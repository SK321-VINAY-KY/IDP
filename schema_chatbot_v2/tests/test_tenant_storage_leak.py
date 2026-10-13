"""
Cross-tenant reproduction test for audit finding #4:
Tenant isolation in storage.py and its callers.

Tests that two owners uploading/extracting the same filename ('resume.pdf'):
1. Overwrite each other's DocumentRecord shadow copy in storage.py.
2. Overwrite each other's MarkdownRecord in storage.py.
3. Cross-contaminate ExtractionRun via get_latest_extracted_data().
4. Leak extraction data across tenants in Query Bot JSON fallback.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.api.pipeline_routes import OUTPUT_DIR
from src.ai.layer3_extraction.storage import (
    init_db,
    SessionLocal,
    DocumentRecord,
    MarkdownRecord,
    ExtractionRun,
    DocumentGraphRecord,
    save_document,
    save_markdown_record,
    get_markdown_record,
    save_extraction_run,
    get_latest_extracted_data,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    session = SessionLocal()
    try:
        session.query(DocumentRecord).delete()
        session.query(MarkdownRecord).delete()
        session.query(ExtractionRun).delete()
        session.query(DocumentGraphRecord).delete()
        session.commit()
    finally:
        session.close()
    yield
    session = SessionLocal()
    try:
        session.query(DocumentRecord).delete()
        session.query(MarkdownRecord).delete()
        session.query(ExtractionRun).delete()
        session.query(DocumentGraphRecord).delete()
        session.commit()
    finally:
        session.close()


def get_token(client, username):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username, "password123", Role.USER)
    resp = client.post("/auth/login", data={"username": username, "password": "password123"})
    return resp.json()["access_token"]


def test_cross_tenant_document_record_collision():
    """
    Two users upload 'resume.pdf'.
    Owner A's document must remain intact and distinct from Owner B's.
    Verified isolated by composite uniqueness (owner, filename).
    """
    # Owner A saves resume.pdf
    save_document(filename="resume.pdf", file_bytes=b"Owner A PDF Content", owner="owner_a", content_type="application/pdf")

    # Owner B saves resume.pdf with different content
    save_document(filename="resume.pdf", file_bytes=b"Owner B PDF Content", owner="owner_b", content_type="application/pdf")

    session = SessionLocal()
    try:
        # Both documents should exist, isolated by owner
        records = session.query(DocumentRecord).filter(DocumentRecord.filename == "resume.pdf").all()
        assert len(records) == 2, f"Expected 2 isolated DocumentRecords for resume.pdf, found {len(records)}"
        owners = {r.owner for r in records}
        assert owners == {"owner_a", "owner_b"}
    finally:
        session.close()


def test_cross_tenant_markdown_record_collision():
    """
    Two users convert 'resume.pdf' to markdown.
    Owner A's markdown must not be overwritten by Owner B.
    Verified isolated by composite uniqueness (owner, doc_id).
    """
    # Owner A saves markdown
    save_markdown_record(
        doc_id="resume.pdf",
        md_filename="resume.md",
        markdown_content="# Resume of Alice (Owner A)",
        owner="owner_a",
    )

    # Owner B saves markdown
    save_markdown_record(
        doc_id="resume.pdf",
        md_filename="resume.md",
        markdown_content="# Resume of Bob (Owner B)",
        owner="owner_b",
    )

    session = SessionLocal()
    try:
        records = session.query(MarkdownRecord).filter(MarkdownRecord.doc_id == "resume.pdf").all()
        assert len(records) == 2, f"Expected 2 isolated MarkdownRecords for resume.pdf, found {len(records)}"
        rec_a = get_markdown_record("resume.pdf", owner="owner_a")
        rec_b = get_markdown_record("resume.pdf", owner="owner_b")
        assert rec_a and "Alice" in rec_a["markdown_content"]
        assert rec_b and "Bob" in rec_b["markdown_content"]
    finally:
        session.close()


def test_cross_tenant_extraction_run_leak():
    """
    Owner A extracts private data from resume.pdf.
    Owner B extracts their data from their own resume.pdf.
    Owner A reading extracted data must get Owner A's data, NOT Owner B's data.
    """
    # Owner A runs extraction
    save_extraction_run(
        doc_id="resume.pdf",
        page_count=1,
        schema_name="resume_schema",
        result_json={"extracted_data": {"candidate_name": "Alice", "salary": 120000}},
        owner="owner_a",
        llm_provider="sarvam",
        model_name="sarvam-2b",
        processing_time_seconds=1.0,
    )

    # Owner B runs extraction afterwards
    save_extraction_run(
        doc_id="resume.pdf",
        page_count=1,
        schema_name="resume_schema",
        result_json={"extracted_data": {"candidate_name": "Bob", "salary": 60000}},
        owner="owner_b",
        llm_provider="sarvam",
        model_name="sarvam-2b",
        processing_time_seconds=1.0,
    )

    # Querying extracted data is scoped by owner.
    data_a = get_latest_extracted_data("resume.pdf", owner="owner_a")
    assert data_a is not None
    assert data_a["candidate_name"] == "Alice", f"Expected Alice's data, got: {data_a}"

    data_b = get_latest_extracted_data("resume.pdf", owner="owner_b")
    assert data_b is not None
    assert data_b["candidate_name"] == "Bob", f"Expected Bob's data, got: {data_b}"


def test_cross_tenant_query_bot_json_fallback_leak():
    """
    Query Bot JSON fallback leak:
    Owner A extracted data for resume.pdf (Alice, salary 120000).
    Owner B has resume.pdf on disk in dataset_output/owner_b/ (so access check passes),
    but Owner B has no document graph.
    When Owner B queries Query Bot about resume.pdf, Query Bot falls back to DB JSON
    and calls get_latest_extracted_data("resume.pdf", owner="owner_b").
    Owner B does NOT get Alice's extraction data!
    """
    client = TestClient(app)
    token_b = get_token(client, "owner_b")

    # Owner A extracted private data in storage under owner_a
    save_extraction_run(
        doc_id="resume.pdf",
        page_count=1,
        schema_name="resume_schema",
        result_json={"extracted_data": {"candidate_name": "Alice Secret", "salary": 120000}},
        owner="owner_a",
        llm_provider="sarvam",
        model_name="sarvam-2b",
        processing_time_seconds=1.0,
    )

    # Owner B extracted their own data in storage under owner_b
    save_extraction_run(
        doc_id="resume.pdf",
        page_count=1,
        schema_name="resume_schema",
        result_json={"extracted_data": {"candidate_name": "Bob Normal", "salary": 60000}},
        owner="owner_b",
        llm_provider="sarvam",
        model_name="sarvam-2b",
        processing_time_seconds=1.0,
    )

    # Owner B has their own resume.pdf sidecar in dataset_output/owner_b/
    b_dir = OUTPUT_DIR / "owner_b"
    b_dir.mkdir(parents=True, exist_ok=True)
    b_sidecar = b_dir / "resume.extracted.json"
    b_sidecar.write_text(json.dumps({"owner": "owner_b", "extracted_data": {"candidate_name": "Bob Normal"}}), encoding="utf-8")

    try:
        resp = client.post(
            "/api/query-bot/ask",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"doc_id": "resume.pdf", "question": "What is the candidate name?"},
        )

        assert resp.status_code == 200
        answer = resp.json()["answer"]
        assert "Alice Secret" not in answer, f"LEAK DETECTED: Owner A's private data leaked to Owner B in answer: {answer}"
    finally:
        b_sidecar.unlink(missing_ok=True)
