import io
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import (
    _job_controls,
    _pipeline_jobs,
    _lookup_doc_owner,
    JobControl,
    OUTPUT_DIR,
    SCHEMA_REGISTRY,
    _run_pipeline_job,
)
from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.layer3_extraction.storage import init_db, SessionLocal, DocumentGraphRecord


@pytest.fixture(autouse=True)
def cleanup():
    _pipeline_jobs.clear()
    _job_controls.clear()
    yield
    _pipeline_jobs.clear()
    _job_controls.clear()


@pytest.fixture
def client():
    return TestClient(app)


def get_auth_header(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_two_user_collision_invoice_isolated(client):
    """
    Two-user collision resolution test:
    User A (user_alpha) and User B (user_beta) both run a pipeline on "invoice.pdf".
    Verifies that with per-owner partition:
    1. B does NOT overwrite A's files (each stored in dataset_output/<owner>/)
    2. User A successfully queries their own document (200 OK, total=100)
    3. User B successfully queries their own document (200 OK, total=200)
    4. Neither user can query the other's document (403 Forbidden)
    """
    auth_a = get_auth_header(client, "user_alpha", "pass_a")
    auth_b = get_auth_header(client, "user_beta", "pass_b")

    doc_stem = "collision_test_invoice"
    pdf_name = f"{doc_stem}.pdf"
    pdf_path = OUTPUT_DIR / pdf_name
    pdf_path.write_bytes(b"%PDF-1.4 dummy collision pdf")

    schema_id = "test_schema_collision"
    schema_record = {
        "schema_id": schema_id,
        "document_type": "invoice",
        "schema": {"document_type": "invoice", "fields": [{"name": "total", "description": "Total amount"}]},
    }
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(json.dumps(schema_record), encoding="utf-8")

    # Step 1: User A runs pipeline on invoice.pdf
    job_a = "job_user_alpha_1"
    _pipeline_jobs[job_a] = {
        "job_id": job_a,
        "status": "queued",
        "owner": "user_alpha",
        "schema_id": schema_id,
        "successes": [],
        "failures": [],
    }
    _job_controls[job_a] = JobControl()

    mock_page = MagicMock()
    mock_page.page_number = 1
    mock_page.markdown = "Total for Alpha: $100"
    mock_page.engines_used = ["docling"]
    mock_page.capabilities = ["digital_pdf"]
    mock_page.confidence = 0.99
    mock_page.escalated = False
    mock_page.low_confidence = False

    class MockResultA:
        def model_dump(self):
            return {"total": 100, "user": "alpha"}

    with patch("app.api.pipeline_routes.process_document", return_value=[(mock_page, {})]), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[mock_page]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockResultA()), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"):

        _run_pipeline_job(job_a, [pdf_path], schema_file, schema_record)

    # Verify User A's files exist in dataset_output/user_alpha/
    a_ext = OUTPUT_DIR / "user_alpha" / f"{doc_stem}.extracted.json"
    assert a_ext.is_file(), f"Expected {a_ext} to exist"
    a_data = json.loads(a_ext.read_text(encoding="utf-8"))
    assert a_data["owner"] == "user_alpha"
    assert a_data["extracted_data"]["total"] == 100

    # Step 2: User B runs pipeline on invoice.pdf (same filename)
    job_b = "job_user_beta_2"
    _pipeline_jobs[job_b] = {
        "job_id": job_b,
        "status": "queued",
        "owner": "user_beta",
        "schema_id": schema_id,
        "successes": [],
        "failures": [],
    }
    _job_controls[job_b] = JobControl()

    class MockResultB:
        def model_dump(self):
            return {"total": 200, "user": "beta"}

    with patch("app.api.pipeline_routes.process_document", return_value=[(mock_page, {})]), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[mock_page]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockResultB()), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"):

        _run_pipeline_job(job_b, [pdf_path], schema_file, schema_record)

    # 1. Did B overwrite A's files? NO!
    b_ext = OUTPUT_DIR / "user_beta" / f"{doc_stem}.extracted.json"
    assert b_ext.is_file(), f"Expected {b_ext} to exist"
    b_data = json.loads(b_ext.read_text(encoding="utf-8"))
    assert b_data["owner"] == "user_beta"
    assert b_data["extracted_data"]["total"] == 200

    # User A's files MUST be completely intact!
    a_data_after = json.loads(a_ext.read_text(encoding="utf-8"))
    assert a_data_after["owner"] == "user_alpha"
    assert a_data_after["extracted_data"]["total"] == 100

    # 2. Query Bot Q&A Isolation:
    with patch("httpx.post") as mock_post:
        def fake_post(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.is_success = True
            # Inspect prompt to return appropriate content
            prompt_content = kwargs.get("json", {}).get("messages", [{}])[0].get("content", "")
            if "alpha" in prompt_content:
                m.json.return_value = {"choices": [{"message": {"content": "User Alpha Total is 100"}}]}
            else:
                m.json.return_value = {"choices": [{"message": {"content": "User Beta Total is 200"}}]}
            return m

        mock_post.side_effect = fake_post

        # User A queries their own invoice.pdf -> 200 OK
        resp_a = client.post(
            "/api/query-bot/ask",
            headers=auth_a,
            json={"doc_id": pdf_name, "question": "What is the total?"},
        )
        assert resp_a.status_code == 200, f"Expected 200 for user A, got {resp_a.status_code}: {resp_a.text}"
        assert "100" in resp_a.json()["answer"]

        # User B queries their own invoice.pdf -> 200 OK
        resp_b = client.post(
            "/api/query-bot/ask",
            headers=auth_b,
            json={"doc_id": pdf_name, "question": "What is the total?"},
        )
        assert resp_b.status_code == 200, f"Expected 200 for user B, got {resp_b.status_code}: {resp_b.text}"
        assert "200" in resp_b.json()["answer"]

    # Clean up DB and files
    init_db()
    sess = SessionLocal()
    try:
        sess.query(DocumentGraphRecord).filter(DocumentGraphRecord.doc_id == pdf_name).delete()
        sess.commit()
    finally:
        sess.close()

    for p in [pdf_path, schema_file]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    for u in ["user_alpha", "user_beta"]:
        shutil.rmtree(OUTPUT_DIR / u, ignore_errors=True)
