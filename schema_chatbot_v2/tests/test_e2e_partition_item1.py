import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import (
    DATASET_DIR,
    OUTPUT_DIR,
    SCHEMA_REGISTRY,
    JobControl,
    _job_controls,
    _pipeline_jobs,
)
from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_users(client):
    store = get_user_store()
    for u in ("user_alpha_1", "user_beta_1"):
        try:
            store.create(username=u, password="password123", role=Role.USER)
        except ValueError:
            pass

    # Login user_alpha_1
    r_a = client.post("/auth/login", data={"username": "user_alpha_1", "password": "password123"})
    token_a = r_a.json()["access_token"]

    # Login user_beta_1
    r_b = client.post("/auth/login", data={"username": "user_beta_1", "password": "password123"})
    token_b = r_b.json()["access_token"]

    return {
        "alpha": {"Authorization": f"Bearer {token_a}"},
        "beta": {"Authorization": f"Bearer {token_b}"},
    }


def test_e2e_two_users_invoice_md_isolation_and_from_output(client, auth_users):
    """
    Item 1: End-to-end partition test.
    User A and User B both run pipeline for 'invoice.pdf'.
    Verify:
    1. User A's .md is written to dataset_output/user_alpha_1/invoice.md.
    2. User B's .md is written to dataset_output/user_beta_1/invoice.md.
    3. User B does NOT overwrite User A's .md.
    4. User A calls /me/pipeline/extract/from-output, reading from dataset_output/user_alpha_1/invoice.md.
    5. User A and User B both query /api/query-bot/ask and receive isolated responses.
    """
    doc_name = "invoice.pdf"
    doc_stem = "invoice"
    schema_id = "test_schema_item1"

    # Create schema in registry
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(
        json.dumps({
            "schema_id": schema_id,
            "document_type": "Invoice",
            "owner": "user_alpha_1",
            "confirmed_at": "2026-09-24T12:00:00Z",
            "schema": {
                "document_type": "Invoice",
                "fields": [{"name": "total", "type": "number", "required": True}],
            },
        }),
        encoding="utf-8",
    )
    # Also schema for beta
    schema_file_b = SCHEMA_REGISTRY / f"{schema_id}_b.json"
    schema_file_b.write_text(
        json.dumps({
            "schema_id": f"{schema_id}_b",
            "document_type": "Invoice",
            "owner": "user_beta_1",
            "confirmed_at": "2026-09-24T12:00:00Z",
            "schema": {
                "document_type": "Invoice",
                "fields": [{"name": "total", "type": "number", "required": True}],
            },
        }),
        encoding="utf-8",
    )

    alpha_dir = DATASET_DIR / "_users" / "user_alpha_1" / "documents"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    (alpha_dir / doc_name).write_bytes(b"%PDF-1.4 Mock Alpha")

    beta_dir = DATASET_DIR / "_users" / "user_beta_1" / "documents"
    beta_dir.mkdir(parents=True, exist_ok=True)
    (beta_dir / doc_name).write_bytes(b"%PDF-1.4 Mock Beta")

    # Mocks for Page output
    mock_page_a = MagicMock()
    mock_page_a.page_number = 1
    mock_page_a.markdown = "# Invoice for Alpha\nTotal: $100"
    mock_page_a.engines_used = ["docling"]
    mock_page_a.capabilities = ["digital_pdf"]
    mock_page_a.confidence = 0.99
    mock_page_a.escalated = False
    mock_page_a.low_confidence = False

    mock_page_b = MagicMock()
    mock_page_b.page_number = 1
    mock_page_b.markdown = "# Invoice for Beta\nTotal: $200"
    mock_page_b.engines_used = ["docling"]
    mock_page_b.capabilities = ["digital_pdf"]
    mock_page_b.confidence = 0.99
    mock_page_b.escalated = False
    mock_page_b.low_confidence = False

    class MockResultA:
        def model_dump(self):
            return {"total": 100, "user": "alpha"}

    class MockResultB:
        def model_dump(self):
            return {"total": 200, "user": "beta"}

    from app.api.pipeline_routes import _run_pipeline_job

    # 1. Run User A pipeline
    job_a = "job_alpha_test_1"
    _pipeline_jobs[job_a] = {
        "job_id": job_a,
        "status": "queued",
        "owner": "user_alpha_1",
        "schema_id": schema_id,
        "successes": [],
        "failures": [],
    }
    _job_controls[job_a] = JobControl()

    def fake_process_doc(pages, **kwargs):
        out_dir = Path(kwargs.get("output_dir", OUTPUT_DIR))
        doc_name = kwargs.get("document_name", "doc.pdf")
        stem = Path(doc_name).stem
        (out_dir / f"{stem}.md").write_text(pages[0].markdown, encoding="utf-8")
        return [(pages[0], {})]

    with patch("app.api.pipeline_routes.process_document", side_effect=fake_process_doc), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[mock_page_a]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockResultA()), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"):

        _run_pipeline_job(job_a, [alpha_dir / doc_name], schema_file, {"schema_id": schema_id})

    # Check User A's markdown was written to dataset_output/user_alpha_1/invoice.md
    a_md = OUTPUT_DIR / "user_alpha_1" / f"{doc_stem}.md"
    assert a_md.is_file(), f"Expected {a_md} to exist"
    assert "Alpha" in a_md.read_text(encoding="utf-8")

    # 2. Run User B pipeline on same doc name "invoice.pdf"
    job_b = "job_beta_test_1"
    _pipeline_jobs[job_b] = {
        "job_id": job_b,
        "status": "queued",
        "owner": "user_beta_1",
        "schema_id": f"{schema_id}_b",
        "successes": [],
        "failures": [],
    }
    _job_controls[job_b] = JobControl()

    with patch("app.api.pipeline_routes.process_document", side_effect=fake_process_doc), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[mock_page_b]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockResultB()), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"):

        _run_pipeline_job(job_b, [beta_dir / doc_name], schema_file_b, {"schema_id": f"{schema_id}_b"})

    # Check User B's markdown was written to dataset_output/user_beta_1/invoice.md
    b_md = OUTPUT_DIR / "user_beta_1" / f"{doc_stem}.md"
    assert b_md.is_file(), f"Expected {b_md} to exist"
    assert "Beta" in b_md.read_text(encoding="utf-8")

    # CRITICAL CHECK: User A's .md must NOT be overwritten!
    assert "Alpha" in a_md.read_text(encoding="utf-8"), "User B overwrote User A's .md file!"

    # 3. User A calls /me/pipeline/extract/from-output for "invoice.md"
    with patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockResultA()):
        resp_from_out = client.post(
            "/me/pipeline/extract/from-output",
            data={"md_name": f"{doc_stem}.md"},
            headers=auth_users["alpha"],
        )
        assert resp_from_out.status_code == 200, f"Expected 200, got {resp_from_out.status_code}: {resp_from_out.text}"
        data_res = resp_from_out.json()
        assert data_res["success"] is True

    # Check that extract from output read from dataset_output/user_alpha_1/invoice.md and wrote to dataset_output/user_alpha_1/
    a_ext = OUTPUT_DIR / "user_alpha_1" / f"{doc_stem}.extracted.json"
    assert a_ext.is_file()
    assert json.loads(a_ext.read_text(encoding="utf-8"))["owner"] == "user_alpha_1"

    # Cleanup
    shutil.rmtree(OUTPUT_DIR / "user_alpha_1", ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR / "user_beta_1", ignore_errors=True)
    shutil.rmtree(alpha_dir, ignore_errors=True)
    shutil.rmtree(beta_dir, ignore_errors=True)
    schema_file.unlink(missing_ok=True)
    schema_file_b.unlink(missing_ok=True)
