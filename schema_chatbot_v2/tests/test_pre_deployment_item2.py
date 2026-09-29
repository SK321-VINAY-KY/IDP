import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import (
    _job_controls,
    _pipeline_jobs,
    JobControl,
    OUTPUT_DIR,
    SCHEMA_REGISTRY,
    _run_pipeline_job,
)
from app.main import app
from app.storage.user_store import Role, get_user_store


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


def test_user_a_pipeline_doc_ownership_and_isolation(client, tmp_path, monkeypatch):
    """
    Item 2:
    - User A runs a pipeline job.
    - _run_pipeline_job writes 'owner': 'user_a' into .extracted.json and .schema_ref.json.
    - User A queries their own doc via /api/query-bot/ask -> 200 OK.
    - User B queries User A's doc -> 403 Forbidden.
    - Non-admin on an ownerless doc -> 403 Forbidden.
    """
    auth_a = get_auth_header(client, "user_a", "pass_a")
    auth_b = get_auth_header(client, "user_b", "pass_b")

    doc_stem = "invoice_user_a_test"
    pdf_name = f"{doc_stem}.pdf"
    pdf_path = OUTPUT_DIR / pdf_name
    pdf_path.write_bytes(b"%PDF-1.4 dummy pdf")

    schema_id = "test_schema_item2"
    schema_record = {
        "schema_id": schema_id,
        "document_type": "invoice",
        "schema": {"document_type": "invoice", "fields": [{"name": "total", "description": "Total amount"}]},
    }
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(json.dumps(schema_record), encoding="utf-8")

    job_id = "job_test_user_a_123"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "owner": "user_a",
        "schema_id": schema_id,
        "successes": [],
        "failures": [],
    }
    _job_controls[job_id] = JobControl()

    mock_page = MagicMock()
    mock_page.page_number = 1
    mock_page.markdown = "Total: $500"
    mock_page.engines_used = ["docling"]
    mock_page.capabilities = ["digital_pdf"]
    mock_page.confidence = 0.99
    mock_page.escalated = False
    mock_page.low_confidence = False

    class MockExtractedResult:
        def model_dump(self):
            return {"total": 500}

    with patch("app.api.pipeline_routes.process_document", return_value=[(mock_page, {})]), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[mock_page]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockExtractedResult()), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"), \
         patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=1), \
         patch("src.ai.layer3_extraction.storage.save_document_graph"):

        _run_pipeline_job(job_id, [pdf_path], schema_file, schema_record)

    # 1. Verify owner written to .schema_ref.json
    ref_path = OUTPUT_DIR / "user_a" / f"{doc_stem}.schema_ref.json"
    if not ref_path.is_file():
        ref_path = OUTPUT_DIR / f"{doc_stem}.schema_ref.json"
    assert ref_path.is_file(), "schema_ref.json was not created"
    ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
    assert ref_data.get("owner") == "user_a", f"Expected owner 'user_a' in {ref_path}, got {ref_data.get('owner')}"

    # 2. Verify owner written to .extracted.json
    ext_path = OUTPUT_DIR / "user_a" / f"{doc_stem}.extracted.json"
    if not ext_path.is_file():
        ext_path = OUTPUT_DIR / f"{doc_stem}.extracted.json"
    assert ext_path.is_file(), "extracted.json was not created"
    ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
    assert ext_data.get("owner") == "user_a", f"Expected owner 'user_a' in {ext_path}, got {ext_data.get('owner')}"

    # 3. User A queries their own doc via /api/query-bot/ask -> 200 OK
    with patch("httpx.post") as mock_http_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {"choices": [{"message": {"content": "The total is $500."}}]}
        mock_http_post.return_value = mock_resp

        resp_a = client.post(
            "/api/query-bot/ask",
            headers=auth_a,
            json={"doc_id": pdf_name, "question": "What is the total?"},
        )
        assert resp_a.status_code == 200, f"User A got {resp_a.status_code}: {resp_a.text}"
        assert "500" in resp_a.json()["answer"]

    # 4. User B queries User A's doc -> 403 Forbidden
    resp_b = client.post(
        "/api/query-bot/ask",
        headers=auth_b,
        json={"doc_id": pdf_name, "question": "What is the total?"},
    )
    assert resp_b.status_code == 403, f"User B got {resp_b.status_code}, expected 403"
    assert "Access denied" in resp_b.json()["detail"]

    # 5. Non-admin queries an ownerless doc -> 403 Forbidden
    ownerless_stem = "ownerless_doc_test"
    ownerless_ext = OUTPUT_DIR / f"{ownerless_stem}.extracted.json"
    ownerless_ext.write_text(json.dumps({"extracted_data": {"val": 123}}), encoding="utf-8")

    resp_ownerless = client.post(
        "/api/query-bot/ask",
        headers=auth_a,
        json={"doc_id": f"{ownerless_stem}.pdf", "question": "What is val?"},
    )
    assert resp_ownerless.status_code == 403, f"Got {resp_ownerless.status_code}, expected 403"
    assert "ownerless document readable by admin only" in resp_ownerless.json()["detail"]

    # Clean up generated files
    import shutil
    for p in [pdf_path, ref_path, ext_path, schema_file, ownerless_ext]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    shutil.rmtree(OUTPUT_DIR / "user_a", ignore_errors=True)


def test_extract_from_output_writes_owner(client):
    admin_auth = get_auth_header(client, "admin", "changeme", role=Role.ADMIN)

    doc_stem = "report_user_x_test"
    md_name = f"{doc_stem}.md"
    md_path = OUTPUT_DIR / md_name
    md_path.write_text("# Report\nContent here", encoding="utf-8")

    schema_id = "test_schema_item2_ext"
    schema_record = {
        "schema_id": schema_id,
        "document_type": "report",
        "schema": {"document_type": "report", "fields": [{"name": "topic", "description": "Topic"}]},
    }
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(json.dumps(schema_record), encoding="utf-8")

    ref_path = OUTPUT_DIR / f"{doc_stem}.schema_ref.json"
    ref_path.write_text(json.dumps({
        "source_pdf": f"{doc_stem}.pdf",
        "schema_id": schema_id,
        "document_type": "report",
        "owner": "user_x",
    }), encoding="utf-8")

    class MockExtractedResult:
        def model_dump(self):
            return {"topic": "AI"}

    with patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=MockExtractedResult()), \
         patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=1):

        resp = client.post(
            "/pipeline/extract/from-output",
            headers=admin_auth,
            data={"md_name": md_name},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    ext_path = OUTPUT_DIR / f"{doc_stem}.extracted.json"
    assert ext_path.is_file(), "extracted.json was not created"
    ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
    assert ext_data.get("owner") == "user_x", f"Expected owner 'user_x' in extracted.json, got {ext_data.get('owner')}"

    # Clean up
    for p in [md_path, ref_path, ext_path, schema_file]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

