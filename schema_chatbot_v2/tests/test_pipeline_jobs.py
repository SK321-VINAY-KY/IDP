import threading
import time
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.api.pipeline_routes import _pipeline_jobs, _job_controls, JobControl


@pytest.fixture(autouse=True)
def cleanup_jobs():
    _pipeline_jobs.clear()
    _job_controls.clear()
    yield
    _pipeline_jobs.clear()
    _job_controls.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_pause_and_resume_job(client, auth_headers):
    job_id = "test_job_1"
    ctrl = JobControl()
    _job_controls[job_id] = ctrl
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf", "doc2.pdf"],
        "successes": [],
        "failures": [],
    }

    # 1. Pause the job
    resp = client.post(f"/pipeline/jobs/{job_id}/pause", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "paused"
    assert not ctrl.pause_event.is_set()
    assert _pipeline_jobs[job_id]["status"] == "paused"

    # Pausing again should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/pause", headers=auth_headers)
    assert resp_again.status_code == 400

    # 2. Resume the job
    resp = client.post(f"/pipeline/jobs/{job_id}/resume", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert ctrl.pause_event.is_set()
    assert _pipeline_jobs[job_id]["status"] == "running"

    # Resuming again when running should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/resume", headers=auth_headers)
    assert resp_again.status_code == 400


def test_kill_running_job(client, auth_headers):
    job_id = "test_job_2"
    ctrl = JobControl()
    _job_controls[job_id] = ctrl
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf", "doc2.pdf"],
        "successes": [],
        "failures": [],
    }

    # Kill the job
    resp = client.post(f"/pipeline/jobs/{job_id}/kill", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "killed"
    assert ctrl.kill_event.is_set()
    assert ctrl.pause_event.is_set()  # unblocks thread
    assert _pipeline_jobs[job_id]["status"] == "killed"
    assert _pipeline_jobs[job_id].get("finished_at") is not None

    # Killing again should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/kill", headers=auth_headers)
    assert resp_again.status_code == 400


def test_kill_paused_job(client, auth_headers):
    job_id = "test_job_3"
    ctrl = JobControl()
    ctrl.pause_event.clear()  # paused
    _job_controls[job_id] = ctrl
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "paused",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf", "doc2.pdf"],
        "successes": [],
        "failures": [],
    }

    # Kill the paused job
    resp = client.post(f"/pipeline/jobs/{job_id}/kill", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "killed"
    assert ctrl.kill_event.is_set()
    assert ctrl.pause_event.is_set()  # Must be set to awaken paused thread


def test_nonexistent_job_returns_404(client, auth_headers):
    resp = client.post("/pipeline/jobs/nonexistent_job/pause", headers=auth_headers)
    assert resp.status_code == 404

    resp = client.post("/pipeline/jobs/nonexistent_job/resume", headers=auth_headers)
    assert resp.status_code == 404

    resp = client.post("/pipeline/jobs/nonexistent_job/kill", headers=auth_headers)
    assert resp.status_code == 404


def test_kill_completed_job_returns_400(client, auth_headers):
    job_id = "test_job_4"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf"],
        "successes": [{"pdf": "doc1.pdf"}],
        "failures": [],
    }

    resp = client.post(f"/pipeline/jobs/{job_id}/kill", headers=auth_headers)
    assert resp.status_code == 400
    assert "Cannot kill job with status 'completed'" in resp.json()["detail"]


def test_pipeline_status_reflects_states(client, auth_headers):
    _pipeline_jobs["job_a"] = {
        "job_id": "job_a",
        "status": "paused",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf", "doc2.pdf"],
        "successes": [],
        "failures": [],
    }
    _pipeline_jobs["job_b"] = {
        "job_id": "job_b",
        "status": "killed",
        "created_at": "2026-08-29T10:05:00+00:00",
        "targets": ["doc1.pdf"],
        "successes": [],
        "failures": [],
    }

    resp = client.get("/pipeline/status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["jobs"]["job_a"]["status"] == "paused"
    assert data["jobs"]["job_b"]["status"] == "killed"


def test_pipeline_status_includes_layer3_strategy(client, auth_headers):
    resp = client.get("/pipeline/status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "layer3_strategy" in data
    assert data["layer3_strategy"] in ("graph_memory", "graph_memory_concurrent")


def test_pipeline_markdown_recovery_when_conversion_empty(tmp_path):
    from unittest.mock import MagicMock, patch
    from pathlib import Path
    from app.api.pipeline_routes import _run_pipeline_job, _pipeline_jobs, _job_controls, JobControl
    from src.ai.schemas.page import PageOutput

    job_id = "test_recovery_job"
    doc_stem = "test_doc"
    fake_pdf = tmp_path / f"{doc_stem}.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 test")

    fake_schema_file = tmp_path / "schema_test.json"
    fake_schema_file.write_text('{"schema_id": "schema_test", "schema": {"fields": [{"name": "patient_name", "description": "Patient Name"}]}}', encoding="utf-8")
    schema_record = {"schema_id": "schema_test", "schema": {"fields": [{"name": "patient_name", "description": "Patient Name"}]}}

    existing_md_text = "# Test Doc\n<!-- IDP Pipeline Output\nAvg conf: 0.95\n-->\n<!-- PAGE 1 -->\nPatient: John Doe\n<!-- /PAGE 1 -->\n"
    existing_md_file = tmp_path / f"{doc_stem}.md"
    existing_md_file.write_text(existing_md_text, encoding="utf-8")

    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": [fake_pdf.name],
        "successes": [],
        "failures": [],
    }
    _job_controls[job_id] = JobControl()

    mock_output = PageOutput(
        page_number=1,
        markdown="",
        confidence=0.0,
        engines_used=["paddleocr_printed"],
        capabilities=["has_printed_scan"],
        escalated=False,
        escalation_attempts=0,
        low_confidence=True,
    )
    mock_meta = MagicMock()

    with patch("app.api.pipeline_routes.OUTPUT_DIR", tmp_path), \
         patch("app.api.pipeline_routes.resolve_owner_output_dir", return_value=tmp_path), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[{"page": 1}]), \
         patch("app.api.pipeline_routes.process_document", return_value=[(mock_output, mock_meta)]), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.extractor.extract_document") as mock_extract, \
         patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=99), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record", return_value=1):

        mock_extract_result = MagicMock()
        mock_extract_result.model_dump.return_value = {"patient_name": "John Doe"}
        mock_extract.return_value = mock_extract_result

        _run_pipeline_job(job_id, [fake_pdf], fake_schema_file, schema_record)

        assert len(_pipeline_jobs[job_id]["successes"]) == 1
        success = _pipeline_jobs[job_id]["successes"][0]
        assert success["chars"] > 0
        assert success["extracted_data"] == {"patient_name": "John Doe"}
        assert existing_md_file.read_text(encoding="utf-8") == existing_md_text



