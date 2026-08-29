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


def test_pause_and_resume_job(client):
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
    resp = client.post(f"/pipeline/jobs/{job_id}/pause")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "paused"
    assert not ctrl.pause_event.is_set()
    assert _pipeline_jobs[job_id]["status"] == "paused"

    # Pausing again should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/pause")
    assert resp_again.status_code == 400

    # 2. Resume the job
    resp = client.post(f"/pipeline/jobs/{job_id}/resume")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert ctrl.pause_event.is_set()
    assert _pipeline_jobs[job_id]["status"] == "running"

    # Resuming again when running should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/resume")
    assert resp_again.status_code == 400


def test_kill_running_job(client):
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
    resp = client.post(f"/pipeline/jobs/{job_id}/kill")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "killed"
    assert ctrl.kill_event.is_set()
    assert ctrl.pause_event.is_set()  # unblocks thread
    assert _pipeline_jobs[job_id]["status"] == "killed"
    assert _pipeline_jobs[job_id].get("finished_at") is not None

    # Killing again should fail with 400
    resp_again = client.post(f"/pipeline/jobs/{job_id}/kill")
    assert resp_again.status_code == 400


def test_kill_paused_job(client):
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
    resp = client.post(f"/pipeline/jobs/{job_id}/kill")
    assert resp.status_code == 200
    assert resp.json()["status"] == "killed"
    assert ctrl.kill_event.is_set()
    assert ctrl.pause_event.is_set()  # Must be set to awaken paused thread


def test_nonexistent_job_returns_404(client):
    resp = client.post("/pipeline/jobs/nonexistent_job/pause")
    assert resp.status_code == 404

    resp = client.post("/pipeline/jobs/nonexistent_job/resume")
    assert resp.status_code == 404

    resp = client.post("/pipeline/jobs/nonexistent_job/kill")
    assert resp.status_code == 404


def test_kill_completed_job_returns_400(client):
    job_id = "test_job_4"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "created_at": "2026-08-29T10:00:00+00:00",
        "targets": ["doc1.pdf"],
        "successes": [{"pdf": "doc1.pdf"}],
        "failures": [],
    }

    resp = client.post(f"/pipeline/jobs/{job_id}/kill")
    assert resp.status_code == 400
    assert "Cannot kill job with status 'completed'" in resp.json()["detail"]


def test_pipeline_status_reflects_states(client):
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

    resp = client.get("/pipeline/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jobs"]["job_a"]["status"] == "paused"
    assert data["jobs"]["job_b"]["status"] == "killed"
