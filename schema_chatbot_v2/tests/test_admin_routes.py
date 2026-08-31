import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import _job_controls, _pipeline_jobs, JobControl
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


def get_token_for(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


def test_admin_create_user(client):
    admin_token = get_token_for(client, "admin", "changeme", role=Role.ADMIN)
    resp = client.post(
        "/admin/users",
        json={"username": "newuser", "password": "newpassword", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "newuser"
    assert data["role"] == "user"

    # User can now login
    login_resp = client.post("/auth/login", data={"username": "newuser", "password": "newpassword"})
    assert login_resp.status_code == 200


def test_regular_user_forbidden_on_admin_routes(client):
    user_token = get_token_for(client, "regular_joe", "pass123", role=Role.USER)
    headers = {"Authorization": f"Bearer {user_token}"}

    # 1. Admin users endpoint
    assert client.post("/admin/users", json={"username": "x", "password": "y"}, headers=headers).status_code == 403

    # 2. Documents endpoints
    assert client.get("/documents", headers=headers).status_code == 403
    assert client.post("/documents/upload", headers=headers).status_code == 403

    # 3. Schemas endpoints
    assert client.get("/schemas", headers=headers).status_code == 403
    assert client.get("/schemas/some_schema", headers=headers).status_code == 403

    # 4. Admin pipeline endpoints
    assert client.get("/pipeline/status", headers=headers).status_code == 403
    assert client.get("/pipeline/jobs/job_1", headers=headers).status_code == 403
    assert client.post("/pipeline/jobs/job_1/pause", headers=headers).status_code == 403
    assert client.post("/pipeline/jobs/job_1/resume", headers=headers).status_code == 403
    assert client.post("/pipeline/jobs/job_1/kill", headers=headers).status_code == 403
    assert client.post("/pipeline/run", headers=headers).status_code == 403


def test_admin_can_control_any_job(client):
    admin_token = get_token_for(client, "admin", "changeme", role=Role.ADMIN)
    headers = {"Authorization": f"Bearer {admin_token}"}

    job_id = "job_owned_by_someone_else"
    ctrl = JobControl()
    _job_controls[job_id] = ctrl
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "owner": "regular_joe",
        "targets": ["doc.pdf"],
        "successes": [],
        "failures": [],
    }

    # Admin pauses the job
    pause_resp = client.post(f"/pipeline/jobs/{job_id}/pause", headers=headers)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["status"] == "paused"

    # Admin resumes the job
    resume_resp = client.post(f"/pipeline/jobs/{job_id}/resume", headers=headers)
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "running"

    # Admin kills the job
    kill_resp = client.post(f"/pipeline/jobs/{job_id}/kill", headers=headers)
    assert kill_resp.status_code == 200
    assert kill_resp.json()["status"] == "killed"
