"""
End-to-End Acceptance Tests verifying the full specification:
- Seeded admin login & access
- User role 403 on admin-only endpoints
- Isolated user document directories & status
- User zero-param pipeline run & 400 validations
- Job isolation & 404 on unowned jobs
- Admin cross-owner job controls
- System log ring buffer & filtered user activity logs
"""
import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import _job_controls, _pipeline_jobs, JobControl, SCHEMA_REGISTRY
from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture(autouse=True)
def cleanup():
    _pipeline_jobs.clear()
    _job_controls.clear()
    test_schema = SCHEMA_REGISTRY / "schema_u1_acc.json"
    if test_schema.exists():
        test_schema.unlink()
    yield
    _pipeline_jobs.clear()
    _job_controls.clear()
    if test_schema.exists():
        test_schema.unlink()


@pytest.fixture
def client():
    return TestClient(app)


def login(client, username, password):
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    return resp.json()["access_token"]


def test_acceptance_criteria_complete_suite(client, monkeypatch):
    store = get_user_store()

    # 1. Admin login with seeded credentials
    admin_token = login(client, "admin", "changeme")
    admin_auth = {"Authorization": f"Bearer {admin_token}"}

    me_admin = client.get("/auth/me", headers=admin_auth)
    assert me_admin.status_code == 200
    assert me_admin.json() == {"username": "admin", "role": "admin"}

    # 2. Admin creates two user-role accounts: user1 and user2
    for u, p in [("user1", "pass1"), ("user2", "pass2")]:
        create_resp = client.post(
            "/admin/users",
            json={"username": u, "password": p, "role": "user"},
            headers=admin_auth,
        )
        assert create_resp.status_code in (201, 400)  # 201 created or 400 already exists

    token1 = login(client, "user1", "pass1")
    auth1 = {"Authorization": f"Bearer {token1}"}

    token2 = login(client, "user2", "pass2")
    auth2 = {"Authorization": f"Bearer {token2}"}

    # 3. User1 gets 403 on every admin-only endpoint
    for path in ["/documents", "/schemas", "/pipeline/status", "/pipeline/jobs/dummy_job_id", "/admin/logs/system", "/admin/logs/users", "/admin/users"]:
        assert client.get(path, headers=auth1).status_code == 403
    for path in ["/documents/upload", "/pipeline/jobs/dummy/pause", "/pipeline/jobs/dummy/resume", "/pipeline/jobs/dummy/kill", "/pipeline/run", "/admin/users"]:
        assert client.post(path, headers=auth1).status_code == 403

    # 4. User1 and User2 upload documents to /me/documents and are isolated
    pdf1 = b"%PDF-1.4 User 1 Private Doc"
    client.post("/me/documents", files=[("files", ("u1_doc.pdf", io.BytesIO(pdf1), "application/pdf"))], headers=auth1)

    pdf2 = b"%PDF-1.4 User 2 Private Doc"
    client.post("/me/documents", files=[("files", ("u2_doc.pdf", io.BytesIO(pdf2), "application/pdf"))], headers=auth2)

    docs1 = [d["name"] for d in client.get("/me/documents", headers=auth1).json()["documents"]]
    docs2 = [d["name"] for d in client.get("/me/documents", headers=auth2).json()["documents"]]
    assert "u1_doc.pdf" in docs1 and "u2_doc.pdf" not in docs1
    assert "u2_doc.pdf" in docs2 and "u1_doc.pdf" not in docs2

    # 5. User zero-param pipeline run validations
    # User2 has uploaded docs but no confirmed schema -> 400 "Build and confirm a schema first"
    resp_no_schema = client.post("/me/pipeline/run", headers=auth2)
    assert resp_no_schema.status_code == 400
    assert "Build and confirm a schema first" in resp_no_schema.json()["detail"]

    # User3 with no docs and no schema -> 400 "Upload documents first"
    store.create(username="user3", password="pass3", role=Role.USER)
    auth3 = {"Authorization": f"Bearer {login(client, 'user3', 'pass3')}"}
    resp_no_docs = client.post("/me/pipeline/run", headers=auth3)
    assert resp_no_docs.status_code == 400
    assert "Upload documents first" in resp_no_docs.json()["detail"]

    # Plant a confirmed schema for User1
    schema_id = "schema_u1_acc"
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(json.dumps({
        "schema_id": schema_id,
        "owner": "user1",
        "document_type": "invoice",
        "confirmed_at": "2026-08-31T15:00:00+00:00",
        "schema": {"document_type": "invoice", "fields": [{"name": "total", "type": "number"}]}
    }), encoding="utf-8")

    # Mock background runner
    def mock_runner(job_id, targets, spath, srec):
        _pipeline_jobs[job_id]["status"] = "running"
        _pipeline_jobs[job_id]["current_document"] = targets[0].name
        _pipeline_jobs[job_id]["successes"].append({"pdf": targets[0].name})
        _pipeline_jobs[job_id]["status"] = "completed"

    from app.api import user_routes
    monkeypatch.setattr(user_routes, "_run_pipeline_job", mock_runner)

    # User1 triggers pipeline run (zero parameters)
    run_resp = client.post("/me/pipeline/run", headers=auth1)
    assert run_resp.status_code == 200
    job_u1_id = run_resp.json()["job_id"]

    # 6. Status isolation: User1 sees job_u1_id, User2 does NOT
    st1 = client.get("/me/pipeline/status", headers=auth1).json()["jobs"]
    st2 = client.get("/me/pipeline/status", headers=auth2).json()["jobs"]
    assert any(j["job_id"] == job_u1_id for j in st1)
    assert not any(j["job_id"] == job_u1_id for j in st2)

    # 7. Guessing job_id: User2 gets 404 (not 403) trying to pause/resume/kill User1's job
    assert client.post(f"/me/pipeline/jobs/{job_u1_id}/pause", headers=auth2).status_code == 404
    assert client.post(f"/me/pipeline/jobs/{job_u1_id}/resume", headers=auth2).status_code == 404
    assert client.post(f"/me/pipeline/jobs/{job_u1_id}/kill", headers=auth2).status_code == 404

    # 8. Admin can control any job regardless of owner
    job_ctrl = JobControl()
    _job_controls["job_running_u1"] = job_ctrl
    _pipeline_jobs["job_running_u1"] = {
        "job_id": "job_running_u1",
        "owner": "user1",
        "status": "running",
        "targets": ["u1_doc.pdf"],
        "successes": [],
        "failures": [],
    }

    assert client.post("/pipeline/jobs/job_running_u1/pause", headers=admin_auth).status_code == 200
    assert _pipeline_jobs["job_running_u1"]["status"] == "paused"
    assert client.post("/pipeline/jobs/job_running_u1/resume", headers=admin_auth).status_code == 200
    assert _pipeline_jobs["job_running_u1"]["status"] == "running"
    assert client.post("/pipeline/jobs/job_running_u1/kill", headers=admin_auth).status_code == 200
    assert _pipeline_jobs["job_running_u1"]["status"] == "killed"

    # 9. Logs: system logs return lines; user activity logs filter by username
    sys_logs = client.get("/admin/logs/system", headers=admin_auth).json()["lines"]
    assert isinstance(sys_logs, list)

    u1_logs = client.get("/admin/logs/users?username=user1", headers=admin_auth).json()["logs"]
    assert len(u1_logs) > 0
    assert all(entry["username"] == "user1" for entry in u1_logs)
