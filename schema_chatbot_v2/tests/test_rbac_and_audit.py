import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.pipeline_routes import _pipeline_jobs, _job_controls, JobControl
from app.storage.audit_log import get_audit_logger
from app.storage.user_store import get_user_store


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


def test_auth_login_and_me(client):
    # Login as user1
    resp = client.post("/auth/login", json={"username": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"]["username"] == "user1"
    assert data["user"]["role"] == "normal"
    token = data["token"]

    # Verify /auth/me with Bearer token
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "user1"


def test_rbac_job_isolation(client):
    # Create a job owned by user1 and another by user2
    _pipeline_jobs["job_user1"] = {
        "job_id": "job_user1",
        "owner": "user1",
        "owner_role": "normal",
        "status": "running",
        "created_at": "2026-08-31T12:00:00+05:30",
        "targets": ["doc1.pdf"],
        "successes": [],
        "failures": [],
    }
    _pipeline_jobs["job_user2"] = {
        "job_id": "job_user2",
        "owner": "user2",
        "owner_role": "normal",
        "status": "completed",
        "created_at": "2026-08-31T12:30:00+05:30",
        "targets": ["doc2.pdf"],
        "successes": [{"pdf": "doc2.pdf"}],
        "failures": [],
    }

    # 1. user1 checks pipeline status -> should ONLY see job_user1
    user1_resp = client.get("/pipeline/status", headers={"X-User-Id": "user1"})
    assert user1_resp.status_code == 200
    data1 = user1_resp.json()
    assert "job_user1" in data1["jobs"]
    assert "job_user2" not in data1["jobs"]
    assert data1["my_summary"]["total"] == 1
    assert data1["my_summary"]["running"] == 1

    # 2. user2 checks pipeline status -> should ONLY see job_user2
    user2_resp = client.get("/pipeline/status", headers={"X-User-Id": "user2"})
    assert user2_resp.status_code == 200
    data2 = user2_resp.json()
    assert "job_user2" in data2["jobs"]
    assert "job_user1" not in data2["jobs"]
    assert data2["my_summary"]["total"] == 1
    assert data2["my_summary"]["completed"] == 1

    # 3. admin checks pipeline status -> should see BOTH jobs
    admin_resp = client.get("/pipeline/status", headers={"X-User-Id": "admin"})
    assert admin_resp.status_code == 200
    data_admin = admin_resp.json()
    assert "job_user1" in data_admin["jobs"]
    assert "job_user2" in data_admin["jobs"]


def test_admin_routes_protection(client):
    # Normal user accessing admin endpoints -> 403 Forbidden
    resp_overview_normal = client.get("/admin/overview", headers={"X-User-Id": "user1"})
    assert resp_overview_normal.status_code == 403

    resp_activity_normal = client.get("/admin/activity", headers={"X-User-Id": "user1"})
    assert resp_activity_normal.status_code == 403

    # Admin user accessing admin endpoints -> 200 OK
    resp_overview_admin = client.get("/admin/overview", headers={"X-User-Id": "admin"})
    assert resp_overview_admin.status_code == 200
    overview = resp_overview_admin.json()
    assert "total_users" in overview
    assert "success_rate" in overview

    resp_activity_admin = client.get("/admin/activity", headers={"X-User-Id": "admin"})
    assert resp_activity_admin.status_code == 200
    assert isinstance(resp_activity_admin.json(), list)


def test_audit_logging_across_actions(client):
    audit = get_audit_logger()
    audit.log_activity(
        username="user1",
        role="normal",
        action="TEST_ACTION",
        details={"test_key": "test_val"},
    )

    activities = audit.get_activities(username="user1", action="TEST_ACTION")
    assert len(activities) >= 1
    assert activities[0].details["test_key"] == "test_val"
