import io
import logging
import pytest
from fastapi.testclient import TestClient

from app.core.activity_log import clear_activity_logs, get_activity_logs, log_activity
from app.core.log_buffer import _log_buffer
from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture(autouse=True)
def reset_logs():
    clear_activity_logs()
    yield
    clear_activity_logs()


@pytest.fixture
def client():
    return TestClient(app)


def get_token(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


def test_system_logs_endpoint(client):
    admin_token = get_token(client, "admin", "changeme", role=Role.ADMIN)
    user_token = get_token(client, "norman", "pass123", role=Role.USER)

    # Emit a test log
    logging.getLogger("test_logger").warning("Test log message for ring buffer")

    # Regular user gets 403
    resp_user = client.get("/admin/logs/system", headers={"Authorization": f"Bearer {user_token}"})
    assert resp_user.status_code == 403

    # Admin gets 200 and sees lines
    resp_admin = client.get("/admin/logs/system?limit=50", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp_admin.status_code == 200
    lines = resp_admin.json()["lines"]
    assert any("Test log message for ring buffer" in l for l in lines)


def test_user_activity_logging_and_filtering(client):
    admin_token = get_token(client, "admin", "changeme", role=Role.ADMIN)

    # 1. Login generates activity log
    get_token(client, "user_alpha", "alphapass", role=Role.USER)
    get_token(client, "user_beta", "betapass", role=Role.USER)

    # 2. Upload document generates activity log
    auth_alpha = {"Authorization": f"Bearer {get_token(client, 'user_alpha', 'alphapass', role=Role.USER)}"}
    pdf_content = b"%PDF-1.4 sample file for alpha"
    files = [("files", ("alpha.pdf", io.BytesIO(pdf_content), "application/pdf"))]
    client.post("/me/documents", files=files, headers=auth_alpha)

    # 3. Query all logs as admin
    resp_all = client.get("/admin/logs/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp_all.status_code == 200
    logs_all = resp_all.json()["logs"]
    actions = [l["action"] for l in logs_all]
    assert "login" in actions
    assert "document_upload" in actions

    # 4. Filter by username=user_alpha
    resp_alpha = client.get("/admin/logs/users?username=user_alpha", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp_alpha.status_code == 200
    logs_alpha = resp_alpha.json()["logs"]
    assert len(logs_alpha) >= 2
    assert all(l["username"] == "user_alpha" for l in logs_alpha)

    # 5. Filter by username=user_beta
    resp_beta = client.get("/admin/logs/users?username=user_beta", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp_beta.status_code == 200
    logs_beta = resp_beta.json()["logs"]
    assert all(l["username"] == "user_beta" for l in logs_beta)
