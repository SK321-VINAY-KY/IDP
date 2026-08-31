import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_login_admin_success(client):
    # Seeded admin is admin/changeme by default
    resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"

    # Test /auth/me with the token
    token = data["access_token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == "admin"
    assert me_data["role"] == "admin"


def test_auth_login_invalid_password(client):
    resp = client.post("/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert resp.status_code == 401
    assert "Incorrect username or password" in resp.json()["detail"]


def test_auth_login_nonexistent_user(client):
    resp = client.post("/auth/login", data={"username": "does_not_exist", "password": "any"})
    assert resp.status_code == 401


def test_auth_me_without_token(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_auth_me_with_invalid_token(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.value"})
    assert resp.status_code == 401


def test_regular_user_auth(client):
    store = get_user_store()
    try:
        store.create(username="charlie", password="charliepassword", role=Role.USER)
    except ValueError:
        pass  # already created

    resp = client.post("/auth/login", data={"username": "charlie", "password": "charliepassword"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "user"

    token = data["access_token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "charlie"
    assert me_resp.json()["role"] == "user"
