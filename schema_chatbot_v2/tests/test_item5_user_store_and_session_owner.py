import os
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store, reset_user_store
from app.storage.session_store import get_session_store, reset_session_store


@pytest.fixture
def client():
    return TestClient(app)


def test_production_staging_refuses_memory_and_file_user_store(monkeypatch):
    reset_user_store()
    monkeypatch.setenv("IDP_APP_ENV", "production")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecurePassword123!")

    # In production, memory must be refused
    monkeypatch.setenv("USER_STORE_TYPE", "memory")
    reset_user_store()
    with pytest.raises(ValueError) as exc1:
        get_user_store()
    assert "refused in production/staging" in str(exc1.value).lower() or "cannot be 'memory'" in str(exc1.value).lower()

    # In production, file must be refused
    monkeypatch.setenv("USER_STORE_TYPE", "file")
    reset_user_store()
    with pytest.raises(ValueError) as exc2:
        get_user_store()
    assert "refused in production/staging" in str(exc2.value).lower() or "cannot be 'file'" in str(exc2.value).lower()

    # In development, memory is accepted
    monkeypatch.setenv("IDP_APP_ENV", "development")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("USER_STORE_TYPE", "memory")
    reset_user_store()
    dev_store = get_user_store()
    assert dev_store is not None
    reset_user_store()


def test_session_id_is_valid_uuid4_and_owner_checked_on_read_write(client):
    store = get_user_store()
    if not store.get_by_username("user_a"):
        store.create("user_a", "password_a", role=Role.USER)
    if not store.get_by_username("user_b"):
        store.create("user_b", "password_b", role=Role.USER)
    if not store.get_by_username("admin_item5"):
        store.create("admin_item5", "admin_pass", role=Role.ADMIN)

    token_a = client.post("/auth/login", data={"username": "user_a", "password": "password_a"}).json()["access_token"]
    token_b = client.post("/auth/login", data={"username": "user_b", "password": "password_b"}).json()["access_token"]
    token_admin = client.post("/auth/login", data={"username": "admin_item5", "password": "admin_pass"}).json()["access_token"]

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    headers_admin = {"Authorization": f"Bearer {token_admin}"}

    # User A starts session
    resp = client.post("/chat", json={"session_id": None, "message": None}, headers=headers_a)
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Confirm session_id is a valid UUID4
    parsed_uuid = uuid.UUID(session_id, version=4)
    assert str(parsed_uuid) == session_id

    # User B attempts to READ User A's session -> 403
    resp_get = client.post(f"/session/{session_id}", headers=headers_b) if False else client.get(f"/session/{session_id}", headers=headers_b)
    assert resp_get.status_code == 403, f"Expected 403, got {resp_get.status_code}"

    # User B attempts to WRITE / continue User A's session -> 403
    resp_chat = client.post("/chat", json={"session_id": session_id, "message": "hello"}, headers=headers_b)
    assert resp_chat.status_code == 403, f"Expected 403, got {resp_chat.status_code}"

    # User B attempts to update schema on User A's session -> 403
    resp_schema = client.post(f"/session/{session_id}/schema", json={"document_type": "invoice", "fields": []}, headers=headers_b)
    assert resp_schema.status_code == 403, f"Expected 403, got {resp_schema.status_code}"

    # User B attempts to reset User A's session -> 403
    resp_reset = client.post(f"/session/{session_id}/reset", headers=headers_b)
    assert resp_reset.status_code == 403, f"Expected 403, got {resp_reset.status_code}"

    # Admin CAN read User A's session -> 200
    resp_admin = client.get(f"/session/{session_id}", headers=headers_admin)
    assert resp_admin.status_code == 200
