import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import get_user_store, Role

client = TestClient(app)


def get_token(username="testuser", password="password123", role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


def test_query_bot_ask_rejects_path_traversal_doc_id():
    """Prove that ../ in doc_id is rejected with 400 Path traversal detected."""
    token = get_token("traversal_user")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/query-bot/ask",
        json={"doc_id": "../../etc/passwd", "question": "What is in this file?"},
        headers=headers,
    )
    assert resp.status_code == 400
    assert "Path traversal detected" in resp.json()["detail"]


def test_query_bot_ask_rejects_absolute_path_doc_id():
    """Prove that absolute paths in doc_id are rejected with 400 Path traversal detected."""
    token = get_token("traversal_user")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/query-bot/ask",
        json={"doc_id": "/etc/shadow", "question": "What is in this file?"},
        headers=headers,
    )
    assert resp.status_code == 400
    assert "Path traversal detected" in resp.json()["detail"]
