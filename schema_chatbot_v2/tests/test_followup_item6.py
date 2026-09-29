import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.storage.user_store import get_user_store, Role

client = TestClient(app)


def get_token(username="item6_user", password="password123", role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


def test_extract_rejects_path_traversal_filename():
    """Prove that filename with path traversal is rejected."""
    token = get_token("item6_user1")
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("../../evil.pdf", io.BytesIO(b"%PDF-1.4 test"), "application/pdf")}
    data = {"raw_schema": '{"fields": ["field1"]}'}

    resp = client.post("/pipeline/extract", headers=headers, files=files, data=data)
    assert resp.status_code == 400
    assert "Path traversal" in resp.json()["detail"] or "Invalid filename" in resp.json()["detail"]


def test_extract_rejects_exceeded_file_size():
    """Prove that file exceeding max upload size is rejected with 413."""
    token = get_token("item6_user2")
    headers = {"Authorization": f"Bearer {token}"}
    # Mock max size to 100 bytes for test
    with patch("app.api.pipeline_routes.MAX_UPLOAD_SIZE_BYTES", 100):
        oversized = b"%PDF-1.4" + b"A" * 200
        files = {"file": ("large.pdf", io.BytesIO(oversized), "application/pdf")}
        data = {"raw_schema": '{"fields": ["field1"]}'}

        resp = client.post("/pipeline/extract", headers=headers, files=files, data=data)
        assert resp.status_code == 413
        assert "maximum allowed upload size" in resp.json()["detail"]


def test_extract_enforces_rate_limit():
    """Prove that sending requests beyond limit returns 429 Too Many Requests."""
    token = get_token("item6_user3")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.api.pipeline_routes.RATE_LIMIT_MAX_REQUESTS", 2), \
         patch("app.api.pipeline_routes.RATE_LIMIT_WINDOW_SECONDS", 60):
        # We need extract to fail fast or succeed, so pass an invalid raw_schema to consume quota
        # But rate limiter triggers first
        for i in range(2):
            resp = client.post(
                "/pipeline/extract",
                headers=headers,
                files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 sample"), "application/pdf")},
                data={"raw_schema": "invalid json"},
            )
            assert resp.status_code == 400

        # 3rd request should hit rate limit
        resp3 = client.post(
            "/pipeline/extract",
            headers=headers,
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 sample"), "application/pdf")},
            data={"raw_schema": "invalid json"},
        )
        assert resp3.status_code == 429
        assert "Rate limit exceeded" in resp3.json()["detail"]
