import io
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.api.pipeline_routes import reset_extract_rate_limits

client = TestClient(app)


def get_token(username, role=Role.USER):
    store = get_user_store()
    user = store.get_by_username(username)
    if not user:
        store.create(username=username, password="password123", role=role)
    resp = client.post("/auth/login", data={"username": username, "password": "password123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(autouse=True)
def clean_rate_limits():
    reset_extract_rate_limits()
    yield
    reset_extract_rate_limits()


def test_me_documents_enforces_size_limit():
    token = get_token("user_limits_1")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.api.pipeline_routes.MAX_UPLOAD_SIZE_BYTES", 100):
        oversized = b"%PDF-1.4" + b"X" * 200
        files = [("files", ("oversized.pdf", io.BytesIO(oversized), "application/pdf"))]
        resp = client.post("/me/documents", headers=headers, files=files)
        assert resp.status_code == 413
        assert "maximum allowed upload size" in resp.json()["detail"]


def test_me_documents_enforces_rate_limit():
    token = get_token("user_limits_2")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.api.pipeline_routes.RATE_LIMIT_MAX_REQUESTS", 2), \
         patch("app.api.pipeline_routes.RATE_LIMIT_WINDOW_SECONDS", 60):
        small = b"%PDF-1.4 sample content"
        for _ in range(2):
            files = [("files", ("test.pdf", io.BytesIO(small), "application/pdf"))]
            r = client.post("/me/documents", headers=headers, files=files)
            assert r.status_code == 200

        # 3rd request must be blocked by rate limiter
        files = [("files", ("test.pdf", io.BytesIO(small), "application/pdf"))]
        r3 = client.post("/me/documents", headers=headers, files=files)
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]


def test_me_pipeline_run_enforces_rate_limit():
    token = get_token("user_limits_3")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.api.pipeline_routes.RATE_LIMIT_MAX_REQUESTS", 2), \
         patch("app.api.pipeline_routes.RATE_LIMIT_WINDOW_SECONDS", 60):
        for _ in range(2):
            # Fails fast due to no documents, but consumes rate limit quota
            r = client.post("/me/pipeline/run", headers=headers)
            assert r.status_code in (200, 400)

        # 3rd request must be blocked with 429
        r3 = client.post("/me/pipeline/run", headers=headers)
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]


def test_schema_infer_enforces_size_limit():
    token = get_token("user_limits_4")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.api.pipeline_routes.MAX_UPLOAD_SIZE_BYTES", 100):
        oversized = b"%PDF-1.4" + b"Y" * 200
        normal = b"%PDF-1.4 valid content"
        files = [
            ("files", ("oversized.pdf", io.BytesIO(oversized), "application/pdf")),
            ("files", ("sample2.pdf", io.BytesIO(normal), "application/pdf")),
        ]
        resp = client.post("/schema/infer", headers=headers, files=files)
        assert resp.status_code == 413
        assert "maximum allowed upload size" in resp.json()["detail"]
