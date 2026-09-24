import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_token(client):
    store = get_user_store()
    if not store.get_by_username("admin"):
        store.create(username="admin", password="changeme", role=Role.ADMIN)
    resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    return resp.json()["access_token"]


@pytest.mark.parametrize("bad_name", [
    "../secret.md",
    "/etc/passwd",
    "a/b.md",
    "x.txt",
    "malicious.json",
])
def test_extract_from_output_path_traversal_rejected_with_400(client, admin_token, bad_name):
    # Create a dummy file outside dataset_output to ensure original code would find it if traversed
    secret_path = Path(__file__).resolve().parents[2] / "secret.md"
    secret_path.write_text("# Secret Markdown", encoding="utf-8")
    try:
        resp = client.post(
            "/pipeline/extract/from-output",
            data={"md_name": bad_name},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400, f"Expected 400 for bad md_name '{bad_name}', got {resp.status_code}: {resp.text}"
    finally:
        if secret_path.exists():
            secret_path.unlink()


@pytest.mark.parametrize("bad_schema_id", [
    "*",
    "../secret",
    "schema;drop",
    "../../etc/passwd",
    "schema*test",
])
def test_extract_uploaded_schema_id_invalid_format_rejected_with_400(client, admin_token, bad_schema_id):
    fake_pdf = b"%PDF-1.4 fake pdf"
    resp = client.post(
        "/pipeline/extract",
        files={"file": ("sample.pdf", fake_pdf, "application/pdf")},
        data={"schema_id": bad_schema_id},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 400, f"Expected 400 for invalid schema_id '{bad_schema_id}', got {resp.status_code}: {resp.text}"


def test_extract_uploaded_schema_id_missing_returns_404(client, admin_token):
    fake_pdf = b"%PDF-1.4 fake pdf"
    resp = client.post(
        "/pipeline/extract",
        files={"file": ("sample.pdf", fake_pdf, "application/pdf")},
        data={"schema_id": "nonexistent_schema_valid_format_999"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 404, f"Expected 404 for missing schema, got {resp.status_code}: {resp.text}"
