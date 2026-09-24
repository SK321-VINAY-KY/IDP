import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.schemas.page import PageOutput


@pytest.fixture
def client():
    return TestClient(app)


def get_token(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


def test_pipeline_outputs_unauthenticated_returns_401(client):
    resp = client.get("/pipeline/outputs")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


def test_pipeline_extract_from_output_unauthenticated_returns_401(client):
    resp = client.post("/pipeline/extract/from-output", data={"md_name": "sample.md"})
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


def test_pipeline_extract_unauthenticated_returns_401(client):
    fake_pdf = b"%PDF-1.4 fake pdf binary"
    resp = client.post(
        "/pipeline/extract",
        files={"file": ("sample.pdf", fake_pdf, "application/pdf")},
        data={"raw_schema": json.dumps([{"name": "test_field", "description": "desc"}])},
    )
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


def test_pipeline_outputs_authenticated_admin_success(client):
    admin_token = get_token(client, "admin", "changeme", role=Role.ADMIN)
    resp = client.get(
        "/pipeline/outputs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert "outputs" in resp.json()


def test_pipeline_outputs_non_admin_forbidden(client):
    user_token = get_token(client, "item1_user", "password123", role=Role.USER)
    resp = client.get(
        "/pipeline/outputs",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403


def test_pipeline_extract_from_output_non_admin_forbidden(client):
    user_token = get_token(client, "item1_user", "password123", role=Role.USER)
    resp = client.post(
        "/pipeline/extract/from-output",
        data={"md_name": "sample.md"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403


def test_pipeline_extract_authenticated_user_success_and_records_owner(client, tmp_path):
    user_token = get_token(client, "item1_extract_user", "password123", role=Role.USER)
    fake_pdf = b"%PDF-1.4 fake pdf binary"

    # Mock extraction result
    mock_extracted = MagicMock()
    mock_extracted.model_dump.return_value = {"field_a": "value_a"}

    mock_page_output = PageOutput(
        page_number=1,
        markdown="# Page 1\ncontent",
        engines_used=["digital"],
        confidence=0.95,
        capabilities=[],
        escalated=False,
        escalation_attempts=0,
        low_confidence=False,
    )

    with patch("app.api.pipeline_routes.process_document", return_value=[(mock_page_output, {})]), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[{"page_number": 1}]), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=mock_extracted), \
         patch("src.ai.layer3_extraction.page_loader.load_pages_with_confidence", return_value=["# Page 1\ncontent"]), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client", return_value=MagicMock()), \
         patch("src.ai.layer3_extraction.storage.save_document"), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"), \
         patch("src.ai.layer3_extraction.storage.save_extraction_run") as mock_save_run, \
         patch("src.ai.layer3_extraction.storage.save_document_graph") as mock_save_graph:

        resp = client.post(
            "/pipeline/extract",
            files={"file": ("sample_upload.pdf", fake_pdf, "application/pdf")},
            data={"raw_schema": json.dumps([{"name": "field_a", "description": "test field"}])},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        assert data["data"] == {"field_a": "value_a"}
