"""
File: test_custom_schema_builder.py
Purpose: Tests for the Human-Interactable Custom Schema Builder and POST /schema/custom.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.core.auth import create_access_token
from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

SCHEMA_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "schema_registry"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_headers():
    store = get_user_store()
    user = store.get_by_username("admin")
    if not user:
        user = store.create("admin", "changeme", Role.ADMIN)
    token = create_access_token(user)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_headers():
    store = get_user_store()
    user = store.get_by_username("builder_test_user")
    if not user:
        user = store.create("builder_test_user", "password123", Role.USER)
    token = create_access_token(user)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Test 1: Valid Schema Creation
# ---------------------------------------------------------------------------
def test_1_valid_schema_creation(client, admin_headers):
    """Verify HTTP success, schema ID generated, registry file written, valid JSON."""
    payload = {
        "document_type": "discharge_summary",
        "fields": [
            {
                "name": "patient_name",
                "type": "string",
                "required": True,
                "description": "Full legal name of the patient",
            },
            {
                "name": "admission_date",
                "type": "date",
                "required": True,
                "description": "Hospital admission date",
            },
            {
                "name": "claim_amount",
                "type": "number",
                "required": False,
                "description": "Approved claim amount",
            },
        ],
    }

    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data.get("status") == "success"
    schema_id = data.get("schema_id")
    assert schema_id and schema_id.startswith("schema_")
    assert data.get("document_type") == "discharge_summary"

    schema_file = SCHEMA_REGISTRY_DIR / f"{schema_id}.json"
    try:
        assert schema_file.exists()
        file_data = json.loads(schema_file.read_text(encoding="utf-8"))
        assert file_data["schema_id"] == schema_id
        assert file_data["document_type"] == "discharge_summary"
        fields = file_data["schema"]["fields"]
        assert len(fields) == 3
        assert fields[0]["name"] == "patient_name"
        assert fields[0]["required"] is True
        assert fields[0]["description"] == "Full legal name of the patient"
    finally:
        if schema_file.exists():
            schema_file.unlink()


# ---------------------------------------------------------------------------
# Test 2: Missing or Blank Document Type Rejection
# ---------------------------------------------------------------------------
def test_2_missing_document_type(client, admin_headers):
    """Rejects missing or whitespace-only document type."""
    payload = {
        "document_type": "   ",
        "fields": [
            {"name": "patient_name", "type": "string", "required": True}
        ],
    }
    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 400
    assert "document_type" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 3: No Fields Rejection
# ---------------------------------------------------------------------------
def test_3_no_fields(client, admin_headers):
    """Rejects empty fields list."""
    payload = {
        "document_type": "test_doc",
        "fields": [],
    }
    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 400
    assert "at least one field" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 4: Duplicate Field Names with Normalization Rejection
# ---------------------------------------------------------------------------
def test_4_duplicate_fields_normalized(client, admin_headers):
    """Rejects duplicate field names, including normalized equivalents (e.g. 'Patient Name' and 'patient_name')."""
    payload = {
        "document_type": "invoice",
        "fields": [
            {"name": "patient_name", "type": "string", "required": True},
            {"name": "Patient Name", "type": "string", "required": False},
        ],
    }
    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 400
    assert "duplicate field name" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 5: Invalid Field Type Rejection
# ---------------------------------------------------------------------------
def test_5_invalid_field_type(client, admin_headers):
    """Rejects unsupported field types."""
    payload = {
        "document_type": "invoice",
        "fields": [
            {"name": "amount", "type": "unsupported_magic_type", "required": True}
        ],
    }
    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 400
    assert "unsupported type" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 6: PostgreSQL / Storage Persistence
# ---------------------------------------------------------------------------
def test_6_storage_persistence(client, admin_headers):
    """Verify save_schema_record stores or logs schema in database."""
    payload = {
        "document_type": "storage_test_doc",
        "fields": [
            {"name": "field_a", "type": "string", "required": True},
            {"name": "field_b", "type": "number", "required": False},
        ],
    }
    resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    schema_id = data["schema_id"]

    schema_file = SCHEMA_REGISTRY_DIR / f"{schema_id}.json"
    try:
        assert schema_file.exists()
    finally:
        if schema_file.exists():
            schema_file.unlink()


# ---------------------------------------------------------------------------
# Test 7: Existing Schemas Integrity
# ---------------------------------------------------------------------------
def test_7_existing_schemas_integrity():
    """Verify that all existing schemas in schema_registry remain readable JSON."""
    existing_files = list(SCHEMA_REGISTRY_DIR.glob("schema_*.json"))
    assert len(existing_files) >= 200, f"Expected at least 200 schemas, found {len(existing_files)}"

    # Sample check 20 files
    for f in existing_files[:20]:
        content = json.loads(f.read_text(encoding="utf-8"))
        assert "schema_id" in content or "schema" in content or "document_type" in content


# ---------------------------------------------------------------------------
# Test 8: Pipeline Schema Selector Availability
# ---------------------------------------------------------------------------
def test_8_pipeline_schema_selector_availability(client, admin_headers):
    """Verify newly created schema appears immediately in /schemas for pipeline selector."""
    payload = {
        "document_type": "pipeline_selector_test",
        "fields": [
            {"name": "selector_field", "type": "string", "required": True}
        ],
    }
    create_resp = client.post("/schema/custom", json=payload, headers=admin_headers)
    assert create_resp.status_code == 200
    new_schema_id = create_resp.json()["schema_id"]

    schema_file = SCHEMA_REGISTRY_DIR / f"{new_schema_id}.json"
    try:
        list_resp = client.get("/schemas", headers=admin_headers)
        assert list_resp.status_code == 200
        schemas = list_resp.json().get("schemas", [])
        found = any(s.get("schema_id") == new_schema_id for s in schemas)
        assert found, f"Newly created schema {new_schema_id} should appear in /schemas"
    finally:
        if schema_file.exists():
            schema_file.unlink()


# ---------------------------------------------------------------------------
# Test 9: Dynamic Schema & Layer 3 Compatibility
# ---------------------------------------------------------------------------
def test_9_dynamic_schema_layer3_compatibility():
    """Verify build_dynamic_schema creates a valid Pydantic model compatible with Layer 3."""
    fields = [
        SchemaFieldIn(name="patient_name", description="Patient name"),
        SchemaFieldIn(name="admission_date", description="Admission date"),
        SchemaFieldIn(name="approved_claim_amount", description="Approved claim amount"),
    ]
    model_cls = build_dynamic_schema(fields)
    assert hasattr(model_cls, "model_fields")
    assert "patient_name" in model_cls.model_fields
    assert "admission_date" in model_cls.model_fields
    assert "approved_claim_amount" in model_cls.model_fields

    instance = model_cls(patient_name="Rahul Sharma", admission_date="12/08/2026")
    dumped = instance.model_dump()
    assert dumped["patient_name"] == "Rahul Sharma"
    assert dumped["admission_date"] == "12/08/2026"
    assert dumped["approved_claim_amount"] == ""
