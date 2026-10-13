"""Tenant-isolation regression test for schema_id -> schema resolution,
including the schema_registry/ DISK FALLBACK.

Threat model: owner A confirms a schema (schema_id S, owner=A). Owner B knows or
guesses S and requests it through every path that can reach the resolver + its
disk fallback. B must be DENIED (get None / 404), never receive A's schema.

Paths exercised (each must deny B):
  1. storage.get_schema_record(S, owner=B)            -- DB record branch
  2. storage.get_schema_record(S, owner=B)            -- DISK FALLBACK branch
                                                         (DB row absent, file on disk)
  3. pipeline_routes.load_strict_schema_record(S, owner=B)  -- web helper (DB)
  4. pipeline_routes.load_strict_schema_record(S, owner=B)  -- web helper DISK FALLBACK

A legitimately gets its own schema through every path (positive control), and a
NULL-owner (shared/admin) schema stays visible to everyone (documented behavior).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHATBOT = ROOT / "schema_chatbot_v2"
if str(CHATBOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT))

from src.ai.layer3_extraction.storage import (
    init_db,
    save_schema_record,
    get_schema_record,
    SessionLocal,
)

OWNER_A = "alice_schematest"
OWNER_B = "bob_schematest"
SCHEMA_ID_DB = "schema_isotest_db_a"       # lives in DB, owner A
SCHEMA_ID_DISK = "schema_isotest_disk_a"   # lives ONLY on disk, owner A
SCHEMA_ID_SHARED = "schema_isotest_shared"  # NULL owner -> shared

SAMPLE_SCHEMA_JSON = {
    "schema": {"fields": [{"name": "patient_name", "description": "Full name"}]},
    "document_type": "medical",
}


def _cleanup_db():
    try:
        from src.ai.layer3_extraction.storage import SchemaRecord
        s = SessionLocal()
        try:
            s.query(SchemaRecord).filter(
                SchemaRecord.schema_id.in_([SCHEMA_ID_DB, SCHEMA_ID_DISK, SCHEMA_ID_SHARED])
            ).delete(synchronize_session=False)
            s.commit()
        finally:
            s.close()
    except Exception:
        pass


@pytest.fixture()
def seeded(tmp_path_factory):
    """Seed: A's DB schema, A's disk-only schema, and a shared NULL-owner schema."""
    init_db()
    _cleanup_db()
    # A's schema in the DB (owner A)
    save_schema_record(
        schema_id=SCHEMA_ID_DB, document_type="medical",
        schema_json={**SAMPLE_SCHEMA_JSON, "owner": OWNER_A},
        owner=OWNER_A, schema_name="iso_db",
    )
    # A disk-only schema file (DB row absent) stamped owner A — forces the disk fallback.
    reg_dir = ROOT / "schema_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    disk_file = reg_dir / f"{SCHEMA_ID_DISK}.json"
    disk_file.write_text(json.dumps({**SAMPLE_SCHEMA_JSON, "owner": OWNER_A}), encoding="utf-8")
    # A shared schema with NULL owner (documented: visible to all)
    save_schema_record(
        schema_id=SCHEMA_ID_SHARED, document_type="medical",
        schema_json=dict(SAMPLE_SCHEMA_JSON), owner=None, schema_name="iso_shared",
    )
    yield
    _cleanup_db()
    try:
        disk_file.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path 1 & 2: storage.get_schema_record (DB branch + disk-fallback branch)
# ---------------------------------------------------------------------------

def test_get_schema_record_db_denies_cross_owner(seeded):
    assert get_schema_record(SCHEMA_ID_DB, owner=OWNER_A) is not None, "A must get its own DB schema"
    assert get_schema_record(SCHEMA_ID_DB, owner=OWNER_B) is None, \
        "LEAK: B received A's DB schema via get_schema_record"


def test_get_schema_record_disk_fallback_denies_cross_owner(seeded):
    # DB row absent for SCHEMA_ID_DISK -> this hits the disk fallback branch.
    assert get_schema_record(SCHEMA_ID_DISK, owner=OWNER_A) is not None, "A must get its own disk schema"
    assert get_schema_record(SCHEMA_ID_DISK, owner=OWNER_B) is None, \
        "LEAK: B received A's schema via the schema_registry DISK FALLBACK"


def test_get_schema_record_shared_null_owner_visible_to_all(seeded):
    # Documented behavior: NULL-owner schemas are shared/admin and visible to everyone.
    assert get_schema_record(SCHEMA_ID_SHARED, owner=OWNER_A) is not None
    assert get_schema_record(SCHEMA_ID_SHARED, owner=OWNER_B) is not None


# ---------------------------------------------------------------------------
# Path 3 & 4: pipeline_routes.load_strict_schema_record (web helper + its disk fallback)
# ---------------------------------------------------------------------------

def test_load_strict_schema_record_db_denies_cross_owner(seeded):
    from app.api.pipeline_routes import load_strict_schema_record
    from fastapi import HTTPException
    # A gets its own
    assert load_strict_schema_record(SCHEMA_ID_DB, owner=OWNER_A) is not None
    # B must be denied (404)
    with pytest.raises(HTTPException) as ei:
        load_strict_schema_record(SCHEMA_ID_DB, owner=OWNER_B)
    assert ei.value.status_code == 404, "LEAK: B resolved A's DB schema via load_strict_schema_record"


def test_load_strict_schema_record_disk_fallback_denies_cross_owner(seeded):
    from app.api.pipeline_routes import load_strict_schema_record
    from fastapi import HTTPException
    assert load_strict_schema_record(SCHEMA_ID_DISK, owner=OWNER_A) is not None
    with pytest.raises(HTTPException) as ei:
        load_strict_schema_record(SCHEMA_ID_DISK, owner=OWNER_B)
    assert ei.value.status_code == 404, \
        "LEAK: B resolved A's schema via load_strict_schema_record DISK FALLBACK"
