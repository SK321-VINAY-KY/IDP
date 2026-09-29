import json
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.layer3_extraction.storage import Base, DocumentGraphRecord
from scripts.backfill_document_owners import backfill_document_owners


def test_backfill_never_overwrites_collisions_and_writes_manifest(tmp_path, monkeypatch):
    test_db_url = f"sqlite:///{tmp_path / 'test_backfill_col.db'}"
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    monkeypatch.setattr("scripts.backfill_document_owners.SessionLocal", TestSession)
    monkeypatch.setattr("scripts.backfill_document_owners.init_db", lambda: None)

    # Insert DB record
    s = TestSession()
    r1 = DocumentGraphRecord(doc_id="doc1.pdf", owner=None, graph_json={"nodes": []})
    r2 = DocumentGraphRecord(doc_id="doc2.pdf", owner=None, graph_json={"nodes": []})
    s.add_all([r1, r2])
    s.commit()
    s.close()

    out_dir = tmp_path / "dataset_output"
    out_dir.mkdir()

    # Create root artifact doc1.md and doc2.md
    (out_dir / "doc1.md").write_text("# New Root Content for Doc 1")
    (out_dir / "doc2.md").write_text("# Doc 2 Root Content")

    # Create a pre-existing destination file for doc1 under user1 to trigger collision
    user1_dir = out_dir / "user1"
    user1_dir.mkdir()
    colliding_dest = user1_dir / "doc1.md"
    colliding_dest.write_text("# Original Existing Destination Content")

    mapping = {"doc1.pdf": "user1", "doc2.pdf": "user1"}

    # Run backfill with apply_changes=True
    counts = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=True,
    )

    # 1. Collision check: doc1.md must NOT overwrite existing file
    assert colliding_dest.read_text() == "# Original Existing Destination Content", "Existing file was overwritten!"
    assert (out_dir / "doc1.md").is_file(), "Colliding source file should be preserved in root"
    assert counts.get("collisions_skipped") == 1, "Collision was not reported in counts"

    # doc2.md should be moved normally
    assert (user1_dir / "doc2.md").is_file()
    assert not (out_dir / "doc2.md").exists()
    assert counts.get("disk_artifacts_moved") == 1

    # 2. Manifest check: manifest file must exist and contain moves + collisions
    manifest_path = out_dir / "backfill_manifest.json"
    assert manifest_path.is_file(), "Manifest file was not written"
    manifest_data = json.loads(manifest_path.read_text())
    assert "timestamp" in manifest_data
    assert len(manifest_data["collisions"]) == 1
    assert "doc1.md" in manifest_data["collisions"][0]["source"]
    assert len(manifest_data["moves"]) == 1
    assert "doc2.md" in manifest_data["moves"][0]["source"]
    assert len(manifest_data["db_updates"]) == 2

    # 3. Idempotent rerun check: running again must succeed without errors, 0 updates, 0 moves
    counts_rerun = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=True,
    )
    assert counts_rerun["db_records_updated"] == 0, "Rerun updated DB records that were already backfilled"
    assert counts_rerun["disk_artifacts_moved"] == 0, "Rerun moved files that were already partitioned"
