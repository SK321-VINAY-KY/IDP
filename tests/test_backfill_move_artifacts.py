import json
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.layer3_extraction.storage import Base, DocumentGraphRecord
from scripts.backfill_document_owners import backfill_document_owners


def test_backfill_moves_legacy_root_artifacts_to_owner_dir(tmp_path, monkeypatch):
    # Setup test sqlite db
    test_db_url = f"sqlite:///{tmp_path / 'test_backfill_move.db'}"
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    monkeypatch.setattr("scripts.backfill_document_owners.SessionLocal", TestSession)
    monkeypatch.setattr("scripts.backfill_document_owners.init_db", lambda: None)

    out_dir = tmp_path / "dataset_output"
    out_dir.mkdir()

    # Create root artifacts for doc1 (mapped to user1) and doc2 (fallback to default admin)
    (out_dir / "doc1.extracted.json").write_text(json.dumps({"source_pdf": "doc1.pdf", "data": 1}))
    (out_dir / "doc1.graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
    (out_dir / "doc1.schema_ref.json").write_text(json.dumps({"source_pdf": "doc1.pdf"}))
    (out_dir / "doc1.md").write_text("# Doc 1 Markdown Content")

    (out_dir / "doc2.extracted.json").write_text(json.dumps({"source_pdf": "doc2.pdf", "data": 2}))
    (out_dir / "doc2.md").write_text("# Doc 2 Markdown Content")

    mapping = {"doc1.pdf": "user1"}

    # 1. Dry run: verify files are NOT moved
    res_dry = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=False,
    )

    assert (out_dir / "doc1.extracted.json").is_file()
    assert (out_dir / "doc1.graph.json").is_file()
    assert (out_dir / "doc1.schema_ref.json").is_file()
    assert (out_dir / "doc1.md").is_file()
    assert (out_dir / "doc2.extracted.json").is_file()
    assert (out_dir / "doc2.md").is_file()
    assert not (out_dir / "user1").exists()
    assert not (out_dir / "admin").exists()

    # 2. Apply: verify all legacy root artifacts are MOVED into owner subdirs
    res_apply = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=True,
    )

    # Root directory must have NO legacy artifacts left
    root_artifacts = [
        p.name for p in out_dir.iterdir()
        if p.is_file() and any(p.name.endswith(sfx) for sfx in [".extracted.json", ".graph.json", ".schema_ref.json", ".md"])
    ]
    assert root_artifacts == [], f"Expected root artifacts to be moved, but found: {root_artifacts}"

    # user1 partition
    user1_dir = out_dir / "user1"
    assert (user1_dir / "doc1.extracted.json").is_file()
    assert (user1_dir / "doc1.graph.json").is_file()
    assert (user1_dir / "doc1.schema_ref.json").is_file()
    assert (user1_dir / "doc1.md").is_file()

    # admin partition
    admin_dir = out_dir / "admin"
    assert (admin_dir / "doc2.extracted.json").is_file()
    assert (admin_dir / "doc2.md").is_file()

    # Check that owner was written into JSON files
    assert json.loads((user1_dir / "doc1.extracted.json").read_text())["owner"] == "user1"
    assert json.loads((admin_dir / "doc2.extracted.json").read_text())["owner"] == "admin"
