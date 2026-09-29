import json
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.layer3_extraction.storage import Base, DocumentGraphRecord
from scripts.backfill_document_owners import backfill_document_owners


def test_backfill_document_owners_dry_run_and_apply(tmp_path, monkeypatch):
    # Setup test sqlite db
    test_db_url = f"sqlite:///{tmp_path / 'test_backfill.db'}"
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    # Monkeypatch storage SessionLocal and init_db
    monkeypatch.setattr("scripts.backfill_document_owners.SessionLocal", TestSession)
    monkeypatch.setattr("scripts.backfill_document_owners.init_db", lambda: None)

    # Insert test records: 2 ownerless, 1 with owner
    s = TestSession()
    r1 = DocumentGraphRecord(doc_id="invoice1.pdf", owner=None, graph_json={"nodes": []})
    r2 = DocumentGraphRecord(doc_id="report2.pdf", owner="", graph_json={"nodes": []})
    r3 = DocumentGraphRecord(doc_id="claim3.pdf", owner="existing_user", graph_json={"nodes": []})
    s.add_all([r1, r2, r3])
    s.commit()
    s.close()

    # Create dummy output directory with .extracted.json and .schema_ref.json
    out_dir = tmp_path / "dataset_output"
    out_dir.mkdir()
    ext_file = out_dir / "invoice1.extracted.json"
    ext_file.write_text(json.dumps({"source_pdf": "invoice1.pdf", "extracted_data": {}}))
    ref_file = out_dir / "report2.schema_ref.json"
    ref_file.write_text(json.dumps({"source_pdf": "report2.pdf"}))

    mapping = {"invoice1.pdf": "finance_team"}

    # 1. Dry run
    res_dry = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=False,
    )
    assert res_dry["db_records_examined"] == 2
    assert res_dry["db_records_updated"] == 2
    assert res_dry["disk_extracted_updated"] == 1
    assert res_dry["disk_schema_ref_updated"] == 1

    # Verify dry-run didn't mutate DB or files
    s = TestSession()
    assert s.query(DocumentGraphRecord).filter_by(doc_id="invoice1.pdf").first().owner is None
    s.close()
    assert "owner" not in json.loads(ext_file.read_text())

    # 2. Apply
    res_apply = backfill_document_owners(
        mapping=mapping,
        default_owner="admin",
        dataset_output_dir=out_dir,
        apply_changes=True,
    )
    assert res_apply["db_records_updated"] == 2

    # Verify changes persisted
    s = TestSession()
    assert s.query(DocumentGraphRecord).filter_by(doc_id="invoice1.pdf").first().owner == "finance_team"
    assert s.query(DocumentGraphRecord).filter_by(doc_id="report2.pdf").first().owner == "admin"
    assert s.query(DocumentGraphRecord).filter_by(doc_id="claim3.pdf").first().owner == "existing_user"
    s.close()

    moved_ext = out_dir / "finance_team" / "invoice1.extracted.json"
    moved_ref = out_dir / "admin" / "report2.schema_ref.json"
    assert moved_ext.is_file()
    assert moved_ref.is_file()
    assert json.loads(moved_ext.read_text())["owner"] == "finance_team"
    assert json.loads(moved_ref.read_text())["owner"] == "admin"
