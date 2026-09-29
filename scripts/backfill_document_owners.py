"""
One-off backfill script to assign owners to existing ownerless documents
and DocumentGraphRecord entries from a supplied mapping.

USAGE:
    python scripts/backfill_document_owners.py --mapping mapping.json --default-owner admin [--apply]

NOTE: Runs in DRY-RUN mode by default. Pass --apply to persist changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ai.layer3_extraction.storage import (
    DocumentGraphRecord,
    SessionLocal,
    init_db,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill_owners")

ARTIFACT_SUFFIXES = (
    ".extracted.json",
    ".graph.json",
    ".schema_ref.json",
    ".md",
)


def _get_artifact_stem(filename: str) -> str:
    for sfx in ARTIFACT_SUFFIXES:
        if filename.endswith(sfx):
            return filename[:-len(sfx)]
    return Path(filename).stem


def backfill_document_owners(
    mapping: Dict[str, str],
    default_owner: Optional[str] = None,
    dataset_output_dir: Optional[Path] = None,
    apply_changes: bool = False,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Backfills owner on ownerless DocumentGraphRecord rows and disk json files,
    and moves legacy root artifacts (.extracted.json, .graph.json, .schema_ref.json, .md)
    into dataset_output/<owner>/.
    Never overwrites existing files (skips and reports collisions).
    Writes a JSON manifest of all moves and collisions.
    Applies all DB updates in one atomic transaction.
    Reruns are idempotent.
    Returns counts of updated records.
    """
    init_db()
    session = SessionLocal()
    counts: Dict[str, Any] = {
        "db_records_examined": 0,
        "db_records_updated": 0,
        "disk_extracted_examined": 0,
        "disk_extracted_updated": 0,
        "disk_schema_ref_examined": 0,
        "disk_schema_ref_updated": 0,
        "disk_artifacts_examined": 0,
        "disk_artifacts_moved": 0,
        "collisions_skipped": 0,
    }

    moves: List[Dict[str, Any]] = []
    collisions: List[Dict[str, Any]] = []
    db_updates: List[Dict[str, Any]] = []

    try:
        # 1. Backfill DB DocumentGraphRecord entries in one atomic transaction
        ownerless_query = session.query(DocumentGraphRecord).filter(
            (DocumentGraphRecord.owner == None) | (DocumentGraphRecord.owner == "")
        )
        ownerless_records = ownerless_query.all()
        counts["db_records_examined"] = len(ownerless_records)

        for rec in ownerless_records:
            doc_id = rec.doc_id or ""
            stem = Path(doc_id).stem
            assigned = mapping.get(doc_id) or mapping.get(stem) or default_owner
            if assigned:
                logger.info(
                    "[%s] DB Record ID=%s (doc_id=%s): assign owner -> %s",
                    "APPLY" if apply_changes else "DRY-RUN",
                    rec.id,
                    doc_id,
                    assigned,
                )
                db_updates.append({
                    "record_id": rec.id,
                    "doc_id": doc_id,
                    "assigned_owner": assigned,
                    "status": "updated" if apply_changes else "planned",
                })
                if apply_changes:
                    rec.owner = assigned
                counts["db_records_updated"] += 1
            else:
                logger.warning("No owner mapping found for DB doc_id: %s (id=%s)", doc_id, rec.id)

        if apply_changes and counts["db_records_updated"] > 0:
            try:
                session.commit()
            except Exception:
                session.rollback()
                raise

        # 2. Backfill and move disk artifacts if directory is provided
        if dataset_output_dir and dataset_output_dir.is_dir():
            root_artifacts = [
                p for p in dataset_output_dir.iterdir()
                if p.is_file() and any(p.name.endswith(sfx) for sfx in ARTIFACT_SUFFIXES)
            ]
            root_artifacts.sort(key=lambda p: p.name)

            for artifact_path in root_artifacts:
                fname = artifact_path.name
                stem = _get_artifact_stem(fname)
                counts["disk_artifacts_examined"] += 1
                if fname.endswith(".extracted.json"):
                    counts["disk_extracted_examined"] += 1
                elif fname.endswith(".schema_ref.json"):
                    counts["disk_schema_ref_examined"] += 1

                # Determine assigned owner
                assigned = None
                json_data = None
                if fname.endswith(".json"):
                    try:
                        json_data = json.loads(artifact_path.read_text(encoding="utf-8"))
                        if isinstance(json_data, dict):
                            assigned = json_data.get("owner")
                            if not assigned and json_data.get("source_pdf"):
                                assigned = mapping.get(json_data["source_pdf"])
                    except Exception as exc:
                        logger.error("Failed to read JSON %s: %s", fname, exc)

                if not assigned:
                    assigned = mapping.get(f"{stem}.pdf") or mapping.get(stem)

                if not assigned:
                    try:
                        db_rec = session.query(DocumentGraphRecord).filter(
                            DocumentGraphRecord.doc_id.in_([f"{stem}.pdf", stem])
                        ).first()
                        if db_rec and db_rec.owner:
                            assigned = db_rec.owner
                    except Exception:
                        pass

                if not assigned:
                    assigned = default_owner

                if assigned:
                    target_dir = dataset_output_dir / assigned
                    dest_file = target_dir / fname

                    # Never overwrite: check for existing destination file
                    if dest_file.exists():
                        collision_info = {
                            "source": str(artifact_path.resolve()),
                            "destination": str(dest_file.resolve()),
                            "owner": assigned,
                            "status": "skipped_collision",
                            "reason": f"Destination file already exists: {dest_file.name}",
                        }
                        collisions.append(collision_info)
                        counts["collisions_skipped"] += 1
                        logger.warning(
                            "[%s] Collision: destination %s already exists. Skipping move of %s",
                            "APPLY" if apply_changes else "DRY-RUN",
                            dest_file,
                            artifact_path,
                        )
                        continue

                    if fname.endswith(".extracted.json"):
                        counts["disk_extracted_updated"] += 1
                    elif fname.endswith(".schema_ref.json"):
                        counts["disk_schema_ref_updated"] += 1

                    logger.info(
                        "[%s] Root artifact %s: assign owner -> %s and move to %s/%s",
                        "APPLY" if apply_changes else "DRY-RUN",
                        fname,
                        assigned,
                        assigned,
                        fname,
                    )

                    move_info = {
                        "source": str(artifact_path.resolve()),
                        "destination": str(dest_file.resolve()),
                        "owner": assigned,
                        "status": "moved" if apply_changes else "planned",
                    }
                    moves.append(move_info)

                    if apply_changes:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        if fname.endswith(".json") and json_data is not None:
                            if json_data.get("owner") != assigned:
                                json_data["owner"] = assigned
                                artifact_path.write_text(json.dumps(json_data, indent=2) + "\n", encoding="utf-8")

                        shutil.move(str(artifact_path), str(dest_file))
                        counts["disk_artifacts_moved"] += 1
                else:
                    logger.warning("No owner assigned for root artifact: %s", fname)

    finally:
        session.close()

    # Build and write manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "apply_changes": apply_changes,
        "summary": counts,
        "moves": moves,
        "collisions": collisions,
        "db_updates": db_updates,
    }

    actual_manifest_path = manifest_path
    if not actual_manifest_path and dataset_output_dir and dataset_output_dir.is_dir():
        actual_manifest_path = dataset_output_dir / "backfill_manifest.json"

    if actual_manifest_path:
        try:
            actual_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            actual_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            logger.info("Wrote backfill manifest to %s", actual_manifest_path)
            counts["manifest_path"] = str(actual_manifest_path)
        except Exception as m_err:
            logger.error("Failed to write backfill manifest to %s: %s", actual_manifest_path, m_err)

    logger.info("Summary of backfill operation: %s", json.dumps(counts, indent=2))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill document owners for IDP")
    parser.add_argument("--mapping", type=Path, help="JSON file mapping doc_id/stem to owner username")
    parser.add_argument("--default-owner", type=str, default="admin", help="Fallback owner for unmapped documents (default: admin)")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_output"), help="Path to dataset_output directory")
    parser.add_argument("--apply", action="store_true", help="Persist updates (defaults to DRY-RUN mode if omitted)")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to write JSON manifest (default: <output-dir>/backfill_manifest.json)")

    args = parser.parse_args()

    mapping: Dict[str, str] = {}
    if args.mapping:
        if not args.mapping.is_file():
            raise FileNotFoundError(f"Mapping file not found: {args.mapping}")
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    backfill_document_owners(
        mapping=mapping,
        default_owner=args.default_owner,
        dataset_output_dir=args.output_dir,
        apply_changes=args.apply,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
