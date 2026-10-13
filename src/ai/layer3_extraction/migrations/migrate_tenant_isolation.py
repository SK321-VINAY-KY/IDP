"""
Database migration script: Tenant Isolation (Audit Finding #4).

Key responsibilities:
1. Replaces global uniqueness on documents.filename with composite (owner, filename).
2. Adds owner NOT NULL to documents, document_markdowns, extraction_runs, page_checkpoints.
3. Drops legacy unique constraints / indexes from Postgres catalog and SQLite table rebuilds.
4. Backfills ownerless legacy rows before quarantining:
   - Inspects dataset_output/<owner>/ directories and registry files.
   - If exactly one owner matches: backfills with that owner.
   - If ambiguous (>1 owner) or unrecoverable (0 owners): quarantines to sentinel '_legacy'.
5. Idempotent: safe to run multiple times with zero changes on subsequent runs.
6. Refuses to run without --backup-acknowledged flag (unless in --dry-run mode).
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SENTINEL_LEGACY = "_legacy"
SENTINEL_ADMIN = "_admin"

DATASET_OUTPUT_DIR = Path("dataset_output")
SCHEMA_REGISTRY_DIR = Path("schema_registry")


def get_candidate_owners_from_filesystem(doc_id: str) -> Set[str]:
    """
    Search dataset_output/ and schema_registry/ for candidate owners of a document.
    """
    candidates: Set[str] = set()
    stem = Path(doc_id).stem
    doc_variants = {doc_id.lower(), stem.lower(), f"{stem}.pdf".lower(), f"{stem}.md".lower()}

    # 1. Check dataset_output/<owner>/ subdirectories
    if DATASET_OUTPUT_DIR.exists():
        for owner_dir in DATASET_OUTPUT_DIR.iterdir():
            if owner_dir.is_dir() and not owner_dir.name.startswith(("_", ".")):
                # Inspect files inside this owner's directory
                for f in owner_dir.iterdir():
                    f_stem = f.stem.replace(".extracted", "").replace(".graph", "").replace(".schema_ref", "")
                    if f.name.lower() in doc_variants or f_stem.lower() == stem.lower():
                        candidates.add(owner_dir.name)
                        break

        # 2. Check dataset_output/ root JSON files for explicit 'owner' field
        for json_file in [
            DATASET_OUTPUT_DIR / f"{stem}.extracted.json",
            DATASET_OUTPUT_DIR / f"{stem}.schema_ref.json",
            DATASET_OUTPUT_DIR / f"{doc_id}.extracted.json",
        ]:
            if json_file.exists():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
                    owner_val = data.get("owner")
                    if owner_val and isinstance(owner_val, str) and not owner_val.startswith("_"):
                        candidates.add(owner_val.strip())
                except Exception:
                    pass

    # 3. Check schema_registry/ for schemas with owner that reference this document
    if SCHEMA_REGISTRY_DIR.exists():
        for reg_file in SCHEMA_REGISTRY_DIR.glob("*.json"):
            try:
                sdata = json.loads(reg_file.read_text(encoding="utf-8", errors="replace"))
                s_owner = sdata.get("owner")
                if s_owner and isinstance(s_owner, str) and not s_owner.startswith("_"):
                    samples = sdata.get("sample_documents") or []
                    for s in samples:
                        s_stem = Path(s).stem
                        if s.lower() in doc_variants or s_stem.lower() == stem.lower():
                            candidates.add(s_owner.strip())
            except Exception:
                pass

    return candidates


def resolve_owner_for_doc(
    doc_identifier: str,
    db_owner_candidates: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    """
    Resolve the owner for an ownerless row.
    Returns (resolved_owner, resolution_status):
    resolution_status is one of: 'backfilled', 'ambiguous', 'quarantined'
    """
    fs_candidates = get_candidate_owners_from_filesystem(doc_identifier)
    all_candidates = set(fs_candidates)
    if db_owner_candidates:
        all_candidates.update({c for c in db_owner_candidates if c and not c.startswith("_")})

    if len(all_candidates) == 1:
        return next(iter(all_candidates)), "backfilled"
    elif len(all_candidates) > 1:
        return SENTINEL_LEGACY, "ambiguous"
    else:
        return SENTINEL_LEGACY, "quarantined"


# ==============================================================================
# SQLite Migration Implementation
# ==============================================================================

def migrate_sqlite(db_file: Path, dry_run: bool = False) -> Dict[str, Any]:
    print(f"\n[SQLite] Connecting to {db_file.resolve()}...")
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    stats = {
        "engine": "sqlite",
        "dry_run": dry_run,
        "backfilled": 0,
        "ambiguous": 0,
        "quarantined": 0,
        "tables_rebuilt": [],
        "already_migrated": False,
    }

    try:
        # Check if already migrated
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
        if not cur.fetchone():
            print("[SQLite] 'documents' table does not exist. Nothing to migrate.")
            stats["already_migrated"] = True
            return stats

        cur.execute("PRAGMA table_info(documents);")
        columns = [row["name"] for row in cur.fetchall()]
        has_owner = "owner" in columns

        cur.execute("PRAGMA index_list(documents);")
        indexes = cur.fetchall()
        has_composite_uq = False
        for idx in indexes:
            cur.execute(f"PRAGMA index_info({idx['name']});")
            idx_cols = [c["name"] for c in cur.fetchall()]
            if idx_cols == ["owner", "filename"] and idx["unique"]:
                has_composite_uq = True

        if has_owner and has_composite_uq:
            # Check if any NULL owners remain
            cur.execute("SELECT count(*) as cnt FROM documents WHERE owner IS NULL;")
            null_cnt = cur.fetchone()["cnt"]
            if null_cnt == 0:
                print("[SQLite] Database is already fully migrated. 0 changes required.")
                stats["already_migrated"] = True
                return stats

        # Build DB graph owner candidates lookup: {doc_stem: {owner, ...}}
        db_doc_owners: Dict[str, Set[str]] = {}
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_graphs';")
        if cur.fetchone():
            cur.execute("SELECT doc_id, owner FROM document_graphs WHERE owner IS NOT NULL AND owner != '';")
            for r in cur.fetchall():
                d_stem = Path(r["doc_id"]).stem.lower()
                db_doc_owners.setdefault(d_stem, set()).add(r["owner"])

        print("[SQLite] Planning migration steps...")

        # 1. Rebuild documents table
        cur.execute("SELECT * FROM documents;")
        doc_rows = cur.fetchall()
        doc_backfills = []
        for r in doc_rows:
            row_dict = dict(r)
            cur_owner = row_dict.get("owner")
            if not cur_owner:
                stem = Path(row_dict["filename"]).stem.lower()
                resolved, status = resolve_owner_for_doc(row_dict["filename"], db_doc_owners.get(stem))
                row_dict["owner"] = resolved
                stats[status] += 1
                doc_backfills.append((row_dict["filename"], resolved, status))
            elif cur_owner.startswith("_"):
                stats["quarantined"] += 1
            else:
                stats["backfilled"] += 1

        # 2. Rebuild document_markdowns table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_markdowns';")
        md_rows = []
        if cur.fetchone():
            cur.execute("SELECT * FROM document_markdowns;")
            for r in cur.fetchall():
                row_dict = dict(r)
                cur_owner = row_dict.get("owner")
                if not cur_owner:
                    stem = Path(row_dict["doc_id"]).stem.lower()
                    resolved, status = resolve_owner_for_doc(row_dict["doc_id"], db_doc_owners.get(stem))
                    row_dict["owner"] = resolved
                    stats[status] += 1
                elif cur_owner.startswith("_"):
                    stats["quarantined"] += 1
                else:
                    stats["backfilled"] += 1
                md_rows.append(row_dict)

        # 3. Rebuild extraction_runs table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_runs';")
        run_rows = []
        if cur.fetchone():
            cur.execute("SELECT * FROM extraction_runs;")
            for r in cur.fetchall():
                row_dict = dict(r)
                cur_owner = row_dict.get("owner")
                if not cur_owner:
                    stem = Path(row_dict["doc_id"]).stem.lower()
                    resolved, status = resolve_owner_for_doc(row_dict["doc_id"], db_doc_owners.get(stem))
                    row_dict["owner"] = resolved
                    stats[status] += 1
                elif cur_owner.startswith("_"):
                    stats["quarantined"] += 1
                else:
                    stats["backfilled"] += 1
                run_rows.append(row_dict)

        # 4. Check page_checkpoints table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='page_checkpoints';")
        checkpoint_rows = []
        if cur.fetchone():
            cur.execute("SELECT * FROM page_checkpoints;")
            for r in cur.fetchall():
                row_dict = dict(r)
                cur_owner = row_dict.get("owner")
                if not cur_owner:
                    row_dict["owner"] = SENTINEL_ADMIN
                checkpoint_rows.append(row_dict)

        print("\n--- Migration Plan (SQLite) ---")
        print(f"Total Backfilled rows : {stats['backfilled']}")
        print(f"Total Ambiguous rows  : {stats['ambiguous']}")
        print(f"Total Quarantined rows: {stats['quarantined']} (assigned to '{SENTINEL_LEGACY}')")

        if dry_run:
            print("\n[DRY RUN] No changes applied to database.")
            return stats

        # Execute table rebuilds inside transaction
        cur.execute("PRAGMA foreign_keys = OFF;")
        cur.execute("BEGIN TRANSACTION;")

        # Rebuild documents
        cur.execute("""
            CREATE TABLE documents_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner VARCHAR NOT NULL,
                filename VARCHAR NOT NULL,
                file_size INTEGER NOT NULL,
                content_type VARCHAR NOT NULL,
                file_hash VARCHAR,
                file_data BLOB NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT uq_documents_owner_filename UNIQUE (owner, filename)
            );
        """)
        for d in doc_rows:
            d_dict = dict(d)
            owner_val = d_dict.get("owner") or resolve_owner_for_doc(d_dict["filename"], db_doc_owners.get(Path(d_dict["filename"]).stem.lower()))[0]
            cur.execute("""
                INSERT INTO documents_new (id, owner, filename, file_size, content_type, file_hash, file_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (d_dict["id"], owner_val, d_dict["filename"], d_dict["file_size"], d_dict["content_type"], d_dict.get("file_hash"), d_dict["file_data"], d_dict["created_at"]))
        cur.execute("DROP TABLE documents;")
        cur.execute("ALTER TABLE documents_new RENAME TO documents;")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash);")
        stats["tables_rebuilt"].append("documents")

        # Rebuild document_markdowns
        if md_rows:
            cur.execute("""
                CREATE TABLE document_markdowns_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner VARCHAR NOT NULL,
                    doc_id VARCHAR NOT NULL,
                    md_filename VARCHAR NOT NULL,
                    markdown_content TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    schema_id VARCHAR,
                    schema_ref_json JSON,
                    pages_json JSON,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_document_markdowns_owner_doc_id UNIQUE (owner, doc_id)
                );
            """)
            for m in md_rows:
                cur.execute("""
                    INSERT INTO document_markdowns_new (id, owner, doc_id, md_filename, markdown_content, page_count, schema_id, schema_ref_json, pages_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (m["id"], m["owner"], m["doc_id"], m["md_filename"], m["markdown_content"], m["page_count"], m.get("schema_id"), m.get("schema_ref_json"), m.get("pages_json"), m["created_at"]))
            cur.execute("DROP TABLE document_markdowns;")
            cur.execute("ALTER TABLE document_markdowns_new RENAME TO document_markdowns;")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_document_markdowns_schema_id ON document_markdowns (schema_id);")
            stats["tables_rebuilt"].append("document_markdowns")

        # Rebuild extraction_runs
        if run_rows:
            cur.execute("""
                CREATE TABLE extraction_runs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner VARCHAR NOT NULL,
                    doc_id VARCHAR NOT NULL,
                    page_count INTEGER,
                    schema_name VARCHAR NOT NULL,
                    result_json JSON NOT NULL,
                    page_details JSON,
                    llm_provider VARCHAR,
                    model_name VARCHAR,
                    processing_time_seconds FLOAT,
                    created_at DATETIME NOT NULL
                );
            """)
            for er in run_rows:
                cur.execute("""
                    INSERT INTO extraction_runs_new (id, owner, doc_id, page_count, schema_name, result_json, page_details, llm_provider, model_name, processing_time_seconds, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (er["id"], er["owner"], er["doc_id"], er.get("page_count"), er["schema_name"], er["result_json"], er.get("page_details"), er.get("llm_provider"), er.get("model_name"), er.get("processing_time_seconds"), er["created_at"]))
            cur.execute("DROP TABLE extraction_runs;")
            cur.execute("ALTER TABLE extraction_runs_new RENAME TO extraction_runs;")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_extraction_runs_owner_doc_id ON extraction_runs (owner, doc_id);")
            stats["tables_rebuilt"].append("extraction_runs")

        # Rebuild page_checkpoints if needed
        if checkpoint_rows:
            cur.execute("PRAGMA table_info(page_checkpoints);")
            cp_cols = [c["name"] for c in cur.fetchall()]
            if "owner" not in cp_cols:
                cur.execute("""
                    CREATE TABLE page_checkpoints_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        checkpoint_key VARCHAR NOT NULL UNIQUE,
                        owner VARCHAR NOT NULL,
                        doc_id VARCHAR NOT NULL,
                        page_number INTEGER NOT NULL,
                        pipeline_version VARCHAR NOT NULL,
                        extraction_version VARCHAR NOT NULL,
                        model_version VARCHAR,
                        prompt_version VARCHAR,
                        schema_version VARCHAR,
                        status VARCHAR NOT NULL,
                        page_delta_json JSON,
                        delta_checksum VARCHAR,
                        attempt_count INTEGER NOT NULL,
                        error_info TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );
                """)
                for cp in checkpoint_rows:
                    cur.execute("""
                        INSERT INTO page_checkpoints_new (id, checkpoint_key, owner, doc_id, page_number, pipeline_version, extraction_version, model_version, prompt_version, schema_version, status, page_delta_json, delta_checksum, attempt_count, error_info, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (cp["id"], cp["checkpoint_key"], cp.get("owner", SENTINEL_ADMIN), cp["doc_id"], cp["page_number"], cp["pipeline_version"], cp["extraction_version"], cp.get("model_version"), cp.get("prompt_version"), cp.get("schema_version"), cp["status"], cp.get("page_delta_json"), cp.get("delta_checksum"), cp["attempt_count"], cp.get("error_info"), cp["created_at"], cp["updated_at"]))
                cur.execute("DROP TABLE page_checkpoints;")
                cur.execute("ALTER TABLE page_checkpoints_new RENAME TO page_checkpoints;")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_page_checkpoints_doc_id ON page_checkpoints (doc_id);")
                stats["tables_rebuilt"].append("page_checkpoints")

        conn.commit()
        cur.execute("PRAGMA foreign_keys = ON;")
        print("[SQLite] Migration completed successfully.")
        return stats

    finally:
        conn.close()


# ==============================================================================
# PostgreSQL Migration Implementation
# ==============================================================================

def migrate_postgres(db_url: str, dry_run: bool = False) -> Dict[str, Any]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    print(f"\n[PostgreSQL] Connecting to PostgreSQL...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=RealDictCursor)

    stats = {
        "engine": "postgres",
        "dry_run": dry_run,
        "backfilled": 0,
        "ambiguous": 0,
        "quarantined": 0,
        "dropped_constraints": [],
        "dropped_indexes": [],
        "created_constraints": [],
        "already_migrated": False,
    }

    try:
        # Check if already migrated:
        # owner column exists in documents and constraint uq_documents_owner_filename exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'owner';
        """)
        has_owner = bool(cur.fetchone())

        cur.execute("""
            SELECT conname FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'documents' AND c.conname = 'uq_documents_owner_filename';
        """)
        has_uq = bool(cur.fetchone())

        if has_owner and has_uq:
            cur.execute("SELECT count(*) as cnt FROM documents WHERE owner IS NULL;")
            if cur.fetchone()["cnt"] == 0:
                print("[PostgreSQL] Database is already fully migrated. 0 changes required.")
                stats["already_migrated"] = True
                return stats

        # Build DB graph owner candidates lookup: {doc_stem: {owner, ...}}
        db_doc_owners: Dict[str, Set[str]] = {}
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'document_graphs'
            );
        """)
        if cur.fetchone()["exists"]:
            cur.execute("SELECT doc_id, owner FROM document_graphs WHERE owner IS NOT NULL AND owner != '';")
            for r in cur.fetchall():
                d_stem = Path(r["doc_id"]).stem.lower()
                db_doc_owners.setdefault(d_stem, set()).add(r["owner"])

        def _has_col(tbl: str, col: str) -> bool:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s AND column_name = %s;
            """, (tbl, col))
            return bool(cur.fetchone())

        # 1. Add owner column to tables if missing (as nullable for backfill)
        tables_to_check = ["documents", "document_markdowns", "extraction_runs", "page_checkpoints"]
        col_status = {}
        for tbl in tables_to_check:
            exists = _has_col(tbl, "owner")
            col_status[tbl] = exists
            if not exists:
                print(f"[PostgreSQL] Planning to add 'owner VARCHAR' to {tbl}...")
                if not dry_run:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN owner VARCHAR;")

        # 2. Backfill rows in documents
        q_docs = "SELECT id, filename, owner FROM documents;" if (col_status["documents"] or not dry_run) else "SELECT id, filename, NULL as owner FROM documents;"
        cur.execute(q_docs)
        for r in cur.fetchall():
            cur_owner = r["owner"]
            if not cur_owner:
                stem = Path(r["filename"]).stem.lower()
                resolved, status = resolve_owner_for_doc(r["filename"], db_doc_owners.get(stem))
                stats[status] += 1
                if not dry_run:
                    cur.execute("UPDATE documents SET owner = %s WHERE id = %s;", (resolved, r["id"]))
            elif cur_owner.startswith("_"):
                stats["quarantined"] += 1
            else:
                stats["backfilled"] += 1

        # 3. Backfill rows in document_markdowns
        q_mds = "SELECT id, doc_id, owner FROM document_markdowns;" if (col_status["document_markdowns"] or not dry_run) else "SELECT id, doc_id, NULL as owner FROM document_markdowns;"
        cur.execute(q_mds)
        for r in cur.fetchall():
            cur_owner = r["owner"]
            if not cur_owner:
                stem = Path(r["doc_id"]).stem.lower()
                resolved, status = resolve_owner_for_doc(r["doc_id"], db_doc_owners.get(stem))
                stats[status] += 1
                if not dry_run:
                    cur.execute("UPDATE document_markdowns SET owner = %s WHERE id = %s;", (resolved, r["id"]))
            elif cur_owner.startswith("_"):
                stats["quarantined"] += 1
            else:
                stats["backfilled"] += 1

        # 4. Backfill rows in extraction_runs
        q_er = "SELECT id, doc_id, owner FROM extraction_runs;" if (col_status["extraction_runs"] or not dry_run) else "SELECT id, doc_id, NULL as owner FROM extraction_runs;"
        cur.execute(q_er)
        for r in cur.fetchall():
            cur_owner = r["owner"]
            if not cur_owner:
                stem = Path(r["doc_id"]).stem.lower()
                resolved, status = resolve_owner_for_doc(r["doc_id"], db_doc_owners.get(stem))
                stats[status] += 1
                if not dry_run:
                    cur.execute("UPDATE extraction_runs SET owner = %s WHERE id = %s;", (resolved, r["id"]))
            elif cur_owner.startswith("_"):
                stats["quarantined"] += 1
            else:
                stats["backfilled"] += 1

        # 5. Backfill page_checkpoints
        q_cp = "SELECT id, owner FROM page_checkpoints;" if (col_status["page_checkpoints"] or not dry_run) else "SELECT id, NULL as owner FROM page_checkpoints;"
        cur.execute(q_cp)
        for r in cur.fetchall():
            if not r["owner"]:
                if not dry_run:
                    cur.execute("UPDATE page_checkpoints SET owner = %s WHERE id = %s;", (SENTINEL_ADMIN, r["id"]))

        # 6. Drop old unique constraints and indexes on documents.filename
        cur.execute("""
            SELECT c.conname FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'documents' AND c.contype = 'u';
        """)
        for c in cur.fetchall():
            conname = c["conname"]
            stats["dropped_constraints"].append(conname)
            if not dry_run:
                cur.execute(f"ALTER TABLE documents DROP CONSTRAINT IF EXISTS {conname};")

        # Also find any unique indexes on filename that aren't formal table constraints
        cur.execute("""
            SELECT i.relname FROM pg_index ix
            JOIN pg_class t ON ix.indrelid = t.oid
            JOIN pg_class i ON ix.indexrelid = i.oid
            WHERE t.relname = 'documents' AND ix.indisunique AND i.relname != 'documents_pkey';
        """)
        for idx in cur.fetchall():
            idx_name = idx["relname"]
            stats["dropped_indexes"].append(idx_name)
            if not dry_run:
                cur.execute(f"DROP INDEX IF EXISTS {idx_name};")

        # 7. Apply NOT NULL and new composite constraints
        for tbl in tables_to_check:
            if not dry_run:
                cur.execute(f"ALTER TABLE {tbl} ALTER COLUMN owner SET NOT NULL;")

        if not dry_run:
            cur.execute("""
                ALTER TABLE documents 
                ADD CONSTRAINT uq_documents_owner_filename UNIQUE (owner, filename);
            """)
            stats["created_constraints"].append("uq_documents_owner_filename")

            cur.execute("""
                ALTER TABLE document_markdowns 
                ADD CONSTRAINT uq_document_markdowns_owner_doc_id UNIQUE (owner, doc_id);
            """)
            stats["created_constraints"].append("uq_document_markdowns_owner_doc_id")

            cur.execute("""
                CREATE INDEX IF NOT EXISTS ix_extraction_runs_owner_doc_id 
                ON extraction_runs (owner, doc_id);
            """)
            stats["created_constraints"].append("ix_extraction_runs_owner_doc_id")

        print("\n--- Migration Plan (PostgreSQL) ---")
        print(f"Total Backfilled rows : {stats['backfilled']}")
        print(f"Total Ambiguous rows  : {stats['ambiguous']}")
        print(f"Total Quarantined rows: {stats['quarantined']} (assigned to '{SENTINEL_LEGACY}')")
        print(f"Dropped constraints   : {stats['dropped_constraints']}")
        print(f"Dropped unique indexes: {stats['dropped_indexes']}")
        print(f"Created constraints   : {stats['created_constraints'] if not dry_run else ['uq_documents_owner_filename', 'uq_document_markdowns_owner_doc_id', 'ix_extraction_runs_owner_doc_id']}")

        if dry_run:
            conn.rollback()
            print("\n[DRY RUN] Rolled back all changes. No tables or data were altered.")
            return stats

        conn.commit()
        print("[PostgreSQL] Migration committed successfully.")
        return stats

    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        cur.close()
        conn.close()


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tenant Isolation Database Migration (Finding #4)")
    parser.add_argument("--db-url", default=None, help="Database URL (e.g. postgresql://... or sqlite://...)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying database")
    parser.add_argument("--backup-acknowledged", action="store_true", help="Acknowledge database backup has been taken")
    args = parser.parse_args()

    if not args.dry_run and not args.backup_acknowledged:
        print("ERROR: Migration modifies database tables. You must provide --backup-acknowledged to confirm that a database backup has been taken.", file=sys.stderr)
        sys.exit(1)

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        try:
            from app.core.config import settings
            db_url = getattr(settings, "database_url", None)
        except Exception:
            pass
    if not db_url:
        db_url = "sqlite:///./idp_storage.db"

    print(f"Target Database URL: {re.sub(r':([^:@]+)@', ':****@', db_url)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")

    if db_url.startswith("sqlite"):
        db_path = db_url.replace("sqlite:///", "").replace("sqlite://", "")
        p = Path(db_path)
        if not p.exists():
            print(f"SQLite file {p} does not exist. Creating empty directory if needed.")
            p.parent.mkdir(parents=True, exist_ok=True)
            # Initialize with init_db if brand new
            from src.ai.layer3_extraction.storage import init_db
            init_db()
        stats = migrate_sqlite(p, dry_run=args.dry_run)
    else:
        pg_conn_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
        stats = migrate_postgres(pg_conn_url, dry_run=args.dry_run)

    print("\nMigration Summary:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
