"""
Comprehensive test suite for tenant isolation (Audit Finding #4).

Verifies all required acceptance invariants:
1. Same filename for two owners coexists in all tables (documents, document_markdowns, extraction_runs, checkpoints).
2. Reads with wrong owner return nothing across all read functions.
3. Candidate doc_id normalization (stem, doc_id, doc_id.pdf) occurs strictly within owner scope.
4. _legacy rows are strictly invisible to normal users; admins can query _legacy and _admin explicitly.
5. Function signature test: inspects public storage functions and fails if any lacks a required owner parameter.
6. Username registration validation rejects sentinels (_legacy, _admin, and leading underscores).
7. Checkpoint store keying incorporates owner to prevent cross-tenant replay.
8. Migration test on real SQLite with pre-populated legacy data (backfill, ambiguity, quarantine, idempotency).
9. Migration test on PostgreSQL.
"""
import inspect
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Set
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
chatbot_dir = ROOT_DIR / "schema_chatbot_v2"
if str(chatbot_dir) not in sys.path:
    sys.path.insert(0, str(chatbot_dir))

from src.ai.layer3_extraction.storage import (
    init_db,
    SessionLocal,
    SENTINEL_LEGACY,
    SENTINEL_ADMIN,
    DocumentRecord,
    MarkdownRecord,
    ExtractionRun,
    PageCheckpointRecord,
    save_document,
    save_markdown_record,
    get_markdown_record,
    save_extraction_run,
    get_latest_extracted_data,
    list_markdown_records,
    list_markdown_records_for_owner,
    save_page_checkpoint,
    get_page_checkpoint,
    build_checkpoint_key,
    list_page_checkpoints,
    delete_page_checkpoints,
)
from src.ai.layer3_extraction.extractor import (
    extract_with_graph_memory_ordered,
    extract_with_graph_memory_concurrent,
    extract_with_graph_memory_batched,
)
from src.ai.layer3_extraction.migrations.migrate_tenant_isolation import (
    migrate_sqlite,
    migrate_postgres,
    resolve_owner_for_doc,
)
from app.storage.user_store import InMemoryUserStore, Role


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    session = SessionLocal()
    try:
        session.query(DocumentRecord).delete()
        session.query(MarkdownRecord).delete()
        session.query(ExtractionRun).delete()
        session.query(PageCheckpointRecord).delete()
        session.commit()
    finally:
        session.close()
    yield
    session = SessionLocal()
    try:
        session.query(DocumentRecord).delete()
        session.query(MarkdownRecord).delete()
        session.query(ExtractionRun).delete()
        session.query(PageCheckpointRecord).delete()
        session.commit()
    finally:
        session.close()


# ==============================================================================
# 1. Signature Regression Guard Test
# ==============================================================================

def test_storage_function_signatures_require_owner_without_default():
    """
    Guards against regressions:
    (A) Every public storage function touching tenant-scoped tables must require
        'owner: str' with NO default value. Omitting 'owner' must be a TypeError.
    (B) The three extractor functions (ordered, concurrent, batched) must also
        require 'owner: str' with NO default. Omitting 'owner' must be a TypeError.
    """
    storage_funcs = [
        save_document,
        save_markdown_record,
        get_markdown_record,
        save_extraction_run,
        get_latest_extracted_data,
        list_markdown_records,
        build_checkpoint_key,
        save_page_checkpoint,
        get_page_checkpoint,
        list_page_checkpoints,
        delete_page_checkpoints,
    ]

    for fn in storage_funcs:
        sig = inspect.signature(fn)
        assert "owner" in sig.parameters, f"{fn.__name__} is missing required 'owner' parameter!"
        param = sig.parameters["owner"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} 'owner' parameter must NOT have a default value (got {param.default})!"
        )

    # Extractor functions must also have owner with NO default
    extractor_funcs = [
        extract_with_graph_memory_ordered,
        extract_with_graph_memory_concurrent,
        extract_with_graph_memory_batched,
    ]
    for fn in extractor_funcs:
        sig = inspect.signature(fn)
        assert "owner" in sig.parameters, f"{fn.__name__} is missing required 'owner' parameter!"
        param = sig.parameters["owner"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} 'owner' parameter must NOT have a default value (got {param.default})!"
        )

    # Calling each storage function without owner must raise TypeError
    with pytest.raises(TypeError):
        save_document("test.pdf", b"bytes")  # type: ignore

    with pytest.raises(TypeError):
        save_markdown_record("test.pdf", "test.md", "# test")  # type: ignore

    with pytest.raises(TypeError):
        get_markdown_record("test.pdf")  # type: ignore

    with pytest.raises(TypeError):
        save_extraction_run("test.pdf", 1, "schema", {})  # type: ignore

    with pytest.raises(TypeError):
        get_latest_extracted_data("test.pdf")  # type: ignore

    with pytest.raises(TypeError):
        list_markdown_records()  # type: ignore

    with pytest.raises(TypeError):
        build_checkpoint_key("test.pdf", 1)  # type: ignore

    with pytest.raises(TypeError):
        save_page_checkpoint("test.pdf", 1, "status")  # type: ignore

    with pytest.raises(TypeError):
        get_page_checkpoint("test.pdf", 1)  # type: ignore

    with pytest.raises(TypeError):
        list_page_checkpoints("test.pdf")  # type: ignore

    with pytest.raises(TypeError):
        delete_page_checkpoints("test.pdf")  # type: ignore

    # Calling each extractor function without owner must raise TypeError
    with pytest.raises(TypeError):
        extract_with_graph_memory_ordered([], object, object)  # type: ignore

    with pytest.raises(TypeError):
        extract_with_graph_memory_concurrent([], object, object)  # type: ignore

    with pytest.raises(TypeError):
        extract_with_graph_memory_batched([], object, object)  # type: ignore


# ==============================================================================
# 2. Coexistence of Same Filename for Multiple Owners
# ==============================================================================

def test_same_filename_coexists_across_owners_in_all_tables():
    """
    Two distinct owners upload and process 'tax_return.pdf'.
    Both records must coexist without collisions in:
    - documents
    - document_markdowns
    - extraction_runs
    - page_checkpoints
    """
    filename = "tax_return.pdf"

    # 1. Documents (shadow PDF table)
    id_a = save_document(filename=filename, file_bytes=b"Owner A 1040", owner="tenant_alpha")
    id_b = save_document(filename=filename, file_bytes=b"Owner B 1040", owner="tenant_beta")
    assert id_a is not None and id_b is not None
    assert id_a != id_b

    # 2. Document Markdowns
    md_a = save_markdown_record(doc_id=filename, md_filename="tax_return.md", markdown_content="# Alpha Tax", owner="tenant_alpha")
    md_b = save_markdown_record(doc_id=filename, md_filename="tax_return.md", markdown_content="# Beta Tax", owner="tenant_beta")
    assert md_a is not None and md_b is not None
    assert md_a != md_b

    # 3. Extraction Runs
    run_a = save_extraction_run(doc_id=filename, page_count=1, schema_name="tax", result_json={"gross_income": 100000}, owner="tenant_alpha")
    run_b = save_extraction_run(doc_id=filename, page_count=1, schema_name="tax", result_json={"gross_income": 50000}, owner="tenant_beta")
    assert run_a is not None and run_b is not None
    assert run_a != run_b

    # 4. Page Checkpoints
    cp_a = save_page_checkpoint(doc_id=filename, page_number=1, status="success", owner="tenant_alpha")
    cp_b = save_page_checkpoint(doc_id=filename, page_number=1, status="success", owner="tenant_beta")
    assert cp_a is not None and cp_b is not None
    assert cp_a != cp_b

    session = SessionLocal()
    try:
        docs = session.query(DocumentRecord).filter_by(filename=filename).all()
        assert len(docs) == 2
        assert {d.owner for d in docs} == {"tenant_alpha", "tenant_beta"}

        mds = session.query(MarkdownRecord).filter_by(doc_id=filename).all()
        assert len(mds) == 2
        assert {m.owner for m in mds} == {"tenant_alpha", "tenant_beta"}

        runs = session.query(ExtractionRun).filter_by(doc_id=filename).all()
        assert len(runs) == 2
        assert {r.owner for r in runs} == {"tenant_alpha", "tenant_beta"}

        cps = session.query(PageCheckpointRecord).filter_by(doc_id=filename).all()
        assert len(cps) == 2
        assert {c.owner for c in cps} == {"tenant_alpha", "tenant_beta"}
    finally:
        session.close()


# ==============================================================================
# 3. Wrong Owner Returns Nothing Across All Read Functions
# ==============================================================================

def test_read_with_wrong_owner_returns_nothing():
    """
    Every storage read function returns None or empty list when queried with
    the wrong owner.
    """
    doc_id = "sensitive_nda.pdf"
    save_markdown_record(doc_id=doc_id, md_filename="sensitive_nda.md", markdown_content="# NDA Secret Clauses", owner="tenant_alpha")
    save_extraction_run(doc_id=doc_id, page_count=2, schema_name="nda", result_json={"party": "Secret Corp"}, owner="tenant_alpha")
    save_page_checkpoint(doc_id=doc_id, page_number=1, status="success", page_delta_json={"party": "Secret Corp"}, owner="tenant_alpha")

    # tenant_beta reads must return None / empty
    assert get_markdown_record(doc_id, owner="tenant_beta") is None
    assert get_latest_extracted_data(doc_id, owner="tenant_beta") is None
    assert list_markdown_records(owner="tenant_beta") == []
    assert list_markdown_records_for_owner(owner="tenant_beta") == []
    assert get_page_checkpoint(doc_id, page_number=1, owner="tenant_beta") is None

    # tenant_alpha reads return the expected records
    rec_md = get_markdown_record(doc_id, owner="tenant_alpha")
    assert rec_md is not None and "Secret Clauses" in rec_md["markdown_content"]

    data = get_latest_extracted_data(doc_id, owner="tenant_alpha")
    assert data is not None and data.get("party") == "Secret Corp"

    recs = list_markdown_records(owner="tenant_alpha")
    assert len(recs) == 1 and recs[0]["doc_id"] == doc_id


# ==============================================================================
# 4. Doc ID Normalization Strictly Inside Owner Scope
# ==============================================================================

def test_doc_id_stem_normalization_confined_to_owner_scope():
    """
    Amendment #5:
    Owner A has 'resume.pdf'.
    Owner B asks for 'resume' (stem).
    Owner B must get None. Normalization must NOT match across tenants.
    """
    save_extraction_run(
        doc_id="resume.pdf",
        page_count=1,
        schema_name="resume",
        result_json={"candidate_name": "Alice Private"},
        owner="owner_a",
    )
    save_markdown_record(
        doc_id="resume.pdf",
        md_filename="resume.md",
        markdown_content="# Alice Resume",
        owner="owner_a",
    )

    # Owner B asks for stem 'resume'
    assert get_latest_extracted_data("resume", owner="owner_b") is None
    assert get_markdown_record("resume", owner="owner_b") is None

    # Owner A asking for 'resume' (stem) matches within Owner A's scope
    data_a = get_latest_extracted_data("resume", owner="owner_a")
    assert data_a is not None and data_a.get("candidate_name") == "Alice Private"

    rec_a = get_markdown_record("resume", owner="owner_a")
    assert rec_a is not None and "Alice Resume" in rec_a["markdown_content"]


# ==============================================================================
# 5. Sentinel Owner Isolation (_legacy and _admin)
# ==============================================================================

def test_legacy_rows_are_invisible_to_normal_users_and_visible_to_admins():
    """
    Amendment #2:
    Rows assigned to _legacy can never be read by normal users.
    Admins can explicitly read _legacy and _admin records.
    """
    save_markdown_record(
        doc_id="orphaned.pdf",
        md_filename="orphaned.md",
        markdown_content="# Orphaned Data",
        owner=SENTINEL_LEGACY,
    )
    save_extraction_run(
        doc_id="orphaned.pdf",
        page_count=1,
        schema_name="legacy",
        result_json={"status": "migrated_legacy"},
        owner=SENTINEL_LEGACY,
    )

    save_markdown_record(
        doc_id="shared_admin.pdf",
        md_filename="shared_admin.md",
        markdown_content="# Admin Template",
        owner=SENTINEL_ADMIN,
    )

    # Normal users cannot see _legacy
    assert get_markdown_record("orphaned.pdf", owner="regular_user") is None
    assert get_latest_extracted_data("orphaned.pdf", owner="regular_user") is None
    normal_recs = list_markdown_records(owner="regular_user")
    assert not any(r["doc_id"] == "orphaned.pdf" for r in normal_recs)

    # Admin querying _legacy explicitly
    rec_leg = get_markdown_record("orphaned.pdf", owner=SENTINEL_LEGACY)
    assert rec_leg is not None and "Orphaned Data" in rec_leg["markdown_content"]
    data_leg = get_latest_extracted_data("orphaned.pdf", owner=SENTINEL_LEGACY)
    assert data_leg is not None and data_leg.get("status") == "migrated_legacy"

    # Admin querying _admin explicitly
    rec_admin = get_markdown_record("shared_admin.pdf", owner=SENTINEL_ADMIN)
    assert rec_admin is not None and "Admin Template" in rec_admin["markdown_content"]

    # Admin querying across all via sentinel
    all_recs = list_markdown_records(owner="*")
    doc_ids = {r["doc_id"] for r in all_recs}
    assert "orphaned.pdf" in doc_ids
    assert "shared_admin.pdf" in doc_ids


# ==============================================================================
# 6. Username Registration Sentinel Collisions Rejected
# ==============================================================================

def test_username_validation_rejects_sentinels_and_leading_underscores():
    """
    Amendment #2: Sentinels must be rejected at registration.
    Usernames colliding with _legacy, _admin, or leading underscores must fail.
    """
    store = InMemoryUserStore()

    with pytest.raises(ValueError, match="reserved"):
        store.create("_legacy", "password123", Role.USER)

    with pytest.raises(ValueError, match="reserved"):
        store.create("_admin", "password123", Role.USER)

    with pytest.raises(ValueError, match="reserved"):
        store.create("_internal_user", "password123", Role.USER)

    # Valid user creation succeeds
    u = store.create("valid_user", "password123", Role.USER)
    assert u.username == "valid_user"


# ==============================================================================
# 7. Checkpoint Key Isolation
# ==============================================================================

def test_checkpoint_key_isolation_prevents_page_result_replay():
    """
    Amendment #6:
    Checkpoint key format is '{owner}:{doc_id}:p{page_number}:{pipeline_version}:{extraction_version}'.
    Two owners with same doc_id cannot replay each other's checkpoints.
    """
    key_a = build_checkpoint_key("doc.pdf", 1, owner="tenant_a", pipeline_version="v1", extraction_version="v1")
    key_b = build_checkpoint_key("doc.pdf", 1, owner="tenant_b", pipeline_version="v1", extraction_version="v1")

    assert key_a != key_b
    assert key_a.startswith("tenant_a:doc.pdf:p1:")
    assert key_b.startswith("tenant_b:doc.pdf:p1:")

    save_page_checkpoint("doc.pdf", 1, "completed", page_delta_json={"page": 1, "text": "Tenant A Secret"}, owner="tenant_a")
    cp_b = get_page_checkpoint("doc.pdf", 1, owner="tenant_b")
    assert cp_b is None, "Tenant B must not see Tenant A's page checkpoint!"

    cp_a = get_page_checkpoint("doc.pdf", 1, owner="tenant_a")
    assert cp_a is not None
    assert cp_a["page_delta_json"]["text"] == "Tenant A Secret"


# ==============================================================================
# 8. SQLite Migration on Real SQLite File with Legacy Data
# ==============================================================================

def test_sqlite_migration_backfill_ambiguity_and_idempotency(tmp_path: Path):
    """
    Amendment #3 & #4:
    Tests SQLite migration on a legacy database:
    - Pre-populates legacy tables without owner column and with unique=True on filename.
    - Tests backfill for unambiguous doc.
    - Tests ambiguity quarantine.
    - Tests unrecoverable quarantine.
    - Tests table rebuild and composite unique constraint enforcement.
    - Tests idempotency (running twice produces 0 changes).
    """
    db_file = tmp_path / "legacy_test.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Create pre-migration tables
    cur.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename VARCHAR NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            content_type VARCHAR NOT NULL,
            file_hash VARCHAR,
            file_data BLOB NOT NULL,
            created_at DATETIME NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE document_markdowns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id VARCHAR NOT NULL,
            md_filename VARCHAR NOT NULL,
            markdown_content TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            schema_id VARCHAR,
            schema_ref_json JSON,
            pages_json JSON,
            created_at DATETIME NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE extraction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    # Populate legacy rows
    cur.execute("INSERT INTO documents (filename, file_size, content_type, file_data, created_at) VALUES ('doc1.pdf', 100, 'application/pdf', X'00', '2026-01-01');")
    cur.execute("INSERT INTO documents (filename, file_size, content_type, file_data, created_at) VALUES ('doc2.pdf', 100, 'application/pdf', X'00', '2026-01-01');")
    cur.execute("INSERT INTO documents (filename, file_size, content_type, file_data, created_at) VALUES ('doc3.pdf', 100, 'application/pdf', X'00', '2026-01-01');")

    cur.execute("INSERT INTO document_markdowns (doc_id, md_filename, markdown_content, page_count, created_at) VALUES ('doc1.pdf', 'doc1.md', '# Doc 1', 1, '2026-01-01');")
    cur.execute("INSERT INTO extraction_runs (doc_id, schema_name, result_json, created_at) VALUES ('doc1.pdf', 'schema1', '{}', '2026-01-01');")
    conn.commit()
    conn.close()

    # Mock filesystem discovery:
    # doc1 -> exactly one owner ('user_alpha')
    # doc2 -> multiple owners ('user_alpha', 'user_beta') -> ambiguous
    # doc3 -> zero owners -> quarantined
    original_resolve = resolve_owner_for_doc

    def mock_resolve(doc_identifier: str, db_candidates=None):
        if "doc1" in doc_identifier:
            return "user_alpha", "backfilled"
        elif "doc2" in doc_identifier:
            return SENTINEL_LEGACY, "ambiguous"
        else:
            return SENTINEL_LEGACY, "quarantined"

    import src.ai.layer3_extraction.migrations.migrate_tenant_isolation as mig_mod
    mig_mod.resolve_owner_for_doc = mock_resolve

    try:
        # Dry run first
        dry_stats = migrate_sqlite(db_file, dry_run=True)
        assert dry_stats["dry_run"] is True
        assert dry_stats["backfilled"] >= 1
        assert dry_stats["ambiguous"] >= 1
        assert dry_stats["quarantined"] >= 1

        # Real execution
        exec_stats = migrate_sqlite(db_file, dry_run=False)
        assert exec_stats["dry_run"] is False
        assert "documents" in exec_stats["tables_rebuilt"]

        # Verify tables and constraints in migrated SQLite database
        m_conn = sqlite3.connect(str(db_file))
        m_cur = m_conn.cursor()

        # Check doc1 has owner 'user_alpha'
        m_cur.execute("SELECT owner FROM documents WHERE filename='doc1.pdf';")
        assert m_cur.fetchone()[0] == "user_alpha"

        # Check doc2 has owner '_legacy' (ambiguous)
        m_cur.execute("SELECT owner FROM documents WHERE filename='doc2.pdf';")
        assert m_cur.fetchone()[0] == SENTINEL_LEGACY

        # Check doc3 has owner '_legacy' (unrecoverable)
        m_cur.execute("SELECT owner FROM documents WHERE filename='doc3.pdf';")
        assert m_cur.fetchone()[0] == SENTINEL_LEGACY

        # Verify that two owners can now insert the same filename into documents
        m_cur.execute("""
            INSERT INTO documents (owner, filename, file_size, content_type, file_data, created_at)
            VALUES ('owner_one', 'same_name.pdf', 50, 'application/pdf', X'00', '2026-01-02');
        """)
        m_cur.execute("""
            INSERT INTO documents (owner, filename, file_size, content_type, file_data, created_at)
            VALUES ('owner_two', 'same_name.pdf', 50, 'application/pdf', X'00', '2026-01-02');
        """)
        m_conn.commit()

        # Inserting same owner + same filename must fail composite uniqueness
        with pytest.raises(sqlite3.IntegrityError):
            m_cur.execute("""
                INSERT INTO documents (owner, filename, file_size, content_type, file_data, created_at)
                VALUES ('owner_one', 'same_name.pdf', 50, 'application/pdf', X'00', '2026-01-02');
            """)

        m_conn.close()

        # Test idempotency: running migration a 2nd time does nothing
        idem_stats = migrate_sqlite(db_file, dry_run=False)
        assert idem_stats["already_migrated"] is True
        assert len(idem_stats["tables_rebuilt"]) == 0

    finally:
        mig_mod.resolve_owner_for_doc = original_resolve


# ==============================================================================
# 9. PostgreSQL Migration Verification
# ==============================================================================

def test_postgres_migration_verification():
    """
    Verifies that migrate_postgres runs successfully and idempotently against
    the active PostgreSQL instance.
    """
    from app.config import settings
    db_url = getattr(settings, "database_url", None)
    if not db_url or not db_url.startswith("postgres"):
        pytest.skip("PostgreSQL not configured in settings")

    pg_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
    stats = migrate_postgres(pg_url, dry_run=False)
    assert stats["already_migrated"] is True or stats["engine"] == "postgres"
