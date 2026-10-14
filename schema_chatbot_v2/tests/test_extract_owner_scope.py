"""
Phase 4 commit 3 — Step 1 verification + Step 2/3 regression tests for
owner-scoped pipeline outputs and production IR persistence.

OFFLINE: process_document, the extraction client and OCR are all mocked, so no
Bedrock and no AWS calls happen. Synthetic 1-page digital PDF bytes only; no
document content is asserted beyond synthetic markers.

Covers:
  - /pipeline/extract writes each owner's .md + .ir.json to a per-owner dir
    (two owners, same filename, no collision; wrong owner cannot read).
  - ir_owner is passed from the route and the IR lands at the owner-scoped path.
  - the doc_id mapping (filename vs stem normalization).
  - a persistence failure does not fail the request and sets ir_persisted=False.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import OUTPUT_DIR, resolve_owner_output_dir
from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.layer3_extraction.ir_store import ir_path, load_page_irs
from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR


DOC_STEM = "owner_scope_invoice"
PDF_NAME = f"{DOC_STEM}.pdf"


@pytest.fixture
def client():
    return TestClient(app)


def _auth(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _mock_page(markdown_marker: str):
    page = MagicMock()
    page.page_number = 1
    page.markdown = markdown_marker
    page.engines_used = ["docling"]
    page.capabilities = ["digital_pdf"]
    page.confidence = 0.99
    page.escalated = False
    page.low_confidence = False
    # a real block so IR persistence has something to serialize
    page.blocks = [Block(block_id="p1-b1", type=BlockType.PARAGRAPH, text=markdown_marker,
                         bbox=BBox(x0=0, y0=0, x1=10, y1=10), reading_order=0, source_engine="docling")]
    return page


def _run_extract(client, auth, marker, total):
    """Drive POST /pipeline/extract fully offline. Returns the response."""
    result_obj = MagicMock()
    result_obj.model_dump.return_value = {"total": total}

    pdf_bytes = b"%PDF-1.4 synthetic one page"
    files = {"file": (PDF_NAME, pdf_bytes, "application/pdf")}
    data = {"raw_schema": json.dumps([{"name": "total", "description": "Total amount"}])}

    with patch("app.api.pipeline_routes.process_document",
               return_value=[(_mock_page(marker), {})]) as _pd, \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[MagicMock()]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=result_obj), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"), \
         patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=1), \
         patch("src.ai.layer3_extraction.storage.save_document"), \
         patch("src.ai.layer3_extraction.s3_documents.is_configured", return_value=False):
        return client.post("/pipeline/extract", headers=auth, files=files, data=data), _pd


@pytest.fixture
def cleanup_dirs():
    yield
    for u in ("extract_owner_a", "extract_owner_b", "extract_fail_owner"):
        shutil.rmtree(OUTPUT_DIR / "_users" / u, ignore_errors=True)
        shutil.rmtree(resolve_owner_output_dir(u), ignore_errors=True)
    # flat leftovers from any regression
    for suffix in (".md", ".extracted.json", ".ir.json"):
        (OUTPUT_DIR / f"{DOC_STEM}{suffix}").unlink(missing_ok=True)


def test_extract_two_owners_same_filename_no_collision(client, cleanup_dirs):
    auth_a = _auth(client, "extract_owner_a", "pw_a")
    auth_b = _auth(client, "extract_owner_b", "pw_b")

    resp_a, pd_a = _run_extract(client, auth_a, "Total A: 100", 100)
    assert resp_a.status_code == 200, resp_a.text
    resp_b, pd_b = _run_extract(client, auth_b, "Total B: 200", 200)
    assert resp_b.status_code == 200, resp_b.text

    # The route must direct process_document to each owner's OWN output_dir and
    # pass that owner as ir_owner — so the same filename for two owners never
    # collides on disk (the .md itself is written by the real md_writer into the
    # output_dir supplied here, which is mocked out in this offline test).
    a_dir = str(resolve_owner_output_dir("extract_owner_a"))
    b_dir = str(resolve_owner_output_dir("extract_owner_b"))
    kw_a = pd_a.call_args.kwargs
    kw_b = pd_b.call_args.kwargs
    assert kw_a["output_dir"] == a_dir, f"owner A output_dir not scoped: {kw_a['output_dir']}"
    assert kw_b["output_dir"] == b_dir, f"owner B output_dir not scoped: {kw_b['output_dir']}"
    assert kw_a["output_dir"] != kw_b["output_dir"]
    assert kw_a["ir_owner"] == "extract_owner_a"
    assert kw_b["ir_owner"] == "extract_owner_b"

    # IR itself (written by the route, not md_writer) is per-owner and isolated.
    assert load_page_irs(owner="extract_owner_a", doc_id=PDF_NAME) is not None
    assert load_page_irs(owner="extract_owner_b", doc_id=PDF_NAME) is not None


def test_extract_persists_ir_per_owner(client, cleanup_dirs):
    auth_a = _auth(client, "extract_owner_a", "pw_a")
    resp, _ = _run_extract(client, auth_a, "IR owner A", 100)
    assert resp.status_code == 200, resp.text
    assert resp.json()["meta"]["ir_persisted"] is True

    # IR present at the owner-scoped path, under the authenticated owner.
    irs = load_page_irs(owner="extract_owner_a", doc_id=PDF_NAME)
    assert irs is not None, "IR not persisted at owner-scoped path"
    assert sum(len(p.blocks) for p in irs) >= 1


def test_extract_doc_id_matches_query_doc_id(client, cleanup_dirs):
    """The IR doc_id must equal the doc_id the Query Bot/graph store use for the
    same document — the FULL filename (e.g. 'owner_scope_invoice.pdf'), and must
    be findable whether queried by filename or by stem (normalized identically
    to ir_store._safe_doc_id)."""
    auth_a = _auth(client, "extract_owner_a", "pw_a")
    resp, _ = _run_extract(client, auth_a, "doc id mapping", 100)
    assert resp.status_code == 200, resp.text
    assert resp.json()["doc_id"] == PDF_NAME  # graph/query-bot doc_id == filename

    # found by the filename doc_id
    assert load_page_irs(owner="extract_owner_a", doc_id=PDF_NAME) is not None
    # the path for 'resume.pdf' and the sanitized stem differ (filename keeps '.pdf'):
    p_full = ir_path(owner="extract_owner_a", doc_id=PDF_NAME)
    p_stem = ir_path(owner="extract_owner_a", doc_id=DOC_STEM)
    assert p_full != p_stem  # documents the normalization: filename != stem


def test_extract_ir_failure_does_not_fail_job(client, cleanup_dirs):
    auth = _auth(client, "extract_fail_owner", "pw_f")
    result_obj = MagicMock()
    result_obj.model_dump.return_value = {"total": 1}
    files = {"file": (PDF_NAME, b"%PDF-1.4 x", "application/pdf")}
    data = {"raw_schema": json.dumps([{"name": "total", "description": "t"}])}

    with patch("app.api.pipeline_routes.process_document",
               return_value=[(_mock_page("fail path"), {})]), \
         patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[MagicMock()]), \
         patch("src.adapters.llm.factory.get_llm_client"), \
         patch("src.adapters.llm.extraction_factory.get_extraction_client"), \
         patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=result_obj), \
         patch("src.ai.layer3_extraction.storage.save_markdown_record"), \
         patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=1), \
         patch("src.ai.layer3_extraction.storage.save_document"), \
         patch("src.ai.layer3_extraction.s3_documents.is_configured", return_value=False), \
         patch("app.api.pipeline_routes.save_page_irs", side_effect=OSError("disk full")):
        resp = client.post("/pipeline/extract", headers=auth, files=files, data=data)

    assert resp.status_code == 200, resp.text
    assert resp.json()["meta"]["ir_persisted"] is False


# --------------------------------------------------------------------------- #
# /pipeline/run (background job) wiring + delete cleanup
# --------------------------------------------------------------------------- #

def test_run_job_passes_ir_owner_and_persists(client, cleanup_dirs):
    """_run_pipeline_job must pass ir_owner=job_owner to process_document and
    record ir_persisted on the per-document success entry."""
    from app.api.pipeline_routes import (
        _job_controls, _pipeline_jobs, JobControl, _run_pipeline_job, SCHEMA_REGISTRY,
    )

    owner = "extract_owner_a"
    stem = "job_scope_doc"
    pdf = resolve_owner_output_dir(owner).parent  # ensure OUTPUT_DIR exists
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    pdf.write_bytes(b"%PDF-1.4 job")

    schema_id = "job_scope_schema"
    schema_record = {"schema_id": schema_id, "document_type": "doc",
                     "schema": {"document_type": "doc", "fields": [{"name": "total", "description": "t"}]}}
    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    schema_file.write_text(json.dumps(schema_record), encoding="utf-8")

    job_id = "job_scope_1"
    _pipeline_jobs[job_id] = {"job_id": job_id, "status": "queued", "owner": owner,
                              "schema_id": schema_id, "successes": [], "failures": []}
    _job_controls[job_id] = JobControl()

    captured = {}

    def _fake_process_document(**kwargs):
        captured.update(kwargs)
        return [(_mock_page("job marker"), {})]

    result_obj = MagicMock()
    result_obj.model_dump.return_value = {"total": 1}

    try:
        with patch("app.api.pipeline_routes.process_document", side_effect=_fake_process_document), \
             patch("app.api.pipeline_routes._build_pages_for_pdf", return_value=[MagicMock()]), \
             patch("src.adapters.llm.factory.get_llm_client"), \
             patch("src.ai.layer3_extraction.schema_validation.extract_with_retry", return_value=result_obj), \
             patch("src.ai.layer3_extraction.storage.save_markdown_record"), \
             patch("src.ai.layer3_extraction.storage.save_extraction_run", return_value=1), \
             patch("src.ai.layer3_extraction.extractor.extract_document", return_value=result_obj):
            _run_pipeline_job(job_id, [pdf], schema_file, schema_record, None, owner)

        assert captured.get("ir_owner") == owner
        assert captured.get("output_dir") == str(resolve_owner_output_dir(owner))
        succ = _pipeline_jobs[job_id]["successes"]
        assert succ and succ[0]["ir_persisted"] is True
        assert load_page_irs(owner=owner, doc_id=f"{stem}.pdf") is not None
    finally:
        _pipeline_jobs.pop(job_id, None)
        _job_controls.pop(job_id, None)
        pdf.unlink(missing_ok=True)
        schema_file.unlink(missing_ok=True)


def test_delete_my_document_removes_ir(client, cleanup_dirs):
    """Deleting a user's own document removes that owner's IR."""
    from src.ai.layer3_extraction.ir_store import save_page_irs
    owner = "extract_owner_a"
    doc = "to_delete.pdf"
    # seed an IR for this owner
    page = PageIR(page_number=1, blocks=[Block(block_id="b1", type=BlockType.PARAGRAPH,
                 text="x", bbox=BBox(x0=0, y0=0, x1=1, y1=1), reading_order=0)], source_engine="docling")
    save_page_irs(owner=owner, doc_id=doc, pages=[page])
    assert load_page_irs(owner=owner, doc_id=doc) is not None

    auth = _auth(client, owner, "pw_a")
    with patch("src.ai.layer3_extraction.s3_documents.is_configured", return_value=False), \
         patch("app.api.user_routes._user_docs_dir") as mock_dir:
        # make the "document file" exist so the route reaches the IR-cleanup step
        d = OUTPUT_DIR / "_users" / owner / "documents"
        d.mkdir(parents=True, exist_ok=True)
        (d / doc).write_bytes(b"%PDF-1.4 x")
        mock_dir.return_value = d
        resp = client.delete(f"/me/documents/{doc}", headers=auth)

    assert resp.status_code == 200, resp.text
    assert resp.json().get("ir_removed") is True
    assert load_page_irs(owner=owner, doc_id=doc) is None
