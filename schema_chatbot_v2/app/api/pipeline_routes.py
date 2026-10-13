from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from PIL import Image

from app.core.activity_log import log_activity
from app.core.auth import get_current_user, require_admin
from app.storage.user_store import User

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "dataset_output"
SCHEMA_REGISTRY = ROOT / "schema_registry"

for d in (DATASET_DIR, OUTPUT_DIR, SCHEMA_REGISTRY):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, PageOutput, VLMAnalysis
from src.ai.layer1_routing.pipeline import process_document
from src.config.settings import settings as a_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
PIPELINE_AVAILABLE = True
router = APIRouter()


class JobControl:
    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.pause_event.set()  # set = running; cleared = paused
        self.kill_event = threading.Event()  # set = kill requested


_pipeline_jobs: Dict[str, Dict[str, Any]] = {}
_job_controls: Dict[str, JobControl] = {}

MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(20 * 1024 * 1024)))  # 20MB
RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_EXTRACT_MAX_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_EXTRACT_WINDOW_SECONDS", "60"))
_extract_rate_limits: Dict[str, List[float]] = {}
_rate_limit_lock = threading.Lock()


def check_user_extract_rate_limit(username: str) -> None:
    """
    Enforces a simple sliding window per-user rate limit on document extraction.
    """
    now = time.time()
    with _rate_limit_lock:
        history = _extract_rate_limits.setdefault(username, [])
        valid_history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(valid_history) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: maximum {RATE_LIMIT_MAX_REQUESTS} extraction requests per {RATE_LIMIT_WINDOW_SECONDS}s allowed."
            )
        valid_history.append(now)
        _extract_rate_limits[username] = valid_history


def reset_extract_rate_limits() -> None:
    """
    Resets in-memory extraction rate limits (for testing and isolation).
    """
    with _rate_limit_lock:
        _extract_rate_limits.clear()


async def validate_uploaded_pdf(file: UploadFile) -> Tuple[str, bytes]:
    """
    Validates that the upload is a non-empty .pdf without path traversal,
    and does not exceed MAX_UPLOAD_SIZE_BYTES.
    Returns (safe_filename, content_bytes).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")
    p = Path(file.filename)
    if p.name != file.filename or ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected.")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_BYTES} bytes."
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return p.name, content


RENDER_DPI = 150
MAT = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)


class MockLLMClient(LLMClient):
    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        raise RuntimeError(
            "MockLLMClient.classify_page called. Pages need a real VLM for scanned/mixed content."
        )

    def analyze_page(self, image_bytes: bytes, page_profile_hint: dict) -> VLMAnalysis:
        raise RuntimeError(
            "MockLLMClient.analyze_page called."
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> Tuple[str, float]:
        raise RuntimeError(
            "MockLLMClient.transcribe_handwriting called."
        )


def _render_page(page: Any) -> Tuple[np.ndarray, bytes]:
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    png = pix.tobytes("png")
    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return arr, png


def _build_pages_for_pdf(pdf_path: Path, pdf_bytes: Optional[bytes] = None) -> List[Dict[str, Any]]:
    # Open from in-memory bytes (S3-sourced) when provided, else from a local path.
    if pdf_bytes is not None:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        arr, png = _render_page(page)
        pages.append({
            "page": page,
            "page_number": i + 1,
            "context": {
                "pdf_path": str(pdf_path),
                "image_array": arr,
                "image_bytes": png,
            },
            "image_bytes": png,
        })
    return pages


# ========================= Documents =========================


@router.get("/documents")
def list_documents(
    _: User = Depends(require_admin),
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    from src.ai.layer3_extraction import s3_documents as s3
    from src.ai.layer3_extraction.storage import list_markdown_records

    # Processed outputs are DB-backed (converted markdown records).
    output_records = list_markdown_records(owner=owner or "_admin")
    outputs = [
        {
            "name": o["name"],
            "size": o["size"],
            "modified": o["modified"],
            "schema_ref": o.get("schema_ref") or ({"schema_id": o["schema_id"]} if o.get("schema_id") else None),
        }
        for o in output_records
    ]
    # Stems that have a processed output, to compute has_output per document.
    processed_stems = set()
    for o in output_records:
        src = o.get("source_pdf") or ""
        processed_stems.add(Path(src).stem if src else Path(o["name"]).stem)

    def _has_output(name: str) -> bool:
        return Path(name).stem in processed_stems

    # Source PDFs are stored in S3 (shared documents prefix). Fall back to the
    # local dataset/ folder only when S3 is not configured (dev convenience).
    if s3.is_configured():
        docs = [
            {
                "name": d["name"],
                "size": d["size"],
                "modified": d["modified"],
                "has_output": _has_output(d["name"]),
                "source": "s3",
            }
            for d in sorted(s3.list_shared_documents(), key=lambda x: x["name"].lower())
        ]
    else:
        pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
        docs = [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "has_output": _has_output(p.name),
                "source": "disk",
            }
            for p in pdfs
        ]

    return {"documents": docs, "outputs": outputs}


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    from src.ai.layer3_extraction import s3_documents as s3
    saved: List[str] = []
    for f in files:
        safe_name, content = await validate_uploaded_pdf(f)
        content_type = f.content_type or "application/pdf"

        # Store the source PDF in S3 (shared documents prefix); fall back to disk.
        if s3.is_configured():
            s3.upload_shared_document(safe_name, content, content_type=content_type)
        else:
            (DATASET_DIR / safe_name).write_bytes(content)
        saved.append(safe_name)

        # Persist uploaded document into PostgreSQL documents table (shadow copy).
        try:
            from src.ai.layer3_extraction.storage import save_document
            doc_owner = "_admin" if getattr(user, "role", None) == Role.ADMIN else user.username
            save_document(filename=safe_name, file_bytes=content, owner=doc_owner, content_type=content_type)
        except Exception:
            pass
    return {"saved": saved, "count": len(saved)}


def _validate_doc_filename(filename: str) -> str:
    """
    Validates a document filename is a safe single-component .pdf name
    (no path traversal). Returns the sanitized name.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    p = Path(filename)
    if p.name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected.")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf documents are supported.")
    return p.name


@router.delete("/documents/{filename}")
def delete_document(filename: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    Delete a shared document PDF from S3 (admin only).
    """
    from src.ai.layer3_extraction import s3_documents as s3
    safe_name = _validate_doc_filename(filename)

    if s3.is_configured():
        if not s3.shared_document_exists(safe_name):
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        if not s3.delete_shared_document(safe_name):
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}' from storage.")
    else:
        target = DATASET_DIR / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        try:
            target.unlink()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}': {exc}")

    logger.info("documents.deleted", doc_name=safe_name, by=_.username)
    return {"deleted": safe_name}


@router.get("/documents/output/{name}")
def download_output(name: str, owner: Optional[str] = None, _: User = Depends(require_admin)) -> Response:
    """
    Download the extracted-JSON output for a processed document (admin).
    `name` may be the output filename (e.g. foo.md / foo.extracted.json) or the
    source pdf/stem. Returns the extracted_data as a JSON attachment.
    """
    from src.ai.layer3_extraction.storage import get_latest_extracted_data
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid output name.")
    stem = Path(name).stem.replace(".extracted", "")
    eff_owner = owner or _lookup_doc_owner(name, stem) or "_admin"
    data = get_latest_extracted_data(stem, owner=eff_owner)
    if data is None and eff_owner != "_admin":
        data = get_latest_extracted_data(stem, owner="_admin")
    if data is None:
        raise HTTPException(status_code=404, detail=f"No extracted output found for '{name}'.")
    payload = json.dumps(data, indent=2, default=str)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{stem}.extracted.json"'},
    )


# ========================= Schemas =========================


@router.get("/schemas")
def list_schemas(_: User = Depends(require_admin)) -> Dict[str, Any]:
    # DB-backed: schemas are read from the PostgreSQL `schemas` table, not disk.
    from src.ai.layer3_extraction.storage import list_schema_records
    records = list_schema_records(owner=None)  # admin sees all
    schemas: List[Dict[str, Any]] = [
        {
            "file": f"{r['schema_id']}.json",
            "schema_id": r["schema_id"],
            "document_type": r["document_type"],
            "confirmed_at": r["confirmed_at"],
            "field_count": r["field_count"],
            "sample_documents": r.get("sample_documents", []),
        }
        for r in records
    ]
    return {"source": "database", "schemas": schemas}


@router.get("/schemas/{schema_id}")
def get_schema(schema_id: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    from src.ai.layer3_extraction.storage import get_schema_record
    record = get_schema_record(schema_id)
    if record is None:
        raise HTTPException(status_code=404, detail="schema not found")
    return record


# ========================= Pipeline =========================


@router.get("/pipeline/status")
def pipeline_status(_: User = Depends(require_admin)) -> Dict[str, Any]:
    routing_mode = getattr(a_settings, "routing_mode", None) if PIPELINE_AVAILABLE else None
    layer3_strategy = getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")
    return {
        "available": PIPELINE_AVAILABLE,
        "routing_mode": routing_mode,
        "layer3_strategy": layer3_strategy,
        "jobs": {
            k: {
                "job_id": k,
                "schema_id": v.get("schema_id"),
                "schema_file": v.get("schema_file"),
                "strategy": v.get("strategy", layer3_strategy),
                "status": v["status"],
                "created_at": v["created_at"],
                "total": len(v.get("targets", [])),
                "completed": len(v.get("successes", [])) + len(v.get("failures", [])),
            }
            for k, v in _pipeline_jobs.items()
        },
    }


@router.get("/pipeline/jobs/{job_id}")
def get_pipeline_job(job_id: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return _pipeline_jobs[job_id]


@router.post("/pipeline/jobs/{job_id}/pause")
def pause_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] not in ("running", "queued"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause job with status '{job['status']}'. Job must be 'running' or 'queued'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.clear()
    job["status"] = "paused"
    logger.info("pipeline.job_paused", job_id=job_id)
    log_activity(admin.username, "job_paused", {"job_id": job_id})
    return {"job_id": job_id, "status": "paused", "message": f"Job {job_id} paused."}


@router.post("/pipeline/jobs/{job_id}/resume")
def resume_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status '{job['status']}'. Job must be 'paused'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.set()
    job["status"] = "running"
    logger.info("pipeline.job_resumed", job_id=job_id)
    log_activity(admin.username, "job_resumed", {"job_id": job_id})
    return {"job_id": job_id, "status": "running", "message": f"Job {job_id} resumed."}


@router.post("/pipeline/jobs/{job_id}/kill")
def kill_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] in ("completed", "killed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot kill job with status '{job['status']}'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.kill_event.set()
        ctrl.pause_event.set()  # Unblock thread if it was paused
    job["status"] = "killed"
    if not job.get("finished_at"):
        job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    logger.info("pipeline.job_kill_requested", job_id=job_id)
    log_activity(admin.username, "job_killed", {"job_id": job_id})
    return {"job_id": job_id, "status": "killed", "message": f"Job {job_id} killed."}


@router.get("/pipeline/jobs/{job_id}/pdf", response_class=Response)
def download_job_pdf_report(job_id: str, user: User = Depends(get_current_user)) -> Response:
    """
    Download the generated PDF report for a job directly from PostgreSQL.
    If not yet stored in PostgreSQL, generate it now, save it, and stream it.
    """
    if job_id in _pipeline_jobs:
        job = _pipeline_jobs[job_id]
        if user.role != "admin" and job.get("owner") != user.username:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this job")

    from src.ai.layer3_extraction.storage import get_job_pdf, save_job_pdf
    from src.utils.job_pdf_report import generate_job_pdf

    res = get_job_pdf(job_id)
    if res:
        pdf_bytes, filename = res
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # If not stored yet in PostgreSQL, check if the job is in memory
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job report not found in database or memory")

    job = _pipeline_jobs[job_id]
    try:
        pdf_bytes = generate_job_pdf(job)
        filename = f"{job_id}_report.pdf"
        save_job_pdf(job_id, pdf_bytes, filename)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("pipeline.job_pdf_generation_failed", job_id=job_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {exc}")


@router.post("/pipeline/run")
async def run_pipeline(
    schema_id: str = Form(...),
    documents: Optional[str] = Form(default=None),
    strategy: Optional[str] = Form(default=None),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Extraction pipeline unavailable (import failed)")

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    # DB-backed: load the confirmed schema from PostgreSQL, not the disk registry.
    from src.ai.layer3_extraction.storage import get_schema_record
    schema_record = get_schema_record(schema_id)
    if schema_record is None:
        raise HTTPException(status_code=404, detail=f"schema {schema_id} not found")
    # Display-only stand-in path; the worker only reads schema_path.name, never the file.
    schema_path = Path(f"{schema_id}.json")

    from src.ai.layer3_extraction import s3_documents as s3

    requested_names: Optional[List[str]] = None
    if documents:
        try:
            parsed = json.loads(documents)
            if isinstance(parsed, list):
                requested_names = [str(n) for n in parsed]
        except Exception:
            requested_names = None

    if s3.is_configured():
        # Resolve target PDFs from S3 (shared documents prefix) into local temp files.
        available = {d["name"]: d["key"] for d in s3.list_shared_documents()}
        names = requested_names if requested_names else sorted(available.keys())
        names = [n for n in names if n in available]
        if not names:
            raise HTTPException(status_code=400, detail="No documents to process")
        tmp_dir = Path(tempfile.mkdtemp(prefix="idp_docs_"))
        targets = []
        for n in names:
            data = s3.download_bytes_by_key(available[n])
            if not data:
                continue
            tp = tmp_dir / n
            tp.write_bytes(data)
            targets.append(tp)
        if not targets:
            raise HTTPException(status_code=400, detail="No documents to process")
    else:
        all_pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
        if requested_names:
            selected = [DATASET_DIR / n for n in requested_names if (DATASET_DIR / n).exists()]
            targets = selected if selected else all_pdfs
        else:
            targets = all_pdfs
        if not targets:
            raise HTTPException(status_code=400, detail="No documents to process")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_id": schema_id,
        "schema_file": schema_path.name,
        "strategy": selected_strategy,
        "targets": [p.name for p in targets],
        "successes": [],
        "failures": [],
        "started_at": None,
        "finished_at": None,
        "wall_time_s": None,
    }
    _job_controls[job_id] = JobControl()

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, targets, schema_path, schema_record, selected_strategy, job.get("owner")),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "targets": len(targets), "strategy": selected_strategy}


def _parse_pages_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Extract individual pages from an already-generated pipeline markdown file."""
    marker = re.compile(r"<!--\s*PAGE\s+(\d+)[^>]*-->", re.I)
    matches = list(marker.finditer(md_text))
    if not matches:
        return [{"markdown": md_text.strip(), "page_number": 1}]
    pages = []
    for i, m in enumerate(matches):
        pg_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end]
        close_match = re.search(rf"<!--\s*/PAGE\s+{pg_num}\s*-->", content, re.I)
        if close_match:
            content = content[:close_match.start()]
        else:
            content = re.sub(r"<!--\s*/PAGE\s+\d+\s*-->", "", content)
        pages.append({"markdown": content.strip(), "page_number": pg_num})
    return pages


def _run_pipeline_job(
    job_id: str,
    targets: List[Path],
    schema_path: Path,
    schema_record: Dict[str, Any],
    strategy: Optional[str] = None,
    owner: Optional[str] = None,
) -> None:
    job = _pipeline_jobs[job_id]
    ctrl = _job_controls.get(job_id)
    if not ctrl:
        ctrl = JobControl()
        _job_controls[job_id] = ctrl

    selected_strategy = strategy or job.get("strategy") or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")
    job["strategy"] = selected_strategy
    job_owner = owner or job.get("owner") or "_admin"
    job["status"] = "running"
    job["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        from src.adapters.llm.factory import get_llm_client
        llm_client = get_llm_client()
    except Exception:
        llm_client = MockLLMClient()
    schema_id = schema_record["schema_id"]

    t_start = time.monotonic()
    for pdf in targets:
        job["current_document"] = pdf.name
        # Check before starting next PDF: pause wait loop and kill check
        while not ctrl.kill_event.is_set():
            if ctrl.pause_event.is_set():
                break
            time.sleep(0.2)

        if ctrl.kill_event.is_set():
            logger.info("pipeline.job_killed", job_id=job_id, pdf=pdf.name)
            job["status"] = "killed"
            job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            job["wall_time_s"] = round(time.monotonic() - t_start, 2)
            return

        owner = job.get("owner")
        job_out_dir = resolve_owner_output_dir(owner) if owner else OUTPUT_DIR
        job_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Discover existing converted markdown (disk or DB) to avoid destructive overwrite
        existing_md_text = None
        md_path = job_out_dir / f"{pdf.stem}.md"
        cand_files = [
            md_path,
            OUTPUT_DIR / f"{pdf.stem}.md",
            DATASET_DIR / f"{pdf.stem}.md",
        ]
        for cand in cand_files:
            if cand.exists() and cand.is_file():
                try:
                    c_txt = cand.read_text(encoding="utf-8")
                    if len(c_txt.strip()) > 100:
                        existing_md_text = c_txt
                        break
                except Exception:
                    pass

        if not existing_md_text:
            try:
                from src.ai.layer3_extraction.storage import get_markdown_record
                rec = get_markdown_record(pdf.name, owner=job_owner) or get_markdown_record(f"{pdf.stem}.md", owner=job_owner)
                if rec and rec.get("markdown_content") and len(rec["markdown_content"].strip()) > 100:
                    existing_md_text = rec["markdown_content"]
            except Exception:
                pass

        t0 = time.monotonic()
        try:
            results = None
            try:
                pages = _build_pages_for_pdf(pdf)
                results = process_document(
                    pages=pages,
                    llm_client=llm_client,
                    document_name=pdf.name,
                    document_id=f"{job_id}-{pdf.stem}",
                    write_output=True,
                    output_dir=str(job_out_dir),
                    overwrite=True,
                )
            except Exception as proc_exc:
                logger.warning("pipeline.process_document_failed", pdf=pdf.name, error=str(proc_exc))
                if not existing_md_text:
                    raise

            outputs = [r[0] for r in results] if results else []
            total_chars = sum(len(o.markdown.strip()) for o in outputs)
            avg_conf = sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0

            pages_for_layer3 = [{"markdown": r[0].markdown, "page_number": r[0].page_number} for r in results] if results else []

            # If local conversion yielded 0 chars (e.g. paddleocr missing in web container),
            # restore pre-existing valid markdown so Layer 3 receives full document text
            if total_chars == 0 and existing_md_text:
                logger.warning(
                    "pipeline.conversion_produced_0_chars_restoring_existing_md",
                    document=pdf.name,
                    existing_chars=len(existing_md_text),
                )
                md_path.write_text(existing_md_text, encoding="utf-8")
                parsed_pages = _parse_pages_from_markdown(existing_md_text)
                if parsed_pages and any(p.get("markdown") for p in parsed_pages):
                    pages_for_layer3 = parsed_pages
                    total_chars = sum(len(p.get("markdown", "").strip()) for p in parsed_pages)
                    conf_match = re.search(r"Avg conf\s*:\s*([\d\.]+)", existing_md_text)
                    if conf_match:
                        try:
                            avg_conf = float(conf_match.group(1))
                        except Exception:
                            avg_conf = 0.95
                    else:
                        avg_conf = 0.95

            ref_path = job_out_dir / f"{pdf.stem}.schema_ref.json"
            ref_dict = {
                "source_pdf": pdf.name,
                "output_md": md_path.name,
                "schema_id": schema_id,
                "schema_registry_file": schema_path.name,
                "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                "pages": [
                    {
                        "page_number": p.get("page_number", idx + 1),
                        "engines_used": getattr(results[idx][0], "engines_used", ["paddleocr_printed"]) if (results and idx < len(results)) else ["paddleocr_printed"],
                        "capabilities": getattr(results[idx][0], "capabilities", ["has_printed_scan"]) if (results and idx < len(results)) else ["has_printed_scan"],
                        "confidence": avg_conf if total_chars > 0 and (not results or not any(r[0].markdown.strip() for r in results)) else (results[idx][0].confidence if (results and idx < len(results)) else avg_conf),
                        "escalated": getattr(results[idx][0], "escalated", False) if (results and idx < len(results)) else False,
                        "low_confidence": False if total_chars > 0 else (results[idx][0].low_confidence if (results and idx < len(results)) else True),
                        "chars": len(p.get("markdown", "").strip()),
                    }
                    for idx, p in enumerate(pages_for_layer3)
                ],
            }
            if job.get("owner"):
                ref_dict["owner"] = job["owner"]
            ref_path.write_text(
                json.dumps(ref_dict, indent=2) + "\n",
                encoding="utf-8",
            )

            # Persist converted Markdown (.md) into PostgreSQL document_markdowns table
            try:
                from src.ai.layer3_extraction.storage import save_markdown_record
                md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                save_markdown_record(
                    doc_id=pdf.name,
                    md_filename=md_path.name,
                    markdown_content=md_text,
                    owner=job_owner,
                    schema_id=schema_id,
                    schema_ref_json=ref_dict,
                    page_count=len(pages_for_layer3),
                    pages_json=pages_for_layer3,
                )
            except Exception:
                pass

            elapsed = time.monotonic() - t0

            # ================= Automatically Continue to Layer 3 Extraction =================
            extracted_data = None
            db_id = None
            db_error = None
            extract_elapsed = 0.0
            graph_meta: Dict[str, Any] = {}

            try:
                from src.adapters.llm.extraction_factory import get_extraction_client
                from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
                from src.ai.layer3_extraction.schema_validation import extract_with_retry
                from src.ai.layer3_extraction.storage import init_db, save_extraction_run, save_document_graph
                from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

                raw_fields = (schema_record.get("schema") or {}).get("fields", [])
                fields_in = []
                for f in raw_fields:
                    fname = f.get("name") if isinstance(f, dict) else str(f)
                    fdesc = f.get("description", "") if isinstance(f, dict) else ""
                    if fname:
                        fields_in.append(SchemaFieldIn(name=fname, description=fdesc))

                if fields_in:
                    dynamic_schema = build_dynamic_schema(fields_in)
                    t_ext0 = time.monotonic()
                    extraction_llm = get_extraction_client()

                    extracted_result = extract_with_retry(
                        lambda: extract_document(
                            pages_for_layer3,
                            dynamic_schema,
                            extraction_llm,
                            strategy=selected_strategy,
                            graph_out=graph_meta,
                            owner=job_owner,
                        ),
                        dynamic_schema,
                    )
                    extract_elapsed = round(time.monotonic() - t_ext0, 2)
                    extracted_data = extracted_result.model_dump()

                    # Write dataset_output/<owner>/<stem>.extracted.json
                    json_out_path = job_out_dir / f"{pdf.stem}.extracted.json"
                    json_payload = {
                        "source_pdf": pdf.name,
                        "schema_used": schema_id,
                        "strategy_used": selected_strategy,
                        "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                        "processing_time_seconds": extract_elapsed,
                        "extracted_data": extracted_data,
                    }
                    if graph_meta:
                        json_payload["graph_memory"] = graph_meta
                    if job.get("owner"):
                        json_payload["owner"] = job["owner"]

                    json_out_path.write_text(
                        json.dumps(json_payload, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    # Save file checkpoint: dataset_output/<owner>/<stem>.graph.json (Requirement 7)
                    if graph_meta and graph_meta.get("snapshot"):
                        try:
                            graph_out_path = job_out_dir / f"{pdf.stem}.graph.json"
                            graph_out_path.write_text(
                                json.dumps(graph_meta["snapshot"], indent=2) + "\n",
                                encoding="utf-8",
                            )
                        except Exception as gf_exc:
                            logger.warning("pipeline.extract.graph_file_save_failed", pdf=pdf.name, error=str(gf_exc))

                    # Persist to PostgreSQL (Requirement 6)
                    try:
                        init_db()
                        db_id = save_extraction_run(
                            doc_id=pdf.name,
                            page_count=len(pages_for_layer3),
                            schema_name=schema_id,
                            result_json=extracted_data,
                            owner=job_owner,
                            llm_provider=a_settings.extraction_backend,
                            model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                            processing_time_seconds=extract_elapsed,
                            page_outputs=outputs if total_chars > 0 and results and any(r[0].markdown.strip() for r in results) else None,
                        )
                        if graph_meta and graph_meta.get("snapshot"):
                            save_document_graph(
                                doc_id=pdf.name,
                                graph_dict=graph_meta["snapshot"],
                                job_id=job_id,
                                owner=job.get("owner"),
                                schema_id=schema_id,
                                strategy=selected_strategy,
                            )
                            logger.info("pipeline.extract.graph_persisted", pdf=pdf.name, job_id=job_id)
                    except Exception as db_exc:
                        db_error = str(db_exc)
                        logger.warning("pipeline.extract.db_save_failed", pdf=pdf.name, error=str(db_exc))
            except Exception as ext_exc:
                logger.error("pipeline.layer3_extraction_failed", pdf=pdf.name, error=str(ext_exc))

            job["successes"].append({
                "pdf": pdf.name,
                "md": md_path.name,
                "schema_ref": ref_path.name,
                "avg_conf": round(avg_conf, 3),
                "chars": total_chars,
                "pages": len(pages_for_layer3),
                "elapsed_s": round(elapsed, 2),
                "strategy": selected_strategy,
                "extracted_json": f"{pdf.stem}.extracted.json" if extracted_data else None,
                "extracted_data": extracted_data,
                "graph_stats": graph_meta.get("stats") if graph_meta else None,
                "graph_nodes": len(graph_meta.get("snapshot", {}).get("nodes", [])) if graph_meta else 0,
                "graph_edges": len(graph_meta.get("snapshot", {}).get("edges", [])) if graph_meta else 0,
                "graph_memory": graph_meta if graph_meta else None,
                "extract_elapsed_s": extract_elapsed,
                "db_run_id": db_id,
                "db_error": db_error,
            })
        except Exception as exc:
            logger.error("pipeline.doc_failed", pdf=pdf.name, error=str(exc), error_type=type(exc).__name__)
            job["failures"].append({
                "pdf": pdf.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(time.monotonic() - t0, 2),
            })

        # Check after PDF completed if kill was requested during doc processing
        if ctrl.kill_event.is_set():
            logger.info("pipeline.job_killed_after_doc", job_id=job_id, pdf=pdf.name)
            job["status"] = "killed"
            job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            job["wall_time_s"] = round(time.monotonic() - t_start, 2)
            return

    if ctrl.kill_event.is_set():
        job["status"] = "killed"
    else:
        job["status"] = "completed"
    job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    job["wall_time_s"] = round(time.monotonic() - t_start, 2)

    # Automatically generate full JSON PDF report and persist into PostgreSQL
    try:
        from src.ai.layer3_extraction.storage import save_job_pdf
        from src.utils.job_pdf_report import generate_job_pdf
        pdf_bytes = generate_job_pdf(job)
        save_job_pdf(job_id=job_id, pdf_bytes=pdf_bytes, filename=f"{job_id}_report.pdf")
        logger.info("pipeline.job_pdf_auto_saved", job_id=job_id, size=len(pdf_bytes))
    except Exception as pdf_exc:
        logger.warning("pipeline.job_pdf_auto_save_failed", job_id=job_id, error=str(pdf_exc))


# ========================= Layer 3 Extraction =========================


# _parse_pages_from_markdown is defined above _run_pipeline_job



_SCHEMA_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-]+$")


def resolve_owner_output_dir(owner: str) -> Path:
    """
    Safely resolves OUTPUT_DIR / owner and guarantees it stays within OUTPUT_DIR.
    Normalizes owner to lowercase and rejects consecutive dots or traversal attempts.
    """
    if not owner or not isinstance(owner, str):
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    owner_clean = owner.strip().lower()
    if not owner_clean or ".." in owner_clean or "/" in owner_clean or "\\" in owner_clean:
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    owner_dir = (OUTPUT_DIR / owner_clean).resolve()
    if not owner_dir.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    return owner_dir


def validate_md_filename(md_name: str, owner: Optional[str] = None) -> Path:
    """
    Validates that md_name is a strict single-component filename ending in .md
    and resolving strictly within OUTPUT_DIR or an owner subfolder.
    """
    if not md_name or not isinstance(md_name, str):
        raise HTTPException(status_code=400, detail="Invalid md_name.")
    p = Path(md_name)
    if p.suffix.lower() != ".md" or p.name != md_name or ".." in md_name or "/" in md_name or "\\" in md_name:
        raise HTTPException(status_code=400, detail="Invalid md_name: must be a simple filename ending with .md")

    # 1. If owner is provided, search exclusively in owner's directory first
    if owner:
        owner_dir = resolve_owner_output_dir(owner)
        owner_file = (owner_dir / p).resolve()
        if not owner_file.is_relative_to(OUTPUT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Path traversal detected.")
        if owner_file.is_file():
            return owner_file

    # 2. Check root OUTPUT_DIR
    resolved = (OUTPUT_DIR / p).resolve()
    if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected.")
    if resolved.is_file():
        return resolved

    # 3. If not in root and no owner specified, search owner subdirectories
    cands = [cand.resolve() for cand in OUTPUT_DIR.glob(f"*/{p.name}") if cand.is_file() and cand.resolve().is_relative_to(OUTPUT_DIR.resolve())]
    if len(cands) == 1:
        return cands[0]
    elif len(cands) > 1:
        owners = sorted(list(set(c.parent.name for c in cands)))
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous md_name '{md_name}' found in multiple owner directories: {owners}. Specify owner or document path explicitly."
        )

    return resolved


def validate_doc_id(doc_id: str) -> str:
    """
    Validates that doc_id is a strict single-component document identifier or filename
    and resolves strictly within OUTPUT_DIR without directory escape.
    """
    if not doc_id or not isinstance(doc_id, str):
        raise HTTPException(status_code=400, detail="Invalid doc_id.")
    p = Path(doc_id)
    if p.name != doc_id or ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Path traversal detected in doc_id.")
    resolved = (OUTPUT_DIR / p).resolve()
    if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected in doc_id.")
    return doc_id


def load_strict_schema_record(schema_id: str, owner: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Validates schema_id against strict regex ^[A-Za-z0-9_\-]+$ and loads the
    confirmed schema record from the PostgreSQL `schemas` table (DB-backed).

    Tenant isolation: when ``owner`` is provided (always, for user-facing
    endpoints), the resolved schema must belong to that owner. This is enforced
    in BOTH the DB path (via get_schema_record(..., owner=owner)) AND this
    function's own schema_registry/ DISK FALLBACK below. A schema whose stored
    ``owner`` is set and differs from the caller is treated as not-found (404),
    so one tenant can never resolve another tenant's schema by guessing its
    schema_id. NULL-owner records are shared/admin and remain visible to all.
    Pass owner=None only on admin-gated code paths.
    """
    if not schema_id or not _SCHEMA_ID_REGEX.fullmatch(schema_id):
        raise HTTPException(status_code=400, detail=f"Invalid schema_id format: '{schema_id}'")
    from src.ai.layer3_extraction.storage import get_schema_record
    record = get_schema_record(schema_id, owner=owner)
    if record is None:
        reg_file = SCHEMA_REGISTRY / f"{schema_id}.json"
        if reg_file.is_file():
            try:
                data = json.loads(reg_file.read_text(encoding="utf-8"))
                # Owner-scope the disk fallback: deny if the file is owned by a
                # different tenant. NULL/absent owner == shared, visible to all.
                if owner is not None:
                    rec_owner = data.get("owner")
                    if rec_owner is not None and rec_owner != owner:
                        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found.")
                return data
            except HTTPException:
                raise
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found.")
    return record


resolve_strict_schema_file = load_strict_schema_record


@router.get("/pipeline/outputs")
def list_pipeline_outputs(_: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    List all documents that have already finished Layer 1 & 2 processing,
    along with their bound target schema and whether Layer 3 extracted JSON exists.
    """
    outputs = sorted([p for p in OUTPUT_DIR.glob("*.md") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for md_path in outputs:
        ref_path = md_path.with_suffix(".schema_ref.json")
        ext_path = md_path.with_name(f"{md_path.stem}.extracted.json")
        ref_data: Dict[str, Any] = {}
        if ref_path.exists():
            try:
                ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
            except Exception:
                ref_data = {}

        schema_id = ref_data.get("schema_id")
        doc_type = ref_data.get("document_type")
        page_count = len(ref_data.get("pages", [])) if ref_data.get("pages") else None

        # Check schema (DB-backed) for field count if available
        field_count = None
        if schema_id:
            try:
                from src.ai.layer3_extraction.storage import get_schema_record
                s_data = get_schema_record(schema_id)
                if s_data:
                    field_count = len((s_data.get("schema") or {}).get("fields", []))
                    if not doc_type:
                        doc_type = s_data.get("document_type")
            except Exception:
                pass

        has_extracted = ext_path.exists()
        extracted_data = None
        if has_extracted:
            try:
                extracted_data = json.loads(ext_path.read_text(encoding="utf-8")).get("extracted_data")
            except Exception:
                pass

        items.append({
            "md_name": md_path.name,
            "stem": md_path.stem,
            "source_pdf": ref_data.get("source_pdf", f"{md_path.stem}.pdf"),
            "schema_id": schema_id,
            "document_type": doc_type or "Document",
            "field_count": field_count,
            "page_count": page_count,
            "modified": datetime.fromtimestamp(md_path.stat().st_mtime).isoformat(),
            "has_extracted": has_extracted,
            "extracted_data": extracted_data,
        })
    return {"outputs": items}


@router.post("/pipeline/extract/from-output")
async def extract_from_existing_output(
    md_name: str = Form(...),
    strategy: Optional[str] = Form(default=None),
    owner: Optional[str] = Form(default=None),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    """
    Run Layer 3 extraction directly on an ALREADY-PROCESSED pipeline document.
    Automatically retrieves the converted Markdown and the schema bound to it during Run Pipeline.
    Saves the extracted structured JSON to dataset_output/<stem>.extracted.json and stores in PostgreSQL.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)
    if not is_admin:
        if not owner or owner != user.username:
            raise HTTPException(status_code=403, detail="Access denied: non-admin users may only extract from their own output partition.")

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    md_path = validate_md_filename(md_name, owner=owner)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{md_name}' not found in dataset_output.")

    ref_path = md_path.with_suffix(".schema_ref.json")
    if not ref_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No schema reference found for '{md_name}'. Make sure the pipeline was run with a target schema.",
        )

    ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
    known_owner = owner or ref_data.get("owner") or _lookup_doc_owner(ref_data.get("source_pdf", md_path.stem), md_path.stem, username=owner)
    schema_id = ref_data.get("schema_id")
    if not schema_id:
        raise HTTPException(status_code=400, detail=f"No schema_id recorded in {ref_path.name}")

    # Owner-scoped: resolve the schema as the document's owner (non-admin callers
    # are already constrained above to owner == their own username).
    schema_json = load_strict_schema_record(schema_id, owner=known_owner)
    raw_fields = (schema_json.get("schema") or {}).get("fields", [])
    fields_in: List[SchemaFieldIn] = []
    for f in raw_fields:
        name = f.get("name") if isinstance(f, dict) else str(f)
        desc = f.get("description", "") if isinstance(f, dict) else ""
        if name:
            fields_in.append(SchemaFieldIn(name=name, description=desc))

    if not fields_in:
        raise HTTPException(status_code=400, detail=f"Schema '{schema_id}' contains no field definitions.")

    dynamic_schema = build_dynamic_schema(fields_in)
    md_content = md_path.read_text(encoding="utf-8")
    if len(md_content.strip()) < 100:
        cand = OUTPUT_DIR / md_path.name
        if cand.exists() and cand != md_path:
            try:
                c_txt = cand.read_text(encoding="utf-8")
                if len(c_txt.strip()) > 100:
                    md_content = c_txt
                    md_path.write_text(md_content, encoding="utf-8")
            except Exception:
                pass
        if len(md_content.strip()) < 100:
            try:
                from src.ai.layer3_extraction.storage import get_markdown_record
                rec = get_markdown_record(md_path.stem, owner=user.username) or get_markdown_record(f"{md_path.stem}.pdf", owner=user.username)
                if rec and rec.get("markdown_content") and len(rec["markdown_content"].strip()) > 100:
                    md_content = rec["markdown_content"]
                    md_path.write_text(md_content, encoding="utf-8")
            except Exception:
                pass
    pages = _parse_pages_from_markdown(md_content)

    def _execute():
        t_start = time.time()
        extraction_llm = get_extraction_client()
        graph_meta: Dict[str, Any] = {}
        extracted_result = extract_with_retry(
            lambda: extract_document(pages, dynamic_schema, extraction_llm, strategy=selected_strategy, graph_out=graph_meta, owner=known_owner or "_admin"),
            dynamic_schema,
        )
        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        # Save to output dir (same dir as source md)
        stem = md_path.stem
        out_dir = md_path.parent
        json_out_path = out_dir / f"{stem}.extracted.json"
        out_payload = {
            "source_doc": md_name,
            "source_pdf": ref_data.get("source_pdf", f"{stem}.pdf"),
            "schema_used": schema_id,
            "strategy_used": selected_strategy,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
        }
        if graph_meta:
            out_payload["graph_memory"] = graph_meta
        if known_owner:
            out_payload["owner"] = known_owner
        json_out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

        if known_owner and not ref_data.get("owner"):
            ref_data["owner"] = known_owner
            try:
                ref_path.write_text(json.dumps(ref_data, indent=2) + "\n", encoding="utf-8")
            except Exception as e_ref:
                logger.warning("pipeline.extract_from_output.schema_ref_update_failed", error=str(e_ref))

        # Save to PostgreSQL
        db_id = None
        db_error = None
        try:
            init_db()
            page_outputs = []
            for p_info in ref_data.get("pages", []):
                page_outputs.append(PageOutput(
                    page_number=p_info.get("page_number", 1),
                    markdown="",
                    engines_used=p_info.get("engines_used", ["pipeline_output"]),
                    confidence=p_info.get("confidence", 0.95),
                    capabilities=p_info.get("capabilities", []),
                    escalated=p_info.get("escalated", False),
                    escalation_attempts=0,
                    low_confidence=p_info.get("low_confidence", False),
                ))

            eff_run_owner = known_owner or user.username
            db_id = save_extraction_run(
                doc_id=ref_data.get("source_pdf", md_name),
                page_count=len(pages),
                schema_name=schema_id,
                result_json=result_dict,
                owner=eff_run_owner,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                processing_time_seconds=elapsed,
                page_outputs=page_outputs if page_outputs else None,
            )
        except Exception as exc:
            db_error = str(exc)
            logger.warning("pipeline.extract_from_output.db_save_failed", error=str(exc))

        return {
            "success": True,
            "doc_id": ref_data.get("source_pdf", md_name),
            "schema": schema_id,
            "document_type": ref_data.get("document_type", "Document"),
            "strategy": selected_strategy,
            "data": result_dict,
            "graph_stats": graph_meta.get("stats") if graph_meta else None,
            "graph_memory": graph_meta if graph_meta else None,
            "meta": {
                "pages_processed": len(pages),
                "processing_time_seconds": elapsed,
                "strategy": selected_strategy,
                "source_md": md_name,
                "output_json": f"{stem}.extracted.json",
                "llm_provider": a_settings.extraction_backend,
                "saved_to_db": db_id is not None,
                "db_run_id": db_id,
                "db_error": db_error,
            }
        }

    try:
        return await run_in_threadpool(_execute)
    except Exception as exc:
        logger.exception("pipeline.extract_from_output.failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


@router.post("/pipeline/extract")
async def extract_uploaded_document(
    file: UploadFile = File(...),
    schema_id: Optional[str] = Form(default=None),
    raw_schema: Optional[str] = Form(default=None),
    strategy: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    End-to-End Extraction (single PDF upload):
      1. Receives PDF file + target schema.
      2. Runs Layer 1 (inspection & routing) + Layer 2 (conversion & OCR).
      3. Saves converted markdown to dataset_output/<stem>.md.
      4. Runs Layer 3 (Sarvam/Ollama page-by-page field extraction with scratchpad & early stopping, or Graph Memory).
      5. Saves output JSON to dataset_output/<stem>.extracted.json.
      6. Persists extraction run into PostgreSQL database.
      7. Returns structured JSON result and metadata.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.adapters.llm.factory import get_llm_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
    from src.ai.layer3_extraction.page_loader import load_pages_with_confidence
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    # Rate limiting
    check_user_extract_rate_limit(user.username)

    # Allowed extension, filename, and size validation
    filename, pdf_bytes = await validate_uploaded_pdf(file)

    # 1. Resolve Target Schema
    fields_in: List[SchemaFieldIn] = []
    schema_name = "custom_schema"
    if schema_id:
        # Owner-scoped: a user may only resolve their own (or a shared) schema.
        schema_data = load_strict_schema_record(schema_id, owner=user.username)
        schema_name = schema_data.get("schema_id", schema_id)
        raw_fields = (schema_data.get("schema") or {}).get("fields", [])
        for f in raw_fields:
            name = f.get("name") if isinstance(f, dict) else str(f)
            desc = f.get("description", "") if isinstance(f, dict) else ""
            if name:
                fields_in.append(SchemaFieldIn(name=name, description=desc))
    elif raw_schema:
        try:
            parsed = json.loads(raw_schema)
            if isinstance(parsed, dict) and "fields" in parsed:
                parsed = parsed["fields"]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        fields_in.append(SchemaFieldIn(name=item.get("name", ""), description=item.get("description", "")))
                    elif isinstance(item, str):
                        fields_in.append(SchemaFieldIn(name=item, description=""))
            schema_name = ",".join(f.name for f in fields_in)[:100]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid raw_schema format: {exc}")

    if not fields_in:
        raise HTTPException(status_code=400, detail="No target schema provided. Please select a schema or specify fields.")

    dynamic_schema = build_dynamic_schema(fields_in)

    # Store the uploaded source PDF in S3 (shared documents prefix) for persistence.
    from src.ai.layer3_extraction import s3_documents as s3
    if s3.is_configured():
        s3.upload_shared_document(filename, pdf_bytes, content_type=file.content_type or "application/pdf")

    # Persist uploaded document into PostgreSQL documents table (shadow copy)
    try:
        from src.ai.layer3_extraction.storage import save_document
        save_document(filename=filename, file_bytes=pdf_bytes, owner=user.username, content_type=file.content_type or "application/pdf")
    except Exception:
        pass

    def _execute():
        t_start = time.time()

        # Render directly from the in-memory upload bytes (no local disk write).
        pages = _build_pages_for_pdf(Path(filename), pdf_bytes=pdf_bytes)
        try:
            llm_client = get_llm_client()
        except Exception:
            llm_client = MockLLMClient()

        results = process_document(
            pages=pages,
            llm_client=llm_client,
            document_name=filename,
            document_id=f"extract-{uuid.uuid4().hex[:6]}",
            write_output=True,
            output_dir=str(OUTPUT_DIR),
            overwrite=True,
        )

        stem = Path(filename).stem
        md_file = OUTPUT_DIR / f"{stem}.md"

        # Persist converted Markdown (.md) into PostgreSQL document_markdowns table
        try:
            from src.ai.layer3_extraction.storage import save_markdown_record
            md_text = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
            save_markdown_record(
                doc_id=filename,
                md_filename=md_file.name,
                markdown_content=md_text,
                owner=user.username,
                schema_id=schema_id or schema_name,
                page_count=len(results),
                pages_json=[{"page_number": r[0].page_number, "markdown": r[0].markdown} for r in results],
            )
        except Exception:
            pass

        page_outputs = [r[0] for r in results]
        pages_md = load_pages_with_confidence(page_outputs)

        extraction_llm = get_extraction_client()
        graph_meta: Dict[str, Any] = {}
        extracted_result = extract_with_retry(
            lambda: extract_document(pages_md, dynamic_schema, extraction_llm, strategy=selected_strategy, graph_out=graph_meta, owner=user.username),
            dynamic_schema,
        )

        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        json_out_path = OUTPUT_DIR / f"{stem}.extracted.json"
        out_payload = {
            "source_pdf": filename,
            "schema_used": schema_name,
            "strategy_used": selected_strategy,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
            "owner": user.username,
        }
        if graph_meta:
            out_payload["graph_memory"] = graph_meta
            try:
                from src.ai.layer3_extraction.storage import save_document_graph
                save_document_graph(
                    doc_id=filename,
                    job_id=None,
                    schema_id=schema_name,
                    strategy=selected_strategy,
                    graph_dict=graph_meta,
                    owner=user.username,
                )
            except Exception as g_err:
                logger.warning("pipeline.extract.save_graph_failed", error=str(g_err))
        json_out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

        db_id = None
        db_error = None
        try:
            init_db()
            db_id = save_extraction_run(
                doc_id=filename,
                page_count=len(page_outputs),
                schema_name=schema_name,
                result_json=result_dict,
                owner=user.username,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                processing_time_seconds=elapsed,
                page_outputs=page_outputs,
            )
        except Exception as exc:
            db_error = str(exc)
            logger.warning("pipeline.extract.db_save_failed", error=str(exc))

        return {
            "success": True,
            "doc_id": filename,
            "schema": schema_name,
            "strategy": selected_strategy,
            "data": result_dict,
            "graph_stats": graph_meta.get("stats") if graph_meta else None,
            "graph_memory": graph_meta if graph_meta else None,
            "meta": {
                "pages_processed": len(page_outputs),
                "processing_time_seconds": elapsed,
                "strategy": selected_strategy,
                "output_md": f"{stem}.md",
                "output_json": f"{stem}.extracted.json",
                "llm_provider": a_settings.extraction_backend,
                "saved_to_db": db_id is not None,
                "db_run_id": db_id,
                "db_error": db_error,
            }
        }

    try:
        return await run_in_threadpool(_execute)
    except Exception as exc:
        logger.exception("pipeline.extract.failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


# ========================= Query Bot (JSON Q&A) =========================

# ========================= Query Bot (Graph Memory & JSON Fallback) =========================

from pydantic import BaseModel, Field
from app.core.auth import get_current_user
from app.storage.user_store import Role


def _lookup_doc_owner(doc_id: str, stem: str, username: Optional[str] = None, job_id: Optional[str] = None) -> Optional[str]:
    """
    Looks up the recorded owner of a document from PostgreSQL DB or disk metadata.
    If username is provided, checks that user's partition only (DB owner==username or OUTPUT_DIR/username).
    """
    from src.ai.layer3_extraction.storage import init_db, SessionLocal, DocumentGraphRecord
    try:
        init_db()
        sess = SessionLocal()
        try:
            candidates = [doc_id, f"{stem}.pdf", stem]
            q = sess.query(DocumentGraphRecord).filter(
                DocumentGraphRecord.doc_id.in_(candidates)
            )
            if job_id:
                q = q.filter_by(job_id=job_id)
            if username:
                user_rec = q.filter_by(owner=username).order_by(DocumentGraphRecord.created_at.desc()).first()
                if user_rec and user_rec.owner:
                    return user_rec.owner
            else:
                rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()
                if rec and rec.owner:
                    return rec.owner
        finally:
            sess.close()
    except Exception:
        pass

    if username:
        try:
            search_dirs = [resolve_owner_output_dir(username), OUTPUT_DIR]
        except HTTPException:
            search_dirs = [OUTPUT_DIR]
    else:
        search_dirs = [OUTPUT_DIR] + [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.resolve().is_relative_to(OUTPUT_DIR.resolve())]

    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for fname in (f"{stem}.extracted.json", f"{doc_id}.extracted.json", f"{stem}.schema_ref.json"):
            p = sdir / fname
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        owner = data.get("owner")
                        if owner:
                            if username:
                                if owner == username:
                                    return owner
                            else:
                                return owner
                except Exception:
                    pass
    return None


def _check_doc_access(owner: Optional[str], user: User, is_admin: bool) -> None:
    """
    Enforces that ownerless records are readable by admins only, and user records
    are readable only by the owning user or admins.
    """
    if is_admin:
        return
    if not owner:
        raise HTTPException(
            status_code=403,
            detail="Access denied: ownerless document readable by admin only.",
        )
    if owner != user.username:
        raise HTTPException(
            status_code=403,
            detail="Access denied: document belongs to another user.",
        )


class QueryBotRequest(BaseModel):
    extracted_data: Optional[Any] = Field(None, description="Full extracted JSON object or record dictionary")
    question: str = Field(..., description="User's natural language question about the extracted data")
    doc_id: Optional[str] = Field(None, description="Optional document name for reference")
    job_id: Optional[str] = Field(None, description="Optional job ID for reference")
    owner: Optional[str] = Field(None, description="Optional document owner for reference (required for admins when doc_id is ambiguous)")



@router.get("/api/query-bot/documents")
def get_query_bot_documents(
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all documents available for Query Bot inspection.
    Gathers documents from:
      1. PostgreSQL DocumentGraphRecord (with tenant isolation for non-admin users)
      2. Disk dataset_output/ (*.extracted.json, *.graph.json, *.schema_ref.json)
      3. Active in-memory pipeline jobs (_pipeline_jobs)
    """
    from src.ai.layer3_extraction.storage import (
        init_db,
        SessionLocal,
        DocumentGraphRecord,
    )

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)

    docs: Dict[str, Dict[str, Any]] = {}

    # Source 1: PostgreSQL Database
    try:
        init_db()
        session = SessionLocal()
        try:
            q = session.query(DocumentGraphRecord)
            if not is_admin:
                # Regular user: see only own documents (no ownerless docs)
                q = q.filter(DocumentGraphRecord.owner == user.username)
            for rec in q.order_by(DocumentGraphRecord.created_at.desc()).all():
                stem = Path(rec.doc_id).stem
                doc_key = f"{stem}.pdf" if not rec.doc_id.lower().endswith(".pdf") else rec.doc_id
                docs[doc_key] = {
                    "doc_id": doc_key,
                    "stem": stem,
                    "label": doc_key,
                    "has_graph": True,
                    "has_extracted": True,
                    "strategy": "graph_memory",
                    "source": "database",
                    "job_id": rec.job_id,
                    "owner": rec.owner,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                }
        finally:
            session.close()
    except Exception as db_err:
        logger.warning("query_bot.list_docs_db_error", error=str(db_err))

    # Source 2 (in-memory active pipeline jobs)
    # NOTE: The former disk source (dataset_output/*.graph.json / *.extracted.json)
    # has been removed. Documents/graphs/extractions are now DB-backed only.
    try:
        for job_id, job in _pipeline_jobs.items():
            if job.get("status") != "completed":
                continue
            job_owner = job.get("owner")
            if not is_admin:
                if not job_owner or job_owner != user.username:
                    continue
            for s in job.get("successes", []):
                doc_id = s.get("pdf") or s.get("doc_id") or (
                    s.get("extracted_json", "").replace(".extracted.json", ".pdf")
                    if s.get("extracted_json") else None
                )
                if doc_id:
                    stem = Path(doc_id).stem
                    strat = s.get("strategy") or job.get("strategy") or "graph_memory_concurrent"
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "doc_id": doc_id,
                            "stem": stem,
                            "label": doc_id,
                            "has_graph": (strat.startswith("graph_memory")),
                            "has_extracted": True,
                            "strategy": strat,
                            "source": "active_job",
                            "job_id": job_id,
                            "owner": job_owner,
                            "created_at": job.get("created_at"),
                        }
                    else:
                        docs[doc_id]["job_id"] = job_id
                        if strat:
                            docs[doc_id]["strategy"] = strat
    except Exception as job_err:
        logger.warning("query_bot.list_docs_job_error", error=str(job_err))

    sorted_docs = sorted(docs.values(), key=lambda d: d.get("created_at") or "", reverse=True)
    return {"documents": sorted_docs, "count": len(sorted_docs)}


@router.post("/api/query-bot/ask")
async def ask_query_bot(
    req: QueryBotRequest,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Query Bot:
    1. If Document Graph Memory is available (from PostgreSQL DB, file checkpoint dataset_output/<stem>.graph.json,
       or request payload's graph_memory), queries the graph via GraphQueryService with bounded multi-hop traversal
       and verbatim source citations (mode: "graph").
    2. Strictly enforces user/document isolation (User A cannot query User B's document graph).
    3. If no graph is found, falls back cleanly to the legacy extracted JSON LLM assistant (mode: "json_fallback").
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from src.ai.layer3_extraction.storage import (
        init_db,
        SessionLocal,
        DocumentGraphRecord,
    )
    try:
        from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
        from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
    except ImportError as _g_imp_err:
        logger.warning("query_bot.graph_import_unavailable", error=str(_g_imp_err))
        GraphMemory = None
        GraphQueryService = None

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)

    if not is_admin and req.owner and req.owner != user.username:
        raise HTTPException(
            status_code=403,
            detail="Access denied: non-admin users cannot specify a different owner.",
        )

    # For admin: require explicit owner or job_id if doc_id is ambiguous across multiple owners
    if is_admin and req.doc_id and not req.owner and not req.job_id:
        stem = Path(req.doc_id).stem
        candidates = [req.doc_id, f"{stem}.pdf", stem]
        ambiguous_owners = set()
        try:
            init_db()
            session = SessionLocal()
            try:
                db_owners = session.query(DocumentGraphRecord.owner).filter(
                    DocumentGraphRecord.doc_id.in_(candidates),
                    DocumentGraphRecord.owner.isnot(None),
                ).distinct().all()
                for (o,) in db_owners:
                    if o:
                        ambiguous_owners.add(o)
            finally:
                session.close()
        except Exception:
            pass

        try:
            if OUTPUT_DIR.is_dir():
                for cdir in OUTPUT_DIR.iterdir():
                    if cdir.is_dir():
                        for fname in (f"{stem}.extracted.json", f"{req.doc_id}.extracted.json", f"{stem}.graph.json", f"{stem}.schema_ref.json"):
                            if (cdir / fname).is_file():
                                ambiguous_owners.add(cdir.name)
                                break
        except Exception:
            pass

        if len(ambiguous_owners) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Ambiguous doc_id '{req.doc_id}' exists for multiple owners ({', '.join(sorted(ambiguous_owners))}). Specify 'owner' or 'job_id' in the request.",
            )

    graph_dict = None

    # Validate doc_id up-front to prevent path traversal
    if req.doc_id:
        req.doc_id = validate_doc_id(req.doc_id)
        stem = Path(req.doc_id).stem
        if not is_admin:
            # Check if requesting user owns a copy in their partition/DB
            user_owner = _lookup_doc_owner(req.doc_id, stem, username=user.username)
            if not user_owner:
                # User does not own a copy; check if the document belongs to another user
                doc_owner = _lookup_doc_owner(req.doc_id, stem)
                if doc_owner and doc_owner != user.username:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied: document owned by '{doc_owner}'.",
                    )

    # Step 1: Check document graph in DB with strict user isolation
    if req.doc_id or req.job_id:
        try:
            init_db()
            session = SessionLocal()
            try:
                q = session.query(DocumentGraphRecord)
                if req.doc_id:
                    stem = Path(req.doc_id).stem
                    doc_candidates = [req.doc_id, f"{stem}.pdf", stem]
                    q = q.filter(DocumentGraphRecord.doc_id.in_(doc_candidates))
                    if req.job_id:
                        sub_q = q.filter_by(job_id=req.job_id)
                        if sub_q.first():
                            q = sub_q
                elif req.job_id:
                    q = q.filter_by(job_id=req.job_id)
                if not is_admin:
                    user_q = q.filter_by(owner=user.username)
                    db_rec = user_q.order_by(DocumentGraphRecord.created_at.desc()).first()
                    if not db_rec:
                        any_rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()
                        if any_rec:
                            _check_doc_access(any_rec.owner, user, is_admin)
                else:
                    if req.owner:
                        q = q.filter_by(owner=req.owner)
                    db_rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()

                if db_rec:
                    # Enforce isolation: ownerless is admin-only, user docs are owner-only
                    _check_doc_access(db_rec.owner, user, is_admin)
                    graph_dict = db_rec.graph_json
            finally:
                session.close()
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("query_bot.db_lookup_error", error=str(exc))

    # Step 2 (removed): the former on-disk graph checkpoint fallback
    # (dataset_output/<stem>.graph.json) has been dropped. Graph memory is now
    # read exclusively from PostgreSQL (Step 1) or the request payload (Step 3).

    # Step 3: Check extracted_data payload for embedded graph
    if not graph_dict and req.extracted_data and isinstance(req.extracted_data, dict):
        if req.doc_id:
            stem = Path(req.doc_id).stem
            payload_owner = _lookup_doc_owner(req.doc_id, stem, username=(req.owner if is_admin else user.username), job_id=req.job_id)
            _check_doc_access(payload_owner, user, is_admin)
        if "graph_memory" in req.extracted_data and isinstance(req.extracted_data["graph_memory"], dict):
            graph_dict = req.extracted_data["graph_memory"].get("snapshot") or req.extracted_data["graph_memory"]
        elif "nodes" in req.extracted_data and "edges" in req.extracted_data:
            graph_dict = req.extracted_data

    # Path A: Graph Memory answering
    if GraphMemory and GraphQueryService and graph_dict and isinstance(graph_dict, dict) and "nodes" in graph_dict:
        try:
            graph_mem = GraphMemory.from_dict(graph_dict)
            service = GraphQueryService()
            result = service.query(graph_mem, req.question)
            return {
                "success": True,
                "question": req.question,
                "answer": result.answer,
                "sources": result.sources[:5] if result.sources else [],
                "doc_id": req.doc_id,
                "mode": "graph",
                "hops_traversed": result.hops_traversed,
                "retrieved_nodes_count": len(result.retrieved_nodes),
                "retrieved_edges_count": len(result.retrieved_edges),
            }
        except Exception as g_err:
            logger.error("query_bot.graph_query_failed", error=str(g_err))
            # Fall back to json_fallback if extracted_data is available

    # Path B: JSON Fallback (DB-backed) — read the latest extraction result for
    # this document from PostgreSQL (extraction_runs) instead of disk.
    if not req.extracted_data and req.doc_id:
        stem = Path(req.doc_id).stem
        eff_owner = req.owner if (is_admin and req.owner) else user.username
        # Enforce the same ownership checks as before using DB/sidecar ownership lookup.
        file_owner = _lookup_doc_owner(
            req.doc_id, stem,
            username=eff_owner,
            job_id=req.job_id,
        )
        _check_doc_access(file_owner, user, is_admin)
        try:
            from src.ai.layer3_extraction.storage import get_latest_extracted_data
            target_owner = req.owner if (is_admin and req.owner) else (file_owner if (is_admin and file_owner) else user.username)
            db_extracted = get_latest_extracted_data(req.doc_id, owner=target_owner)
            if db_extracted:
                req.extracted_data = db_extracted
            elif target_owner:
                owner_dir = resolve_owner_output_dir(target_owner)
                owner_ext_file = owner_dir / f"{stem}.extracted.json"
                if owner_ext_file.is_file():
                    ext_json = json.loads(owner_ext_file.read_text(encoding="utf-8"))
                    req.extracted_data = ext_json.get("extracted_data") or ext_json
        except Exception as ex_err:
            logger.warning("query_bot.db_extracted_lookup_error", error=str(ex_err))

    if not req.extracted_data:
        raise HTTPException(
            status_code=400,
            detail="No document graph found and no extracted JSON data provided to answer the question.",
        )

    import os
    import httpx
    from starlette.concurrency import run_in_threadpool
    from app.config import settings as app_settings

    def _query_llm() -> str:
        data_json_str = (
            req.extracted_data
            if isinstance(req.extracted_data, str)
            else json.dumps(req.extracted_data, indent=2)
        )

        prompt = (
            "You are a friendly document assistant. The user has uploaded a document and the following data was extracted from it:\n\n"
            f"{data_json_str}\n\n"
            "Answer the user's question naturally and conversationally, like a helpful human assistant. "
            "Use the data above to answer. Do not show JSON, field names, asterisks, or technical formatting. "
            "Just give a clear, plain English answer. If the information is not in the data, say so politely.\n\n"
            f"User: {req.question}\nAssistant:"
        )

        api_key = (
            getattr(app_settings, "sarvam_api_key", None)
            or getattr(a_settings, "sarvam_api_key", None)
            or os.getenv("IDP_SARVAM_API_KEY")
            or os.getenv("SARVAM_API_KEY")
            or ""
        )
        base_url = (
            getattr(app_settings, "sarvam_base_url", None)
            or getattr(a_settings, "sarvam_base_url", None)
            or os.getenv("IDP_SARVAM_BASE_URL")
            or os.getenv("SARVAM_BASE_URL")
            or "https://api.sarvam.ai/v1"
        ).rstrip("/")
        model_name = (
            getattr(app_settings, "sarvam_model", None)
            or getattr(a_settings, "sarvam_model_name", None)
            or os.getenv("IDP_SARVAM_MODEL_NAME")
            or os.getenv("SARVAM_MODEL")
            or "sarvam-105b"
        )
        backend = (
            getattr(app_settings, "llm_provider", None)
            or getattr(a_settings, "extraction_backend", None)
            or os.getenv("IDP_EXTRACTION_BACKEND")
            or os.getenv("LLM_PROVIDER")
            or "bedrock"
        ).lower()

        try:
            if backend == "bedrock":
                bedrock_model_id = (
                    getattr(app_settings, "bedrock_model_id", None)
                    or getattr(a_settings, "bedrock_model_id", None)
                    or os.getenv("IDP_BEDROCK_MODEL_ID")
                    or os.getenv("BEDROCK_MODEL_ID")
                    or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
                )
                bedrock_region = (
                    getattr(app_settings, "bedrock_region", None)
                    or getattr(a_settings, "bedrock_region", None)
                    or os.getenv("IDP_BEDROCK_REGION")
                    or os.getenv("BEDROCK_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "ap-south-1"
                )
                import boto3
                profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
                if profile:
                    b_client = boto3.Session(profile_name=profile, region_name=bedrock_region).client("bedrock-runtime", region_name=bedrock_region)
                else:
                    b_client = boto3.client("bedrock-runtime", region_name=bedrock_region)

                kwargs = {
                    "modelId": bedrock_model_id,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"temperature": 0.2, "maxTokens": 600},
                }
                effort = getattr(app_settings, "bedrock_reasoning_effort", None) or getattr(a_settings, "bedrock_reasoning_effort", None) or "low"
                if effort and ("gpt-oss" in bedrock_model_id.lower() or "qdtz23c8eis1" in bedrock_model_id):
                    kwargs["additionalModelRequestFields"] = {"reasoning_effort": effort}

                try:
                    b_resp = b_client.converse(**kwargs)
                except Exception:
                    kwargs.pop("additionalModelRequestFields", None)
                    b_resp = b_client.converse(**kwargs)

                content = b_resp.get("output", {}).get("message", {}).get("content", [])
                text = next((block["text"] for block in content if isinstance(block, dict) and "text" in block), "")
                if text.strip():
                    return text.strip()

            elif (backend == "sarvam" or api_key) and api_key:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                    "reasoning_effort": None,
                }
                headers = {
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                resp = httpx.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                if resp.is_success:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content") or msg.get("reasoning_content") or ""
                        # Strip internal thinking tags if present
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                        # Strip markdown bold/italic markers
                        content = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", content)
                        # Strip "Assistant:" prefix if model echoed it
                        content = re.sub(r"^Assistant:\s*", "", content.strip())
                        if content.strip():
                            return content.strip()
                logger.warning(
                    "query_bot.sarvam_error",
                    status_code=resp.status_code,
                    body=resp.text[:200],
                )
                if resp.status_code in (401, 402, 403):
                    return f"Sarvam API Auth Error ({resp.status_code}): {resp.text}"

            elif backend == "ollama":
                # Fallback to Ollama (only if explicitly requested)
                ollama_url = getattr(a_settings, "ollama_base_url", "http://localhost:11434/v1").rstrip("/v1")
                ollama_model = getattr(a_settings, "extraction_model_name", "llama3.1")
                resp = httpx.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
                return f"LLM error: HTTP {resp.status_code}"
            else:
                pass
        except Exception as e:
            logger.warning("query_bot.llm_call_failed", error=str(e))

        # Deterministic extraction fallback from extracted_data if LLM was unavailable
        try:
            extracted_dict = req.extracted_data if isinstance(req.extracted_data, dict) else {}
            norm_q = req.question.lower()
            for k, v in extracted_dict.items():
                if k.replace("_", " ") in norm_q or norm_q in str(v).lower():
                    return f"{k.replace('_', ' ').title()}: {v}"
        except Exception:
            pass

        return f"Error communicating with LLM: {str(e)}" if "e" in locals() else "Unable to answer: no response from primary LLM provider."

    try:
        answer = await run_in_threadpool(_query_llm)
        return {
            "success": True,
            "question": req.question,
            "answer": answer,
            "sources": [],
            "doc_id": req.doc_id,
            "mode": "json_fallback",
        }
    except Exception as exc:
        logger.exception("query_bot.error")
        raise HTTPException(status_code=500, detail=f"Query bot failed: {exc}")





