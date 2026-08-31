from __future__ import annotations

import io
import json
import logging
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image

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

from app.core.auth import get_current_user
from app.models.auth_models import User, UserRole
from app.storage.audit_log import get_audit_logger

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


def _build_pages_for_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
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
def list_documents(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
    outputs = sorted([p for p in OUTPUT_DIR.glob("*.md") if p.is_file()])

    def _output_info(md_path: Path) -> Dict[str, Any]:
        ref_path = md_path.with_suffix(".schema_ref.json")
        ref: Optional[Dict[str, Any]] = None
        if ref_path.exists():
            try:
                ref = json.loads(ref_path.read_text(encoding="utf-8"))
            except Exception:
                ref = None
        return {
            "name": md_path.name,
            "size": md_path.stat().st_size,
            "modified": datetime.fromtimestamp(md_path.stat().st_mtime).isoformat(),
            "schema_ref": ref,
        }

    return {
        "dataset_dir": str(DATASET_DIR),
        "output_dir": str(OUTPUT_DIR),
        "documents": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "has_output": (OUTPUT_DIR / f"{p.stem}.md").exists(),
            }
            for p in pdfs
        ],
        "outputs": [_output_info(md) for md in outputs],
    }


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    saved: List[str] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf") and f.content_type != "application/pdf":
            continue
        content = await f.read()
        target = DATASET_DIR / Path(f.filename or f"doc_{uuid.uuid4().hex[:8]}.pdf").name
        target.write_bytes(content)
        saved.append(target.name)

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="DOCUMENT_UPLOAD",
        details={"count": len(saved), "files": saved},
    )

    return {"saved": saved, "count": len(saved), "uploaded_by": current_user.username}


# ========================= Schemas =========================


@router.get("/schemas")
def list_schemas() -> Dict[str, Any]:
    files = sorted(SCHEMA_REGISTRY.glob("schema_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    schemas: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            schemas.append({
                "file": f.name,
                "schema_id": data.get("schema_id"),
                "document_type": (data.get("schema") or {}).get("document_type") or data.get("document_type"),
                "confirmed_at": data.get("confirmed_at"),
                "field_count": len((data.get("schema") or {}).get("fields", [])),
                "sample_documents": data.get("sample_documents", []),
                "created_by": data.get("created_by", "user1"),
            })
        except Exception as exc:
            logger.warning("schema.parse_failed", file=f.name, error=str(exc))
    return {"schema_registry_dir": str(SCHEMA_REGISTRY), "schemas": schemas}


@router.get("/schemas/{schema_id}")
def get_schema(schema_id: str) -> Dict[str, Any]:
    target = SCHEMA_REGISTRY / f"{schema_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="schema not found")
    return json.loads(target.read_text(encoding="utf-8"))


# ========================= Pipeline =========================


@router.get("/pipeline/status")
def pipeline_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    routing_mode = getattr(a_settings, "routing_mode", None) if PIPELINE_AVAILABLE else None

    # Role-based filtering: Normal users see their own jobs; Admin sees all
    is_admin = current_user.role == UserRole.ADMIN

    filtered_jobs = {}
    for k, v in _pipeline_jobs.items():
        job_owner = v.get("owner", "user1")
        if is_admin or job_owner.lower() == current_user.username.lower():
            filtered_jobs[k] = {
                "job_id": k,
                "schema_id": v.get("schema_id"),
                "schema_file": v.get("schema_file"),
                "status": v["status"],
                "owner": job_owner,
                "owner_role": v.get("owner_role", "normal"),
                "created_at": v["created_at"],
                "total": len(v.get("targets", [])),
                "completed": len(v.get("successes", [])) + len(v.get("failures", [])),
                "successes_count": len(v.get("successes", [])),
                "failures_count": len(v.get("failures", [])),
            }

    # Summary metrics for current user
    user_jobs = list(filtered_jobs.values())
    total_jobs = len(user_jobs)
    completed_jobs = sum(1 for j in user_jobs if j["status"] == "completed")
    running_jobs = sum(1 for j in user_jobs if j["status"] == "running")
    paused_jobs = sum(1 for j in user_jobs if j["status"] == "paused")
    remaining_jobs = sum(1 for j in user_jobs if j["status"] in ("running", "queued", "paused"))
    error_jobs = sum(1 for j in user_jobs if j["failures_count"] > 0 or j["status"] == "failed")

    return {
        "available": PIPELINE_AVAILABLE,
        "routing_mode": routing_mode,
        "current_user": {
            "username": current_user.username,
            "role": current_user.role.value,
            "full_name": current_user.full_name,
        },
        "my_summary": {
            "total": total_jobs,
            "completed": completed_jobs,
            "running": running_jobs,
            "paused": paused_jobs,
            "remaining": remaining_jobs,
            "errors": error_jobs,
        },
        "jobs": filtered_jobs,
    }


@router.get("/pipeline/jobs/{job_id}")
def get_pipeline_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]

    # Non-admins can only see their own jobs
    if current_user.role != UserRole.ADMIN and job.get("owner") and job.get("owner").lower() != current_user.username.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to view other users' jobs",
        )

    return job


@router.post("/pipeline/jobs/{job_id}/pause")
def pause_pipeline_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]

    if current_user.role != UserRole.ADMIN and job.get("owner") and job.get("owner").lower() != current_user.username.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot pause another user's job",
        )

    if job["status"] not in ("running", "queued"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause job with status '{job['status']}'. Job must be 'running' or 'queued'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.clear()
    job["status"] = "paused"
    logger.info("pipeline.job_paused", job_id=job_id, user=current_user.username)

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="JOB_PAUSE",
        details={"job_id": job_id},
    )

    return {"job_id": job_id, "status": "paused", "message": f"Job {job_id} paused."}


@router.post("/pipeline/jobs/{job_id}/resume")
def resume_pipeline_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]

    if current_user.role != UserRole.ADMIN and job.get("owner") and job.get("owner").lower() != current_user.username.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot resume another user's job",
        )

    if job["status"] != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status '{job['status']}'. Job must be 'paused'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.set()
    job["status"] = "running"
    logger.info("pipeline.job_resumed", job_id=job_id, user=current_user.username)

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="JOB_RESUME",
        details={"job_id": job_id},
    )

    return {"job_id": job_id, "status": "running", "message": f"Job {job_id} resumed."}


@router.post("/pipeline/jobs/{job_id}/kill")
def kill_pipeline_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]

    if current_user.role != UserRole.ADMIN and job.get("owner") and job.get("owner").lower() != current_user.username.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot kill another user's job",
        )

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
    logger.info("pipeline.job_kill_requested", job_id=job_id, user=current_user.username)

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="JOB_KILL",
        details={"job_id": job_id},
    )

    return {"job_id": job_id, "status": "killed", "message": f"Job {job_id} killed."}


@router.post("/pipeline/run")
async def run_pipeline(
    schema_id: str = Form(...),
    documents: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Extraction pipeline unavailable (import failed)")

    schema_path = SCHEMA_REGISTRY / f"{schema_id}.json"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail=f"schema {schema_id} not found in registry")

    try:
        schema_record = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid schema file: {exc}")

    all_pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
    if documents:
        try:
            doc_list = json.loads(documents)
            selected = []
            for name in doc_list:
                p = DATASET_DIR / name
                if p.exists():
                    selected.append(p)
            targets = selected if selected else all_pdfs
        except Exception:
            targets = all_pdfs
    else:
        targets = all_pdfs

    if not targets:
        raise HTTPException(status_code=400, detail="No documents to process")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "owner": current_user.username,
        "owner_role": current_user.role.value,
        "status": "queued",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_id": schema_id,
        "schema_file": schema_path.name,
        "targets": [p.name for p in targets],
        "successes": [],
        "failures": [],
        "started_at": None,
        "finished_at": None,
        "wall_time_s": None,
    }
    _job_controls[job_id] = JobControl()

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="JOB_START",
        details={
            "job_id": job_id,
            "schema_id": schema_id,
            "targets_count": len(targets),
            "targets": [p.name for p in targets],
        },
    )

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, targets, schema_path, schema_record),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "targets": len(targets),
        "owner": current_user.username,
    }


def _run_pipeline_job(
    job_id: str,
    targets: List[Path],
    schema_path: Path,
    schema_record: Dict[str, Any],
) -> None:
    job = _pipeline_jobs[job_id]
    ctrl = _job_controls.get(job_id)
    if not ctrl:
        ctrl = JobControl()
        _job_controls[job_id] = ctrl

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

        t0 = time.monotonic()
        try:
            pages = _build_pages_for_pdf(pdf)
            results = process_document(
                pages=pages,
                llm_client=llm_client,
                document_name=pdf.name,
                document_id=f"{job_id}-{pdf.stem}",
                write_output=True,
                output_dir=str(OUTPUT_DIR),
                overwrite=True,
            )

            md_path = OUTPUT_DIR / f"{pdf.stem}.md"
            ref_path = OUTPUT_DIR / f"{pdf.stem}.schema_ref.json"
            ref_path.write_text(
                json.dumps({
                    "source_pdf": pdf.name,
                    "output_md": md_path.name,
                    "schema_id": schema_id,
                    "schema_registry_file": schema_path.name,
                    "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                    "pages": [
                        {
                            "page_number": output.page_number,
                            "engines_used": output.engines_used,
                            "capabilities": output.capabilities,
                            "confidence": output.confidence,
                            "escalated": output.escalated,
                            "low_confidence": output.low_confidence,
                            "chars": len(output.markdown.strip()),
                        }
                        for output, _meta in results
                    ],
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            elapsed = time.monotonic() - t0
            outputs = [r[0] for r in results]
            avg_conf = sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0
            total_chars = sum(len(o.markdown.strip()) for o in outputs)
            job["successes"].append({
                "pdf": pdf.name,
                "md": str(md_path.name),
                "schema_ref": str(ref_path.name),
                "avg_conf": round(avg_conf, 3),
                "chars": total_chars,
                "pages": len(outputs),
                "elapsed_s": round(elapsed, 2),
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

    audit = get_audit_logger()
    audit.log_activity(
        username=job.get("owner", "user1"),
        role=job.get("owner_role", "normal"),
        action="JOB_COMPLETE",
        details={
            "job_id": job_id,
            "status": job["status"],
            "success_count": len(job.get("successes", [])),
            "failure_count": len(job.get("failures", [])),
            "total_docs": len(targets),
            "wall_time_s": job["wall_time_s"],
        },
        status="success" if not job.get("failures") else "warning",
    )
