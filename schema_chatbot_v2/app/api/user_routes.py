"""
Per-user endpoints: private document management, latest-schema resolution,
one-click isolated pipeline trigger, and ownership-scoped job monitoring.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.pipeline_routes import (
    DATASET_DIR,
    SCHEMA_REGISTRY,
    JobControl,
    _job_controls,
    _pipeline_jobs,
    _run_pipeline_job,
)
from app.core.activity_log import log_activity
from app.core.auth import get_current_user
from app.storage.user_store import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/me", tags=["me"])


def _user_docs_dir(username: str) -> Path:
    """
    Returns and ensures the user's isolated documents directory exists.
    """
    d = DATASET_DIR / "_users" / username / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_owned_schema(username: str) -> Optional[Path]:
    """
    Scans SCHEMA_REGISTRY for schemas owned by username, returning the Path
    of the one with the latest confirmed_at timestamp (or None if none exist).
    """
    candidate_files = sorted(SCHEMA_REGISTRY.glob("schema_*.json"))
    matched: List[tuple[str, Path]] = []
    for f in candidate_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("owner") == username:
                confirmed_at = str(data.get("confirmed_at") or "")
                matched.append((confirmed_at, f))
        except Exception as exc:
            logger.warning("user_schema.parse_failed", file=f.name, error=str(exc))
    if not matched:
        return None
    matched.sort(key=lambda item: item[0], reverse=True)
    return matched[0][1]


@router.post("/documents")
async def upload_my_documents(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Uploads PDFs to the authenticated user's isolated document directory.
    """
    user_dir = _user_docs_dir(user.username)
    saved: List[str] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf") and f.content_type != "application/pdf":
            continue
        content = await f.read()
        target = user_dir / Path(f.filename or f"doc_{uuid.uuid4().hex[:8]}.pdf").name
        target.write_bytes(content)
        saved.append(target.name)
    log_activity(user.username, "document_upload", {"files": saved, "count": len(saved)})
    return {"saved": saved, "count": len(saved)}


@router.get("/documents")
def list_my_documents(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Lists all PDF documents in the authenticated user's private workspace.
    """
    user_dir = _user_docs_dir(user.username)
    pdfs = sorted([p for p in user_dir.glob("*.pdf") if p.is_file()])
    return {
        "user_docs_dir": str(user_dir),
        "documents": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            }
            for p in pdfs
        ],
    }


@router.post("/pipeline/run")
async def run_my_pipeline(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Zero-parameter trigger for regular users. Automatically uses the documents
    uploaded to the user's workspace and their most recently confirmed schema.
    """
    user_dir = _user_docs_dir(user.username)
    targets = sorted([p for p in user_dir.glob("*.pdf") if p.is_file()])
    if not targets:
        raise HTTPException(status_code=400, detail="Upload documents first")

    schema_path = _latest_owned_schema(user.username)
    if not schema_path:
        raise HTTPException(status_code=400, detail="Build and confirm a schema first")

    try:
        schema_record = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid schema file: {exc}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_id": schema_record.get("schema_id", schema_path.stem),
        "schema_file": schema_path.name,
        "targets": [p.name for p in targets],
        "successes": [],
        "failures": [],
        "started_at": None,
        "finished_at": None,
        "wall_time_s": None,
        "owner": user.username,
        "current_document": None,
    }
    _job_controls[job_id] = JobControl()

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, targets, schema_path, schema_record),
        daemon=True,
    )
    thread.start()
    log_activity(user.username, "pipeline_run", {"job_id": job_id, "targets": len(targets), "schema_id": schema_record.get("schema_id")})

    return {"job_id": job_id, "status": "queued", "targets": len(targets)}


@router.get("/pipeline/status")
def my_pipeline_status(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns progress and status for only the authenticated user's jobs.
    """
    user_jobs = []
    for k, v in _pipeline_jobs.items():
        if v.get("owner") == user.username:
            total = len(v.get("targets", []))
            succeeded = len(v.get("successes", []))
            failed = len(v.get("failures", []))
            remaining = max(0, total - succeeded - failed)
            user_jobs.append({
                "job_id": v["job_id"],
                "status": v["status"],
                "currently_processing": v.get("current_document"),
                "documents": v.get("targets", []),
                "succeeded": succeeded,
                "failed": failed,
                "remaining": remaining,
                "total": total,
                "created_at": v.get("created_at"),
            })
    return {"jobs": user_jobs}


@router.post("/pipeline/jobs/{job_id}/pause")
def pause_my_pipeline_job(job_id: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    job = _pipeline_jobs.get(job_id)
    if not job or job.get("owner") != user.username:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] not in ("running", "queued"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause job with status '{job['status']}'. Job must be 'running' or 'queued'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.clear()
    job["status"] = "paused"
    logger.info("pipeline.user_job_paused", job_id=job_id, username=user.username)
    log_activity(user.username, "job_paused", {"job_id": job_id})
    return {"job_id": job_id, "status": "paused", "message": f"Job {job_id} paused."}


@router.post("/pipeline/jobs/{job_id}/resume")
def resume_my_pipeline_job(job_id: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    job = _pipeline_jobs.get(job_id)
    if not job or job.get("owner") != user.username:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status '{job['status']}'. Job must be 'paused'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.set()
    job["status"] = "running"
    logger.info("pipeline.user_job_resumed", job_id=job_id, username=user.username)
    log_activity(user.username, "job_resumed", {"job_id": job_id})
    return {"job_id": job_id, "status": "running", "message": f"Job {job_id} resumed."}


@router.post("/pipeline/jobs/{job_id}/kill")
def kill_my_pipeline_job(job_id: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    job = _pipeline_jobs.get(job_id)
    if not job or job.get("owner") != user.username:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] in ("completed", "killed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot kill job with status '{job['status']}'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.kill_event.set()
        ctrl.pause_event.set()
    job["status"] = "killed"
    if not job.get("finished_at"):
        job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    logger.info("pipeline.user_job_kill_requested", job_id=job_id, username=user.username)
    log_activity(user.username, "job_killed", {"job_id": job_id})
    return {"job_id": job_id, "status": "killed", "message": f"Job {job_id} killed."}
