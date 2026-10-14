"""
Per-user endpoints: private document management, latest-schema resolution,
one-click isolated pipeline trigger, and ownership-scoped job monitoring and downloads.
"""
from __future__ import annotations

import json
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from app.api.pipeline_routes import (
    DATASET_DIR,
    SCHEMA_REGISTRY,
    JobControl,
    _job_controls,
    _pipeline_jobs,
    _run_pipeline_job,
    download_job_pdf_report,
)
from app.core.activity_log import log_activity
from app.core.auth import get_current_user
from app.storage.user_store import User
from src.config.settings import settings as a_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/me", tags=["me"])


def _user_docs_dir(username: str) -> Path:
    """
    Returns and ensures the user's isolated documents directory exists.
    """
    d = DATASET_DIR / "_users" / username / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_owned_schema(username: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Returns (schema_id, schema_record) for the schema owned by username with
    the latest confirmed_at timestamp (DB-backed), or None if none exist.
    """
    from src.ai.layer3_extraction.storage import get_latest_owned_schema_id, get_schema_record
    schema_id = get_latest_owned_schema_id(username)
    if not schema_id:
        return None
    record = get_schema_record(schema_id)
    if record is None:
        return None
    return schema_id, record


@router.post("/documents")
async def upload_my_documents(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Uploads PDFs to the authenticated user's isolated document directory.
    """
    from app.api.pipeline_routes import check_user_extract_rate_limit, validate_uploaded_pdf
    from src.ai.layer3_extraction import s3_documents as s3
    check_user_extract_rate_limit(user.username)
    saved: List[str] = []
    for f in files:
        safe_name, content = await validate_uploaded_pdf(f)
        content_type = f.content_type or "application/pdf"
        # Store per-user PDFs in S3 under the user's isolated prefix; fall back to disk.
        if s3.is_configured():
            s3.upload_user_document(user.username, safe_name, content, content_type=content_type)
        else:
            (_user_docs_dir(user.username) / safe_name).write_bytes(content)
        saved.append(safe_name)
    log_activity(user.username, "document_upload", {"files": saved, "count": len(saved)})
    return {"saved": saved, "count": len(saved)}


@router.get("/documents")
def list_my_documents(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Lists uploaded documents belonging to the calling user.
    """
    from src.ai.layer3_extraction import s3_documents as s3
    if s3.is_configured():
        docs = [
            {"name": d["name"], "size": d["size"], "modified": d["modified"], "source": "s3"}
            for d in sorted(s3.list_user_documents(user.username), key=lambda x: x["name"].lower())
        ]
        return {"documents": docs}
    user_dir = _user_docs_dir(user.username)
    pdfs = sorted([p for p in user_dir.glob("*.pdf") if p.is_file()])
    return {
        "documents": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "source": "disk",
            }
            for p in pdfs
        ],
    }


@router.delete("/documents/{filename}")
def delete_my_document(filename: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Delete one of the calling user's own documents from their isolated S3 prefix.
    """
    from app.api.pipeline_routes import _validate_doc_filename
    from src.ai.layer3_extraction import s3_documents as s3
    safe_name = _validate_doc_filename(filename)

    if s3.is_configured():
        if not s3.user_document_exists(user.username, safe_name):
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        if not s3.delete_user_document(user.username, safe_name):
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}' from storage.")
    else:
        target = _user_docs_dir(user.username) / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        try:
            target.unlink()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}': {exc}")

    # Clean up this owner's persisted DocumentIR for the same doc_id (== filename).
    # Best-effort: a failure here must not fail the document delete.
    ir_removed = False
    try:
        from src.ai.layer3_extraction.ir_store import delete_page_irs
        ir_removed = delete_page_irs(owner=user.username, doc_id=safe_name)
    except Exception:
        ir_removed = False

    log_activity(user.username, "document_delete", {"file": safe_name, "ir_removed": ir_removed})
    return {"deleted": safe_name, "ir_removed": ir_removed}


@router.get("/outputs")
def list_my_outputs(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    List processed-output (extracted) files that belong to the calling user.
    """
    from src.ai.layer3_extraction.storage import list_markdown_records_for_owner
    records = list_markdown_records_for_owner(user.username)
    outputs = [
        {
            "name": o["name"],
            "size": o["size"],
            "modified": o["modified"],
            "schema_ref": o.get("schema_ref") or ({"schema_id": o["schema_id"]} if o.get("schema_id") else None),
        }
        for o in records
    ]
    return {"outputs": outputs}


@router.get("/outputs/output/{name}")
def download_my_output(name: str, user: User = Depends(get_current_user)) -> Response:
    """
    Download the extracted-JSON output for one of the calling user's documents.
    """
    from src.ai.layer3_extraction.storage import list_markdown_records_for_owner, get_latest_extracted_data
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid output name.")
    stem = Path(name).stem.replace(".extracted", "")

    # Ownership gate: the requested output must be in the user's own output set.
    owned = list_markdown_records_for_owner(user.username)
    owned_stems = {Path(o["name"]).stem.replace(".extracted", "") for o in owned}
    owned_stems |= {Path(o.get("source_pdf", "")).stem for o in owned if o.get("source_pdf")}
    if stem not in owned_stems:
        raise HTTPException(status_code=403, detail="You can only download your own outputs.")

    data = get_latest_extracted_data(stem, owner=user.username)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No extracted output found for '{name}'.")
    payload = json.dumps(data, indent=2, default=str)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{stem}.extracted.json"'},
    )


@router.post("/pipeline/run")
async def run_my_pipeline(
    strategy: Optional[str] = Form(default=None),
    schema_id: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Zero-parameter trigger for regular users. Automatically uses the documents
    uploaded to the user's workspace and their most recently confirmed schema.
    """
    from app.api.pipeline_routes import check_user_extract_rate_limit
    from src.ai.layer3_extraction import s3_documents as s3
    check_user_extract_rate_limit(user.username)

    if s3.is_configured():
        # Download the user's S3 PDFs into local temp files for the worker to render.
        user_docs = s3.list_user_documents(user.username)
        if not user_docs:
            raise HTTPException(status_code=400, detail="Upload documents first")
        tmp_dir = Path(tempfile.mkdtemp(prefix="idp_userdocs_"))
        targets = []
        for d in user_docs:
            data = s3.download_bytes_by_key(d["key"])
            if not data:
                continue
            tp = tmp_dir / d["name"]
            tp.write_bytes(data)
            targets.append(tp)
        if not targets:
            raise HTTPException(status_code=400, detail="Upload documents first")
    else:
        user_dir = _user_docs_dir(user.username)
        targets = sorted([p for p in user_dir.glob("*.pdf") if p.is_file()])
        if not targets:
            raise HTTPException(status_code=400, detail="Upload documents first")

    # Resolve the target schema: use the user's chosen schema (must be owned by
    # them), otherwise fall back to their most recently confirmed schema.
    from src.ai.layer3_extraction.storage import get_schema_record
    schema_record = None
    schema_id_val = None
    if schema_id:
        rec = get_schema_record(schema_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Schema not found")
        if rec.get("owner") and rec.get("owner") != user.username:
            raise HTTPException(status_code=403, detail="You can only run your own schemas")
        schema_record, schema_id_val = rec, schema_id
    else:
        owned = _latest_owned_schema(user.username)
        if not owned:
            raise HTTPException(status_code=400, detail="Build and confirm a schema first")
        schema_id_val, schema_record = owned

    # Display-only stand-in path; the worker only reads schema_path.name, never the file.
    schema_path = Path(f"{schema_id_val}.json")

    # Normal users always run the recommended concurrent strategy (not user-selectable).
    selected_strategy = "graph_memory_concurrent"

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_id": schema_record.get("schema_id", schema_id_val),
        "schema_file": schema_path.name,
        "strategy": selected_strategy,
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
    log_activity(user.username, "pipeline_run", {"job_id": job_id, "targets": len(targets), "schema_id": schema_record.get("schema_id"), "strategy": selected_strategy})

    return {"job_id": job_id, "status": "queued", "targets": len(targets), "strategy": selected_strategy}


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


@router.get("/pipeline/jobs/{job_id}")
def get_my_pipeline_job(job_id: str, user: User = Depends(get_current_user)) -> Dict[str, Any]:
    job = _pipeline_jobs.get(job_id)
    if not job or job.get("owner") != user.username:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/pipeline/jobs/{job_id}/pdf", response_class=Response)
def download_my_pipeline_job_pdf(job_id: str, user: User = Depends(get_current_user)) -> Response:
    return download_job_pdf_report(job_id=job_id, user=user)


@router.get("/schemas")
def list_my_schemas(user: User = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """
    Returns list of confirmed schemas owned by the authenticated user.
    """
    from src.ai.layer3_extraction.storage import list_schema_records
    results: List[Dict[str, Any]] = []
    for r in list_schema_records(owner=user.username):
        results.append({
            "schema_id": r["schema_id"],
            "document_type": r["document_type"],
            "schema_name": r.get("schema_name"),
            "version": r.get("version"),
            "confirmed_at": r["confirmed_at"],
            "field_count": r["field_count"],
            "owner": r.get("owner") or user.username,
        })
    return results


@router.post("/pipeline/extract/from-output")
async def extract_my_existing_output(
    md_name: str = Form(...),
    strategy: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    User-facing endpoint to run Layer 3 extraction on an already-processed document
    belonging to the calling user in their isolated partition.
    """
    from app.api.pipeline_routes import check_user_extract_rate_limit, extract_from_existing_output
    check_user_extract_rate_limit(user.username)
    return await extract_from_existing_output(
        md_name=md_name,
        strategy=strategy,
        owner=user.username,
        user=user,
    )

