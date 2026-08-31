from __future__ import annotations

import io
import json
import logging
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "dataset_output"
SCHEMA_REGISTRY = ROOT / "schema_registry"

for d in (DATASET_DIR, OUTPUT_DIR, SCHEMA_REGISTRY):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "backend"))
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
def list_documents() -> Dict[str, Any]:
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
async def upload_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    saved: List[str] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf") and f.content_type != "application/pdf":
            continue
        content = await f.read()
        target = DATASET_DIR / Path(f.filename or f"doc_{uuid.uuid4().hex[:8]}.pdf").name
        target.write_bytes(content)
        saved.append(target.name)
    return {"saved": saved, "count": len(saved)}


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
def pipeline_status() -> Dict[str, Any]:
    routing_mode = getattr(a_settings, "routing_mode", None) if PIPELINE_AVAILABLE else None
    return {
        "available": PIPELINE_AVAILABLE,
        "routing_mode": routing_mode,
        "jobs": {
            k: {
                "job_id": k,
                "schema_id": v.get("schema_id"),
                "schema_file": v.get("schema_file"),
                "status": v["status"],
                "created_at": v["created_at"],
                "total": len(v.get("targets", [])),
                "completed": len(v.get("successes", [])) + len(v.get("failures", [])),
            }
            for k, v in _pipeline_jobs.items()
        },
    }


@router.get("/pipeline/jobs/{job_id}")
def get_pipeline_job(job_id: str) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return _pipeline_jobs[job_id]


@router.post("/pipeline/jobs/{job_id}/pause")
def pause_pipeline_job(job_id: str) -> Dict[str, Any]:
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
    return {"job_id": job_id, "status": "paused", "message": f"Job {job_id} paused."}


@router.post("/pipeline/jobs/{job_id}/resume")
def resume_pipeline_job(job_id: str) -> Dict[str, Any]:
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
    return {"job_id": job_id, "status": "running", "message": f"Job {job_id} resumed."}


@router.post("/pipeline/jobs/{job_id}/kill")
def kill_pipeline_job(job_id: str) -> Dict[str, Any]:
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
    return {"job_id": job_id, "status": "killed", "message": f"Job {job_id} killed."}


@router.post("/pipeline/run")
async def run_pipeline(
    schema_id: str = Form(...),
    documents: Optional[str] = Form(default=None),
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

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, targets, schema_path, schema_record),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "targets": len(targets)}


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

            # ================= Automatically Continue to Layer 3 Extraction =================
            extracted_data = None
            db_id = None
            db_error = None
            extract_elapsed = 0.0

            try:
                from src.adapters.llm.extraction_factory import get_extraction_client
                from src.ai.layer3_extraction.extractor import extract_by_page_scan
                from src.ai.layer3_extraction.schema_validation import extract_with_retry
                from src.ai.layer3_extraction.storage import init_db, save_extraction_run
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
                    pages_for_layer3 = [{"markdown": r[0].markdown, "page_number": r[0].page_number} for r in results]

                    extracted_result = extract_with_retry(
                        lambda: extract_by_page_scan(pages_for_layer3, dynamic_schema, extraction_llm),
                        dynamic_schema,
                    )
                    extract_elapsed = round(time.monotonic() - t_ext0, 2)
                    extracted_data = extracted_result.model_dump()

                    # Write dataset_output/<stem>.extracted.json
                    json_out_path = OUTPUT_DIR / f"{pdf.stem}.extracted.json"
                    json_out_path.write_text(
                        json.dumps({
                            "source_pdf": pdf.name,
                            "schema_used": schema_id,
                            "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                            "processing_time_seconds": extract_elapsed,
                            "extracted_data": extracted_data,
                        }, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    # Persist to PostgreSQL
                    try:
                        init_db()
                        db_id = save_extraction_run(
                            doc_id=pdf.name,
                            page_count=len(outputs),
                            schema_name=schema_id,
                            result_json=extracted_data,
                            llm_provider=a_settings.extraction_backend,
                            model_name=a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name,
                            processing_time_seconds=extract_elapsed,
                            page_outputs=outputs,
                        )
                    except Exception as db_exc:
                        db_error = str(db_exc)
                        logger.warning("pipeline.extract.db_save_failed", pdf=pdf.name, error=str(db_exc))
            except Exception as ext_exc:
                logger.error("pipeline.layer3_extraction_failed", pdf=pdf.name, error=str(ext_exc))

            job["successes"].append({
                "pdf": pdf.name,
                "md": str(md_path.name),
                "schema_ref": str(ref_path.name),
                "avg_conf": round(avg_conf, 3),
                "chars": total_chars,
                "pages": len(outputs),
                "elapsed_s": round(elapsed, 2),
                "extracted_json": f"{pdf.stem}.extracted.json" if extracted_data else None,
                "extracted_data": extracted_data,
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


# ========================= Layer 3 Extraction =========================


def _parse_pages_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Extract individual pages from an already-generated pipeline markdown file."""
    marker = re.compile(r"<!--\s*PAGE\s+(\d+)[^>]*-->", re.I)
    matches = list(marker.finditer(md_text))
    if not matches:
        return [{"markdown": md_text.strip(), "page_number": 1}]
    pages = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end]
        content = re.sub(r"<!--\s*/PAGE\s+\d+\s*-->", "", content).strip()
        pages.append({"markdown": content, "page_number": int(m.group(1))})
    return pages


@router.get("/pipeline/outputs")
def list_pipeline_outputs() -> Dict[str, Any]:
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

        # Check schema registry for field count if available
        field_count = None
        if schema_id:
            schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
            if schema_file.exists():
                try:
                    s_data = json.loads(schema_file.read_text(encoding="utf-8"))
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
) -> Dict[str, Any]:
    """
    Run Layer 3 extraction directly on an ALREADY-PROCESSED pipeline document.
    Automatically retrieves the converted Markdown and the schema bound to it during Run Pipeline.
    Saves the extracted structured JSON to dataset_output/<stem>.extracted.json and stores in PostgreSQL.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    md_path = OUTPUT_DIR / md_name
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{md_name}' not found in dataset_output.")

    ref_path = md_path.with_suffix(".schema_ref.json")
    if not ref_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No schema reference found for '{md_name}'. Make sure the pipeline was run with a target schema.",
        )

    ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
    schema_id = ref_data.get("schema_id")
    if not schema_id:
        raise HTTPException(status_code=400, detail=f"No schema_id recorded in {ref_path.name}")

    schema_file = SCHEMA_REGISTRY / f"{schema_id}.json"
    if not schema_file.exists():
        # Fallback to search
        cand = list(SCHEMA_REGISTRY.glob(f"*{schema_id}*.json"))
        if cand:
            schema_file = cand[0]
        else:
            raise HTTPException(status_code=404, detail=f"Schema file '{schema_id}.json' not found in schema_registry.")

    schema_json = json.loads(schema_file.read_text(encoding="utf-8"))
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
    pages = _parse_pages_from_markdown(md_content)

    def _execute():
        t_start = time.time()
        extraction_llm = get_extraction_client()
        extracted_result = extract_with_retry(
            lambda: extract_by_page_scan(pages, dynamic_schema, extraction_llm),
            dynamic_schema,
        )
        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        # Save to dataset_output/
        stem = md_path.stem
        json_out_path = OUTPUT_DIR / f"{stem}.extracted.json"
        json_out_path.write_text(json.dumps({
            "source_doc": md_name,
            "source_pdf": ref_data.get("source_pdf", f"{stem}.pdf"),
            "schema_used": schema_id,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
        }, indent=2) + "\n", encoding="utf-8")

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

            db_id = save_extraction_run(
                doc_id=ref_data.get("source_pdf", md_name),
                page_count=len(pages),
                schema_name=schema_id,
                result_json=result_dict,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name,
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
            "data": result_dict,
            "meta": {
                "pages_processed": len(pages),
                "processing_time_seconds": elapsed,
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
async def extract_document(
    file: UploadFile = File(...),
    schema_id: Optional[str] = Form(default=None),
    raw_schema: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    """
    End-to-End Extraction (single PDF upload):
      1. Receives PDF file + target schema.
      2. Runs Layer 1 (inspection & routing) + Layer 2 (conversion & OCR).
      3. Saves converted markdown to dataset_output/<stem>.md.
      4. Runs Layer 3 (Sarvam/Ollama page-by-page field extraction with scratchpad & early stopping).
      5. Saves output JSON to dataset_output/<stem>.extracted.json.
      6. Persists extraction run into PostgreSQL database.
      7. Returns structured JSON result and metadata.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.adapters.llm.factory import get_llm_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan
    from src.ai.layer3_extraction.page_loader import load_pages_with_confidence
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Resolve Target Schema
    fields_in: List[SchemaFieldIn] = []
    schema_name = "custom_schema"
    if schema_id:
        schema_path = SCHEMA_REGISTRY / f"{schema_id}.json"
        if not schema_path.exists():
            cand = list(SCHEMA_REGISTRY.glob(f"*{schema_id}*.json"))
            if cand:
                schema_path = cand[0]
            else:
                raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found in registry.")
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
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

    # 2. Save uploaded PDF temporarily
    pdf_bytes = await file.read()
    temp_pdf = DATASET_DIR / file.filename
    temp_pdf.write_bytes(pdf_bytes)

    def _execute():
        t_start = time.time()

        pages = _build_pages_for_pdf(temp_pdf)
        try:
            llm_client = get_llm_client()
        except Exception:
            llm_client = MockLLMClient()

        results = process_document(
            pages=pages,
            llm_client=llm_client,
            document_name=file.filename,
            document_id=f"extract-{uuid.uuid4().hex[:6]}",
            write_output=True,
            output_dir=str(OUTPUT_DIR),
            overwrite=True,
        )

        page_outputs = [r[0] for r in results]
        pages_md = load_pages_with_confidence(page_outputs)

        extraction_llm = get_extraction_client()
        extracted_result = extract_with_retry(
            lambda: extract_by_page_scan(pages_md, dynamic_schema, extraction_llm),
            dynamic_schema,
        )

        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        stem = Path(file.filename).stem
        json_out_path = OUTPUT_DIR / f"{stem}.extracted.json"
        json_out_path.write_text(json.dumps({
            "source_pdf": file.filename,
            "schema_used": schema_name,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
        }, indent=2) + "\n", encoding="utf-8")

        db_id = None
        db_error = None
        try:
            init_db()
            db_id = save_extraction_run(
                doc_id=file.filename,
                page_count=len(page_outputs),
                schema_name=schema_name,
                result_json=result_dict,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name,
                processing_time_seconds=elapsed,
                page_outputs=page_outputs,
            )
        except Exception as exc:
            db_error = str(exc)
            logger.warning("pipeline.extract.db_save_failed", error=str(exc))

        return {
            "success": True,
            "doc_id": file.filename,
            "schema": schema_name,
            "data": result_dict,
            "meta": {
                "pages_processed": len(page_outputs),
                "processing_time_seconds": elapsed,
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


