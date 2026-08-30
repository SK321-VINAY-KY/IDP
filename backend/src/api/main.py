"""
File: main.py
Purpose: HTTP API for the frontend's user flow — upload a PDF + a target
         schema, run it through Layer 1 (routing) -> Layer 2 (conversion) ->
         Layer 3 (extraction), return structured JSON. Admin endpoints are
         out of scope for this pilot; only the extraction path is wired up.
Owner: api@idp-pilot
Created: 2026-08-26
"""
import json
import os
import shutil
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.llm.extraction_factory import get_extraction_client
from src.adapters.llm.ollama_client import OllamaClient
from src.ai.layer1_routing.pipeline import process_document
from src.ai.layer3_extraction.extractor import extract_by_page_scan
from src.ai.layer3_extraction.page_loader import load_pages_with_confidence
from src.ai.layer3_extraction.schema_validation import extract_with_retry
from src.ai.layer3_extraction.storage import init_db, save_processing_result
from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="IDP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.on_event("startup")
def _on_startup() -> None:
    # Creates the processing_results table if it doesn't exist yet. If Postgres
    # isn't reachable, the app still starts — extraction works, only the save
    # step at the end of each request will fail (and be reported, not silent).
    try:
        init_db()
    except Exception as exc:
        logger.error("api.startup.db_unavailable", error=str(exc))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...), target_schema: str = Form(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        raw_fields = json.loads(target_schema)
        fields = [SchemaFieldIn(**f) for f in raw_fields]
        dynamic_schema = build_dynamic_schema(fields)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target schema: {exc}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="idp_upload_")
    pdf_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        start = time.time()
        page_outputs = _run_layer1_2(pdf_path)
        pages_md = load_pages_with_confidence(page_outputs)

        extraction_llm = get_extraction_client()
        result = extract_with_retry(
            lambda: extract_by_page_scan(pages_md, dynamic_schema, extraction_llm),
            dynamic_schema,
        )
        elapsed = round(time.time() - start, 2)

    except HTTPException:
        raise
    except Exception as exc:  # pipeline/model errors surface as a clean 502
        logger.error("api.extract.failed", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    strategy_used = "page_scan"
    schema_name = ",".join(sorted(dynamic_schema.model_fields.keys()))[:250]

    db_result_id, db_error = _save_to_db(
        doc_id=file.filename,
        page_outputs=page_outputs,
        schema_name=schema_name,
        result_json=result.model_dump(),
        strategy_used=strategy_used,
        elapsed=elapsed,
    )

    return {
        "success": True,
        "data": result.model_dump(),
        "meta": {
            "strategy_used": strategy_used,
            "processing_time_seconds": elapsed,
            "llm_provider": settings.extraction_backend,
            "saved_to_db": db_result_id is not None,
            "db_result_id": db_result_id,
        },
    }


def _run_layer1_2(pdf_path: str) -> list:
    """Opens the PDF, runs it through Layer 1/2, and always closes the doc."""
    from src.api.document_processor import build_pages_for_document

    doc, pages_data = build_pages_for_document(pdf_path)
    try:
        vision_llm = OllamaClient()
        return process_document(pages_data, vision_llm)
    finally:
        doc.close()


def _save_to_db(
    doc_id: str,
    page_outputs: list,
    schema_name: str,
    result_json: dict,
    strategy_used: str,
    elapsed: float,
) -> tuple[int | None, str | None]:
    """
    Persists one row per extraction run. Failure here (e.g. Postgres is down)
    never breaks the response the frontend already has — it's reported back
    in `meta.db_error` instead of raised.
    """
    confidences = [p.confidence for p in page_outputs]
    primary_scripts = sorted({s for p in page_outputs if (s := getattr(p, "primary_script", None))})
    complexity_scores = [c for p in page_outputs if (c := getattr(p, "complexity_score", None)) is not None]
    has_images_flags = [getattr(p, "has_images", None) for p in page_outputs]
    try:
        result_id = save_processing_result(
            doc_id=doc_id,
            page_count=len(page_outputs),
            routes_used=sorted({p.engine_used for p in page_outputs}),
            engines_used=sorted({p.engine_used for p in page_outputs}),
            primary_scripts=primary_scripts or None,
            avg_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
            min_confidence=round(min(confidences), 4) if confidences else None,
            low_confidence_page_count=sum(1 for p in page_outputs if p.low_confidence),
            escalated_page_count=sum(1 for p in page_outputs if p.escalated),
            total_escalation_attempts=sum(p.escalation_attempts for p in page_outputs),
            max_complexity_score=max(complexity_scores, default=None),
            has_images=any(has_images_flags) if any(f is not None for f in has_images_flags) else None,
            schema_name=schema_name,
            result_json=result_json,
            llm_provider=settings.extraction_backend,
            model_name=(
                settings.extraction_model_name
                if settings.extraction_backend == "ollama"
                else settings.sarvam_model_name
            ),
            strategy_used=strategy_used,
            processing_time_seconds=elapsed,
        )
        return result_id, None
    except Exception as exc:
        logger.error("api.extract.db_save_failed", error=str(exc), error_type=type(exc).__name__)
        return None, str(exc)