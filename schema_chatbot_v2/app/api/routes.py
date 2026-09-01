from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.auth import get_current_user
from app.core.conversation_manager import MAX_DOCUMENT_SAMPLES, MIN_DOCUMENT_SAMPLES, ConversationManager, TurnResult
from app.llm.factory import get_llm_adapter
from app.models.api_models import ChatRequest, ChatResponse, UpdateSchemaRequest
from app.output.schema_renderer import render_json, render_pdf
from app.storage.session_store import get_session_store
from app.storage.user_store import User

SCHEMA_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "schema_registry"

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conversation_manager() -> ConversationManager:
    return ConversationManager(llm=get_llm_adapter(), store=get_session_store())


def _to_response(result: TurnResult) -> ChatResponse:
    return ChatResponse(
        session_id=result.session_id,
        message=result.message,
        state=result.state,
        schema=result.schema,
        completed=result.completed,
        schema_id=result.schema_id,
        errors=result.errors,
    )


@router.post("/chat", response_model=ChatResponse, response_model_by_alias=True)
async def chat(
    req: ChatRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    from starlette.concurrency import run_in_threadpool

    if not req.session_id:
        result = await run_in_threadpool(manager.start_session)
        session = manager.store.get(result.session_id)
        if session:
            session.owner = user.username
            manager.store.save(session)
        return _to_response(result)

    if not req.message:
        raise HTTPException(status_code=400, detail="message is required when session_id is provided")

    try:
        result = await run_in_threadpool(manager.handle_message, req.session_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(result)


@router.post("/schema/infer", response_model=ChatResponse, response_model_by_alias=True)
async def infer_schema(
    files: List[UploadFile] = File(..., description=f"{MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample PDFs of the same document type"),
    session_id: str | None = Form(
        default=None,
        description="Optional - feed samples into an existing session (e.g. one already mid-chat) instead of starting a new one",
    ),
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    if not (MIN_DOCUMENT_SAMPLES <= len(files) <= MAX_DOCUMENT_SAMPLES):
        raise HTTPException(
            status_code=400,
            detail=f"upload between {MIN_DOCUMENT_SAMPLES} and {MAX_DOCUMENT_SAMPLES} sample documents (got {len(files)})",
        )

    samples = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf") and f.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"'{f.filename}' doesn't look like a PDF")
        content = await f.read()
        samples.append(content)

        # Store input sample document in PostgreSQL
        try:
            from src.ai.layer3_extraction.storage import save_document
            save_document(filename=f.filename or f"sample_{len(samples)}.pdf", file_bytes=content, content_type=f.content_type or "application/pdf")
        except Exception:
            pass

    from starlette.concurrency import run_in_threadpool

    try:
        result = await run_in_threadpool(manager.start_from_documents, samples, session_id=session_id)
        if session_id is None:
            session = manager.store.get(result.session_id)
            if session:
                session.owner = user.username
                manager.store.save(session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(result)


@router.get("/session/{session_id}", response_model=ChatResponse, response_model_by_alias=True)
def get_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    session = manager.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return ChatResponse(
        session_id=session.session_id,
        message="",
        state=session.state.value,
        schema=session.schema_state.to_json_schema(),
        completed=session.completed,
        schema_id=session.schema_id,
    )


@router.post("/session/{session_id}/reset", response_model=ChatResponse, response_model_by_alias=True)
def reset_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    manager.store.delete(session_id)
    result = manager.start_session()
    session = manager.store.get(result.session_id)
    if session:
        session.owner = user.username
        manager.store.save(session)
    return _to_response(result)


@router.post("/session/{session_id}/schema", response_model=ChatResponse, response_model_by_alias=True)
def update_schema(
    session_id: str,
    req: UpdateSchemaRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    try:
        result = manager.update_schema_manually(session_id, req.document_type, req.fields)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _to_response(result)


# ---------------------------------------------------------------------------
# Schema download endpoints
# ---------------------------------------------------------------------------

def _load_schema_record(schema_id: str) -> dict:
    """
    Load a confirmed schema record from the schema_registry directory.
    Raises HTTPException 404 if the file does not exist.
    """
    path = SCHEMA_REGISTRY_DIR / f"{schema_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"schema '{schema_id}' not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not read schema file: {exc}")


@router.get(
    "/schema/{schema_id}/download/json",
    summary="Download confirmed schema as JSON",
    response_class=Response,
    responses={
        200: {
            "content": {"application/json": {}},
            "description": "JSON file containing the full confirmed schema record",
        },
        404: {"description": "Schema not found"},
    },
)
def download_schema_json(
    schema_id: str,
    user: User = Depends(get_current_user),
) -> Response:
    """
    Download the confirmed extraction schema as a formatted JSON file.

    The file contains the complete schema record — document_type, all fields
    with their types/constraints, schema_id, confirmed_at timestamp, and
    session metadata. Suitable for direct import into downstream pipeline
    configuration or manual inspection.
    """
    record = _load_schema_record(schema_id)
    doc_type = (record.get("document_type") or "schema").replace(" ", "_")
    filename = f"{doc_type}_{schema_id}.json"

    payload = render_json(record)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/schema/{schema_id}/download/pdf",
    summary="Download confirmed schema as PDF",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF report of the confirmed schema with a formatted fields table",
        },
        404: {"description": "Schema not found"},
        500: {"description": "PDF generation failed"},
    },
)
def download_schema_pdf(
    schema_id: str,
    user: User = Depends(get_current_user),
) -> Response:
    """
    Download the confirmed extraction schema as a human-readable PDF.

    The PDF contains a metadata header (document_type, schema_id,
    confirmed_at) and a formatted table of all fields — name, type, required
    flag, and any additional constraints (item_type, currency, pattern,
    description). Suitable for review, sign-off, or sharing with non-technical
    stakeholders.
    """
    record = _load_schema_record(schema_id)
    doc_type = (record.get("document_type") or "schema").replace(" ", "_")
    filename = f"{doc_type}_{schema_id}.pdf"

    try:
        payload = render_pdf(record)
    except Exception as exc:
        logger.exception("PDF generation failed for schema_id=%s", schema_id)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
