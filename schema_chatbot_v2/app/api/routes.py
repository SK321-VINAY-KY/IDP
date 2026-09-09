from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.activity_log import log_activity
from app.core.auth import get_current_user
from app.core.conversation_manager import MAX_DOCUMENT_SAMPLES, MIN_DOCUMENT_SAMPLES, ConversationManager, TurnResult
from app.core.schema_state import _normalize_field_name, normalize_type, SUPPORTED_TYPES, SchemaState
from app.core.validator import validate_schema
from app.llm.factory import get_llm_adapter
from app.models.api_models import ChatRequest, ChatResponse, UpdateSchemaRequest
from app.output.schema_renderer import render_json, render_pdf
from app.storage.session_store import get_session_store
from app.storage.user_store import User

SCHEMA_REGISTRY_DIR = Path(__file__).resolve().parents[3] / "schema_registry"
if not SCHEMA_REGISTRY_DIR.exists():
    # Fallback if running from a different root
    alt = Path(__file__).resolve().parents[2] / "schema_registry"
    if alt.exists():
        SCHEMA_REGISTRY_DIR = alt

logger = logging.getLogger(__name__)
router = APIRouter()


class CustomFieldIn(BaseModel):
    name: str
    type: str = "string"
    required: bool = True
    description: Optional[str] = ""
    item_type: Optional[str] = None
    pattern: Optional[str] = None
    currency: Optional[str] = None


class CreateCustomSchemaRequest(BaseModel):
    document_type: str
    fields: List[CustomFieldIn] = Field(default_factory=list)


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


@router.post(
    "/schema/custom",
    summary="Create and persist a custom schema directly from the interactive builder",
)
def create_custom_schema(
    req: CreateCustomSchemaRequest,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    doc_type = (req.document_type or "").strip()
    if not doc_type:
        raise HTTPException(status_code=400, detail="document_type cannot be empty or whitespace-only")

    if not req.fields:
        raise HTTPException(status_code=400, detail="schema must contain at least one field")

    seen_normalized: set[str] = set()
    schema_state = SchemaState()
    schema_state.set_document_type(doc_type)

    for idx, f in enumerate(req.fields):
        raw_name = (f.name or "").strip()
        if not raw_name:
            raise HTTPException(status_code=400, detail=f"field at index {idx} has an empty or blank name")

        norm_name = _normalize_field_name(raw_name)
        if not norm_name:
            raise HTTPException(status_code=400, detail=f"field name '{f.name}' normalizes to empty string")

        if norm_name in seen_normalized:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate field name detected: '{f.name}' (normalized: '{norm_name}')",
            )
        seen_normalized.add(norm_name)

        raw_type = (f.type or "string").strip()
        item_type = f.item_type
        if "[" in raw_type and "]" in raw_type:
            item_type = raw_type[raw_type.find("[") + 1 : raw_type.find("]")].strip()
            raw_type = "array"

        norm_type, _ = normalize_type(raw_type)
        if not norm_type or norm_type not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported type '{f.type}' for field '{f.name}'. Supported types: {sorted(SUPPORTED_TYPES)}",
            )

        if norm_type == "array" and not item_type:
            item_type = "string"

        schema_state.add_field(
            norm_name,
            type=norm_type,
            required=bool(f.required),
            description=(f.description or "").strip(),
            item_type=item_type,
            pattern=f.pattern,
            currency=f.currency,
        )

    # Authoritative validation
    errors = validate_schema(schema_state)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Generate unique schema ID
    schema_id = f"schema_{uuid.uuid4().hex[:12]}"
    while (SCHEMA_REGISTRY_DIR / f"{schema_id}.json").exists():
        schema_id = f"schema_{uuid.uuid4().hex[:12]}"

    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    record = {
        "schema_id": schema_id,
        "document_type": schema_state.document_type,
        "confirmed_at": now_iso,
        "session_id": None,
        "turn_count_at_confirm": 1,
        "schema": schema_state.to_json_schema(),
        "sample_documents": [],
        "owner": user.username if user else "admin",
    }

    SCHEMA_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = SCHEMA_REGISTRY_DIR / f"{schema_id}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # Persist to PostgreSQL schemas table
    try:
        from src.ai.layer3_extraction.storage import save_schema_record
        save_schema_record(
            schema_id=schema_id,
            document_type=schema_state.document_type or "document",
            schema_json=record,
            session_id=None,
            sample_documents=[],
        )
    except Exception as exc:
        logger.warning("storage.schema_save_failed for schema_id=%s: %s", schema_id, exc)

    # Audit log
    try:
        log_activity(
            user.username if user else "admin",
            "schema_confirmed",
            {"schema_id": schema_id, "document_type": schema_state.document_type},
        )
    except Exception:
        pass

    return {
        "status": "success",
        "schema_id": schema_id,
        "document_type": schema_state.document_type,
        "confirmed_at": now_iso,
        "schema": schema_state.to_json_schema(),
        "record": record,
    }


# ---------------------------------------------------------------------------
# Schema download endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/schema/my",
    summary="List schemas owned by the current user",
    response_model=list,
)
def list_my_schemas(
    user: User = Depends(get_current_user),
) -> list:
    """
    Returns a list of confirmed schema summaries owned by the authenticated user.
    ADMIN sees all schemas. USER sees only their own.

    Each item contains: schema_id, document_type, confirmed_at, field_count.
    """
    from app.storage.user_store import Role

    if not SCHEMA_REGISTRY_DIR.exists():
        return []

    results = []
    for path in sorted(SCHEMA_REGISTRY_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        owner = record.get("owner")
        if user.role != Role.ADMIN and owner and owner != user.username:
            continue

        results.append({
            "schema_id":    record.get("schema_id"),
            "document_type": record.get("document_type"),
            "confirmed_at": record.get("confirmed_at"),
            "field_count":  len(record.get("schema", {}).get("fields", [])),
            "owner":        owner,
        })

    return results

def _load_schema_record(schema_id: str, requesting_user: User) -> dict:
    """
    Load a confirmed schema record from the schema_registry directory.

    Access rules:
      - ADMIN: can download any schema.
      - USER:  can only download schemas they own (record.owner == username).
                If the record has no owner set (legacy records), access is
                granted to avoid breaking existing data.

    Raises 404 if the file does not exist, 403 if the user does not own it.
    """
    from app.storage.user_store import Role

    path = SCHEMA_REGISTRY_DIR / f"{schema_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"schema '{schema_id}' not found")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not read schema file: {exc}")

    # Ownership check for non-admin users
    if requesting_user.role != Role.ADMIN:
        owner = record.get("owner")
        if owner and owner != requesting_user.username:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to download this schema.",
            )

    return record


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
@router.get(
    "/schema/{schema_id}/json",
    summary="Download confirmed schema as JSON (alias)",
    response_class=Response,
    include_in_schema=False,
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
    record = _load_schema_record(schema_id, requesting_user=user)
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
@router.get(
    "/schema/{schema_id}/pdf",
    summary="Download confirmed schema as PDF (alias)",
    response_class=Response,
    include_in_schema=False,
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
    record = _load_schema_record(schema_id, requesting_user=user)
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
