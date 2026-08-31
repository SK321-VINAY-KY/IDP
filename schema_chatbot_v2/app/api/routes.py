from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.auth import get_current_user
from app.core.conversation_manager import MAX_DOCUMENT_SAMPLES, MIN_DOCUMENT_SAMPLES, ConversationManager, TurnResult
from app.llm.factory import get_llm_adapter
from app.models.api_models import ChatRequest, ChatResponse, UpdateSchemaRequest
from app.models.auth_models import User
from app.storage.audit_log import get_audit_logger
from app.storage.session_store import get_session_store

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
def chat(
    req: ChatRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    if not req.session_id:
        result = manager.start_session()
        return _to_response(result)

    if not req.message:
        raise HTTPException(status_code=400, detail="message is required when session_id is provided")

    try:
        result = manager.handle_message(req.session_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    if result.completed and result.schema_id:
        audit = get_audit_logger()
        audit.log_activity(
            username=current_user.username,
            role=current_user.role.value,
            action="SCHEMA_CONFIRM",
            details={
                "schema_id": result.schema_id,
                "document_type": (result.schema or {}).get("document_type"),
                "fields_count": len((result.schema or {}).get("fields", [])),
            },
        )
        # Update schema json file to record creator
        try:
            from app.core.conversation_manager import SCHEMA_REGISTRY_DIR
            schema_file = SCHEMA_REGISTRY_DIR / f"{result.schema_id}.json"
            if schema_file.exists():
                data = json.loads(schema_file.read_text(encoding="utf-8"))
                data["created_by"] = current_user.username
                data["created_by_role"] = current_user.role.value
                schema_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    return _to_response(result)


@router.post("/schema/infer", response_model=ChatResponse, response_model_by_alias=True)
async def infer_schema(
    files: List[UploadFile] = File(..., description=f"{MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample PDFs of the same document type"),
    session_id: str | None = Form(
        default=None,
        description="Optional - feed samples into an existing session (e.g. one already mid-chat) instead of starting a new one",
    ),
    manager: ConversationManager = Depends(get_conversation_manager),
    current_user: User = Depends(get_current_user),
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
        samples.append(await f.read())

    try:
        result = manager.start_from_documents(samples, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="SCHEMA_INFER",
        details={
            "samples_count": len(files),
            "document_type": (result.schema or {}).get("document_type"),
            "fields_inferred": len((result.schema or {}).get("fields", [])),
        },
    )

    return _to_response(result)


@router.get("/session/{session_id}", response_model=ChatResponse, response_model_by_alias=True)
def get_session(session_id: str, manager: ConversationManager = Depends(get_conversation_manager)) -> ChatResponse:
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
def reset_session(session_id: str, manager: ConversationManager = Depends(get_conversation_manager)) -> ChatResponse:
    manager.store.delete(session_id)
    result = manager.start_session()
    return _to_response(result)


@router.post("/session/{session_id}/schema", response_model=ChatResponse, response_model_by_alias=True)
def update_schema(
    session_id: str,
    req: UpdateSchemaRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    try:
        result = manager.update_schema_manually(session_id, req.document_type, req.fields)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    audit = get_audit_logger()
    audit.log_activity(
        username=current_user.username,
        role=current_user.role.value,
        action="SCHEMA_EDIT",
        details={
            "document_type": req.document_type,
            "fields_count": len(req.fields),
        },
    )

    return _to_response(result)
