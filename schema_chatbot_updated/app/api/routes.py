from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.conversation_manager import MAX_DOCUMENT_SAMPLES, MIN_DOCUMENT_SAMPLES, ConversationManager, TurnResult
from app.llm.factory import get_llm_adapter
from app.models.api_models import ChatRequest, ChatResponse
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
def chat(req: ChatRequest, manager: ConversationManager = Depends(get_conversation_manager)) -> ChatResponse:
    if not req.session_id:
        result = manager.start_session()
        return _to_response(result)

    if not req.message:
        raise HTTPException(status_code=400, detail="message is required when session_id is provided")

    try:
        result = manager.handle_message(req.session_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(result)


@router.post("/schema/infer", response_model=ChatResponse, response_model_by_alias=True)
async def infer_schema(
    files: List[UploadFile] = File(..., description=f"{MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample PDFs of the same document type"),
    manager: ConversationManager = Depends(get_conversation_manager),
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
        result = manager.start_from_documents(samples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
