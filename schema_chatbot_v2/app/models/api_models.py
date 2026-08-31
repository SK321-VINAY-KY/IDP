from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # omit to start a new session
    message: Optional[str] = None  # omit when session_id is also omitted (first turn)


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str
    message: str
    state: str
    # "schema" is reserved-ish on BaseModel; the Python attribute is schema_
    # but it serializes as "schema" in the JSON body via the alias below.
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    completed: bool
    schema_id: Optional[str] = None
    errors: Optional[List[str]] = None


class UpdateSchemaRequest(BaseModel):
    document_type: Optional[str] = None
    fields: List[Dict[str, Any]] = Field(default_factory=list)
