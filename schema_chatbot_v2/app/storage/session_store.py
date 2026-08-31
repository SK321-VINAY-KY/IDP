"""
Session persistence, abstracted behind an interface so the in-memory
implementation used now can be swapped for DynamoDB/Redis later without
touching the conversation manager.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel, Field

from app.core.schema_state import SchemaState
from app.core.state_machine import ConversationState


class Session(BaseModel):
    session_id: str
    state: ConversationState = ConversationState.START
    schema_state: SchemaState = Field(default_factory=SchemaState)
    turn_count: int = 0
    completed: bool = False
    schema_id: Optional[str] = None
    owner: Optional[str] = None


class SessionStore(ABC):
    @abstractmethod
    def create(self) -> Session: ...

    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]: ...

    @abstractmethod
    def save(self, session: Session) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    """
    Development store. Structured so a future DynamoDBSessionStore /
    RedisSessionStore just needs to implement the same four methods -
    conversation_manager.py never touches storage internals directly.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(session_id=str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Process-wide singleton for the MVP. Swap for dependency injection if/when
# multiple store backends need to coexist (e.g. tests vs prod).
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store
