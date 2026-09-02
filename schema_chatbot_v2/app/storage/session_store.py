"""
Session persistence, abstracted behind an interface so the in-memory
implementation used now can be swapped for DynamoDB/Redis later without
touching the conversation manager.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from app.core.schema_state import SchemaState
from app.core.state_machine import ConversationState

logger = logging.getLogger(__name__)


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


class JSONFileSessionStore(SessionStore):
    """
    File-backed session storage implementation persisting sessions to a JSON file on disk.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()
        self._sessions: Dict[str, Session] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.file_path.exists():
            return
        try:
            content = self.file_path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("Session store file %s is empty; initializing empty store.", self.file_path)
                return
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    try:
                        s = Session(**item)
                        self._sessions[s.session_id] = s
                    except Exception as exc:
                        logger.warning("Skipping invalid session entry in %s: %s", self.file_path, exc)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt JSON in %s (%s); initializing empty store.", self.file_path, exc)
        except Exception as exc:
            logger.error("Error reading %s: %s", self.file_path, exc)

    def _save_to_disk(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                s.model_dump(mode="json") if hasattr(s, "model_dump") else json.loads(s.json())
                for s in self._sessions.values()
            ]
            temp_file = self.file_path.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(self.file_path)
        except Exception as exc:
            logger.error("Failed to write sessions to %s: %s", self.file_path, exc)

    def create(self) -> Session:
        session = Session(session_id=str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        self._save_to_disk()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        self._save_to_disk()

    def delete(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_to_disk()


# Process-wide singleton for the MVP. Swap for dependency injection if/when
# multiple store backends need to coexist (e.g. tests vs prod).
_store: Optional[SessionStore] = None


def reset_session_store() -> None:
    """Reset the singleton instance (primarily for tests)."""
    global _store
    _store = None


def get_session_store(file_path: Optional[str | Path] = None) -> SessionStore:
    global _store
    if _store is None:
        store_mode = os.getenv("SESSION_STORE", "memory").lower()
        if store_mode == "file" or store_mode == "json":
            if file_path is None:
                raw_path = os.getenv("SESSION_STORE_FILE")
                if raw_path:
                    path = Path(raw_path)
                else:
                    path = Path(__file__).resolve().parent.parent.parent / "data" / "sessions.json"
            else:
                path = Path(file_path)
            _store = JSONFileSessionStore(path)
        else:
            _store = InMemorySessionStore()
    return _store
