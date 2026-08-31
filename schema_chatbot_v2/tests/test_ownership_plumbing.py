import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.conversation_manager import ConversationManager
from app.llm.mock_adapter import MockLLMAdapter
from app.main import app
from app.storage.session_store import InMemorySessionStore, Session
from app.storage.user_store import Role, get_user_store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_session_model_has_owner():
    session = Session(session_id="s1", owner="alice")
    assert session.owner == "alice"


def test_persist_confirmed_schema_includes_owner(tmp_path, monkeypatch):
    from app.core import conversation_manager as cm
    monkeypatch.setattr(cm, "SCHEMA_REGISTRY_DIR", tmp_path)

    mgr = ConversationManager(llm=MockLLMAdapter(), store=InMemorySessionStore())
    session = mgr.store.create()
    session.schema_id = "schema_test123"
    session.owner = "testuser"
    session.schema_state.set_document_type("receipt")

    saved_path = mgr._persist_confirmed_schema(session)
    assert saved_path.exists()
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["owner"] == "testuser"
    assert data["schema_id"] == "schema_test123"


def test_chat_unauthenticated_fails(client):
    resp = client.post("/chat", json={"session_id": None, "message": None})
    assert resp.status_code == 401


def test_chat_authenticated_sets_owner(client, auth_headers):
    resp = client.post("/chat", json={"session_id": None, "message": None}, headers=auth_headers)
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # Verify session in store has owner == admin
    from app.storage.session_store import get_session_store
    session = get_session_store().get(sid)
    assert session is not None
    assert session.owner == "admin"
