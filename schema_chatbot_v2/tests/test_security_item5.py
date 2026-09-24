from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.layer3_extraction.storage import (
    init_db,
    SessionLocal,
    DocumentGraphRecord,
)


@pytest.fixture
def client():
    return TestClient(app)


def get_token(client, username, password, role=Role.USER):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    return resp.json()["access_token"]


@pytest.fixture
def tokens(client):
    admin_token = get_token(client, "admin", "changeme", role=Role.ADMIN)
    user_a_token = get_token(client, "item5_user_a", "password123", role=Role.USER)
    user_b_token = get_token(client, "item5_user_b", "password123", role=Role.USER)
    return {
        "admin": admin_token,
        "user_a": user_a_token,
        "user_b": user_b_token,
    }


def _seed_graph(doc_id: str, owner: str | None):
    init_db()
    session = SessionLocal()
    try:
        session.query(DocumentGraphRecord).filter_by(doc_id=doc_id).delete()
        rec = DocumentGraphRecord(
            doc_id=doc_id,
            owner=owner,
            graph_json={
                "nodes": [{"id": "n1", "label": "Invoice", "value": "100", "source_pages": [1]}],
                "edges": [],
            },
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()


def test_query_bot_documents_anonymous_returns_401(client):
    resp = client.get("/api/query-bot/documents")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


def test_query_bot_ask_anonymous_returns_401(client):
    resp = client.post("/api/query-bot/ask", json={"question": "What is total?", "doc_id": "test.pdf"})
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


def test_query_bot_ask_user_a_cannot_query_user_b_doc(client, tokens):
    doc_id = "user_b_doc.pdf"
    _seed_graph(doc_id, owner="item5_user_b")

    resp = client.post(
        "/api/query-bot/ask",
        json={"question": "What is value?", "doc_id": doc_id},
        headers={"Authorization": f"Bearer {tokens['user_a']}"},
    )
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_query_bot_ask_normal_user_cannot_query_ownerless_doc(client, tokens):
    doc_id = "ownerless_doc.pdf"
    _seed_graph(doc_id, owner=None)

    resp = client.post(
        "/api/query-bot/ask",
        json={"question": "What is value?", "doc_id": doc_id},
        headers={"Authorization": f"Bearer {tokens['user_a']}"},
    )
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_query_bot_ask_admin_can_query_ownerless_doc(client, tokens):
    doc_id = "admin_ownerless_doc.pdf"
    _seed_graph(doc_id, owner=None)

    mock_result = MagicMock()
    mock_result.answer = "The value is 100"
    mock_result.sources = []
    mock_result.hops_traversed = 1
    mock_result.retrieved_nodes = ["n1"]
    mock_result.retrieved_edges = []

    with patch("src.ai.layer3_extraction.graph_agent.query_service.GraphQueryService.query", return_value=mock_result):
        resp = client.post(
            "/api/query-bot/ask",
            json={"question": "What is value?", "doc_id": doc_id},
            headers={"Authorization": f"Bearer {tokens['admin']}"},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        assert data["answer"] == "The value is 100"


def test_query_bot_ask_owner_can_query_own_doc(client, tokens):
    doc_id = "user_a_own_doc.pdf"
    _seed_graph(doc_id, owner="item5_user_a")

    mock_result = MagicMock()
    mock_result.answer = "The value is 100"
    mock_result.sources = []
    mock_result.hops_traversed = 1
    mock_result.retrieved_nodes = ["n1"]
    mock_result.retrieved_edges = []

    with patch("src.ai.layer3_extraction.graph_agent.query_service.GraphQueryService.query", return_value=mock_result):
        resp = client.post(
            "/api/query-bot/ask",
            json={"question": "What is value?", "doc_id": doc_id},
            headers={"Authorization": f"Bearer {tokens['user_a']}"},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["success"] is True


def test_query_bot_documents_normal_user_sees_only_own_docs(client, tokens):
    _seed_graph("doc_user_a.pdf", owner="item5_user_a")
    _seed_graph("doc_user_b.pdf", owner="item5_user_b")
    _seed_graph("doc_ownerless.pdf", owner=None)

    resp = client.get(
        "/api/query-bot/documents",
        headers={"Authorization": f"Bearer {tokens['user_a']}"},
    )
    assert resp.status_code == 200
    doc_ids = [d["doc_id"] for d in resp.json()["documents"]]
    assert "doc_user_a.pdf" in doc_ids
    assert "doc_user_b.pdf" not in doc_ids
    assert "doc_ownerless.pdf" not in doc_ids
