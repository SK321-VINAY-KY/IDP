import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import (
    OUTPUT_DIR,
)
from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.layer3_extraction.storage import init_db, SessionLocal, DocumentGraphRecord


@pytest.fixture
def client():
    return TestClient(app)


def get_auth_header(client, username, password, role=Role.USER):
    store = get_user_store()
    user = store.get_by_username(username)
    if not user:
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_admin_ambiguous_doc_requires_owner_or_job_id(client):
    admin_auth = get_auth_header(client, "admin_test_user", "adminpass", role=Role.ADMIN)
    auth_a = get_auth_header(client, "user_alice", "pass_a", role=Role.USER)
    auth_b = get_auth_header(client, "user_bob", "pass_b", role=Role.USER)

    doc_name = "ambiguous_invoice.pdf"
    stem = "ambiguous_invoice"

    # Setup directories
    dir_a = OUTPUT_DIR / "user_alice"
    dir_b = OUTPUT_DIR / "user_bob"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    # Write disk files for both users
    (dir_a / f"{stem}.extracted.json").write_text(
        json.dumps({"owner": "user_alice", "extracted_data": {"user": "alice", "total": 111}}),
        encoding="utf-8",
    )
    (dir_b / f"{stem}.extracted.json").write_text(
        json.dumps({"owner": "user_bob", "extracted_data": {"user": "bob", "total": 222}}),
        encoding="utf-8",
    )

    # Write DB records for both users
    init_db()
    sess = SessionLocal()
    try:
        sess.query(DocumentGraphRecord).filter(DocumentGraphRecord.doc_id == doc_name).delete()
        sess.commit()

        rec_a = DocumentGraphRecord(
            doc_id=doc_name,
            job_id="job_alice_101",
            owner="user_alice",
            node_count=2,
            edge_count=1,
            graph_json={"nodes": [{"id": "n1", "label": "AliceInvoice"}], "edges": []},
        )
        rec_b = DocumentGraphRecord(
            doc_id=doc_name,
            job_id="job_bob_202",
            owner="user_bob",
            node_count=2,
            edge_count=1,
            graph_json={"nodes": [{"id": "n2", "label": "BobInvoice"}], "edges": []},
        )
        sess.add(rec_a)
        sess.add(rec_b)
        sess.commit()
    finally:
        sess.close()

    try:
        with patch("src.ai.layer3_extraction.graph_agent.query_service.GraphQueryService.query") as mock_query:
            def fake_query(graph_mem, question):
                m = MagicMock()
                # If alice's node is in graph
                node_ids = list(graph_mem.nodes.keys())
                if "n1" in node_ids:
                    m.answer = "Answer for Alice: total 111"
                else:
                    m.answer = "Answer for Bob: total 222"
                m.sources = ["invoice"]
                m.hops_traversed = 1
                m.retrieved_nodes = ["node"]
                m.retrieved_edges = []
                return m

            mock_query.side_effect = fake_query

            # 1. Admin asks for ambiguous document WITHOUT owner or job_id
            # MUST fail with 400 Bad Request demanding owner or job_id
            resp_ambig = client.post(
                "/api/query-bot/ask",
                headers=admin_auth,
                json={"doc_id": doc_name, "question": "What is the total?"},
            )
            assert resp_ambig.status_code == 400, f"Expected 400 for ambiguous doc, got {resp_ambig.status_code}: {resp_ambig.text}"
            err_msg = resp_ambig.json().get("detail", "")
            assert "Ambiguous" in err_msg or "ambiguous" in err_msg
            assert "owner" in err_msg or "job_id" in err_msg

            # 2. Admin asks with explicit owner="user_alice" -> 200 OK, Alice's data
            resp_alice = client.post(
                "/api/query-bot/ask",
                headers=admin_auth,
                json={"doc_id": doc_name, "owner": "user_alice", "question": "What is the total?"},
            )
            assert resp_alice.status_code == 200, f"Expected 200 for admin with owner, got {resp_alice.status_code}: {resp_alice.text}"
            assert "Alice" in resp_alice.json()["answer"]

            # 3. Admin asks with explicit job_id="job_bob_202" -> 200 OK, Bob's data
            resp_bob = client.post(
                "/api/query-bot/ask",
                headers=admin_auth,
                json={"doc_id": doc_name, "job_id": "job_bob_202", "question": "What is the total?"},
            )
            assert resp_bob.status_code == 200, f"Expected 200 for admin with job_id, got {resp_bob.status_code}: {resp_bob.text}"
            assert "Bob" in resp_bob.json()["answer"]

            # 4. Non-admin user_alice queries without owner -> 200 OK, gets Alice's data
            resp_user = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"doc_id": doc_name, "question": "What is the total?"},
            )
            assert resp_user.status_code == 200
            assert "Alice" in resp_user.json()["answer"]

            # 5. Non-admin user_alice tries to query user_bob's partition -> 403 Forbidden
            resp_tamper = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"doc_id": doc_name, "owner": "user_bob", "question": "What is the total?"},
            )
            assert resp_tamper.status_code == 403
    finally:
        # Cleanup
        sess = SessionLocal()
        try:
            sess.query(DocumentGraphRecord).filter(DocumentGraphRecord.doc_id == doc_name).delete()
            sess.commit()
        finally:
            sess.close()
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)
