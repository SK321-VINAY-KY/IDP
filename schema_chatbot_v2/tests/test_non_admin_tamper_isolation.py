import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import OUTPUT_DIR
from app.main import app
from app.storage.user_store import Role, get_user_store
from src.ai.layer3_extraction.storage import Base, DocumentGraphRecord, SessionLocal, init_db

client = TestClient(app)


def get_auth_token(username, password="password123", role=Role.USER):
    store = get_user_store()
    user = store.get_by_username(username)
    if not user:
        store.create(username=username, password=password, role=role)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_non_admin_cannot_use_other_user_job_id_or_owner():
    token_a = get_auth_token("user_alpha_tamper")
    token_b = get_auth_token("user_beta_tamper")
    auth_a = {"Authorization": f"Bearer {token_a}"}
    auth_b = {"Authorization": f"Bearer {token_b}"}

    doc_b = "beta_secret_doc.pdf"
    job_b = "job_beta_secret_777"

    # Setup DB record belonging to user_beta_tamper
    init_db()
    sess = SessionLocal()
    try:
        sess.query(DocumentGraphRecord).filter(DocumentGraphRecord.doc_id == doc_b).delete()
        sess.commit()

        rec_b = DocumentGraphRecord(
            doc_id=doc_b,
            job_id=job_b,
            owner="user_beta_tamper",
            node_count=1,
            edge_count=0,
            graph_json={"nodes": [{"id": "n1", "label": "SecretBetaContent"}], "edges": []},
        )
        sess.add(rec_b)
        sess.commit()
    finally:
        sess.close()

    # Also setup disk files in user_beta_tamper partition
    beta_dir = OUTPUT_DIR / "user_beta_tamper"
    beta_dir.mkdir(parents=True, exist_ok=True)
    (beta_dir / "beta_secret_doc.extracted.json").write_text(
        json.dumps({"owner": "user_beta_tamper", "extracted_data": {"secret": "beta_data"}}),
        encoding="utf-8",
    )

    try:
        with patch("src.ai.layer3_extraction.graph_agent.query_service.GraphQueryService.query") as mock_query:
            m = MagicMock()
            m.answer = "Secret Answer for Beta"
            m.sources = ["beta"]
            m.hops_traversed = 1
            m.retrieved_nodes = ["n1"]
            m.retrieved_edges = []
            mock_query.return_value = m

            # Case 1: user_alpha specifies owner="user_beta_tamper" -> MUST return 403
            resp1 = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"doc_id": doc_b, "owner": "user_beta_tamper", "question": "What is the secret?"},
            )
            assert resp1.status_code == 403, f"Expected 403, got {resp1.status_code}: {resp1.text}"
            assert "Access denied" in resp1.json()["detail"]

            # Case 2: user_alpha specifies job_id=job_b (with doc_id=doc_b) -> MUST return 403
            resp2 = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"doc_id": doc_b, "job_id": job_b, "question": "What is the secret?"},
            )
            assert resp2.status_code == 403, f"Expected 403, got {resp2.status_code}: {resp2.text}"
            assert "Access denied" in resp2.json()["detail"]

            # Case 3: user_alpha specifies ONLY job_id=job_b (no doc_id) -> MUST return 403
            resp3 = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"job_id": job_b, "question": "What is the secret?"},
            )
            assert resp3.status_code == 403, f"Expected 403, got {resp3.status_code}: {resp3.text}"
            assert "Access denied" in resp3.json()["detail"]

            # Case 4: user_alpha specifies doc_id=doc_b without owner/job_id -> MUST return 403
            resp4 = client.post(
                "/api/query-bot/ask",
                headers=auth_a,
                json={"doc_id": doc_b, "question": "What is the secret?"},
            )
            assert resp4.status_code == 403, f"Expected 403, got {resp4.status_code}: {resp4.text}"
            assert "Access denied" in resp4.json()["detail"]

            # Case 5: user_beta (legitimate owner) accesses their own doc -> MUST return 200 OK
            resp5 = client.post(
                "/api/query-bot/ask",
                headers=auth_b,
                json={"doc_id": doc_b, "job_id": job_b, "question": "What is the secret?"},
            )
            assert resp5.status_code == 200, f"Expected 200, got {resp5.status_code}: {resp5.text}"
            assert resp5.json()["answer"] == "Secret Answer for Beta"
    finally:
        sess = SessionLocal()
        try:
            sess.query(DocumentGraphRecord).filter(DocumentGraphRecord.doc_id == doc_b).delete()
            sess.commit()
        finally:
            sess.close()
        shutil.rmtree(beta_dir, ignore_errors=True)
