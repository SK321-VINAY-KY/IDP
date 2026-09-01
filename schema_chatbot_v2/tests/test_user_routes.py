import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.pipeline_routes import _job_controls, _pipeline_jobs, JobControl, SCHEMA_REGISTRY
from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture(autouse=True)
def cleanup():
    _pipeline_jobs.clear()
    _job_controls.clear()
    yield
    _pipeline_jobs.clear()
    _job_controls.clear()


@pytest.fixture
def client():
    return TestClient(app)


def get_auth_header(client, username, password):
    store = get_user_store()
    if not store.get_by_username(username):
        store.create(username=username, password=password, role=Role.USER)
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_user_document_isolation(client):
    auth_a = get_auth_header(client, "user_a", "pass_a")
    auth_b = get_auth_header(client, "user_b", "pass_b")

    # User A uploads a doc
    pdf_content = b"%PDF-1.4 test document user a"
    files = [("files", ("doc_a.pdf", io.BytesIO(pdf_content), "application/pdf"))]
    resp = client.post("/me/documents", files=files, headers=auth_a)
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    # User A lists docs
    resp_a = client.get("/me/documents", headers=auth_a)
    assert resp_a.status_code == 200
    doc_names_a = [d["name"] for d in resp_a.json()["documents"]]
    assert "doc_a.pdf" in doc_names_a

    # User B lists docs (should NOT see doc_a.pdf)
    resp_b = client.get("/me/documents", headers=auth_b)
    assert resp_b.status_code == 200
    doc_names_b = [d["name"] for d in resp_b.json()["documents"]]
    assert "doc_a.pdf" not in doc_names_b


def test_run_my_pipeline_no_documents(client):
    auth_c = get_auth_header(client, "user_c", "pass_c")
    # No documents uploaded yet
    resp = client.post("/me/pipeline/run", headers=auth_c)
    assert resp.status_code == 400
    assert "Upload documents first" in resp.json()["detail"]


def test_run_my_pipeline_no_schema(client):
    auth_d = get_auth_header(client, "user_d", "pass_d")
    # Upload doc but have no schema
    pdf_content = b"%PDF-1.4 sample user d"
    files = [("files", ("sample_d.pdf", io.BytesIO(pdf_content), "application/pdf"))]
    client.post("/me/documents", files=files, headers=auth_d)

    resp = client.post("/me/pipeline/run", headers=auth_d)
    assert resp.status_code == 400
    assert "Build and confirm a schema first" in resp.json()["detail"]


def test_run_my_pipeline_and_status(client, monkeypatch):
    auth_e = get_auth_header(client, "user_e", "pass_e")

    # 1. Upload doc
    pdf_content = b"%PDF-1.4 sample user e"
    files = [("files", ("invoice_e.pdf", io.BytesIO(pdf_content), "application/pdf"))]
    client.post("/me/documents", files=files, headers=auth_e)

    # 2. Plant a confirmed schema owned by user_e in SCHEMA_REGISTRY
    schema_id = "schema_e_test_1"
    schema_record = {
        "schema_id": schema_id,
        "document_type": "invoice",
        "confirmed_at": "2026-08-31T12:00:00+00:00",
        "owner": "user_e",
        "schema": {"document_type": "invoice", "fields": [{"name": "total", "type": "number"}]},
    }
    target_path = SCHEMA_REGISTRY / f"{schema_id}.json"
    target_path.write_text(json.dumps(schema_record), encoding="utf-8")

    # Mock _run_pipeline_job so background thread doesn't do heavy OCR
    def dummy_run(job_id, targets, spath, srec):
        _pipeline_jobs[job_id]["status"] = "running"
        _pipeline_jobs[job_id]["current_document"] = "invoice_e.pdf"
        _pipeline_jobs[job_id]["successes"].append({"pdf": "invoice_e.pdf"})
        _pipeline_jobs[job_id]["status"] = "completed"

    from app.api import user_routes
    monkeypatch.setattr(user_routes, "_run_pipeline_job", dummy_run)

    # 3. Run pipeline
    resp = client.post("/me/pipeline/run", headers=auth_e)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    job_id = data["job_id"]
    assert _pipeline_jobs[job_id]["owner"] == "user_e"

    # 4. Check status via /me/pipeline/status
    st_resp = client.get("/me/pipeline/status", headers=auth_e)
    assert st_resp.status_code == 200
    jobs = st_resp.json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)
    user_e_job = next(j for j in jobs if j["job_id"] == job_id)
    assert "invoice_e.pdf" in user_e_job["documents"]
    assert "schema_id" not in user_e_job  # do not leak schema_id to regular user status response


def test_user_job_isolation_and_controls(client):
    auth_x = get_auth_header(client, "user_x", "pass_x")
    auth_y = get_auth_header(client, "user_y", "pass_y")

    # Create a job owned by user_x
    job_id = "job_x_100"
    ctrl = JobControl()
    _job_controls[job_id] = ctrl
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "owner": "user_x",
        "targets": ["doc.pdf"],
        "successes": [],
        "failures": [],
    }

    # User Y tries to view status - should NOT see user_x's job
    resp_y = client.get("/me/pipeline/status", headers=auth_y)
    assert resp_y.status_code == 200
    assert not any(j["job_id"] == job_id for j in resp_y.json()["jobs"])

    # User Y tries to pause/resume/kill user_x's job - should return 404 (not 403, don't leak existence)
    assert client.post(f"/me/pipeline/jobs/{job_id}/pause", headers=auth_y).status_code == 404
    assert client.post(f"/me/pipeline/jobs/{job_id}/resume", headers=auth_y).status_code == 404
    assert client.post(f"/me/pipeline/jobs/{job_id}/kill", headers=auth_y).status_code == 404

    # User X pauses their own job
    pause_resp = client.post(f"/me/pipeline/jobs/{job_id}/pause", headers=auth_x)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["status"] == "paused"
    assert not ctrl.pause_event.is_set()

    # User X resumes their own job
    resume_resp = client.post(f"/me/pipeline/jobs/{job_id}/resume", headers=auth_x)
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "running"
    assert ctrl.pause_event.is_set()

    # User X kills their own job
    kill_resp = client.post(f"/me/pipeline/jobs/{job_id}/kill", headers=auth_x)
    assert kill_resp.status_code == 200
    assert kill_resp.json()["status"] == "killed"
    assert ctrl.kill_event.is_set()


def test_user_job_detail_and_pdf_download(client, monkeypatch):
    auth_u = get_auth_header(client, "user_download_test", "pass_dl")
    auth_other = get_auth_header(client, "user_other", "pass_other")

    job_id = "job_dl_123"
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "owner": "user_download_test",
        "targets": ["invoice.pdf"],
        "successes": [{"pdf": "invoice.pdf", "pages": 1, "extracted_data": {"total": 500}}],
        "failures": [],
        "created_at": "2026-09-01T12:00:00Z",
        "finished_at": "2026-09-01T12:01:00Z",
    }

    # 1. Owner can get job details via /me/pipeline/jobs/{job_id}
    res_b = client.get(f"/me/pipeline/jobs/{job_id}", headers=auth_u)
    assert res_b.status_code == 200
    assert res_b.json()["job_id"] == job_id

    # 2. Non-admin gets 403 on admin-only route /pipeline/jobs/{job_id}
    assert client.get(f"/pipeline/jobs/{job_id}", headers=auth_u).status_code == 403

    # 3. Other user cannot access owner's job detail via /me/pipeline/jobs/{job_id} (returns 404)
    assert client.get(f"/me/pipeline/jobs/{job_id}", headers=auth_other).status_code == 404

    # 4. Owner can download job PDF report via /me/pipeline/jobs/{job_id}/pdf
    pdf_resp = client.get(f"/me/pipeline/jobs/{job_id}/pdf", headers=auth_u)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 0

    # 5. Other user cannot download owner's PDF report via /me/pipeline/jobs/{job_id}/pdf
    pdf_forbidden = client.get(f"/me/pipeline/jobs/{job_id}/pdf", headers=auth_other)
    assert pdf_forbidden.status_code == 403


def test_user_schema_download_pdf_and_json(client):
    auth_s = get_auth_header(client, "user_schema_downloader", "pass_s")

    schema_id = "schema_dl_test_456"
    schema_record = {
        "schema_id": schema_id,
        "document_type": "receipt",
        "confirmed_at": "2026-09-01T10:00:00Z",
        "owner": "user_schema_downloader",
        "schema": {
            "document_type": "receipt",
            "fields": [
                {"name": "merchant", "type": "string", "required": True},
                {"name": "amount", "type": "number", "required": True, "currency": "USD"},
            ],
        },
    }
    target_path = SCHEMA_REGISTRY / f"{schema_id}.json"
    target_path.write_text(json.dumps(schema_record), encoding="utf-8")

    # 1. User can download schema JSON
    json_resp = client.get(f"/schema/{schema_id}/json", headers=auth_s)
    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"] == "application/json"
    data = json.loads(json_resp.content.decode("utf-8"))
    assert data["schema_id"] == schema_id
    assert data["document_type"] == "receipt"

    # 2. User can download schema PDF
    pdf_resp = client.get(f"/schema/{schema_id}/pdf", headers=auth_s)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 0

