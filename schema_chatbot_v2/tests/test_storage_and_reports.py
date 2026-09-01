"""
Tests for PostgreSQL/SQLite storage fallback and Job PDF report generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.core.auth import create_access_token
from src.ai.layer3_extraction.storage import (
    init_db,
    save_document,
    save_markdown_record,
    save_extraction_run,
    save_job_pdf,
    get_job_pdf,
)
from src.utils.job_pdf_report import generate_job_pdf


@pytest.fixture
def admin_token():
    store = get_user_store()
    user = store.get_by_username("admin")
    return create_access_token(user)


@pytest.fixture
def user_token():
    store = get_user_store()
    user = store.get_by_username("test_storage_user")
    if not user:
        user = store.create("test_storage_user", "password123", Role.USER)
    return create_access_token(user)


def test_storage_fallback_and_operations():
    """Verify storage initializes (SQLite fallback when Postgres is offline) and persists records."""
    init_db()

    # Save Document
    doc_id = save_document(
        filename="test_invoice.pdf",
        file_bytes=b"%PDF-1.4 test binary data",
        content_type="application/pdf",
    )
    assert doc_id is not None

    # Save Markdown record
    md_id = save_markdown_record(
        doc_id="test_invoice.pdf",
        md_filename="test_invoice.md",
        markdown_content="# Invoice Summary\nTotal: $500",
        schema_id="invoice_schema",
        page_count=1,
    )
    assert md_id is not None

    # Save Extraction Run
    run_id = save_extraction_run(
        doc_id="test_invoice.pdf",
        page_count=1,
        schema_name="invoice_schema",
        result_json={"invoice_number": "INV-1001", "total_amount": "$500.00"},
        llm_provider="sarvam",
        model_name="sarvam-105b",
        processing_time_seconds=1.25,
    )
    assert run_id is not None

    # Save & Retrieve Job PDF
    save_job_pdf("job_test_001", b"%PDF report binary", "job_test_001_report.pdf")
    retrieved = get_job_pdf("job_test_001")
    assert retrieved is not None
    pdf_bytes, fname = retrieved
    assert pdf_bytes == b"%PDF report binary"
    assert fname == "job_test_001_report.pdf"


def test_job_pdf_report_generator():
    """Verify ReportLab builds a valid PDF binary from job data."""
    mock_job = {
        "job_id": "job_sample_123",
        "status": "completed",
        "schema_id": "invoice_v1",
        "created_at": "2026-08-31T12:00:00Z",
        "finished_at": "2026-08-31T12:00:05Z",
        "targets": ["sample1.pdf", "sample2.pdf"],
        "results": [
            {
                "pdf": "sample1.pdf",
                "pages": 2,
                "elapsed_s": 1.4,
                "extract_elapsed_s": 2.1,
                "avg_conf": 0.98,
                "md": "sample1.md",
                "extracted_json": "sample1.extracted.json",
                "extracted_data": {"patient_name": "John Doe", "amount": 1500},
            }
        ],
        "failures": [],
    }

    pdf_bytes = generate_job_pdf(mock_job)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_pdf_job_download_endpoint(admin_token):
    """Verify GET /pipeline/jobs/{job_id}/pdf endpoint."""
    client = TestClient(app)

    # Put a mock job in memory or storage
    from app.api.pipeline_routes import _pipeline_jobs
    _pipeline_jobs["job_download_test"] = {
        "job_id": "job_download_test",
        "status": "completed",
        "schema_id": "test_schema",
        "targets": ["doc.pdf"],
        "results": [{"pdf": "doc.pdf", "pages": 1, "elapsed_s": 0.5}],
        "failures": [],
    }

    resp = client.get(
        "/pipeline/jobs/job_download_test/pdf",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
