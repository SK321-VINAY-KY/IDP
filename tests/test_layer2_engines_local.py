"""
Unit and integration tests for Layer 2 isolated engine implementations and Lambda handlers.
Verifies contract compliance and local execution across Docling, PaddleOCR Printed, and PaddleOCR Handwritten.
"""
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import docker.docling.engine as docling_engine
import docker.docling.handler as docling_handler
import docker.paddle_handwritten.engine as paddle_hand_engine
import docker.paddle_handwritten.handler as paddle_hand_handler
import docker.paddle_printed.engine as paddle_printed_engine
import docker.paddle_printed.handler as paddle_printed_handler

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIGITAL_PDF = ROOT_DIR / "dataset" / "test.pdf"
SAMPLE_SCANNED_PDF = ROOT_DIR / "dataset" / "165.pdf"


@pytest.fixture
def digital_pdf_path() -> str:
    assert SAMPLE_DIGITAL_PDF.exists(), f"Sample PDF not found at {SAMPLE_DIGITAL_PDF}"
    return str(SAMPLE_DIGITAL_PDF)


@pytest.fixture
def scanned_pdf_path() -> str:
    assert SAMPLE_SCANNED_PDF.exists(), f"Sample PDF not found at {SAMPLE_SCANNED_PDF}"
    return str(SAMPLE_SCANNED_PDF)


class DummyContext:
    aws_request_id = "test-req-layer2-12345"


# ==============================================================================
# Docling Engine & Handler Tests
# ==============================================================================

def test_docling_engine_local_path(digital_pdf_path):
    results = docling_engine.process_pages(digital_pdf_path, [1])
    assert len(results) == 1
    res = results[0]
    assert res["page_number"] == 1
    assert res["engine"] == "docling"
    assert res["status"] == "SUCCESS"
    assert res["confidence"] >= 0.50
    assert res["lines_extracted"] > 0
    assert len(res["markdown"]) > 20
    assert res["latency_ms"] >= 0


def test_docling_handler_local_path(digital_pdf_path):
    event = {
        "document_id": "test.pdf",
        "local_path": digital_pdf_path,
        "pages": [1],
    }
    resp = docling_handler.lambda_handler(event, DummyContext())
    assert resp["statusCode"] == 200
    body = resp["body"]
    assert body["document_id"] == "test.pdf"
    assert body["engine"] == "docling"
    assert len(body["results"]) == 1
    assert body["results"][0]["page_number"] == 1
    assert body["results"][0]["lines_extracted"] > 0


def test_docling_handler_mocked_s3(digital_pdf_path):
    pdf_bytes = Path(digital_pdf_path).read_bytes()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: pdf_bytes)}

    event = {
        "document_id": "test.pdf",
        "s3_bucket": "test-bucket",
        "s3_key": "documents/test.pdf",
        "page_number": 1,
    }

    with patch.object(docling_handler, "get_s3_client", return_value=mock_s3):
        resp = docling_handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert body["engine"] == "docling"
    assert len(body["results"]) == 1


# ==============================================================================
# PaddleOCR Printed Engine & Handler Tests
# ==============================================================================

def test_paddle_printed_engine_local_path(scanned_pdf_path):
    results = paddle_printed_engine.process_pdf_pages(scanned_pdf_path, [1])
    assert len(results) == 1
    res = results[0]
    assert res["page_number"] == 1
    assert res["engine"] == "paddleocr_printed"
    assert res["status"] == "SUCCESS"
    assert res["latency_ms"] >= 0


def test_paddle_printed_handler_mocked_s3(scanned_pdf_path):
    pdf_bytes = Path(scanned_pdf_path).read_bytes()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: pdf_bytes)}

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "test-bucket",
        "s3_key": "documents/165.pdf",
        "pages": [1],
    }

    with patch.object(paddle_printed_handler, "get_s3_client", return_value=mock_s3):
        resp = paddle_printed_handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert body["engine"] == "paddleocr_printed"
    assert len(body["results"]) == 1


# ==============================================================================
# PaddleOCR Handwritten Engine & Handler Tests
# ==============================================================================

def test_paddle_handwritten_engine_local_path(scanned_pdf_path):
    results = paddle_hand_engine.process_pdf_pages(scanned_pdf_path, [1])
    assert len(results) == 1
    res = results[0]
    assert res["page_number"] == 1
    assert res["engine"] == "paddleocr_handwritten"
    assert res["status"] == "SUCCESS"
    assert res["latency_ms"] >= 0


def test_paddle_handwritten_handler_mocked_s3(scanned_pdf_path):
    pdf_bytes = Path(scanned_pdf_path).read_bytes()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: pdf_bytes)}

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "test-bucket",
        "s3_key": "documents/165.pdf",
        "page_number": 1,
    }

    with patch.object(paddle_hand_handler, "get_s3_client", return_value=mock_s3):
        resp = paddle_hand_handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert body["engine"] == "paddleocr_handwritten"
    assert len(body["results"]) == 1


# ==============================================================================
# Standard Contract Equality Verification
# ==============================================================================

def test_standard_contract_consistency():
    """Verify all 3 engines output identical key schemas."""
    expected_keys = {"page_number", "engine", "markdown", "confidence", "lines_extracted", "latency_ms", "status"}
    
    docling_res = docling_engine.process_pages(str(SAMPLE_DIGITAL_PDF), [1])[0]
    paddle_p_res = paddle_printed_engine.process_pdf_pages(str(SAMPLE_DIGITAL_PDF), [1])[0]
    paddle_h_res = paddle_hand_engine.process_pdf_pages(str(SAMPLE_DIGITAL_PDF), [1])[0]

    assert expected_keys.issubset(docling_res.keys())
    assert expected_keys.issubset(paddle_p_res.keys())
    assert expected_keys.issubset(paddle_h_res.keys())
