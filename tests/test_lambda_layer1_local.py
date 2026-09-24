"""
Unit tests for Layer 1 AWS Lambda Handler (local test with mocked S3).
"""
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.lambda_layer1_handler import lambda_handler


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    sample_path = Path(__file__).resolve().parents[1] / "dataset" / "test.pdf"
    assert sample_path.exists(), f"Sample test PDF not found at {sample_path}"
    return sample_path.read_bytes()


def test_lambda_invalid_event():
    class DummyContext:
        aws_request_id = "test-req-invalid"

    res = lambda_handler({}, DummyContext())
    assert res["statusCode"] == 400
    body = json.loads(res["body"])
    assert "error" in body


def test_lambda_direct_invocation_mocked_s3(sample_pdf_bytes):
    class DummyContext:
        aws_request_id = "test-req-direct"

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: sample_pdf_bytes)
    }

    event = {
        "s3_bucket": "vinay-rag-documents-2026",
        "s3_key": "documents/test.pdf",
    }

    with patch("src.api.lambda_layer1_handler.get_s3_client", return_value=mock_s3):
        res = lambda_handler(event, DummyContext())

    assert res["statusCode"] == 200
    body = res["body"]
    assert body["document_name"] == "test.pdf"
    assert body["s3_bucket"] == "vinay-rag-documents-2026"
    assert body["s3_key"] == "documents/test.pdf"
    assert body["total_pages"] > 0
    assert len(body["pages"]) == body["total_pages"]

    first_page = body["pages"][0]
    assert first_page["page_number"] == 1
    assert "profile" in first_page
    assert "route" in first_page
    assert "tasks" in first_page
    assert isinstance(first_page["tasks"], list)
    assert len(first_page["tasks"]) > 0


def test_lambda_s3_event_invocation(sample_pdf_bytes):
    class DummyContext:
        aws_request_id = "test-req-s3-event"

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: sample_pdf_bytes)
    }

    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "vinay-rag-documents-2026"},
                    "object": {"key": "documents%2Ftest.pdf"},
                }
            }
        ]
    }

    with patch("src.api.lambda_layer1_handler.get_s3_client", return_value=mock_s3):
        res = lambda_handler(event, DummyContext())

    assert res["statusCode"] == 200
    body = res["body"]
    assert body["s3_bucket"] == "vinay-rag-documents-2026"
    assert body["s3_key"] == "documents/test.pdf"
    assert body["total_pages"] > 0


def test_lambda_base64_payload(sample_pdf_bytes):
    class DummyContext:
        aws_request_id = "test-req-base64"

    event = {
        "pdf_base64": base64.b64encode(sample_pdf_bytes).decode("ascii"),
        "document_name": "inline_test.pdf",
    }

    res = lambda_handler(event, DummyContext())
    assert res["statusCode"] == 200
    body = res["body"]
    assert body["document_name"] == "inline_test.pdf"
    assert body["total_pages"] > 0


def test_lambda_s3_read_error():
    class DummyContext:
        aws_request_id = "test-req-s3-fail"

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("NoSuchKey")

    event = {
        "s3_bucket": "vinay-rag-documents-2026",
        "s3_key": "missing.pdf",
    }

    with patch("src.api.lambda_layer1_handler.get_s3_client", return_value=mock_s3):
        res = lambda_handler(event, DummyContext())

    assert res["statusCode"] == 500
    body = json.loads(res["body"])
    assert "NoSuchKey" in body["error"]
