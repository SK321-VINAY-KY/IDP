"""
Tests for SarvamAdapter truncation detection and completion handling.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from app.llm.sarvam_adapter import SarvamAdapter, TruncatedCompletionError


@pytest.fixture
def adapter():
    return SarvamAdapter(api_key="test_sk_123", model="sarvam-105b", base_url="http://mock-sarvam.ai")


def test_chat_raises_truncated_completion_error_on_length_finish_reason(adapter):
    """Assert _chat() raises TruncatedCompletionError when finish_reason is 'length'."""
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": '{"document_type": "invoice", "fields": [{"name": "invo',
                    "reasoning_content": "Thinking about the document...",
                },
            }
        ]
    }

    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(TruncatedCompletionError) as exc_info:
            adapter._chat(
                system="system prompt",
                user="user prompt",
                max_tokens=1536,
            )

    assert "truncated" in str(exc_info.value).lower()
    assert exc_info.value.finish_reason == "length"


def test_infer_schema_from_pdfs_sets_extraction_failed_on_truncated_response(adapter):
    """Given a truncated _chat() completion, infer_schema_from_pdfs sets extraction_failed=True and failure_reason."""
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": '{"document_type": "resume", "fields": [',
                    "reasoning_content": "partial reasoning...",
                },
            }
        ]
    }

    with patch.object(adapter, "_digitise", return_value="# Sample Resume\nName: John Doe\nSkills: Python"):
        with patch("httpx.post", return_value=mock_resp):
            proposal = adapter.infer_schema_from_pdfs([b"%PDF-sample-1"])

    assert proposal.extraction_failed is True
    assert proposal.failure_reason is not None
    assert "truncated" in proposal.failure_reason.lower()
    assert "max_tokens" in proposal.failure_reason.lower()


def test_infer_schema_from_pdfs_succeeds_on_normal_stop_finish_reason(adapter):
    """Control test: normal finish_reason='stop' returns successful SchemaProposal with document_type."""
    valid_schema_json = {
        "document_type": "medical_bill",
        "fields": [
            {
                "name": "patient_name",
                "type": "string",
                "required": True,
                "seen_in_samples": 1,
            },
            {
                "name": "total_amount",
                "type": "number",
                "required": True,
                "seen_in_samples": 1,
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(valid_schema_json),
                    "reasoning_content": "Identified medical bill with patient name and total amount.",
                },
            }
        ]
    }

    with patch.object(adapter, "_digitise", return_value="# Medical Bill\nPatient: Jane Doe\nTotal: $1200"):
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            proposal = adapter.infer_schema_from_pdfs([b"%PDF-sample-1"])

    assert proposal.extraction_failed is False
    assert proposal.failure_reason is None
    assert proposal.document_type == "medical_bill"
    assert len(proposal.fields) == 2
    assert proposal.fields[0].name == "patient_name"
    assert proposal.fields[0].total_samples == 1

    # Verify payload uses reasoning_effort=None and max_tokens=2560
    sent_payload = mock_post.call_args.kwargs["json"]
    assert "reasoning_effort" in sent_payload
    assert sent_payload["reasoning_effort"] is None
    assert sent_payload["max_tokens"] == 2560


def test_chat_sends_literal_reasoning_effort_none_in_payload(adapter):
    """When reasoning_effort=None, _chat() must explicitly include 'reasoning_effort': None in JSON payload."""
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "ok", "reasoning_content": ""},
            }
        ]
    }

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        adapter._chat(
            system="system prompt",
            user="user prompt",
            reasoning_effort=None,
            max_tokens=2560,
        )

    sent_payload = mock_post.call_args.kwargs["json"]
    assert "reasoning_effort" in sent_payload
    assert sent_payload["reasoning_effort"] is None


def test_infer_schema_from_pdfs_succeeds_with_large_reasoning_and_valid_content(adapter):
    """Regression test for max_tokens=2560: reasoning_content (~5,800 chars / ~1,475 tokens) + valid content JSON."""
    large_reasoning = "We need to analyze this document carefully. " * 130  # ~5,850 chars
    valid_schema_json = {
        "document_type": "resume",
        "fields": [
            {"name": "full_name", "type": "string", "required": True, "seen_in_samples": 1},
            {"name": "skills", "type": "string", "required": False, "seen_in_samples": 1},
            {"name": "experience_years", "type": "string", "required": False, "seen_in_samples": 1},
        ],
    }

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "usage": {
            "prompt_tokens": 1650,
            "completion_tokens": 1950,
            "total_tokens": 3600,
        },
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(valid_schema_json),
                    "reasoning_content": large_reasoning,
                },
            }
        ],
    }

    with patch.object(adapter, "_digitise", return_value="# Resume\nName: Alice\nSkills: Python, Go"):
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            proposal = adapter.infer_schema_from_pdfs([b"%PDF-sample"])

    assert proposal.extraction_failed is False
    assert proposal.failure_reason is None
    assert proposal.document_type == "resume"
    assert len(proposal.fields) == 3
    assert proposal.fields[0].name == "full_name"

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["max_tokens"] == 2560
    assert sent_payload["reasoning_effort"] is None


