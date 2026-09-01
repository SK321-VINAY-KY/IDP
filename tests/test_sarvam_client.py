"""
Unit tests for SarvamExtractionClient.
"""
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field
import pytest

from src.adapters.llm.sarvam_client import SarvamExtractionClient
from src.adapters.llm.extraction_base import ExtractionLLMClient


class SampleSchema(BaseModel):
    patient_name: str = Field(default="", description="Name of the patient")
    total_amount: str = Field(default="", description="Total invoice amount")


def test_sarvam_client_implements_protocol():
    """Verify SarvamExtractionClient adheres to ExtractionLLMClient protocol."""
    client = SarvamExtractionClient()
    assert isinstance(client, ExtractionLLMClient)


def test_sarvam_summarize_page():
    """Verify summarize_page renders prompt and queries LLM."""
    client = SarvamExtractionClient()
    mock_choice = MagicMock()
    mock_choice.message.content = "Summary of medical bill page 1."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch.object(client.raw_client.chat.completions, "create", return_value=mock_resp):
        summary = client.summarize_page("Page 1 Markdown text", max_words=50)
        assert summary == "Summary of medical bill page 1."


def test_sarvam_navigate():
    """Verify navigate maps field names to page numbers."""
    client = SarvamExtractionClient()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"patient_name": [1], "total_amount": [1, 2]}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch.object(client.raw_client.chat.completions, "create", return_value=mock_resp):
        nav_map = client.navigate(["Page 1 Summary", "Page 2 Summary"], ["patient_name", "total_amount"])
        assert nav_map == {"patient_name": [1], "total_amount": [1, 2]}


def test_sarvam_check_page_for_fields():
    """Verify check_page_for_fields extracts match list."""
    client = SarvamExtractionClient()
    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"matches": [{"field": "patient_name", "value": "Alice Doe"}]}\n```'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch.object(client.raw_client.chat.completions, "create", return_value=mock_resp):
        schema_fields = [{"name": "patient_name", "description": ""}]
        matches = client.check_page_for_fields("# Bill\nPatient: Alice Doe", schema_fields, 1, 1)
        assert matches == [{"field": "patient_name", "value": "Alice Doe"}]
