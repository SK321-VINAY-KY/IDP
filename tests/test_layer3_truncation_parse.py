"""
Tests for audit finding #1 fixes in the Layer-3 Bedrock call path.

Commit 2a (this file's initial content): configurable graph-extraction output
budget (default 8000) + reasoning-model id markers moved into config.
Later commits (2b-2e) append their own cases below.

All Converse responses are mocked; no real AWS. owner stays required where it was.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.llm import llm_metrics as m
from src.adapters.llm.bedrock_client import BedrockExtractionClient, _is_reasoning_model
from src.config.settings import settings


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _resp(stop_reason, text, out_tokens=50, in_tokens=100, with_usage=True,
          with_reasoning=False):
    content = []
    if with_reasoning:
        content.append({"reasoningContent": {"reasoningText": {"text": "thinking"}}})
    content.append({"text": text})
    r = {"output": {"message": {"content": content}}, "stopReason": stop_reason}
    if with_usage:
        r["usage"] = {"inputTokens": in_tokens, "outputTokens": out_tokens,
                      "totalTokens": in_tokens + out_tokens}
    return r


ENTITIES_JSON = ('{"entities": [{"id":"e1","type":"Person","value":"X"}], '
                 '"relationships": [], "reference_resolutions": []}')

SCHEMA_FIELDS = [{"name": "patient_name", "description": "n"}]


def _make_client(responses):
    """responses: a callable(**kw)->resp, or a list consumed one per converse call."""
    client = BedrockExtractionClient(model_id="test-gpt-oss-model", region="ap-south-1")
    fake = MagicMock()
    if callable(responses):
        fake.converse = responses
    else:
        seq = list(responses)
        def _next(**kw):
            if not seq:
                raise AssertionError("more converse calls than mocked responses")
            r = seq.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        fake.converse = _next
    client._client = fake
    return client


@pytest.fixture()
def fresh():
    c = m.get_collector()
    c.reset()
    yield c
    c.reset()


# --------------------------------------------------------------------------- #
# 2a — configurable budget + reasoning-model markers in config
# --------------------------------------------------------------------------- #

def test_graph_extraction_uses_configured_budget_default_8000(fresh):
    seen = {}
    def capture(**kw):
        seen["maxTokens"] = kw["inferenceConfig"]["maxTokens"]
        return _resp("end_turn", ENTITIES_JSON)
    client = _make_client(capture)
    client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=1)
    # BEFORE fix: hardcoded 4000. AFTER: settings.graph_extraction_max_tokens (8000).
    assert seen["maxTokens"] == settings.graph_extraction_max_tokens == 8000


def test_graph_extraction_budget_is_configurable(fresh, monkeypatch):
    monkeypatch.setattr(settings, "graph_extraction_max_tokens", 9000, raising=False)
    seen = {}
    def capture(**kw):
        seen["maxTokens"] = kw["inferenceConfig"]["maxTokens"]
        return _resp("end_turn", ENTITIES_JSON)
    client = _make_client(capture)
    client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=1)
    assert seen["maxTokens"] == 9000


def test_reasoning_model_marker_is_config_driven(monkeypatch):
    assert _is_reasoning_model("some-gpt-oss-120b") is True
    assert _is_reasoning_model("arn:...:qdtz23c8eis1") is True
    assert _is_reasoning_model("anthropic.claude-3") is False
    monkeypatch.setattr(settings, "bedrock_reasoning_model_markers", ["claude-3"], raising=False)
    assert _is_reasoning_model("anthropic.claude-3") is True
    assert _is_reasoning_model("some-gpt-oss-120b") is False
