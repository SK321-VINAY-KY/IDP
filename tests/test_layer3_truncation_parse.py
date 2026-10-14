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


# --------------------------------------------------------------------------- #
# 2b — stopReason: one larger-budget retry on max_tokens, else degraded
# --------------------------------------------------------------------------- #

def test_max_tokens_retries_once_at_ceiling_then_succeeds(fresh, monkeypatch):
    monkeypatch.setattr(settings, "graph_extraction_max_tokens", 8000, raising=False)
    monkeypatch.setattr(settings, "graph_extraction_max_tokens_ceiling", 12000, raising=False)
    budgets = []
    def seq(**kw):
        budgets.append(kw["inferenceConfig"]["maxTokens"])
        if len(budgets) == 1:
            return _resp("max_tokens", ENTITIES_JSON, out_tokens=8000)
        return _resp("end_turn", ENTITIES_JSON, out_tokens=5000)
    client = _make_client(seq)
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=1)
    assert budgets == [8000, 12000]              # retried once at the ceiling
    assert out["entities"]                       # recovered content
    assert out["quality"] == "clean"
    s = fresh.summarize()
    assert s["retry_attempts"] >= 1
    assert s["degraded_pages"] == 0


def test_max_tokens_twice_marks_page_degraded(fresh, monkeypatch):
    monkeypatch.setattr(settings, "graph_extraction_max_tokens", 8000, raising=False)
    monkeypatch.setattr(settings, "graph_extraction_max_tokens_ceiling", 12000, raising=False)
    def seq(**kw):
        return _resp("max_tokens", ENTITIES_JSON, out_tokens=kw["inferenceConfig"]["maxTokens"])
    client = _make_client(seq)
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=2)
    assert out["quality"] == "degraded"
    assert out["entities"] == []                 # cannot trust a cut-off object
    s = fresh.summarize()
    assert s["degraded_pages"] == 1
    assert s["max_tokens_stops"] == 2            # original + retry both truncated


def test_clean_page_has_clean_quality_and_no_retry(fresh):
    client = _make_client(lambda **kw: _resp("end_turn", ENTITIES_JSON, out_tokens=1000))
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=1)
    assert out["quality"] == "clean"
    s = fresh.summarize()
    assert s["retry_attempts"] == 0
    assert s["degraded_pages"] == 0
    assert s["clean_pages"] == 1


def test_page_delta_carries_quality_through_agent():
    """The agent must thread the client's quality onto the PageDelta."""
    import asyncio
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
    from pydantic import BaseModel, Field

    class S(BaseModel):
        patient_name: str = Field(default="", description="n")

    class FakeLLM:
        def extract_graph_from_page(self, **kw):
            return {"entities": [], "relationships": [], "reference_resolutions": [],
                    "quality": "degraded"}

    agent = GraphExtractionAgent(schema=S, graph=GraphMemory(), llm=FakeLLM())
    delta = asyncio.run(agent.build_page_delta_async(
        page_number=1, markdown="some text", total_pages=1, context=[]))
    assert delta.quality == "degraded"
    # round-trips through checkpoint serialization
    from src.ai.layer3_extraction.graph_agent.models import PageDelta
    assert PageDelta.from_dict(delta.to_dict()).quality == "degraded"


# --------------------------------------------------------------------------- #
# 2c — parse failures: re-ask once; salvage -> partial; invalid twice -> degraded
# --------------------------------------------------------------------------- #

# The exact Step-1 observed signature: a structurally complete single object with an
# unescaped inner double-quote in a string value -> json.loads raises
# "Expecting ',' delimiter" though braces balance; stopReason end_turn, under budget.
UNESCAPED_QUOTE_JSON = (
    '{"entities": [{"id":"e1","type":"Amount","value":"he said "hi" to me",'
    '"evidence":"x"}], "relationships": [], "reference_resolutions": []}'
)


def test_observed_signature_invalid_quote_reask_then_accepts(fresh):
    """First parse fails on the unescaped-quote signature -> one re-ask -> valid
    second response accepted. Exactly one re-ask is issued."""
    calls = []
    def seq(**kw):
        calls.append(1)
        if len(calls) == 1:
            return _resp("end_turn", UNESCAPED_QUOTE_JSON, out_tokens=2000)
        return _resp("end_turn", ENTITIES_JSON, out_tokens=2100)
    client = _make_client(seq)
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=17)
    assert len(calls) == 2                       # exactly one re-ask
    assert out["entities"]                       # second valid response accepted
    assert out["quality"] == "clean"
    s = fresh.summarize()
    assert s["reask_count"] == 1
    assert s["reask_recovered"] == 1
    assert s["degraded_pages"] == 0


def test_invalid_json_twice_leaves_page_degraded(fresh):
    def seq(**kw):
        return _resp("end_turn", UNESCAPED_QUOTE_JSON, out_tokens=2000)
    client = _make_client(seq)
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=17)
    assert out["quality"] == "degraded"
    assert out["entities"] == []
    s = fresh.summarize()
    assert s["reask_count"] == 1
    assert s["reask_recovered"] == 0
    assert s["degraded_pages"] == 1


def test_salvage_path_flags_partial_not_clean(fresh):
    """Trailing non-JSON after the object -> deterministic substring salvage
    recovers it; flagged partial (never clean), no re-ask."""
    salvageable = ENTITIES_JSON + "   trailing noise not json"
    client = _make_client(lambda **kw: _resp("end_turn", salvageable, out_tokens=2000))
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=4)
    assert out["entities"]                       # salvaged
    assert out["quality"] == "partial"
    s = fresh.summarize()
    assert s["partial_pages"] == 1
    assert s["reask_count"] == 0                 # salvage is deterministic, no re-ask


def test_fenced_json_is_clean_no_reask(fresh):
    fenced = "```json\n" + ENTITIES_JSON + "\n```"
    client = _make_client(lambda **kw: _resp("end_turn", fenced, out_tokens=2000))
    out = client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=1)
    assert out["entities"]
    assert out["quality"] == "clean"             # fence strip is deterministic
    s = fresh.summarize()
    assert s["reask_count"] == 0


def test_reask_tokens_and_cost_recorded_as_extra_call(fresh):
    """The re-ask must go through Converse so its tokens are recorded separately."""
    calls = []
    def seq(**kw):
        calls.append(1)
        if len(calls) == 1:
            return _resp("end_turn", UNESCAPED_QUOTE_JSON, out_tokens=2000, in_tokens=1500)
        return _resp("end_turn", ENTITIES_JSON, out_tokens=300, in_tokens=1600)
    client = _make_client(seq)
    client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=17)
    reask_recs = [r for r in fresh.records_for() if r.purpose == m.PURPOSE_GRAPH_REASK]
    assert len(reask_recs) == 1
    assert reask_recs[0].output_tokens == 300    # the extra call's tokens are captured
    assert reask_recs[0].input_tokens == 1600


def test_combined_18_scored_fields_resolver_reask(fresh):
    """Resolver also re-asks once on invalid JSON (both methods per amendment)."""
    from pydantic import BaseModel, Field

    class S(BaseModel):
        a: str = Field(default="", description="x")

    calls = []
    def seq(**kw):
        calls.append(1)
        if len(calls) == 1:
            return _resp("end_turn", '{"a": "v" bad}', out_tokens=50)
        return _resp("end_turn", '{"a": "value"}', out_tokens=50)
    client = _make_client(seq)
    out = client.resolve_schema_from_graph("evidence", S)
    assert out.a == "value"                      # recovered on re-ask
    assert len(calls) == 2
    reask = [r for r in fresh.records_for() if r.purpose == m.PURPOSE_RESOLVER_REASK]
    assert len(reask) == 1


# --------------------------------------------------------------------------- #
# 2d — no response text in parse-failure logs (length + category only)
# --------------------------------------------------------------------------- #

import logging  # noqa: E402

SECRET = "CONFIDENTIAL_" + "RESPONSE_TEXT_SECRET_98765"


def _log_blob(caplog):
    return ("\n".join(r.getMessage() for r in caplog.records)
            + "\n" + str([r.__dict__ for r in caplog.records]))


def test_graph_parse_failure_log_has_no_response_text(fresh, caplog):
    leaky = ('{"entities": [{"id":"e1","value":"' + SECRET + '" invalid}], '
             '"relationships": [], "reference_resolutions": []}')
    # stays invalid on the re-ask too so both parse-fail log sites fire
    client = _make_client(lambda **kw: _resp("end_turn", leaky, out_tokens=2000))
    with caplog.at_level(logging.WARNING):
        client.extract_graph_from_page("page", SCHEMA_FIELDS, [], page_number=17)
    blob = _log_blob(caplog)
    assert SECRET not in blob, "response text leaked into graph parse-failure log"
    # the metadata we DO log is present
    assert any("parse_failed" in r.getMessage() or "reask_still_invalid" in r.getMessage()
               for r in caplog.records)


def test_resolver_parse_failure_log_has_no_response_text(fresh, caplog):
    from pydantic import BaseModel, Field

    class S(BaseModel):
        a: str = Field(default="", description="x")

    leaky = '{"a": "' + SECRET + '" invalid json}'
    client = _make_client(lambda **kw: _resp("end_turn", leaky, out_tokens=50))
    with caplog.at_level(logging.WARNING):
        client.resolve_schema_from_graph("evidence", S)
    assert SECRET not in _log_blob(caplog), "response text leaked into resolver parse-failure log"


def test_extract_parse_failure_log_has_no_response_text(fresh, caplog):
    from pydantic import BaseModel, Field

    class S(BaseModel):
        a: str = Field(default="", description="x")

    leaky = "{ invalid " + SECRET + " not json ::::"
    client = _make_client(lambda **kw: _resp("end_turn", leaky, out_tokens=50))
    with caplog.at_level(logging.WARNING):
        client.extract("some content", S)
    assert SECRET not in _log_blob(caplog), "response text leaked into extract parse-failure log"


def test_navigate_parse_failure_log_has_no_response_text(fresh, caplog):
    leaky = "not json " + SECRET + " ::::"
    client = _make_client(lambda **kw: _resp("end_turn", leaky, out_tokens=50))
    with caplog.at_level(logging.WARNING):
        client.navigate(["summary one"], ["field_a"])
    assert SECRET not in _log_blob(caplog), "response text leaked into navigate parse-failure log"


def test_parse_failure_category_is_metadata_only():
    from src.adapters.llm.bedrock_client import _parse_failure_category
    assert _parse_failure_category("```json\n{}\n```") == "fenced"
    assert _parse_failure_category("no object here") == "no_json_object"
    assert _parse_failure_category('{"a": 1') == "unbalanced_braces"
    assert _parse_failure_category('prefix {"a":1} suffix') == "surrounding_text"
    assert _parse_failure_category('{"a": "b" "c"}') == "invalid_json_balanced"
    # the category never contains the input text
    secret = "SENSITIVE_VALUE_XYZ"
    assert secret not in _parse_failure_category('{"a":"' + secret + '" x}')
