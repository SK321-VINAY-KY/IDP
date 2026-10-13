"""Tests for the measurement-only LLM Converse instrumentation (audit finding #1).

Covers mocked Converse responses: normal stop, max_tokens, unparseable JSON,
empty entities, throttling exception, missing usage field. Plus a strict test
that NO prompt / document / model-output text appears in the log output or the
owner-partitioned summary file.
"""
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.llm import llm_metrics as m
from src.adapters.llm.bedrock_client import BedrockExtractionClient


# ---- fixtures --------------------------------------------------------------

def _resp(stop_reason, out_tokens=50, in_tokens=100, text='{"entities": [{"id":"e1","type":"Person","value":"X"}]}',
          with_usage=True):
    r = {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": stop_reason,
    }
    if with_usage:
        r["usage"] = {"inputTokens": in_tokens, "outputTokens": out_tokens, "totalTokens": in_tokens + out_tokens}
    return r


@pytest.fixture()
def fresh_collector():
    c = m.get_collector()
    c.reset()
    yield c
    c.reset()


def _make_client_with_mock(mock_converse):
    client = BedrockExtractionClient(model_id="test-model", region="ap-south-1")
    fake = MagicMock()
    fake.converse = mock_converse
    client._client = fake  # inject mock, no real AWS
    return client


# ---- Converse-response cases ----------------------------------------------

def test_normal_stop_records_ok(fresh_collector):
    client = _make_client_with_mock(lambda **kw: _resp("end_turn", out_tokens=120))
    out = client.extract_graph_from_page("page", [{"name": "patient_name", "description": "n"}], [], page_number=1)
    assert out["entities"]  # pass-through returned the parsed entities unchanged
    recs = fresh_collector.records_for()
    assert len(recs) == 1
    assert recs[0].stop_reason == "end_turn"
    assert recs[0].output_tokens == 120
    assert recs[0].outcome == m.OUTCOME_OK


def test_max_tokens_stop_is_recorded(fresh_collector):
    # Truncated: valid-looking but we still read stopReason=max_tokens.
    # audit #1 (2b): a max_tokens stop now triggers ONE larger-budget retry. This
    # mock returns max_tokens on every call, so both the original and the retry
    # truncate -> 2 recorded max_tokens stops and a degraded page.
    client = _make_client_with_mock(lambda **kw: _resp("max_tokens", out_tokens=4000))
    client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=5)
    s = fresh_collector.summarize()
    assert s["stop_reason_distribution"].get("max_tokens") == 2
    assert s["max_tokens_stops"] == 2
    assert s["degraded_pages"] == 1


def test_unparseable_json_tagged_parse_failed(fresh_collector):
    # Non-truncated (end_turn) but invalid JSON. audit #1 (2c): one re-ask is
    # issued; this mock stays invalid on the re-ask too -> page degraded, empty.
    client = _make_client_with_mock(lambda **kw: _resp("end_turn", text="{ this is not valid json ::::"))
    out = client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=3)
    assert out["entities"] == [] and out["relationships"] == [] and out["reference_resolutions"] == []
    assert out["quality"] == "degraded"
    s = fresh_collector.summarize()
    # invalid JSON -> tagged parse_failed (amended from ok)
    assert s["parse_failures"] >= 1


def test_empty_entities_tagged_empty_result(fresh_collector):
    client = _make_client_with_mock(lambda **kw: _resp("end_turn", text='{"entities": [], "relationships": []}'))
    client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=7)
    s = fresh_collector.summarize()
    assert s["empty_result_calls"] == 1


def test_throttling_exception_recorded(fresh_collector):
    def boom(**kw):
        raise RuntimeError("ThrottlingException: rate exceeded")
    # Name the exception type ThrottlingException so the recorder counts it.
    class ThrottlingException(Exception):
        pass
    def boom2(**kw):
        raise ThrottlingException("rate exceeded")
    client = _make_client_with_mock(boom2)
    with pytest.raises(ThrottlingException):
        client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=9)
    recs = fresh_collector.records_for()
    assert len(recs) == 1
    assert recs[0].outcome == m.OUTCOME_EXCEPTION
    assert recs[0].stop_reason == "ThrottlingException"
    s = fresh_collector.summarize()
    assert s["throttling_exceptions"] == 1


def test_missing_usage_field_handled(fresh_collector):
    client = _make_client_with_mock(lambda **kw: _resp("end_turn", with_usage=False))
    client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=2)
    recs = fresh_collector.records_for()
    assert len(recs) == 1
    assert recs[0].input_tokens is None
    assert recs[0].output_tokens is None  # no crash on missing usage


# ---- pass-through guarantee ------------------------------------------------

def test_response_passed_through_unchanged(fresh_collector):
    sentinel = _resp("end_turn", out_tokens=77)
    captured = {}

    def capture(**kw):
        return sentinel

    client = _make_client_with_mock(capture)
    client.extract_graph_from_page("page", [{"name": "x", "description": "y"}], [], page_number=1)
    # The recorder must not have mutated the response object.
    assert sentinel["stopReason"] == "end_turn"
    assert sentinel["usage"]["outputTokens"] == 77
    assert sentinel["output"]["message"]["content"][0]["text"]  # content still intact


# ---- NO CONTENT in logs or summary ----------------------------------------

SECRET_PROMPT = "CONFIDENTIAL_PATIENT_NAME_John_Q_Secret"
SECRET_OUTPUT = "CONFIDENTIAL_MODEL_OUTPUT_diagnosis_text"


def test_no_prompt_or_output_text_in_logs_or_summary(fresh_collector, tmp_path, caplog):
    # Response text contains a secret marker; prompt contains another.
    resp_text = '{"entities": [{"id":"e1","type":"Person","value":"' + SECRET_OUTPUT + '"}]}'
    client = _make_client_with_mock(lambda **kw: _resp("end_turn", text=resp_text))

    m.set_run_context(owner="alice", doc_id="docX")
    try:
        with caplog.at_level(logging.INFO):
            client.extract_graph_from_page(SECRET_PROMPT, [{"name": "x", "description": "y"}], [], page_number=1)
    finally:
        m.clear_run_context()

    # 1. No secret in any captured log record (message or args).
    log_blob = "\n".join(r.getMessage() for r in caplog.records) + "\n" + str([r.__dict__ for r in caplog.records])
    assert SECRET_PROMPT not in log_blob, "prompt text leaked into logs"
    assert SECRET_OUTPUT not in log_blob, "model output text leaked into logs"

    # 2. No secret in the owner-partitioned summary file.
    path = fresh_collector.write_summary(owner="alice", doc_id="docX", output_root=tmp_path)
    blob = Path(path).read_text(encoding="utf-8")
    assert SECRET_PROMPT not in blob
    assert SECRET_OUTPUT not in blob
    # And the summary is real metadata.
    data = json.loads(blob)
    assert data["owner"] == "alice"
    assert data["total_calls"] == 1


def test_summary_write_requires_owner(fresh_collector, tmp_path):
    with pytest.raises(ValueError):
        fresh_collector.write_summary(owner="", doc_id="d", output_root=tmp_path)
