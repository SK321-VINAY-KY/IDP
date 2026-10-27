"""Unit tests for scripts/summarize_probe_results.py — synthetic log fixtures,
no AWS/network. Exercises the summary math, the harmful-count non-zero exit, and
the cannot-compute path.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_probe_results as s  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "probe"
LOGS_CLEAN = FIXTURES / "logs_clean.ndjson"
LABELS_CLEAN = FIXTURES / "labels_clean.csv"
LABELS_HARMFUL = FIXTURES / "labels_harmful.csv"
LOGS_PAGE_DECISION = FIXTURES / "logs_page_decision.ndjson"
LABELS_PAGE_DECISION = FIXTURES / "labels_page_decision.csv"
LOGS_FINAL_ENGINE = FIXTURES / "logs_final_engine.ndjson"
LABELS_FINAL_ENGINE = FIXTURES / "labels_final_engine.csv"
FMT_A = FIXTURES / "fmt_a_ndjson.ndjson"
FMT_B = FIXTURES / "fmt_b_filterlogevents.json"
FMT_C = FIXTURES / "fmt_c_concatenated.json"
SCRIPT = ROOT / "scripts" / "summarize_probe_results.py"


@pytest.fixture
def summary():
    records = list(s.iter_log_records(LOGS_CLEAN))
    return s.summarize(records)


def test_pages_per_route(summary):
    # docling->digital (p1), paddleocr_printed->scanned (p2),
    # final engine of p3 is vlm_transcribe (escalation ladder appended it).
    assert summary["pages_per_route"] == {
        "digital": 1,
        "scanned": 1,
        "vlm_transcribe": 1,
    }


def test_escalation_rate(summary):
    assert summary["saw_escalated_field"] is True
    assert summary["escalated_per_route"] == {"vlm_transcribe": 1}


def test_vlm_calls_per_doc(summary):
    # p1,p2,p3 carry a vlm_script; p4 has vlm_script=null -> not counted.
    assert summary["vlm_calls_per_doc"] == {"docA": 3}


def test_agree_disagree(summary):
    assert summary["agree"] == 2
    assert summary["disagree"] == 1


def test_harmful_count_clean(summary):
    labels = s.load_labels(LABELS_CLEAN)  # docA=latin -> nothing harmful
    assert s.harmful_rows(summary, labels) == []


def test_harmful_count_detects_non_latin_on_paddle(summary):
    labels = s.load_labels(LABELS_HARMFUL)  # docA=devanagari
    rows = s.harmful_rows(summary, labels)
    assert len(rows) == 1
    row = rows[0]
    assert row["doc"] == "docA"
    assert row["page"] == 2
    assert row["engine"] == "paddleocr_printed"
    assert row["label"] == "devanagari"


def test_script_accuracy_joins_labels(summary):
    labels = s.load_labels(LABELS_HARMFUL)  # docA=devanagari
    acc = s.script_accuracy(summary, labels)
    # page_script: p1=latin, p2=devanagari, p3=tamil (vlm_script wins),
    # p4=latin (lipika_script, vlm_script null). All 4 joined to label
    # 'devanagari'; only p2 matches.
    assert acc is not None
    assert acc["devanagari"]["total"] == 4
    assert acc["devanagari"]["correct"] == 1
    assert acc["devanagari"]["accuracy"] == round(1 / 4, 4)


def test_script_accuracy_cannot_join_returns_none(summary):
    # Labels for a doc that never appears in the logs -> cannot join.
    labels = {"docZZZ": "latin"}
    assert s.script_accuracy(summary, labels) is None


def test_render_cannot_compute_message_when_no_page_complete():
    # Only probe events, no pipeline.page_complete -> pages/escalation not available.
    records = [
        {"event": "probe.lipika_vs_vlm", "correlation_id": "d", "page_number": 1,
         "lipika_script": "latin", "vlm_script": "latin", "agree": True},
    ]
    summ = s.summarize(records)
    report, code = s.render(summ, labels=None)
    assert code == 0
    assert s._NOT_AVAILABLE in report


def test_cli_clean_exits_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(LOGS_CLEAN),
         "--labels", str(LABELS_CLEAN)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "HARMFUL" in r.stdout
    assert "HARMFUL (labelled non-Latin page routed to a Paddle engine): 0" in r.stdout


def test_cli_harmful_exits_non_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(LOGS_CLEAN),
         "--labels", str(LABELS_HARMFUL)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1, r.stdout
    assert "paddleocr_printed" in r.stdout
    assert "page=2" in r.stdout


# ============================================================ probe.page_decision (item 3)
@pytest.fixture
def pd_summary():
    records = list(s.iter_log_records(LOGS_PAGE_DECISION))
    return s.summarize(records)


def test_pages_per_probe_route(pd_summary):
    # probe_route distribution (PRIMARY source): 3 printed + 1 vlm_transcribe.
    assert pd_summary["saw_page_decision"] is True
    assert pd_summary["pages_per_route"] == {
        "paddleocr_printed": 3,
        "vlm_transcribe": 1,
    }


def test_pages_per_final_route(pd_summary):
    # FINAL engine_used distribution (engine->route): p1,p2 stay printed
    # (paddleocr_printed -> "scanned"); p3 escalated to vlm; p4 vlm ->
    # 2 "scanned", 2 "vlm_transcribe".
    assert pd_summary["pages_per_final_route"] == {
        "scanned": 2,
        "vlm_transcribe": 2,
    }


def test_escalation_rate_per_probe_route(pd_summary):
    # paddleocr_printed route: p2,p3 escalated of p1,p2,p3 -> 2 of 3.
    assert pd_summary["route_total"]["paddleocr_printed"] == 3
    assert pd_summary["escalated_per_route"]["paddleocr_printed"] == 2
    # vlm_transcribe route: p4, not escalated.
    assert pd_summary["escalated_per_route"].get("vlm_transcribe", 0) == 0


def test_harmful_after_escalation_still_paddle(pd_summary):
    labels = s.load_labels(LABELS_PAGE_DECISION)  # docP=devanagari
    rows = s.harmful_rows(pd_summary, labels)
    # p1 (printed, no escalation) and p2 (escalated but STILL paddle) are harmful;
    # p3 escalated AWAY to vlm (not harmful); p4 routed to vlm (not harmful).
    harmful_pages = sorted(r["page"] for r in rows)
    assert harmful_pages == [1, 2]
    p2 = next(r for r in rows if r["page"] == 2)
    assert p2["engine"] == "paddleocr_printed"  # final engine, after escalation


def test_near_miss_escalated_away_not_harmful(pd_summary):
    labels = s.load_labels(LABELS_PAGE_DECISION)
    near = s.near_miss_rows(pd_summary, labels)
    assert len(near) == 1
    assert near[0]["page"] == 3
    assert near[0]["probe_route"] == "paddleocr_printed"
    assert near[0]["final_engine"] == "vlm_transcribe"
    # near-miss page 3 must NOT be counted as harmful
    harmful_pages = {r["page"] for r in s.harmful_rows(pd_summary, labels)}
    assert 3 not in harmful_pages


def test_page_decision_script_accuracy(pd_summary):
    labels = s.load_labels(LABELS_PAGE_DECISION)  # docP=devanagari
    acc = s.script_accuracy(pd_summary, labels)
    # probe_script per page: p1=latin, p2=devanagari, p3=devanagari, p4=telugu.
    # All 4 join to label devanagari; p2,p3 correct -> 2/4.
    assert acc is not None
    assert acc["devanagari"]["total"] == 4
    assert acc["devanagari"]["correct"] == 2
    assert acc["devanagari"]["accuracy"] == round(2 / 4, 4)


def test_vlm_calls_still_from_lipika_vs_vlm(pd_summary):
    # VLM calls remain sourced from probe.lipika_vs_vlm (3 records w/ vlm_script).
    assert pd_summary["vlm_calls_per_doc"] == {"docP": 3}
    assert pd_summary["agree"] == 2
    assert pd_summary["disagree"] == 1


def test_cli_page_decision_harmful_exits_non_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(LOGS_PAGE_DECISION),
         "--labels", str(LABELS_PAGE_DECISION)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1, r.stdout
    assert "HARMFUL (labelled non-Latin page routed to a Paddle engine): 2" in r.stdout
    assert "near-misses (reported, NOT harmful)" in r.stdout
    # the near-miss (page 3) is reported in its own block, count 1
    assert "escalated away: 1" in r.stdout


def test_cli_page_decision_primary_source_labelled_in_output():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(LOGS_PAGE_DECISION)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "probe.page_decision (primary)" in r.stdout
    assert "Pages per FINAL engine" in r.stdout


# ============================================================ final_engine (item 2)
@pytest.fixture
def fe_summary():
    records = list(s.iter_log_records(LOGS_FINAL_ENGINE))
    return s.summarize(records)


def test_final_engine_harmful_and_near_miss(fe_summary):
    """logs_final_engine.ndjson: docF=devanagari.
    p1: probe_route=paddleocr_printed, escalated, final_engine=paddleocr_printed
        (STILL paddle after escalation) -> HARMFUL.
    p2: probe_route=paddleocr_printed, escalated AWAY, final_engine=vlm_transcribe
        -> near-miss, NOT harmful. engine_used (primary) is paddleocr_printed on
        both pages, proving the summariser keys off final_engine, not engine_used.
    """
    labels = s.load_labels(LABELS_FINAL_ENGINE)
    harmful = s.harmful_rows(fe_summary, labels)
    assert len(harmful) == 1
    assert harmful[0]["page"] == 1
    assert harmful[0]["engine"] == "paddleocr_printed"

    near = s.near_miss_rows(fe_summary, labels)
    assert len(near) == 1
    assert near[0]["page"] == 2
    assert near[0]["probe_route"] == "paddleocr_printed"
    assert near[0]["final_engine"] == "vlm_transcribe"
    # the escalated-away page is NOT counted harmful
    assert 2 not in {r["page"] for r in harmful}


def test_final_engine_fallback_to_engines_used_last():
    """No final_engine field but engines_used present -> use engines_used[-1]."""
    records = [
        {"event": "probe.page_decision", "correlation_id": "docX", "document_id": "docX",
         "page_number": 1, "probe_route": "paddleocr_printed", "probe_script": "devanagari",
         "engine_used": "paddleocr_printed",
         "engines_used": ["paddleocr_printed", "vlm_transcribe"], "escalated": True},
    ]
    summ = s.summarize(records)
    assert summ["page_engine"][("docX", 1)] == "vlm_transcribe"
    labels = {"docX": "devanagari"}
    # final engine is vlm (from engines_used[-1]) -> not harmful, near-miss=1
    assert s.harmful_rows(summ, labels) == []
    assert len(s.near_miss_rows(summ, labels)) == 1


def test_final_engine_fallback_to_engine_used():
    """Neither final_engine nor engines_used -> fall back to engine_used
    (back-compat with existing logs_page_decision.ndjson rows)."""
    records = [
        {"event": "probe.page_decision", "correlation_id": "docY", "document_id": "docY",
         "page_number": 1, "probe_route": "paddleocr_printed", "probe_script": "devanagari",
         "engine_used": "paddleocr_printed", "escalated": True},
    ]
    summ = s.summarize(records)
    assert summ["page_engine"][("docY", 1)] == "paddleocr_printed"
    labels = {"docY": "devanagari"}
    assert len(s.harmful_rows(summ, labels)) == 1


def test_cli_final_engine_harmful_exits_non_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(LOGS_FINAL_ENGINE),
         "--labels", str(LABELS_FINAL_ENGINE)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1, r.stdout
    assert "HARMFUL (labelled non-Latin page routed to a Paddle engine): 1" in r.stdout
    assert "escalated away: 1" in r.stdout


def test_cannot_compute_when_no_page_decision_and_no_page_complete():
    # Only probe.lipika_vs_vlm -> pages/escalation + near-miss cannot be computed.
    records = [
        {"event": "probe.lipika_vs_vlm", "correlation_id": "d", "page_number": 1,
         "lipika_script": "latin", "vlm_script": "latin", "agree": True},
    ]
    summ = s.summarize(records)
    assert summ["saw_page_decision"] is False
    assert summ["saw_page_complete"] is False
    report, code = s.render(summ, labels={"d": "devanagari"})
    assert code == 0
    assert s._NOT_AVAILABLE in report
    # near-miss cannot be computed without probe.page_decision
    assert "near-miss requires probe.page_decision" in report


# ============================================================ multi-format --logs input
# All three fixtures carry the SAME two log records (one docling page_complete +
# one latin lipika_vs_vlm for docFmt) in different wire formats, so the parsed
# result must be identical. Each also includes one non-JSON line to exercise the
# skip counter.

def _records_from(path):
    recs, skipped = s.read_log_records(path)
    return recs, skipped


def _assert_docfmt_records(recs):
    events = sorted(r["event"] for r in recs)
    assert events == ["pipeline.page_complete", "probe.lipika_vs_vlm"]
    summ = s.summarize(recs)
    assert summ["pages_per_route"] == {"digital": 1}
    assert summ["vlm_calls_per_doc"] == {"docFmt": 1}
    assert summ["agree"] == 1 and summ["disagree"] == 0


def test_format_a_ndjson():
    recs, skipped = _records_from(FMT_A)
    _assert_docfmt_records(recs)
    assert skipped == 1  # the "not valid json at all" line


def test_format_b_filter_log_events():
    # a single filter-log-events page: 2 events whose "message" is a JSON string,
    # plus one trailing junk line.
    recs, skipped = _records_from(FMT_B)
    _assert_docfmt_records(recs)
    assert skipped == 1  # the "this line is not json" line; the events page = 1 ok line


def test_format_c_concatenated_pages():
    # two filter-log-events pages, one JSON object per line, no junk.
    recs, skipped = _records_from(FMT_C)
    _assert_docfmt_records(recs)
    assert skipped == 0


def test_all_three_formats_parse_equal():
    a, _ = _records_from(FMT_A)
    b, _ = _records_from(FMT_B)
    c, _ = _records_from(FMT_C)
    key = lambda recs: sorted(json.dumps(r, sort_keys=True) for r in recs)
    assert key(a) == key(b) == key(c)


def test_skip_count_printed_in_cli():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(FMT_A)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "Input lines skipped (not valid JSON): 1" in r.stdout


def test_cli_filter_log_events_format():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(FMT_B)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "Input lines skipped (not valid JSON): 1" in r.stdout
