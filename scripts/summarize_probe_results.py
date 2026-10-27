"""
Summarise probe behaviour from the structured JSON LOG stream.

Per the measurement-path investigation (see the task report, finding F5): per-page
probe decisions are NOT persisted to disk or DB today — PageMetadata.extra["probe"]
is never written to the .md footer, the job result, or storage. The ONLY
runtime-reachable probe signal is the structured JSON log stream (src/utils/logger.py
JSONFormatter -> stdout -> CloudWatch log group /ecs/idp-layer3-app, stream prefix
"web"). So this script consumes a CloudWatch export OR a local logs/pipeline.log,
one JSON object per line.

Events consumed (fields per src/ai/layer1_routing/pipeline.py and
src/ai/layer1_probe/probe.py):
  probe.page_decision : document_id, page_number, probe_route, probe_ocr_lang,
                        probe_script, probe_script_confidence,
                        probe_handwriting_prob, probe_has_table, probe_is_blank,
                        engine_used, confidence, escalated   (+ correlation_id)
                        -- PRIMARY source. engine_used/confidence/escalated are
                        the FINAL post-ladder values; probe_route is the route
                        the probe chose BEFORE the escalation ladder ran.
  probe.lipika_vs_vlm : page_number, lipika_script, lipika_conf, margin,
                        n_windows, vlm_script, agree   (+ correlation_id)
  pipeline.page_complete : page_number, engines_used, confidence, escalated,
                        low_confidence                 (+ correlation_id)
                        -- FALLBACK source for pages-per-route / escalation when
                        no probe.page_decision records are present.
  probe.failed : document_id, error                    (+ correlation_id)

Grouping caveat: probe.lipika_vs_vlm carries NO document_id — pages are tied to a
document only via correlation_id (one per pipeline execution). We group by
correlation_id and fall back to a single synthetic document when it is absent.
probe.page_decision DOES carry document_id, but to join it against the VLM events
we still group by correlation_id (the only key probe.lipika_vs_vlm has).

Metrics printed (and, where a metric cannot be computed from what is stored, the
script SAYS SO rather than inventing a number):
  - pages per PROBE route (from probe.page_decision.probe_route when present;
    else by FINAL engine from pipeline.page_complete). The FINAL engine_used
    distribution is also shown (engine->route via the inverse of _ROUTE_TO_ENGINE).
  - escalation rate per route (from probe.page_decision.escalated when present;
    else from the 'escalated' flag on pipeline.page_complete)
  - VLM probe calls per document (count of probe.lipika_vs_vlm with vlm_script set,
    joined to a document by correlation_id — probe.lipika_vs_vlm has no document_id)
  - lipika-vs-VLM agree/disagree
  - with --labels CSV (doc,script): per-script accuracy (of probe.page_decision
    probe_script when present, else the detected script from probe.lipika_vs_vlm)
    and the HARMFUL COUNT = labelled NON-Latin pages whose FINAL engine is a
    Paddle engine EVEN AFTER escalation. HARMFUL must be 0; if >0 the offending
    rows are printed and the script exits NON-ZERO.
  - near-misses (reported, NOT harmful): labelled NON-Latin pages the PROBE
    ROUTED to Paddle but that ESCALATED AWAY to a non-Paddle final engine.
    Reported separately; does NOT affect the exit code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

# Engine -> route, the inverse of _ROUTE_TO_ENGINE (pipeline.py:48-55). Mirrored
# here so the summariser does not import the heavy pipeline module.
_ENGINE_TO_ROUTE = {
    "docling": "digital",
    "paddleocr_printed": "scanned",
    "paddleocr_handwritten": "handwritten",
    "vlm_transcribe": "vlm_transcribe",
    "skip": "skip",
}

# Paddle engines a non-Latin page must NOT be routed to (pipeline.py:51-52).
_PADDLE_ENGINES = {"paddleocr_printed", "paddleocr_handwritten"}

# Scripts considered Latin for the harmful check. Everything else is non-Latin.
_LATIN_SCRIPTS = {"latin", "latn", "english", "en"}

_NOT_AVAILABLE = "not available from the logs provided"


def _unwrap_filter_log_events(obj: dict) -> Optional[list[dict]]:
    """If `obj` is a `filter-log-events` page ({"events": [{"message": ...}, ...]}),
    parse each event's "message" (itself a JSON log line) and return the list of
    parsed log objects. Events whose "message" is not valid JSON are dropped here
    and surface to the caller as skips (see read_log_records). Returns None if
    `obj` is not a filter-log-events page."""
    events = obj.get("events")
    if not isinstance(events, list):
        return None
    out: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        if not isinstance(msg, str):
            continue
        try:
            parsed = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def read_log_records(path: Path) -> tuple[list[dict], int]:
    """Read `--logs` in any of three auto-detected formats and return
    (records, skipped). `skipped` counts input lines that were NOT valid JSON.

    Formats (auto-detected per non-blank line):
      (a) NDJSON          — one JSON log object per line (a bare log dict).
      (b) filter-log-events — the raw JSON of `aws logs filter-log-events`,
          {"events": [{"message": "<json log line>", ...}], ...}; each event's
          "message" is itself a JSON string that is parsed into a log object.
      (c) concatenated pages — several filter-log-events JSON objects, one per
          line (e.g. multiple `--next-token` pages appended to one file).

    A whole filter-log-events page may also be pretty-printed across many lines;
    to support that, if line-by-line parsing yields nothing but the whole file is
    a single JSON object/array, the file is parsed as a whole as a fallback.

    Invalid-JSON lines are skipped and counted. An events-page is counted as ONE
    line regardless of how many messages it carries; malformed inner messages are
    not added to the skip count (they are dropped silently inside the page)."""
    records: list[dict] = []
    skipped = 0
    any_line_parsed = False
    raw = path.read_text(encoding="utf-8")

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        any_line_parsed = True
        _ingest_json_obj(obj, records)

    # Fallback: a pretty-printed (multi-line) single JSON doc that the per-line
    # pass could not parse. Only attempt when NOTHING parsed line-by-line.
    if not any_line_parsed and raw.strip():
        try:
            whole = json.loads(raw)
        except json.JSONDecodeError:
            whole = None
        if whole is not None:
            skipped = 0  # the per-line skips were just fragments of one JSON doc
            _ingest_json_obj(whole, records)

    return records, skipped


def _ingest_json_obj(obj, records: list[dict]) -> None:
    """Append log records from a parsed JSON value: a filter-log-events page
    (unwrap its messages), a bare log dict, or a list of either."""
    if isinstance(obj, list):
        for item in obj:
            _ingest_json_obj(item, records)
        return
    if not isinstance(obj, dict):
        return
    unwrapped = _unwrap_filter_log_events(obj)
    if unwrapped is not None:
        records.extend(unwrapped)
    else:
        records.append(obj)


def iter_log_records(path: Path) -> Iterable[dict]:
    """Backward-compatible NDJSON reader: yield parsed JSON objects, one per
    non-blank line, skipping bad lines. Retained for callers/tests that expect a
    simple generator; new code should use read_log_records (multi-format +
    skip count)."""
    records, _ = read_log_records(path)
    yield from records


def _doc_key(rec: dict) -> str:
    return rec.get("correlation_id") or rec.get("document_id") or "__single__"


def _engine_route(engine: str) -> str:
    return _ENGINE_TO_ROUTE.get(engine, engine)


def _is_non_latin(script: Optional[str]) -> bool:
    if not script:
        return False
    return script.strip().lower() not in _LATIN_SCRIPTS


def load_labels(path: Path) -> dict[str, str]:
    """CSV with columns doc,script (header optional) -> {doc: script}."""
    labels: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            doc, script = row[0].strip(), row[1].strip()
            if doc.lower() == "doc" and script.lower() == "script":
                continue  # header
            labels[doc] = script
    return labels


def summarize(records: list[dict]) -> dict:
    """Compute the log-derived metrics. Returns a result dict (no printing).

    probe.page_decision is the PRIMARY source: when present for a (doc, page) it
    supplies the probe route, the FINAL engine_used/escalated, and the probe's
    own script. pipeline.page_complete is retained as the FALLBACK for
    pages-per-route / escalation so logs without probe.page_decision still work.
    """
    vlm_calls_per_doc: Counter = Counter()
    agree = 0
    disagree = 0
    saw_page_complete = False
    saw_escalated_field = False
    saw_page_decision = False

    # ── probe.page_decision (PRIMARY) keyed by (doc, page) ──────────────────
    # pd_route: the probe's pre-ladder route; pd_final_engine: post-ladder final
    # engine; pd_escalated: post-ladder escalation flag; pd_script: probe script.
    pd_route: dict[tuple[str, int], str] = {}
    pd_final_engine: dict[tuple[str, int], str] = {}
    pd_escalated: dict[tuple[str, int], bool] = {}
    pd_script: dict[tuple[str, int], str] = {}

    # ── pipeline.page_complete (FALLBACK) keyed by (doc, page) ──────────────
    pc_final_engine: dict[tuple[str, int], str] = {}
    pc_escalated: dict[tuple[str, int], bool] = {}

    # (doc, page) -> detected script from probe.lipika_vs_vlm (fallback script).
    lv_script: dict[tuple[str, int], str] = {}
    docs: set[str] = set()

    for rec in records:
        event = rec.get("event")
        doc = _doc_key(rec)
        docs.add(doc)

        if event == "probe.page_decision":
            saw_page_decision = True
            pn = rec.get("page_number")
            if pn is None:
                continue
            key = (doc, int(pn))
            route = rec.get("probe_route")
            if route is not None:
                pd_route[key] = route
            # FINAL adopted engine, 3-level fallback for back-compat with older
            # log lines: prefer the explicit `final_engine` field; else the last
            # entry of `engines_used`; else the legacy `engine_used`. NOTE: with
            # the current pipeline, `engine_used` is the PRIMARY (engines_used[0])
            # and is NOT the adopted final engine — it is used here ONLY as the
            # last-resort fallback for rows predating `final_engine`/`engines_used`.
            final_engine = rec.get("final_engine")
            if final_engine is None:
                engines_list = rec.get("engines_used")
                if isinstance(engines_list, list) and engines_list:
                    final_engine = engines_list[-1]
            if final_engine is None:
                final_engine = rec.get("engine_used")
            if final_engine is not None:
                pd_final_engine[key] = final_engine
            if "escalated" in rec:
                pd_escalated[key] = bool(rec.get("escalated"))
            script = rec.get("probe_script")
            if script:
                pd_script[key] = script

        elif event == "pipeline.page_complete":
            saw_page_complete = True
            engines = rec.get("engines_used") or []
            final_engine = engines[-1] if engines else "unknown"
            pn = rec.get("page_number")
            if pn is not None:
                key = (doc, int(pn))
                pc_final_engine[key] = final_engine
                if "escalated" in rec:
                    saw_escalated_field = True
                    pc_escalated[key] = bool(rec.get("escalated"))

        elif event == "probe.lipika_vs_vlm":
            vlm_script = rec.get("vlm_script")
            if vlm_script is not None:
                vlm_calls_per_doc[doc] += 1
                if rec.get("agree"):
                    agree += 1
                else:
                    disagree += 1
            pn = rec.get("page_number")
            if pn is not None:
                detected = vlm_script or rec.get("lipika_script")
                if detected:
                    lv_script[(doc, int(pn))] = detected

    # ── Derive the reported structures, preferring probe.page_decision ──────
    # pages_per_route: PROBE route (pd_route) when any page_decision exists,
    # else fall back to the final-engine route from pipeline.page_complete.
    pages_per_route: Counter = Counter()
    if saw_page_decision:
        for route in pd_route.values():
            pages_per_route[route] += 1
    else:
        for engine in pc_final_engine.values():
            pages_per_route[_engine_route(engine)] += 1

    # Final engine_used distribution (route via _engine_route), always shown.
    # Prefer page_decision's final engine, else page_complete's.
    final_engine_by_page: dict[tuple[str, int], str] = dict(pc_final_engine)
    final_engine_by_page.update(pd_final_engine)  # page_decision wins on overlap
    pages_per_final_route: Counter = Counter()
    for engine in final_engine_by_page.values():
        pages_per_final_route[_engine_route(engine)] += 1

    # escalation rate per route: keyed to the PROBE route when page_decision
    # exists, else to the final-engine route from page_complete.
    escalated_per_route: Counter = Counter()
    route_total: Counter = Counter()
    if saw_page_decision:
        for key, route in pd_route.items():
            route_total[route] += 1
            if pd_escalated.get(key):
                escalated_per_route[route] += 1
    else:
        for key, engine in pc_final_engine.items():
            route = _engine_route(engine)
            route_total[route] += 1
            if pc_escalated.get(key):
                saw_escalated_field = True
                escalated_per_route[route] += 1

    # page_engine for the harmful check: FINAL engine (page_decision preferred).
    page_engine: dict[tuple[str, int], str] = dict(final_engine_by_page)

    # page_script for accuracy: probe.page_decision.probe_script wins, else the
    # detected script from probe.lipika_vs_vlm.
    page_script: dict[tuple[str, int], str] = dict(lv_script)
    page_script.update(pd_script)

    return {
        "pages_per_route": dict(pages_per_route),
        "pages_per_final_route": dict(pages_per_final_route),
        "escalated_per_route": dict(escalated_per_route),
        "route_total": dict(route_total),
        "vlm_calls_per_doc": dict(vlm_calls_per_doc),
        "agree": agree,
        "disagree": disagree,
        "saw_page_complete": saw_page_complete,
        "saw_page_decision": saw_page_decision,
        "saw_escalated_field": saw_escalated_field if not saw_page_decision else True,
        "page_engine": page_engine,
        "page_script": page_script,
        # per-(doc,page): the probe's pre-ladder route + its post-ladder escalated
        # flag, used by the near-miss metric.
        "pd_route": pd_route,
        "pd_escalated": pd_escalated,
        "docs": docs,
    }


def harmful_rows(summary: dict, labels: dict[str, str]) -> list[dict]:
    """Rows where a labelled NON-Latin doc ran a Paddle engine on some page."""
    rows: list[dict] = []
    for (doc, page), engine in summary["page_engine"].items():
        label = labels.get(doc)
        if label is None:
            continue
        if _is_non_latin(label) and engine in _PADDLE_ENGINES:
            rows.append({"doc": doc, "page": page, "label": label, "engine": engine})
    return sorted(rows, key=lambda r: (r["doc"], r["page"]))


def near_miss_rows(summary: dict, labels: dict[str, str]) -> list[dict]:
    """Near-misses (reported, NOT harmful): labelled NON-Latin pages the PROBE
    ROUTED to a Paddle engine but which ESCALATED AWAY to a non-Paddle final
    engine. Requires probe.page_decision (pd_route/pd_escalated); empty when
    only pipeline.page_complete is available (the probe route is unknown there).
    """
    rows: list[dict] = []
    pd_route = summary.get("pd_route", {})
    pd_escalated = summary.get("pd_escalated", {})
    final_engine = summary["page_engine"]
    for key, route in pd_route.items():
        doc, page = key
        label = labels.get(doc)
        if label is None or not _is_non_latin(label):
            continue
        if route not in _PADDLE_ENGINES:
            continue
        if not pd_escalated.get(key):
            continue
        final = final_engine.get(key)
        if final is not None and final not in _PADDLE_ENGINES:
            rows.append({"doc": doc, "page": page, "label": label,
                         "probe_route": route, "final_engine": final})
    return sorted(rows, key=lambda r: (r["doc"], r["page"]))


def script_accuracy(summary: dict, labels: dict[str, str]) -> Optional[dict]:
    """Per-script accuracy, or None if labels cannot be joined (no doc id in logs).

    The join is (doc, page) -> detected script vs the doc's labelled script.
    Returns None when NONE of the labelled docs appear in the probe logs (the
    'cannot join' case the brief asks us to report instead of inventing).
    """
    per_script_total: Counter = Counter()
    per_script_correct: Counter = Counter()
    joined_any = False
    for (doc, _page), detected in summary["page_script"].items():
        label = labels.get(doc)
        if label is None:
            continue
        joined_any = True
        per_script_total[label] += 1
        if detected.strip().lower() == label.strip().lower():
            per_script_correct[label] += 1
    if not joined_any:
        return None
    return {
        s: {
            "total": per_script_total[s],
            "correct": per_script_correct[s],
            "accuracy": round(per_script_correct[s] / per_script_total[s], 4),
        }
        for s in per_script_total
    }


def render(summary: dict, labels: Optional[dict[str, str]]) -> tuple[str, int]:
    """Build the text report and the intended exit code (non-zero iff HARMFUL>0)."""
    lines: list[str] = []
    exit_code = 0

    lines.append("=== Probe results summary (from logs) ===")

    _primary = summary.get("saw_page_decision")
    _src = ("probe.page_decision (primary)" if _primary
            else "pipeline.page_complete final engine (fallback)")

    lines.append(f"\nPages per PROBE route [{_src}]:")
    if summary["saw_page_decision"] or summary["saw_page_complete"]:
        for route, n in sorted(summary["pages_per_route"].items()):
            lines.append(f"  {route}: {n}")
    else:
        lines.append(f"  {_NOT_AVAILABLE} "
                     "(no probe.page_decision and no pipeline.page_complete events)")

    lines.append("\nPages per FINAL engine (route of engine_used):")
    if summary.get("pages_per_final_route"):
        for route, n in sorted(summary["pages_per_final_route"].items()):
            lines.append(f"  {route}: {n}")
    else:
        lines.append(f"  {_NOT_AVAILABLE} "
                     "(no probe.page_decision and no pipeline.page_complete events)")

    _esc_src = ("probe.page_decision.escalated (primary)" if _primary
                else "pipeline.page_complete.escalated (fallback)")
    lines.append(f"\nEscalation rate per route [{_esc_src}]:")
    if not (summary["saw_page_decision"] or summary["saw_page_complete"]):
        lines.append(f"  {_NOT_AVAILABLE} "
                     "(no probe.page_decision and no pipeline.page_complete events)")
    elif not summary["saw_escalated_field"]:
        lines.append("  not available: no 'escalated' field in the events provided")
    else:
        for route, total in sorted(summary["route_total"].items()):
            esc = summary["escalated_per_route"].get(route, 0)
            rate = round(esc / total, 4) if total else 0.0
            lines.append(f"  {route}: {esc}/{total} = {rate}")

    lines.append("\nVLM probe calls per document "
                 "(from probe.lipika_vs_vlm, joined by correlation_id):")
    if summary["vlm_calls_per_doc"]:
        for doc, n in sorted(summary["vlm_calls_per_doc"].items()):
            lines.append(f"  {doc}: {n}")
    else:
        lines.append("  0 (no probe.lipika_vs_vlm events with a vlm_script)")

    total_cmp = summary["agree"] + summary["disagree"]
    lines.append("\nlipika-vs-VLM agreement:")
    if total_cmp:
        lines.append(f"  agree={summary['agree']}  disagree={summary['disagree']}  "
                     f"(of {total_cmp} compared)")
    else:
        lines.append("  not available (no probe.lipika_vs_vlm events with a vlm_script)")

    if labels is not None:
        lines.append("\nPer-script accuracy (detected vs label):")
        acc = script_accuracy(summary, labels)
        if acc is None:
            lines.append("  cannot join labels (no matching doc id/correlation_id in probe logs)")
        else:
            for script, st in sorted(acc.items()):
                lines.append(f"  {script}: {st['correct']}/{st['total']} = {st['accuracy']}")

        rows = harmful_rows(summary, labels)
        # Wording kept stable for existing tooling; semantics: the FINAL engine
        # (post-escalation) of a labelled non-Latin page is a Paddle engine.
        lines.append(f"\nHARMFUL (labelled non-Latin page routed to a Paddle engine): {len(rows)}")
        if rows:
            exit_code = 1
            for r in rows:
                lines.append(f"  doc={r['doc']} page={r['page']} label={r['label']} "
                             f"engine={r['engine']}")

        near = near_miss_rows(summary, labels)
        lines.append(f"\nnear-misses (reported, NOT harmful) — non-Latin pages the "
                     f"probe ROUTED to Paddle but that escalated away: {len(near)}")
        if near:
            for r in near:
                lines.append(f"  doc={r['doc']} page={r['page']} label={r['label']} "
                             f"probe_route={r['probe_route']} final_engine={r['final_engine']}")
        elif not summary.get("saw_page_decision"):
            lines.append("  not available: near-miss requires probe.page_decision "
                         "(the pre-ladder probe route is unknown from pipeline.page_complete)")

    return "\n".join(lines), exit_code


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True,
                    help="log file: NDJSON (one JSON object per line), raw "
                         "`aws logs filter-log-events` JSON ({\"events\":[{\"message\":...}]}), "
                         "or several such pages concatenated. Format auto-detected; "
                         "non-JSON lines are skipped and counted.")
    ap.add_argument("--labels", help="CSV of doc,script for accuracy + harmful check")
    args = ap.parse_args(argv)

    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"error: logs file not found: {logs_path}", file=sys.stderr)
        return 2

    records, skipped = read_log_records(logs_path)
    summary = summarize(records)
    labels = load_labels(Path(args.labels)) if args.labels else None

    report, exit_code = render(summary, labels)
    print(report)
    print(f"\nInput lines skipped (not valid JSON): {skipped}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
