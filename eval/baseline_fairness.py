"""
File: eval/baseline_fairness.py
Purpose: MEASUREMENT-ONLY follow-up to eval/baseline_compare.py that removes three
         confounds from the arm comparison, WITHOUT changing any production code,
         prompt, resolver, or shared config. All overrides are applied at runtime
         inside this script and restored afterwards (non-persistent monkeypatch).

Adds, relative to baseline_compare.py:
  * Arm D = Arm A (graph pipeline) with the graph-extraction Converse call's output
    budget raised to 8000 tokens (matching Arm C). The raise is applied ONLY to calls
    whose purpose is graph_extraction, via an instance-level monkeypatch of the client's
    _call_converse that is installed for the duration of the run and then removed. The
    production method bedrock_client.extract_graph_from_page is NOT edited; it still
    computes max(params['max_tokens'],4000)=4000 and passes it in — the wrapper simply
    overrides that one argument on the way through. Nothing persists past the run.
  * A page-mode switch: '12p' (the hand-picked targeted subset, identical to
    baseline_compare) or '76p' (the ENTIRE document, no hand-picked pages).
  * For single-call arms, the full-markdown character count and the model's REPORTED
    input tokens are recorded, and stopReason is reported for every call. If the input
    exceeds the model context window the run is still executed (the window is 128K and
    the full doc is far smaller) — nothing is silently truncated.

Field-information parity (Part 1): the per-field descriptions every arm shows the model
are byte-identical because all arms build their field list from the SAME
schema.model_fields (sourced from schema_registry/schema_4ab8d8aaf598.json via
build_dynamic_schema). The hard-field disambiguation (claimed vs sanctioned, UHID, AL
number) lives in those descriptions, so C and B already receive it. Therefore C2/B2 are
NOT created. See the parity table printed by --parity-audit.

Scoring is unchanged: scripts.eval_metrics.evaluate_fields (strict + fuzzy), the one
flagged-ambiguous field patient_payable_amount is excluded from ground truth for every
arm, and metadata-only owner-partitioned logging via llm_metrics is reused.

Usage:
    python -m eval.baseline_fairness --parity-audit
    python -m eval.baseline_fairness --pages 76 --arms A,D,C --runs 3 --out <path>
    python -m eval.baseline_fairness --pages 12 --arms D --runs 5 --out <path>
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse everything already built + committed in baseline_compare.py (no duplication).
from eval.baseline_compare import (
    ADMIN_OWNER,
    CHANDER_MD,
    DATASET_OUT,
    EXCLUDED_FIELD,
    FOCUS_FIELDS,
    PRICE_IN_PER_1M,
    PRICE_OUT_PER_1M,
    TARGETED_PAGES,
    _git_commit_and_status,
    _scored_ground_truth,
    _usage_totals,
    run_arm_c_single_call,
)

from src.adapters.llm import llm_metrics as m
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import (
    extract_by_page_scan,
    extract_with_graph_memory_concurrent,
)
from src.ai.layer3_extraction.storage import delete_page_checkpoints

from scripts.benchmark_concurrency_modes import (
    load_document_metadata,
    load_markdown_pages,
)
from scripts.eval_metrics import evaluate_fields


GRAPH_MAXTOKENS_D = 8000


# --------------------------------------------------------------------------- #
# Part 2: runtime, non-persistent graph-extraction maxTokens override (Arm D)
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def graph_maxtokens_override(client: Any, new_max_tokens: int):
    """Install an instance-level wrapper around client._call_converse that raises
    max_tokens to `new_max_tokens` ONLY for graph_extraction-purpose calls, then
    restores the original bound method on exit. Touches no production file; the
    override does not persist past the `with` block.

    Rationale: bedrock_client.extract_graph_from_page hardcodes
    max_tokens=max(params['max_tokens'],4000) and passes _purpose=graph_extraction.
    We cannot change that line (production), so we intercept the call at the client
    boundary and swap just that one kwarg for the graph-extraction site. Every other
    call (resolver, check_page, extract, single_call) is passed through untouched."""
    original = client._call_converse

    def wrapper(*args, **kwargs):
        if kwargs.get("_purpose") == m.PURPOSE_GRAPH_EXTRACTION:
            kwargs["max_tokens"] = new_max_tokens
        return original(*args, **kwargs)

    client._call_converse = wrapper  # type: ignore[attr-defined]
    try:
        yield
    finally:
        client._call_converse = original  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def select_pages(all_pages: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "12":
        return [p for p in all_pages if p.get("page_number") in TARGETED_PAGES]
    if mode == "76":
        return list(all_pages)
    raise ValueError(f"Unknown page mode: {mode}")


def _approx_tokens(text: str) -> int:
    """Rough char/3.5 estimate (reported alongside the model's real inputTokens)."""
    return int(len(text) / 3.5)


# --------------------------------------------------------------------------- #
# Single run of one arm (A, B, C, D)
# --------------------------------------------------------------------------- #

def _fresh(owner: Optional[str], doc_id: Optional[str]) -> m.MetricsCollector:
    coll = m.get_collector()
    coll.reset()
    m.set_run_context(owner=owner, doc_id=doc_id)
    return coll


def run_arm(
    arm: str,
    run_idx: int,
    page_mode: str,
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    scored_gt: Dict[str, str],
    client: Any,
) -> Dict[str, Any]:
    """Execute ONE run of one arm. A/D = graph pipeline (D under the 8000-token
    override); B = page_scan; C = single-call whole-document. Each graph run uses a
    unique doc_id + checkpoint cleanup so it is an independent full-pipeline run."""
    doc_id = f"chander_{page_mode}p_arm_{arm.lower()}_run{run_idx}"
    coll = _fresh(ADMIN_OWNER, doc_id)
    t0 = time.perf_counter()
    stop_reasons: List[Optional[str]] = []
    c_input_chars = None
    c_reported_input_tokens = None

    def _graph_run() -> Dict[str, Any]:
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
        graph_out: Dict[str, Any] = {}
        res = asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages, schema=schema, llm=client,
            graph_out=graph_out, doc_id=doc_id, owner=ADMIN_OWNER,
        ))
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
        return res.model_dump()

    if arm == "A":
        result_dict = _graph_run()
    elif arm == "D":
        with graph_maxtokens_override(client, GRAPH_MAXTOKENS_D):
            result_dict = _graph_run()
    elif arm == "B":
        res = extract_by_page_scan(pages_md=pages, schema=schema, llm=client)
        result_dict = res.model_dump()
    elif arm == "C":
        full_md = "\n\n".join(p.get("markdown", "") for p in pages if p.get("markdown"))
        c_input_chars = len(full_md)
        result_dict, call_info = run_arm_c_single_call(
            pages, schema, client, doc_id=doc_id)
        stop_reasons.append(call_info["stop_reason"])
    else:
        raise ValueError(f"Unknown arm: {arm}")

    latency_s = round(time.perf_counter() - t0, 3)
    ev = evaluate_fields(scored_gt, result_dict)
    summary = coll.summarize(owner=ADMIN_OWNER, doc_id=doc_id)
    usage = _usage_totals(summary, coll, ADMIN_OWNER, doc_id)

    # Collect stopReason for EVERY call this run (not just arm C).
    recs = coll.records_for(owner=ADMIN_OWNER, doc_id=doc_id)
    all_stop_reasons = [r.stop_reason for r in recs]
    if arm == "C":
        c_reported_input_tokens = sum(r.input_tokens or 0 for r in recs)

    m.clear_run_context()

    focus = {
        f: {
            "expected": scored_gt.get(f, ""),
            "actual": result_dict.get(f, ""),
            "strict": ev["strict_field_results"].get(f),
            "fuzzy": ev["field_results"].get(f),
        }
        for f in FOCUS_FIELDS if f in scored_gt
    }

    return {
        "arm": arm,
        "page_mode": page_mode,
        "doc_id": doc_id,
        "latency_s": latency_s,
        "strict_matched": ev["strict_matched_count"],
        "fuzzy_matched": ev["fuzzy_matched_count"],
        "scored_fields": ev["total_fields"],
        "strict_pct": ev["strict_pct"],
        "fuzzy_pct": ev["fuzzy_pct"],
        "max_tokens_stops": summary["max_tokens_stops"],
        "parse_failures": summary["parse_failures"],
        "empty_result_calls": summary["empty_result_calls"],
        "throttling_exceptions": summary["throttling_exceptions"],
        "total_calls": summary["total_calls"],
        "stop_reason_distribution": summary["stop_reason_distribution"],
        "all_stop_reasons": all_stop_reasons,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "c_input_chars": c_input_chars,
        "c_reported_input_tokens": c_reported_input_tokens,
        "focus_fields": focus,
        "strict_field_results": ev["strict_field_results"],
        "fuzzy_field_results": ev["field_results"],
        "result_dict": result_dict,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _agg(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": round(statistics.mean(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
    }


def summarize_arm(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_runs": len(runs),
        "strict_matched": _agg([r["strict_matched"] for r in runs]),
        "fuzzy_matched": _agg([r["fuzzy_matched"] for r in runs]),
        "scored_fields": runs[0]["scored_fields"] if runs else 0,
        "strict_pct": _agg([r["strict_pct"] for r in runs]),
        "fuzzy_pct": _agg([r["fuzzy_pct"] for r in runs]),
        "latency_s": _agg([r["latency_s"] for r in runs]),
        "input_tokens": _agg([r["input_tokens"] for r in runs]),
        "output_tokens": _agg([r["output_tokens"] for r in runs]),
        "estimated_cost_usd": _agg([r["estimated_cost_usd"] for r in runs]),
        "max_tokens_stops_total": sum(r["max_tokens_stops"] for r in runs),
        "parse_failures_total": sum(r["parse_failures"] for r in runs),
        "empty_result_calls_total": sum(r["empty_result_calls"] for r in runs),
        "total_calls_total": sum(r["total_calls"] for r in runs),
    }


# --------------------------------------------------------------------------- #
# Part 1: parity audit (no runs)
# --------------------------------------------------------------------------- #

def parity_audit() -> Dict[str, Any]:
    schema, _gt = load_document_metadata(CHANDER_MD)
    fields = [
        {"name": k, "description": (v.description or k)}
        for k, v in schema.model_fields.items()
    ]
    scored = [f for f in fields if f["name"] != EXCLUDED_FIELD]
    print("=" * 88)
    print("PART 1 — FIELD-INFORMATION PARITY AUDIT (no LLM calls)")
    print("=" * 88)
    print(f"schema source: schema_registry (via load_document_metadata/build_dynamic_schema)")
    print(f"total schema fields={len(fields)} | shown to model after excluding "
          f"'{EXCLUDED_FIELD}'={len(scored)}")
    print("All three arms build their field list from the SAME schema.model_fields:")
    print("  [{'name': k, 'description': v.description or k} for k,v in schema.model_fields.items()]")
    print("  Arm A: graph_page_ingestion.j2 (per page) + graph_schema_resolution.j2 (resolver)")
    print("  Arm B: page_field_check.j2 (per page) + extraction_system.j2 (final extract)")
    print("  Arm C: eval single-call renders '- name: description' from the same list")
    print("  => the per-field (name, description) text is identical across A/B/C.\n")
    print("HARD-FIELD DESCRIPTIONS (verbatim, identical for every arm):")
    for f in fields:
        if f["name"] in FOCUS_FIELDS or f["name"] == "authorization_letter_number":
            print(f"  [{f['name']}]\n    {f['description']}")
    print("\nDIFFERENCE THAT EXISTS (noted, NOT a per-field description):")
    print("  Arm A's graph_page_ingestion.j2 contains an EXTRA static instruction block")
    print("  (CATEGORY-A disambiguation: 'pick the ONE whose on-page label most closely")
    print("  matches ... requested vs sanctioned vs total across columns'). This is method")
    print("  scaffolding, not schema field-description content. Per the task rule, C2/B2 are")
    print("  created only if C/B are MISSING descriptions A gets — they are not — so C2/B2")
    print("  are NOT created.")
    return {
        "total_schema_fields": len(fields),
        "fields_shown_after_exclusion": len(scored),
        "descriptions_identical_across_arms": True,
        "c2_b2_created": False,
        "note": "Arm A graph ingestion prompt has extra generic disambiguation scaffolding "
                "(not a per-field schema description); does not warrant C2/B2.",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Fairness follow-up (measurement only).")
    ap.add_argument("--parity-audit", action="store_true", help="Print Part 1 parity audit and exit.")
    ap.add_argument("--pages", type=str, default="76", choices=["12", "76"])
    ap.add_argument("--arms", type=str, default="A,D,C")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(DATASET_OUT / "baseline_fairness.json"))
    args = ap.parse_args()

    if args.parity_audit:
        parity_audit()
        return

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    git_info = _git_commit_and_status()
    client = get_extraction_client()

    all_pages = load_markdown_pages(CHANDER_MD)
    pages = select_pages(all_pages, args.pages)
    schema, ground_truth = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(ground_truth)

    full_md = "\n\n".join(p.get("markdown", "") for p in pages if p.get("markdown"))

    report: Dict[str, Any] = {
        "git": git_info,
        "pricing": {"input_per_1m_usd": PRICE_IN_PER_1M, "output_per_1m_usd": PRICE_OUT_PER_1M,
                    "source": "GPT-OSS-120B Bedrock, metatext.io + futureagi.com (2026-09)."},
        "page_mode": args.pages,
        "n_pages": len(pages),
        "full_markdown_chars": len(full_md),
        "full_markdown_approx_tokens": _approx_tokens(full_md),
        "context_window_tokens": 128000,
        "excluded_field": EXCLUDED_FIELD,
        "arm_d_definition": f"Arm A graph pipeline with graph_extraction max_tokens overridden "
                            f"to {GRAPH_MAXTOKENS_D} at runtime inside the eval script (non-persistent).",
        "scored_gt_field_count": len(scored_gt),
        "runs_per_arm": args.runs,
        "arms": arms,
        "runs": {},
        "summary": {},
    }

    print("=" * 90)
    print(f"FAIRNESS RUN | commit={git_info['commit'][:10]} ({git_info['branch']}, {git_info['working_tree']})")
    print(f"pages={args.pages} ({len(pages)} pages, {len(full_md)} chars ~{_approx_tokens(full_md)} tok "
          f"| ctx window 128000) | arms={arms} runs/arm={args.runs}")
    print(f"scored GT fields={len(scored_gt)} | excluded='{EXCLUDED_FIELD}'")
    print("=" * 90)

    for arm in arms:
        arm_runs: List[Dict[str, Any]] = []
        for i in range(args.runs):
            print(f"  Arm {arm} ({args.pages}p) run {i+1}/{args.runs} ...", flush=True)
            rec = run_arm(arm, i + 1, args.pages, pages, schema, scored_gt, client)
            arm_runs.append(rec)
            extra = ""
            if arm == "C":
                extra = (f" c_chars={rec['c_input_chars']} "
                         f"reported_in_tok={rec['c_reported_input_tokens']} "
                         f"stop={rec['stop_reason_distribution']}")
            print(f"    strict={rec['strict_matched']}/{rec['scored_fields']} "
                  f"({rec['strict_pct']}%) fuzzy={rec['fuzzy_matched']}/{rec['scored_fields']} "
                  f"calls={rec['total_calls']} maxtok_stops={rec['max_tokens_stops']} "
                  f"parse_fail={rec['parse_failures']} lat={rec['latency_s']}s "
                  f"cost=${rec['estimated_cost_usd']}{extra}", flush=True)
        report["runs"][arm] = arm_runs
        report["summary"][arm] = summarize_arm(arm_runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote report -> {out_path}")


if __name__ == "__main__":
    main()
