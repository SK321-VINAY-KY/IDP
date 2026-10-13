"""
File: eval/baseline_compare.py
Purpose: MEASUREMENT-ONLY baseline comparison of extraction arms on the Chander
         Kochhar ground-truth document, plus coverage-only measurement on resume
         fixtures. This is a standalone evaluation script: it imports and calls
         existing production entry points UNCHANGED and adds NO behavior to the
         production pipeline, resolver, or prompts. The only new extraction logic
         here is Arm C (single-call), which reuses the production Bedrock client's
         own `_call_converse` so it is auto-instrumented and consistent.

Arms
----
  A  Current graph pipeline     -> extract_with_graph_memory_concurrent (same entry
                                   point as the earlier 3-run baseline), owner="_admin".
  B  page_scan                  -> extract_by_page_scan (existing per-page scan path),
                                   used as-is.
  C  Single-call                -> ONE structured-output Converse call with the whole
                                   targeted-12-page markdown + the schema field list,
                                   a minimal neutral prompt, high maxTokens (so output
                                   truncation can't confound it), stopReason reported.
  D  Larger-maxTokens graph     -> SKIPPED. Raising graph-extraction maxTokens is not a
                                   pure standalone-eval config value: extract_graph_from_page
                                   applies a hardcoded floor max(params["max_tokens"],4000)
                                   in bedrock_client, and the configurable value lives in the
                                   shared production prompt_config.yaml. Changing either is a
                                   production code/shared-config change, which this task forbids.
                                   See the printed note and the final report.

Scoring
-------
  Chander: strict + fuzzy via scripts.eval_metrics.evaluate_fields. The single flagged-
           ambiguous ground-truth field (patient_payable_amount, _needs_human_decision)
           is EXCLUDED from the ground truth for EVERY arm before scoring.
  Resumes: COVERAGE ONLY (fraction of schema fields non-empty per arm). No accuracy claim.

Instrumentation
----------------
  Each run resets the process-global metrics collector, sets the run context, executes
  the arm, then reads the collector summary (metadata only: stopReason distribution,
  max_tokens stops, parse failures, empty results, token distributions). Arm A also
  writes an owner-partitioned summary file via write_summary. Content is never logged.

Cost
----
  GPT-OSS-120B on Bedrock pricing used: $0.15 / 1M input tokens, $0.60 / 1M output tokens
  (metatext.io and futureagi.com agree exactly, 2026-09). Some calculators list a higher
  $0.18 / $0.72 (likely GovCloud/markup); noted in the report, not used here.

Usage
-----
  Credentials/region must already be in the environment (see repo .env2 loader pattern).
  EXTRACTION_BACKEND=bedrock must be set.

    python -m eval.baseline_compare --runs 3 --out dataset_output/baseline_compare.json
    python -m eval.baseline_compare --runs 3 --arms A,B,C --out <path>
    python -m eval.baseline_compare --resumes-only --out <path>

  --runs N        runs per arm (default 3). Extend to 5 externally by passing --runs 5
                  when two arms land within the observed 11.6-point variance band
                  (the script prints a within-variance advisory after the Chander arms).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pydantic import BaseModel

from src.adapters.llm.bedrock_client import (
    _extract_text_from_converse_response,
    _strip_fences,
)
from src.adapters.llm.extraction_factory import get_extraction_client
from src.adapters.llm import llm_metrics as m
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


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATASET_OUT = ROOT_DIR / "dataset_output"
CHANDER_MD = DATASET_OUT / "Chander Kochhar 01_compressed 2.md"

# Same targeted 12-page subset used by the earlier 52-64% baseline.
TARGETED_PAGES = {1, 5, 6, 7, 9, 10, 17, 56, 57, 63, 68, 75}

# The single flagged-ambiguous ground-truth field, excluded from every arm.
EXCLUDED_FIELD = "patient_payable_amount"

# Known-hard fields called out explicitly in the per-field report.
FOCUS_FIELDS = [
    "patient_uhid",
    "claimed_amount",
    "sanctioned_amount",
    "admission_date",
]

# Resume fixtures: coverage only (no ground-truth accuracy claim).
RESUME_STEMS = ["165", "166", "167", "168", "170", "Vinay_Resume"]

# GPT-OSS-120B on Bedrock, 2026-09 (metatext.io + futureagi.com agree).
PRICE_IN_PER_1M = 0.15
PRICE_OUT_PER_1M = 0.60

# Observed variance band from the earlier baseline (strict %, points).
VARIANCE_BAND_PTS = 11.6

ADMIN_OWNER = "_admin"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _git_commit_and_status() -> Dict[str, str]:
    """Record HEAD commit + working-tree dirtiness for reproducibility."""
    def _run(args: List[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
            ).decode("utf-8", "replace").strip()
        except Exception:
            return ""
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "working_tree": "dirty" if status else "clean",
        "status_lines": status.count("\n") + 1 if status else 0,
    }


def _targeted_pages(all_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in all_pages if p.get("page_number") in TARGETED_PAGES]


def _scored_ground_truth(ground_truth: Dict[str, str]) -> Dict[str, str]:
    """Ground truth with the flagged-ambiguous field removed, for scoring."""
    return {k: v for k, v in ground_truth.items() if k != EXCLUDED_FIELD}


def _coverage(result_dict: Dict[str, Any], schema: type[BaseModel]) -> Dict[str, Any]:
    """Fraction of schema fields that are non-empty in the result."""
    field_names = list(schema.model_fields.keys())
    non_empty = [
        k for k in field_names
        if str(result_dict.get(k, "") or "").strip() not in ("", "None")
    ]
    total = len(field_names)
    pct = round((len(non_empty) / total) * 100, 1) if total else 0.0
    return {
        "total_fields": total,
        "non_empty_fields": len(non_empty),
        "coverage_pct": pct,
        "non_empty_field_names": non_empty,
    }


def _usage_totals(summary: Dict[str, Any], collector: m.MetricsCollector,
                  owner: Optional[str], doc_id: Optional[str]) -> Dict[str, Any]:
    """Aggregate input/output tokens + cost from the raw records of this run."""
    recs = collector.records_for(owner=owner, doc_id=doc_id)
    in_tok = sum(r.input_tokens or 0 for r in recs)
    out_tok = sum(r.output_tokens or 0 for r in recs)
    cost = (in_tok / 1_000_000.0) * PRICE_IN_PER_1M + (out_tok / 1_000_000.0) * PRICE_OUT_PER_1M
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "estimated_cost_usd": round(cost, 6),
    }


# --------------------------------------------------------------------------- #
# Arm C: single-call extraction (new code, reuses production client)
# --------------------------------------------------------------------------- #

def run_arm_c_single_call(
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    client: Any,
    doc_id: str,
    max_tokens: int = 8000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Single structured-output call: whole targeted markdown + schema field list in ONE
    Converse call. Minimal, neutral prompt (NOT tuned against the ground truth). Reuses
    the production client's _call_converse so the call is recorded by llm_metrics with
    purpose 'single_call' and parsed exactly like the production extract() path.
    Returns (result_dict, call_info{stop_reason}).
    """
    schema_fields = [
        {"name": k, "description": (v.description or k)}
        for k, v in schema.model_fields.items()
    ]
    field_lines = "\n".join(f"- {f['name']}: {f['description']}" for f in schema_fields)
    field_names = [f["name"] for f in schema_fields]

    # Minimal, neutral instruction. No field-specific hints, no GT-derived cues.
    system_prompt = (
        "You extract structured data from a document. "
        "Return ONLY a single JSON object with exactly these keys; "
        "use an empty string for any value not present in the document. "
        "Do not add commentary.\n\n"
        "Fields:\n" + field_lines
    )

    full_markdown = "\n\n".join(
        p.get("markdown", "") for p in pages if p.get("markdown")
    )

    resp = client._call_converse(
        messages=[{"role": "user", "content": [{"text": full_markdown}]}],
        system=system_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        _purpose="single_call",
        _owner=ADMIN_OWNER,
        _doc_id=doc_id,
    )

    stop_reason = m.stop_reason_from_response(resp)
    raw = _strip_fences(_extract_text_from_converse_response(resp))

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw[start:end + 1])
            except Exception:
                parsed = None

    if parsed is None:
        # parse failure: record it (metadata only) and return empty schema instance
        m.amend_outcome("single_call", m.OUTCOME_PARSE_FAILED)
        result_dict = {k: "" for k in field_names}
    else:
        # Keep only known schema keys; coerce to str to match other arms' dicts.
        result_dict = {
            k: ("" if parsed.get(k) is None else str(parsed.get(k)))
            for k in field_names
        }

    return result_dict, {"stop_reason": stop_reason, "parsed_ok": parsed is not None}


# --------------------------------------------------------------------------- #
# Single-run drivers (one run of one arm)
# --------------------------------------------------------------------------- #

def _fresh_run(owner: Optional[str], doc_id: Optional[str]):
    """Reset the collector + set run context; return the collector."""
    collector = m.get_collector()
    collector.reset()
    m.set_run_context(owner=owner, doc_id=doc_id)
    return collector


def run_chander_arm(
    arm: str,
    run_idx: int,
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    scored_gt: Dict[str, str],
    client: Any,
) -> Dict[str, Any]:
    """Execute ONE run of one Chander arm and return the per-run record.

    IMPORTANT: Arm A (graph pipeline) persists per-page checkpoints keyed by doc_id.
    To keep every run an INDEPENDENT full-pipeline execution (not a cheap resolver-only
    resume off a previous run's checkpoints), each run gets a UNIQUE doc_id and its
    checkpoints are deleted before and after.
    """
    import asyncio

    doc_id = f"chander_arm_{arm.lower()}_run{run_idx}"
    collector = _fresh_run(ADMIN_OWNER, doc_id)
    t0 = time.perf_counter()

    stop_reason = None
    if arm == "A":
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
        graph_out: Dict[str, Any] = {}
        res = asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages, schema=schema, llm=client,
            graph_out=graph_out, doc_id=doc_id, owner=ADMIN_OWNER,
        ))
        result_dict = res.model_dump()
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
    elif arm == "B":
        res = extract_by_page_scan(pages_md=pages, schema=schema, llm=client)
        result_dict = res.model_dump()
    elif arm == "C":
        result_dict, call_info = run_arm_c_single_call(
            pages, schema, client, doc_id=doc_id)
        stop_reason = call_info["stop_reason"]
    else:
        raise ValueError(f"Unknown arm: {arm}")

    latency_s = round(time.perf_counter() - t0, 3)

    ev = evaluate_fields(scored_gt, result_dict)
    summary = collector.summarize(owner=ADMIN_OWNER, doc_id=doc_id)
    usage = _usage_totals(summary, collector, ADMIN_OWNER, doc_id)

    m.clear_run_context()

    # Per-field correctness (strict + fuzzy) restricted to focus fields for the table.
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
        "doc_id": doc_id,
        "latency_s": latency_s,
        "strict_pct": ev["strict_pct"],
        "fuzzy_pct": ev["fuzzy_pct"],
        "strict_matched": ev["strict_matched_count"],
        "fuzzy_matched": ev["fuzzy_matched_count"],
        "scored_fields": ev["total_fields"],
        "stop_reason": stop_reason,
        "max_tokens_stops": summary["max_tokens_stops"],
        "parse_failures": summary["parse_failures"],
        "empty_result_calls": summary["empty_result_calls"],
        "throttling_exceptions": summary["throttling_exceptions"],
        "total_calls": summary["total_calls"],
        "stop_reason_distribution": summary["stop_reason_distribution"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "focus_fields": focus,
        "strict_field_results": ev["strict_field_results"],
        "fuzzy_field_results": ev["field_results"],
        "result_dict": result_dict,
    }


def run_resume_arm(
    arm: str,
    stem: str,
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    client: Any,
) -> Dict[str, Any]:
    """Execute ONE resume fixture under one arm; coverage only (no accuracy)."""
    import asyncio

    doc_id = f"resume_{stem}_arm_{arm.lower()}"
    collector = _fresh_run(ADMIN_OWNER, doc_id)
    t0 = time.perf_counter()
    stop_reason = None

    if arm == "A":
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
        graph_out: Dict[str, Any] = {}
        res = asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages, schema=schema, llm=client,
            graph_out=graph_out, doc_id=doc_id, owner=ADMIN_OWNER,
        ))
        result_dict = res.model_dump()
        try:
            delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
        except Exception:
            pass
    elif arm == "B":
        res = extract_by_page_scan(pages_md=pages, schema=schema, llm=client)
        result_dict = res.model_dump()
    elif arm == "C":
        result_dict, call_info = run_arm_c_single_call(
            pages, schema, client, doc_id=doc_id)
        stop_reason = call_info["stop_reason"]
    else:
        raise ValueError(f"Unknown arm: {arm}")

    latency_s = round(time.perf_counter() - t0, 3)
    cov = _coverage(result_dict, schema)
    summary = collector.summarize(owner=ADMIN_OWNER, doc_id=doc_id)
    usage = _usage_totals(summary, collector, ADMIN_OWNER, doc_id)
    m.clear_run_context()

    return {
        "arm": arm,
        "stem": stem,
        "latency_s": latency_s,
        "coverage_pct": cov["coverage_pct"],
        "non_empty_fields": cov["non_empty_fields"],
        "total_fields": cov["total_fields"],
        "stop_reason": stop_reason,
        "max_tokens_stops": summary["max_tokens_stops"],
        "parse_failures": summary["parse_failures"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _agg(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def summarize_chander_arm(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    strict = [r["strict_pct"] for r in runs]
    fuzzy = [r["fuzzy_pct"] for r in runs]
    lat = [r["latency_s"] for r in runs]
    cost = [r["estimated_cost_usd"] for r in runs]
    out_tok = [r["output_tokens"] for r in runs]
    in_tok = [r["input_tokens"] for r in runs]
    return {
        "n_runs": len(runs),
        "strict_pct": _agg(strict),
        "fuzzy_pct": _agg(fuzzy),
        "latency_s": _agg(lat),
        "estimated_cost_usd": _agg(cost),
        "input_tokens": _agg(in_tok),
        "output_tokens": _agg(out_tok),
        "max_tokens_stops_total": sum(r["max_tokens_stops"] for r in runs),
        "parse_failures_total": sum(r["parse_failures"] for r in runs),
        "empty_result_calls_total": sum(r["empty_result_calls"] for r in runs),
        "throttling_exceptions_total": sum(r["throttling_exceptions"] for r in runs),
        "total_calls_total": sum(r["total_calls"] for r in runs),
    }


def within_variance_advisory(arm_summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    """Flag any pair of arms whose mean strict% lands within the variance band."""
    advisories: List[str] = []
    arms = list(arm_summaries.keys())
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            ma = arm_summaries[a]["strict_pct"]["mean"]
            mb = arm_summaries[b]["strict_pct"]["mean"]
            if abs(ma - mb) <= VARIANCE_BAND_PTS:
                advisories.append(
                    f"Arms {a} ({ma}%) and {b} ({mb}%) differ by "
                    f"{round(abs(ma - mb), 2)}pt <= {VARIANCE_BAND_PTS}pt band: "
                    f"extend to 5 runs/arm for a defensible comparison."
                )
    return advisories


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline extraction-arm comparison (measurement only).")
    ap.add_argument("--runs", type=int, default=3, help="Runs per arm (default 3; use 5 if within variance band).")
    ap.add_argument("--arms", type=str, default="A,B,C", help="Comma-separated arms to run.")
    ap.add_argument("--out", type=str, default=str(DATASET_OUT / "baseline_compare.json"))
    ap.add_argument("--resumes-only", action="store_true", help="Run only the resume coverage pass.")
    ap.add_argument("--skip-resumes", action="store_true", help="Skip the resume coverage pass.")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    git_info = _git_commit_and_status()
    client = get_extraction_client()

    report: Dict[str, Any] = {
        "git": git_info,
        "pricing": {
            "input_per_1m_usd": PRICE_IN_PER_1M,
            "output_per_1m_usd": PRICE_OUT_PER_1M,
            "source": "GPT-OSS-120B Bedrock, metatext.io + futureagi.com (2026-09); "
                      "alternate $0.18/$0.72 seen on some calculators (GovCloud/markup) NOT used.",
        },
        "arm_d_skipped_reason": (
            "Arm D (larger graph-extraction maxTokens) requires a production code or "
            "shared-config change: extract_graph_from_page applies a hardcoded floor "
            "max(params['max_tokens'],4000) in bedrock_client, and the configurable value "
            "lives in the shared production prompt_config.yaml. Neither is a pure "
            "standalone-eval config value, so it is skipped per the task rules."
        ),
        "excluded_field": EXCLUDED_FIELD,
        "variance_band_pts": VARIANCE_BAND_PTS,
        "runs_per_arm": args.runs,
        "arms": arms,
    }

    print("=" * 90)
    print(f"BASELINE COMPARE | commit={git_info['commit'][:10]} ({git_info['branch']}, {git_info['working_tree']})")
    print(f"Arms={arms} runs/arm={args.runs} | excluded field='{EXCLUDED_FIELD}'")
    print(f"Pricing: ${PRICE_IN_PER_1M}/1M in, ${PRICE_OUT_PER_1M}/1M out | Arm D: SKIPPED (not pure config)")
    print("=" * 90)

    # ---- Chander arms ----------------------------------------------------- #
    if not args.resumes_only:
        all_pages = load_markdown_pages(CHANDER_MD)
        pages = _targeted_pages(all_pages)
        schema_model, ground_truth = load_document_metadata(CHANDER_MD)
        scored_gt = _scored_ground_truth(ground_truth)

        print(f"\nChander: {len(pages)} targeted pages | schema fields={len(schema_model.model_fields)} "
              f"| scored GT fields={len(scored_gt)} (excluded '{EXCLUDED_FIELD}')")

        chander: Dict[str, Any] = {
            "targeted_pages": sorted(TARGETED_PAGES),
            "n_targeted_pages": len(pages),
            "schema_field_count": len(schema_model.model_fields),
            "scored_gt_field_count": len(scored_gt),
            "runs": {},
            "summary": {},
        }

        for arm in arms:
            arm_runs: List[Dict[str, Any]] = []
            for i in range(args.runs):
                print(f"  Arm {arm} run {i + 1}/{args.runs} ...", flush=True)
                rec = run_chander_arm(arm, i + 1, pages, schema_model, scored_gt, client)
                arm_runs.append(rec)
                print(f"    strict={rec['strict_pct']}% fuzzy={rec['fuzzy_pct']}% "
                      f"lat={rec['latency_s']}s calls={rec['total_calls']} "
                      f"maxtok_stops={rec['max_tokens_stops']} parse_fail={rec['parse_failures']} "
                      f"cost=${rec['estimated_cost_usd']}", flush=True)
            chander["runs"][arm] = arm_runs
            chander["summary"][arm] = summarize_chander_arm(arm_runs)

        advisories = within_variance_advisory(chander["summary"])
        chander["within_variance_advisories"] = advisories
        if advisories:
            print("\n[WITHIN-VARIANCE ADVISORY]")
            for a in advisories:
                print(f"  - {a}")

        report["chander"] = chander

    # ---- Resume coverage pass -------------------------------------------- #
    if not args.skip_resumes:
        resumes: Dict[str, Any] = {}
        for stem in RESUME_STEMS:
            md_path = DATASET_OUT / f"{stem}.md"
            if not md_path.exists():
                print(f"  [skip] resume {stem}: {md_path.name} not found")
                continue
            r_pages = load_markdown_pages(md_path)
            r_schema, _ = load_document_metadata(md_path)
            resumes[stem] = {"schema_field_count": len(r_schema.model_fields), "by_arm": {}}
            for arm in arms:
                print(f"  Resume {stem} arm {arm} ...", flush=True)
                rec = run_resume_arm(arm, stem, r_pages, r_schema, client)
                resumes[stem]["by_arm"][arm] = rec
                print(f"    coverage={rec['coverage_pct']}% "
                      f"({rec['non_empty_fields']}/{rec['total_fields']}) "
                      f"lat={rec['latency_s']}s cost=${rec['estimated_cost_usd']}", flush=True)
        report["resumes_coverage_only"] = resumes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote report -> {out_path}")


if __name__ == "__main__":
    main()
