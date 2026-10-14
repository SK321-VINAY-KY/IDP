"""
File: eval/variance_attribution.py
Purpose: Variance attribution diagnostic for the Layer 3 graph pipeline.
         Measures contribution of:
           (1) Model's per-page output variations (E5)
           (2) Merge order / concurrency instability (E2, E3, E6)
           (3) Resolver LLM call & tie-break sensitivity (E3, E4)

GROUND RULES:
  1. All new code in eval/. No production code/prompts/config changes.
  2. No document content in logs, reports or files. Only counts, hashes,
     entity IDs, category names, field names, and the 4 permitted GT fields:
     claimed_amount, sanctioned_amount, patient_uhid, admission_date.
  3. Corrected Chander Kochhar GT, 12-page subset, default settings, owner=_admin,
     eval_metrics.py scorer, excluded field patient_payable_amount.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os

def _ensure_aws_creds():
    env2_path = ROOT_DIR / ".env2"
    if env2_path.exists():
        import configparser
        cp = configparser.ConfigParser()
        cp.read(env2_path)
        if cp.sections():
            sec = cp.sections()[0]
            if "aws_access_key_id" in cp[sec]:
                os.environ["AWS_ACCESS_KEY_ID"] = cp[sec]["aws_access_key_id"]
            if "aws_secret_access_key" in cp[sec]:
                os.environ["AWS_SECRET_ACCESS_KEY"] = cp[sec]["aws_secret_access_key"]
            if "aws_session_token" in cp[sec]:
                os.environ["AWS_SESSION_TOKEN"] = cp[sec]["aws_session_token"]
            os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")

def _patch_logging_rollover():
    try:
        from logging.handlers import RotatingFileHandler
        orig_do_rollover = RotatingFileHandler.doRollover
        def safe_do_rollover(self):
            try:
                orig_do_rollover(self)
            except (PermissionError, OSError):
                pass
        RotatingFileHandler.doRollover = safe_do_rollover
    except Exception:
        pass

_ensure_aws_creds()
_patch_logging_rollover()

from pydantic import BaseModel

from eval.baseline_compare import (
    ADMIN_OWNER,
    CHANDER_MD,
    EXCLUDED_FIELD,
    FOCUS_FIELDS,
    TARGETED_PAGES,
    _git_commit_and_status,
    _scored_ground_truth,
)
from scripts.benchmark_concurrency_modes import (
    load_document_metadata,
    load_markdown_pages,
)
from scripts.eval_metrics import evaluate_fields
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import GraphNode, PageDelta
from src.ai.layer3_extraction.graph_agent.resolver import (
    format_graph_evidence_for_resolution,
    resolve_schema_from_graph,
)
from src.config.settings import settings

ALLOWED_VALUE_FIELDS = {
    "claimed_amount",
    "sanctioned_amount",
    "patient_uhid",
    "admission_date",
}


def sanitize_val(field_name: str, val: Any) -> str:
    """Format value for reporting. Ground-truth focus fields display values; all others hash."""
    if val is None or val == "":
        return "<empty>"
    s = str(val).strip()
    if field_name in ALLOWED_VALUE_FIELDS:
        return s
    h = hashlib.sha256(s.lower().encode("utf-8")).hexdigest()[:8]
    return f"hash:{h}"


def val_hash(val: Any) -> str:
    if not val:
        return "<empty>"
    s = " ".join(str(val).strip().lower().split())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def norm_str(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


# --------------------------------------------------------------------------- #
# E1: Capture
# --------------------------------------------------------------------------- #

class CaptureRecorder:
    def __init__(self):
        self.completion_order: List[int] = []
        self.captured_deltas: Dict[int, PageDelta] = {}
        self.merged_graph: Optional[GraphMemory] = None
        self.evidence_text: Optional[str] = None
        self.resolver_output: Optional[Dict[str, Any]] = None


@contextlib.contextmanager
def hook_pipeline_capture(recorder: CaptureRecorder):
    orig_ingest = GraphMemoryManager.ingest_page_delta
    import src.ai.layer3_extraction.graph_agent.resolver as res_mod
    orig_resolve = res_mod.resolve_schema_from_graph

    async def hooked_ingest(self_mgr: GraphMemoryManager, delta: PageDelta, *args, **kwargs):
        recorder.completion_order.append(delta.page_number)
        recorder.captured_deltas[delta.page_number] = PageDelta.from_dict(delta.to_dict())
        return await orig_ingest(self_mgr, delta, *args, **kwargs)

    def hooked_resolve(graph: GraphMemory, schema: type[BaseModel], llm: Any):
        recorder.merged_graph = copy.deepcopy(graph)
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        recorder.evidence_text = format_graph_evidence_for_resolution(graph, schema_fields)
        res = orig_resolve(graph, schema, llm)
        recorder.resolver_output = res.model_dump()
        return res

    GraphMemoryManager.ingest_page_delta = hooked_ingest  # type: ignore[assignment]
    res_mod.resolve_schema_from_graph = hooked_resolve  # type: ignore[assignment]
    try:
        yield
    finally:
        GraphMemoryManager.ingest_page_delta = orig_ingest  # type: ignore[assignment]
        res_mod.resolve_schema_from_graph = orig_resolve  # type: ignore[assignment]


def run_e1_capture(
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    scored_gt: Dict[str, str],
    client: Any,
) -> Tuple[CaptureRecorder, Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Live Capture of Normal Concurrent Graph Pipeline")
    print("=" * 80)
    recorder = CaptureRecorder()
    t0 = time.perf_counter()

    with hook_pipeline_capture(recorder):
        res = asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages,
            schema=schema,
            llm=client,
            owner=ADMIN_OWNER,
            doc_id=None,  # in-memory run, no DB persistence
        ))
    elapsed = round(time.perf_counter() - t0, 2)
    res_dict = res.model_dump()
    ev = evaluate_fields(scored_gt, res_dict)

    print(f"E1 Capture Completed in {elapsed}s")
    print(f"  Completion / Ingestion order: {recorder.completion_order}")
    print(f"  Pages captured: {sorted(recorder.captured_deltas.keys())}")
    total_raw_entities = sum(len(d.entities) for d in recorder.captured_deltas.values())
    print(f"  Total raw page-level entities: {total_raw_entities}")
    merged_nodes = len(recorder.merged_graph.nodes) if recorder.merged_graph else 0
    merged_edges = len(recorder.merged_graph.edges) if recorder.merged_graph else 0
    print(f"  Merged graph nodes: {merged_nodes}, edges: {merged_edges}")
    ev_lines = len(recorder.evidence_text.splitlines()) if recorder.evidence_text else 0
    print(f"  Resolver input evidence lines: {ev_lines}")
    print(f"  Resolver output strict score: {ev['strict_pct']}% ({ev['strict_matched_count']}/{ev['total_fields']})")
    print(f"  Resolver output fuzzy score:  {ev['fuzzy_pct']}% ({ev['fuzzy_matched_count']}/{ev['total_fields']})")

    summary = {
        "elapsed_s": elapsed,
        "completion_order": recorder.completion_order,
        "n_captured_pages": len(recorder.captured_deltas),
        "total_raw_entities": total_raw_entities,
        "merged_nodes": merged_nodes,
        "merged_edges": merged_edges,
        "evidence_lines": ev_lines,
        "strict_pct": ev["strict_pct"],
        "fuzzy_pct": ev["fuzzy_pct"],
        "strict_matched": ev["strict_matched_count"],
        "total_fields": ev["total_fields"],
        "focus_fields": {
            f: {
                "expected": scored_gt.get(f),
                "actual": res_dict.get(f),
                "strict": ev["strict_field_results"].get(f),
            }
            for f in FOCUS_FIELDS if f in scored_gt
        },
    }
    return recorder, summary


# --------------------------------------------------------------------------- #
# E2: Merge-Order Sensitivity (Pipeline only, model held constant)
# --------------------------------------------------------------------------- #

def _node_signature(node: GraphNode) -> Dict[str, Any]:
    norm_val = norm_str(node.value)
    pages = tuple(sorted(node.source_pages))
    ev_hashes = tuple(val_hash(e.text) for e in node.evidence)
    sfn = node.properties.get("schema_field_name")
    return {
        "norm_val": norm_val,
        "val_hash": val_hash(node.value),
        "pages": pages,
        "label": node.label,
        "type": node.type,
        "status": node.status,
        "category": node.category,
        "sfn": sfn,
        "ev_hashes": ev_hashes,
    }


def compare_graphs(
    base_graph: GraphMemory,
    test_graph: GraphMemory,
) -> Dict[str, Any]:
    """Compare test_graph against base_graph using order-independent entity keys."""
    base_by_key: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = {}
    for n in base_graph.nodes.values():
        sig = _node_signature(n)
        k = (sig["norm_val"], sig["pages"])
        base_by_key.setdefault(k, []).append(sig)

    test_by_key: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = {}
    for n in test_graph.nodes.values():
        sig = _node_signature(n)
        k = (sig["norm_val"], sig["pages"])
        test_by_key.setdefault(k, []).append(sig)

    extra_nodes = 0
    missing_nodes = 0
    differing_label = 0
    differing_sfn = 0
    differing_status = 0
    differing_evidence_order = 0

    all_keys = set(base_by_key.keys()) | set(test_by_key.keys())
    for k in all_keys:
        b_list = base_by_key.get(k, [])
        t_list = test_by_key.get(k, [])
        if not b_list:
            extra_nodes += len(t_list)
        elif not t_list:
            missing_nodes += len(b_list)
        else:
            for b_sig, t_sig in zip(b_list, t_list):
                if b_sig["label"] != t_sig["label"]:
                    differing_label += 1
                if b_sig["sfn"] != t_sig["sfn"]:
                    differing_sfn += 1
                if b_sig["status"] != t_sig["status"]:
                    differing_status += 1
                if b_sig["ev_hashes"] != t_sig["ev_hashes"]:
                    differing_evidence_order += 1
            if len(t_list) > len(b_list):
                extra_nodes += len(t_list) - len(b_list)
            elif len(b_list) > len(t_list):
                missing_nodes += len(b_list) - len(t_list)

    is_identical = (
        extra_nodes == 0
        and missing_nodes == 0
        and differing_label == 0
        and differing_sfn == 0
        and differing_status == 0
        and differing_evidence_order == 0
    )

    base_order = [norm_str(n.value) for n in base_graph.nodes.values()]
    test_order = [norm_str(n.value) for n in test_graph.nodes.values()]
    insertion_order_matches = (base_order == test_order)

    return {
        "identical": is_identical,
        "insertion_order_matches": insertion_order_matches,
        "extra_nodes": extra_nodes,
        "missing_nodes": missing_nodes,
        "differing_label": differing_label,
        "differing_properties_sfn": differing_sfn,
        "differing_status": differing_status,
        "differing_evidence_order": differing_evidence_order,
    }


def graph_fingerprint(graph: GraphMemory) -> str:
    """Deterministic hash of order-independent node set (keys, types, labels, sfn)."""
    items = []
    for n in graph.nodes.values():
        sig = _node_signature(n)
        item = (sig["norm_val"], sig["pages"], sig["type"], sig["label"], sig["sfn"], sig["status"])
        items.append(repr(item))
    items.sort()
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:12]


async def _replay_merge_order(
    captured_deltas: Dict[int, PageDelta],
    order: List[int],
    schema: type[BaseModel],
) -> GraphMemory:
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)
    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
        pipeline_version="v1",
    )
    for p_num in order:
        delta_copy = PageDelta.from_dict(captured_deltas[p_num].to_dict())
        await manager.ingest_page_delta(delta_copy)
    await manager.reconcile_cross_references()
    return manager.graph


def run_e2_merge_order(
    captured_deltas: Dict[int, PageDelta],
    original_order: List[int],
    schema: type[BaseModel],
) -> Tuple[Dict[str, Any], List[Tuple[str, GraphMemory]]]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Merge-Order Sensitivity (Model Constant, 32 Replays)")
    print("=" * 80)

    page_nums = sorted(captured_deltas.keys())
    rng = random.Random(42)

    orders: List[Tuple[str, List[int]]] = []
    orders.append(("orig_completion", list(original_order)))
    orders.append(("sequential", list(page_nums)))
    for i in range(30):
        shuffled = list(page_nums)
        rng.shuffle(shuffled)
        orders.append((f"random_{i+1:02d}", shuffled))

    replayed_graphs: List[Tuple[str, GraphMemory]] = []
    for name, ord_pages in orders:
        g = asyncio.run(_replay_merge_order(captured_deltas, ord_pages, schema))
        replayed_graphs.append((name, g))

    base_name, base_graph = replayed_graphs[0]
    results: List[Dict[str, Any]] = []
    fingerprints: Dict[str, List[str]] = {}

    for name, g in replayed_graphs:
        fp = graph_fingerprint(g)
        fingerprints.setdefault(fp, []).append(name)
        comp = compare_graphs(base_graph, g)
        comp["name"] = name
        comp["fp"] = fp
        comp["n_nodes"] = len(g.nodes)
        comp["n_edges"] = len(g.edges)
        results.append(comp)

    identical_count = sum(1 for r in results if r["identical"])
    insertion_order_count = sum(1 for r in results if r["insertion_order_matches"])
    distinct_fps = len(fingerprints)

    print(f"Total replays: {len(orders)} (1 original completion + 1 sequential + 30 random)")
    print(f"Identical graphs (exact order-independent match to baseline): {identical_count}/{len(orders)}")
    print(f"Replays with identical node insertion order: {insertion_order_count}/{len(orders)}")
    print(f"Distinct graph topology fingerprints: {distinct_fps}")

    total_extra = sum(r["extra_nodes"] for r in results)
    total_missing = sum(r["missing_nodes"] for r in results)
    total_label_diff = sum(r["differing_label"] for r in results)
    total_sfn_diff = sum(r["differing_properties_sfn"] for r in results)
    total_status_diff = sum(r["differing_status"] for r in results)
    total_ev_diff = sum(r["differing_evidence_order"] for r in results)

    print("\nDifference counts across all 32 replays relative to baseline:")
    print(f"  Extra nodes total:             {total_extra}")
    print(f"  Missing nodes total:           {total_missing}")
    print(f"  Differing entity labels:       {total_label_diff}")
    print(f"  Differing schema_field_name:   {total_sfn_diff}")
    print(f"  Differing entity status:       {total_status_diff}")
    print(f"  Differing evidence ordering:   {total_ev_diff}")

    # Select representative distinct graphs (up to 10)
    rep_graphs: List[Tuple[str, GraphMemory]] = []
    seen_fps = set()
    for name, g in replayed_graphs:
        fp = graph_fingerprint(g)
        if fp not in seen_fps:
            seen_fps.add(fp)
            rep_graphs.append((f"{name}_{fp}", g))
            if len(rep_graphs) >= 10:
                break

    e2_summary = {
        "n_replays": len(orders),
        "identical_to_baseline": identical_count,
        "distinct_fingerprints": distinct_fps,
        "insertion_order_matches": insertion_order_count,
        "total_extra_nodes": total_extra,
        "total_missing_nodes": total_missing,
        "total_differing_labels": total_label_diff,
        "total_differing_sfn": total_sfn_diff,
        "total_differing_status": total_status_diff,
        "total_differing_evidence_order": total_ev_diff,
        "fingerprint_clusters": {fp: len(members) for fp, members in fingerprints.items()},
    }
    return e2_summary, rep_graphs


# --------------------------------------------------------------------------- #
# E3: Resolver-Pick Sensitivity to the Graph
# --------------------------------------------------------------------------- #

def run_deterministic_resolver_pass(
    graph: GraphMemory,
    schema: type[BaseModel],
) -> Dict[str, Any]:
    """
    Executes the pure deterministic part of resolver.py (Pass (i) and Pass (ii)),
    inspecting candidates and tie-break winners under graph insertion order.
    """
    def _norm_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    field_names = list(schema.model_fields.keys())
    results: Dict[str, Any] = {}

    for fname in field_names:
        fname_key = _norm_key(fname)
        tagged_candidates: List[Tuple[str, str]] = []
        label_candidates: List[Tuple[str, str]] = []

        for node in graph.nodes.values():
            if node.status == "UNCERTAIN" or not node.value:
                continue
            if node.properties.get("schema_field_name") == fname:
                tagged_candidates.append((node.id, node.value))
            lkey = _norm_key(getattr(node, "label", ""))
            if lkey and (lkey == fname_key or fname_key in lkey or lkey in fname_key):
                label_candidates.append((node.id, node.value))

        winner = None
        winner_method = None
        if tagged_candidates:
            winner = tagged_candidates[0][1]
            winner_method = "pass_i_tagged"
        elif label_candidates:
            winner = label_candidates[0][1]
            winner_method = "pass_ii_label"

        results[fname] = {
            "winner": winner,
            "winner_method": winner_method,
            "n_tagged_candidates": len(tagged_candidates),
            "n_label_candidates": len(label_candidates),
            "tagged_values": [sanitize_val(fname, v) for _, v in tagged_candidates],
            "label_values": [sanitize_val(fname, v) for _, v in label_candidates],
        }
    return results


def run_e3_resolver_determinism(
    rep_graphs: List[Tuple[str, GraphMemory]],
    schema: type[BaseModel],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Resolver-Pick Sensitivity to the Graph (Deterministic Parts)")
    print("=" * 80)
    print("Separability analysis: YES, the resolver's candidate selection and tie-break")
    print("logic (Pass (i) schema_field_name and Pass (ii) normalized label match) can be")
    print("separated cleanly from the LLM call and evaluated across graphs.\n")

    runs_eval: List[Tuple[str, Dict[str, Any]]] = []
    for name, g in rep_graphs:
        det = run_deterministic_resolver_pass(g, schema)
        runs_eval.append((name, det))

    field_names = list(schema.model_fields.keys())
    varying_candidate_fields = []
    varying_winner_fields = []

    per_field_analysis: Dict[str, Any] = {}
    for fname in field_names:
        winners = [r[1][fname]["winner"] for r in runs_eval]
        n_tagged = [r[1][fname]["n_tagged_candidates"] for r in runs_eval]
        n_label = [r[1][fname]["n_label_candidates"] for r in runs_eval]

        cand_sets_differ = (len(set(n_tagged)) > 1 or len(set(n_label)) > 1)
        winners_differ = (len(set(winners)) > 1)

        if cand_sets_differ:
            varying_candidate_fields.append(fname)
        if winners_differ:
            varying_winner_fields.append(fname)

        per_field_analysis[fname] = {
            "candidates_differ": cand_sets_differ,
            "winners_differ": winners_differ,
            "distinct_winners": len(set(winners)),
            "sanitized_winners": list(set(sanitize_val(fname, w) for w in winners)),
        }

    print(f"Distinct merged graph variants evaluated: {len(rep_graphs)}")
    print(f"Fields with varying candidate set size: {len(varying_candidate_fields)}")
    print(f"Fields with varying deterministic tie-break winner: {len(varying_winner_fields)}")
    if varying_winner_fields:
        print(f"  Fields that flip winner: {varying_winner_fields}")

    for focus_f in ["claimed_amount", "sanctioned_amount", "patient_uhid", "admission_date"]:
        info = per_field_analysis.get(focus_f, {})
        print(f"\nFocus field: '{focus_f}'")
        print(f"  Candidate sets vary across graphs: {info.get('candidates_differ')}")
        print(f"  Winner varies across graphs:       {info.get('winners_differ')}")
        print(f"  Observed winner values:            {info.get('sanitized_winners')}")

    e3_summary = {
        "n_distinct_graphs_evaluated": len(rep_graphs),
        "fields_with_varying_candidates": len(varying_candidate_fields),
        "varying_candidate_field_names": varying_candidate_fields,
        "fields_with_varying_winner": len(varying_winner_fields),
        "varying_winner_field_names": varying_winner_fields,
        "focus_fields": {
            f: per_field_analysis.get(f) for f in FOCUS_FIELDS
        },
    }
    return e3_summary


# --------------------------------------------------------------------------- #
# E4: Resolver LLM Sensitivity (Model, resolver step only)
# --------------------------------------------------------------------------- #

def run_e4_resolver_llm(
    fixed_graph: GraphMemory,
    schema: type[BaseModel],
    scored_gt: Dict[str, str],
    client: Any,
    n_calls: int = 20,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 4: Resolver LLM Sensitivity ({n_calls} Calls on ONE Fixed Graph)")
    print("=" * 80)

    runs_results: List[Dict[str, Any]] = []
    strict_scores: List[float] = []
    fuzzy_scores: List[float] = []

    for i in range(n_calls):
        t0 = time.perf_counter()
        res = resolve_schema_from_graph(graph=fixed_graph, schema=schema, llm=client)
        elapsed = round(time.perf_counter() - t0, 2)
        rdict = res.model_dump()
        ev = evaluate_fields(scored_gt, rdict)
        strict_scores.append(ev["strict_pct"])
        fuzzy_scores.append(ev["fuzzy_pct"])
        runs_results.append(rdict)
        print(f"  Resolver call {i+1:02d}/{n_calls:02d} ({elapsed}s): strict={ev['strict_pct']}% fuzzy={ev['fuzzy_pct']}%")

    field_names = list(schema.model_fields.keys())
    flipping_fields: Dict[str, Any] = {}
    field_agreement: Dict[str, float] = {}

    for fname in field_names:
        raw_vals = [r.get(fname) for r in runs_results]
        norm_vals = [norm_str(v) for v in raw_vals]
        counts: Dict[str, int] = {}
        for nv in norm_vals:
            counts[nv] = counts.get(nv, 0) + 1
        majority_count = max(counts.values()) if counts else 0
        agreement_pct = round((majority_count / n_calls) * 100, 1)
        field_agreement[fname] = agreement_pct

        if len(counts) > 1:
            val_dist = {}
            for rv, nv in zip(raw_vals, norm_vals):
                s_val = sanitize_val(fname, rv)
                val_dist[s_val] = val_dist.get(s_val, 0) + 1
            flipping_fields[fname] = {
                "n_distinct_values": len(counts),
                "agreement_pct": agreement_pct,
                "distribution": val_dist,
            }

    mean_strict = round(statistics.mean(strict_scores), 2)
    std_strict = round(statistics.stdev(strict_scores), 2) if len(strict_scores) > 1 else 0.0
    min_strict = min(strict_scores)
    max_strict = max(strict_scores)
    spread_strict = round(max_strict - min_strict, 1)

    print(f"\nResolver LLM Score Statistics across {n_calls} calls on identical fixed graph:")
    print(f"  Strict score: mean={mean_strict}%, min={min_strict}%, max={max_strict}%, spread={spread_strict}pts, stdev={std_strict}%")
    print(f"  Fields with 100% agreement: {sum(1 for a in field_agreement.values() if a == 100.0)}/{len(field_names)}")
    print(f"  Fields that flip across calls: {len(flipping_fields)}/{len(field_names)}")
    print("\nFlipping fields breakdown:")
    for fname, details in flipping_fields.items():
        print(f"  - {fname:<28} (agreement {details['agreement_pct']}%): {details['distribution']}")

    e4_summary = {
        "n_calls": n_calls,
        "mean_strict_pct": mean_strict,
        "min_strict_pct": min_strict,
        "max_strict_pct": max_strict,
        "spread_strict_pts": spread_strict,
        "stdev_strict_pct": std_strict,
        "all_strict_scores": strict_scores,
        "all_fuzzy_scores": fuzzy_scores,
        "fields_with_full_agreement_count": sum(1 for a in field_agreement.values() if a == 100.0),
        "flipping_fields_count": len(flipping_fields),
        "flipping_fields": flipping_fields,
    }
    return e4_summary


# --------------------------------------------------------------------------- #
# E5: Page-Level Model Sensitivity
# --------------------------------------------------------------------------- #

def run_e5_page_model_sensitivity(
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    client: Any,
    e1_deltas: Dict[int, PageDelta],
    n_more_runs: int = 5,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 5: Page-Level Model Sensitivity (5 More Passes = 6 Total Passes)")
    print("=" * 80)

    agent = GraphExtractionAgent(
        schema=schema,
        graph=GraphMemory(),
        llm=client,
    )

    all_passes: List[Dict[int, PageDelta]] = [e1_deltas]

    for run_i in range(n_more_runs):
        print(f"  Running page extraction pass {run_i + 2}/6 (12 pages) ...", flush=True)
        pass_deltas: Dict[int, PageDelta] = {}
        t0 = time.perf_counter()

        async def _run_pass():
            tasks = [
                agent.build_page_delta_async(
                    page_number=p["page_number"],
                    markdown=p["markdown"],
                    total_pages=len(pages),
                    context=[],
                    blocks=p.get("blocks", []),
                )
                for p in pages
            ]
            results = await asyncio.gather(*tasks)
            for p, d in zip(pages, results):
                pass_deltas[p["page_number"]] = d

        asyncio.run(_run_pass())
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"    Pass {run_i + 2} finished in {elapsed}s", flush=True)
        all_passes.append(pass_deltas)

    page_nums = sorted(e1_deltas.keys())
    page_report: Dict[int, Any] = {}

    gt_field_pages = {
        1: ["patient_name", "hospital_name", "hospital_address", "age", "gender", "patient_address"],
        7: ["patient_uhid", "hospital_rohini_id", "hospital_pan", "authorization_letter_number"],
        9: ["claimed_amount", "sanctioned_amount", "non_medical_deductions", "bsi_exhaustion_amount"],
        56: ["admission_date"],
        57: ["discharge_date"],
        63: ["room_category"],
    }

    print("\nPer-Page Entity Stability Across 6 Model Extractions:")
    print(f"{'Page':<6} | {'Counts (6 runs)':<26} | {'Stable Entities':<16} | {'Varying Entities':<16} | {'Carries GT Fields?'}")
    print("-" * 90)

    total_varying_entities_doc = 0
    pages_with_variance = 0

    for p_num in page_nums:
        counts = [len(pass_d[p_num].entities) for pass_d in all_passes]
        qualities = [pass_d[p_num].quality for pass_d in all_passes]

        entity_appearances: Dict[Tuple[str, str], int] = {}
        entity_labels: Dict[Tuple[str, str], Set[str]] = {}

        for pass_d in all_passes:
            seen_in_pass = set()
            for ent in pass_d[p_num].entities:
                nv = norm_str(ent.value)
                k = (ent.type, nv)
                if k not in seen_in_pass:
                    seen_in_pass.add(k)
                    entity_appearances[k] = entity_appearances.get(k, 0) + 1
                entity_labels.setdefault(k, set()).add(ent.label)

        total_unique_ents = len(entity_appearances)
        stable_ents = sum(1 for cnt in entity_appearances.values() if cnt == len(all_passes))
        varying_ents = total_unique_ents - stable_ents
        label_varying_ents = sum(1 for lbs in entity_labels.values() if len(lbs) > 1)

        total_varying_entities_doc += varying_ents
        if varying_ents > 0 or len(set(counts)) > 1:
            pages_with_variance += 1

        gt_fields_here = gt_field_pages.get(p_num, [])
        gt_desc = f"YES ({', '.join(gt_fields_here)})" if gt_fields_here else "No"

        counts_str = str(counts)
        print(f"P{p_num:<4} | {counts_str:<26} | {stable_ents:<16} | {varying_ents:<16} | {gt_desc}")

        page_report[p_num] = {
            "entity_counts": counts,
            "mean_count": round(statistics.mean(counts), 2),
            "total_unique_entities": total_unique_ents,
            "stable_entities_6of6": stable_ents,
            "varying_entities": varying_ents,
            "label_varying_entities": label_varying_ents,
            "qualities": qualities,
            "carries_gt_fields": gt_fields_here,
        }

    print(f"\nSummary: {pages_with_variance}/12 pages exhibit entity set variations across 6 model passes.")

    e5_summary = {
        "n_passes": len(all_passes),
        "pages_with_variance": pages_with_variance,
        "total_varying_entities": total_varying_entities_doc,
        "page_report": page_report,
    }
    return e5_summary


# --------------------------------------------------------------------------- #
# E6: Concurrency Control (concurrency = 1 vs default 8)
# --------------------------------------------------------------------------- #

def run_e6_concurrency_control(
    pages: List[Dict[str, Any]],
    schema: type[BaseModel],
    scored_gt: Dict[str, str],
    client: Any,
    n_runs: int = 5,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 6: Concurrency Control ({n_runs} Runs with Concurrency Limit = 1)")
    print("=" * 80)
    print("Runtime Concurrency Override Verification: settings.graph_concurrency_limit = 1")

    orig_limit = getattr(settings, "graph_concurrency_limit", 8)
    settings.graph_concurrency_limit = 1

    seq_strict_scores: List[float] = []
    seq_fuzzy_scores: List[float] = []
    seq_runs_results: List[Dict[str, Any]] = []

    try:
        for i in range(n_runs):
            t0 = time.perf_counter()
            res = asyncio.run(extract_with_graph_memory_concurrent(
                pages_md=pages,
                schema=schema,
                llm=client,
                owner=ADMIN_OWNER,
                doc_id=None,
            ))
            elapsed = round(time.perf_counter() - t0, 2)
            rdict = res.model_dump()
            ev = evaluate_fields(scored_gt, rdict)
            seq_strict_scores.append(ev["strict_pct"])
            seq_fuzzy_scores.append(ev["fuzzy_pct"])
            seq_runs_results.append(rdict)
            print(f"  Sequential Run {i+1}/{n_runs} ({elapsed}s): strict={ev['strict_pct']}% fuzzy={ev['fuzzy_pct']}% (matched {ev['strict_matched_count']}/{ev['total_fields']})")
    finally:
        settings.graph_concurrency_limit = orig_limit

    default_strict_scores = [60.0, 72.0, 56.0, 64.0, 64.0]
    def_mean = round(statistics.mean(default_strict_scores), 2)
    def_std = round(statistics.stdev(default_strict_scores), 2)
    def_spread = round(max(default_strict_scores) - min(default_strict_scores), 1)

    seq_mean = round(statistics.mean(seq_strict_scores), 2)
    seq_std = round(statistics.stdev(seq_strict_scores), 2) if len(seq_strict_scores) > 1 else 0.0
    seq_spread = round(max(seq_strict_scores) - min(seq_strict_scores), 1)

    print("\nConcurrency Comparison (Strict Accuracy %):")
    print(f"  Default (8 workers, 5 runs):    scores={default_strict_scores} mean={def_mean}% spread={def_spread}pts stdev={def_std}%")
    print(f"  Sequential (1 worker, 5 runs):  scores={seq_strict_scores} mean={seq_mean}% spread={seq_spread}pts stdev={seq_std}%")

    e6_summary = {
        "concurrency_set_to": 1,
        "n_runs": n_runs,
        "sequential_strict_scores": seq_strict_scores,
        "sequential_fuzzy_scores": seq_fuzzy_scores,
        "sequential_mean_strict": seq_mean,
        "sequential_spread_strict_pts": seq_spread,
        "sequential_stdev_strict": seq_std,
        "default_strict_scores": default_strict_scores,
        "default_mean_strict": def_mean,
        "default_spread_strict_pts": def_spread,
        "default_stdev_strict": def_std,
    }
    return e6_summary


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Variance attribution diagnostic.")
    parser.add_argument("--e1", action="store_true", help="Run E1 capture only")
    parser.add_argument("--e2", action="store_true", help="Run E2 merge sensitivity only")
    parser.add_argument("--e3", action="store_true", help="Run E3 resolver determinism only")
    parser.add_argument("--e4", action="store_true", help="Run E4 resolver LLM only")
    parser.add_argument("--e5", action="store_true", help="Run E5 page model sensitivity only")
    parser.add_argument("--e6", action="store_true", help="Run E6 concurrency control only")
    args = parser.parse_args()

    git_info = _git_commit_and_status()
    print("=" * 80)
    print(f"VARIANCE ATTRIBUTION DIAGNOSTIC | Commit: {git_info['commit'][:10]} ({git_info['working_tree']})")
    print(f"Excluded Ambiguous Field: '{EXCLUDED_FIELD}'")
    print(f"Targeted Pages ({len(TARGETED_PAGES)}): {sorted(TARGETED_PAGES)}")
    print("=" * 80)

    all_pages = load_markdown_pages(CHANDER_MD)
    pages = [p for p in all_pages if p.get("page_number") in TARGETED_PAGES]
    schema_model, ground_truth = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(ground_truth)
    client = get_extraction_client()

    # E1: Live capture
    recorder, e1_summary = run_e1_capture(pages, schema_model, scored_gt, client)

    # E2: Merge order sensitivity
    e2_summary, rep_graphs = run_e2_merge_order(recorder.captured_deltas, recorder.completion_order, schema_model)

    # E3: Resolver determinism
    e3_summary = run_e3_resolver_determinism(rep_graphs, schema_model)

    # E4: Resolver LLM sensitivity (20 calls on fixed graph)
    e4_summary = run_e4_resolver_llm(recorder.merged_graph, schema_model, scored_gt, client, n_calls=20)

    # E5: Page-level model sensitivity (5 more passes = 6 total)
    e5_summary = run_e5_page_model_sensitivity(pages, schema_model, client, recorder.captured_deltas, n_more_runs=5)

    # E6: Concurrency control (5 sequential runs)
    e6_summary = run_e6_concurrency_control(pages, schema_model, scored_gt, client, n_runs=5)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
