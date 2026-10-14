"""
File: eval/merge_determinism_measure.py
Purpose: Step-2 measurement for the merge-determinism fix. Reuses the capture/replay
         machinery from eval/variance_attribution.py.

  (2.1) Replay check (model held constant, in memory only): capture the real
        pipeline's 12 page outputs once, then replay the FIXED merge contract
        (extractor.merge_page_deltas_sorted) under 32 arrival orders. Expected after
        the fix: 1 distinct graph. Reports distinct-graph count and, if >1, what
        still differs by category.

  (2.2) End-to-end: run the real pipeline N times on 12 pages and N times on 76,
        reporting strict mean / min-max in FIELDS/25, the 4 permitted per-field
        results, and the number of distinct merged graphs captured.

GROUND RULES: all new code in eval/; NO document content in logs/report/files
(only counts, hashes, entity ids, category names, field names, and the 4 permitted
GT fields). owner stays required. Corrected Chander GT, excluded field unchanged.

Usage:
    python -m eval.merge_determinism_measure --replay --out dataset_output/md_replay.json
    python -m eval.merge_determinism_measure --e2e --pages 12 --runs 5 --out <path>
    python -m eval.merge_determinism_measure --e2e --pages 76 --runs 5 --out <path>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# variance_attribution runs _ensure_aws_creds()/_patch_logging_rollover() on import
from eval.variance_attribution import (
    CaptureRecorder,
    compare_graphs,
    graph_fingerprint,
    hook_pipeline_capture,
    run_e1_capture,
)
from eval.baseline_compare import (
    ADMIN_OWNER,
    CHANDER_MD,
    EXCLUDED_FIELD,
    FOCUS_FIELDS,
    PRICE_IN_PER_1M,
    PRICE_OUT_PER_1M,
    TARGETED_PAGES,
    _git_commit_and_status,
    _scored_ground_truth,
    _usage_totals,
)
from scripts.benchmark_concurrency_modes import load_document_metadata, load_markdown_pages
from scripts.eval_metrics import evaluate_fields

from src.adapters.llm import llm_metrics as m
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import (
    extract_with_graph_memory_concurrent,
    merge_page_deltas_sorted,
)
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import PageDelta
from src.config.settings import settings


def _load_pages(page_mode: str) -> List[Dict[str, Any]]:
    allp = load_markdown_pages(CHANDER_MD)
    if page_mode == "12":
        return [p for p in allp if p.get("page_number") in TARGETED_PAGES]
    return list(allp)


async def _replay_sorted(captured: Dict[int, PageDelta], arrival_order: List[int], schema):
    """Replay the FIXED merge contract: deltas arrive in arrival_order, merge sorted."""
    mgr = GraphMemoryManager(
        anchor_types=getattr(settings, "graph_anchor_types", None) or [],
        anchor_max_k=getattr(settings, "graph_anchor_max_k", 5),
        schema=schema, pipeline_version="v1",
    )
    arrived = [PageDelta.from_dict(captured[p].to_dict()) for p in arrival_order]
    await merge_page_deltas_sorted(mgr, arrived)
    await mgr.reconcile_cross_references()
    return mgr.graph


async def _merge_with_gate(captured: Dict[int, PageDelta], schema, gate_on: bool):
    """Merge the captured deltas (sorted) with the label gate off/on. Returns the graph."""
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph  # noqa
    old = getattr(settings, "graph_merge_label_gate", True)
    settings.graph_merge_label_gate = gate_on
    try:
        mgr = GraphMemoryManager(
            anchor_types=getattr(settings, "graph_anchor_types", None) or [],
            anchor_max_k=getattr(settings, "graph_anchor_max_k", 5),
            schema=schema, pipeline_version="v1",
        )
        await merge_page_deltas_sorted(mgr, [PageDelta.from_dict(captured[p].to_dict()) for p in sorted(captured)])
        await mgr.reconcile_cross_references()
        return mgr.graph
    finally:
        settings.graph_merge_label_gate = old


def run_ab_replay(out_path: str, n_captures: int = 5, resolver_calls: int = 3) -> Dict[str, Any]:
    """Step 2.1: capture n live runs' page deltas (12p); for each, merge with the gate
    OFF and ON (model held constant), then run the REAL resolver `resolver_calls` times
    on each merged graph. Count per-field correct out of (n_captures * resolver_calls)."""
    import random
    from eval.variance_attribution import graph_fingerprint
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    git_info = _git_commit_and_status()
    pages = _load_pages("12")
    schema, gt = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(gt)
    client = get_extraction_client()

    def _num(v):
        import re as _re
        s = _re.sub(r"[^\d.]", "", str(v or ""))
        try:
            return float(s) if s else None
        except Exception:
            return None

    FIELDS = ["claimed_amount", "sanctioned_amount", "patient_uhid", "admission_date"]
    want = {f: _num(scored_gt.get(f)) if f in ("claimed_amount", "sanctioned_amount") else str(scored_gt.get(f, "")).strip()
            for f in FIELDS}

    captures = []
    for i in range(n_captures):
        print(f"  capture {i+1}/{n_captures} ...", flush=True)
        rec = CaptureRecorder()
        with hook_pipeline_capture(rec):
            asyncio.run(extract_with_graph_memory_concurrent(
                pages_md=pages, schema=schema, llm=client, owner=ADMIN_OWNER, doc_id=None))
        captures.append(rec.captured_deltas)

    def _field_correct(resolved_val, field):
        if field in ("claimed_amount", "sanctioned_amount"):
            rv = _num(resolved_val)
            return rv is not None and want[field] is not None and abs(rv - want[field]) < 1e-6
        return str(resolved_val or "").strip() == want[field] or (
            want[field] and want[field] in str(resolved_val or ""))

    results = {}
    for gate_on in (False, True):
        cond = "on" if gate_on else "off"
        per_field = {f: 0 for f in FIELDS}
        total_calls = 0
        fps = set()
        for cap in captures:
            g = asyncio.run(_merge_with_gate(cap, schema, gate_on))
            fps.add(graph_fingerprint(g))
            for _ in range(resolver_calls):
                res = resolve_schema_from_graph(graph=g, schema=schema, llm=client).model_dump()
                total_calls += 1
                for f in FIELDS:
                    if _field_correct(res.get(f), f):
                        per_field[f] += 1
        results[cond] = {
            "per_field_correct": per_field,
            "total_resolver_calls": total_calls,
            "distinct_merged_graphs": len(fps),
        }
        print(f"  gate {cond}: {per_field} / {total_calls} calls | distinct_graphs={len(fps)}", flush=True)

    report = {"git": git_info, "n_captures": n_captures, "resolver_calls_per_graph": resolver_calls,
              "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nA/B REPLAY commit {git_info['commit'][:10]} {git_info['working_tree']} -> {out_path}")
    return report


def run_replay(out_path: str) -> Dict[str, Any]:
    import random
    git_info = _git_commit_and_status()
    pages = _load_pages("12")
    schema, gt = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(gt)
    client = get_extraction_client()

    # Capture the real pipeline's 12 page outputs once (model held constant after).
    recorder, e1 = run_e1_capture(pages, schema, scored_gt, client)
    captured = recorder.captured_deltas
    page_nums = sorted(captured.keys())

    rng = random.Random(42)
    orders = [("sequential", list(page_nums)),
              ("completion", list(recorder.completion_order))]
    for i in range(30):
        s = list(page_nums); rng.shuffle(s)
        orders.append((f"random_{i+1:02d}", s))

    graphs = [(name, asyncio.run(_replay_sorted(captured, order, schema))) for name, order in orders]
    base = graphs[0][1]
    fps: Dict[str, List[str]] = {}
    diffs = {"extra_nodes": 0, "missing_nodes": 0, "differing_label": 0,
             "differing_properties_sfn": 0, "differing_status": 0, "differing_evidence_order": 0}
    for name, g in graphs:
        fp = graph_fingerprint(g)
        fps.setdefault(fp, []).append(name)
        c = compare_graphs(base, g)
        for k in diffs:
            diffs[k] += c[k]

    distinct = len(fps)
    report = {
        "git": git_info,
        "n_replays": len(orders),
        "distinct_graphs": distinct,
        "fingerprint_clusters": {fp: len(v) for fp, v in fps.items()},
        "difference_totals_vs_baseline": diffs,
        "captured_pages": page_nums,
        "capture_strict_matched": e1["strict_matched"],
        "capture_total_fields": e1["total_fields"],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"REPLAY CHECK (fixed sorted merge) | commit {git_info['commit'][:10]} {git_info['working_tree']}")
    print(f"  replays={len(orders)}  DISTINCT GRAPHS={distinct}  (expected 1)")
    if distinct > 1:
        print(f"  still-differing categories: {diffs}")
    print(f"  wrote -> {out_path}")
    return report


def _e2e_run(client, pages, schema, scored_gt, doc_id):
    coll = m.get_collector(); coll.reset()
    m.set_run_context(owner=ADMIN_OWNER, doc_id=doc_id)
    recorder = CaptureRecorder()
    with hook_pipeline_capture(recorder):
        res = asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages, schema=schema, llm=client, owner=ADMIN_OWNER, doc_id=None))
    ev = evaluate_fields(scored_gt, res.model_dump())
    summ = coll.summarize(owner=ADMIN_OWNER, doc_id=doc_id)
    usage = _usage_totals(summ, coll, ADMIN_OWNER, doc_id)
    m.clear_run_context()
    fp = graph_fingerprint(recorder.merged_graph) if recorder.merged_graph else None
    # Cross-page edges: edges whose source and target nodes have disjoint source_pages
    # (counts only, no content). Measures how much cross-page linking survived.
    cross_page_edges = 0
    g = recorder.merged_graph
    if g is not None:
        for e in g.edges.values():
            s = g.get_node(e.source_node)
            t = g.get_node(e.target_node)
            if s and t:
                sp = set(s.source_pages or [])
                tp = set(t.source_pages or [])
                if sp and tp and sp.isdisjoint(tp):
                    cross_page_edges += 1
    focus = {f: {"expected": scored_gt.get(f), "actual": res.model_dump().get(f),
                 "strict": ev["strict_field_results"].get(f)}
             for f in FOCUS_FIELDS if f in scored_gt}
    return {
        "strict_matched": ev["strict_matched_count"], "total_fields": ev["total_fields"],
        "strict_pct": ev["strict_pct"], "fuzzy_matched": ev["fuzzy_matched_count"],
        "graph_fingerprint": fp, "focus_fields": focus,
        "cross_page_edges": cross_page_edges,
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }


def run_e2e(page_mode: str, runs: int, out_path: str) -> Dict[str, Any]:
    git_info = _git_commit_and_status()
    pages = _load_pages(page_mode)
    schema, gt = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(gt)
    client = get_extraction_client()

    run_recs = []
    for i in range(runs):
        print(f"  e2e {page_mode}p run {i+1}/{runs} ...", flush=True)
        rec = _e2e_run(client, pages, schema, scored_gt, f"md_e2e_{page_mode}_{i+1}")
        run_recs.append(rec)
        print(f"    strict={rec['strict_matched']}/{rec['total_fields']} fp={rec['graph_fingerprint']} "
              f"cost=${rec['estimated_cost_usd']}", flush=True)

    strict = [r["strict_matched"] for r in run_recs]
    fps = {r["graph_fingerprint"] for r in run_recs}
    report = {
        "git": git_info, "page_mode": page_mode, "runs": runs,
        "strict_matched_mean": round(statistics.mean(strict), 2),
        "strict_matched_min": min(strict), "strict_matched_max": max(strict),
        "strict_matched_stdev": round(statistics.stdev(strict), 2) if len(strict) > 1 else 0.0,
        "distinct_graph_fingerprints": len(fps),
        "per_run": run_recs,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nE2E {page_mode}p: strict/25 mean={report['strict_matched_mean']} "
          f"({report['strict_matched_min']}-{report['strict_matched_max']}) "
          f"distinct_graphs={len(fps)} -> {out_path}")
    return report


def _count_stages(all_runs, fld):
    out = {}
    for t in all_runs:
        key = t[fld]["value_lost_at_stage"].split(" ")[0]
        out[key] = out.get(key, 0) + 1
    return out


def _num(v):
    import re as _re
    s = _re.sub(r"[^\d.]", "", str(v or ""))
    try:
        return float(s) if s else None
    except Exception:
        return None


def run_trace(out_path: str) -> Dict[str, Any]:
    """Trace claimed_amount (189771) and sanctioned_amount (80698) through the stages
    on ONE captured run. Reports only ids/labels/schema tags/page numbers/counts and
    the two permitted GT values. No evidence text."""
    git_info = _git_commit_and_status()
    pages = _load_pages("12")
    schema, gt = load_document_metadata(CHANDER_MD)
    scored_gt = _scored_ground_truth(gt)
    client = get_extraction_client()

    targets = {"claimed_amount": _num(scored_gt.get("claimed_amount")),
               "sanctioned_amount": _num(scored_gt.get("sanctioned_amount"))}

    recorder, _e1 = run_e1_capture(pages, schema, scored_gt, client)
    captured = recorder.captured_deltas
    merged = recorder.merged_graph
    resolver_out = recorder.resolver_output or {}

    def _node_features(node):
        """Metadata-only numeric features the resolver can distinguish nodes on."""
        return {
            "node_id": node.id, "label": node.label, "type": node.type,
            "schema_tag": node.properties.get("schema_field_name", ""),
            "pages": sorted(node.source_pages), "n_pages": len(node.source_pages or []),
            "evidence_count": len(node.evidence or []),
            "confidence": round(float(getattr(node, "confidence", 0.0) or 0.0), 3),
            "status": node.status, "category": node.category,
            "n_observations": len((node.properties or {}).get("observations", [])),
            "n_aliases": len(node.aliases or []),
        }

    trace = {}
    for fld, tval in targets.items():
        # Stage 1: page-level extraction — any delta entity carrying the target value
        page_hits = []
        for pnum, delta in captured.items():
            for ent in delta.entities:
                if _num(ent.value) is not None and tval is not None and abs(_num(ent.value) - tval) < 1e-6:
                    page_hits.append({"page": pnum, "node_id": ent.id, "label": ent.label,
                                      "schema_tag": ent.properties.get("schema_field_name", ""),
                                      "type": ent.type})
        # Stage 2: survived merge — canonical node with the target value
        merged_hits = []
        if merged:
            for nid, node in merged.nodes.items():
                if _num(node.value) is not None and tval is not None and abs(_num(node.value) - tval) < 1e-6:
                    merged_hits.append({"node_id": nid, "label": node.label,
                                        "schema_tag": node.properties.get("schema_field_name", ""),
                                        "pages": sorted(node.source_pages), "status": node.status})
        # Stage 3: resolver output chosen value for this field
        chosen_val = resolver_out.get(fld)
        chosen_num = _num(chosen_val)
        chosen_matches = (chosen_num is not None and tval is not None and abs(chosen_num - tval) < 1e-6)
        # which merged node (if any) carries the value the resolver actually chose
        chosen_node = None
        if merged and chosen_num is not None:
            for nid, node in merged.nodes.items():
                if _num(node.value) is not None and abs(_num(node.value) - chosen_num) < 1e-6:
                    chosen_node = {"node_id": nid, "label": node.label,
                                   "schema_tag": node.properties.get("schema_field_name", ""),
                                   "pages": sorted(node.source_pages)}
                    break

        if not page_hits:
            lost = "page_extraction (no page emitted a node with the correct value)"
        elif not merged_hits:
            lost = "merge (correct value present at page level but no canonical node kept it)"
        elif not chosen_matches:
            lost = "resolver (correct canonical node existed but resolver chose a different value)"
        else:
            lost = "none (correct value survives to the resolver output)"

        # Competing-candidate features: the node carrying the EXPECTED value vs the
        # node carrying the value the resolver actually CHOSE (metadata only).
        competing = []
        if merged:
            want_vals = {tval}
            if chosen_num is not None:
                want_vals.add(chosen_num)
            seen_ids = set()
            for nid, node in merged.nodes.items():
                nv = _num(node.value)
                if nv is not None and any(abs(nv - w) < 1e-6 for w in want_vals if w is not None):
                    if nid not in seen_ids:
                        feats = _node_features(node)
                        feats["carries_value"] = nv
                        feats["is_expected"] = (tval is not None and abs(nv - tval) < 1e-6)
                        competing.append(feats)
                        seen_ids.add(nid)

        trace[fld] = {
            "expected_value": tval,
            "stage1_page_hits": page_hits,
            "stage2_merged_hits": merged_hits,
            "stage2_kept_schema_tag": any(h["schema_tag"] == fld for h in merged_hits),
            "stage2_merged_with_different_label": any(
                len({o.get("label") for o in
                     (merged.nodes[h["node_id"]].properties.get("observations", []) if merged and h["node_id"] in merged.nodes else [])
                     if o.get("label")}) > 1
                for h in merged_hits
            ),
            "stage3_resolver_chose_value": chosen_val,
            "stage3_resolver_matches_expected": chosen_matches,
            "stage3_resolver_chosen_node": chosen_node,
            "competing_candidates": competing,
            "value_lost_at_stage": lost,
        }

    report = {"git": git_info, "trace": trace}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    print("TRACE (one captured run)")
    for fld, t in trace.items():
        print(f"  {fld} (expected {t['expected_value']}):")
        print(f"    page hits: {[(h['page'], h['schema_tag']) for h in t['stage1_page_hits']]}")
        print(f"    merged nodes w/ value: {[(h['node_id'], h['schema_tag'], h['pages']) for h in t['stage2_merged_hits']]}")
        print(f"    resolver chose: {t['stage3_resolver_chose_value']} (matches={t['stage3_resolver_matches_expected']})")
        print(f"    => lost at: {t['value_lost_at_stage']}")
    print(f"  wrote -> {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--ab-replay", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-runs", type=int, default=1)
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--pages", type=str, default="12", choices=["12", "76"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.replay:
        run_replay(args.out)
    elif args.ab_replay:
        run_ab_replay(args.out)
    elif args.trace:
        if args.trace_runs <= 1:
            run_trace(args.out)
        else:
            all_runs = []
            for i in range(args.trace_runs):
                print(f"\n### TRACE RUN {i+1}/{args.trace_runs} ###", flush=True)
                rep = run_trace(args.out.replace(".json", f"_run{i+1}.json"))
                all_runs.append(rep["trace"])
            # aggregate counts across runs
            agg = {}
            for fld in ("claimed_amount", "sanctioned_amount"):
                agg[fld] = {
                    "runs": args.trace_runs,
                    "page_had_correct_value_with_tag": sum(
                        1 for t in all_runs if any(h["schema_tag"] == fld for h in t[fld]["stage1_page_hits"])),
                    "survived_merge": sum(1 for t in all_runs if t[fld]["stage2_merged_hits"]),
                    "merged_with_different_label": sum(1 for t in all_runs if t[fld]["stage2_merged_with_different_label"]),
                    "resolver_matched_expected": sum(1 for t in all_runs if t[fld]["stage3_resolver_matches_expected"]),
                    "lost_at_stage_counts": _count_stages(all_runs, fld),
                }
            out = Path(args.out)
            out.write_text(json.dumps({"aggregate": agg, "runs": all_runs}, indent=2, default=str), encoding="utf-8")
            print("\n" + "=" * 70)
            print(f"TRACE AGGREGATE across {args.trace_runs} runs:")
            for fld, a in agg.items():
                print(f"  {fld}: tag@page={a['page_had_correct_value_with_tag']}/{a['runs']} "
                      f"survived_merge={a['survived_merge']}/{a['runs']} "
                      f"resolver_correct={a['resolver_matched_expected']}/{a['runs']} "
                      f"lost_at={a['lost_at_stage_counts']}")
            print(f"  wrote -> {out}")
    elif args.e2e:
        run_e2e(args.pages, args.runs, args.out)
    else:
        ap.error("choose --replay, --trace or --e2e")


if __name__ == "__main__":
    main()
