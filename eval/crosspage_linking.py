"""
File: eval/crosspage_linking.py
Purpose: Cross-page-linking diagnosis + offline replay (restore-linking task).

Budget discipline: --capture makes ONE live 12-page run (~13 Bedrock calls) and
persists the page deltas to a gitignored JSON. Every other mode is OFFLINE and
replays that capture with NO model calls (merge / connectivity / linking A/B).

Ground rules: synthetic/metadata only in reports (counts, ids, types, page sets,
edge types, page distance; NO document values). owner required.

Modes:
  --capture --out <deltas.json>         : ONE live 12p capture -> persist deltas
  --connectivity --deltas <f>           : offline connectivity report (Step 0c)
  --unresolved --deltas <f>             : offline unresolved-ref trace (Step 0a)
  --ab --deltas <f> --out <f>           : offline linking OFF/ON replay (Step 2)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# variance_attribution sets creds + logging rollover on import
from eval.variance_attribution import (
    CaptureRecorder, hook_pipeline_capture, graph_fingerprint,
)
from eval.baseline_compare import ADMIN_OWNER, CHANDER_MD, TARGETED_PAGES, _git_commit_and_status
from scripts.benchmark_concurrency_modes import load_document_metadata, load_markdown_pages

from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent, merge_page_deltas_sorted
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import PageDelta
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.config.settings import settings


def _load_pages(mode: str):
    allp = load_markdown_pages(CHANDER_MD)
    if mode == "12":
        return [p for p in allp if p.get("page_number") in TARGETED_PAGES]
    return list(allp)


def do_capture(out: str) -> None:
    """ONE live 12-page capture; persist deltas (metadata+structure only, no doc text
    is printed — the file is gitignored and used only for offline replay)."""
    pages = _load_pages("12")
    schema, _gt = load_document_metadata(CHANDER_MD)
    client = get_extraction_client()
    rec = CaptureRecorder()
    with hook_pipeline_capture(rec):
        asyncio.run(extract_with_graph_memory_concurrent(
            pages_md=pages, schema=schema, llm=client, owner=ADMIN_OWNER, doc_id=None))
    payload = {p: d.to_dict() for p, d in rec.captured_deltas.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, default=str), encoding="utf-8")
    print(f"CAPTURED {len(payload)} page deltas -> {out} (LIVE calls made this run)")


def _load_deltas(path: str) -> Dict[int, PageDelta]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(p): PageDelta.from_dict(d) for p, d in raw.items()}


def _merge(deltas: Dict[int, PageDelta], schema, link_on=None) -> GraphMemory:
    old = getattr(settings, "graph_crosspage_linking", None)
    if link_on is not None:
        settings.graph_crosspage_linking = link_on
    try:
        mgr = GraphMemoryManager(
            anchor_types=getattr(settings, "graph_anchor_types", None) or [],
            anchor_max_k=getattr(settings, "graph_anchor_max_k", 5),
            schema=schema, pipeline_version="v1",
        )
        asyncio.run(_do_merge(mgr, deltas))
        return mgr.graph
    finally:
        if link_on is not None and old is not None:
            settings.graph_crosspage_linking = old


async def _do_merge(mgr, deltas):
    await merge_page_deltas_sorted(mgr, [PageDelta.from_dict(deltas[p].to_dict()) for p in sorted(deltas)])
    await mgr.reconcile_cross_references()


def _cross_page_edges(graph: GraphMemory):
    """(count, by_type, page_distance_list, meaningful_count) — meaningful = both ends
    share a normalized entity type OR a normalized alias/value."""
    nodes = graph.nodes
    cpe = 0
    by_type: Dict[str, int] = {}
    pdist: List[int] = []
    meaningful = 0
    for e in graph.edges.values():
        s = nodes.get(e.source_node); t = nodes.get(e.target_node)
        if not (s and t):
            continue
        sp = set(s.source_pages or []); tp = set(t.source_pages or [])
        if sp and tp and sp.isdisjoint(tp):
            cpe += 1
            by_type[e.relationship] = by_type.get(e.relationship, 0) + 1
            pdist.append(abs(min(tp) - max(sp)))
            same_type = graph._normalize(s.type) == graph._normalize(t.type)
            same_val = graph._normalize(s.value) == graph._normalize(t.value)
            if same_type or same_val:
                meaningful += 1
    return cpe, by_type, pdist, meaningful


def _connectivity(graph: GraphMemory) -> Dict[str, Any]:
    nodes = graph.nodes
    multi_page = [n for n in nodes.values() if len(set(n.source_pages or [])) > 1]
    # undirected adjacency from edges
    adj: Dict[str, set] = {nid: set() for nid in nodes}
    for e in graph.edges.values():
        if e.source_node in adj and e.target_node in adj:
            adj[e.source_node].add(e.target_node)
            adj[e.target_node].add(e.source_node)
    # connected components (sorted ids for determinism)
    seen = set(); comps = []
    for nid in sorted(nodes):
        if nid in seen:
            continue
        comp = []; dq = deque([nid]); seen.add(nid)
        while dq:
            c = dq.popleft(); comp.append(c)
            for nb in sorted(adj[c]):
                if nb not in seen:
                    seen.add(nb); dq.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    # pages reachable from a page-1 node (via edges)
    page1 = [nid for nid, n in nodes.items() if 1 in (n.source_pages or [])]
    reachable_pages = set()
    if page1:
        start = sorted(page1)[0]
        seen2 = {start}; dq = deque([start])
        while dq:
            c = dq.popleft()
            for p in (nodes[c].source_pages or []):
                reachable_pages.add(p)
            for nb in sorted(adj[c]):
                if nb not in seen2:
                    seen2.add(nb); dq.append(nb)
    # pages reachable via SHARED multi-page nodes alone (no edges) — the latent linking
    page_sets = [set(n.source_pages or []) for n in nodes.values() if n.source_pages]
    return {
        "n_nodes": len(nodes),
        "n_edges": len(graph.edges),
        "nodes_spanning_multi_page": len(multi_page),
        "n_connected_components": len(comps),
        "largest_component_size": len(comps[0]) if comps else 0,
        "pages_reachable_from_page1_via_edges": sorted(reachable_pages),
        "distinct_pages_total": sorted({p for ps in page_sets for p in ps}),
    }


def do_connectivity(deltas_path: str) -> None:
    schema, _gt = load_document_metadata(CHANDER_MD)
    deltas = _load_deltas(deltas_path)
    git = _git_commit_and_status()
    g = _merge(deltas, schema, link_on=None)  # current behavior (no flag needed in Step 0)
    conn = _connectivity(g)
    cpe, by_type, pdist, meaningful = _cross_page_edges(g)
    print(f"CONNECTIVITY (current merge, linking off) commit={git['commit'][:10]} {git['working_tree']}")
    print(f"  nodes={conn['n_nodes']} edges={conn['n_edges']} cross_page_edges={cpe} meaningful={meaningful}")
    print(f"  nodes_spanning_multi_page={conn['nodes_spanning_multi_page']}")
    print(f"  connected_components={conn['n_connected_components']} largest={conn['largest_component_size']}")
    print(f"  pages_reachable_from_page1_via_edges={conn['pages_reachable_from_page1_via_edges']}")
    print(f"  distinct_pages_total={conn['distinct_pages_total']}")
    print(f"  cross_page_edge_types={by_type} page_distances={sorted(pdist)}")


def do_unresolved(deltas_path: str) -> None:
    deltas = _load_deltas(deltas_path)
    total_refs = 0; with_target = 0
    reused = 0
    for p, d in deltas.items():
        total_refs += len(d.unresolved_references)
        for ur in d.unresolved_references:
            if (ur.target_type or "").strip():
                with_target += 1
        for ent in d.entities:
            if ent.properties.get("reused_node_id"):
                reused += 1
    # count REFERS_TO edges already present inside deltas (context/local-resolved)
    delta_ref_edges = sum(1 for d in deltas.values() for e in d.relationships
                          if (e.relationship or "").upper() == "REFERS_TO")
    print("UNRESOLVED-REF TRACE (offline, from capture):")
    print(f"  pages={len(deltas)}")
    print(f"  total unresolved_references across deltas = {total_refs}")
    print(f"  unresolved_references with a target_type = {with_target}")
    print(f"  entities carrying reused_node_id = {reused}")
    print(f"  REFERS_TO edges already in deltas (context/local-resolved) = {delta_ref_edges}")


def do_ab(deltas_path: str, out: str) -> None:
    schema, _gt = load_document_metadata(CHANDER_MD)
    deltas = _load_deltas(deltas_path)
    git = _git_commit_and_status()
    rows = {}
    for link_on in (False, True):
        g = _merge(deltas, schema, link_on=link_on)
        conn = _connectivity(g)
        cpe, by_type, pdist, meaningful = _cross_page_edges(g)
        rows["on" if link_on else "off"] = {
            "cross_page_edges": cpe, "meaningful": meaningful, "by_type": by_type,
            "page_distances": sorted(pdist), "fingerprint": graph_fingerprint(g), **conn,
        }
    report = {"git": git, "results": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for cond in ("off", "on"):
        r = rows[cond]
        print(f"  linking {cond}: cpe={r['cross_page_edges']} meaningful={r['meaningful']} "
              f"components={r['n_connected_components']} largest={r['largest_component_size']} "
              f"multi_page_nodes={r['nodes_spanning_multi_page']} fp={r['fingerprint']}")
    print(f"wrote -> {out} commit={git['commit'][:10]} {git['working_tree']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--connectivity", action="store_true")
    ap.add_argument("--unresolved", action="store_true")
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--deltas", type=str)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()
    if args.capture:
        do_capture(args.out)
    elif args.connectivity:
        do_connectivity(args.deltas)
    elif args.unresolved:
        do_unresolved(args.deltas)
    elif args.ab:
        do_ab(args.deltas, args.out)
    else:
        ap.error("choose a mode")


if __name__ == "__main__":
    main()
