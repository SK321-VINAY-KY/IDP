"""
Deterministic merge-order tests (graph-merging determinism task).

Synthetic PageDelta fixtures only — NO document content. Verifies that in concurrent
mode the same set of page deltas produces an identical graph regardless of the order
the workers finish in, that canonical ids are reproducible, that a resumed run equals
a fresh run, and that failed/degraded pages behave as before.
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

import pytest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import GraphNode, GraphEdge, PageDelta, Evidence


# --------------------------------------------------------------------------- #
# Synthetic schema + deltas (no document content)
# --------------------------------------------------------------------------- #

class SynthSchema(BaseModel):
    field_a: str = Field(default="", description="field a")
    field_b: str = Field(default="", description="field b")
    field_c: str = Field(default="", description="field c")


def _node(nid: str, value: str, label: str, page: int, ntype: str = "Value",
          sfn: str = "") -> GraphNode:
    props = {}
    if sfn:
        props["schema_field_name"] = sfn
    return GraphNode(
        id=nid, type=ntype, label=label, value=value,
        properties=props, source_pages=[page], confidence=0.9,
        status="ASSERTED", category="EXPLICIT",
        evidence=[Evidence(page_number=page, text=f"ev-{nid}", confidence=0.9)],
    )


def _delta(page: int, nodes: List[GraphNode], edges: List[GraphEdge] = None) -> PageDelta:
    return PageDelta(page_number=page, entities=nodes, relationships=edges or [])


def make_synthetic_deltas() -> Dict[int, PageDelta]:
    """A small multi-page set with a repeated entity (same normalized value on two
    pages) so the merge path's resolution/fusion is exercised, plus distinct ones."""
    return {
        1: _delta(1, [
            _node("p1_patient", "ALICE EXAMPLE", "patient_name", 1, "Person", "field_a"),
            _node("p1_amt", "100", "requested", 1, "Amount", "field_b"),
        ]),
        5: _delta(5, [
            _node("p5_amt2", "200", "sanctioned", 5, "Amount", "field_c"),
        ]),
        3: _delta(3, [
            # same normalized value as p1_patient -> should fuse into one canonical node
            _node("p3_patient", "Alice Example", "patient_name", 3, "Person", "field_a"),
            _node("p3_org", "ACME CORP", "org", 3, "Organization"),
        ]),
    }


async def _merge_arrival(deltas: Dict[int, PageDelta], arrival_order: List[int],
                         schema=SynthSchema) -> GraphMemory:
    """Simulate the FIXED concurrent extractor: deltas become available in
    `arrival_order` (worker completion order), but the merge phase sorts by page
    number before ingesting. This mirrors merge_page_deltas_sorted in the extractor."""
    from src.ai.layer3_extraction.extractor import merge_page_deltas_sorted
    mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=schema, pipeline_version="v1")
    # fresh copies, delivered in arrival (shuffled) order
    arrived = [PageDelta.from_dict(deltas[p].to_dict()) for p in arrival_order]
    await merge_page_deltas_sorted(mgr, arrived)
    await mgr.reconcile_cross_references()
    return mgr.graph


def _graph_sig(graph: GraphMemory) -> str:
    """Order-INDEPENDENT content signature (sorted): value|pages|type|label|sfn|status."""
    items = []
    for n in graph.nodes.values():
        sfn = n.properties.get("schema_field_name", "")
        items.append("|".join([
            " ".join(n.value.lower().split()),
            ",".join(map(str, sorted(n.source_pages))),
            n.type, n.label, str(sfn), n.status,
        ]))
    items.sort()
    return "\n".join(items)


def _canonical_ids(graph: GraphMemory) -> List[str]:
    return sorted(graph.nodes.keys())


def _node_insertion_order(graph: GraphMemory) -> List[str]:
    return [" ".join(n.value.lower().split()) for n in graph.nodes.values()]


def _evidence_order(graph: GraphMemory) -> Dict[str, List[str]]:
    """Per-canonical-node evidence text order (synthetic ev tags, not doc content)."""
    out = {}
    for n in graph.nodes.values():
        key = " ".join(n.value.lower().split())
        out[key] = [e.text for e in n.evidence]
    return out


# --------------------------------------------------------------------------- #
# The core regression: shuffled arrival orders -> identical graph
# --------------------------------------------------------------------------- #

def test_shuffled_arrival_orders_produce_identical_graph():
    import random
    deltas = make_synthetic_deltas()
    page_nums = list(deltas.keys())

    baseline = asyncio.run(_merge_arrival(deltas, sorted(page_nums)))
    base_sig = _graph_sig(baseline)
    base_ids = _canonical_ids(baseline)
    base_ins = _node_insertion_order(baseline)
    base_ev = _evidence_order(baseline)

    rng = random.Random(1234)
    distinct_sigs = {base_sig}
    distinct_id_sets = {tuple(base_ids)}
    for _ in range(20):
        order = list(page_nums)
        rng.shuffle(order)
        g = asyncio.run(_merge_arrival(deltas, order))
        # content identical regardless of arrival order
        assert _graph_sig(g) == base_sig, f"graph content differs for order {order}"
        # canonical ids reproducible (deterministic, no uuid)
        assert _canonical_ids(g) == base_ids, f"canonical ids differ for order {order}"
        # node insertion order deterministic (sorted by page)
        assert _node_insertion_order(g) == base_ins, f"insertion order differs for order {order}"
        # evidence order per node deterministic
        assert _evidence_order(g) == base_ev, f"evidence order differs for order {order}"
        distinct_sigs.add(_graph_sig(g))
        distinct_id_sets.add(tuple(_canonical_ids(g)))

    assert len(distinct_sigs) == 1, "merge produced more than one distinct graph content"
    assert len(distinct_id_sets) == 1, "canonical id sets are not reproducible"


def test_canonical_ids_contain_no_random_uuid():
    """Deterministic ids must not embed a random uuid hex suffix."""
    import re
    deltas = make_synthetic_deltas()
    g1 = asyncio.run(_merge_arrival(deltas, sorted(deltas.keys())))
    g2 = asyncio.run(_merge_arrival(deltas, sorted(deltas.keys())))
    assert _canonical_ids(g1) == _canonical_ids(g2)
    # repeated identical runs give identical id sets (uuid would differ)
    for nid in g1.nodes.keys():
        # any generated node id must be stable across the two runs
        assert nid in g2.nodes, f"node id {nid} not reproduced in a second run"



# --------------------------------------------------------------------------- #
# End-to-end concurrent path: resume == fresh; failed/degraded unchanged
# --------------------------------------------------------------------------- #

from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent
from src.ai.layer3_extraction import storage as _storage


class FakeGraphLLM:
    """Synthetic LLM: returns a fixed graph-extraction output per page number.
    No document content. `quality_by_page` lets a page be degraded; `raise_pages`
    makes a page's build raise (worker failure)."""
    def __init__(self, outputs_by_page, quality_by_page=None, raise_pages=None):
        self.outputs_by_page = outputs_by_page
        self.quality_by_page = quality_by_page or {}
        self.raise_pages = set(raise_pages or [])

    def extract_graph_from_page(self, *, page_md, schema_fields, existing_nodes,
                                page_number, total_pages):
        if page_number in self.raise_pages:
            raise RuntimeError(f"synthetic worker failure on page {page_number}")
        out = dict(self.outputs_by_page.get(page_number, {"entities": [], "relationships": [],
                                                          "reference_resolutions": []}))
        out["quality"] = self.quality_by_page.get(page_number, "clean")
        return out

    # resolver call used at the end of the concurrent pipeline
    def resolve_schema_from_graph(self, graph_evidence, schema):
        return schema()


def _ent(nid, value, label, sfn=""):
    e = {"id": nid, "type": "Value", "label": label, "value": value,
         "evidence": f"ev-{nid}", "confidence": 0.9, "is_schema_field": bool(sfn)}
    if sfn:
        e["schema_field_name"] = sfn
    return e


def _outputs():
    return {
        1: {"entities": [_ent("p1a", "ALICE EXAMPLE", "patient_name", "field_a"),
                         _ent("p1b", "100", "requested", "field_b")],
            "relationships": [], "reference_resolutions": []},
        2: {"entities": [_ent("p2a", "200", "sanctioned", "field_c")],
            "relationships": [], "reference_resolutions": []},
        3: {"entities": [_ent("p3a", "Alice Example", "patient_name", "field_a")],
            "relationships": [], "reference_resolutions": []},
    }


def _pages(n):
    return [{"markdown": f"page {i} text", "page_number": i} for i in range(1, n + 1)]


def _run_concurrent(llm, pages, owner, doc_id=None):
    graph_out = {}
    import asyncio as _a
    _a.run(extract_with_graph_memory_concurrent(
        pages_md=pages, schema=SynthSchema, llm=llm, owner=owner,
        graph_out=graph_out, doc_id=doc_id,
    ))
    return graph_out


def test_concurrent_fresh_run_is_deterministic(monkeypatch):
    # No doc_id => no DB checkpoints; pure in-memory determinism across repeats.
    sigs = set()
    idsets = set()
    for _ in range(3):
        g = _run_concurrent(FakeGraphLLM(_outputs()), _pages(3), owner="_t")
        snap = g["snapshot"]
        nodes = snap["nodes"] if isinstance(snap, dict) else []
        sig = tuple(sorted((" ".join(str(n.get("value", "")).lower().split()),
                            ",".join(map(str, sorted(n.get("source_pages", [])))))
                           for n in nodes))
        sigs.add(sig)
        idsets.add(tuple(sorted(n.get("id") for n in nodes)))
    assert len(sigs) == 1, "concurrent fresh runs produced different graph content"
    assert len(idsets) == 1, "concurrent fresh runs produced different canonical ids"


def test_resumed_run_equals_fresh_run(monkeypatch, tmp_path):
    # Use a temp sqlite DB so checkpoints persist within the test only.
    monkeypatch.setenv("IDP_DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    # Force storage to re-init against the temp DB.
    import importlib
    importlib.reload(_storage)
    from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent as ex

    owner = "_t"
    # Fresh full run under a doc_id (writes checkpoints for all pages).
    doc_fresh = "doc_fresh"
    _storage.delete_page_checkpoints(doc_fresh, owner=owner)
    g_fresh = {}
    import asyncio as _a
    _a.run(ex(pages_md=_pages(3), schema=SynthSchema, llm=FakeGraphLLM(_outputs()),
              owner=owner, graph_out=g_fresh, doc_id=doc_fresh))

    # Simulate a resume: a new doc_id, pre-seed checkpoints for pages 1 and 3 by
    # running them first, then a full run that recovers 1&3 and builds 2.
    doc_resume = "doc_resume"
    _storage.delete_page_checkpoints(doc_resume, owner=owner)
    # partial run: only pages 1 and 3 available first
    _a.run(ex(pages_md=[{"markdown": "page 1 text", "page_number": 1},
                        {"markdown": "page 3 text", "page_number": 3}],
              schema=SynthSchema, llm=FakeGraphLLM(_outputs()), owner=owner,
              doc_id=doc_resume))
    # resumed full run: pages 1&3 recovered from checkpoint, page 2 built fresh
    g_resume = {}
    _a.run(ex(pages_md=_pages(3), schema=SynthSchema, llm=FakeGraphLLM(_outputs()),
              owner=owner, graph_out=g_resume, doc_id=doc_resume))

    def _sig(go):
        nodes = go["snapshot"]["nodes"]
        return tuple(sorted((" ".join(str(n.get("value","")).lower().split()),
                             ",".join(map(str, sorted(n.get("source_pages", [])))),
                             str(n.get("properties", {}).get("schema_field_name", "")))
                            for n in nodes))
    def _ids(go):
        return tuple(sorted(n.get("id") for n in go["snapshot"]["nodes"]))

    assert _sig(g_resume) == _sig(g_fresh), "resumed run graph content != fresh run"
    assert _ids(g_resume) == _ids(g_fresh), "resumed run canonical ids != fresh run"
    _storage.delete_page_checkpoints(doc_fresh, owner=owner)
    _storage.delete_page_checkpoints(doc_resume, owner=owner)


def test_failed_page_still_recorded_as_failed(monkeypatch):
    llm = FakeGraphLLM(_outputs(), raise_pages={2})
    g = _run_concurrent(llm, _pages(3), owner="_t")
    assert 2 in g["failed_pages"]
    assert g["pages_failed"] == 1
    # the other pages still merged
    assert g["pages_succeeded"] == 2


def test_degraded_page_excluded_from_success(monkeypatch):
    llm = FakeGraphLLM(_outputs(), quality_by_page={2: "degraded"})
    g = _run_concurrent(llm, _pages(3), owner="_t")
    assert g["pages_degraded"] == 1
    assert 2 in g["degraded_pages"]
    # degraded page not counted as succeeded (pages 1 and 3 only)
    assert g["pages_succeeded"] == 2
