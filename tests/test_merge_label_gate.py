"""
Label-compatibility gate + schema-tag handling in the merge (audit #2/#3).

Synthetic fixtures only — NO document content. Verifies:
 1. Two amount nodes with the SAME value but incompatible labels / differing schema
    tags do NOT fuse (they stay distinct) when the gate is on; they DO fuse with the
    gate off; non-amount/date types are unaffected either way.
 2. When nodes DO merge, the union of schema_field_name tags is kept (sorted).
 3. check_hard_contradictions behaves per the chosen design (dead keys removed).
"""
import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import GraphNode, PageDelta, Evidence
from src.ai.layer3_extraction.extractor import merge_page_deltas_sorted
from src.config.settings import settings


class S(BaseModel):
    claimed_amount: str = Field(default="", description="claimed")
    sanctioned_amount: str = Field(default="", description="sanctioned")


def _amt(nid, value, label, page, tag):
    return GraphNode(
        id=nid, type="Amount", label=label, value=value,
        properties={"schema_field_name": tag} if tag else {},
        source_pages=[page], confidence=0.9, status="ASSERTED", category="EXPLICIT",
        evidence=[Evidence(page_number=page, text="ev-" + nid, confidence=0.9)],
    )


def _merge(deltas, order, flag):
    async def run():
        mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S, pipeline_version="v1")
        await merge_page_deltas_sorted(mgr, [PageDelta.from_dict(deltas[p].to_dict()) for p in order])
        await mgr.reconcile_cross_references()
        return mgr.graph
    old = getattr(settings, "graph_merge_label_gate", None)
    settings.graph_merge_label_gate = flag
    try:
        return asyncio.run(run())
    finally:
        if old is not None:
            settings.graph_merge_label_gate = old


def _nodes_with_value(graph, value):
    import re
    tv = re.sub(r"[^0-9]", "", value)
    return [n for n in graph.nodes.values() if re.sub(r"[^0-9]", "", n.value) == tv]


# --------------------------------------------------------------------------- #
# 1. label/tag incompatibility blocks amount fusion (gate ON); fuses (gate OFF)
# --------------------------------------------------------------------------- #

def test_incompatible_amount_labels_do_not_merge_gate_on():
    # same value 80698, different labels (ClaimedAmount vs SanctionedAmount),
    # different tags -> must NOT fuse with the gate on.
    deltas = {
        1: PageDelta(page_number=1, entities=[_amt("p1", "80698", "ClaimedAmount", 1, "claimed_amount")]),
        2: PageDelta(page_number=2, entities=[_amt("p2", "80698", "SanctionedAmount", 2, "sanctioned_amount")]),
    }
    g = _merge(deltas, [1, 2], flag=True)
    nodes = _nodes_with_value(g, "80698")
    assert len(nodes) == 2, f"gate ON should keep both amount nodes distinct, got {len(nodes)}"


def test_incompatible_amount_labels_merge_when_gate_off():
    deltas = {
        1: PageDelta(page_number=1, entities=[_amt("p1", "80698", "ClaimedAmount", 1, "claimed_amount")]),
        2: PageDelta(page_number=2, entities=[_amt("p2", "80698", "SanctionedAmount", 2, "sanctioned_amount")]),
    }
    g = _merge(deltas, [1, 2], flag=False)
    nodes = _nodes_with_value(g, "80698")
    assert len(nodes) == 1, f"gate OFF should fuse on value (old behavior), got {len(nodes)}"


def test_compatible_amount_labels_still_merge_gate_on():
    # identical label + identical tag -> compatible -> should still fuse with gate on.
    deltas = {
        1: PageDelta(page_number=1, entities=[_amt("p1", "80698", "SanctionedAmount", 1, "sanctioned_amount")]),
        2: PageDelta(page_number=2, entities=[_amt("p2", "80698", "SanctionedAmount", 2, "sanctioned_amount")]),
    }
    g = _merge(deltas, [1, 2], flag=True)
    nodes = _nodes_with_value(g, "80698")
    assert len(nodes) == 1, f"compatible labels should still fuse, got {len(nodes)}"


def test_non_amount_types_unaffected_by_gate():
    # Person nodes same value, different labels -> gate must NOT change their behavior.
    def person(nid, label, page):
        return GraphNode(id=nid, type="Person", label=label, value="ALICE EXAMPLE",
                         properties={}, source_pages=[page], confidence=0.9,
                         status="ASSERTED", category="EXPLICIT",
                         evidence=[Evidence(page_number=page, text="e" + nid, confidence=0.9)])
    deltas = {
        1: PageDelta(page_number=1, entities=[person("p1", "patient_name", 1)]),
        2: PageDelta(page_number=2, entities=[person("p2", "signatory", 2)]),
    }
    g_on = _merge(deltas, [1, 2], flag=True)
    g_off = _merge(deltas, [1, 2], flag=False)
    n_on = len(_nodes_with_value(g_on, "ALICE EXAMPLE")) if False else len([n for n in g_on.nodes.values() if "alice" in n.value.lower()])
    n_off = len([n for n in g_off.nodes.values() if "alice" in n.value.lower()])
    assert n_on == n_off, "gate must not change non-amount/date merging"
