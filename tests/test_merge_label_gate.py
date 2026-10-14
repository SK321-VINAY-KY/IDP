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


# --------------------------------------------------------------------------- #
# 2. tag union on merge (gate OFF so the two tags actually meet on one node)
# --------------------------------------------------------------------------- #

def test_merged_node_keeps_union_of_tags():
    # Two nodes, SAME value + SAME label (so they fuse even with the gate on),
    # but DIFFERENT schema tags -> merged node must keep BOTH tags (sorted union),
    # not drop the incoming one.
    deltas = {
        1: PageDelta(page_number=1, entities=[_amt("p1", "500", "TotalAmount", 1, "total_hospital_bill_amount")]),
        2: PageDelta(page_number=2, entities=[_amt("p2", "500", "TotalAmount", 2, "claimed_amount")]),
    }
    # same label 'TotalAmount' -> labels compatible; tags differ. With the gate ON a
    # tag mismatch would BLOCK the merge, so test the union with the gate OFF (merge
    # happens) which is the scenario the union handling must cover.
    g = _merge(deltas, [1, 2], flag=False)
    nodes = _nodes_with_value(g, "500")
    assert len(nodes) == 1, "nodes should have fused (gate off)"
    node = nodes[0]
    union = node.properties.get("schema_field_names")
    assert union == ["claimed_amount", "total_hospital_bill_amount"], f"tags not unioned+sorted: {union}"
    # primary single tag stays readable (first sorted) for back-compat
    assert node.properties.get("schema_field_name") in union


def test_resolver_fallback_reads_union_tag():
    # A node whose union contains the field name must be found by the resolver's
    # exact-tag fallback pass even if it's not the primary tag.
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory

    class FakeLLM:
        def resolve_schema_from_graph(self, graph_evidence, schema):
            return schema()  # LLM returns empty -> forces the tag fallback

    g = GraphMemory()
    n = _amt("n1", "500", "TotalAmount", 1, "")
    n.properties["schema_field_names"] = ["claimed_amount", "total_hospital_bill_amount"]
    n.properties["schema_field_name"] = "claimed_amount"
    g.create_node(n)
    out = resolve_schema_from_graph(graph=g, schema=S, llm=FakeLLM())
    # claimed_amount should be filled from the union-tagged node
    assert out.claimed_amount == "500"


# --------------------------------------------------------------------------- #
# 3. contradiction gates: dead property-key checks removed; live gates kept
# --------------------------------------------------------------------------- #

def test_contradiction_gate_retained_and_works_when_keys_populated():
    """DECISION (audit #3): the id_keys / DOB / gender property-key checks are RETAINED,
    not removed. They are inert for the live pipeline (the agent never writes those
    keys) but are correct, defensive, and test-covered: when a caller DOES populate
    them, a conflict must still block the merge. This asserts that behaviour so the
    gate's retention is justified by an exercised path."""
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager

    class S4(BaseModel):
        patient_name: str = Field(default="", description="n")

    mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S4, pipeline_version="v1")
    # conflicting UHID property on same-value nodes -> blocked
    a = GraphNode(id="a", type="Patient", label="Patient", value="John Doe",
                  properties={"uhid": "UHID-123"}, source_pages=[1], confidence=0.9,
                  status="ASSERTED", category="EXPLICIT")
    b = GraphNode(id="b", type="Patient", label="Patient", value="John Doe",
                  properties={"uhid": "UHID-456"}, source_pages=[2], confidence=0.9,
                  status="ASSERTED", category="EXPLICIT")
    has_c, reasons = mgr.check_hard_contradictions(a, b)
    assert has_c and any("CONFLICTING_IDENTIFIERS" in r for r in reasons), reasons
    # conflicting DOB -> blocked
    c = GraphNode(id="c", type="Patient", label="Patient", value="John Doe",
                  properties={"dob": "1950-01-01"}, source_pages=[1], confidence=0.9,
                  status="ASSERTED", category="EXPLICIT")
    d = GraphNode(id="d", type="Patient", label="Patient", value="John Doe",
                  properties={"dob": "1960-02-02"}, source_pages=[2], confidence=0.9,
                  status="ASSERTED", category="EXPLICIT")
    has_c2, reasons2 = mgr.check_hard_contradictions(c, d)
    assert has_c2 and any("CONFLICTING_DOB" in r for r in reasons2), reasons2


def test_contradiction_gate_live_identifier_value_conflict_still_blocks():
    # Two identifier-type nodes with different values must still be a contradiction
    # (this is a LIVE gate that acts on the node value, not a dead property key).
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager

    class S2(BaseModel):
        patient_uhid: str = Field(default="", description="uhid")

    mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S2, pipeline_version="v1")
    a = GraphNode(id="a", type="Identifier", label="uhid", value="111",
                  properties={}, source_pages=[1], confidence=0.9, status="ASSERTED", category="EXPLICIT")
    b = GraphNode(id="b", type="Identifier", label="uhid", value="222",
                  properties={}, source_pages=[2], confidence=0.9, status="ASSERTED", category="EXPLICIT")
    has_c, reasons = mgr.check_hard_contradictions(a, b)
    assert has_c and any("IDENTIFIER" in r for r in reasons), f"identifier value conflict should block: {reasons}"


def test_contradiction_gate_person_name_conflict_still_blocks():
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager

    class S3(BaseModel):
        patient_name: str = Field(default="", description="n")

    mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S3, pipeline_version="v1")
    a = GraphNode(id="a", type="Person", label="patient_name", value="Arun Sharma",
                  properties={}, source_pages=[1], confidence=0.9, status="ASSERTED", category="EXPLICIT")
    b = GraphNode(id="b", type="Person", label="patient_name", value="Ramesh Sharma",
                  properties={}, source_pages=[2], confidence=0.9, status="ASSERTED", category="EXPLICIT")
    has_c, reasons = mgr.check_hard_contradictions(a, b)
    assert has_c and any("PERSON_NAMES" in r for r in reasons), f"distinct names should block: {reasons}"
