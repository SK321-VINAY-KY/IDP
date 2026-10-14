"""
Deterministic post-merge cross-page linking pass (restore-linking task, Candidate 2).

Synthetic fixtures only — NO document content. Verifies:
 - same entity on two pages gets a SAME_ENTITY_AS link; distinct entities don't
 - amount/date-semantic nodes are never linked (consistent with the label gate)
 - generic short / pure-digit / stoplist values are excluded
 - fan-out cap respected and deterministic (sorted ids)
 - identical result under shuffled arrival order and PYTHONHASHSEED (seed test is in
   test_merge_determinism_crossproc via the real pipeline; here we check arrival order)
 - derived edges are marked derived with the rule
 - a traversal multi-hop path exists with linking ON and not with it OFF
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
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.extractor import merge_page_deltas_sorted
from src.config.settings import settings


class S(BaseModel):
    patient_name: str = Field(default="", description="n")


def _node(nid, value, label, page, ntype="Person", aliases=None):
    return GraphNode(
        id=nid, type=ntype, label=label, value=value,
        properties={}, source_pages=[page], confidence=0.9, status="ASSERTED",
        category="EXPLICIT", aliases=list(aliases or []),
        evidence=[Evidence(page_number=page, text="ev-" + nid, confidence=0.9)],
    )


def _merge(deltas, order, link_on):
    old = getattr(settings, "graph_crosspage_linking", None)
    settings.graph_crosspage_linking = link_on
    try:
        async def run():
            mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S, pipeline_version="v1")
            await merge_page_deltas_sorted(mgr, [PageDelta.from_dict(deltas[p].to_dict()) for p in order])
            await mgr.reconcile_cross_references()
            return mgr.graph
        return asyncio.run(run())
    finally:
        if old is not None:
            settings.graph_crosspage_linking = old


def _same_entity_edges(graph: GraphMemory):
    return [e for e in graph.edges.values() if e.relationship == "SAME_ENTITY_AS"]


# --------------------------------------------------------------------------- #
# core linking behaviour
# --------------------------------------------------------------------------- #

def test_same_entity_two_pages_gets_linked():
    # Two Organization nodes that stay DISTINCT after merge (different values, so the
    # value-match merge doesn't fire) but share an explicit normalized alias across
    # pages -> deterministic SAME_ENTITY_AS link.
    deltas = {
        1: PageDelta(page_number=1, entities=[_node("p1", "ACME HOSPITAL CENTRAL WING", "hospital", 1, "Organization", aliases=["acme hospital registry"])]),
        2: PageDelta(page_number=2, entities=[_node("p2", "ACME HOSPITAL NORTH BLOCK", "hospital", 2, "Organization", aliases=["acme hospital registry"])]),
    }
    g = _merge(deltas, [1, 2], link_on=True)
    # two distinct nodes remain
    assert len([n for n in g.nodes.values() if "acme" in n.value.lower()]) == 2
    links = _same_entity_edges(g)
    assert len(links) >= 1, "expected a SAME_ENTITY_AS link between the two org nodes"
    e = links[0]
    assert e.properties.get("derived") is True
    assert e.properties.get("rule")
    assert e.status == "DERIVED"


def test_distinct_entities_not_linked():
    deltas = {
        1: PageDelta(page_number=1, entities=[_node("p1", "ARUN SHARMA", "doctor", 1, "Doctor")]),
        2: PageDelta(page_number=2, entities=[_node("p2", "RAMESH GUPTA", "doctor", 2, "Doctor")]),
    }
    g = _merge(deltas, [1, 2], link_on=True)
    assert len(_same_entity_edges(g)) == 0


def test_amount_and_date_nodes_never_linked():
    deltas = {
        1: PageDelta(page_number=1, entities=[_node("a1", "80698", "SanctionedAmount", 1, "Amount")]),
        2: PageDelta(page_number=2, entities=[_node("a2", "80698", "SanctionedAmount", 2, "Amount")]),
        3: PageDelta(page_number=3, entities=[_node("d1", "2025-02-21", "AdmissionDate", 3, "Date")]),
        4: PageDelta(page_number=4, entities=[_node("d2", "2025-02-21", "AdmissionDate", 4, "Date")]),
    }
    # NOTE: same value+type+label amount nodes MERGE (one node), so they can't be linked
    # anyway; the gate must also refuse to LINK amount/date nodes that stay distinct.
    deltas[2].entities[0] = _node("a2", "80698", "SanctionedAmount2", 2, "Amount")  # diff label -> stays distinct
    deltas[4].entities[0] = _node("d2", "2025-02-21", "AdmissionDate2", 4, "Date")
    g = _merge(deltas, [1, 2, 3, 4], link_on=True)
    assert len(_same_entity_edges(g)) == 0, "amount/date nodes must never be SAME_ENTITY_AS linked"


def test_generic_short_and_digit_and_stoplist_values_excluded():
    deltas = {
        1: PageDelta(page_number=1, entities=[
            _node("g1", "F", "gender", 1, "Attribute"),            # stoplist (gender)
            _node("n1", "12", "code", 1, "Identifier"),            # pure digits, short
            _node("s1", "ok", "flag", 1, "Status"),                # too short
        ]),
        2: PageDelta(page_number=2, entities=[
            _node("g2", "F", "gender", 2, "Attribute"),
            _node("n2", "12", "code", 2, "Identifier"),
            _node("s2", "ok", "flag", 2, "Status"),
        ]),
    }
    g = _merge(deltas, [1, 2], link_on=True)
    assert len(_same_entity_edges(g)) == 0, "generic short/digit/stoplist values must not link"


def test_fan_out_cap_respected_and_deterministic():
    # one value on page 1, same value on pages 2..6 (6 same-entity nodes) -> cap limits
    # how many links a single node gets.
    deltas = {1: PageDelta(page_number=1, entities=[_node("hub", "ACME HOSPITAL", "hospital", 1, "Organization")])}
    for p in range(2, 7):
        deltas[p] = PageDelta(page_number=p, entities=[_node(f"n{p}", "ACME HOSPITAL", "hospital", p, "Organization")])
    # same value+type across pages would MERGE into one node; to keep them distinct for a
    # fan-out test, vary the value slightly but share a normalized alias.
    deltas = {1: PageDelta(page_number=1, entities=[_node("hub", "ACME HOSPITAL CENTRAL", "hospital", 1, "Organization", aliases=["acme hospital"])])}
    for p in range(2, 7):
        deltas[p] = PageDelta(page_number=p, entities=[_node(f"n{p}", f"ACME HOSPITAL BRANCH {p}", "hospital", p, "Organization", aliases=["acme hospital"])])
    cap = int(getattr(settings, "graph_crosspage_link_max_fanout", 3))
    g = _merge(deltas, list(deltas.keys()), link_on=True)
    # count links incident on the hub node
    hub_links = [e for e in _same_entity_edges(g) if "hub" in (e.source_node, e.target_node)]
    assert len(hub_links) <= cap, f"fan-out not capped: {len(hub_links)} > {cap}"
    # deterministic: same links under shuffled order
    g2 = _merge(deltas, list(reversed(list(deltas.keys()))), link_on=True)
    def sig(gr):
        return sorted((e.source_node, e.target_node) for e in _same_entity_edges(gr))
    assert sig(g) == sig(g2), "links differ under shuffled arrival order"


def test_linking_identical_under_shuffled_order():
    deltas = {
        1: PageDelta(page_number=1, entities=[_node("p1", "DR ARUN SHARMA", "doctor", 1, "Doctor", aliases=["arun sharma"])]),
        3: PageDelta(page_number=3, entities=[_node("p3", "ARUN SHARMA", "doctor", 3, "Doctor")]),
        2: PageDelta(page_number=2, entities=[_node("p2", "CITY HOSPITAL", "hospital", 2, "Organization")]),
    }
    def sig(order):
        g = _merge(deltas, order, link_on=True)
        return sorted((e.source_node, e.target_node, e.id) for e in _same_entity_edges(g))
    assert sig([1, 2, 3]) == sig([3, 1, 2]) == sig([2, 3, 1])


def test_links_off_produce_no_edges():
    deltas = {
        1: PageDelta(page_number=1, entities=[_node("p1", "DR ARUN SHARMA", "doctor", 1, "Doctor", aliases=["arun sharma"])]),
        2: PageDelta(page_number=2, entities=[_node("p2", "ARUN SHARMA", "doctor", 2, "Doctor")]),
    }
    g = _merge(deltas, [1, 2], link_on=False)
    assert len(_same_entity_edges(g)) == 0


# --------------------------------------------------------------------------- #
# traversal: multi-hop path exists with linking ON, absent with OFF
# --------------------------------------------------------------------------- #

def test_traversal_multihop_path_requires_linking():
    from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
    # page 1: patient --HAS_DOCTOR--> doctor(DR ARUN SHARMA, alias 'arun sharma')
    # page 3: doctor(ARUN SHARMA) --PERFORMED--> procedure
    # Without a cross-page link between the two doctor nodes, a BFS from the patient
    # cannot reach the procedure; with SAME_ENTITY_AS it can.
    # doc1/doc2 are distinct Organization-typed provider nodes (different values) that
    # share an explicit alias, so they stay separate after merge and are linked only by
    # the cross-page pass. (Avoids the honorific name-merge that fuses doctor nodes.)
    d1 = _node("pat", "RAHUL PATIENT RECORD", "patient", 1, "Patient")
    doc1 = _node("doc1", "PROVIDER UNIT ALPHA", "provider", 1, "Organization", aliases=["provider group x"])
    doc2 = _node("doc2", "PROVIDER UNIT BETA", "provider", 3, "Organization", aliases=["provider group x"])
    proc = _node("proc", "APPENDECTOMY PROCEDURE", "procedure", 3, "Procedure")
    from src.ai.layer3_extraction.graph_agent.models import GraphEdge
    e1 = GraphEdge(id="e1", source_node="pat", target_node="doc1", relationship="HAS_DOCTOR", source_page=1)
    e2 = GraphEdge(id="e2", source_node="doc2", target_node="proc", relationship="PERFORMED", source_page=3)
    deltas = {
        1: PageDelta(page_number=1, entities=[d1, doc1], relationships=[e1]),
        3: PageDelta(page_number=3, entities=[doc2, proc], relationships=[e2]),
    }

    def reachable(order, link_on):
        g = _merge(deltas, order, link_on=link_on)
        svc = GraphQueryService(max_hops=3)
        seed = next(n for n in g.nodes.values() if n.id == "pat" or "pat" in n.id)
        nodes, _edges, _h = svc.traverse_subgraph(g, [seed])
        return any("APPENDECTOMY" in n.value.upper() for n in nodes)

    assert reachable([1, 3], link_on=True), "with linking ON the procedure must be reachable from the patient"
    assert not reachable([1, 3], link_on=False), "with linking OFF the procedure must NOT be reachable"
