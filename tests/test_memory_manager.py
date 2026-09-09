"""
File: test_memory_manager.py
Purpose: Test suite for concurrent Blackboard / GraphMemoryManager architecture.
Verifies entity fusion, local-to-canonical ID remapping, edge rewriting,
parallel edges, cross-reference reconciliation, delta isolation, targeted context,
dynamic anchor derivation, concurrent overlap, and failure isolation.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel, Field

from src.ai.layer3_extraction.extractor import (
    extract_document,
    extract_with_graph_memory_concurrent,
)
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import (
    GraphMemoryManager,
    PageDelta,
    UnresolvedRef,
    derive_anchor_types_from_schema,
)
from src.ai.layer3_extraction.graph_agent.models import Evidence, GraphEdge, GraphNode


def async_test(coro_fn):
    """Decorator to run an async test inside asyncio.run without requiring pytest-asyncio."""
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro_fn(*args, **kwargs))
    return wrapper


# Sample schemas
class PatientClaimSchema(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Name of the patient")
    hospital_name: Optional[str] = Field(default=None, description="Name of the hospital")
    claim_amount: Optional[str] = Field(default=None, description="Total claim amount")


class FinancialSchema(BaseModel):
    customer_name: Optional[str] = Field(default=None, description="Customer full name")
    account_number: Optional[str] = Field(default=None, description="Bank account number")
    transaction_amount: Optional[str] = Field(default=None, description="Transaction amount")


class LegalSchema(BaseModel):
    plaintiff_name: Optional[str] = Field(default=None, description="Plaintiff name")
    defendant_name: Optional[str] = Field(default=None, description="Defendant name")
    case_number: Optional[str] = Field(default=None, description="Court case number")


# -------------------------------------------------------------------------
# TEST 1 — concurrent alias/entity fusion
# -------------------------------------------------------------------------
@async_test
async def test_1_concurrent_alias_entity_fusion():
    manager = GraphMemoryManager(anchor_types=["Identifier", "Patient"])

    # Delta 1 from Page 1: UHID with alias "HH NO: 192673"
    node1 = GraphNode(
        id="p1_node_1",
        type="Identifier",
        label="UHID",
        value="192673",
        aliases=["HH NO: 192673"],
        evidence=[Evidence(page_number=1, text="UHID: 192673")],
        source_pages=[1],
    )
    delta1 = PageDelta(page_number=1, entities=[node1])

    # Delta 2 from Page 2: HH NO with matching normalized value
    node2 = GraphNode(
        id="p2_node_1",
        type="Identifier",
        label="HH NO",
        value="192673",
        aliases=["UHID: 192673"],
        evidence=[Evidence(page_number=2, text="HH NO: 192673")],
        source_pages=[2],
    )
    delta2 = PageDelta(page_number=2, entities=[node2])

    # Ingest both deltas concurrently
    await asyncio.gather(
        manager.ingest_page_delta(delta1),
        manager.ingest_page_delta(delta2),
    )

    # Must be fused into ONE canonical entity
    assert len(manager.graph.nodes) == 1
    canonical_node = list(manager.graph.nodes.values())[0]
    assert canonical_node.value == "192673"
    assert 1 in canonical_node.source_pages
    assert 2 in canonical_node.source_pages
    assert len(canonical_node.evidence) == 2


# -------------------------------------------------------------------------
# TEST 2 — edge endpoint canonicalization
# -------------------------------------------------------------------------
@async_test
async def test_2_edge_endpoint_canonicalization():
    manager = GraphMemoryManager(anchor_types=["Patient", "Identifier"])

    # Pre-existing canonical node in manager's graph
    canon_patient = GraphNode(
        id="canonical_patient_001",
        type="Patient",
        label="Patient",
        value="John Doe",
        source_pages=[1],
    )
    manager.graph.create_node(canon_patient)

    # Page 2 worker emitted delta with worker-local IDs
    local_patient = GraphNode(
        id="local_patient_p2",
        type="Patient",
        label="Patient",
        value="John Doe",  # matches canonical
        source_pages=[2],
    )
    local_uhid = GraphNode(
        id="local_uhid_p2",
        type="Identifier",
        label="UHID",
        value="UHID-98765",
        source_pages=[2],
    )
    edge = GraphEdge(
        id="local_edge_1",
        source_node="local_patient_p2",
        target_node="local_uhid_p2",
        relationship="HAS_UHID",
        source_page=2,
    )
    delta = PageDelta(
        page_number=2,
        entities=[local_patient, local_uhid],
        relationships=[edge],
    )

    await manager.ingest_page_delta(delta)

    # Assert local_patient_p2 was canonicalized to canonical_patient_001
    assert "local_patient_p2" not in manager.graph.nodes
    assert "canonical_patient_001" in manager.graph.nodes

    # Assert edge was rewritten to canonical IDs without stale worker-local IDs
    graph_edges = list(manager.graph.edges.values())
    assert len(graph_edges) == 1
    stored_edge = graph_edges[0]
    assert stored_edge.source_node == "canonical_patient_001"
    assert stored_edge.target_node == "local_uhid_p2"
    assert stored_edge.relationship == "HAS_UHID"


# -------------------------------------------------------------------------
# TEST 3 — parallel edge support
# -------------------------------------------------------------------------
@async_test
async def test_3_parallel_edge_support():
    manager = GraphMemoryManager()

    node_a = GraphNode(id="node_a", type="Doctor", label="Doctor", value="Dr. Arun Sharma")
    node_b = GraphNode(id="node_b", type="Patient", label="Patient", value="Rahul Sharma")
    manager.graph.create_node(node_a)
    manager.graph.create_node(node_b)

    # Two distinct relationships between the same endpoints
    edge1 = GraphEdge(
        id="e1",
        source_node="node_a",
        target_node="node_b",
        relationship="TREATED",
        evidence="Primary treating physician",
        source_page=1,
    )
    edge2 = GraphEdge(
        id="e2",
        source_node="node_a",
        target_node="node_b",
        relationship="CONSULTED_WITH",
        evidence="Second opinion consult",
        source_page=2,
    )
    delta = PageDelta(page_number=2, relationships=[edge1, edge2])

    await manager.ingest_page_delta(delta)

    # Assert both distinct edges are preserved in NetworkX MultiDiGraph
    assert len(manager.graph.edges) == 2
    rels = {e.relationship for e in manager.graph.edges.values()}
    assert rels == {"TREATED", "CONSULTED_WITH"}


# -------------------------------------------------------------------------
# TEST 4 — cross-reference reconciliation
# -------------------------------------------------------------------------
@async_test
async def test_4_cross_reference_reconciliation():
    manager = GraphMemoryManager(anchor_types=["Patient"])

    # Page 1: Patient established
    p1_patient = GraphNode(
        id="pat_001",
        type="Patient",
        label="Patient",
        value="CHANDER KOCHHAR",
        source_pages=[1],
        confidence=1.0,
        category="EXPLICIT",
    )
    delta1 = PageDelta(page_number=1, entities=[p1_patient])

    # Page 2: Admission node referencing "the above patient"
    p2_adm = GraphNode(
        id="adm_001",
        type="Admission",
        label="Admission",
        value="IPD-10023",
        source_pages=[2],
    )
    unresolved_ref = UnresolvedRef(
        source_node_id="adm_001",
        phrase="the above patient",
        target_type="Patient",
        page_number=2,
        rationale="Patient admitted under IPD-10023",
    )
    delta2 = PageDelta(
        page_number=2,
        entities=[p2_adm],
        unresolved_references=[unresolved_ref],
    )

    await manager.ingest_page_delta(delta1)
    await manager.ingest_page_delta(delta2)
    await manager.reconcile_cross_references()

    # Verify REFERS_TO edge exists
    ref_edges = manager.graph.get_edges(source_node="adm_001", relationship="REFERS_TO")
    assert len(ref_edges) == 1
    assert ref_edges[0].target_node == "pat_001"
    assert ref_edges[0].status == "INFERRED"

    # Negative test: Ambiguous reference should remain unresolved
    manager2 = GraphMemoryManager(anchor_types=["Patient"])
    cand_a = GraphNode(id="pat_a", type="Patient", label="Patient", value="Alice Smith", source_pages=[1], confidence=1.0)
    cand_b = GraphNode(id="pat_b", type="Patient", label="Patient", value="Bob Jones", source_pages=[1], confidence=1.0)
    delta_amb = PageDelta(
        page_number=2,
        entities=[GraphNode(id="rec_01", type="Record", label="Record", value="Chart 1", source_pages=[2])],
        unresolved_references=[
            UnresolvedRef(
                source_node_id="rec_01",
                phrase="the patient",
                target_type="Patient",
                page_number=2,
            )
        ],
    )
    manager2.graph.create_node(cand_a)
    manager2.graph.create_node(cand_b)
    await manager2.ingest_page_delta(delta_amb)
    await manager2.reconcile_cross_references()

    # Since two patients are equally tied on Page 1, no false edge should be created
    amb_edges = manager2.graph.get_edges(source_node="rec_01", relationship="REFERS_TO")
    assert len(amb_edges) == 0


# -------------------------------------------------------------------------
# TEST 5 — PageDelta isolation
# -------------------------------------------------------------------------
@async_test
async def test_5_page_delta_isolation():
    mock_llm = MagicMock()
    mock_llm.extract_graph_from_page.return_value = {
        "entities": [
            {"id": "p1_e1", "type": "Patient", "label": "Patient", "value": "Isolated Patient"}
        ],
        "relationships": [],
        "reference_resolutions": [],
    }

    graph = GraphMemory()
    agent = GraphExtractionAgent(schema=PatientClaimSchema, graph=graph, llm=mock_llm)

    assert len(graph.nodes) == 0

    delta = await agent.build_page_delta_async(
        page_number=1,
        markdown="Patient: Isolated Patient",
        total_pages=1,
        context=[],
    )

    assert isinstance(delta, PageDelta)
    assert len(delta.entities) == 1
    assert delta.entities[0].value == "Isolated Patient"

    # CRITICAL: agent.graph must remain untouched
    assert len(graph.nodes) == 0


# -------------------------------------------------------------------------
# TEST 6 — targeted anchor context
# -------------------------------------------------------------------------
@async_test
async def test_6_targeted_anchor_context():
    manager = GraphMemoryManager(
        anchor_types=["Patient", "Hospital"],
        anchor_max_k=3,
    )

    # Insert 10 filler nodes
    for i in range(10):
        manager.graph.create_node(
            GraphNode(id=f"misc_{i}", type="Note", label="Note", value=f"Note text {i}", source_pages=[1])
        )

    # Insert 2 Patients and 2 Hospitals
    manager.graph.create_node(
        GraphNode(id="p_1", type="Patient", label="Patient", value="P One", source_pages=[1], category="EXPLICIT")
    )
    manager.graph.create_node(
        GraphNode(id="p_2", type="Patient", label="Patient", value="P Two", source_pages=[2], category="EXPLICIT")
    )
    manager.graph.create_node(
        GraphNode(id="h_1", type="Hospital", label="Hospital", value="H One", source_pages=[1], category="EXPLICIT")
    )
    manager.graph.create_node(
        GraphNode(id="h_2", type="Hospital", label="Hospital", value="H Two", source_pages=[2], category="EXPLICIT")
    )

    context = await manager.get_context_for_page(page_number=3)

    # Must respect max_k = 3
    assert len(context) == 3

    # Must contain targeted anchors only, not miscellaneous notes
    for item in context:
        assert item["type"] in ["Patient", "Hospital"]


# -------------------------------------------------------------------------
# TEST 7 — dynamic anchor derivation
# -------------------------------------------------------------------------
def test_7_dynamic_anchor_derivation():
    # Medical
    med_anchors = derive_anchor_types_from_schema(PatientClaimSchema)
    assert "Patient" in med_anchors
    assert "Hospital" in med_anchors
    assert "Claim" in med_anchors

    # Financial
    fin_anchors = derive_anchor_types_from_schema(FinancialSchema)
    assert "Customer" in fin_anchors
    assert "Account" in fin_anchors
    assert "Transaction" in fin_anchors

    # Legal
    legal_anchors = derive_anchor_types_from_schema(LegalSchema)
    assert "Plaintiff" in legal_anchors
    assert "Defendant" in legal_anchors
    assert "Case" in legal_anchors


# -------------------------------------------------------------------------
# TEST 8 — concurrent E2E
# -------------------------------------------------------------------------
@async_test
async def test_8_concurrent_e2e():
    mock_llm = MagicMock()

    def mock_extract_page(page_md, schema_fields, existing_nodes, page_number, total_pages):
        if page_number == 1:
            return {
                "entities": [
                    {"id": "p1_e1", "type": "Patient", "label": "patient_name", "value": "Rahul Sharma", "is_schema_field": True, "schema_field_name": "patient_name"},
                    {"id": "p1_e2", "type": "Hospital", "label": "hospital_name", "value": "City Hospital", "is_schema_field": True, "schema_field_name": "hospital_name"},
                ],
                "relationships": [
                    {"source_node": "p1_e1", "target_node": "p1_e2", "relationship": "ADMITTED_TO"}
                ],
                "reference_resolutions": [],
            }
        else:
            return {
                "entities": [
                    {"id": "p2_e1", "type": "Claim", "label": "claim_amount", "value": "INR 50,000", "is_schema_field": True, "schema_field_name": "claim_amount"},
                ],
                "relationships": [],
                "reference_resolutions": [
                    {"phrase": "the above patient", "target_type": "Patient", "resolved_to_node_id": "p1_e1"}
                ],
            }

    mock_llm.extract_graph_from_page.side_effect = mock_extract_page
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Rahul Sharma",
        hospital_name="City Hospital",
        claim_amount="INR 50,000",
    )

    pages = [
        {"markdown": "Patient: Rahul Sharma at City Hospital", "page_number": 1},
        {"markdown": "Claim of INR 50,000 for the above patient", "page_number": 2},
    ]

    graph_out: Dict[str, Any] = {}
    result = await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        graph_out=graph_out,
    )

    assert result.patient_name == "Rahul Sharma"
    assert result.hospital_name == "City Hospital"
    assert result.claim_amount == "INR 50,000"

    assert graph_out["strategy"] == "graph_memory_concurrent"
    assert graph_out["pages_succeeded"] == 2
    assert graph_out["pages_failed"] == 0
    assert graph_out["stats"]["total_nodes"] >= 3


# -------------------------------------------------------------------------
# TEST 9 — concurrency actually overlaps
# -------------------------------------------------------------------------
@async_test
async def test_9_concurrency_actually_overlaps(monkeypatch):
    from src.config.settings import settings
    monkeypatch.setattr(settings, "graph_concurrency_limit", 4)

    active_calls = 0
    max_active_calls = 0

    mock_llm = MagicMock()

    def mock_extract_overlapping(page_md, schema_fields, existing_nodes, page_number, total_pages):
        nonlocal active_calls, max_active_calls
        # Simulate blocking LLM latency inside asyncio.to_thread
        active_calls += 1
        if active_calls > max_active_calls:
            max_active_calls = active_calls
        time.sleep(0.15)
        active_calls -= 1
        return {
            "entities": [{"id": f"p{page_number}_e1", "type": "Patient", "value": f"Patient {page_number}"}],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract_overlapping
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Patient 1")

    pages = [{"markdown": f"Page content {i}", "page_number": i} for i in range(1, 5)]

    t0 = time.monotonic()
    await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
    )
    elapsed = time.monotonic() - t0

    # If sequential, 4 * 0.15s = 0.60s minimum.
    # Concurrent with 4 workers: should complete in ~0.20-0.45s.
    assert max_active_calls > 1, f"Expected concurrency overlap, max active was {max_active_calls}"
    assert elapsed < 0.50, f"Expected overlap speedup, took {elapsed:.2f}s"


# -------------------------------------------------------------------------
# TEST 10 — failure isolation
# -------------------------------------------------------------------------
@async_test
async def test_10_failure_isolation():
    mock_llm = MagicMock()

    def mock_extract_with_failure(page_md, schema_fields, existing_nodes, page_number, total_pages):
        if page_number == 2:
            raise RuntimeError("LLM API Provider Error on Page 2")
        return {
            "entities": [{"id": f"p{page_number}_e1", "type": "Patient", "value": f"Patient {page_number}"}],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract_with_failure
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Patient 1")

    pages = [
        {"markdown": "Page 1 info", "page_number": 1},
        {"markdown": "Page 2 info", "page_number": 2},
        {"markdown": "Page 3 info", "page_number": 3},
    ]

    graph_out: Dict[str, Any] = {}
    result = await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        graph_out=graph_out,
    )

    # Other pages must still succeed
    assert graph_out["pages_total"] == 3
    assert graph_out["pages_succeeded"] == 2
    assert graph_out["pages_failed"] == 1
    assert graph_out["failed_pages"] == [2]

    # Graph must be valid and contain nodes from pages 1 and 3
    assert graph_out["stats"]["total_nodes"] == 2


def test_11_extract_document_sync_wrapper():
    """Verify synchronous callers of extract_document can use graph_memory_concurrent."""
    mock_llm = MagicMock()
    mock_llm.extract_graph_from_page.return_value = {
        "entities": [{"id": "p1_e1", "type": "Patient", "value": "Sync Patient"}],
        "relationships": [],
        "reference_resolutions": [],
    }
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Sync Patient")

    pages = [{"markdown": "Patient: Sync Patient", "page_number": 1}]
    graph_out: Dict[str, Any] = {}

    # Called synchronously
    res = extract_document(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        strategy="graph_memory_concurrent",
        graph_out=graph_out,
    )

    assert res.patient_name == "Sync Patient"
    assert graph_out["strategy"] == "graph_memory_concurrent"
    assert graph_out["stats"]["total_nodes"] == 1
