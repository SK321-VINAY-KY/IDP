"""
File: test_layer3_hardening.py
Purpose: Comprehensive test suite for hardened Layer 3 Graph Memory.
Verifies:
1. False-merge protection (exact/alias, wrong LLM suggestion, type contradiction, identifier conflict,
   contamination prevention, high similarity with contradiction, ambiguity margins, provenance).
2. Uncertainty model (distinct confidence dimensions, unresolved uncertain references, promotion
   via resolution pipeline, protection against uncertain contamination in schema output, edge confidence).
3. Page-level checkpointing and recovery (durable checkpoint creation, idempotent retries, checksum conflict,
   crash-safe replay, malformed delta rejection, partial document resumption).
4. Concurrency semantics (ordered context access, concurrent snapshot isolation, epoch mode, lock isolation).
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field

from src.ai.layer3_extraction.extractor import (
    extract_document,
    extract_with_graph_memory_batched,
    extract_with_graph_memory_concurrent,
    extract_with_graph_memory_ordered,
)
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import (
    GraphMemoryManager,
    PageDelta,
    UnresolvedRef,
)
from src.ai.layer3_extraction.graph_agent.models import (
    Evidence,
    GraphEdge,
    GraphNode,
    ResolutionStatus,
    TypeCompatibility,
    validate_page_delta,
)
from src.ai.layer3_extraction.graph_agent.resolver import (
    format_graph_evidence_for_resolution,
    resolve_schema_from_graph,
)
from src.ai.layer3_extraction.storage import (
    compute_page_delta_checksum,
    delete_page_checkpoints,
    get_page_checkpoint,
    init_db,
    list_page_checkpoints,
    save_page_checkpoint,
)


def async_test(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro_fn(*args, **kwargs))
    return wrapper


class PatientClaimSchema(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Patient name")
    hospital_name: Optional[str] = Field(default=None, description="Hospital name")
    claim_amount: Optional[str] = Field(default=None, description="Claim amount")


class HospitalPANSchema(BaseModel):
    hospital_name: Optional[str] = Field(default=None, description="Hospital name")
    hospital_pan: Optional[str] = Field(default=None, description="Hospital PAN number")


# ==============================================================================
# 1. FALSE-MERGE TESTS (Sections 3, 4, 6, 24)
# ==============================================================================

@async_test
async def test_fm_1_exact_and_alias_match_merges():
    """Exact/alias match -> confirmed match -> fused into ONE canonical entity."""
    manager = GraphMemoryManager(anchor_types=["Patient"])

    node1 = GraphNode(
        id="p1_pat",
        type="Patient",
        label="Patient",
        value="Arun Sharma",
        aliases=["A. Sharma"],
        evidence=[Evidence(page_number=1, text="Patient: Arun Sharma")],
        source_pages=[1],
    )
    node2 = GraphNode(
        id="p2_pat",
        type="Patient",
        label="Patient",
        value="A. Sharma",
        evidence=[Evidence(page_number=2, text="Subject: A. Sharma")],
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[node1]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[node2]))

    # Confirmed match -> single canonical node
    assert len(manager.graph.nodes) == 1
    canonical = list(manager.graph.nodes.values())[0]
    assert canonical.value == "Arun Sharma"
    assert 1 in canonical.source_pages and 2 in canonical.source_pages
    assert len(canonical.evidence) == 2


@async_test
async def test_fm_2_llm_wrong_reused_node_id_rejected():
    """MANDATORY: LLM proposes reused_node_id, but candidate has contradictory identity evidence -> NO merge."""
    manager = GraphMemoryManager(anchor_types=["Doctor"])

    # Established canonical doctor
    canon_doc = GraphNode(
        id="canonical_doc_42",
        type="Doctor",
        label="Doctor",
        value="Dr. Arun Sharma",
        source_pages=[1],
    )
    manager.graph.create_node(canon_doc)

    # Worker LLM hallucinates reused_node_id = canonical_doc_42 for a DIFFERENT doctor
    incoming_doc = GraphNode(
        id="p2_doc_1",
        type="Doctor",
        label="Doctor",
        value="Dr. Ramesh Sharma",  # Conflicting distinct person!
        properties={"reused_node_id": "canonical_doc_42"},
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[incoming_doc]))

    # Must NOT merge into canonical_doc_42; must be separate entities
    assert len(manager.graph.nodes) == 2
    doc_values = {n.value for n in manager.graph.nodes.values()}
    assert "Dr. Arun Sharma" in doc_values
    assert "Dr. Ramesh Sharma" in doc_values


@async_test
async def test_fm_3_type_contradiction_blocks_merge():
    """Explicitly incompatible types (e.g. Patient vs Hospital) -> merge blocked even if values match."""
    manager = GraphMemoryManager()

    pat_node = GraphNode(
        id="pat_01",
        type="Patient",
        label="Patient",
        value="City Care",
        source_pages=[1],
    )
    hosp_node = GraphNode(
        id="hosp_01",
        type="Hospital",
        label="Hospital",
        value="City Care",  # Same string, but incompatible type!
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[pat_node]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[hosp_node]))

    # Incompatible types -> 2 distinct canonical nodes
    assert len(manager.graph.nodes) == 2
    types = {n.type for n in manager.graph.nodes.values()}
    assert types == {"Patient", "Hospital"}


@async_test
async def test_fm_4_conflicting_identifier_blocks_merge():
    """Observations with conflicting explicit identifiers (e.g. UHID=123 vs UHID=456) -> merge blocked."""
    manager = GraphMemoryManager(anchor_types=["Patient"])

    pat1 = GraphNode(
        id="pat_1",
        type="Patient",
        label="Patient",
        value="John Doe",
        properties={"uhid": "UHID-123"},
        source_pages=[1],
    )
    pat2 = GraphNode(
        id="pat_2",
        type="Patient",
        label="Patient",
        value="John Doe",  # Same name, but conflicting UHID!
        properties={"uhid": "UHID-456"},
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[pat1]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[pat2]))

    # Conflicting UHIDs must NOT merge
    assert len(manager.graph.nodes) == 2


@async_test
async def test_fm_5_false_merge_contamination_prevented():
    """
    Page 1: Arun Sharma, DOB=1980, UHID=123
    Page 2: Arun Sharma, DOB=1990, UHID=456
    Expected: TWO entities, NOT one entity containing both identities.
    """
    manager = GraphMemoryManager(anchor_types=["Patient"])

    pat1 = GraphNode(
        id="pat_p1",
        type="Patient",
        label="Patient",
        value="Arun Sharma",
        properties={"dob": "1980-01-01", "uhid": "UHID-123"},
        evidence=[Evidence(page_number=1, text="Patient Arun Sharma, DOB: 1980-01-01, UHID-123")],
        source_pages=[1],
    )
    pat2 = GraphNode(
        id="pat_p2",
        type="Patient",
        label="Patient",
        value="Arun Sharma",
        properties={"dob": "1990-05-12", "uhid": "UHID-456"},
        evidence=[Evidence(page_number=2, text="Patient Arun Sharma, DOB: 1990-05-12, UHID-456")],
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[pat1]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[pat2]))

    # Must preserve TWO distinct entities
    assert len(manager.graph.nodes) == 2
    dobs = {n.properties.get("dob") for n in manager.graph.nodes.values()}
    assert dobs == {"1980-01-01", "1990-05-12"}


@async_test
async def test_fm_6_high_similarity_with_contradiction_rejected():
    """High textual similarity (token overlap) + contradictory identifier -> merge blocked."""
    manager = GraphMemoryManager(anchor_types=["Account"])

    acc1 = GraphNode(
        id="acc_1",
        type="Account",
        label="Account",
        value="Acme Corp General Corporate Reserve Account",
        properties={"account_no": "ACC-001"},
        source_pages=[1],
    )
    acc2 = GraphNode(
        id="acc_2",
        type="Account",
        label="Account",
        value="Acme Corp General Corporate Reserve Account",
        properties={"account_no": "ACC-002"},  # Different account number!
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[acc1]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[acc2]))

    assert len(manager.graph.nodes) == 2


@async_test
async def test_fm_7_ambiguous_candidates_remain_uncertain():
    """When top two candidates are close in score (margin < ambiguity_margin), status becomes UNCERTAIN."""
    manager = GraphMemoryManager(anchor_types=["Patient"], ambiguity_margin=0.20)

    # Establish two distinct candidates
    cand1 = GraphNode(id="cand_1", type="Patient", label="Patient", value="Robert James Smith", source_pages=[1])
    cand2 = GraphNode(id="cand_2", type="Patient", label="Patient", value="Robert John Smith", source_pages=[1])
    manager.graph.create_node(cand1)
    manager.graph.create_node(cand2)

    # Incoming observation that is equally similar to both
    incoming = GraphNode(id="inc_1", type="Patient", label="Patient", value="Robert Smith", source_pages=[2])

    decision = manager.resolve_entity(incoming, page_num=2)
    assert decision.status == ResolutionStatus.UNCERTAIN_MATCH
    assert decision.decision_margin < manager.ambiguity_margin

    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[incoming]))

    # Node created with status UNCERTAIN, neither candidate merged
    unc_nodes = [n for n in manager.graph.nodes.values() if n.status == "UNCERTAIN"]
    assert len(unc_nodes) == 1
    assert unc_nodes[0].properties.get("uncertain_candidate_id") is not None


@async_test
async def test_fm_8_confirmed_merge_retains_provenance_and_audit():
    """Confirmed merge retains source pages, evidence, aliases, observations, and structured merge audit."""
    manager = GraphMemoryManager(anchor_types=["Patient"])

    pat1 = GraphNode(
        id="pat_001",
        type="Patient",
        label="Patient",
        value="Alice Johnson",
        aliases=["Alice J."],
        evidence=[Evidence(page_number=1, text="Patient: Alice Johnson")],
        source_pages=[1],
    )
    pat2 = GraphNode(
        id="pat_p2",
        type="Patient",
        label="Patient",
        value="Alice Johnson",
        aliases=["Ms. Johnson"],
        evidence=[Evidence(page_number=2, text="Admitted: Alice Johnson")],
        source_pages=[2],
    )

    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[pat1]))
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[pat2]))

    canonical = manager.graph.get_node("pat_001")
    assert canonical is not None
    assert 1 in canonical.source_pages and 2 in canonical.source_pages
    assert len(canonical.evidence) == 2
    assert "Ms. Johnson" in canonical.aliases

    # Check merge audit ledger
    ledger = manager.get_merge_ledger()
    assert len(ledger) >= 1
    record = ledger[0]
    assert record.source_node_id == "pat_p2"
    assert record.canonical_node_id == "pat_001"
    assert record.contradiction_result == "PASSED_HARD_GATES"

    # Canonical node properties contain audit record
    assert "merge_audit" in canonical.properties
    assert len(canonical.properties["observations"]) >= 1


# ==============================================================================
# 2. UNCERTAINTY MODEL TESTS (Sections 9, 10, 13, 23, 25)
# ==============================================================================

def test_unc_1_distinct_confidence_dimensions():
    """Extraction confidence and resolution confidence remain distinct on GraphNode."""
    node = GraphNode(
        id="test_node",
        type="Patient",
        label="Patient",
        value="Jane Doe",
        extraction_confidence=0.85,
        resolution_confidence=0.92,
        evidence_strength=0.90,
    )
    d = node.to_dict()
    assert d["extraction_confidence"] == 0.85
    assert d["resolution_confidence"] == 0.92

    reconstructed = GraphNode.from_dict(d)
    assert reconstructed.extraction_confidence == 0.85
    assert reconstructed.resolution_confidence == 0.92


@async_test
async def test_unc_1b_new_entity_resolution_confidence_semantics_and_reuse():
    """
    Focused regression test for NEW_ENTITY resolution confidence semantics:
    1. When an observation becomes NEW_ENTITY, resolution_confidence is NOT 1.0 (it is None).
    2. Serialization round-trip (to_dict / from_dict) cleanly preserves resolution_confidence=None.
    3. An actually reused/resolved entity still receives its existing resolution confidence behavior
       (candidate_score from confirmed match).
    """
    manager = GraphMemoryManager(anchor_types=["Patient"])

    # --- Step 1: Ingest initial observation (Page 1) ---
    # Since graph is empty, this observation must be created as NEW_ENTITY
    obs_p1 = GraphNode(
        id="raw_p1_pat",
        type="Patient",
        label="Patient",
        value="Carlos Mendoza",
        aliases=["C. Mendoza"],
        evidence=[Evidence(page_number=1, text="Patient: Carlos Mendoza")],
        source_pages=[1],
        extraction_confidence=0.88,
    )
    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[obs_p1]))

    canonical_node = manager.graph.get_node("raw_p1_pat")
    assert canonical_node is not None

    # Regression check: NEW_ENTITY must NOT have resolution_confidence == 1.0
    assert canonical_node.resolution_confidence != 1.0
    assert canonical_node.resolution_confidence is None

    # Serialization check: to_dict / from_dict preserves None
    serialized = canonical_node.to_dict()
    assert serialized["resolution_confidence"] is None
    restored = GraphNode.from_dict(serialized)
    assert restored.resolution_confidence is None

    # --- Step 2: Ingest second observation matching alias (Page 2) ---
    # This observation matches existing Carlos Mendoza -> CONFIRMED_MATCH reuse
    obs_p2 = GraphNode(
        id="raw_p2_pat",
        type="Patient",
        label="Patient",
        value="C. Mendoza",
        evidence=[Evidence(page_number=2, text="Admitted: C. Mendoza")],
        source_pages=[2],
        extraction_confidence=0.91,
    )
    await manager.ingest_page_delta(PageDelta(page_number=2, entities=[obs_p2]))

    # Must still be one canonical node (fused)
    assert len(manager.graph.nodes) == 1
    reused_node = manager.graph.get_node("raw_p1_pat")
    assert reused_node is not None

    # Reused/resolved entity receives actual resolution confidence from candidate evaluation
    assert reused_node.resolution_confidence is not None
    assert isinstance(reused_node.resolution_confidence, float)
    assert 0.0 <= reused_node.resolution_confidence <= 1.0
    assert reused_node.resolution_confidence > 0.5
    assert 2 in reused_node.source_pages


def test_unc_2_edge_confidence_distinct():
    """Edge relationship confidence is distinguished from extraction confidence."""
    edge = GraphEdge(
        id="edge_1",
        source_node="n1",
        target_node="n2",
        relationship="TREATS",
        extraction_confidence=0.80,
        relationship_confidence=0.95,
        evidence_strength=0.90,
    )
    d = edge.to_dict()
    assert d["extraction_confidence"] == 0.80
    assert d["relationship_confidence"] == 0.95


def test_unc_3_uncertain_facts_do_not_contaminate_schema():
    """Uncertain nodes are flagged in prompt and excluded from fallback resolution."""
    graph = GraphMemory()
    # Authoritative hospital
    graph.create_node(
        GraphNode(
            id="h_1",
            type="Hospital",
            label="hospital_name",
            value="Confirmed City Hospital",
            status="ASSERTED",
            properties={"schema_field_name": "hospital_name"},
        )
    )
    # UNCERTAIN candidate patient
    graph.create_node(
        GraphNode(
            id="p_unc",
            type="Patient",
            label="patient_name",
            value="Unverified Suspect",
            status="UNCERTAIN",
            properties={"schema_field_name": "patient_name"},
        )
    )

    evidence = format_graph_evidence_for_resolution(graph, [{"name": "patient_name", "description": "Patient"}])
    assert "[UNCERTAIN - UNVERIFIED Candidate]" in evidence
    assert "[ASSERTED - Primary Evidence]" in evidence

    # LLM fails; test fallback reconciliation
    mock_llm = MagicMock()
    mock_llm.resolve_schema_from_graph.side_effect = RuntimeError("LLM unavailable")

    result = resolve_schema_from_graph(graph=graph, schema=PatientClaimSchema, llm=mock_llm)
    # The ASSERTED hospital must be populated via fallback
    assert result.hospital_name == "Confirmed City Hospital"
    # The UNCERTAIN patient must NOT be silently populated into authoritative output!
    assert result.patient_name is None


def test_schema_resolution_fuzzy_candidate_must_not_populate_empty_field():
    """
    Test 1 — weak fuzzy candidate must NOT populate field.
    An ASSERTED Organization node 'Fortis Hospital' matching generic fuzzy search for
    'hospital_pan' must NOT populate 'hospital_pan'.
    """
    graph = GraphMemory()
    graph.create_node(
        GraphNode(
            id="org_1",
            type="Organization",
            label="Hospital",
            value="Fortis Hospital",
            status="ASSERTED",
        )
    )

    mock_llm = MagicMock()
    # LLM leaves hospital_pan empty
    mock_llm.resolve_schema_from_graph.return_value = HospitalPANSchema(
        hospital_name="Fortis Hospital",
        hospital_pan=None,
    )

    result = resolve_schema_from_graph(graph=graph, schema=HospitalPANSchema, llm=mock_llm)
    assert result.hospital_name == "Fortis Hospital"
    # CRITICAL: hospital_pan remains empty and is NOT populated with "Fortis Hospital"
    assert result.hospital_pan is None


def test_schema_resolution_explicit_schema_field_association_still_works():
    """
    Test 2 — explicit schema-field association still works.
    When a node is explicitly tagged with properties={'schema_field_name': 'hospital_pan'},
    deterministic reconciliation populates the empty field from that node.
    """
    graph = GraphMemory()
    graph.create_node(
        GraphNode(
            id="pan_1",
            type="Identifier",
            label="PAN",
            value="ABCDE1234F",
            status="ASSERTED",
            properties={"schema_field_name": "hospital_pan"},
        )
    )

    mock_llm = MagicMock()
    mock_llm.resolve_schema_from_graph.return_value = HospitalPANSchema(
        hospital_name=None,
        hospital_pan=None,
    )

    result = resolve_schema_from_graph(graph=graph, schema=HospitalPANSchema, llm=mock_llm)
    assert result.hospital_pan == "ABCDE1234F"


def test_schema_resolution_uncertain_explicit_candidate_remains_excluded():
    """
    Test 3 — UNCERTAIN explicit candidate remains excluded.
    Even when explicitly tagged with properties={'schema_field_name': 'hospital_pan'},
    a node with status='UNCERTAIN' must NOT populate the schema field.
    """
    graph = GraphMemory()
    graph.create_node(
        GraphNode(
            id="pan_unc",
            type="Identifier",
            label="PAN",
            value="ABCDE1234F",
            status="UNCERTAIN",
            properties={"schema_field_name": "hospital_pan"},
        )
    )

    mock_llm = MagicMock()
    mock_llm.resolve_schema_from_graph.return_value = HospitalPANSchema(
        hospital_name=None,
        hospital_pan=None,
    )

    result = resolve_schema_from_graph(graph=graph, schema=HospitalPANSchema, llm=mock_llm)
    assert result.hospital_pan is None


# ==============================================================================
# 3. PAGE CHECKPOINTING & RECOVERY TESTS (Sections 14, 15, 16, 17, 18, 26)
# ==============================================================================

def test_chk_1_successful_page_checkpoint_persists():
    """Page checkpoint persists to database with status and delta checksum."""
    doc_id = "test_doc_chk_1"
    delete_page_checkpoints(doc_id)

    delta = PageDelta(
        page_number=1,
        entities=[GraphNode(id="n1", type="Patient", label="Patient", value="Test Patient")],
    )
    delta_dict = delta.to_dict()
    chk = compute_page_delta_checksum(delta_dict)

    rec_id = save_page_checkpoint(
        doc_id=doc_id,
        page_number=1,
        status="DELTA_PERSISTED",
        page_delta_json=delta_dict,
        delta_checksum=chk,
    )
    assert rec_id is not None

    cp = get_page_checkpoint(doc_id=doc_id, page_number=1)
    assert cp is not None
    assert cp["status"] == "DELTA_PERSISTED"
    assert cp["delta_checksum"] == chk

    delete_page_checkpoints(doc_id)


@async_test
async def test_chk_2_idempotent_retry_with_same_checksum():
    """Ingesting the same PageDelta with the same checksum is a safe, idempotent no-op."""
    manager = GraphMemoryManager()

    node = GraphNode(id="p1_n1", type="Patient", label="Patient", value="Idempotent Patient", source_pages=[1])
    delta = PageDelta(page_number=1, entities=[node])
    checksum = compute_page_delta_checksum(delta.to_dict())

    # Ingest once
    await manager.ingest_page_delta(delta, delta_checksum=checksum)
    assert len(manager.graph.nodes) == 1

    # Ingest same delta with same checksum again
    await manager.ingest_page_delta(delta, delta_checksum=checksum)
    # Must NOT duplicate nodes
    assert len(manager.graph.nodes) == 1


def test_chk_3_malformed_page_delta_rejected_by_validator():
    """Malformed PageDelta (missing value, invalid confidence, destructive keys) fails validation."""
    # Missing entity value
    bad_delta1 = PageDelta(
        page_number=1,
        entities=[GraphNode(id="n1", type="Patient", label="Patient", value="")],
    )
    res1 = validate_page_delta(bad_delta1)
    assert not res1.is_valid
    assert any("empty value" in e for e in res1.errors)

    # Invalid confidence range
    bad_delta2 = PageDelta(
        page_number=1,
        entities=[GraphNode(id="n2", type="Patient", label="Patient", value="Good", confidence=1.5)],
    )
    res2 = validate_page_delta(bad_delta2)
    assert not res2.is_valid
    assert any("out of range" in e for e in res2.errors)

    # Destructive command key
    bad_delta3 = PageDelta(
        page_number=1,
        entities=[GraphNode(id="n3", type="Patient", label="Patient", value="Good", properties={"delete_node": "all"})],
    )
    res3 = validate_page_delta(bad_delta3)
    assert not res3.is_valid
    assert any("Destructive mutation" in e for e in res3.errors)


@async_test
async def test_chk_4_crash_recovery_skips_completed_pages():
    """When recovering a partially completed document, completed pages are replayed and skipped from LLM extraction."""
    doc_id = "test_doc_recovery_4"
    delete_page_checkpoints(doc_id)

    # Page 1 was already completed in a prior run
    p1_delta = PageDelta(
        page_number=1,
        entities=[
            GraphNode(
                id="p1_n1",
                type="Patient",
                label="patient_name",
                value="Saved Patient",
                category="EXPLICIT",
                properties={"schema_field_name": "patient_name"},
            )
        ],
    )
    p1_dict = p1_delta.to_dict()
    p1_chk = compute_page_delta_checksum(p1_dict)
    save_page_checkpoint(doc_id=doc_id, page_number=1, status="COMPLETED", page_delta_json=p1_dict, delta_checksum=p1_chk)

    mock_llm = MagicMock()
    # Mock LLM should only be called for Page 2, NEVER for Page 1!
    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        assert page_number == 2, f"LLM was called for page {page_number}, but page 1 was already completed!"
        return {
            "entities": [{"id": "p2_n1", "type": "Hospital", "label": "hospital_name", "value": "Saved Hospital", "is_schema_field": True, "schema_field_name": "hospital_name"}],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Saved Patient",
        hospital_name="Saved Hospital",
    )

    pages = [
        {"markdown": "Page 1 Content", "page_number": 1},
        {"markdown": "Page 2 Content", "page_number": 2},
    ]

    graph_out: Dict[str, Any] = {}
    result = await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        doc_id=doc_id,
        graph_out=graph_out,
    )

    assert result.patient_name == "Saved Patient"
    assert result.hospital_name == "Saved Hospital"
    assert graph_out["pages_succeeded"] == 2
    assert graph_out["stats"]["total_nodes"] == 2

    delete_page_checkpoints(doc_id)


@async_test
async def test_chk_7_ordered_mode_crash_recovery_replays_delta_persisted():
    """
    Ordered-mode crash recovery test:
    When a checkpoint has status DELTA_PERSISTED and non-null page_delta_json,
    it must be recovered and replayed into GraphMemory without calling LLM extraction for that page.
    Also asserts that delta_checksum is passed through during replay.
    """
    doc_id = "test_doc_recovery_ordered_delta_persisted"
    delete_page_checkpoints(doc_id)

    # Page 1 crashed after DELTA_PERSISTED before reaching COMPLETED
    p1_delta = PageDelta(
        page_number=1,
        entities=[
            GraphNode(
                id="p1_n1",
                type="Patient",
                label="patient_name",
                value="Persisted Patient Ordered",
                category="EXPLICIT",
                properties={"schema_field_name": "patient_name"},
            )
        ],
    )
    p1_dict = p1_delta.to_dict()
    p1_chk = compute_page_delta_checksum(p1_dict)
    save_page_checkpoint(
        doc_id=doc_id,
        page_number=1,
        status="DELTA_PERSISTED",
        page_delta_json=p1_dict,
        delta_checksum=p1_chk,
    )

    mock_llm = MagicMock()
    # Mock LLM must only be called for Page 2, NEVER for Page 1!
    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        assert page_number == 2, f"LLM extraction called for page {page_number}, but page 1 was DELTA_PERSISTED!"
        return {
            "entities": [
                {
                    "id": "p2_n1",
                    "type": "Hospital",
                    "label": "hospital_name",
                    "value": "Fresh Hospital Ordered",
                    "is_schema_field": True,
                    "schema_field_name": "hospital_name",
                }
            ],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Persisted Patient Ordered",
        hospital_name="Fresh Hospital Ordered",
    )

    pages = [
        {"markdown": "Page 1 Content", "page_number": 1},
        {"markdown": "Page 2 Content", "page_number": 2},
    ]

    # Track checksums passed to ingest_page_delta
    orig_ingest = GraphMemoryManager.ingest_page_delta
    recorded_replays: List[Tuple[int, Optional[str]]] = []

    async def spy_ingest(self, delta, delta_checksum=None):
        recorded_replays.append((delta.page_number, delta_checksum))
        return await orig_ingest(self, delta, delta_checksum=delta_checksum)

    graph_out: Dict[str, Any] = {}
    with patch.object(GraphMemoryManager, "ingest_page_delta", side_effect=spy_ingest, autospec=True):
        result = await extract_with_graph_memory_ordered(
            pages_md=pages,
            schema=PatientClaimSchema,
            llm=mock_llm,
            doc_id=doc_id,
            graph_out=graph_out,
        )

    # 1. Assert replayed PageDelta entities are in the resolved schema & graph
    assert result.patient_name == "Persisted Patient Ordered"
    assert result.hospital_name == "Fresh Hospital Ordered"
    assert graph_out["pages_succeeded"] == 2
    assert graph_out["stats"]["total_nodes"] == 2

    # 2. Assert build_page_delta_async / LLM was NOT called for page 1
    assert mock_llm.extract_graph_from_page.call_count == 1

    # 3. Assert checksum was passed through during replay
    assert (1, p1_chk) in recorded_replays

    delete_page_checkpoints(doc_id)


@async_test
async def test_chk_8_concurrent_mode_crash_recovery_replays_delta_persisted():
    """
    Concurrent-mode crash recovery test:
    When a checkpoint has status DELTA_PERSISTED and non-null page_delta_json,
    it must be recovered and replayed into GraphMemory without calling LLM extraction for that page.
    Also asserts that delta_checksum is passed through during replay.
    """
    doc_id = "test_doc_recovery_concurrent_delta_persisted"
    delete_page_checkpoints(doc_id)

    # Page 1 crashed after DELTA_PERSISTED before reaching COMPLETED
    p1_delta = PageDelta(
        page_number=1,
        entities=[
            GraphNode(
                id="p1_n1",
                type="Patient",
                label="patient_name",
                value="Persisted Patient Concurrent",
                category="EXPLICIT",
                properties={"schema_field_name": "patient_name"},
            )
        ],
    )
    p1_dict = p1_delta.to_dict()
    p1_chk = compute_page_delta_checksum(p1_dict)
    save_page_checkpoint(
        doc_id=doc_id,
        page_number=1,
        status="DELTA_PERSISTED",
        page_delta_json=p1_dict,
        delta_checksum=p1_chk,
    )

    mock_llm = MagicMock()
    # Mock LLM must only be called for Page 2, NEVER for Page 1!
    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        assert page_number == 2, f"LLM extraction called for page {page_number}, but page 1 was DELTA_PERSISTED!"
        return {
            "entities": [
                {
                    "id": "p2_n1",
                    "type": "Hospital",
                    "label": "hospital_name",
                    "value": "Fresh Hospital Concurrent",
                    "is_schema_field": True,
                    "schema_field_name": "hospital_name",
                }
            ],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Persisted Patient Concurrent",
        hospital_name="Fresh Hospital Concurrent",
    )

    pages = [
        {"markdown": "Page 1 Content", "page_number": 1},
        {"markdown": "Page 2 Content", "page_number": 2},
    ]

    # Track checksums passed to ingest_page_delta
    orig_ingest = GraphMemoryManager.ingest_page_delta
    recorded_replays: List[Tuple[int, Optional[str]]] = []

    async def spy_ingest(self, delta, delta_checksum=None):
        recorded_replays.append((delta.page_number, delta_checksum))
        return await orig_ingest(self, delta, delta_checksum=delta_checksum)

    graph_out: Dict[str, Any] = {}
    with patch.object(GraphMemoryManager, "ingest_page_delta", side_effect=spy_ingest, autospec=True):
        result = await extract_with_graph_memory_concurrent(
            pages_md=pages,
            schema=PatientClaimSchema,
            llm=mock_llm,
            doc_id=doc_id,
            graph_out=graph_out,
        )

    # 1. Assert replayed PageDelta entities are in the resolved schema & graph
    assert result.patient_name == "Persisted Patient Concurrent"
    assert result.hospital_name == "Fresh Hospital Concurrent"
    assert graph_out["pages_succeeded"] == 2
    assert graph_out["stats"]["total_nodes"] == 2

    # 2. Assert build_page_delta_async / LLM was NOT called for page 1
    assert mock_llm.extract_graph_from_page.call_count == 1

    # 3. Assert checksum was passed through during replay
    assert (1, p1_chk) in recorded_replays

    delete_page_checkpoints(doc_id)


@async_test
async def test_chk_9_page_delta_checksum_conflict_prevents_graph_mutation():
    """
    Focused regression test for PageDelta checksum conflict:
    A. Creates a GraphMemoryManager.
    B. Ingests PageDelta(page_number=1) with checksum A.
    C. Records the graph state after successful ingestion.
    D. Attempts to ingest a different PageDelta for page 1 with checksum B.
    E. Verifies:
       - checksum conflict is detected (recorded in manager.checksum_conflicts)
       - the second delta is NOT merged
       - node count is unchanged
       - edge count is unchanged
       - the original page/checksum state remains intact
    F. Also verifies idempotency:
       - ingesting the exact same PageDelta with checksum A twice remains a no-op
    """
    manager = GraphMemoryManager()

    # Step B: Ingest PageDelta(page_number=1) with checksum A
    node_a1 = GraphNode(
        id="p1_n1",
        type="Patient",
        label="patient_name",
        value="Patient Alpha",
        source_pages=[1],
        extraction_confidence=0.9,
    )
    node_a2 = GraphNode(
        id="p1_h1",
        type="Hospital",
        label="hospital_name",
        value="City Hospital",
        source_pages=[1],
        extraction_confidence=0.9,
    )
    edge_a = GraphEdge(
        id="e1",
        source_node="p1_n1",
        target_node="p1_h1",
        relationship="TREATED_AT",
        source_page=1,
    )
    delta_a = PageDelta(page_number=1, entities=[node_a1, node_a2], relationships=[edge_a])
    checksum_a = compute_page_delta_checksum(delta_a.to_dict())

    res_a = await manager.ingest_page_delta(delta_a, delta_checksum=checksum_a)
    assert res_a is True

    # Step C: Record the graph state after successful ingestion
    initial_node_count = len(manager.graph.nodes)
    initial_edge_count = len(manager.graph.edges)
    assert initial_node_count == 2
    assert initial_edge_count == 1
    initial_node = manager.graph.get_node("p1_n1")
    assert initial_node is not None
    assert initial_node.value == "Patient Alpha"
    assert manager._ingested_deltas[1] == checksum_a

    # Step F: Ingest exact same PageDelta with checksum A twice -> no-op
    res_noop = await manager.ingest_page_delta(delta_a, delta_checksum=checksum_a)
    assert res_noop is True
    assert len(manager.graph.nodes) == initial_node_count
    assert len(manager.graph.edges) == initial_edge_count
    assert manager._ingested_deltas[1] == checksum_a

    # Step D: Attempt to ingest a different PageDelta for page 1 with checksum B
    node_b = GraphNode(
        id="p1_n2",
        type="Hospital",
        label="hospital_name",
        value="Conflicting Hospital Beta",
        source_pages=[1],
        extraction_confidence=0.85,
    )
    edge_b = GraphEdge(
        id="e2",
        source_node="p1_n2",
        target_node="p1_h1",
        relationship="AFFILIATED",
        source_page=1,
    )
    delta_b = PageDelta(page_number=1, entities=[node_b], relationships=[edge_b])
    checksum_b = compute_page_delta_checksum(delta_b.to_dict())
    assert checksum_b != checksum_a

    # Ingest conflicting delta
    res_b = await manager.ingest_page_delta(delta_b, delta_checksum=checksum_b)
    assert res_b is False

    # Step E: Verifications
    # 1. Checksum conflict is detected
    assert len(manager.checksum_conflicts) == 1
    conflict = manager.checksum_conflicts[0]
    assert conflict["page_number"] == 1
    assert conflict["stored_checksum"] == checksum_a
    assert conflict["incoming_checksum"] == checksum_b

    # 2. The second delta is NOT merged (node_b not in graph)
    assert manager.graph.get_node("p1_n2") is None
    assert all(n.value != "Conflicting Hospital Beta" for n in manager.graph.nodes.values())

    # 3. Node count is unchanged
    assert len(manager.graph.nodes) == initial_node_count

    # 4. Edge count is unchanged
    assert len(manager.graph.edges) == initial_edge_count

    # 5. The original page/checksum state remains intact
    assert manager._ingested_deltas[1] == checksum_a
    node_after = manager.graph.get_node("p1_n1")
    assert node_after is not None
    assert node_after.value == "Patient Alpha"
    assert node_after.properties.get("observations") is None or len(node_after.properties.get("observations", [])) == 1


# ==============================================================================
# 4. CONCURRENCY SEMANTICS TESTS (Sections 19, 20, 21, 27)
# ==============================================================================

@async_test
async def test_conc_1_ordered_context_mode_sees_prior_pages():
    """In Mode A (Ordered Context Mode), Page N observes canonical memory from pages 1..N-1."""
    mock_llm = MagicMock()
    received_contexts: Dict[int, List[Dict[str, Any]]] = {}

    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        received_contexts[page_number] = existing_nodes
        if page_number == 1:
            return {
                "entities": [{"id": "p1_pat", "type": "Patient", "label": "Patient", "value": "Ordered Patient"}],
                "relationships": [],
                "reference_resolutions": [],
            }
        else:
            return {
                "entities": [{"id": "p2_hosp", "type": "Hospital", "label": "Hospital", "value": "Ordered Hospital"}],
                "relationships": [],
                "reference_resolutions": [],
            }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Ordered Patient",
        hospital_name="Ordered Hospital",
    )

    pages = [
        {"markdown": "P1 info", "page_number": 1},
        {"markdown": "P2 info", "page_number": 2},
    ]

    await extract_with_graph_memory_ordered(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
    )

    # Page 1 started with empty context
    assert len(received_contexts[1]) == 0
    # Page 2 MUST receive Page 1's entity in its context
    assert len(received_contexts[2]) >= 1
    assert any(c["value"] == "Ordered Patient" for c in received_contexts[2])


@async_test
async def test_conc_2_epoch_batch_mode_behavior():
    """Mode C (Epoch / Batch Mode) runs concurrent workers within epoch and merges between epochs."""
    mock_llm = MagicMock()

    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        return {
            "entities": [{"id": f"p{page_number}_e", "type": "Patient", "value": f"Batch Pat {page_number}"}],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Batch Pat 1")

    pages = [{"markdown": f"Content {p}", "page_number": p} for p in range(1, 5)]
    graph_out: Dict[str, Any] = {}

    await extract_with_graph_memory_batched(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        epoch_size=2,  # 2 epochs of 2 pages
        graph_out=graph_out,
    )

    assert graph_out["strategy"] == "graph_memory_batched"
    assert graph_out["pages_succeeded"] == 4
    assert graph_out["stats"]["total_nodes"] == 4


@async_test
async def test_conc_3_workers_do_not_mutate_shared_graph_directly():
    """Page workers building PageDelta do NOT mutate the central GraphMemory directly."""
    manager = GraphMemoryManager()
    mock_llm = MagicMock()
    mock_llm.extract_graph_from_page.return_value = {
        "entities": [{"id": "p1_worker_node", "type": "Patient", "value": "Worker Only"}],
        "relationships": [],
        "reference_resolutions": [],
    }
    agent = GraphExtractionAgent(schema=PatientClaimSchema, graph=manager.graph, llm=mock_llm)

    assert len(manager.graph.nodes) == 0

    delta = await agent.build_page_delta_async(
        page_number=1,
        markdown="Text",
        total_pages=1,
        context=[],
    )

    # Delta contains the entity, but manager.graph MUST still be empty
    assert len(delta.entities) == 1
    assert len(manager.graph.nodes) == 0


@async_test
async def test_unc_4_later_evidence_promotes_uncertain_reference():
    """Uncertain entity on Page 1 is promoted to confirmed canonical match when Page 2 provides corroborating evidence."""
    manager = GraphMemoryManager(anchor_types=["Patient"])

    # Page 1: Initial vague mention creates UNCERTAIN candidate
    delta1 = PageDelta(
        page_number=1,
        entities=[
            GraphNode(
                id="pat_unc_1",
                type="Patient",
                label="Patient",
                value="P. Sharma",
                status="UNCERTAIN",
                source_pages=[1],
            )
        ],
    )
    await manager.ingest_page_delta(delta1)
    assert len(manager.graph.nodes) == 1
    assert list(manager.graph.nodes.values())[0].status == "UNCERTAIN"

    # Page 2: Authoritative explicit mention with full name and matching alias
    delta2 = PageDelta(
        page_number=2,
        entities=[
            GraphNode(
                id="pat_auth_2",
                type="Patient",
                label="Patient",
                value="Pooja Sharma",
                aliases=["P. Sharma"],
                status="ASSERTED",
                evidence=[Evidence(page_number=2, text="Patient Full Name: Pooja Sharma (P. Sharma)")],
                source_pages=[2],
            )
        ],
    )
    await manager.ingest_page_delta(delta2)

    # Post-merge reconciliation: evaluate if uncertain nodes can now be promoted
    await manager.reconcile_cross_references()

    # Canonical memory now has confirmed Pooja Sharma with alias P. Sharma
    nodes = list(manager.graph.nodes.values())
    assert any(n.value == "Pooja Sharma" and "P. Sharma" in n.aliases for n in nodes)


def test_chk_5_same_key_different_checksum_detected():
    """When a checkpoint key is saved with a different checksum, attempt count increments and updated checksum is recorded."""
    doc_id = "test_doc_conflict_5"
    delete_page_checkpoints(doc_id)

    delta_v1 = {"page_number": 1, "entities": [{"id": "n1", "value": "Version 1"}]}
    chk1 = compute_page_delta_checksum(delta_v1)
    save_page_checkpoint(doc_id=doc_id, page_number=1, status="DELTA_PERSISTED", page_delta_json=delta_v1, delta_checksum=chk1)

    # Re-saving with different content / checksum
    delta_v2 = {"page_number": 1, "entities": [{"id": "n1", "value": "Version 2 Modified"}]}
    chk2 = compute_page_delta_checksum(delta_v2)
    save_page_checkpoint(doc_id=doc_id, page_number=1, status="DELTA_PERSISTED", page_delta_json=delta_v2, delta_checksum=chk2)

    cp = get_page_checkpoint(doc_id=doc_id, page_number=1)
    assert cp is not None
    assert cp["delta_checksum"] == chk2
    assert cp["attempt_count"] == 2

    delete_page_checkpoints(doc_id)


@async_test
async def test_chk_6_worker_failure_persists_failed_state_and_allows_retry():
    """Worker failure marks checkpoint as FAILED, allowing subsequent retry to succeed."""
    doc_id = "test_doc_retry_6"
    delete_page_checkpoints(doc_id)

    mock_llm = MagicMock()
    attempt = 0

    def mock_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("Temporary upstream timeout")
        return {
            "entities": [{"id": "p1_e", "type": "Patient", "label": "patient_name", "value": "Retried Patient"}],
            "relationships": [],
            "reference_resolutions": [],
        }

    mock_llm.extract_graph_from_page.side_effect = mock_extract
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Retried Patient")

    pages = [{"markdown": "Patient content", "page_number": 1}]

    # Run 1: fails
    out1: Dict[str, Any] = {}
    await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        doc_id=doc_id,
        graph_out=out1,
    )
    assert out1["pages_failed"] == 1
    cp1 = get_page_checkpoint(doc_id=doc_id, page_number=1)
    assert cp1["status"] == "FAILED"
    assert "timeout" in cp1["error_info"].lower()

    # Run 2: retries and succeeds!
    out2: Dict[str, Any] = {}
    res = await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=PatientClaimSchema,
        llm=mock_llm,
        doc_id=doc_id,
        graph_out=out2,
    )
    assert out2["pages_succeeded"] == 1
    assert res.patient_name == "Retried Patient"
    cp2 = get_page_checkpoint(doc_id=doc_id, page_number=1)
    assert cp2["status"] == "COMPLETED"

    delete_page_checkpoints(doc_id)


@async_test
async def test_reconciliation_idempotency_and_ambiguous_retention():
    """
    Focused regression test for cross-reference reconciliation idempotency:
    A. Create GraphMemoryManager(anchor_types=["Patient"]).
    B. Ingest Page 1 containing a Patient node.
    C. Ingest Page 2 containing an Admission node plus an UnresolvedRef ("the above patient" -> Patient).
    D. Call await manager.reconcile_cross_references().
    E. Verify:
       - exactly one REFERS_TO edge exists
       - edge points to the correct patient
       - edge status is INFERRED
       - unresolved_count no longer includes that resolved reference
    F. Call reconcile_cross_references() again.
    G. Verify:
       - still exactly one REFERS_TO edge
       - no duplicate REFERS_TO edge was created
       - unresolved_count remains zero
    H. Verify that an ambiguous unresolved reference remains pending after reconciliation,
       so later evidence can still resolve it.
    """
    # A. Create GraphMemoryManager(anchor_types=["Patient"])
    manager = GraphMemoryManager(anchor_types=["Patient"])

    # B. Ingest Page 1 containing a Patient node
    p1_node = GraphNode(
        id="pat_001",
        type="Patient",
        label="patient_name",
        value="John Doe",
        source_pages=[1],
        confidence=1.0,
        status="ASSERTED",
    )
    delta1 = PageDelta(page_number=1, entities=[p1_node])
    await manager.ingest_page_delta(delta1)

    # C. Ingest Page 2 containing an Admission node plus an UnresolvedRef
    p2_adm = GraphNode(
        id="adm_001",
        type="Admission",
        label="admission_record",
        value="Admission #12345",
        source_pages=[2],
        confidence=1.0,
        status="ASSERTED",
    )
    unresolved_ref = UnresolvedRef(
        source_node_id="adm_001",
        phrase="the above patient",
        target_type="Patient",
        page_number=2,
        rationale="Page 2 refers to the above patient admitted previously",
    )
    delta2 = PageDelta(
        page_number=2,
        entities=[p2_adm],
        unresolved_references=[unresolved_ref],
    )
    await manager.ingest_page_delta(delta2)

    assert manager.unresolved_count == 1

    # D. Call reconcile_cross_references()
    await manager.reconcile_cross_references()

    # E. Verify:
    # - exactly one REFERS_TO edge exists
    # - edge points to the correct patient
    # - edge status is INFERRED
    # - unresolved_count no longer includes that resolved reference
    ref_edges = manager.graph.get_edges(source_node="adm_001", relationship="REFERS_TO")
    assert len(ref_edges) == 1
    assert ref_edges[0].target_node == "pat_001"
    assert ref_edges[0].status == "INFERRED"
    assert manager.unresolved_count == 0

    # F. Call reconcile_cross_references() again
    await manager.reconcile_cross_references()

    # G. Verify:
    # - still exactly one REFERS_TO edge
    # - no duplicate REFERS_TO edge was created
    # - unresolved_count remains zero
    ref_edges_second = manager.graph.get_edges(source_node="adm_001", relationship="REFERS_TO")
    assert len(ref_edges_second) == 1
    assert manager.unresolved_count == 0

    # H. Verify that an ambiguous unresolved reference remains pending after reconciliation,
    # so later evidence can still resolve it.
    amb_manager = GraphMemoryManager(anchor_types=["Patient"])
    cand_a = GraphNode(id="pat_a", type="Patient", label="Patient", value="Alice Smith", source_pages=[1], confidence=1.0)
    cand_b = GraphNode(id="pat_b", type="Patient", label="Patient", value="Bob Jones", source_pages=[1], confidence=1.0)
    await amb_manager.ingest_page_delta(PageDelta(page_number=1, entities=[cand_a, cand_b]))

    amb_rec = GraphNode(id="rec_01", type="Record", label="Record", value="Chart 1", source_pages=[2])
    amb_ref = UnresolvedRef(
        source_node_id="rec_01",
        phrase="the patient",
        target_type="Patient",
        page_number=2,
    )
    await amb_manager.ingest_page_delta(PageDelta(page_number=2, entities=[amb_rec], unresolved_references=[amb_ref]))

    assert amb_manager.unresolved_count == 1

    # Ambiguous reconciliation attempt
    await amb_manager.reconcile_cross_references()

    # Must NOT create an edge and must remain pending
    assert len(amb_manager.graph.get_edges(source_node="rec_01", relationship="REFERS_TO")) == 0
    assert amb_manager.unresolved_count == 1

    # Later evidence arrives on Page 3 that provides stronger proximity / category evidence for Alice Smith
    p3_node = GraphNode(id="pat_a_p3", type="Patient", label="Patient", value="Alice Smith", source_pages=[3], confidence=1.0, category="EXPLICIT")
    await amb_manager.ingest_page_delta(PageDelta(page_number=3, entities=[p3_node]))

    # Now reconcile again
    await amb_manager.reconcile_cross_references()

    # Now Alice Smith has category EXPLICIT, breaking the tie and resolving the ambiguity
    resolved_edges = amb_manager.graph.get_edges(source_node="rec_01", relationship="REFERS_TO")
    assert len(resolved_edges) == 1
    assert resolved_edges[0].target_node == "pat_a"
    assert amb_manager.unresolved_count == 0


@async_test
async def test_relationship_endpoint_unsafe_search_fallback_removed():
    """
    Focused regression test proving:
    1. A relationship whose endpoint is an unknown textual value does NOT get attached
       to an unrelated node merely because graph.search_nodes() could return a result.
    2. An ambiguous/loosely searchable endpoint does NOT result in an arbitrary first-match relationship.
    3. Existing safe behavior still works:
       - an endpoint resolved through _id_remap is used correctly;
       - an exact existing canonical node ID is used correctly.
    """
    manager = GraphMemoryManager()

    # Step 1: Ingest Page 1 with canonical entities that loose search could match
    pat_1 = GraphNode(id="pat_apollo", type="Patient", label="Patient", value="Apollo Creed", source_pages=[1])
    hosp_1 = GraphNode(id="hosp_apollo", type="Hospital", label="Hospital", value="Apollo Hospitals", source_pages=[1])
    await manager.ingest_page_delta(PageDelta(page_number=1, entities=[pat_1, hosp_1]))

    # Confirm that graph.search_nodes("Apollo") would return results
    loose_results = manager.graph.search_nodes(query="Apollo", limit=5)
    assert len(loose_results) >= 2

    # Step 2: Ingest Page 2 with a new entity and an edge whose target is loose textual query "Apollo"
    # Under old logic, "Apollo" would loose-search and arbitrarily pick pat_apollo or hosp_apollo!
    doc_1 = GraphNode(id="p2_doc1", type="Doctor", label="Doctor", value="Dr. Strange", source_pages=[2])
    unsafe_edge = GraphEdge(
        id="edge_unsafe_1",
        source_node="p2_doc1",
        target_node="Apollo",  # Loose query, not in _id_remap, not an exact node ID
        relationship="WORKS_AT",
        source_page=2,
    )
    delta_unsafe = PageDelta(page_number=2, entities=[doc_1], relationships=[unsafe_edge])
    await manager.ingest_page_delta(delta_unsafe)

    # Verifications 1 & 2:
    # The unsafe edge MUST NOT be created
    assert len(manager.graph.edges) == 0
    assert len(manager.graph.get_edges(source_node="p2_doc1")) == 0
    assert len(manager.graph.get_edges(target_node="pat_apollo")) == 0
    assert len(manager.graph.get_edges(target_node="hosp_apollo")) == 0

    # Step 3: Verify existing safe behavior works:
    # 3a. Target resolved through _id_remap on the same page
    hosp_local = GraphNode(id="p2_hosp", type="Hospital", label="Hospital", value="City Clinic", source_pages=[2])
    safe_edge_remap = GraphEdge(
        id="edge_safe_remap",
        source_node="p2_doc1",  # in _id_remap
        target_node="p2_hosp",  # in _id_remap
        relationship="AFFILIATED_WITH",
        source_page=2,
    )
    # 3b. Target resolved through exact existing canonical node ID ("hosp_apollo")
    safe_edge_canon = GraphEdge(
        id="edge_safe_canon",
        source_node="p2_doc1",      # in _id_remap
        target_node="hosp_apollo",  # exact canonical node ID in self.graph._nodes
        relationship="CONSULTANT_AT",
        source_page=2,
    )
    delta_safe = PageDelta(
        page_number=2,
        entities=[hosp_local],
        relationships=[safe_edge_remap, safe_edge_canon],
    )
    await manager.ingest_page_delta(delta_safe)

    # Verify both safe edges were created
    edges = list(manager.graph.edges.values())
    assert len(edges) == 2

    # Verify edge 3a (remap -> remap)
    remap_edges = manager.graph.get_edges(source_node="p2_doc1", target_node="p2_hosp", relationship="AFFILIATED_WITH")
    assert len(remap_edges) == 1
    assert remap_edges[0].source_page == 2

    # Verify edge 3b (remap -> exact canonical ID)
    canon_edges = manager.graph.get_edges(source_node="p2_doc1", target_node="hosp_apollo", relationship="CONSULTANT_AT")
    assert len(canon_edges) == 1
    assert canon_edges[0].source_page == 2


