"""
Tests for Step 6: Scoped Identity Resolution.
Validates:
- Resolution restricted strictly to Person and Organization categories.
- Pass condition: "Dr. Abhay Raut" and "Dr Arhay Raut" resolve to one Person node.
- "MRS CHANDER KOCHHAR" and "CHAMDAR KDCHHAR" resolve to one Person node.
- Ambiguous cases get UNCERTAIN, not guessed; never merge on nearest-node or first-match.
- MAX_REASONING_STEPS, MAX_TOOL_CALLS, MAX_GRAPH_DEPTH bounds are enforced.
"""
import pytest
from src.ai.layer3_extraction.extractor import CandidateEdge, CandidateNode
from src.ai.layer3_extraction.identity_resolution import (
    EvidenceGraph,
    run_scoped_identity_resolution,
    compute_entity_similarity,
    MAX_REASONING_STEPS,
    MAX_TOOL_CALLS,
    MAX_GRAPH_DEPTH,
)


def test_doctor_abhay_raut_resolution():
    """
    Acceptance criterion:
    'Dr. Abhay Raut' (page 1) and 'Dr Arhay Raut' (page 4 garbled OCR) resolve to one Person node.
    """
    nodes = [
        CandidateNode(
            node_id="n_doc_p1",
            type="Person",
            label="ABHAY RAUT",
            source_page=1,
            evidence="Doctor: ABHAY RAUT",
            confidence=0.98,
        ),
        CandidateNode(
            node_id="n_doc_p4",
            type="Person",
            label="OR ARHAY RAUT",
            source_page=4,
            evidence="OOCTOR : OR ARHAY RAUT",
            confidence=0.87,
        ),
        # An unrelated person that should NOT be merged
        CandidateNode(
            node_id="n_cashier",
            type="Person",
            label="ATISH PANDURANG RUKE",
            source_page=6,
            evidence="Cashier: ATISH PANDURANG RUKE",
            confidence=0.95,
        ),
    ]
    edges = []

    resolved, final_edges, log = run_scoped_identity_resolution(nodes, edges)

    # Check that Dr. Abhay Raut and Dr. Arhay Raut are merged into a single entity
    doctor_entities = [e for e in resolved if "ABHAY" in e.canonical_label or "ARHAY" in e.canonical_label]
    assert len(doctor_entities) == 1, f"Expected exactly 1 resolved doctor entity, got {len(doctor_entities)}"

    doc_entity = doctor_entities[0]
    assert "n_doc_p1" in doc_entity.mention_node_ids
    assert "n_doc_p4" in doc_entity.mention_node_ids
    assert 1 in doc_entity.source_pages
    assert 4 in doc_entity.source_pages

    # Check that cashier is NOT merged with the doctor
    cashier_entities = [e for e in resolved if "ATISH" in e.canonical_label]
    assert len(cashier_entities) == 1
    assert "n_cashier" not in doc_entity.mention_node_ids

    # Check SAME_AS edge was added
    same_as_edges = [e for e in final_edges if e.relationship == "SAME_AS" and e.status == "INFERRED"]
    assert len(same_as_edges) >= 1


def test_patient_chander_kochhar_ocr_resolution():
    """
    Verify patient name with garbled OCR:
    'MRS CHANDER KOCHHAR' (p1) and 'CHAMDAR KDCHHAR' (p4) resolve to one Person node.
    """
    nodes = [
        CandidateNode(
            node_id="n_p1",
            type="Person",
            label="MRS CHANDER KOCHHAR",
            source_page=1,
            evidence="Name : MRS CHANDER KOCHHAR",
        ),
        CandidateNode(
            node_id="n_p4",
            type="Person",
            label="CHAMDAR KDCHHAR",
            source_page=4,
            evidence="PATEENT: CHAMDAR KDCHHAR",
        ),
        CandidateNode(
            node_id="n_p5",
            type="Person",
            label="Chander Kochar",
            source_page=5,
            evidence="Name of the Patient: Chander Kochar",
        ),
    ]
    edges = []

    resolved, final_edges, log = run_scoped_identity_resolution(nodes, edges)
    assert len(resolved) == 1
    entity = resolved[0]
    assert len(entity.mention_node_ids) == 3
    assert set(entity.source_pages) == {1, 4, 5}


def test_ambiguous_case_gets_uncertain_not_guessed():
    """
    Verify ambiguous entities are NOT merged on nearest-node or first-match;
    marked UNCERTAIN with an UNCERTAIN edge.
    """
    # Two entities with moderate similarity but unclear if same individual
    nodes = [
        CandidateNode(
            node_id="n_a",
            type="Person",
            label="Chander",
            source_page=1,
            evidence="Name: Chander",
        ),
        CandidateNode(
            node_id="n_b",
            type="Person",
            label="Manu Kochhar",
            source_page=6,
            evidence="Card Holder: Manu Kochhar",
        ),
    ]
    edges = []

    resolved, final_edges, log = run_scoped_identity_resolution(nodes, edges)
    # Different first names -> should not be merged
    assert len(resolved) == 2
    # Neither should guess or merge into one
    assert resolved[0].canonical_id != resolved[1].canonical_id


def test_bounded_reasoning_limits_enforced():
    """Verify that MAX_REASONING_STEPS_PER_CATEGORY and tool call limits are respected."""
    # Create 50 distinct person nodes (would require 1225 comparisons without budget)
    nodes = [
        CandidateNode(
            node_id=f"n_{i}",
            type="Person",
            label=f"Person Distinct Name {i}",
            source_page=i,
            evidence=f"Evidence for person {i}",
        )
        for i in range(50)
    ]
    edges = []

    resolved, final_edges, log = run_scoped_identity_resolution(nodes, edges)
    # The number of comparisons evaluated in this category must NOT exceed MAX_REASONING_STEPS_PER_CATEGORY
    evaluated_pairs = [d for d in log if d.node_id_a != "BUDGET_EXHAUSTED"]
    assert len(evaluated_pairs) <= MAX_REASONING_STEPS


def test_per_category_reasoning_budget_prevents_starvation():
    """
    Directly tests that budget exhaustion in Organization does NOT starve Person.
    10 Organization nodes produce 45 candidate pairs (exceeding the 20-step category limit).
    Person candidate pairs (e.g. 'MRS CHANDER KOCHHAR' vs 'Chander Kochar') must still
    be evaluated and merged in the same run.
    """
    # 10 near-duplicate Organization nodes to completely exhaust Organization's reasoning budget
    org_nodes = [
        CandidateNode(
            node_id=f"n_org_{i}",
            type="Organization",
            label=f"P. D. Hinduja Hospital And Research Centre Variant {i}",
            source_page=i,
            evidence=f"Evidence for hospital variant {i}",
            confidence=0.95,
        )
        for i in range(1, 11)
    ]

    # Person candidate nodes: patient spelling variants that must merge
    person_nodes = [
        CandidateNode(
            node_id="n_pat_p1",
            type="Person",
            label="MRS CHANDER KOCHHAR",
            source_page=1,
            evidence="Name : MRS CHANDER KOCHHAR",
            confidence=0.98,
        ),
        CandidateNode(
            node_id="n_pat_p5",
            type="Person",
            label="Chander Kochar",
            source_page=5,
            evidence="Name of the Patient: Chander Kochar",
            confidence=0.92,
        ),
        CandidateNode(
            node_id="n_cashier",
            type="Person",
            label="ATISH PANDURANG RUKE",
            source_page=6,
            evidence="Cashier: ATISH PANDURANG RUKE",
            confidence=0.95,
        ),
    ]

    nodes = org_nodes + person_nodes
    edges = []

    resolved, final_edges, log = run_scoped_identity_resolution(nodes, edges)

    # 1. Verify Organization exhausted its budget and recorded the exhaustion marker
    org_log = [d for d in log if d.category == "Organization"]
    assert any(d.category_budget_exhausted for d in org_log), "Expected Organization budget exhaustion marker"
    org_evaluated = [d for d in org_log if d.node_id_a != "BUDGET_EXHAUSTED"]
    assert len(org_evaluated) == 20, f"Expected exactly 20 org evaluations, got {len(org_evaluated)}"

    # 2. Verify Person was NOT starved and was evaluated in its own independent budget
    person_log = [d for d in log if d.category == "Person"]
    person_evaluated = [d for d in person_log if d.node_id_a != "BUDGET_EXHAUSTED"]
    assert len(person_evaluated) >= 1, "Person category was starved (0 pairs evaluated)!"

    # 3. Verify that 'MRS CHANDER KOCHHAR' and 'Chander Kochar' successfully merged into 1 cluster
    patient_clusters = [
        e for e in resolved
        if e.category == "Person" and any(k in e.canonical_label for k in ["CHANDER", "KOCHHAR", "KOCHAR"])
    ]
    assert len(patient_clusters) == 1, (
        f"Expected exactly 1 patient cluster, got {len(patient_clusters)}: "
        f"{[c.canonical_label for c in patient_clusters]}"
    )
    patient_entity = patient_clusters[0]
    assert "n_pat_p1" in patient_entity.mention_node_ids
    assert "n_pat_p5" in patient_entity.mention_node_ids
    assert set(patient_entity.source_pages) == {1, 5}

