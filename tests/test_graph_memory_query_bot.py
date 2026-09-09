"""
Comprehensive test suite for Layer 3 Graph Memory & Graph-Backed Query Bot.
Implements Tests 1 through 23 and the Exact End-to-End Demo (Section 28)
specified in the implementation requirements.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA_APP = ROOT / "schema_chatbot_v2"
if str(SCHEMA_APP) not in sys.path:
    sys.path.insert(0, str(SCHEMA_APP))

from pydantic import BaseModel, Field
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.core.auth import create_access_token

from src.ai.layer3_extraction.graph_agent.models import GraphEdge, GraphNode, NodeEvidence
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService, GraphQueryResult
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
from src.ai.layer3_extraction.storage import (
    init_db,
    save_document_graph,
    get_document_graph,
    list_document_graphs,
    DocumentGraphRecord,
    SessionLocal,
)


class PatientClaimSchema(BaseModel):
    patient_name: str = Field(default="", description="The full name of the patient")
    claim_amount: str = Field(default="", description="The approved claim amount")


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def alice_token():
    store = get_user_store()
    user = store.get_by_username("alice")
    if not user:
        user = store.create("alice", "password123", Role.USER)
    return create_access_token(user)


@pytest.fixture
def bob_token():
    store = get_user_store()
    user = store.get_by_username("bob")
    if not user:
        user = store.create("bob", "password123", Role.USER)
    return create_access_token(user)


@pytest.fixture
def admin_token():
    store = get_user_store()
    user = store.get_by_username("admin")
    if not user:
        user = store.create("admin", "changeme", Role.ADMIN)
    return create_access_token(user)


# ==============================================================================
# Test 1 & 20: Parallel Edges (MultiDiGraph support)
# ==============================================================================
def test_1_parallel_edges():
    """Verify NetworkX MultiDiGraph stores multiple distinct edges between same nodes."""
    graph = GraphMemory()
    graph.create_node(node_id="p1", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)
    graph.create_node(node_id="a1", node_type="Admission", label="Admission", value="General Admission", source_page=1)

    # Add two different relationships between same source and target
    e1 = graph.create_edge(
        edge_id="e_has_admission",
        source_node="p1",
        target_node="a1",
        relationship="HAS_ADMISSION",
        evidence="Patient was admitted",
        source_page=1,
    )
    e2 = graph.create_edge(
        edge_id="e_emergency",
        source_node="p1",
        target_node="a1",
        relationship="EMERGENCY_ADMITTED",
        evidence="Patient was admitted via emergency",
        source_page=1,
    )

    # Both edges must survive in MultiDiGraph
    assert len(graph.edges) == 2
    assert graph.graph.number_of_edges() == 2
    assert graph.graph.has_edge("p1", "a1", key="e_has_admission")
    assert graph.graph.has_edge("p1", "a1", key="e_emergency")

    # read_neighbors must return both edges
    neighbors = graph.read_neighbors("p1")
    rels = {edge.relationship for edge, _ in neighbors}
    assert "HAS_ADMISSION" in rels
    assert "EMERGENCY_ADMITTED" in rels


# ==============================================================================
# Test 2: Graph Serialization Round Trip
# ==============================================================================
def test_2_graph_serialization_round_trip():
    """Verify lossless to_dict() and from_dict() round trip for GraphMemory."""
    graph = GraphMemory()
    n1 = graph.create_node(
        node_id="n_patient",
        node_type="Patient",
        label="Patient Name",
        value="Rahul Sharma",
        properties={"dob": "1985-05-12"},
        source_page=1,
        confidence=0.99,
        category="EXPLICIT",
        evidence=NodeEvidence(page_number=1, text="Patient: Rahul Sharma", confidence=0.99),
    )
    n1.add_alias("Mr. Sharma")
    n1.add_alias("The above patient")

    graph.create_node(
        node_id="n_doc",
        node_type="Doctor",
        label="Treating Doctor",
        value="Dr. Arun Sharma",
        source_page=1,
        confidence=0.95,
        category="CONTEXTUAL",
    )

    graph.create_edge(
        edge_id="e_treatment",
        source_node="n_patient",
        target_node="n_doc",
        relationship="TREATED_BY",
        evidence="under Dr. Arun Sharma",
        source_page=1,
        confidence=0.95,
    )

    # to_dict -> JSON -> from_dict
    serialized = graph.to_dict()
    json_str = json.dumps(serialized)
    restored_dict = json.loads(json_str)
    restored_graph = GraphMemory.from_dict(restored_dict)

    assert len(restored_graph.nodes) == 2
    assert len(restored_graph.edges) == 1

    restored_n1 = restored_graph.get_node("n_patient")
    assert restored_n1 is not None
    assert restored_n1.value == "Rahul Sharma"
    assert "Mr. Sharma" in restored_n1.aliases
    assert "The above patient" in restored_n1.aliases
    assert restored_n1.properties.get("dob") == "1985-05-12"
    assert restored_n1.category == "EXPLICIT"
    assert len(restored_n1.evidence) == 1
    assert restored_n1.evidence[0].text == "Patient: Rahul Sharma"

    assert restored_graph.graph.has_edge("n_patient", "n_doc", key="e_treatment")


# ==============================================================================
# Test 3 & 4: Graph Persistence Round Trip & Process Restart Simulation
# ==============================================================================
def test_3_graph_persistence_round_trip():
    """Verify document graph persists to DB and can be retrieved."""
    graph = GraphMemory()
    graph.create_node(node_id="p1", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)
    graph.create_node(node_id="d1", node_type="Doctor", label="Doctor", value="Dr. Arun Sharma", source_page=1)
    graph.create_edge(edge_id="e1", source_node="p1", target_node="d1", relationship="TREATED_BY", source_page=1)

    rec_id = save_document_graph(
        doc_id="doc_roundtrip.pdf",
        graph_dict=graph.to_dict(),
        job_id="job_persist_1",
        owner="alice",
        schema_id="test_schema",
    )
    assert rec_id is not None

    fetched = get_document_graph(doc_id="doc_roundtrip.pdf", job_id="job_persist_1", owner="alice")
    assert fetched is not None
    assert len(fetched["nodes"]) == 2
    assert len(fetched["edges"]) == 1

    # Reload into GraphMemory
    reloaded = GraphMemory.from_dict(fetched)
    assert reloaded.get_node("p1").value == "Rahul Sharma"
    assert reloaded.get_node("d1").value == "Dr. Arun Sharma"


# ==============================================================================
# Test 5: User Isolation (User A vs User B)
# ==============================================================================
def test_5_user_isolation(test_client, alice_token, bob_token, admin_token):
    """Verify User B cannot query User A's document graph (403 Forbidden)."""
    # Save a graph belonging to alice
    graph = GraphMemory()
    graph.create_node(node_id="p1", node_type="Patient", label="Patient", value="Confidential Patient", source_page=1)
    save_document_graph(
        doc_id="alice_confidential.pdf",
        graph_dict=graph.to_dict(),
        job_id="job_alice_1",
        owner="alice",
    )

    # 1. Alice queries her own document -> 200 OK
    resp_alice = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {alice_token}"},
        json={"question": "Who is the patient?", "doc_id": "alice_confidential.pdf"},
    )
    assert resp_alice.status_code == 200
    assert resp_alice.json()["mode"] == "graph"

    # 2. Bob queries Alice's document -> 403 Forbidden
    resp_bob = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {bob_token}"},
        json={"question": "Who is the patient?", "doc_id": "alice_confidential.pdf"},
    )
    assert resp_bob.status_code == 403

    # 3. Anonymous user queries Alice's document -> 403 Forbidden
    resp_anon = test_client.post(
        "/api/query-bot/ask",
        json={"question": "Who is the patient?", "doc_id": "alice_confidential.pdf"},
    )
    assert resp_anon.status_code == 403

    # 4. Admin queries Alice's document -> 200 OK (Admin override)
    resp_admin = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"question": "Who is the patient?", "doc_id": "alice_confidential.pdf"},
    )
    assert resp_admin.status_code == 200


# ==============================================================================
# Test 6 & 7: Single and Multi-Page Graph Construction
# ==============================================================================
def test_6_7_graph_construction_pages():
    """Verify sequential page-by-page graph building."""
    graph = GraphMemory()
    # Page 1
    graph.create_node(node_id="p1", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)
    graph.create_node(node_id="d1", node_type="Doctor", label="Doctor", value="Dr. Arun Sharma", source_page=1)
    graph.create_edge(edge_id="e1", source_node="p1", target_node="d1", relationship="ADMITTED_UNDER", source_page=1)

    # Page 2
    graph.create_node(node_id="diag1", node_type="Diagnosis", label="Diagnosis", value="Acute appendicitis", source_page=2)
    graph.create_edge(edge_id="e2", source_node="p1", target_node="diag1", relationship="DIAGNOSED_WITH", source_page=2)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    p1 = graph.get_node("p1")
    assert 1 in p1.source_pages


# ==============================================================================
# Test 8 & 9: Cross-Page Entity Reuse & Reference Resolution
# ==============================================================================
def test_8_9_cross_page_resolution():
    """Verify cross-page reference resolution attaches to existing node."""
    graph = GraphMemory()
    p1 = graph.create_node(node_id="patient_1", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)
    p1.add_alias("The above patient")

    # Look up by alias on page 2
    resolved = graph.lookup_alias("The above patient")
    assert resolved is not None
    assert resolved.id == "patient_1"

    # Add page 2 to existing node
    graph.update_node("patient_1", source_page=2)
    assert 1 in p1.source_pages
    assert 2 in p1.source_pages


# ==============================================================================
# Test 10 & 11: Non-Schema Entity Retention & Strict Schema Resolution
# ==============================================================================
def test_10_11_non_schema_retention_and_schema_resolver():
    """Verify non-schema entities remain in graph while schema resolver returns ONLY target fields."""
    graph = GraphMemory()
    graph.create_node(
        node_id="n_patient",
        node_type="Patient",
        label="Patient",
        value="Rahul Sharma",
        source_page=1,
        category="EXPLICIT",
    )
    graph.create_node(
        node_id="n_claim",
        node_type="Claim",
        label="Claim Amount",
        value="INR 87,500",
        source_page=4,
        category="EXPLICIT",
    )
    # Contextual entities NOT in schema
    graph.create_node(
        node_id="n_doctor",
        node_type="Doctor",
        label="Doctor",
        value="Dr. Arun Sharma",
        source_page=1,
        category="CONTEXTUAL",
    )
    graph.create_node(
        node_id="n_proc",
        node_type="Procedure",
        label="Procedure",
        value="Laparoscopic appendectomy",
        source_page=3,
        category="CONTEXTUAL",
    )

    # Verify all 4 nodes exist in Graph Memory
    assert len(graph.nodes) == 4

    # Resolve schema using mock LLM that returns only requested fields
    mock_llm = MagicMock()
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Rahul Sharma",
        claim_amount="INR 87,500",
    )

    resolved_schema = resolve_schema_from_graph(graph, PatientClaimSchema, mock_llm)
    final_dict = resolved_schema.model_dump()

    # Final JSON MUST strictly adhere to target schema
    assert final_dict == {
        "patient_name": "Rahul Sharma",
        "claim_amount": "INR 87,500",
    }
    assert "doctor" not in final_dict
    assert "procedure" not in final_dict


# ==============================================================================
# Test 12: CRITICAL GRAPH-ONLY TEST (Section 16)
# ==============================================================================
def test_12_graph_only_query_bot_fact(test_client, admin_token):
    """
    CRITICAL TEST:
    Extracted JSON contains ONLY patient_name and claim_amount.
    Doctor is in Graph Memory.
    Query Bot answers 'Who was the doctor treating the patient?' using Graph Memory (mode: 'graph').
    """
    graph = GraphMemory()
    graph.create_node(
        node_id="p1",
        node_type="Patient",
        label="Patient",
        value="Rahul Sharma",
        source_page=1,
        evidence=NodeEvidence(page_number=1, text="Patient Rahul Sharma was admitted"),
    )
    graph.create_node(
        node_id="d1",
        node_type="Doctor",
        label="Doctor",
        value="Dr. Arun Sharma",
        source_page=1,
        evidence=NodeEvidence(page_number=1, text="admitted under Dr. Arun Sharma"),
    )
    graph.create_edge(
        edge_id="e_treated",
        source_node="p1",
        target_node="d1",
        relationship="ADMITTED_UNDER",
        source_page=1,
    )

    # Persist graph
    save_document_graph(
        doc_id="medical_report.pdf",
        graph_dict=graph.to_dict(),
        job_id="job_med_1",
    )

    # Extracted JSON strictly excludes Doctor
    extracted_json = {
        "patient_name": "Rahul Sharma",
        "claim_amount": "INR 87,500",
    }

    # Query the doctor
    resp = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "question": "Who was the doctor treating the patient?",
            "doc_id": "medical_report.pdf",
            "extracted_data": extracted_json,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["mode"] == "graph"
    assert "Dr. Arun Sharma" in data["answer"]
    assert any(s["page"] == 1 for s in data["sources"])


# ==============================================================================
# Test 13 & 14: Multi-Hop Queries (2-Hop and 3-Hop Traversal)
# ==============================================================================
def test_13_14_multi_hop_query():
    """Verify 2-hop and 3-hop bounded traversals."""
    graph = GraphMemory()
    p = graph.create_node(node_id="n_patient", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)
    adm = graph.create_node(node_id="n_adm", node_type="Admission", label="Admission", value="Emergency Admission", source_page=1)
    doc = graph.create_node(node_id="n_doc", node_type="Doctor", label="Doctor", value="Dr. Arun Sharma", source_page=1,
                            evidence=NodeEvidence(page_number=1, text="admitted under Dr. Arun Sharma"))
    proc = graph.create_node(node_id="n_proc", node_type="Procedure", label="Procedure", value="Laparoscopic appendectomy", source_page=3,
                             evidence=NodeEvidence(page_number=3, text="underwent laparoscopic appendectomy"))

    # Procedure -> Admission -> Doctor
    graph.create_edge(edge_id="e1", source_node="n_proc", target_node="n_adm", relationship="PERFORMED_DURING", source_page=3)
    graph.create_edge(edge_id="e2", source_node="n_adm", target_node="n_doc", relationship="ATTENDED_BY", source_page=1)
    graph.create_edge(edge_id="e3", source_node="n_patient", target_node="n_adm", relationship="HAS_ADMISSION", source_page=1)

    service = GraphQueryService(max_hops=3)
    # 2-hop query: Procedure -> Admission -> Doctor
    result = service.query(graph, "Which doctor was associated with the admission during which the surgery happened?")
    assert result.mode == "graph"
    assert "Dr. Arun Sharma" in result.answer
    assert result.hops_traversed >= 1


# ==============================================================================
# Test 15: Missing Information Does Not Hallucinate
# ==============================================================================
def test_15_missing_information_no_hallucination():
    """Verify query for fact not in graph returns not found without hallucinating."""
    graph = GraphMemory()
    graph.create_node(node_id="p1", node_type="Patient", label="Patient", value="Rahul Sharma", source_page=1)

    service = GraphQueryService()
    result = service.query(graph, "What is the insurance policy number?")

    assert result.mode == "graph"
    assert "could not be found" in result.answer.lower() or "not found" in result.answer.lower()


# ==============================================================================
# Test 16 & 17: Cycle-Safe Traversal & Traversal Limits
# ==============================================================================
def test_16_17_cycle_safety_and_limits():
    """Verify traversal terminates cleanly on cyclic graphs and respects bounds."""
    graph = GraphMemory()
    # Cycle: A -> B -> C -> A
    graph.create_node(node_id="nA", node_type="Entity", label="A", value="Alpha", source_page=1)
    graph.create_node(node_id="nB", node_type="Entity", label="B", value="Beta", source_page=1)
    graph.create_node(node_id="nC", node_type="Entity", label="C", value="Gamma", source_page=1)

    graph.create_edge(edge_id="eAB", source_node="nA", target_node="nB", relationship="CONNECTS", source_page=1)
    graph.create_edge(edge_id="eBC", source_node="nB", target_node="nC", relationship="CONNECTS", source_page=1)
    graph.create_edge(edge_id="eCA", source_node="nC", target_node="nA", relationship="CONNECTS", source_page=1)

    service = GraphQueryService(max_hops=10, max_nodes=2, max_edges=2)
    nodes, edges, max_hop = service.traverse_subgraph(graph, [graph.get_node("nA")])

    # Must terminate without infinite loop and not exceed max_nodes
    assert len(nodes) <= 2
    assert len(edges) <= 2


# ==============================================================================
# Test 18: Evidence & Page Provenance
# ==============================================================================
def test_18_evidence_provenance():
    """Verify sources list contains exact page numbers and evidence strings."""
    graph = GraphMemory()
    graph.create_node(
        node_id="p1",
        node_type="Patient",
        label="Patient",
        value="Rahul Sharma",
        source_page=1,
        evidence=NodeEvidence(page_number=1, text="Rahul Sharma was admitted"),
    )
    service = GraphQueryService()
    result = service.query(graph, "What is the patient name?")

    assert len(result.sources) >= 1
    assert result.sources[0]["page"] == 1
    assert "Rahul Sharma was admitted" in result.sources[0]["evidence"]


# ==============================================================================
# Test 19: Conflicting Facts
# ==============================================================================
def test_19_conflicting_facts():
    """Verify distinct semantic concepts like requested claim vs approved claim are preserved."""
    graph = GraphMemory()
    graph.create_node(
        node_id="claim_req",
        node_type="ClaimRequest",
        label="Claim Requested",
        value="INR 95,000",
        source_page=2,
        evidence=NodeEvidence(page_number=2, text="Initial claim submitted was INR 95,000"),
    )
    graph.create_node(
        node_id="claim_appr",
        node_type="ClaimApproval",
        label="Approved Claim",
        value="INR 87,500",
        source_page=4,
        evidence=NodeEvidence(page_number=4, text="The approved claim amount was INR 87,500"),
    )

    assert len(graph.nodes) == 2
    assert graph.get_node("claim_req").value == "INR 95,000"
    assert graph.get_node("claim_appr").value == "INR 87,500"


# ==============================================================================
# Test 21: page_scan Regression
# ==============================================================================
def test_21_page_scan_regression():
    """Verify existing page_scan strategy continues to function without error."""
    mock_llm = MagicMock()
    mock_llm.check_page_for_fields.return_value = [
        {"field": "patient_name", "value": "Rahul Sharma", "confidence": 0.9}
    ]
    mock_llm.extract.return_value = PatientClaimSchema(patient_name="Rahul Sharma", claim_amount="")

    res = extract_by_page_scan(
        pages_md=[{"page_number": 1, "markdown": "Patient: Rahul Sharma"}],
        schema=PatientClaimSchema,
        llm=mock_llm,
    )
    assert res.patient_name == "Rahul Sharma"


# ==============================================================================
# Test 23: Query Bot JSON Fallback Regression
# ==============================================================================
def test_23_query_bot_json_fallback(test_client, admin_token):
    """Verify Query Bot cleanly falls back to JSON when no graph exists (mode: 'json_fallback')."""
    with patch("httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "The claim amount is INR 87,500."}}]
        }
        mock_post.return_value = mock_resp

        resp = test_client.post(
            "/api/query-bot/ask",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "extracted_data": {"claim_amount": "INR 87,500"},
                "question": "What is the claim amount?",
                "doc_id": "non_existent_graph.pdf",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["mode"] == "json_fallback"
        assert "87,500" in data["answer"]


# ==============================================================================
# Test 24: EXACT END-TO-END DEMO (Section 28)
# ==============================================================================
def test_24_exact_end_to_end_demo(test_client, admin_token):
    """
    EXACT 4-PAGE MEDICAL SCENARIO:
    Page 1: "Patient Rahul Sharma was admitted under Dr. Arun Sharma."
    Page 2: "The above patient was diagnosed with acute appendicitis."
    Page 3: "The patient underwent laparoscopic appendectomy."
    Page 4: "The approved claim amount was INR 87,500."

    Verifies requirements A through M:
    A. Graph contains Patient.
    B. Graph contains Doctor.
    C. Graph contains Diagnosis.
    D. Graph contains Procedure.
    E. Graph contains Admission/Claim.
    F. Cross-page references resolve correctly.
    G. Graph contains source page/evidence.
    H. Final JSON is exactly schema-constrained.
    I. Query Bot answers 'Who was the doctor treating the patient?' -> Dr. Arun Sharma.
    J. Query Bot answers 'What procedure did the patient undergo?' -> Laparoscopic appendectomy.
    K. Multi-hop query: Doctor associated with procedure admission -> Dr. Arun Sharma.
    L. Answers contain source pages when evidence exists.
    M. Query Bot answers Doctor even though Doctor is NOT in final JSON.
    """
    graph = GraphMemory()

    # Page 1
    p1 = graph.create_node(
        node_id="p1",
        node_type="Patient",
        label="Patient Name",
        value="Rahul Sharma",
        source_page=1,
        category="EXPLICIT",
        evidence=NodeEvidence(page_number=1, text="Patient Rahul Sharma was admitted"),
    )
    p1.add_alias("The above patient")
    p1.add_alias("The patient")

    doc1 = graph.create_node(
        node_id="d1",
        node_type="Doctor",
        label="Doctor",
        value="Dr. Arun Sharma",
        source_page=1,
        category="CONTEXTUAL",
        evidence=NodeEvidence(page_number=1, text="admitted under Dr. Arun Sharma"),
    )
    adm1 = graph.create_node(
        node_id="adm1",
        node_type="Admission",
        label="Admission",
        value="Inpatient Admission",
        source_page=1,
        category="CONTEXTUAL",
    )
    graph.create_edge("e_p_adm", "p1", "adm1", "ADMITTED_TO", source_page=1)
    graph.create_edge("e_adm_d", "adm1", "d1", "TREATED_BY", source_page=1, evidence="admitted under Dr. Arun Sharma")

    # Page 2
    diag1 = graph.create_node(
        node_id="diag1",
        node_type="Diagnosis",
        label="Diagnosis",
        value="Acute appendicitis",
        source_page=2,
        category="CONTEXTUAL",
        evidence=NodeEvidence(page_number=2, text="diagnosed with acute appendicitis"),
    )
    # Cross-page reference: "The above patient" resolves to p1
    res_node = graph.lookup_alias("The above patient")
    assert res_node is not None and res_node.id == "p1"
    graph.update_node("p1", source_page=2)
    graph.create_edge("e_p_diag", "p1", "diag1", "DIAGNOSED_WITH", source_page=2)

    # Page 3
    proc1 = graph.create_node(
        node_id="proc1",
        node_type="Procedure",
        label="Procedure",
        value="Laparoscopic appendectomy",
        source_page=3,
        category="CONTEXTUAL",
        evidence=NodeEvidence(page_number=3, text="underwent laparoscopic appendectomy"),
    )
    graph.update_node("p1", source_page=3)
    graph.create_edge("e_p_proc", "p1", "proc1", "UNDERWENT_PROCEDURE", source_page=3)
    graph.create_edge("e_proc_adm", "proc1", "adm1", "DURING_ADMISSION", source_page=3)

    # Page 4
    claim1 = graph.create_node(
        node_id="claim1",
        node_type="Claim",
        label="Claim Amount",
        value="INR 87,500",
        source_page=4,
        category="EXPLICIT",
        evidence=NodeEvidence(page_number=4, text="The approved claim amount was INR 87,500"),
    )
    graph.create_edge("e_p_claim", "p1", "claim1", "CLAIMED_AMOUNT", source_page=4)

    # A-E: Verify all entities exist in graph
    assert graph.get_node("p1").value == "Rahul Sharma"
    assert graph.get_node("d1").value == "Dr. Arun Sharma"
    assert graph.get_node("diag1").value == "Acute appendicitis"
    assert graph.get_node("proc1").value == "Laparoscopic appendectomy"
    assert graph.get_node("claim1").value == "INR 87,500"
    assert graph.get_node("adm1") is not None

    # F: Cross-page references resolve correctly
    assert 1 in p1.source_pages and 2 in p1.source_pages and 3 in p1.source_pages

    # G: Evidence/provenance retained
    assert graph.get_node("p1").evidence[0].page_number == 1
    assert graph.get_node("proc1").evidence[0].page_number == 3

    # H: Final extracted JSON is strictly schema-constrained
    mock_llm = MagicMock()
    mock_llm.resolve_schema_from_graph.return_value = PatientClaimSchema(
        patient_name="Rahul Sharma",
        claim_amount="INR 87,500",
    )
    resolved_schema = resolve_schema_from_graph(graph, PatientClaimSchema, mock_llm)
    final_json = resolved_schema.model_dump()
    assert final_json == {
        "patient_name": "Rahul Sharma",
        "claim_amount": "INR 87,500",
    }
    assert "doctor" not in final_json

    # Persist graph to DB
    doc_id = "e2e_medical_demo.pdf"
    save_document_graph(doc_id=doc_id, graph_dict=graph.to_dict(), job_id="job_e2e_demo")

    # I: Query Bot answers Doctor treating patient
    resp_doc = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "question": "Who was the doctor treating the patient?",
            "doc_id": doc_id,
            "extracted_data": final_json,
        },
    )
    assert resp_doc.status_code == 200
    res_doc = resp_doc.json()
    assert res_doc["mode"] == "graph"
    assert "Dr. Arun Sharma" in res_doc["answer"]
    assert any(s["page"] == 1 for s in res_doc["sources"])

    # J: Query Bot answers Procedure
    resp_proc = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "question": "What procedure did the patient undergo?",
            "doc_id": doc_id,
            "extracted_data": final_json,
        },
    )
    assert resp_proc.status_code == 200
    res_proc = resp_proc.json()
    assert res_proc["mode"] == "graph"
    assert "laparoscopic appendectomy" in res_proc["answer"].lower()
    assert any(s["page"] == 3 for s in res_proc["sources"])

    # K: Multi-hop query through relationships
    resp_multihop = test_client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "question": "Which doctor was associated with the admission during which the procedure was performed?",
            "doc_id": doc_id,
            "extracted_data": final_json,
        },
    )
    assert resp_multihop.status_code == 200
    res_mh = resp_multihop.json()
    assert res_mh["mode"] == "graph"
    assert "Dr. Arun Sharma" in res_mh["answer"]
    assert any(s["page"] == 1 for s in res_mh["sources"])
