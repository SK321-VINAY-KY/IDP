"""
File: test_layer3_identifier_matching.py
Purpose: Tests for deterministic, zero-LLM Identifier-type node matching.
         Covers:
         1. Unit test: Page 5 ambulance bill ("98924625") vs Page 57 TPA form ("9892462685")
            fixture -> assert merge with state=INFERRED and both source values preserved in evidence.
         2. Unit test: Raised Rule A exact match (len >= 6 -> ASSERTED 1.0, len 4-5 -> INFERRED 0.75,
            len < 4 -> DIFFERENT).
         3. Unit test: Strict Kind-Gating negative test (Bill No vs Admission No with identical digits
            and Bill No vs Phone No with >= 7 digit prefix overlap -> assert NO merge, NO edges).
         4. Unit test: Explicit UNCERTAIN near-miss band (prefix overlap of 6 digits -> UNCERTAIN edge,
            no cluster merge).
         5. Live integration test: Real 76-page Chander Kochhar run with mocking disabled
            (Sarvam AI) -> verify wall-clock timing, phone merge, kind-gating live, and log trace.

Owner: engineer-a@idp-pilot
Created: 2026-09-07
"""
from pathlib import Path
import time
import pytest

from src.adapters.llm.sarvam_client import get_run_log_file
from src.ai.layer3_extraction.extractor import CandidateEdge, CandidateNode
from src.ai.layer3_extraction.identifier_matching import (
    detect_identifier_kind,
    normalize_identifier_value,
    run_identifier_matching,
)
from src.ai.layer3_extraction.pipeline import run_pipeline


def test_unit_page5_page57_phone_number_merge():
    """
    Test 1: Confirmed real case from Chander Kochhar 76-page sample.
    Page 5 ambulance bill OCR: "98924625" (handwriting-derived, truncated, conf 0.915)
    Page 57 TPA undertaking form: "9892462685" (clearly printed, conf 0.938)
    Both are kind 'phone'. First 7 digits match ("9892462") before page 5 truncates.
    Assert they merge with state=INFERRED and both source values preserved in evidence.
    """
    node_p5 = CandidateNode(
        node_id="n_p5_ambulance",
        type="Identifier",
        label="Phone: 98924625",
        source_page=5,
        evidence="Contact No.98924625",
        confidence=0.915,
    )
    node_p57 = CandidateNode(
        node_id="n_p57_tpa",
        type="Identifier",
        label="Phone: 9892462685",
        source_page=57,
        evidence="9892462685\n9819862685\nContact No",
        confidence=0.938,
    )

    resolved, edges, log = run_identifier_matching([node_p5, node_p57], [])

    # Exactly 1 SAME_AS edge created
    assert len(edges) == 1, f"Expected 1 SAME_AS edge, got {len(edges)}"
    edge = edges[0]
    assert edge.relationship == "SAME_AS"
    assert edge.status == "INFERRED"
    assert edge.confidence == 0.7  # 7 matching prefix digits out of 10 max len

    # Evidence on the edge must preserve BOTH source values and BOTH source pages
    assert "98924625" in edge.evidence
    assert "9892462685" in edge.evidence
    assert "5" in edge.evidence
    assert "57" in edge.evidence

    # Both source nodes must have their evidence updated to preserve both source values
    assert "98924625" in node_p5.evidence
    assert "9892462685" in node_p5.evidence
    assert "98924625" in node_p57.evidence
    assert "9892462685" in node_p57.evidence

    # Exactly 1 resolved entity cluster produced
    assert len(resolved) == 1
    entity = resolved[0]
    assert entity.category == "Identifier"
    assert entity.status == "INFERRED"
    assert entity.canonical_label == "Phone: 9892462685"
    assert entity.aliases == ["Phone: 98924625"]
    assert sorted(entity.source_pages) == [5, 57]
    assert "98924625" in entity.resolution_note
    assert "9892462685" in entity.resolution_note


def test_unit_exact_match_rule_a_raised_thresholds():
    """
    Test 2: Rule A raised thresholds:
    2a: HS No. "192673" (len 6) -> ASSERTED 1.0.
    2b: Low entropy fragment "123" (len 3) -> rejected (DIFFERENT, no edge).
    2c: Length 4-5 exact match (e.g. "12345") with matching kind -> INFERRED 0.75.
    """
    # 2a: High entropy (len 6) exact match
    node_p1 = CandidateNode(
        node_id="n_p1_hs",
        type="Identifier",
        label="HS No: 192673",
        source_page=1,
        evidence="H.S. NO: 192673",
        confidence=1.0,
    )
    node_p6 = CandidateNode(
        node_id="n_p6_hs",
        type="Identifier",
        label="HS No: 192673",
        source_page=6,
        evidence="H.S. NO: 192673",
        confidence=1.0,
    )

    resolved, edges, log = run_identifier_matching([node_p1, node_p6], [])
    assert len(edges) == 1
    assert edges[0].status == "ASSERTED"
    assert edges[0].confidence == 1.0
    assert len(resolved) == 1
    assert resolved[0].status == "ASSERTED"
    assert resolved[0].resolution_confidence == 1.0

    # 2b: Low entropy fragment (len 3) -> Must NOT merge
    node_short1 = CandidateNode(
        node_id="n_s1",
        type="Identifier",
        label="HS No: 123",
        source_page=1,
        evidence="Bed 123",
        confidence=1.0,
    )
    node_short2 = CandidateNode(
        node_id="n_s2",
        type="Identifier",
        label="HS No: 123",
        source_page=2,
        evidence="Room 123",
        confidence=1.0,
    )
    resolved_short, edges_short, log_short = run_identifier_matching([node_short1, node_short2], [])
    assert len(edges_short) == 0, "Length 3 exact match must NOT create an edge"
    assert len(resolved_short) == 2, "Must remain separate entities"
    assert log_short[0].decision == "DIFFERENT"

    # 2c: Intermediate length 4-5 (e.g. len 5) with matching kind -> INFERRED 0.75
    node_mid1 = CandidateNode(
        node_id="n_m1",
        type="Identifier",
        label="HS No: 12345",
        source_page=1,
        evidence="Code 12345",
        confidence=1.0,
    )
    node_mid2 = CandidateNode(
        node_id="n_m2",
        type="Identifier",
        label="HS No: 12345",
        source_page=2,
        evidence="Code 12345",
        confidence=1.0,
    )
    resolved_mid, edges_mid, log_mid = run_identifier_matching([node_mid1, node_mid2], [])
    assert len(edges_mid) == 1
    assert edges_mid[0].status == "INFERRED"
    assert edges_mid[0].confidence == 0.75
    assert len(resolved_mid) == 1
    assert resolved_mid[0].status == "INFERRED"


def test_unit_kind_gating_negative_tests():
    """
    Test 3: CRITICAL KIND-GATING TEST.
    Two identifiers of DIFFERENT kinds must NEVER merge, even if numerically identical
    or having >= 7 digit prefix overlap.
    """
    # Case 3a: Bill No vs Admission No with IDENTICAL 6-digit number "220912"
    node_bill = CandidateNode(
        node_id="n_bill_1",
        type="Identifier",
        label="Bill No: 220912",
        source_page=4,
        evidence="Hayaat Chemist Cash Memo / Bill No. 220912",
        confidence=0.95,
    )
    node_admission = CandidateNode(
        node_id="n_adm_1",
        type="Identifier",
        label="Admission No: 220912",
        source_page=1,
        evidence="IPD Admission No: 220912",
        confidence=0.95,
    )

    resolved, edges, log = run_identifier_matching([node_bill, node_admission], [])

    # Kind gating MUST prevent merge:
    assert len(edges) == 0, "Identifiers of different kinds (bill vs admission) must NOT create an edge!"
    assert len(resolved) == 2, "Must remain separate entities!"
    assert len(log) == 1
    assert log[0].match_type == "KIND_MISMATCH"
    assert log[0].decision == "DIFFERENT"
    assert "Kind mismatch" in log[0].reason

    # Case 3b: Bill No vs Phone No with >= 7 digit prefix overlap
    node_bill_close = CandidateNode(
        node_id="n_bill_close",
        type="Identifier",
        label="Bill No: 98924625",
        source_page=5,
        evidence="Invoice 98924625",
        confidence=0.95,
    )
    node_phone = CandidateNode(
        node_id="n_phone_close",
        type="Identifier",
        label="Phone: 9892462685",
        source_page=57,
        evidence="Contact: 9892462685",
        confidence=0.95,
    )

    resolved_b, edges_b, log_b = run_identifier_matching([node_bill_close, node_phone], [])
    assert len(edges_b) == 0, "Different kinds (bill vs phone) must NOT match under Rule B prefix overlap!"
    assert len(resolved_b) == 2
    assert log_b[0].match_type == "KIND_MISMATCH"
    assert log_b[0].decision == "DIFFERENT"


def test_unit_uncertain_near_miss_band():
    """
    Test 4: Explicit UNCERTAIN Near-Miss Band.
    Phone numbers with prefix overlap of exactly 6 digits (below 7-digit threshold)
    on 10-digit numbers:
    - Generates an UNCERTAIN audit edge per Section 18 of Graph Memory POC.
    - Does NOT merge into the same entity cluster.
    """
    node_p1 = CandidateNode(
        node_id="n_tel1",
        type="Identifier",
        label="Phone: 9892461111",
        source_page=10,
        evidence="Contact: 9892461111",
        confidence=0.90,
    )
    node_p2 = CandidateNode(
        node_id="n_tel2",
        type="Identifier",
        label="Phone: 9892462222",
        source_page=20,
        evidence="Contact: 9892462222",
        confidence=0.90,
    )

    resolved, edges, log = run_identifier_matching([node_p1, node_p2], [])

    # An audit edge with status UNCERTAIN must be created
    assert len(edges) == 1, f"Expected 1 UNCERTAIN edge, got {len(edges)}"
    assert edges[0].status == "UNCERTAIN"
    assert edges[0].confidence == 0.6  # 6 / 10

    # Crucially, UNCERTAIN pairs must NOT be merged into the same entity cluster
    assert len(resolved) == 2, "UNCERTAIN pairs must NOT be unioned into the same entity cluster"
    assert log[0].decision == "UNCERTAIN"
    assert log[0].match_type == "AMBIGUOUS_PREFIX"


def test_live_chander_kochhar_pipeline_and_kind_gating():
    """
    Test 5: Live integration run against the real 76-page sample with mocking disabled
    (Sarvam AI extraction client).
    Verifies:
    1. Wall-clock timing is genuine (not mocked).
    2. Phone nodes 98924625 (p5) and 9892462685 (p57) merge into an INFERRED phone entity.
    3. Kind-gating verified live: Bill numbers and Admission/HS numbers never merge.
    4. Prints the actual log trace from get_run_log_file().
    """
    doc_path = Path("dataset_output/Chander Kochhar 01_compressed 2.md")
    if not doc_path.exists():
        pytest.skip(f"Document file not found at {doc_path}")

    start_wall = time.perf_counter()
    result = run_pipeline(doc_path)
    elapsed_wall = time.perf_counter() - start_wall

    print("\n" + "=" * 80)
    print(f"LIVE RUN WALL-CLOCK TIME: {elapsed_wall:.2f}s ({elapsed_wall/60:.2f} min)")
    print(f"RUN LOG FILE: {get_run_log_file()}")
    print("=" * 80)

    # 1. Genuine timing check (mocked runs finish in < 2 seconds; real run takes > 10s)
    assert elapsed_wall > 10.0, f"Wall clock {elapsed_wall:.2f}s is too fast — check for mock leakage!"

    # 2. Verify phone nodes merge
    id_nodes = [n for n in result.candidate_nodes if n.type == "Identifier"]
    p5_phones = [n for n in id_nodes if n.source_page == 5 and "98924625" in n.label]
    p57_phones = [n for n in id_nodes if n.source_page == 57 and "9892462685" in n.label]

    assert len(p5_phones) >= 1, f"Expected page 5 phone node with 98924625, found: {p5_phones}"
    assert len(p57_phones) >= 1, f"Expected page 57 phone node with 9892462685, found: {p57_phones}"

    # Verify SAME_AS edge exists between page 5 and page 57 phone nodes
    phone_edges = [
        e for e in result.candidate_edges
        if e.relationship == "SAME_AS"
        and ("98924625" in e.evidence and "9892462685" in e.evidence)
    ]
    assert len(phone_edges) >= 1, "Expected SAME_AS edge connecting 98924625 and 9892462685"
    assert phone_edges[0].status == "INFERRED"

    # Verify resolved entity cluster links both pages 5 and 57
    merged_phone_entities = [
        e for e in result.resolved_entities
        if e.category == "Identifier"
        and (5 in e.source_pages and 57 in e.source_pages)
        and any("9892462685" in s or "98924625" in s for s in [e.canonical_label] + e.aliases)
    ]
    assert len(merged_phone_entities) >= 1, (
        f"Expected merged Identifier entity spanning pages 5 and 57, found {merged_phone_entities}"
    )
    merged_phone = merged_phone_entities[0]
    assert merged_phone.status == "INFERRED"
    assert 5 in merged_phone.source_pages
    assert 57 in merged_phone.source_pages

    # 3. Verify kind-gating live on the real sample:
    # Ensure bill numbers and admission/HS numbers in result.resolved_entities never share a cluster
    for ent in result.resolved_entities:
        if ent.category == "Identifier":
            all_labels = [ent.canonical_label] + ent.aliases
            kinds_in_entity = {detect_identifier_kind(lbl) for lbl in all_labels if detect_identifier_kind(lbl)}
            assert len(kinds_in_entity) <= 1, (
                f"Kind-gating violation in resolved entity '{ent.canonical_label}': "
                f"mixed kinds {kinds_in_entity} in aliases {ent.aliases}"
            )

    # 4. Extract and print log trace excerpt for 98924625 / 9892462685 comparison
    found_comparison_log = False
    for item in result.identifier_match_log:
        labels = (item.label_a, item.label_b)
        if ("98924625" in labels[0] or "98924625" in labels[1]) and ("9892462685" in labels[0] or "9892462685" in labels[1]):
            found_comparison_log = True
            print("\n" + "-" * 80)
            print("ACTUAL AUDIT LOG EXCERPT FOR 98924625 / 9892462685 PAIR:")
            print(f"  Node A      : {item.label_a} (id={item.node_id_a}, kind={item.kind_a})")
            print(f"  Node B      : {item.label_b} (id={item.node_id_b}, kind={item.kind_b})")
            print(f"  Normalized  : '{item.normalized_a}' vs '{item.normalized_b}'")
            print(f"  Match Type  : {item.match_type}")
            print(f"  Decision    : {item.decision}")
            print(f"  Confidence  : {item.confidence:.3f}")
            print(f"  State       : {item.state}")
            print(f"  Reason      : {item.reason}")
            print("-" * 80)
            break
    assert found_comparison_log, "Comparison log for 98924625 / 9892462685 was not recorded in identifier_match_log"
