"""
Tests for Step 5: Deterministic Recall Check.
Validates:
- Hand-authored expectation table keyed on doc_type_hint (zero LLM calls).
- Applied ONLY to Identifier and Amount categories.
- Location and Date fail open.
- On failure, re-run Step 4 for that specific segment + category only (never document-wide).
- On pass, no re-run is triggered.
"""
import pytest
from src.ai.layer3_extraction.expectation_check import (
    get_expected_categories,
    run_expectation_check,
)
from src.ai.layer3_extraction.extractor import CandidateEdge, CandidateNode
from src.ai.layer3_extraction.segmentation import Segment


def test_expected_categories_table():
    """Verify hand-authored expectation table maps doc types correctly and only contains Identifier/Amount."""
    # Bills and receipts must expect Identifier and Amount
    pharmacy_exp = get_expected_categories("pharmacy bill")
    assert "Identifier" in pharmacy_exp
    assert "Amount" in pharmacy_exp
    assert len(pharmacy_exp.difference({"Identifier", "Amount"})) == 0

    ambulance_exp = get_expected_categories("ambulance bill")
    assert "Identifier" in ambulance_exp
    assert "Amount" in ambulance_exp

    deposit_exp = get_expected_categories("deposit receipt")
    assert "Identifier" in deposit_exp
    assert "Amount" in deposit_exp

    auth_exp = get_expected_categories("authorization letter")
    assert "Identifier" in auth_exp
    assert "Amount" in auth_exp

    # Discharge summary expects Identifier (not Amount)
    discharge_exp = get_expected_categories("discharge summary")
    assert "Identifier" in discharge_exp
    assert "Amount" not in discharge_exp

    # Location and Date fail open (never expected)
    for hint in ["pharmacy bill", "discharge summary", "deposit receipt", "unknown"]:
        exp = get_expected_categories(hint)
        assert "Location" not in exp
        assert "Date" not in exp


def test_expectation_check_passing_case():
    """When all expected categories are present, no re-run is triggered."""
    segments = [
        Segment(segment_id="seg_01", page_range=[1, 1], doc_type_hint="discharge summary", one_line_summary="..."),
        Segment(segment_id="seg_02", page_range=[4, 4], doc_type_hint="pharmacy bill", one_line_summary="..."),
    ]
    nodes = [
        CandidateNode(node_id="n_01", type="Identifier", label="HS No: 123", source_page=1, evidence="..."),
        CandidateNode(node_id="n_02", type="Identifier", label="Bill No: 456", source_page=4, evidence="..."),
        CandidateNode(node_id="n_03", type="Amount", label="INR 1485", source_page=4, evidence="..."),
    ]
    edges = []
    pages = [
        {"page_number": 1, "markdown": "HS No: 123"},
        {"page_number": 4, "markdown": "Bill No: 456 Amount: 1485"},
    ]

    upd_nodes, upd_edges, results = run_expectation_check(segments, nodes, edges, pages)
    assert len(results) == 2
    assert all(r.passed for r in results)
    assert not any(r.rerun_triggered for r in results)


def test_expectation_check_segment_targeted_rerun():
    """When expected category is missing, only that specific segment is re-run."""
    segments = [
        Segment(segment_id="seg_01", page_range=[1, 1], doc_type_hint="discharge summary", one_line_summary="..."),
        Segment(segment_id="seg_02", page_range=[4, 4], doc_type_hint="pharmacy bill", one_line_summary="..."),
    ]
    # seg_02 is missing Amount!
    nodes = [
        CandidateNode(node_id="n_01", type="Identifier", label="HS No: 123", source_page=1, evidence="..."),
        CandidateNode(node_id="n_02", type="Identifier", label="Bill No: 456", source_page=4, evidence="..."),
    ]
    edges = []
    pages = [
        {"page_number": 1, "markdown": "HS No: 123"},
        {"page_number": 4, "markdown": "HAYAAT CHEMIST Bill No: 456 AOTAL ANT:- 1485.75"},
    ]

    upd_nodes, upd_edges, results = run_expectation_check(segments, nodes, edges, pages)
    assert len(results) == 2
    assert results[0].passed is True

    # Segment 2 failed initially because Amount was missing
    assert results[1].passed is False
    assert results[1].missing_categories == ["Amount"]
    assert results[1].rerun_triggered is True

    # Check that Amount was recovered in the segment re-run
    seg2_amounts = [n for n in upd_nodes if n.source_page == 4 and n.type == "Amount"]
    assert len(seg2_amounts) > 0
    assert any("1485" in n.label for n in seg2_amounts)
