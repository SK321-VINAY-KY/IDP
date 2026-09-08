"""
Tests for Step 3: Phase B Navigation.
Validates:
- Exactly 1 LLM call per ontology category (9 calls total).
- Output matches Section 10 NavigationMap structure.
- Section 11 Experiment B: Navigation for Identifier and Amount recovers matches
  from both page 1 and page 6 (not stopping after the first hit).
"""
from typing import Any, Dict, List, Optional
import pytest
from src.ai.layer3_extraction.navigation import NavigationMap, navigate_categories
from src.ai.layer3_extraction.segmentation import Segment
from src.config.ontology import load_ontology


class MockNavigationLLM:
    def __init__(self):
        self.nav_calls = 0
        self.requested_categories = []
        self.last_extraction_focus = None

    def navigate_category(self, segment_summaries, category, category_description, category_examples=None, extraction_focus=None):
        self.nav_calls += 1
        self.requested_categories.append(category)
        self.last_extraction_focus = extraction_focus

        # Realistic mapping per Section 10
        if category == "Amount":
            # Recovers both page 4 pharmacy, page 5 ambulance, page 6 deposit, page 7 authorization
            return ["seg_02", "seg_03", "seg_04", "seg_05"]
        elif category == "Identifier":
            # Recovers both page 1 discharge summary and page 6 deposit receipt (Experiment B requirement)
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        elif category == "Person":
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        elif category == "Organization":
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        elif category == "Document/Record":
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        else:
            return ["seg_01"]

    def extract(self, content: str, schema: Any) -> Any:
        return schema()

    def summarize_page(self, page_md: str, max_words: int = 50) -> str:
        return ""

    def navigate(self, page_summaries: list[str], schema_fields: list[str]) -> dict[str, list[int]]:
        return {}

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: list[dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> list[dict[str, Any]]:
        return []

    def summarize_segment(self, segment_text: str, page_range: list[int]) -> dict[str, str]:
        return {"doc_type_hint": "document", "one_line_summary": "summary"}

    def extract_page_ontology(self, page_md, page_number, extraction_focus=None):
        return {"nodes": [], "edges": []}

    def extract_key_findings(self, segments, candidate_nodes, extraction_focus=None):
        return {"key_findings": []}


def test_navigation_call_count_and_experiment_b():
    ontology = load_ontology()
    assert len(ontology.categories) == 9

    segments = [
        Segment(segment_id="seg_01", page_range=[1, 3], doc_type_hint="discharge summary", one_line_summary="Discharge summary for Chander Kochhar, Hs No 192673, Adm No 68049"),
        Segment(segment_id="seg_02", page_range=[4, 4], doc_type_hint="pharmacy bill", one_line_summary="Hayaat Chemist bill, amount 1485.75, Doctor Or Arhay Raut"),
        Segment(segment_id="seg_03", page_range=[5, 5], doc_type_hint="ambulance bill", one_line_summary="Jeevan Ambulance Service bill no 20570, amount 4500"),
        Segment(segment_id="seg_04", page_range=[6, 6], doc_type_hint="deposit receipt", one_line_summary="Hinduja Hospital in-patient deposit receipt, HS No 192673, amount 10000"),
        Segment(segment_id="seg_05", page_range=[7, 7], doc_type_hint="authorization letter", one_line_summary="ICICI Lombard authorization letter, AL Number 110100834860-, amount 80698"),
    ]

    mock_llm = MockNavigationLLM()
    nav_map = navigate_categories(segments, mock_llm, ontology)

    # 1. Verify call count: exactly 9 calls (1 per ontology category)
    assert mock_llm.nav_calls == 9
    assert nav_map.total_llm_calls == 9

    # 2. Verify all 9 categories were navigated
    for cat in ontology.category_names():
        assert cat in nav_map.category_segments

    # 3. Experiment B pass condition:
    # Recovers HS No./admission no. mentions from both page 1 and page 6;
    # recovers deposit amount (page 6) and authorization amount (page 7) — does NOT stop after first hit!
    id_pages = nav_map.category_pages["Identifier"]
    amt_pages = nav_map.category_pages["Amount"]

    assert 1 in id_pages, "Identifier must recover page 1"
    assert 6 in id_pages, "Identifier must recover page 6 (not stop after page 1)"
    assert 6 in amt_pages, "Amount must recover page 6"
    assert 7 in amt_pages, "Amount must recover page 7"

    # 4. Verify Phase B JSON output format from Section 10
    summary_dict = nav_map.to_summary_dict()
    assert isinstance(summary_dict, dict)
    assert "Amount" in summary_dict
    assert "Identifier" in summary_dict
    assert "seg_04" in summary_dict["Amount"]
    assert "seg_01" in summary_dict["Identifier"]
    assert "seg_04" in summary_dict["Identifier"]


def test_navigation_with_extraction_focus():
    """Verify that extraction_focus steering hints are forwarded to the LLM during category navigation."""
    ontology = load_ontology()
    segments = [
        Segment(segment_id="seg_01", page_range=[1, 2], doc_type_hint="hospital bill", one_line_summary="Inpatient final bill"),
    ]
    mock_llm = MockNavigationLLM()
    focus_hints = ["Focus on room rent charges", "Verify surgeon charges"]

    nav_map = navigate_categories(segments, mock_llm, ontology, extraction_focus=focus_hints)
    assert mock_llm.last_extraction_focus == focus_hints
    assert nav_map.total_llm_calls == 9


def test_heuristic_fallback_navigation_when_llm_empty():
    """Verify that when LLM returns empty segment lists, zero-LLM heuristic fallback navigates correctly."""
    class EmptyNavigationLLM:
        def extract(self, content: str, schema: Any) -> Any:
            return schema()

        def summarize_page(self, page_md: str, max_words: int = 50) -> str:
            return ""

        def navigate(self, page_summaries: list[str], schema_fields: list[str]) -> dict[str, list[int]]:
            return {}

        def check_page_for_fields(
            self,
            page_md: str,
            schema_fields: list[dict[str, str]],
            page_number: int = 0,
            total_pages: int = 0,
        ) -> list[dict[str, Any]]:
            return []

        def summarize_segment(self, segment_text: str, page_range: list[int]) -> dict[str, str]:
            return {"doc_type_hint": "document", "one_line_summary": "summary"}

        def navigate_category(self, segment_summaries, category, category_description, category_examples=None, extraction_focus=None):
            return []  # Simulates LLM returning empty matches

        def extract_page_ontology(self, page_md, page_number=0, extraction_focus=None):
            return {"nodes": [], "edges": []}

        def extract_key_findings(self, segments, candidate_nodes, extraction_focus=None):
            return {"key_findings": []}

    segments = [
        Segment(segment_id="seg_01", page_range=[1, 3], doc_type_hint="discharge summary", one_line_summary="Discharge summary for patient Chander Kochhar"),
        Segment(segment_id="seg_02", page_range=[4, 4], doc_type_hint="pharmacy bill", one_line_summary="Chemist medicine bill with amount 1485.75"),
        Segment(segment_id="seg_03", page_range=[5, 5], doc_type_hint="ambulance bill", one_line_summary="Ambulance service bill"),
    ]

    ontology = load_ontology()
    empty_llm = EmptyNavigationLLM()
    nav_map = navigate_categories(segments, empty_llm, ontology)

    # Fallback should associate Amount with pharmacy bill
    assert "seg_02" in nav_map.category_segments["Amount"]
    # Fallback should associate Person with discharge summary
    assert "seg_01" in nav_map.category_segments["Person"]
    # Fallback should associate Organization with ambulance service
    assert "seg_03" in nav_map.category_segments["Organization"]
