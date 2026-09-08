"""
Tests for Step 2: Phase A Segmentation.
Validates:
- Heuristic boundary detection using Layer 1-2 signals (zero LLM calls).
- Section 11 Experiment A: Segment boundaries align with actual 5 sub-document types
  (discharge summary, pharmacy bill, ambulance bill, deposit receipt, authorization letter) within +/- 1 page.
- Section 11 Experiment C: Page 4's garbled pharmacy bill (0.874 confidence, "CHAMDAR KDCHHAR")
  is correctly isolated as its own segment despite OCR noise.
- O(S) LLM summary calls via build_document_segments.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from pydantic import BaseModel

from src.ai.layer3_extraction.page_loader import load_pages_from_fixture
from src.ai.layer3_extraction.segmentation import (
    Segment,
    detect_segment_boundaries,
    build_document_segments,
)


class MockExtractionLLM:
    def __init__(self):
        self.summary_calls = 0

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        return schema.model_construct()

    def summarize_page(self, page_md: str, max_words: int) -> str:
        return ""

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        return {}

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        return []

    def summarize_segment(self, segment_text: str, page_range: List[int]) -> Dict[str, str]:
        self.summary_calls += 1
        return {
            "doc_type_hint": "sub_document",
            "one_line_summary": f"Summary for pages {page_range[0]}-{page_range[1]}",
        }

    def navigate_category(
        self,
        segment_summaries: List[Dict[str, Any]],
        category: str,
        category_description: str,
        category_examples: Optional[List[str]] = None,
        extraction_focus: Optional[List[str]] = None,
    ) -> List[str]:
        return []

    def extract_page_ontology(
        self,
        page_md: str,
        page_number: int,
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {"nodes": [], "edges": []}

    def extract_key_findings(
        self,
        segments: List[Dict[str, Any]],
        candidate_nodes: List[Dict[str, Any]],
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {"key_findings": []}


def test_synthetic_boundary_detection():
    """Verify individual boundary detection signals on synthetic pages."""
    pages = [
        # Segment 1: Digital discharge summary
        {"page_number": 1, "markdown": "DISCHARGE SUMMARY Patient: John Doe", "confidence": 0.99, "engines_used": ["docling"], "chars": 1500},
        {"page_number": 2, "markdown": "Clinical findings continued...", "confidence": 0.98, "engines_used": ["docling"], "chars": 1200},
        # Signal: Keyword marker & confidence drop -> Segment 2
        {"page_number": 3, "markdown": "HAYAAT CHEMIST BILL Rx details", "confidence": 0.86, "engines_used": ["paddleocr_printed"], "chars": 400},
        # Signal: Engine shift & ambulance marker -> Segment 3
        {"page_number": 4, "markdown": "AMBULANCE SERVICE BILL transport charges", "confidence": 0.95, "engines_used": ["paddleocr_handwritten"], "chars": 900},
        # Signal: Blank page hard boundary -> Segment 4
        {"page_number": 5, "markdown": "", "confidence": 1.0, "engines_used": [], "chars": 0},
        # Segment 5: Deposit receipt
        {"page_number": 6, "markdown": "IN-PATIENT DEPOSIT RECEIPT Amount 5000", "confidence": 0.97, "engines_used": ["paddleocr_printed"], "chars": 800},
    ]

    spans = detect_segment_boundaries(pages)
    assert len(spans) >= 4
    # Check that page 1 and 2 are grouped together
    assert spans[0] == (1, 2)
    # Check that page 3 is isolated as pharmacy bill
    assert (3, 3) in spans


def test_chander_kochhar_experiment_a_and_c():
    """
    Test against real Chander Kochhar 76-page bundle.
    Experiment A: Segment boundaries align with the actual 5 sub-document types:
      1. Discharge summary (pages 1-3)
      2. Pharmacy bill (page 4)
      3. Ambulance bill (page 5)
      4. Deposit receipt (page 6)
      5. Authorization letter (page 7)
    Experiment C: Page 4 (0.874 confidence, garbled OCR) is isolated as [4, 4].
    """
    pages = load_pages_from_fixture("chander_kochhar")
    assert len(pages) == 76

    # Load OCR metadata
    schema_ref_path = Path("tests/fixtures/chander_kochhar.schema_ref.json")
    if schema_ref_path.exists():
        with open(schema_ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        ref_map = {p["page_number"]: p for p in ref.get("pages", [])}
        for p in pages:
            meta = ref_map.get(p["page_number"], {})
            p["confidence"] = meta.get("confidence", 1.0)
            p["engines_used"] = meta.get("engines_used", [])
            p["chars"] = meta.get("chars", len(p["markdown"]))

    spans = detect_segment_boundaries(pages)
    assert len(spans) > 0

    # Experiment A checks:
    # 1. Discharge summary spans pages 1 to 3
    assert spans[0] == (1, 3), f"Expected discharge summary span [1, 3], got {spans[0]}"

    # 2. Pharmacy bill is isolated at page 4 (Experiment C pass condition)
    assert spans[1] == (4, 4), f"Expected page 4 pharmacy bill isolated as [4, 4], got {spans[1]}"

    # 3. Ambulance bill is at page 5
    assert spans[2] == (5, 5), f"Expected ambulance bill at [5, 5], got {spans[2]}"

    # 4. Deposit receipt is at page 6
    assert spans[3] == (6, 6), f"Expected deposit receipt at [6, 6], got {spans[3]}"

    # 5. Authorization letter is at page 7
    assert spans[4] == (7, 7), f"Expected authorization letter at [7, 7], got {spans[4]}"


def test_build_document_segments_call_count():
    """Verify that build_document_segments makes exactly S LLM calls (1 per segment)."""
    pages = [
        {"page_number": 1, "markdown": "DISCHARGE SUMMARY Patient: Chander Kochhar", "confidence": 0.99, "engines_used": ["paddleocr_printed"], "chars": 1500},
        {"page_number": 2, "markdown": "HAYAAT CHEMIST BILL Amount: 1485", "confidence": 0.87, "engines_used": ["paddleocr_printed"], "chars": 500},
        {"page_number": 3, "markdown": "AMBULANCE SERVICE Bill No: 20570", "confidence": 0.91, "engines_used": ["paddleocr_printed"], "chars": 800},
    ]
    mock_llm = MockExtractionLLM()
    segments = build_document_segments(pages, mock_llm)

    assert len(segments) == 3
    assert mock_llm.summary_calls == 3
    assert segments[0].segment_id == "seg_01"
    assert segments[1].segment_id == "seg_02"
    assert segments[2].segment_id == "seg_03"


def test_build_document_segments_chander_kochhar_full():
    """Verify build_document_segments constructs valid Segment objects for the full test bundle."""
    pages = load_pages_from_fixture("chander_kochhar")
    mock_llm = MockExtractionLLM()
    segments = build_document_segments(pages, mock_llm)

    assert len(segments) >= 5
    assert mock_llm.summary_calls == len(segments)
    for seg in segments:
        assert seg.segment_id.startswith("seg_")
        assert len(seg.page_range) == 2
        assert seg.page_range[0] <= seg.page_range[1]
        assert seg.doc_type_hint != ""
        assert seg.one_line_summary != ""
