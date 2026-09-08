"""
Tests for Step 4: Targeted Extraction.
Validates:
- extract_candidate_graph writes candidate nodes and edges with provenance.
- Write-only contract (per graph-memory POC Phase 1): no cross-page reads/merging.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.ai.layer3_extraction.extractor import (
    extract_candidate_graph,
)
from src.ai.layer3_extraction.page_loader import load_pages_from_fixture


def test_candidate_graph_write_only_extraction():
    """
    Verify Phase C candidate graph extraction on Chander Kochhar pages:
    - Emits candidate nodes for Organization, Person, Identifier, Amount.
    - Write-only: each page emits nodes with source_page.
    """
    pages = load_pages_from_fixture("chander_kochhar")
    # Extract only on pages [1, 4, 5, 6, 7]
    nodes, edges = extract_candidate_graph(pages, target_pages=[1, 4, 5, 6, 7])

    assert len(nodes) > 0
    assert len(edges) > 0

    node_types = {n.type for n in nodes}
    assert "Person" in node_types
    assert "Organization" in node_types
    assert "Identifier" in node_types
    assert "Amount" in node_types

    # Verify Page 1 nodes
    p1_nodes = [n for n in nodes if n.source_page == 1]
    patient_p1 = next(n for n in p1_nodes if "CHANDER KOCHHAR" in n.label)
    assert patient_p1.importance == "high"
    assert patient_p1.importance_reason == "matched person pattern for Patient"
    assert patient_p1.subtype == "Patient"

    doctor_p1 = next(n for n in p1_nodes if "ABHAY RAUT" in n.label)
    assert doctor_p1.importance == "high"
    assert doctor_p1.importance_reason == "matched person pattern for Doctor"
    assert doctor_p1.subtype == "Doctor"

    hs_p1 = next(n for n in p1_nodes if "192673" in n.label)  # HS No
    assert hs_p1.importance == "high"
    assert hs_p1.importance_reason == "matched identifier pattern for HS No"
    assert hs_p1.subtype == "HS No"

    # Verify Page 4 nodes (pharmacy bill)
    p4_nodes = [n for n in nodes if n.source_page == 4]
    chemist_p4 = next(n for n in p4_nodes if "HAYAAT CHEMIST" in n.label)
    assert chemist_p4.importance == "medium"
    assert chemist_p4.importance_reason == "matched issuing organization pattern"

    amt_p4 = next(n for n in p4_nodes if "1485" in n.label)
    assert amt_p4.importance == "high"
    assert amt_p4.importance_reason == "matched primary amount pattern"

    assert any("ARHAY RAUT" in n.label for n in p4_nodes)

    # Verify Page 6 nodes (deposit receipt)
    p6_nodes = [n for n in nodes if n.source_page == 6]
    assert any("10000" in n.label for n in p6_nodes)
    assert any("192673" in n.label for n in p6_nodes)  # HS No restated on page 6

    # Verify Page 7 nodes (authorization letter)
    p7_nodes = [n for n in nodes if n.source_page == 7]
    assert any("80698" in n.label for n in p7_nodes)
    assert any("110100834860" in n.label for n in p7_nodes)


def test_candidate_graph_llm_ontology_extraction():
    """
    Verify Phase C LLM-driven ontology extraction:
    - Extracts all 9 categories dynamically (including Location, Event, Item).
    - Links semantic relationships into CandidateEdge instances.
    - Gracefully falls back to heuristic extraction on error.
    """
    class MockLLM:
        def __init__(self):
            self.extracted_pages: List[int] = []

        def extract(self, content: str, schema: Any) -> Any:
            return schema()

        def summarize_page(self, page_md: str, max_words: int = 50) -> str:
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
            return {"doc_type_hint": "document", "one_line_summary": "summary"}

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
            page_number: int = 0,
            extraction_focus: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            self.extracted_pages.append(page_number)
            if page_number == 99:
                raise RuntimeError("Simulated API failure on page 99")

            return {
                "nodes": [
                    {"category": "Person", "label": "Dr. Abhay Raut", "evidence": "DOCTOR : OR ARHAY RAUT", "importance": "high", "importance_reason": "Attending physician"},
                    {"category": "Organization", "label": "Apollo Hospitals", "evidence": "Apollo Hospitals Bilaspur", "importance": "high", "importance_reason": "Issuing hospital"},
                    {"category": "Location", "label": "ICU Bed 4", "evidence": "Ward: ICU Bed 4", "importance": "medium"},
                    {"category": "Event", "label": "Emergency Admission", "evidence": "Admitted via Emergency", "importance": "high"},
                    {"category": "Item", "label": "Paracetamol 500mg", "evidence": "Tab Paracetamol 500mg", "importance": "low"},
                    {"category": "Amount", "label": "INR 500.00", "evidence": "Total: INR 500.00", "importance": "high"},
                    {"category": "Identifier", "label": "Bill No: 9988", "evidence": "Bill No: 9988", "importance": "high", "subtype": "Bill No"},
                    {"category": "Date", "label": "2025-02-24", "evidence": "Date: 2025-02-24", "importance": "medium"},
                    {"category": "Document/Record", "label": "Pharmacy Bill", "evidence": "Tax Invoice / Pharmacy Bill", "importance": "medium"},
                ],
                "edges": [
                    {"source_label": "Pharmacy Bill", "relationship": "CONTAINS", "target_label": "Bill No: 9988", "evidence": "Bill contains number"},
                    {"source_label": "Emergency Admission", "relationship": "PERFORMED_BY", "target_label": "Dr. Abhay Raut", "evidence": "Doctor performed admission"},
                ],
            }

        def extract_key_findings(
            self,
            segments: List[Dict[str, Any]],
            candidate_nodes: List[Dict[str, Any]],
            extraction_focus: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            return {"key_findings": []}

    pages = [
        {"page_number": 1, "markdown": "Medical record page 1 with Apollo Hospitals"},
        {"page_number": 99, "markdown": "Patient Name: CHANDER KOCHHAR, Doctor: ABHAY RAUT, HS No: 192673"},
    ]

    mock_llm = MockLLM()
    nodes, edges = extract_candidate_graph(pages, target_pages=[1, 99], llm=mock_llm)

    assert 1 in mock_llm.extracted_pages
    assert 99 in mock_llm.extracted_pages

    # Page 1 must have nodes for all 9 categories
    p1_nodes = [n for n in nodes if n.source_page == 1]
    p1_types = {n.type for n in p1_nodes}
    assert {"Person", "Organization", "Location", "Event", "Item", "Amount", "Identifier", "Date", "Document/Record"}.issubset(p1_types)

    # Check importance and reason preserved from LLM
    doc_node = next(n for n in p1_nodes if n.label == "Dr. Abhay Raut")
    assert doc_node.importance == "high"
    assert doc_node.importance_reason == "Attending physician"

    # Page 1 must have the semantic edges extracted by the LLM
    p1_edges = [e for e in edges if e.source_page == 1]
    rels = {e.relationship for e in p1_edges}
    assert "CONTAINS" in rels
    assert "PERFORMED_BY" in rels

    # Page 99 encountered exception -> verified heuristic fallback kicked in
    p99_nodes = [n for n in nodes if n.source_page == 99]
    assert len(p99_nodes) > 0
    assert any("CHANDER KOCHHAR" in n.label for n in p99_nodes)


def test_candidate_node_model_fields_and_other_type():
    """Verify CandidateNode accepts importance, importance_reason, subtype, and type='Other'."""
    from src.ai.layer3_extraction.extractor import CandidateNode

    node_default = CandidateNode(
        node_id="n_001",
        type="Person",
        label="Test Person",
        source_page=1,
        evidence="Test evidence",
    )
    assert node_default.importance == "medium"
    assert node_default.importance_reason == ""
    assert node_default.subtype is None

    node_other = CandidateNode(
        node_id="n_002",
        type="Other",
        label="Acute Bronchitis",
        source_page=1,
        evidence="Diagnosis: Acute Bronchitis",
        importance="high",
        importance_reason="Primary medical condition justifying admission",
        subtype="Medical Diagnosis",
    )
    assert node_other.type == "Other"
    assert node_other.importance == "high"
    assert node_other.subtype == "Medical Diagnosis"


def test_unlisted_category_mapped_to_other():
    """Verify unlisted categories returned by LLM are coerced to 'Other' with subtype set."""
    class MockUnlistedLLM:
        def extract(self, content: str, schema: Any) -> Any:
            return schema()

        def summarize_page(self, page_md: str, max_words: int = 50) -> str:
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
            return {"doc_type_hint": "document", "one_line_summary": "summary"}

        def navigate_category(
            self,
            segment_summaries: List[Dict[str, Any]],
            category: str,
            category_description: str,
            category_examples: Optional[List[str]] = None,
            extraction_focus: Optional[List[str]] = None,
        ) -> List[str]:
            return []

        def extract_page_ontology(self, page_md: str, page_number: int = 0, extraction_focus=None) -> Dict[str, Any]:
            return {
                "nodes": [
                    {
                        "category": "Diagnosis",  # Not in 9 baseline ontology categories
                        "label": "Hypertension",
                        "evidence": "Diagnosis: Hypertension",
                        "importance": "medium",
                        "importance_reason": "Secondary condition",
                    }
                ],
                "edges": [],
            }

        def extract_key_findings(
            self,
            segments: List[Dict[str, Any]],
            candidate_nodes: List[Dict[str, Any]],
            extraction_focus: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            return {"key_findings": []}

    pages = [{"page_number": 1, "markdown": "Patient with Hypertension"}]
    nodes, _ = extract_candidate_graph(pages, llm=MockUnlistedLLM())
    assert len(nodes) == 1
    assert nodes[0].type == "Other"
    assert nodes[0].subtype == "Diagnosis"
    assert nodes[0].label == "Hypertension"
    assert nodes[0].importance == "medium"


def test_prompt_templates_render_with_and_without_extraction_focus():
    """Verify category_navigation, page_ontology_extraction, and document_key_findings templates."""
    from src.adapters.llm.sarvam_client import render_prompt

    # 1. category_navigation without focus vs extraction_focus=None (identical output)
    p1 = render_prompt(
        "category_navigation",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        category="Amount",
        category_description="Charges",
        category_examples=["Rs 100"],
    )
    p1_none = render_prompt(
        "category_navigation",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        category="Amount",
        category_description="Charges",
        category_examples=["Rs 100"],
        extraction_focus=None,
    )
    assert p1 == p1_none, "category_navigation with extraction_focus=None must be identical to omitting parameter"
    assert "The user has told you the following matters" not in p1

    # category_navigation with focus
    p1_focused = render_prompt(
        "category_navigation",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        category="Amount",
        category_description="Charges",
        category_examples=["Rs 100"],
        extraction_focus=["Prioritize pharmacy medications", "Ignore ambulance costs"],
    )
    assert "The user has told you the following matters" in p1_focused
    assert "Prioritize pharmacy medications" in p1_focused
    assert "Ignore ambulance costs" in p1_focused

    # 2. page_ontology_extraction without focus vs extraction_focus=None (identical output)
    p2 = render_prompt("page_ontology_extraction", page_md="Sample page", page_number=1)
    p2_none = render_prompt("page_ontology_extraction", page_md="Sample page", page_number=1, extraction_focus=None)
    assert p2 == p2_none, "page_ontology_extraction with extraction_focus=None must be identical to omitting parameter"
    assert "The user has told you the following matters" not in p2

    p2_focused = render_prompt(
        "page_ontology_extraction",
        page_md="Sample page",
        page_number=1,
        extraction_focus=["Focus on admission dates"],
    )
    assert "The user has told you the following matters" in p2_focused
    assert "Focus on admission dates" in p2_focused

    # 3. document_key_findings without focus vs extraction_focus=None (identical output)
    p3 = render_prompt(
        "document_key_findings",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        candidate_nodes=[{"type": "Person", "label": "John Doe", "source_page": 1, "importance": "high"}],
    )
    p3_none = render_prompt(
        "document_key_findings",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        candidate_nodes=[{"type": "Person", "label": "John Doe", "source_page": 1, "importance": "high"}],
        extraction_focus=None,
    )
    assert p3 == p3_none, "document_key_findings with extraction_focus=None must be identical to omitting parameter"
    assert "The user has told you the following matters" not in p3

    p3_focused = render_prompt(
        "document_key_findings",
        segments=[{"segment_id": "seg_01", "page_range": [1, 2], "doc_type_hint": "bill", "one_line_summary": "bill"}],
        candidate_nodes=[{"type": "Person", "label": "John Doe", "source_page": 1, "importance": "high"}],
        extraction_focus=["Check insurance policy coverage"],
    )
    assert "The user has told you the following matters" in p3_focused
    assert "Check insurance policy coverage" in p3_focused


def test_extraction_focus_forwarded_to_llm_kwargs():
    """Verify extraction_focus parameter is forwarded to LLM client methods."""
    class KwargSpyLLM:
        def __init__(self):
            self.nav_focus = None
            self.extract_focus = None

        def extract(self, content: str, schema: Any) -> Any:
            return schema()

        def summarize_page(self, page_md: str, max_words: int = 50) -> str:
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
            return {"doc_type_hint": "document", "one_line_summary": "summary"}

        def navigate_category(self, segment_summaries, category, category_description, category_examples=None, extraction_focus=None):
            self.nav_focus = extraction_focus
            return ["seg_01"]

        def extract_page_ontology(self, page_md, page_number=0, extraction_focus=None):
            self.extract_focus = extraction_focus
            return {"nodes": [], "edges": []}

        def extract_key_findings(self, segments, candidate_nodes, extraction_focus=None):
            return {"key_findings": []}

    spy = KwargSpyLLM()
    focus = ["Focus on ICU medication"]

    # 1. extract_candidate_graph
    extract_candidate_graph([{"page_number": 1, "markdown": "test"}], llm=spy, extraction_focus=focus)
    assert spy.extract_focus == focus

    # 2. navigate_categories
    from src.ai.layer3_extraction.navigation import navigate_categories
    from src.ai.layer3_extraction.segmentation import Segment
    seg = Segment(segment_id="seg_01", page_range=[1, 1], doc_type_hint="bill", one_line_summary="bill")
    navigate_categories([seg], llm=spy, extraction_focus=focus)
    assert spy.nav_focus == focus
