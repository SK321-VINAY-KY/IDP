"""
File: expectation_check.py
Purpose: Step 5 Deterministic Recall Check for Layer 3.
         Implements resolution to POC Section 8.1:
         - Hand-authored expectation table keyed on doc_type_hint (zero LLM calls).
         - Applied ONLY to Identifier and Amount categories.
         - On failure, re-run Step 4 targeted extraction for that specific segment
           and category only — never a document-wide fallback.
         - Location and Date explicitly fail open.
Owner: engineer-a@idp-pilot
Created: 2026-09-07 | References: docs/poc/layer3_pageindex_navigation_poc.md (Section 8.1)
"""
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.extractor import (
    CandidateEdge,
    CandidateNode,
    extract_candidate_graph,
)
from src.ai.layer3_extraction.segmentation import Segment
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Hand-authored expectation table keyed on doc_type_hint.
# Strictly applied ONLY to "Identifier" and "Amount".
# Any category not in {"Identifier", "Amount"} is ignored.
DOC_TYPE_EXPECTATIONS: Dict[str, Set[str]] = {
    "pharmacy bill": {"Identifier", "Amount"},
    "chemist bill": {"Identifier", "Amount"},
    "ambulance bill": {"Identifier", "Amount"},
    "deposit receipt": {"Identifier", "Amount"},
    "receipt": {"Identifier", "Amount"},
    "authorization letter": {"Identifier", "Amount"},
    "cashless authorization": {"Identifier", "Amount"},
    "final bill": {"Identifier", "Amount"},
    "inpatient final bill": {"Identifier", "Amount"},
    "bill of supply": {"Identifier", "Amount"},
    "invoice": {"Identifier", "Amount"},
    "tax invoice": {"Identifier", "Amount"},
    "discharge summary": {"Identifier"},
    "investigation report": {"Identifier"},
    "laboratory report": {"Identifier"},
    "diagnostic report": {"Identifier"},
}


class ExpectationCheckResult(BaseModel):
    passed: bool
    segment_id: str
    doc_type_hint: str
    expected_categories: List[str]
    missing_categories: List[str]
    rerun_triggered: bool = False
    new_nodes_extracted: int = 0


def get_expected_categories(doc_type_hint: str) -> Set[str]:
    """
    Look up hand-authored expectations for a doc_type_hint.
    Only returns subsets of {"Identifier", "Amount"}.
    Location and Date explicitly fail open (never expected).
    """
    hint_lower = doc_type_hint.lower().strip()
    if not hint_lower or hint_lower == "unknown":
        return set()

    # Check exact match
    if hint_lower in DOC_TYPE_EXPECTATIONS:
        return DOC_TYPE_EXPECTATIONS[hint_lower]

    # Substring matching for flexible hint variants (require non-empty key and hint)
    for key, expected in DOC_TYPE_EXPECTATIONS.items():
        if key in hint_lower or hint_lower in key:
            return expected

    return set()


def run_expectation_check(
    segments: List[Segment],
    nodes: List[CandidateNode],
    edges: List[CandidateEdge],
    pages_md: List[Dict[str, Any]],
    llm: Optional[ExtractionLLMClient] = None,
) -> Tuple[List[CandidateNode], List[CandidateEdge], List[ExpectationCheckResult]]:
    """
    Run deterministic recall check over all segments.
    If an expected category (Identifier or Amount) is missing for a segment,
    re-run Step 4 targeted extraction for that specific segment + category only.

    Returns:
        (updated_nodes, updated_edges, check_results)
    """
    updated_nodes = list(nodes)
    updated_edges = list(edges)
    check_results: List[ExpectationCheckResult] = []

    # Map each page number to its candidate categories already extracted
    page_to_categories: Dict[int, Set[str]] = {}
    for node in updated_nodes:
        page_to_categories.setdefault(node.source_page, set()).add(node.type)

    for seg in segments:
        seg_pages = set(range(seg.page_range[0], seg.page_range[1] + 1))
        expected_cats = get_expected_categories(seg.doc_type_hint)

        # Restricted strictly to Identifier and Amount
        restricted_expected = expected_cats.intersection({"Identifier", "Amount"})

        # Categories present across the segment's pages
        present_in_seg: Set[str] = set()
        for p in seg_pages:
            present_in_seg.update(page_to_categories.get(p, set()))

        missing = sorted(list(restricted_expected - present_in_seg))

        result = ExpectationCheckResult(
            passed=len(missing) == 0,
            segment_id=seg.segment_id,
            doc_type_hint=seg.doc_type_hint,
            expected_categories=sorted(list(restricted_expected)),
            missing_categories=missing,
        )

        if not result.passed:
            logger.warning(
                "expectation_check.failure_detected",
                segment_id=seg.segment_id,
                doc_type=seg.doc_type_hint,
                missing=missing,
                pages=sorted(list(seg_pages)),
            )

            # Re-run Step 4 for this specific segment only (never document-wide)
            result.rerun_triggered = True
            seg_page_list = sorted(list(seg_pages))
            new_nodes, new_edges = extract_candidate_graph(
                pages_md=pages_md,
                target_pages=seg_page_list,
                llm=llm,
            )

            # Filter newly extracted nodes to only missing categories
            targeted_new_nodes = [n for n in new_nodes if n.type in missing]
            result.new_nodes_extracted = len(targeted_new_nodes)

            updated_nodes.extend(targeted_new_nodes)
            updated_edges.extend(new_edges)

            # Update present categories
            for n in targeted_new_nodes:
                page_to_categories.setdefault(n.source_page, set()).add(n.type)

            logger.info(
                "expectation_check.segment_rerun_completed",
                segment_id=seg.segment_id,
                new_nodes_recovered=len(targeted_new_nodes),
            )

        check_results.append(result)

    total_failures = sum(1 for r in check_results if not r.passed)
    logger.info(
        "expectation_check.completed",
        total_segments=len(segments),
        failures=total_failures,
    )
    return updated_nodes, updated_edges, check_results
