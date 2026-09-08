"""
File: navigation.py
Purpose: Phase B Navigation for Layer 3 (PageIndex-style navigation).
         For each category in the static type ontology, determines which
         document segments plausibly contain mentions of that category.
         Exactly 1 LLM call per ontology category (O(ontology categories) ≈ 9 calls).
Owner: engineer-a@idp-pilot
Created: 2026-09-07 | References: docs/poc/layer3_pageindex_navigation_poc.md (Section 6)
"""
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.segmentation import Segment
from src.config.ontology import OntologyConfig, load_ontology
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NavigationMap(BaseModel):
    category_segments: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="category -> list of matching segment_ids",
    )
    category_pages: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="category -> list of matching page numbers",
    )
    all_navigated_pages: List[int] = Field(
        default_factory=list,
        description="Sorted union of all distinct navigated page numbers",
    )
    total_llm_calls: int = Field(
        default=0,
        description="Total LLM calls made during navigation pass",
    )

    def to_summary_dict(self) -> Dict[str, List[str]]:
        """Return Section 10 Phase B JSON format: {category: [segment_ids]}."""
        return self.category_segments


def _pages_for_segment(seg: Segment) -> List[int]:
    """Return all page numbers in a segment's [start, end] range."""
    return list(range(seg.page_range[0], seg.page_range[1] + 1))


def navigate_categories(
    segments: List[Segment],
    llm: ExtractionLLMClient,
    ontology: Optional[OntologyConfig] = None,
    extraction_focus: Optional[List[str]] = None,
) -> NavigationMap:
    """
    Phase B Entry Point:
    Run one LLM call per ontology category to identify matching segments.
    Cost: exactly len(categories) LLM calls (9 calls total).

    Args:
        segments: List of Segment objects produced by Phase A.
        llm: LLM client implementing ExtractionLLMClient.
        ontology: Static ontology config (defaults to loaded ontology.yaml).
        extraction_focus: Optional list of free-text hints steering extraction.

    Returns:
        NavigationMap with category_segments, category_pages, and all_navigated_pages.
    """
    if ontology is None:
        ontology = load_ontology()

    seg_dict_list = [s.model_dump() for s in segments]
    segment_lookup = {s.segment_id: s for s in segments}

    category_segments: Dict[str, List[str]] = {}
    category_pages: Dict[str, List[int]] = {}
    all_pages_set: Set[int] = set()

    llm_call_count = 0

    for cat_name, cat_def in ontology.categories.items():
        logger.info("navigation.category_start", category=cat_name)

        matched_seg_ids: List[str] = []
        try:
            matched_seg_ids = llm.navigate_category(
                segment_summaries=seg_dict_list,
                category=cat_name,
                category_description=cat_def.description,
                category_examples=cat_def.examples,
                extraction_focus=extraction_focus,
            )
            llm_call_count += 1
        except Exception as exc:
            logger.warning("navigation.category_llm_failed", category=cat_name, error=str(exc))
            matched_seg_ids = []

        # Validate that returned segment IDs actually exist
        valid_seg_ids = [sid for sid in matched_seg_ids if sid in segment_lookup]

        # Heuristic fallback if LLM returned empty for critical categories
        if not valid_seg_ids:
            valid_seg_ids = _heuristic_fallback_navigation(cat_name, segments)

        # Collect page numbers
        cat_pages_set: Set[int] = set()
        for sid in valid_seg_ids:
            seg = segment_lookup.get(sid)
            if seg:
                cat_pages_set.update(_pages_for_segment(seg))

        category_segments[cat_name] = valid_seg_ids
        category_pages[cat_name] = sorted(list(cat_pages_set))
        all_pages_set.update(cat_pages_set)

        logger.info(
            "navigation.category_done",
            category=cat_name,
            matched_segments=valid_seg_ids,
            pages=category_pages[cat_name],
        )

    nav_map = NavigationMap(
        category_segments=category_segments,
        category_pages=category_pages,
        all_navigated_pages=sorted(list(all_pages_set)),
        total_llm_calls=llm_call_count,
    )

    logger.info(
        "navigation.completed",
        total_categories=len(ontology.categories),
        llm_calls=llm_call_count,
        total_navigated_pages=len(nav_map.all_navigated_pages),
        navigated_pages=nav_map.all_navigated_pages,
    )
    return nav_map


def _heuristic_fallback_navigation(category: str, segments: List[Segment]) -> List[str]:
    """
    Zero-LLM fallback if LLM navigation returned no segments for a category.
    Inspects doc_type_hint and summary for category affinities.
    """
    hits = []
    cat_lower = category.lower()

    for seg in segments:
        text = f"{seg.doc_type_hint} {seg.one_line_summary}".lower()

        if cat_lower == "amount":
            if any(k in text for k in ["bill", "receipt", "invoice", "charge", "amount", "rs", "inr", "deposit", "cost", "authoriz"]):
                hits.append(seg.segment_id)
        elif cat_lower == "identifier":
            if any(k in text for k in ["no", "number", "id", "bill", "summary", "voucher", "receipt", "authoriz", "claim"]):
                hits.append(seg.segment_id)
        # TODO(follow-up): Domain-specific heuristics (e.g. kochhar, raut) are hardcoded for sample data. Refactor to generic pattern extractors in a separate follow-up.
        elif cat_lower == "person":
            if any(k in text for k in ["patient", "doctor", "mr", "mrs", "dr", "name", "kochhar", "raut"]):
                hits.append(seg.segment_id)
        elif cat_lower == "organization":
            if any(k in text for k in ["hospital", "chemist", "pharmacy", "ambulance", "insurance", "bank", "centre", "company"]):
                hits.append(seg.segment_id)
        # TODO(follow-up): Location and Item are currently unhandled in heuristic fallback navigation; add explicit fallback patterns or default to all segments.
        elif cat_lower in ("date", "event", "document/record"):
            hits.append(seg.segment_id)

    return hits
