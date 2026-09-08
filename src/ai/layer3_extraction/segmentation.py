"""
File: segmentation.py
Purpose: Phase A Segmentation for Layer 3 (PageIndex-style navigation).
         Cheaply groups contiguous pages into sub-document spans without
         assuming document structure in advance.
         - Heuristic boundary detection: zero LLM calls.
         - Segment summary: exactly 1 LLM call per segment (O(S) calls).
Owner: engineer-a@idp-pilot
Created: 2026-09-07 | References: docs/poc/layer3_pageindex_navigation_poc.md (Section 5)
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.schemas.page import PageOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Small, hand-authored list of document-type markers (boundary hints, not a schema)
DOCUMENT_TYPE_MARKERS = [
    (r"DISCHARGE\s+SUMMARY", "discharge summary"),
    (r"(?:HAYAAT\s+CHEMIST|CHEMIST|PHARMACY(?:\s+BILL|\s+INVOICE)?)", "pharmacy bill"),
    (r"(?:AMBULANCE\s+SERVICE|AMBULANCE\s+BILL)", "ambulance bill"),
    (r"(?:IN-?PATIENT\s+DEPOSIT\s+RECEIPT|DEPOSIT\s+RECEIPT|RECEIPT\s+VOUCHER)", "deposit receipt"),
    (r"(?:AUTHORIZATION\s+LETTER|PRE-?AUTHORIZATION\s+(?:REQUEST|LETTER)|GUARANTEE\s+OF\s+PAYMENT)", "authorization letter"),
    (r"(?:INPATIENT\s+FINAL\s+BILL|FINAL\s+BILL(?:\s*\(SUMMARY\))?|BILL\s+OF\s+SUPPLY)", "final bill"),
    (r"(?:TAX\s+INVOICE|INVOICE\s+NO)", "invoice"),
    (r"(?:INVESTIGATION\s+REPORT|LABORATORY\s+REPORT|DIAGNOSTIC\s+REPORT)", "investigation report"),
    (r"(?:PRESCRIPTION|RX)", "prescription"),
]


class Segment(BaseModel):
    segment_id: str
    page_range: List[int] = Field(description="[start_page, end_page] inclusive")
    doc_type_hint: str = Field(default="unknown", description="Concise sub-document type hint")
    one_line_summary: str = Field(default="", description="One-line summary of segment contents")


def _extract_page_metadata(page: Union[Dict[str, Any], PageOutput, str], default_page_num: int) -> Dict[str, Any]:
    """Normalize page into a dictionary with markdown and OCR metadata."""
    if isinstance(page, PageOutput):
        caps = getattr(page, "capabilities", [])
        is_blank = "is_blank" in caps or getattr(page, "is_blank", False) or len(page.markdown.strip()) < 30
        return {
            "page_number": page.page_number,
            "markdown": page.markdown,
            "confidence": page.confidence,
            "engines_used": list(page.engines_used),
            "chars": len(page.markdown),
            "is_blank": is_blank,
        }

    if isinstance(page, str):
        md = page
        pnum = default_page_num
    else:
        md = page.get("markdown", "")
        pnum = page.get("page_number", default_page_num)

    # Parse confidence and engines from markdown comments if available
    conf = 1.0
    engines: List[str] = []
    chars = len(md)

    if isinstance(page, dict):
        if "confidence" in page:
            conf = float(page["confidence"])
        if "engines_used" in page:
            engines = list(page["engines_used"])
        if "chars" in page:
            chars = int(page["chars"])

    # Fallback to comment header parsing if confidence was default
    if conf == 1.0 and "<!-- PAGE" in md:
        conf_match = re.search(r"conf=([0-9.]+)", md)
        if conf_match:
            try:
                conf = float(conf_match.group(1))
            except ValueError:
                pass
        eng_match = re.search(r"engine=([a-zA-Z0-9_]+)", md)
        if eng_match:
            engines = [eng_match.group(1)]

    is_blank = chars < 30 or not md.strip()

    return {
        "page_number": pnum,
        "markdown": md,
        "confidence": conf,
        "engines_used": engines,
        "chars": chars,
        "is_blank": is_blank,
    }


def _match_keyword_marker(markdown: str) -> Optional[str]:
    """
    Check if the first portion of the page matches any hand-authored document-type marker.
    Looks primarily in the header region (first 1000 chars) to catch title blocks.
    """
    head = markdown[:1200].upper()
    for pattern, doc_type in DOCUMENT_TYPE_MARKERS:
        if re.search(pattern, head, re.IGNORECASE):
            return doc_type
    return None


def detect_segment_boundaries(
    pages_input: Union[List[Dict[str, Any]], List[PageOutput], List[str], Sequence[Any]],
) -> List[Tuple[int, int]]:
    """
    Heuristic boundary detection using signals already logged by Layers 1-2.
    Zero LLM calls.

    Signals:
    1. Keyword hits against document-type markers in page header.
    2. Blank/skip-routed pages (hard boundary).
    3. OCR engine change between consecutive pages.
    4. Confidence discontinuity (drop/rise vs. rolling average).
    5. Character-count jump (dense page followed by near-empty page).

    Returns:
        List of [start_page, end_page] tuples (1-indexed, inclusive).
    """
    if not pages_input:
        return []

    normalized_pages = [
        _extract_page_metadata(p, idx + 1)
        for idx, p in enumerate(pages_input)
    ]

    boundaries: List[int] = [0]  # Indices into normalized_pages where a segment starts
    rolling_conf: List[float] = []

    last_doc_type: Optional[str] = None

    for i, p in enumerate(normalized_pages):
        pnum = p["page_number"]
        conf = p["confidence"]
        chars = p["chars"]
        is_blank = p["is_blank"]
        doc_type_hit = _match_keyword_marker(p["markdown"])

        if i == 0:
            rolling_conf.append(conf)
            last_doc_type = doc_type_hit
            continue

        prev = normalized_pages[i - 1]
        prev_conf = prev["confidence"]
        prev_chars = prev["chars"]
        prev_is_blank = prev["is_blank"]

        is_boundary = False
        reason = ""

        # Signal 1: Hard boundary from blank / near-empty page
        if is_blank != prev_is_blank:
            is_boundary = True
            reason = "blank_page_transition"

        # Signal 2: Strong keyword hit indicating a different document type
        elif doc_type_hit is not None and doc_type_hit != last_doc_type:
            is_boundary = True
            reason = f"doc_type_marker_{doc_type_hit}"

        # Signal 3: OCR engine change between consecutive pages
        elif p["engines_used"] and prev["engines_used"] and set(p["engines_used"]) != set(prev["engines_used"]):
            is_boundary = True
            reason = "engine_shift"

        # Signal 4: Confidence discontinuity vs rolling average (e.g. sharp drop like page 4)
        else:
            avg_conf = sum(rolling_conf) / len(rolling_conf) if rolling_conf else conf
            conf_drop = avg_conf - conf
            if conf_drop >= 0.08 or (conf < 0.88 and avg_conf >= 0.94):
                is_boundary = True
                reason = f"confidence_discontinuity (avg={avg_conf:.3f}, curr={conf:.3f})"

            # Signal 5: Severe character count jump (e.g., dense report followed by very sparse page)
            elif prev_chars > 800 and chars < 200:
                is_boundary = True
                reason = f"char_count_drop ({prev_chars} -> {chars})"
            elif prev_chars < 200 and chars > 800:
                is_boundary = True
                reason = f"char_count_surge ({prev_chars} -> {chars})"

        if is_boundary:
            boundaries.append(i)
            rolling_conf = [conf]
            last_doc_type = doc_type_hit
            logger.debug(
                "segmentation.boundary_detected",
                page=pnum,
                reason=reason,
                doc_type=doc_type_hit,
            )
        else:
            rolling_conf.append(conf)
            if doc_type_hit:
                last_doc_type = doc_type_hit

    # Convert split indices into [start_page, end_page] tuples
    spans: List[Tuple[int, int]] = []
    for idx, start_idx in enumerate(boundaries):
        end_idx = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else len(normalized_pages) - 1
        start_page = normalized_pages[start_idx]["page_number"]
        end_page = normalized_pages[end_idx]["page_number"]
        spans.append((start_page, end_page))

    logger.info(
        "segmentation.boundaries_built",
        total_pages=len(normalized_pages),
        total_segments=len(spans),
        spans=spans,
    )
    return spans


def build_document_segments(
    pages_input: Union[List[Dict[str, Any]], List[PageOutput], List[str], Sequence[Any]],
    llm: Optional[ExtractionLLMClient] = None,
) -> List[Segment]:
    """
    Phase A entry point:
    1. Heuristic boundary detection (zero LLM calls).
    2. Exactly 1 LLM call per segment (O(S) calls) to populate doc_type_hint and one_line_summary.

    Returns:
        List of Segment objects.
    """
    spans = detect_segment_boundaries(pages_input)
    normalized_pages = [
        _extract_page_metadata(p, idx + 1)
        for idx, p in enumerate(pages_input)
    ]
    page_map = {p["page_number"]: p["markdown"] for p in normalized_pages}

    segments: List[Segment] = []
    for i, (start_p, end_p) in enumerate(spans, start=1):
        seg_id = f"seg_{i:02d}"
        page_range = [start_p, end_p]

        # Combine text across the segment
        seg_texts = [page_map.get(p_num, "") for p_num in range(start_p, end_p + 1)]
        full_text = "\n\n".join([t for t in seg_texts if t.strip()])

        # Fallback doc_type from heuristic marker if LLM call is unavailable or fails
        heuristic_hint = _match_keyword_marker(full_text) or "document"

        summary_info: Dict[str, Any] = {}
        if llm is not None and hasattr(llm, "summarize_segment"):
            try:
                summary_info = llm.summarize_segment(full_text, page_range)
            except Exception as exc:
                logger.warning("segmentation.llm_summary_failed", seg_id=seg_id, error=str(exc))

        doc_type_hint = summary_info.get("doc_type_hint") or heuristic_hint
        one_line_summary = summary_info.get("one_line_summary") or ""

        if not one_line_summary:
            # First non-empty lines as fallback summary
            clean_lines = [
                line_text
                for line in full_text.splitlines()
                if (line_text := line.strip()) and not line_text.startswith("<!--")
            ]
            one_line_summary = " // ".join(clean_lines[:2])[:150]

        segment = Segment(
            segment_id=seg_id,
            page_range=page_range,
            doc_type_hint=doc_type_hint.lower(),
            one_line_summary=one_line_summary,
        )
        segments.append(segment)
        logger.info(
            "segmentation.segment_ready",
            segment_id=seg_id,
            page_range=page_range,
            doc_type_hint=segment.doc_type_hint,
            summary=segment.one_line_summary[:80],
        )

    return segments
