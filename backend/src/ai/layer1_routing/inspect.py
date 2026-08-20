"""
File: inspect.py
Purpose: Layer 1 Step A — programmatic page inspection (PyMuPDF, no GPU, ~10ms/page).
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: pymupdf (fitz)
"""

from typing import Any

from src.ai.schemas.page import PageProfile
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Unicode block prefixes -> script name, used for cheap script detection
# without pulling in a full language-detection dependency.
_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "tamil": (0x0B80, 0x0BFF),
    "bengali": (0x0980, 0x09FF),
    "gujarati": (0x0A80, 0x0AFF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "odia": (0x0B00, 0x0B7F),
    "telugu": (0x0C00, 0x0C7F),
}


def detect_unicode_script(words: list) -> str:
    """Inspect embedded text and return the dominant script, defaulting to latin."""
    if not words:
        return "unknown"

    script_counts: dict[str, int] = {}
    for w in words:
        text = w[4] if len(w) > 4 else ""
        for ch in text:
            code = ord(ch)
            for script, (lo, hi) in _SCRIPT_RANGES.items():
                if lo <= code <= hi:
                    script_counts[script] = script_counts.get(script, 0) + 1
                    break

    if not script_counts:
        return "latin"
    return max(script_counts, key=script_counts.get)


def _detect_tables(page: Any) -> bool:
    """Heuristic: multiple horizontal + vertical vector lines suggest a table grid."""
    drawings = page.get_drawings()
    h_lines = sum(1 for d in drawings if d.get("type") == "l" and _is_horizontal(d))
    v_lines = sum(1 for d in drawings if d.get("type") == "l" and _is_vertical(d))
    return h_lines >= 2 and v_lines >= 2


def _is_horizontal(drawing: dict) -> bool:
    items = drawing.get("items", [])
    if not items:
        return False
    return True  # placeholder — real geometry check goes here in full impl


def _is_vertical(drawing: dict) -> bool:
    items = drawing.get("items", [])
    if not items:
        return False
    return True  # placeholder — real geometry check goes here in full impl


def inspect_page(page: Any, page_number: int) -> PageProfile:
    """
    Programmatic inspection of a single PyMuPDF page object.
    No GPU, no model calls — pure heuristics. Should complete in ~10ms.
    """
    words = page.get_text("words")
    char_count = sum(len(w[4]) for w in words) if words else 0

    images = page.get_image_info()
    image_area = sum(
        max(0, (i["bbox"][2] - i["bbox"][0])) * max(0, (i["bbox"][3] - i["bbox"][1]))
        for i in images
    )
    page_area = page.rect.width * page.rect.height
    image_coverage = image_area / page_area if page_area > 0 else 0.0

    has_tables = _detect_tables(page)
    has_vector_drawings = len(page.get_drawings()) > 0
    primary_script = detect_unicode_script(words)
    is_scanned = (
        char_count < settings.scanned_char_count_threshold
        and image_coverage > settings.scanned_image_coverage_threshold
    )

    complexity_score = _compute_complexity_score(
        has_tables=has_tables,
        has_vector_drawings=has_vector_drawings,
        char_count=char_count,
        image_coverage=image_coverage,
    )

    profile = PageProfile(
        page_number=page_number,
        has_text=char_count > 0,
        char_count=char_count,
        image_coverage=round(image_coverage, 4),
        has_tables=has_tables,
        is_scanned=is_scanned,
        has_vector_drawings=has_vector_drawings,
        primary_script=primary_script,
        complexity_score=complexity_score,
        dpi_estimate=_estimate_dpi(page, images),
    )

    logger.info(
        "page.inspected",
        page_number=page_number,
        char_count=char_count,
        image_coverage=round(image_coverage, 4),
        is_scanned=is_scanned,
        primary_script=primary_script,
        complexity_score=complexity_score,
    )
    return profile


def _compute_complexity_score(
    has_tables: bool, has_vector_drawings: bool, char_count: int, image_coverage: float
) -> int:
    """0-5 heuristic complexity score. Used by the router to force escalation
    on dense/complex layouts even when other signals look clean."""
    score = 0
    if has_tables:
        score += 2
    if has_vector_drawings:
        score += 1
    if 0 < char_count < 50:
        score += 1  # sparse text is often a form or noisy scan
    if 0.05 < image_coverage < 0.5:
        score += 1  # partial image coverage suggests mixed content
    return min(score, 5)


def _estimate_dpi(page: Any, images: list) -> int:
    """Rough DPI estimate for scanned pages, used to flag upscaling needs."""
    if not images:
        return 0
    page_width_in = page.rect.width / 72.0  # points -> inches
    if page_width_in <= 0:
        return 0
    widest_image = max((i["bbox"][2] - i["bbox"][0] for i in images), default=0)
    if widest_image <= 0:
        return 0
    return int(widest_image / page_width_in)
