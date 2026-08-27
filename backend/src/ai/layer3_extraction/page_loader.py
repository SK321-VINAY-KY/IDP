"""
File: page_loader.py
Purpose: Two ways to get pages into Layer 3's list[str] contract:
         1. load_pages_from_outputs() — the REAL path. Converts Layer 1/2's
            list[PageOutput] (from pipeline.process_document()) into the
            list[str] every Layer 3 strategy expects.
         2. load_pages_from_fixture() — TESTING ONLY. Reads a saved .md
            fixture file with "<!-- Page N | ... -->" markers, for quick
            manual testing without running the full Layer 1/2 pipeline.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-20 (added real PageOutput adapter)
"""
import re
from pathlib import Path
from typing import List

from src.ai.schemas.page import PageOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s*\|[^>]*-->")


def load_pages_from_outputs(page_outputs: List[PageOutput]) -> List[str]:
    """
    THE REAL PATH. Converts Layer 1/2's list[PageOutput] into the list[str]
    contract every Layer 3 strategy expects — one Markdown string per page,
    in page order.

    Call this with the output of:
        from src.ai.layer1_routing.pipeline import process_document
        page_outputs = process_document(pages, llm_client)
        pages_md = load_pages_from_outputs(page_outputs)
    """
    sorted_outputs = sorted(page_outputs, key=lambda p: p.page_number)

    low_conf_pages = [p.page_number for p in sorted_outputs if p.low_confidence]
    if low_conf_pages:
        logger.warning(
            "page_loader.low_confidence_pages_included",
            pages=low_conf_pages,
            total_pages=len(sorted_outputs),
        )

    return [p.markdown for p in sorted_outputs]


def load_pages_from_fixture(doc_id: str) -> List[str]:
    """
    TESTING ONLY. Reads tests/fixtures/<doc_id>.md, splits on
    "<!-- Page N | route=... | ... -->" markers. Use this for manual
    testing against saved Layer 2 sample output, not in the real pipeline.
    """
    file_path = FIXTURES_DIR / f"{doc_id}.md"
    if not file_path.exists():
        raise FileNotFoundError(f"No fixture found for doc_id={doc_id!r} at {file_path}")

    content = file_path.read_text(encoding="utf-8")
    matches = list(PAGE_MARKER.finditer(content))
    if not matches:
        return [content.strip()]

    pages = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        pages.append(content[start:end].strip())

    return pages


def load_pages(doc_id: str) -> List[str]:
    """Backward-compatible name for the fixture loader."""
    return load_pages_from_fixture(doc_id)