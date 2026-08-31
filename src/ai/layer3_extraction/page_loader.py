"""
File: page_loader.py
Purpose: Convert Layer 1/2 PageOutput objects into the list[dict] contract
         that Layer 3's extractor expects.

Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-28
"""
import re
from pathlib import Path
from typing import Dict, List

from src.ai.schemas.page import PageOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s*\|[^>]*-->", re.IGNORECASE)
PAGE_CLOSING_MARKER = re.compile(r"<!--\s*/PAGE(?:\s+\d+)?\s*-->", re.IGNORECASE)


def load_pages_with_confidence(page_outputs: List[PageOutput]) -> List[Dict]:
    """Real pipeline path — kept for API use."""
    sorted_outputs = sorted(page_outputs, key=lambda p: p.page_number)
    low_conf_pages = [p.page_number for p in sorted_outputs if p.low_confidence]
    if low_conf_pages:
        logger.warning("page_loader.low_confidence_pages_included",
                       pages=low_conf_pages, total_pages=len(sorted_outputs))
    return [{"markdown": p.markdown, "page_number": p.page_number} for p in sorted_outputs]


def load_pages_from_outputs(page_outputs: List[PageOutput]) -> List[str]:
    """Backward-compatible: returns plain list[str]."""
    return [p["markdown"] for p in load_pages_with_confidence(page_outputs)]


def load_pages_from_fixture(doc_id: str) -> List[Dict]:
    """TESTING ONLY. Reads fixture .md file and returns list[dict]."""
    file_path = FIXTURES_DIR / f"{doc_id}.md"
    if not file_path.exists():
        raise FileNotFoundError(f"No fixture found for doc_id={doc_id!r} at {file_path}")

    content = file_path.read_text(encoding="utf-8")
    matches = list(PAGE_MARKER.finditer(content))
    if not matches:
        return [{"markdown": content.strip(), "page_number": 1}]

    pages = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        raw_slice = content[start:end].strip()
        cleaned_markdown = PAGE_CLOSING_MARKER.split(raw_slice)[0].strip()
        pages.append({
            "markdown":    cleaned_markdown,
            "page_number": int(match.group(1)),
        })
    return pages


def load_pages(doc_id: str) -> List[Dict]:
    return load_pages_from_fixture(doc_id)
