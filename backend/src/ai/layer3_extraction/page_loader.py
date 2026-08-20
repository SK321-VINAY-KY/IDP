"""
File: page_loader.py
Purpose: Load per-page Markdown for Layer 3 extraction, parsed from Layer 2's
         real single-file output format (pages separated by
         "<!-- Page N | route=... | ... -->" comment markers).
Owner: genai-platform@shellkode
Created: 2026-08-20
"""
import re
from pathlib import Path
from typing import List

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"

PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s*\|[^>]*-->")


def load_pages(doc_id: str) -> List[str]:
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