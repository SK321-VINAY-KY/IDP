"""
File: strategy_long_pageindex.py
Purpose: Strategy B — PageIndex tree navigation for long docs.
Owner: engineer-b@idp-pilot
Created: 2026-08-20
"""
from typing import List
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_long_doc_pageindex(pages_md: List[str], schema: type[BaseModel], llm: ExtractionLLMClient) -> BaseModel:
    page_summaries = [
        f"Page {i + 1}: {llm.summarize_page(md, settings.page_summary_max_words)}"
        for i, md in enumerate(pages_md)
    ]

    # Log how much content each page has — empty pages here means Layer 2 produced no text
    empty_pages = [i + 1 for i, md in enumerate(pages_md) if not md.strip()]
    if empty_pages:
        logger.warning("pageindex.empty_pages_detected", empty_pages=empty_pages, total_pages=len(pages_md))
    logger.info(
        "pageindex.page_content_sizes",
        total_pages=len(pages_md),
        avg_chars=round(sum(len(md) for md in pages_md) / len(pages_md)) if pages_md else 0,
        empty_page_count=len(empty_pages),
    )

    schema_fields = list(schema.model_fields.keys())
    page_map = llm.navigate(page_summaries, schema_fields)
    logger.info("pageindex.page_map", page_map=page_map)

    # CHANGED: int(p) — the model can return page numbers as strings ("1")
    # instead of integers (1); this normalizes either case before the
    # arithmetic below (pages_md[p - 1]) runs.
    relevant_pages = sorted({int(p) for pages in page_map.values() for p in pages})

    if not relevant_pages:
        logger.warning("pageindex.navigation_empty_fallback_full_doc", total_pages=len(pages_md))
        relevant_pages = list(range(1, len(pages_md) + 1))

    logger.info(
        "pageindex.navigated",
        total_pages=len(pages_md),
        relevant_pages=relevant_pages,
        reduction_pct=round(100 * (1 - len(relevant_pages) / len(pages_md)), 1),
    )

    relevant_content = "\n\n---\n\n".join(
        f"[Page {p}]\n{pages_md[p - 1]}" for p in relevant_pages
    )
    logger.info(
        "pageindex.extract_input",
        relevant_pages=relevant_pages,
        content_chars=len(relevant_content),
        content_preview=relevant_content[:300],
    )
    result = llm.extract(relevant_content, schema)
    complete_from_text = getattr(schema, "complete_from_text", None)
    if complete_from_text is not None:
        return complete_from_text(result, relevant_content)
    return result