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

    schema_fields = list(schema.model_fields.keys())
    page_map = llm.navigate(page_summaries, schema_fields)

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
    return llm.extract(relevant_content, schema)