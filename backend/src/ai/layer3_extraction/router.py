"""
File: router.py
Purpose: Route to Strategy A (short) or Strategy B (long/PageIndex) by page count.
Owner: genai-platform@shellkode
Created: 2026-08-20
"""
from typing import List
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.strategy_short import extract_short_doc
from src.ai.layer3_extraction.strategy_long_pageindex import extract_long_doc_pageindex
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def route_and_extract(pages_md: List[str], schema: type[BaseModel], llm: ExtractionLLMClient) -> BaseModel:
    strategy = "A_short" if len(pages_md) < settings.short_doc_page_limit else "B_pageindex"
    logger.info("router.decision", page_count=len(pages_md), strategy=strategy)

    if strategy == "A_short":
        return extract_short_doc(pages_md, schema, llm)
    return extract_long_doc_pageindex(pages_md, schema, llm)