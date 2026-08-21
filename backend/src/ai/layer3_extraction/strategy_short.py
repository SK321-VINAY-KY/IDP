"""
File: strategy_short.py
Purpose: Strategy A — concatenate all pages, extract schema directly.
Owner: engineer-b@idp-pilot
Created: 2026-08-20
"""
from typing import List
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient


def extract_short_doc(pages_md: List[str], schema: type[BaseModel], llm: ExtractionLLMClient) -> BaseModel:
    full_content = "\n\n---\n\n".join(
        f"[Page {i + 1}]\n{md}" for i, md in enumerate(pages_md)
    )
    return llm.extract(full_content, schema)