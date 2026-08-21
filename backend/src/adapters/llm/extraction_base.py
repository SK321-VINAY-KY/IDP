"""
File: extraction_base.py
Purpose: Interface for Layer 3's text-extraction LLM client. Distinct from
         src/adapters/llm/base.py (LLMClient), which is Layer 1/2's
         vision-classification interface — different job, different model type.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: pydantic
"""
from typing import Protocol, List, Dict, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class ExtractionLLMClient(Protocol):
    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        ...

    def summarize_page(self, page_md: str, max_words: int) -> str:
        ...

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        ...