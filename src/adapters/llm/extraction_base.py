"""
File: extraction_base.py
Purpose: Interface for Layer 3's text-extraction LLM client. Distinct from
         src/adapters/llm/base.py (LLMClient), which is Layer 1/2's
         vision-classification interface — different job, different model type.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: pydantic
"""
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class ExtractionLLMClient(Protocol):
    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        ...

    def summarize_page(self, page_md: str, max_words: int) -> str:
        ...

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        ...

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        ...

    def summarize_segment(
        self,
        segment_text: str,
        page_range: List[int],
    ) -> Dict[str, str]:
        ...

    def navigate_category(
        self,
        segment_summaries: List[Dict[str, Any]],
        category: str,
        category_description: str,
        category_examples: Optional[List[str]] = None,
        extraction_focus: Optional[List[str]] = None,
    ) -> List[str]:
        ...

    def extract_page_ontology(
        self,
        page_md: str,
        page_number: int,
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ...

    def extract_key_findings(
        self,
        segments: List[Dict[str, Any]],
        candidate_nodes: List[Dict[str, Any]],
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ...