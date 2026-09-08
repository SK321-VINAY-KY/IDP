"""
File: extraction_client.py
Purpose: Ollama-backed text extraction for Layer 3 (Qwen2.5, text-only).
         Reuses settings.ollama_base_url already defined for Layer 1/2's
         vision client — same Ollama instance, different model pulled.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: instructor, openai
"""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.config.settings import settings
from src.ai.layer3_extraction.prompts.loader import render_prompt, prompt_params
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import instructor
    from openai import OpenAI
except ImportError:
    instructor = None
    OpenAI = None


class OllamaExtractionClient:
    def __init__(self) -> None:
        if instructor is None or OpenAI is None:
            raise ImportError("pip install instructor openai")
        self.model = settings.extraction_model_name
        self.summary_model = settings.summary_model_name
        self.raw_client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # pragma: allowlist secret
            timeout=120.0,
        )
        self.client = instructor.from_openai(
            self.raw_client,
            mode=instructor.Mode.JSON,
        )

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        # Build list of {name, description} dicts so the prompt can show
        # field descriptions alongside names for better extraction accuracy.
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        system_prompt = render_prompt("extraction", schema_fields=schema_fields)
        params = prompt_params("extraction")
        logger.info("extraction.request", model=self.model, content_chars=len(content))
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_model=schema,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 16384}},
        )
        logger.info("extraction.ok", model=self.model)
        return result

    def summarize_page(self, page_md: str, max_words: int) -> str:
        prompt = render_prompt("page_summary", page_md=page_md, max_words=max_words)
        params = prompt_params("page_summary")
        resp = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 4096}},
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        prompt = render_prompt("navigation", page_summaries="\n".join(page_summaries), schema_fields=schema_fields)
        params = prompt_params("navigation")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 8192}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("navigate.parse_failed", raw=raw[:200])
            return {f: [] for f in schema_fields}

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        prompt = render_prompt(
            "page_field_check",
            page_md=page_md,
            schema_fields=schema_fields,
            page_number=page_number,
            total_pages=total_pages,
        )
        params = prompt_params("page_field_check")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 8192}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            return parsed.get("matches", [])
        except json.JSONDecodeError:
            logger.warning("ollama.check_page_for_fields.parse_failed", raw=raw[:200])
            return []

    def summarize_segment(
        self,
        segment_text: str,
        page_range: List[int],
    ) -> Dict[str, str]:
        prompt = render_prompt(
            "segment_summary",
            segment_text=segment_text[:4000],
            page_range=page_range,
        )
        params = prompt_params("segment_summary")
        resp = self.client.chat.completions.create(
            model=self.summary_model or self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 4096}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {
                    "doc_type_hint": str(parsed.get("doc_type_hint", "unknown")),
                    "one_line_summary": str(parsed.get("one_line_summary", "")),
                }
        except Exception:
            logger.warning("ollama.summarize_segment.parse_failed", raw=raw[:200])
        return {"doc_type_hint": "unknown", "one_line_summary": raw.strip()[:200]}

    def navigate_category(
        self,
        segment_summaries: List[Dict[str, Any]],
        category: str,
        category_description: str,
        category_examples: Optional[List[str]] = None,
        extraction_focus: Optional[List[str]] = None,
    ) -> List[str]:
        prompt = render_prompt(
            "category_navigation",
            segments=segment_summaries,
            category=category,
            category_description=category_description,
            category_examples=category_examples or [],
            extraction_focus=extraction_focus,
        )
        params = prompt_params("category_navigation")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 8192}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "segment_ids" in parsed:
                return [str(s) for s in parsed["segment_ids"]]
        except Exception:
            logger.warning("ollama.category_navigation.parse_failed", category=category, raw=raw[:200])
        return []

    def extract_page_ontology(
        self,
        page_md: str,
        page_number: int,
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = render_prompt(
            "page_ontology_extraction",
            page_md=page_md[:6000],
            page_number=page_number,
            extraction_focus=extraction_focus,
        )
        params = prompt_params("page_ontology_extraction")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 8192}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {
                    "nodes": parsed.get("nodes", []),
                    "edges": parsed.get("edges", []),
                }
        except Exception:
            logger.warning("ollama.extract_page_ontology.parse_failed", page=page_number, raw=raw[:200])
        return {"nodes": [], "edges": []}

    def extract_key_findings(
        self,
        segments: List[Dict[str, Any]],
        candidate_nodes: List[Dict[str, Any]],
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = render_prompt(
            "document_key_findings",
            segments=segments,
            candidate_nodes=candidate_nodes,
            extraction_focus=extraction_focus,
        )
        params = prompt_params("document_key_findings")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"options": {"num_ctx": 8192}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {"key_findings": parsed.get("key_findings", [])}
        except Exception:
            logger.warning("ollama.extract_key_findings.parse_failed", raw=raw[:200])
        return {"key_findings": []}


# Generic alias for callers importing ExtractionClient
ExtractionClient = OllamaExtractionClient