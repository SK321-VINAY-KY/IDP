"""
File: extraction_client.py
Purpose: Ollama-backed text extraction for Layer 3 (Qwen2.5, text-only).
         Reuses settings.ollama_base_url already defined for Layer 1/2's
         vision client — same Ollama instance, different model pulled.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: instructor, openai
"""
import json
from typing import List, Dict
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
        if instructor is None:
            raise ImportError("pip install instructor openai")
        self.model = settings.extraction_model_name
        self.client = instructor.from_openai(
            OpenAI(base_url=settings.ollama_base_url, api_key="ollama"),  # pragma: allowlist secret
            mode=instructor.Mode.JSON,
        )

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        system_prompt = render_prompt("extraction", schema_fields=list(schema.model_fields.keys()))
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
        )
        logger.info("extraction.ok", model=self.model)
        return result

    def summarize_page(self, page_md: str, max_words: int) -> str:
        prompt = render_prompt("page_summary", page_md=page_md, max_words=max_words)
        params = prompt_params("page_summary")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return resp.choices[0].message.content.strip()

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        prompt = render_prompt("navigation", page_summaries="\n".join(page_summaries), schema_fields=schema_fields)
        params = prompt_params("navigation")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("navigate.parse_failed", raw=raw[:200])
            return {f: [] for f in schema_fields}