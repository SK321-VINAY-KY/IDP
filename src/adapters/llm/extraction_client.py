"""
File: extraction_client.py
Purpose: Ollama-backed text extraction for Layer 3 (Qwen2.5, text-only).
         Reuses settings.ollama_base_url already defined for Layer 1/2's
         vision client — same Ollama instance, different model pulled.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: instructor, openai
"""
import json
from typing import List, Dict, Any
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