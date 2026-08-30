"""
File: sarvam_client.py
Purpose: Sarvam AI implementation of ExtractionLLMClient.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-28
Deps: instructor, openai
"""

import json
import re
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


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences (```json ... ```) before JSON parsing."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class SarvamExtractionClient:
    def __init__(self) -> None:
        if instructor is None:
            raise ImportError("pip install instructor openai")
        if not settings.sarvam_api_key:
            raise ValueError("IDP_SARVAM_API_KEY is not set.")
        self.model = settings.sarvam_model_name
        # instructor client — used for extract() with structured response_model
        self.client = instructor.from_openai(
            OpenAI(base_url=settings.sarvam_base_url, api_key=settings.sarvam_api_key),
            mode=instructor.Mode.JSON,
        )
        # raw OpenAI client — used for check_page_for_fields() which returns free-form JSON
        self.raw_client = OpenAI(
            base_url=settings.sarvam_base_url,
            api_key=settings.sarvam_api_key,
        )

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        system_prompt = render_prompt("extraction", schema_fields=schema_fields)
        params = prompt_params("extraction")
        logger.info("sarvam.extract.request", model=self.model, content_chars=len(content))

        common_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_model=schema,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"reasoning_effort": None},
        )
        try:
            result, completion = self.client.chat.completions.create_with_completion(**common_kwargs)
            finish_reason = completion.choices[0].finish_reason if completion.choices else None
            if finish_reason == "length":
                logger.warning("sarvam.extract.truncated", model=self.model, max_tokens=params["max_tokens"])
            if all(v in ("", None) for v in result.model_dump().values()):
                logger.warning("sarvam.extract.all_fields_empty", model=self.model, content_chars=len(content))
        except AttributeError:
            result = self.client.chat.completions.create(**common_kwargs)

        logger.info("sarvam.extract.ok", model=self.model)
        return result

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Scan a single page for schema fields.
        Returns list of {"field", "value"} dicts — empty list if nothing found.
        """
        prompt = render_prompt(
            "page_field_check",
            page_md=page_md,
            schema_fields=schema_fields,
            page_number=page_number,
            total_pages=total_pages,
        )
        params = prompt_params("page_field_check")

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"reasoning_effort": None},
        )
        raw = resp.choices[0].message.content
        if raw is None:
            logger.warning("sarvam.check_page_for_fields.empty_content",
                           finish_reason=resp.choices[0].finish_reason)
            return []
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            matches = parsed.get("matches", [])
            if not isinstance(matches, list):
                logger.warning("sarvam.check_page_for_fields.bad_matches", raw=raw[:200])
                return []
            valid_names = {f["name"] for f in schema_fields}
            result = []
            for m in matches:
                field = m.get("field", "")
                value = m.get("value", "")
                if field in valid_names and value not in ("", None):
                    result.append({"field": field, "value": value})
            return result
        except json.JSONDecodeError:
            logger.warning("sarvam.check_page_for_fields.parse_failed", raw=raw[:200])
            return []
