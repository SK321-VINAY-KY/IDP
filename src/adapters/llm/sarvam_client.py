"""
File: sarvam_client.py
Purpose: Sarvam AI implementation of ExtractionLLMClient for Layer 3.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-31
Deps: instructor, openai
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
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
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class SarvamExtractionClient:
    def __init__(self) -> None:
        if instructor is None or OpenAI is None:
            raise ImportError("pip install instructor openai")

        api_key = settings.sarvam_api_key or os.getenv("IDP_SARVAM_API_KEY") or os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            raise ValueError("IDP_SARVAM_API_KEY or SARVAM_API_KEY is not set.")

        base_url = (settings.sarvam_base_url or os.getenv("IDP_SARVAM_BASE_URL") or os.getenv("SARVAM_BASE_URL") or "https://api.sarvam.ai/v1").rstrip("/")
        self.model = settings.sarvam_model_name or os.getenv("IDP_SARVAM_MODEL_NAME") or os.getenv("SARVAM_MODEL") or "sarvam-105b"
        self.timeout = float(getattr(settings, "sarvam_timeout_s", None) or os.getenv("SARVAM_TIMEOUT_S") or 180.0)

        # Base OpenAI client configured for Sarvam
        self.raw_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout,
        )

        # Instructor client for structured extraction
        self.client = instructor.from_openai(
            self.raw_client,
            mode=instructor.Mode.JSON,
        )

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        """
        Extract structured Pydantic schema from content using Sarvam LLM.
        """
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        system_prompt = render_prompt("extraction", schema_fields=schema_fields)
        params = prompt_params("extraction")
        logger.info("sarvam.extract.request", model=self.model, content_chars=len(content))

        common_kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_model=schema,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 4000),
        )

        try:
            try:
                result, completion = self.client.chat.completions.create_with_completion(**common_kwargs)
                finish_reason = completion.choices[0].finish_reason if completion.choices else None
                if finish_reason == "length":
                    logger.warning("sarvam.extract.truncated", model=self.model, max_tokens=params.get("max_tokens"))
                if all(v in ("", None) for v in result.model_dump().values()):
                    logger.warning("sarvam.extract.all_fields_empty", model=self.model, content_chars=len(content))
                return result
            except AttributeError:
                result = self.client.chat.completions.create(**common_kwargs)
                return result
        except Exception as exc:
            logger.warning("sarvam.extract.instructor_failed", error=str(exc))
            # Fallback to direct raw JSON completion
            try:
                raw_resp = self.raw_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching the schema fields."},
                        {"role": "user", "content": content},
                    ],
                    temperature=params.get("temperature", 0.0),
                    max_tokens=params.get("max_tokens", 4000),
                )
                choice = raw_resp.choices[0]
                text = choice.message.content or getattr(choice.message, "reasoning_content", "") or ""
                text = _strip_fences(text)
                parsed = json.loads(text)
                return schema.model_validate(parsed)
            except Exception as raw_exc:
                logger.error("sarvam.extract.raw_fallback_failed", error=str(raw_exc))
                raise exc

    def summarize_page(self, page_md: str, max_words: int = 150) -> str:
        """
        Summarize a single document page in max_words or fewer.
        """
        prompt = render_prompt("page_summary", page_md=page_md, max_words=max_words)
        params = prompt_params("page_summary")

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 500),
        )
        choice = resp.choices[0]
        content = choice.message.content or getattr(choice.message, "reasoning_content", "") or ""
        return content.strip()

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        """
        Map target schema fields to likely page numbers using page summaries.
        """
        prompt = render_prompt(
            "navigation",
            page_summaries="\n".join(page_summaries),
            schema_fields=schema_fields,
        )
        params = prompt_params("navigation")

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 1000),
        )
        choice = resp.choices[0]
        raw = choice.message.content or getattr(choice.message, "reasoning_content", "") or ""
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): list(v) if isinstance(v, list) else [] for k, v in parsed.items()}
            return {f: [] for f in schema_fields}
        except json.JSONDecodeError:
            logger.warning("sarvam.navigate.parse_failed", raw=raw[:200])
            return {f: [] for f in schema_fields}

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Scan a single page for target schema fields.
        Supports both {"matches": [{"field": ..., "value": ...}]} and direct {"field": "value"} maps.
        """
        valid_names = {f["name"] for f in schema_fields}
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
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 4000),
        )
        choice = resp.choices[0]
        raw = choice.message.content or getattr(choice.message, "reasoning_content", "") or ""
        if not raw:
            logger.warning("sarvam.check_page_for_fields.empty_content",
                           finish_reason=choice.finish_reason)
            return []

        raw = _strip_fences(raw)
        matches: List[Any] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if "matches" in parsed and isinstance(parsed["matches"], list):
                    matches = parsed["matches"]
                else:
                    # Direct dictionary format {field_name: value}
                    matches = [{"field": k, "value": str(v)} for k, v in parsed.items() if k in valid_names]
            elif isinstance(parsed, list):
                matches = parsed
        except json.JSONDecodeError:
            # Fallback 1: repair incomplete JSON object or array
            try:
                last_obj_end = raw.rfind("}")
                if last_obj_end != -1:
                    repaired = raw[:last_obj_end + 1] + "\n]}"
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        matches = parsed.get("matches", [])
                    elif isinstance(parsed, list):
                        matches = parsed
            except Exception:
                pass

            # Fallback 2: Regex extraction of {"field": "...", "value": "..."} pairs
            if not matches:
                pattern = re.compile(
                    r'\{\s*"field"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    re.DOTALL,
                )
                for m in pattern.finditer(raw):
                    f_name = m.group(1)
                    val = m.group(2).encode().decode("unicode_escape", errors="ignore")
                    matches.append({"field": f_name, "value": val})

            if not matches:
                logger.warning("sarvam.check_page_for_fields.parse_failed", raw=raw[:200])
                return []

        if not isinstance(matches, list):
            logger.warning("sarvam.check_page_for_fields.bad_matches", raw=raw[:200])
            return []

        result = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            field = m.get("field", "")
            value = m.get("value", "")
            if field in valid_names and value not in ("", None):
                result.append({"field": str(field), "value": str(value)})
        return result
