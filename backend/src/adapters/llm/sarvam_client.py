"""
File: sarvam_client.py
Purpose: Sarvam AI implementation of ExtractionLLMClient. Uses their
         OpenAI-compatible endpoint (Bearer token auth) so the same
         `instructor` code path works unchanged — only base_url,
         api_key, and model name differ from the Ollama client.

         NOTE: Sarvam's "thinking mode" is ON by default (reasoning_effort
         defaults to "low"). For simple tasks like summarization and
         navigation, this can consume the entire max_tokens budget on
         internal reasoning, leaving message.content = None. Every call
         below explicitly passes reasoning_effort=None to disable this.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-26 (extraction diagnostics; raised max_tokens)
Deps: instructor, openai
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


class SarvamExtractionClient:
    """
    Sarvam AI, OpenAI-compatible mode. Requires IDP_SARVAM_API_KEY set —
    this is a paid API, not free/local like Ollama. See Sarvam's pricing
    page before using this for anything beyond light testing.
    """

    def __init__(self) -> None:
        if instructor is None:
            raise ImportError("pip install instructor openai")
        if not settings.sarvam_api_key:
            raise ValueError(
                "IDP_SARVAM_API_KEY is not set. Add it to .env before using the sarvam backend."
            )
        self.model = settings.sarvam_model_name
        self.client = instructor.from_openai(
            OpenAI(
                base_url=settings.sarvam_base_url,
                api_key=settings.sarvam_api_key,
            ),
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
        logger.info("sarvam.extract.request", model=self.model, content_chars=len(content))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        common_kwargs = dict(
            model=self.model,
            messages=messages,
            response_model=schema,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"reasoning_effort": None},
        )

        # DIAGNOSTIC (2026-08-26): extract() previously had no visibility into
        # whether Sarvam's reasoning mode ate the whole max_tokens budget before
        # emitting an answer — unlike summarize_page/navigate, which log and
        # guard against that. create_with_completion() gives us the raw
        # completion alongside the parsed model so we can catch that case here
        # too, instead of silently returning an all-default-values object.
        try:
            result, completion = self.client.chat.completions.create_with_completion(
                **common_kwargs
            )
            finish_reason = completion.choices[0].finish_reason if completion.choices else None
            if finish_reason == "length":
                logger.warning(
                    "sarvam.extract.truncated",
                    model=self.model,
                    finish_reason=finish_reason,
                    max_tokens=params["max_tokens"],
                )
            if all(v in ("", None) for v in result.model_dump().values()):
                logger.warning(
                    "sarvam.extract.all_fields_empty",
                    model=self.model,
                    finish_reason=finish_reason,
                    content_chars=len(content),
                )
        except AttributeError:
            # Older instructor versions without create_with_completion — fall
            # back to the original call with no extra diagnostics.
            result = self.client.chat.completions.create(**common_kwargs)

        logger.info("sarvam.extract.ok", model=self.model)
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
            extra_body={"reasoning_effort": None},
        )
        content = resp.choices[0].message.content
        if content is None:
            logger.warning("sarvam.summarize_page.empty_content", finish_reason=resp.choices[0].finish_reason)
            return ""
        return content.strip()

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        prompt = render_prompt(
            "navigation",
            page_summaries="\n".join(page_summaries),
            schema_fields=schema_fields,
        )
        params = prompt_params("navigation")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"reasoning_effort": None},
        )
        raw = resp.choices[0].message.content
        if raw is None:
            logger.warning("sarvam.navigate.empty_content", finish_reason=resp.choices[0].finish_reason)
            return {f: [] for f in schema_fields}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("sarvam.navigate.parse_failed", raw=raw[:200])
            return {f: [] for f in schema_fields}