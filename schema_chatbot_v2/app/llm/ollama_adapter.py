"""
Ollama adapter - talks to a local Ollama server (default http://localhost:11434).

Uses Ollama's `format: "json"` constrained-decoding mode so we get parseable
JSON back instead of having to regex free text out of a chat response.

Swap-out note: OllamaAdapter and BedrockAdapter both implement LLMAdapter,
so moving from local dev to AWS is a one-line config change
(LLM_PROVIDER=bedrock), not a code change in conversation_manager.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx

from app.config import settings
from app.llm.base import ExtractionResult, LLMAdapter, SchemaProposal
from app.llm.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt, fallback_question

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout_s = timeout_s or settings.ollama_timeout_s

    def extract(self, state: str, user_message: str, context: Dict[str, Any]) -> ExtractionResult:
        user_prompt = build_extraction_user_prompt(state, user_message, context)
        try:
            raw = self._chat_json(EXTRACTION_SYSTEM_PROMPT, user_prompt)
        except Exception:
            logger.exception("Ollama extraction call failed")
            return ExtractionResult(extraction_failed=True, needs_clarification=True,
                                     clarification_reason="LLM provider error")

        try:
            return ExtractionResult.model_validate(raw)
        except Exception:
            logger.exception("Ollama returned JSON that didn't match ExtractionResult: %r", raw)
            return ExtractionResult(extraction_failed=True, needs_clarification=True,
                                     clarification_reason="malformed model output")

    def phrase_question(self, gap_field: str, gap_attribute: str, context: Dict[str, Any]) -> str:
        system = (
            "Rephrase the following into one short, friendly, plain-English "
            "question for a non-technical user. Respond with ONLY the question, "
            "no quotes, no preamble."
        )
        template = fallback_question(gap_field, gap_attribute)
        try:
            resp = self._chat_text(system, template)
            return resp.strip() or template
        except Exception:
            logger.warning("Ollama phrase_question call failed, using template fallback")
            return template

    def infer_schema_from_pdfs(self, samples: List[bytes]) -> SchemaProposal:
        # Local Ollama models here (llama3.1) are text-only - no reliable way
        # to read a PDF. Degrade gracefully rather than raising, so the
        # caller falls back to the plain-text interview (see README's
        # "LLM failures degrade gracefully" principle).
        logger.warning("OllamaAdapter.infer_schema_from_pdfs called - document intake is not supported on this provider")
        return SchemaProposal(
            extraction_failed=True,
            failure_reason=(
                "document upload isn't supported with LLM_PROVIDER=ollama "
                "(no vision model configured) - switch to sarvam or bedrock, "
                "or continue by chat"
            ),
        )

    # ---- low-level HTTP ----

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)

    def _chat_text(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        }
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
