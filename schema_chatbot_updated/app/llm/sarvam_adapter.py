"""
Sarvam AI adapter (https://docs.sarvam.ai).

Uses the OpenAI-compatible /v1/chat/completions endpoint with the
`response_format: {"type": "json_schema", ...}` structured-outputs mode,
which constrains the model's output to match ExtractionResult's shape
directly - no free-text JSON parsing needed.

Auth: api-subscription-key header (sk_xxx). Get a key at
https://dashboard.sarvam.ai and set SARVAM_API_KEY.

Same interface as OllamaAdapter/BedrockAdapter - swap in via
LLM_PROVIDER=sarvam, no other code changes required.
"""
from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from typing import Any, Dict, List

import httpx

from app.config import settings
from app.llm.base import ExtractionResult, LLMAdapter, SchemaProposal
from app.llm.prompts import (
    DOCUMENT_INFERENCE_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    build_document_inference_prompt,
    build_extraction_user_prompt,
    fallback_question,
)

logger = logging.getLogger(__name__)

# Reuses the same shape as the Bedrock tool schema - one schema, one
# source of truth for "what an extraction looks like" across providers.
_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": ["string", "null"]},
        "new_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": ["string", "null"]},
                    "required": {"type": ["boolean", "null"]},
                    "currency": {"type": ["string", "null"]},
                    "item_type": {"type": ["string", "null"]},
                },
                "required": ["name"],
            },
        },
        "field_answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value": {},
                },
                "required": ["field_name", "attribute", "value"],
            },
        },
        "removals": {"type": "array", "items": {"type": "string"}},
        "confirmation": {"type": ["boolean", "null"]},
        "needs_clarification": {"type": "boolean"},
        "clarification_reason": {"type": ["string", "null"]},
    },
    "required": ["needs_clarification"],
}

# Same shape as SchemaProposal - one source of truth for "what a document
# schema inference looks like", mirroring the extraction schema above.
_SCHEMA_PROPOSAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": ["string", "null"]},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": ["string", "null"]},
                    "required": {"type": ["boolean", "null"]},
                    "item_type": {"type": ["string", "null"]},
                    "pattern": {"type": ["string", "null"]},
                    "currency": {"type": ["string", "null"]},
                    "seen_in_samples": {"type": "integer"},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["name", "seen_in_samples"],
            },
        },
    },
    "required": ["fields"],
}


class SarvamAdapter(LLMAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ):
        self.api_key = api_key or settings.sarvam_api_key
        self.model = model or settings.sarvam_model
        self.base_url = (base_url or settings.sarvam_base_url).rstrip("/")
        self.timeout_s = timeout_s or settings.sarvam_timeout_s
        if not self.api_key:
            logger.warning("SarvamAdapter initialized without SARVAM_API_KEY - calls will fail with 403")
        # Lazy - the sarvamai SDK is only needed for Document AI (digitise),
        # not for plain chat completions, which use raw httpx like the rest
        # of this adapter.
        self._doc_ai_client = None

    def extract(self, state: str, user_message: str, context: Dict[str, Any]) -> ExtractionResult:
        user_prompt = build_extraction_user_prompt(state, user_message, context)
        try:
            raw = self._chat(
                system=EXTRACTION_SYSTEM_PROMPT,
                user=user_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema_extraction",
                        "strict": False,  # our schema has null-unions; keep non-strict for robustness
                        "schema": _EXTRACTION_JSON_SCHEMA,
                    },
                },
                # This is a structured extraction task, not open-ended reasoning -
                # disable thinking mode for lower latency/cost.
                reasoning_effort=None,
                temperature=0.1,
            )
            parsed = json.loads(raw)
        except Exception:
            logger.exception("Sarvam extraction call failed")
            return ExtractionResult(extraction_failed=True, needs_clarification=True,
                                     clarification_reason="LLM provider error")

        try:
            return ExtractionResult.model_validate(parsed)
        except Exception:
            logger.exception("Sarvam returned JSON that didn't match ExtractionResult: %r", parsed)
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
            text = self._chat(system=system, user=template, reasoning_effort=None, temperature=0.3)
            return text.strip() or template
        except Exception:
            logger.warning("Sarvam phrase_question call failed, using template fallback")
            return template

    def infer_schema_from_pdfs(self, samples: List[bytes]) -> SchemaProposal:
        """
        Two steps, deliberately kept separate:
          1. OCR each sample via Sarvam's Document AI "Digitise" job
             (powered by Sarvam Vision) -> clean Markdown text.
          2. Feed all N markdown texts into the plain text-only sarvam-105b
             model, via the exact same structured-output pattern as
             extract() above, to compare samples and propose a schema.
        This keeps the vision step (OCR) and the reasoning step (schema
        inference) decoupled - sarvam-105b itself never sees an image.
        """
        try:
            texts = [self._digitise(i, pdf_bytes) for i, pdf_bytes in enumerate(samples, start=1)]
        except Exception:
            logger.exception("Sarvam Document AI digitise failed")
            return SchemaProposal(extraction_failed=True, failure_reason="document OCR failed")

        # Truncate OCR text to avoid hitting API limits (max ~10K chars per sample)
        truncated_texts = [text[:10000] for text in texts]
        user_prompt = build_document_inference_prompt(truncated_texts)
        try:
            raw = self._chat(
                system=DOCUMENT_INFERENCE_SYSTEM_PROMPT,
                user=user_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema_proposal",
                        "strict": False,
                        "schema": _SCHEMA_PROPOSAL_JSON_SCHEMA,
                    },
                },
                reasoning_effort=None,
                temperature=0.1,
            )
            parsed = json.loads(raw)
        except Exception:
            logger.exception("Sarvam schema-inference call failed")
            return SchemaProposal(extraction_failed=True, failure_reason="LLM provider error")

        try:
            proposal = SchemaProposal.model_validate(parsed)
        except Exception:
            logger.exception("Sarvam returned JSON that didn't match SchemaProposal: %r", parsed)
            return SchemaProposal(extraction_failed=True, failure_reason="malformed model output")

        # total_samples is a fact we already know - don't trust the model for it.
        for field in proposal.fields:
            field.total_samples = len(samples)
        return proposal

    # ---- Document AI (OCR) ----

    @property
    def doc_ai_client(self):
        if self._doc_ai_client is None:
            from sarvamai import SarvamAI  # local import: keep sarvamai optional unless doc intake is used

            self._doc_ai_client = SarvamAI(api_subscription_key=self.api_key)
        return self._doc_ai_client

    def _digitise(self, sample_index: int, pdf_bytes: bytes) -> str:
        """Submits one sample to Sarvam's Digitise job and returns its Markdown text."""
        job = self.doc_ai_client.doc_ai.digitise(
            file=[(f"sample_{sample_index}.pdf", io.BytesIO(pdf_bytes), "application/pdf")],
            language=settings.sarvam_doc_ai_language,
            output_format="md",
        )

        terminal = {"completed", "partially_completed", "failed", "rejected"}
        deadline = time.monotonic() + settings.sarvam_doc_ai_timeout_s
        status = job.status
        job_id = job.job_id
        while status.lower() not in terminal:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Sarvam digitise job {job_id} did not finish within {settings.sarvam_doc_ai_timeout_s}s")
            time.sleep(settings.sarvam_doc_ai_poll_interval_s)
            status = self.doc_ai_client.doc_ai.get_status(job_id=job_id).status

        if status.lower() in ("failed", "rejected"):
            raise RuntimeError(f"Sarvam digitise job {job_id} ended with status {status!r}")

        download = self.doc_ai_client.doc_ai.get_download_url(job_id=job_id)
        resp = httpx.get(download.url, timeout=self.timeout_s)
        resp.raise_for_status()
        return self._extract_markdown(resp.content)

    @staticmethod
    def _extract_markdown(zip_bytes: bytes) -> str:
        """Digitise results download as a ZIP: the primary .md file, plus
        metadata/page_NNN.json per page and a manifest.json."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            candidates = [n for n in zf.namelist() if n.endswith(".md") and "metadata/" not in n]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.endswith(".md")]
            if not candidates:
                raise RuntimeError("Sarvam digitise result did not contain a markdown file")
            return zf.read(candidates[0]).decode("utf-8", errors="replace")

    # ---- low-level HTTP ----

    def _chat(
        self,
        system: str,
        user: str,
        response_format: Dict[str, Any] | None = None,
        reasoning_effort: str | None = "medium",
        temperature: float = 0.2,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "api-subscription-key": self.api_key or "",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
