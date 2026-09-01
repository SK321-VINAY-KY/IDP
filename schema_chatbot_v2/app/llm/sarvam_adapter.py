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


class TruncatedCompletionError(RuntimeError):
    """Raised when the LLM response is truncated due to hitting max_tokens (finish_reason='length')."""

    def __init__(self, message: str, finish_reason: str = "length"):
        super().__init__(message)
        self.finish_reason = finish_reason


class ChatResult(str):
    """Subclass of str carrying response metadata (reasoning_content, reasoning_len, completion_tokens, finish_reason)."""
    reasoning_content: str
    reasoning_len: int
    completion_tokens: int | None
    finish_reason: str | None

    def __new__(
        cls,
        content: str,
        reasoning_content: str = "",
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ):
        obj = super().__new__(cls, content)
        obj.reasoning_content = reasoning_content
        obj.reasoning_len = len(reasoning_content)
        obj.completion_tokens = completion_tokens
        obj.finish_reason = finish_reason
        return obj

# Reuses the same shape as the Bedrock tool schema - one schema, one
# source of truth for "what an extraction looks like" across providers.
# One "operations" list (add/update/remove) replaces the old separate
# new_fields/field_answers/removals lists, so a single reply can add,
# correct, and remove fields together; "reply" lets the model compose the
# actual response text shown to the user instead of a template picking it.
_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": ["string", "null"]},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "remove"]},
                    "field_name": {"type": "string"},
                    "type": {"type": ["string", "null"]},
                    "required": {"type": ["boolean", "null"]},
                    "currency": {"type": ["string", "null"]},
                    "item_type": {"type": ["string", "null"]},
                    "pattern": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                },
                "required": ["op", "field_name"],
            },
        },
        "confirmation": {"type": ["boolean", "null"]},
        "reply": {"type": ["string", "null"]},
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
        sys_prompt = (
            EXTRACTION_SYSTEM_PROMPT
            + "\n\nCRITICAL: Respond with ONLY the raw valid JSON object matching the ExtractionResult schema. No prose or explanations."
        )
        try:
            raw = self._chat(
                system=sys_prompt,
                user=user_prompt,
                temperature=0.1,
                max_tokens=4096,
            )
            parsed = self._parse_json(raw)
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
            text = self._chat(system=system, user=template, reasoning_effort=None, temperature=0.3, max_tokens=512)
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

        # Cap each sample to ~3,500 characters to ensure fast, high-quality inference without timeouts
        capped_texts = [t[:3500] if len(t) > 3500 else t for t in texts]
        user_prompt = build_document_inference_prompt(capped_texts)
        sys_prompt = (
            DOCUMENT_INFERENCE_SYSTEM_PROMPT
            + "\n\nCRITICAL: Keep reasoning brief. Output ONLY the valid JSON object with keys 'document_type' and 'fields'. Do not wrap in commentary."
        )
        try:
            raw = self._chat(
                system=sys_prompt,
                user=user_prompt,
                temperature=0.1,
                reasoning_effort=None,
                max_tokens=2560,
            )
            reasoning_len = getattr(raw, "reasoning_len", len(getattr(raw, "reasoning_content", "")))
            completion_tokens = getattr(raw, "completion_tokens", "N/A")
            logger.info("schema-inference reasoning_len=%d completion_tokens=%s", reasoning_len, completion_tokens)
            parsed = self._parse_json(raw)
        except TruncatedCompletionError as exc:
            logger.warning("Sarvam schema-inference output truncated: %s", exc)
            return SchemaProposal(
                extraction_failed=True,
                failure_reason="model output truncated (hit max_tokens)",
            )
        except Exception:
            logger.exception("Sarvam schema-inference call failed")
            return SchemaProposal(extraction_failed=True, failure_reason="LLM provider error")

        # Normalize fields dict -> list if model formatted fields as an object map
        if isinstance(parsed, dict) and isinstance(parsed.get("fields"), dict):
            normalized_fields = []
            for k, v in parsed["fields"].items():
                if isinstance(v, dict):
                    field_dict = {"name": str(k)}
                    for subk, subv in v.items():
                        if subk not in field_dict:
                            field_dict[subk] = subv
                    normalized_fields.append(field_dict)
                elif isinstance(v, str):
                    normalized_fields.append({"name": str(k), "type": v})
                else:
                    normalized_fields.append({"name": str(k)})
            parsed["fields"] = normalized_fields

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

    @staticmethod
    def _parse_json(raw: str | None) -> Any:
        if not raw or not isinstance(raw, str):
            raise ValueError(f"Expected JSON string, got {type(raw)}")
        raw = raw.strip()

        # 1. Direct parse attempt
        try:
            return json.loads(raw)
        except Exception:
            pass

        # 2. Markdown code fences
        if "```" in raw:
            blocks = raw.split("```")
            for block in blocks:
                cleaned = block.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except Exception:
                    # try raw_decode on code block
                    start = cleaned.find("{")
                    if start != -1:
                        try:
                            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                            return obj
                        except Exception:
                            pass

        # 3. Iterative raw_decode across the text to find the schema dictionary
        start = 0
        decoder = json.JSONDecoder()
        candidates = []
        while start < len(raw):
            idx = raw.find("{", start)
            if idx == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(raw[idx:])
                if isinstance(obj, dict):
                    # prioritize the object that contains schema keys
                    if "document_type" in obj or "fields" in obj:
                        return obj
                    candidates.append(obj)
                start = idx + max(1, end_idx)
            except Exception:
                start = idx + 1

        if candidates:
            return candidates[0]

        # 4. Fallback slice
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])

        raise ValueError(f"Could not parse valid JSON from model output: {raw[:200]}")

    # ---- low-level HTTP ----

    def _chat(
        self,
        system: str,
        user: str,
        response_format: Dict[str, Any] | None = None,
        reasoning_effort: str | None = "low",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "api-subscription-key": self.api_key or "",
                "Content-Type": "application/json",
            },
            timeout=max(self.timeout_s, 180.0),
        )
        if not resp.is_success:
            logger.error(
                "Sarvam API error HTTP %d: %s | Model: %s",
                resp.status_code,
                resp.text,
                self.model,
            )
            resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens")

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Sarvam API returned no choices: {data}")
        choice = choices[0]
        finish_reason = (
            choice.get("finish_reason")
            if isinstance(choice, dict)
            else getattr(choice, "finish_reason", None)
        )
        if finish_reason == "length":
            raise TruncatedCompletionError(
                f"Sarvam completion was truncated by max_tokens (finish_reason='{finish_reason}', max_tokens={max_tokens})",
                finish_reason=finish_reason,
            )

        msg = (
            choice.get("message", {})
            if isinstance(choice, dict)
            else getattr(choice, "message", {})
        )
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        ) or ""
        reasoning = (
            msg.get("reasoning_content")
            if isinstance(msg, dict)
            else getattr(msg, "reasoning_content", None)
        ) or ""

        out_text = content
        if not content:
            out_text = reasoning
        elif "{" in reasoning and "{" not in content:
            out_text = reasoning

        return ChatResult(
            out_text,
            reasoning_content=reasoning,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
