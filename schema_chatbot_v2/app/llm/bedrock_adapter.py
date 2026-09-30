"""
Bedrock adapter - for the eventual AWS deployment target.

Implements the exact same LLMAdapter interface as OllamaAdapter, using
Bedrock's Converse API with a tool-use (function-calling) definition so the
model is constrained to return the ExtractionResult shape directly, rather
than relying on prompted JSON mode the way the Ollama adapter does.

Not exercised by the test suite here (no AWS credentials in this
environment) - structurally complete and ready to point at a real Bedrock
endpoint by setting LLM_PROVIDER=bedrock. Verify the tool-use response
parsing against a real Bedrock account before relying on it in production.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from app.config import settings
from app.llm.base import ExtractionResult, LLMAdapter, SchemaProposal
from app.llm.prompts import (
    DOCUMENT_INFERENCE_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
    document_inference_user_text,
    fallback_question,
)

logger = logging.getLogger(__name__)

# "operations" (add/update/remove) replaces the old new_fields/field_answers/
# removals split - one reply can add, correct, and remove fields together.
# "reply" lets the model compose the actual response text.
_EXTRACTION_TOOL = {
    "toolSpec": {
        "name": "record_extraction",
        "description": "Record what was extracted from the user's message.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "document_type": {"type": ["string", "null"]},
                    "operations": {"type": "array", "items": {"type": "object"}},
                    "confirmation": {"type": ["boolean", "null"]},
                    "reply": {"type": ["string", "null"]},
                    "needs_clarification": {"type": "boolean"},
                    "clarification_reason": {"type": ["string", "null"]},
                },
                "required": ["needs_clarification"],
            }
        },
    }
}

# Same shape as SchemaProposal, for the document-intake tool-use call.
_SCHEMA_PROPOSAL_TOOL = {
    "toolSpec": {
        "name": "propose_schema",
        "description": "Propose a target extraction schema after comparing N sample documents of the same type.",
        "inputSchema": {
            "json": {
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
        },
    }
}


class BedrockAdapter(LLMAdapter):
    def __init__(self, region: str | None = None, model_id: str | None = None):
        self.region = region or settings.bedrock_region
        self.model_id = model_id or settings.bedrock_model_id
        self._client: Any = None  # lazy, so importing this module doesn't require boto3/creds

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # local import: keep boto3 optional unless bedrock is actually used
            profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=self.region)
                self._client = session.client("bedrock-runtime", region_name=self.region)
            else:
                self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _additional_fields(self) -> Dict[str, Any]:
        effort = getattr(settings, "bedrock_reasoning_effort", "low")
        if effort and ("gpt-oss" in self.model_id.lower() or "qdtz23c8eis1" in self.model_id):
            return {"reasoning_effort": effort}
        return {}

    def extract(self, state: str, user_message: str, context: Dict[str, Any]) -> ExtractionResult:
        user_prompt = build_extraction_user_prompt(state, user_message, context)
        try:
            kwargs: Dict[str, Any] = {
                "modelId": self.model_id,
                "system": [{"text": EXTRACTION_SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                "toolConfig": {"tools": [_EXTRACTION_TOOL], "toolChoice": {"tool": {"name": "record_extraction"}}},
                "inferenceConfig": {"temperature": 0.1, "maxTokens": 4000},
            }
            extra = self._additional_fields()
            if extra:
                kwargs["additionalModelRequestFields"] = extra
            try:
                response = self.client.converse(**kwargs)
            except Exception:
                kwargs.pop("additionalModelRequestFields", None)
                response = self.client.converse(**kwargs)

            tool_input = self._extract_tool_input(response)
            return ExtractionResult.model_validate(tool_input)
        except Exception:
            logger.exception("Bedrock extraction call failed")
            return ExtractionResult(extraction_failed=True, needs_clarification=True,
                                     clarification_reason="LLM provider error")

    def phrase_question(self, gap_field: str, gap_attribute: str, context: Dict[str, Any]) -> str:
        template = fallback_question(gap_field, gap_attribute)
        try:
            kwargs: Dict[str, Any] = {
                "modelId": self.model_id,
                "system": [{"text": "Rephrase as one short, friendly question. Reply with ONLY the question."}],
                "messages": [{"role": "user", "content": [{"text": template}]}],
                "inferenceConfig": {"temperature": 0.3, "maxTokens": 2000},
            }
            extra = self._additional_fields()
            if extra:
                kwargs["additionalModelRequestFields"] = extra
            try:
                response = self.client.converse(**kwargs)
            except Exception:
                kwargs.pop("additionalModelRequestFields", None)
                response = self.client.converse(**kwargs)

            content = response.get("output", {}).get("message", {}).get("content", [])
            text = next((block["text"] for block in content if isinstance(block, dict) and "text" in block), "")
            return text.strip() or template
        except Exception:
            logger.warning("Bedrock phrase_question call failed, using template fallback")
            return template

    def infer_schema_from_pdfs(self, samples: List[bytes]) -> SchemaProposal:
        """
        Infers target extraction schema after comparing N sample documents.
        Supports native document blocks for multimodal models and automatic
        text-extraction fallback for text-only models (e.g. OpenAI GPT OSS 120B).
        """
        # Try native document blocks first
        content: List[Dict[str, Any]] = [{"text": document_inference_user_text(len(samples))}]
        for i, pdf_bytes in enumerate(samples, start=1):
            content.append(
                {
                    "document": {
                        "format": "pdf",
                        "name": self._safe_document_name(f"sample-{i}"),
                        "source": {"bytes": pdf_bytes},
                    }
                }
            )

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": DOCUMENT_INFERENCE_SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": content}],
            "toolConfig": {"tools": [_SCHEMA_PROPOSAL_TOOL], "toolChoice": {"tool": {"name": "propose_schema"}}},
            "inferenceConfig": {"temperature": 0.1, "maxTokens": 4000},
        }
        extra = self._additional_fields()
        if extra:
            kwargs["additionalModelRequestFields"] = extra

        try:
            try:
                response = self.client.converse(**kwargs)
            except Exception as exc:
                err_str = str(exc)
                if "doesn't support documents" in err_str or "ValidationException" in type(exc).__name__:
                    # Fallback to text extraction for text-only LLMs
                    logger.info("Bedrock model does not support document blocks; falling back to extracted text")
                    text_content: List[Dict[str, Any]] = [{"text": document_inference_user_text(len(samples))}]
                    for i, pdf_bytes in enumerate(samples, start=1):
                        extracted_text = self._extract_text_from_pdf_bytes(pdf_bytes)
                        text_content.append({"text": f"--- SAMPLE DOCUMENT {i} ---\n{extracted_text}"})
                    kwargs["messages"] = [{"role": "user", "content": text_content}]
                    kwargs.pop("additionalModelRequestFields", None)
                    if extra:
                        kwargs["additionalModelRequestFields"] = extra
                    response = self.client.converse(**kwargs)
                else:
                    raise exc

            tool_input = self._extract_tool_input(response)
            proposal = SchemaProposal.model_validate(tool_input)
        except Exception:
            logger.exception("Bedrock document-schema inference failed")
            return SchemaProposal(extraction_failed=True, failure_reason="LLM provider error")

        for field in proposal.fields:
            field.total_samples = len(samples)
        return proposal

    @staticmethod
    def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            texts: list[str] = [str(page.get_text()) for page in doc]
            return "\n".join(texts)
        except Exception:
            return pdf_bytes.decode("utf-8", errors="replace")[:10000]

    @staticmethod
    def _safe_document_name(name: str) -> str:
        # Converse's document `name` field only allows a limited charset and
        # is technically prompt-injectable, so keep it a neutral, fixed pattern.
        return re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "_", name)

    @staticmethod
    def _extract_tool_input(response: Dict[str, Any]) -> Dict[str, Any]:
        content = response.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                return block["toolUse"].get("input", {})
        # Fallback: check if model returned JSON in a text block
        import json
        for block in content:
            if isinstance(block, dict) and "text" in block:
                raw = block["text"].strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw).strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    try:
                        return json.loads(raw[start : end + 1])
                    except Exception:
                        pass
        raise ValueError("no toolUse or valid JSON block in Bedrock response")

