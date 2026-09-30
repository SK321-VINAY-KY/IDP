"""
File: bedrock_client.py
Purpose: Amazon Bedrock implementation of ExtractionLLMClient for Layer 3 extraction.
Supports Application Inference Profiles and Foundation Models (including OpenAI GPT OSS 120B).
Owner: engineer-b@idp-pilot
Created: 2026-09-29 | Deps: boto3, pydantic
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, VLMAnalysis
from src.config.settings import settings
from src.ai.layer3_extraction.prompts.loader import render_prompt, prompt_params
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import boto3
except ImportError:
    boto3: Any = None


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences (```json ... ```) before JSON parsing."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _extract_text_from_converse_response(response: Dict[str, Any]) -> str:
    """Safely extract plain text from Bedrock Converse API response, skipping reasoningContent."""
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    for block in content:
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return ""


class BedrockExtractionClient:
    """
    Amazon Bedrock extraction client using Bedrock Runtime Converse API.
    Fully compatible with OpenAI GPT OSS 120B (and other Bedrock models),
    handling reasoningContent blocks, toolUse, and robust JSON schema recovery.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        if boto3 is None:
            raise ImportError("pip install boto3")

        self.region = (
            region
            or getattr(settings, "bedrock_region", None)
            or os.getenv("IDP_BEDROCK_REGION")
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )
        self.model_id = (
            model_id
            or getattr(settings, "bedrock_model_id", None)
            or os.getenv("IDP_BEDROCK_MODEL_ID")
            or os.getenv("BEDROCK_MODEL_ID")
            or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
        )
        self.reasoning_effort = (
            reasoning_effort
            or getattr(settings, "bedrock_reasoning_effort", None)
            or os.getenv("IDP_BEDROCK_REASONING_EFFORT")
            or os.getenv("BEDROCK_REASONING_EFFORT")
            or "low"
        )
        self._client: Any = None

    @property
    def client(self) -> Any:
        if boto3 is None:
            raise ImportError("pip install boto3")
        if self._client is None:
            # Check for AWS profile in development or use default IAM role credentials in ECS/Lambda
            profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=self.region)
                self._client = session.client("bedrock-runtime", region_name=self.region)
            else:
                self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _call_converse(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tool_config:
            kwargs["toolConfig"] = tool_config
        if self.reasoning_effort and ("gpt-oss" in self.model_id.lower() or "qdtz23c8eis1" in self.model_id):
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": self.reasoning_effort}

        try:
            return self.client.converse(**kwargs)
        except Exception as exc:
            # If additionalModelRequestFields failed, retry without it
            if "additionalModelRequestFields" in kwargs:
                kwargs.pop("additionalModelRequestFields", None)
                return self.client.converse(**kwargs)
            raise exc

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        """
        Extract structured Pydantic schema from content using Bedrock LLM.
        """
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        system_prompt = (
            render_prompt("extraction", schema_fields=schema_fields)
            + "\nCRITICAL: Return ONLY valid JSON matching the requested schema fields. No conversational prose."
        )
        params = prompt_params("extraction")
        logger.info("bedrock.extract.request", model=self.model_id, content_chars=len(content))

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": content}]}],
            system=system_prompt,
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 4000), 4000),
        )

        raw = _extract_text_from_converse_response(resp)
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            return schema.model_validate(parsed)
        except Exception:
            # Substring JSON search
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    return schema.model_validate(parsed)
                except Exception:
                    pass
            logger.warning("bedrock.extract.parse_failed", raw=raw[:200])
            # Return empty schema instance rather than crashing
            return schema.model_validate({})

    def summarize_page(self, page_md: str, max_words: int = 150) -> str:
        """
        Summarize a single document page in max_words or fewer.
        """
        prompt = render_prompt("page_summary", page_md=page_md, max_words=max_words)
        params = prompt_params("page_summary")

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 1000), 2000),
        )
        raw = _extract_text_from_converse_response(resp)
        return raw.strip()

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

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system="Return ONLY valid JSON mapping schema fields to integer page lists.",
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 1000), 2000),
        )
        raw = _extract_text_from_converse_response(resp)
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): list(v) if isinstance(v, list) else [] for k, v in parsed.items()}
            return {f: [] for f in schema_fields}
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    if isinstance(parsed, dict):
                        return {str(k): list(v) if isinstance(v, list) else [] for k, v in parsed.items()}
                except Exception:
                    pass
            logger.warning("bedrock.navigate.parse_failed", raw=raw[:200])
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

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system="Return ONLY valid JSON array or object with extracted fields and values.",
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 4000), 4000),
        )
        raw = _extract_text_from_converse_response(resp)
        if not raw:
            return []

        raw = _strip_fences(raw)
        matches: List[Any] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if "matches" in parsed and isinstance(parsed["matches"], list):
                    matches = parsed["matches"]
                else:
                    matches = [{"field": k, "value": str(v)} for k, v in parsed.items() if k in valid_names]
            elif isinstance(parsed, list):
                matches = parsed
        except json.JSONDecodeError:
            # Fallback regex extraction
            pattern = re.compile(
                r'\{\s*"field"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"',
                re.DOTALL,
            )
            for m in pattern.finditer(raw):
                f_name = m.group(1)
                val = m.group(2).encode().decode("unicode_escape", errors="ignore")
                matches.append({"field": f_name, "value": val})

        result = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            field = m.get("field", "")
            value = m.get("value", "")
            if field in valid_names and value not in ("", None):
                result.append({"field": str(field), "value": str(value)})
        return result

    def extract_graph_from_page(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        existing_nodes: List[Dict[str, Any]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract entities, contextual relationships, and reference resolutions from a page.
        """
        lines = []
        for n in existing_nodes[:35]:
            pages_str = ",".join(map(str, n.get("source_pages", [])))
            lines.append(f"- [{n.get('id')}] {n.get('type')}: \"{n.get('value')}\" (P{pages_str})")
        existing_context = "\n".join(lines) if lines else "No existing entities in graph memory."

        prompt = render_prompt(
            "graph_page_ingestion",
            page_number=page_number,
            total_pages=total_pages,
            page_md=page_md,
            schema_fields=schema_fields,
            existing_graph_context=existing_context,
        )
        params = prompt_params("graph_page_ingestion")

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system="You extract entities and relationships for a Document Knowledge Graph. Return ONLY valid JSON.",
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 4000), 4000),
        )
        raw = _extract_text_from_converse_response(resp)
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {
                    "entities": parsed.get("entities", []),
                    "relationships": parsed.get("relationships", []),
                    "reference_resolutions": parsed.get("reference_resolutions", []),
                }
            return {"entities": [], "relationships": [], "reference_resolutions": []}
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    if isinstance(parsed, dict):
                        return {
                            "entities": parsed.get("entities", []),
                            "relationships": parsed.get("relationships", []),
                            "reference_resolutions": parsed.get("reference_resolutions", []),
                        }
                except Exception:
                    pass
            logger.warning("bedrock.extract_graph.parse_failed", raw=raw[:200])
            return {"entities": [], "relationships": [], "reference_resolutions": []}

    def resolve_schema_from_graph(
        self,
        graph_evidence: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """
        Synthesize final target schema fields using structured Graph Evidence.
        """
        schema_fields = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        prompt = render_prompt(
            "graph_schema_resolution",
            schema_fields=schema_fields,
            graph_evidence=graph_evidence,
        )
        params = prompt_params("graph_schema_resolution")

        resp = self._call_converse(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system="You extract structured schema fields using a Document Knowledge Graph. Return ONLY valid JSON matching the schema.",
            temperature=params.get("temperature", 0.0),
            max_tokens=max(params.get("max_tokens", 4000), 4000),
        )
        raw = _extract_text_from_converse_response(resp)
        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            return schema.model_validate(parsed)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    return schema.model_validate(parsed)
                except Exception:
                    pass
            logger.warning("bedrock.resolve_schema.parse_failed", raw=raw[:200])
            return schema.model_validate({})


class BedrockLLMClient(LLMClient):
    """
    Amazon Bedrock multimodal implementation of LLMClient for Layer 1 page inspection,
    classification, and handwriting transcription.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        if boto3 is None:
            raise ImportError("pip install boto3")

        self.region = (
            region
            or getattr(settings, "bedrock_region", None)
            or os.getenv("IDP_BEDROCK_REGION")
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )
        self.model_id = (
            model_id
            or getattr(settings, "bedrock_vlm_model_id", None)
            or os.getenv("IDP_BEDROCK_VLM_MODEL_ID")
            or os.getenv("BEDROCK_VLM_MODEL_ID")
            or getattr(settings, "vlm_model_name", None)
            or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/lmbukv3mwnhm"
        )
        self.reasoning_effort = (
            reasoning_effort
            or getattr(settings, "bedrock_reasoning_effort", None)
            or os.getenv("IDP_BEDROCK_REASONING_EFFORT")
            or os.getenv("BEDROCK_REASONING_EFFORT")
            or "low"
        )
        profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
        if profile:
            session = boto3.Session(profile_name=profile, region_name=self.region)
            self._client = session.client("bedrock-runtime", region_name=self.region)
        else:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def _call_converse_vision(self, prompt: str, image_bytes: bytes, max_tokens: int = 1500) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        ]
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {"temperature": 0.0, "maxTokens": max_tokens},
        }
        if self.reasoning_effort and ("gpt-oss" in self.model_id.lower() or "qdtz23c8eis1" in self.model_id):
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": self.reasoning_effort}

        try:
            resp = self._client.converse(**kwargs)
            return _extract_text_from_converse_response(resp)
        except Exception:
            kwargs.pop("additionalModelRequestFields", None)
            resp = self._client.converse(**kwargs)
            return _extract_text_from_converse_response(resp)

    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        prompt = (
            "Classify this document page image. Return a JSON object with: "
            "'route' ('digital', 'scanned', 'handwritten', or 'skip'), "
            "'confidence' (float 0.0-1.0), "
            "'language_hint' (string, e.g. 'en'), "
            "'handwriting_pct' (float 0.0-1.0 estimating proportion of handwritten content), "
            "'noise_level' (float 0.0-1.0), "
            "'needs_preprocessing' (list of strings). "
            f"\nHints from page profile: {page_profile_hint}"
        )
        try:
            raw = self._call_converse_vision(prompt, image_bytes, max_tokens=1000)
            raw = _strip_fences(raw)
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            parsed = json.loads(raw)
            return PageClassification.model_validate(parsed)
        except Exception as exc:
            logger.warning("bedrock.classify_page.vision_fallback", error=str(exc))
            is_scanned = bool(page_profile_hint.get("is_scanned", False))
            char_count = int(page_profile_hint.get("char_count", 0))
            hw_pct = 0.5 if is_scanned and char_count < 50 else 0.0
            route = "handwritten" if hw_pct > 0.3 else ("scanned" if is_scanned else "digital")
            return PageClassification(
                route=route,
                confidence=0.85,
                language_hint="en",
                handwriting_pct=hw_pct,
                noise_level=0.1,
                needs_preprocessing=[],
            )

    def analyze_page(self, image_bytes: bytes, page_profile_hint: dict) -> VLMAnalysis:
        prompt = (
            "Analyze this document page image for OCR and extraction planning. Return a JSON object with: "
            "'can_extract_directly' (boolean, true only if page has legible printed/digital text and can be fully transcribed directly), "
            "'confidence' (float 0.0-1.0), "
            "'detected_capabilities' (list of strings from: 'ocr', 'handwriting', 'table', 'figure'), "
            "'required_capabilities' (list of strings from: 'ocr', 'handwriting', 'table'), "
            "'reason' (short explanation), "
            "'extracted_markdown' (string, full page transcription in markdown if can_extract_directly is true, else empty string). "
            f"\nHints from page profile: {page_profile_hint}"
        )
        try:
            raw = self._call_converse_vision(prompt, image_bytes, max_tokens=2048)
            raw = _strip_fences(raw)
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start : end + 1])
                return VLMAnalysis.model_validate(parsed)
        except Exception as exc:
            logger.warning("bedrock.analyze_page.vision_fallback", error=str(exc))

        classification = self.classify_page(image_bytes, page_profile_hint)
        capabilities = {"handwriting"} if classification.handwriting_pct > 0.10 else {"ocr"}
        return VLMAnalysis(
            confidence=classification.confidence,
            detected_capabilities=capabilities,
            required_capabilities=capabilities,
            reason="bedrock classification; specialized processing plan",
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        prompt = (
            "You are an expert transcription model. Transcribe all visible text, handwritten notes, "
            "and tabular data in this image into accurate Markdown. Output ONLY the transcribed Markdown."
        )
        try:
            text = self._call_converse_vision(prompt, image_bytes, max_tokens=2048)
            conf = 0.92 if len(text.strip()) > 20 else 0.40
            return text.strip(), conf
        except Exception as exc:
            logger.warning("bedrock.transcribe_handwriting.failed", error=str(exc))
            return "", 0.0
