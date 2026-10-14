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


def _is_reasoning_model(model_id: str) -> bool:
    """True if the model id matches any configured reasoning-model marker.
    Replaces the hardcoded 'gpt-oss'/'qdtz23c8eis1' conditional (audit #1, 2a)."""
    markers = getattr(settings, "bedrock_reasoning_model_markers", None) or ["gpt-oss", "qdtz23c8eis1"]
    mid = (model_id or "").lower()
    return any(str(mk).lower() in mid for mk in markers)


_THROTTLE_MARKERS = (
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "ServiceUnavailable", "InternalServerException", "ModelTimeoutException",
    "ModelNotReadyException",
)


def _is_throttling_or_transient(exc: Exception) -> bool:
    """True for throttling / transient service errors that warrant a backoff retry
    (audit #1, 2e). Matches on the exception class name and, for botocore
    ClientError, the response error Code. Metadata only."""
    name = type(exc).__name__
    if any(mk in name for mk in _THROTTLE_MARKERS):
        return True
    # botocore ClientError carries the service error code in .response
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
        if any(mk in code for mk in _THROTTLE_MARKERS):
            return True
    return False


def _parse_failure_category(raw: str) -> str:
    """Classify a parse failure using METADATA-ONLY features (audit #1, 2d).
    Returns a short category label — NEVER any response content. Mirrors the
    Step-1 diagnosis categories so logs stay useful without leaking text."""
    if raw is None:
        return "none"
    s = raw.strip()
    if not s:
        return "empty"
    if s.lstrip().startswith("```"):
        return "fenced"
    first, last = s.find("{"), s.rfind("}")
    if first == -1 and last == -1:
        return "no_json_object"
    if s.count("{") != s.count("}"):
        return "unbalanced_braces"
    if first == -1 or last == -1:
        return "no_json_object"
    if first > 0 or last < len(s) - 1:
        return "surrounding_text"
    return "invalid_json_balanced"   # Step-1 signature: balanced but invalid


def _parse_graph_json(raw: str):
    """Parse a graph-extraction JSON payload (audit #1, 2c).

    Returns (parsed_dict | None, used_salvage: bool):
      - direct json.loads success        -> (dict, False)   [clean]
      - needed the substring-{...} salvage -> (dict, True)   [partial]
      - unparseable even after salvage    -> (None, False)   [invalid -> re-ask]
    The salvage recovers deterministic trailing/leading noise; it does NOT repair
    the model-side invalid JSON (unescaped inner quotes) found in Step 1 — those
    fall through to None so the caller can issue a single re-ask."""
    try:
        parsed = json.loads(raw)
        return (parsed if isinstance(parsed, dict) else {}), False
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass
        return None, False


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
        _purpose: str = "other",
        _owner: Optional[str] = None,
        _doc_id: Optional[str] = None,
        _page_number: Optional[int] = None,
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
        if self.reasoning_effort and _is_reasoning_model(self.model_id):
            kwargs["additionalModelRequestFields"] = {"reasoning_effort": self.reasoning_effort}

        # ---- MEASUREMENT ONLY (audit finding #1) -------------------------------
        # Record Converse metadata (stopReason / usage / latency / outcome). This is
        # a pure observer: the response is returned UNCHANGED and behavior is identical
        # whether or not recording succeeds. Never reads prompt/response content.
        import time as _time
        import random as _random
        from src.adapters.llm import llm_metrics as _m

        def _invoke() -> Dict[str, Any]:
            # Existing behavior: drop additionalModelRequestFields once and retry.
            try:
                return self.client.converse(**kwargs)
            except Exception as exc:
                if "additionalModelRequestFields" in kwargs and not _is_throttling_or_transient(exc):
                    kwargs.pop("additionalModelRequestFields", None)
                    return self.client.converse(**kwargs)
                raise

        # 2e: bounded exponential backoff + jitter for throttling / transient errors.
        max_retries = int(getattr(settings, "bedrock_throttle_max_retries", 4))
        base_delay = float(getattr(settings, "bedrock_throttle_base_delay_s", 0.5))
        max_delay = float(getattr(settings, "bedrock_throttle_max_delay_s", 8.0))

        _t0 = _time.perf_counter()
        retry_attempts = 0
        while True:
            try:
                resp = _invoke()
                break
            except Exception as exc:
                transient = _is_throttling_or_transient(exc)
                if transient and retry_attempts < max_retries:
                    delay = min(max_delay, base_delay * (2 ** retry_attempts))
                    delay = delay * (0.5 + _random.random())  # full-ish jitter
                    logger.warning("bedrock.converse.throttled_retry",
                                   purpose=_purpose, attempt=retry_attempts + 1,
                                   exception_type=type(exc).__name__)
                    _time.sleep(delay)
                    retry_attempts += 1
                    continue
                # Exhausted or non-transient: record exception (with attempts) and raise.
                try:
                    _m.record_converse(
                        purpose=_purpose, model_id=self.model_id, max_tokens=max_tokens,
                        response=None, latency_ms=(_time.perf_counter() - _t0) * 1000.0,
                        outcome=_m.OUTCOME_EXCEPTION, owner=_owner, doc_id=_doc_id,
                        page_number=_page_number, exception_type=type(exc).__name__,
                        retry_attempts=retry_attempts,
                    )
                except Exception:
                    pass
                raise
        try:
            _m.record_converse(
                purpose=_purpose, model_id=self.model_id, max_tokens=max_tokens,
                response=resp, latency_ms=(_time.perf_counter() - _t0) * 1000.0,
                outcome=_m.OUTCOME_OK, owner=_owner, doc_id=_doc_id, page_number=_page_number,
                retry_attempts=retry_attempts,
            )
        except Exception:
            pass
        return resp

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
            logger.warning("bedrock.extract.parse_failed",
                           raw_len=len(raw), category=_parse_failure_category(raw))
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
            logger.warning("bedrock.navigate.parse_failed",
                           raw_len=len(raw), category=_parse_failure_category(raw))
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

        from src.adapters.llm import llm_metrics as _m
        # Audit #1 (2a): the graph-extraction output budget comes from settings
        # (default 8000). Audit #1 (2b): if the call stops on max_tokens, retry
        # ONCE at the ceiling budget; if it still truncates, the page is degraded.
        graph_budget = int(getattr(settings, "graph_extraction_max_tokens", 8000))
        ceiling = int(getattr(settings, "graph_extraction_max_tokens_ceiling", 12000))
        base_messages = [{"role": "user", "content": [{"text": prompt}]}]
        base_system = "You extract entities and relationships for a Document Knowledge Graph. Return ONLY valid JSON."

        resp = self._call_converse(
            messages=base_messages,
            system=base_system,
            temperature=params.get("temperature", 0.0),
            max_tokens=graph_budget,
            _purpose=_m.PURPOSE_GRAPH_EXTRACTION,
            _page_number=page_number,
        )

        # 2b: one larger-budget retry on truncation.
        truncated = _m.stop_reason_from_response(resp) == "max_tokens"
        if truncated and ceiling > graph_budget:
            resp = self._call_converse(
                messages=base_messages,
                system=base_system,
                temperature=params.get("temperature", 0.0),
                max_tokens=ceiling,
                _purpose=_m.PURPOSE_GRAPH_EXTRACTION,
                _page_number=page_number,
            )
            _m.amend_retry_attempts(_m.PURPOSE_GRAPH_EXTRACTION, 1, page_number=page_number)
            truncated = _m.stop_reason_from_response(resp) == "max_tokens"

        raw = _strip_fences(_extract_text_from_converse_response(resp))

        def _finish(entities, relationships, refs, quality):
            # 2b/2c: a degraded page produced truncated/invalid output; a page with
            # no entities is tagged empty_result; otherwise the metrics 'ok' stands.
            if quality == _m.PAGE_QUALITY_DEGRADED:
                _m.amend_outcome(_m.PURPOSE_GRAPH_EXTRACTION, _m.OUTCOME_PARSE_FAILED, page_number=page_number)
            elif not entities:
                _m.amend_outcome(_m.PURPOSE_GRAPH_EXTRACTION, _m.OUTCOME_EMPTY_RESULT, page_number=page_number)
            _m.record_page_quality(quality, page_number=page_number)
            return {
                "entities": entities,
                "relationships": relationships,
                "reference_resolutions": refs,
                "quality": quality,
            }

        # If still truncated after the retry, the page is degraded regardless of
        # whether the fragment happens to parse — we cannot trust a cut-off object.
        if truncated:
            logger.warning("bedrock.extract_graph.truncated_after_retry",
                           page_number=page_number, out_len=len(raw))
            return _finish([], [], [], _m.PAGE_QUALITY_DEGRADED)

        # 2c: parse. Clean = direct parse. Partial = needed the deterministic
        # substring salvage. Invalid = unparseable even after salvage -> one re-ask.
        parsed, used_salvage = _parse_graph_json(raw)
        if parsed is not None:
            quality = _m.PAGE_QUALITY_PARTIAL if used_salvage else _m.PAGE_QUALITY_CLEAN
            if used_salvage:
                logger.warning("bedrock.extract_graph.salvaged_partial", page_number=page_number)
            return _finish(parsed.get("entities", []), parsed.get("relationships", []),
                           parsed.get("reference_resolutions", []), quality)

        # Invalid JSON after salvage. Tag the first call parse_failed, then do ONE
        # bounded re-ask: same prompt + a short GENERIC instruction (no document
        # content) that the previous reply was not valid JSON. The re-ask goes
        # through _call_converse under a distinct purpose so its tokens/cost are
        # recorded; its recovered/still_invalid outcome is captured.
        _m.amend_outcome(_m.PURPOSE_GRAPH_EXTRACTION, _m.OUTCOME_PARSE_FAILED, page_number=page_number)
        logger.warning("bedrock.extract_graph.parse_failed",
                       page_number=page_number, raw_len=len(raw),
                       category=_parse_failure_category(raw))

        reask_messages = base_messages + [
            {"role": "assistant", "content": [{"text": "(previous reply omitted)"}]},
            {"role": "user", "content": [{"text":
                "Your previous reply was not valid JSON. Reply with ONE valid JSON "
                "object only, matching the structure requested above. No prose, no "
                "code fences, and make sure every string value is properly escaped."}]},
        ]
        reask_resp = self._call_converse(
            messages=reask_messages,
            system=base_system,
            temperature=params.get("temperature", 0.0),
            max_tokens=ceiling,
            _purpose=_m.PURPOSE_GRAPH_REASK,
            _page_number=page_number,
        )
        reask_raw = _strip_fences(_extract_text_from_converse_response(reask_resp))
        reask_parsed, reask_salvage = _parse_graph_json(reask_raw)
        if reask_parsed is not None and _m.stop_reason_from_response(reask_resp) != "max_tokens":
            # re-ask recovered: the reask record stays OUTCOME_OK (counts as recovered).
            quality = _m.PAGE_QUALITY_PARTIAL if reask_salvage else _m.PAGE_QUALITY_CLEAN
            return _finish(reask_parsed.get("entities", []), reask_parsed.get("relationships", []),
                           reask_parsed.get("reference_resolutions", []), quality)

        # Still invalid after the single re-ask -> degraded.
        _m.amend_outcome(_m.PURPOSE_GRAPH_REASK, _m.OUTCOME_PARSE_FAILED, page_number=page_number)
        logger.warning("bedrock.extract_graph.reask_still_invalid",
                       page_number=page_number, raw_len=len(reask_raw))
        return _finish([], [], [], _m.PAGE_QUALITY_DEGRADED)

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

        from src.adapters.llm import llm_metrics as _m
        base_messages = [{"role": "user", "content": [{"text": prompt}]}]
        base_system = "You extract structured schema fields using a Document Knowledge Graph. Return ONLY valid JSON matching the schema."
        resolver_budget = max(params.get("max_tokens", 4000), 4000)
        resp = self._call_converse(
            messages=base_messages,
            system=base_system,
            temperature=params.get("temperature", 0.0),
            max_tokens=resolver_budget,
            _purpose=_m.PURPOSE_RESOLVER,
        )
        raw = _strip_fences(_extract_text_from_converse_response(resp))

        def _validate(obj) -> Optional[BaseModel]:
            try:
                return schema.model_validate(obj)
            except Exception:
                return None

        parsed, _salvage = _parse_graph_json(raw)
        if parsed is not None:
            model = _validate(parsed)
            if model is not None:
                return model

        # 2c: invalid JSON (or schema-invalid) -> one bounded re-ask, generic
        # instruction only, no document content in the added message.
        _m.amend_outcome(_m.PURPOSE_RESOLVER, _m.OUTCOME_PARSE_FAILED)
        logger.warning("bedrock.resolve_schema.parse_failed", raw_len=len(raw))
        reask_messages = base_messages + [
            {"role": "assistant", "content": [{"text": "(previous reply omitted)"}]},
            {"role": "user", "content": [{"text":
                "Your previous reply was not valid JSON. Reply with ONE valid JSON "
                "object only, matching the requested schema field names as keys. No "
                "prose, no code fences, and properly escape every string value."}]},
        ]
        reask_resp = self._call_converse(
            messages=reask_messages,
            system=base_system,
            temperature=params.get("temperature", 0.0),
            max_tokens=resolver_budget,
            _purpose=_m.PURPOSE_RESOLVER_REASK,
        )
        reask_raw = _strip_fences(_extract_text_from_converse_response(reask_resp))
        reask_parsed, _ = _parse_graph_json(reask_raw)
        if reask_parsed is not None:
            model = _validate(reask_parsed)
            if model is not None:
                return model
        _m.amend_outcome(_m.PURPOSE_RESOLVER_REASK, _m.OUTCOME_PARSE_FAILED)
        logger.warning("bedrock.resolve_schema.reask_still_invalid", raw_len=len(reask_raw))
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
