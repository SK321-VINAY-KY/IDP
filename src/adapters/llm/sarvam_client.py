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
from typing import Any, Dict, List, Optional, Union
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


import datetime
import time
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_current_run_log_file: Optional[Path] = None


def set_run_log_file(path: Union[str, Path]) -> Path:
    global _current_run_log_file
    _current_run_log_file = Path(path)
    _current_run_log_file.parent.mkdir(parents=True, exist_ok=True)
    return _current_run_log_file


def get_run_log_file() -> Path:
    global _current_run_log_file
    if _current_run_log_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _current_run_log_file = LOGS_DIR / f"layer3_run_{ts}.log"
    return _current_run_log_file


def log_sarvam_request(phase: str, prompt: str, response: str, model: str, latency_s: float) -> None:
    log_file = get_run_log_file()
    ts_str = datetime.datetime.now().isoformat()
    entry = (
        f"\n{'=' * 80}\n"
        f"TIMESTAMP: {ts_str}\n"
        f"PHASE: {phase}\n"
        f"MODEL: {model} | LATENCY: {latency_s:.2f}s\n"
        f"{'-' * 35} PROMPT SENT {'-' * 35}\n"
        f"{prompt.strip()}\n"
        f"{'-' * 35} RESPONSE RECEIVED {'-' * 35}\n"
        f"{response.strip()}\n"
        f"{'=' * 80}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()



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
        self.reasoning_effort = getattr(settings, "sarvam_reasoning_effort", None) or os.getenv("IDP_SARVAM_REASONING_EFFORT") or os.getenv("SARVAM_REASONING_EFFORT", None)

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
            extra_body={"reasoning_effort": self.reasoning_effort},
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
                    extra_body={"reasoning_effort": self.reasoning_effort},
                )
                choice = raw_resp.choices[0]
                text = choice.message.content or ""
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
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
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
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
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
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
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

    def summarize_segment(
        self,
        segment_text: str,
        page_range: List[int],
    ) -> Dict[str, str]:
        """
        Summarize a multi-page document segment to extract doc_type_hint and one_line_summary.
        """
        prompt = render_prompt(
            "segment_summary",
            segment_text=segment_text[:4000],
            page_range=page_range,
        )
        params = prompt_params("segment_summary")
        t0 = time.time()

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        latency = time.time() - t0
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if not raw and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            raw = choice.message.reasoning_content
        log_sarvam_request("segmentation summary", prompt, raw, self.model, latency)

        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {
                    "doc_type_hint": str(parsed.get("doc_type_hint", "unknown")),
                    "one_line_summary": str(parsed.get("one_line_summary", "")),
                }
        except Exception:
            logger.warning("sarvam.summarize_segment.parse_failed", raw=raw[:200])
        return {"doc_type_hint": "unknown", "one_line_summary": raw.strip()[:200]}

    def navigate_category(
        self,
        segment_summaries: List[Dict[str, Any]],
        category: str,
        category_description: str,
        category_examples: Optional[List[str]] = None,
        extraction_focus: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Identify segment_ids that plausibly contain mentions of the specified ontology category.
        """
        prompt = render_prompt(
            "category_navigation",
            segments=segment_summaries,
            category=category,
            category_description=category_description,
            category_examples=category_examples or [],
            extraction_focus=extraction_focus,
        )
        params = prompt_params("category_navigation")
        t0 = time.time()

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        latency = time.time() - t0
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if not raw and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            raw = choice.message.reasoning_content
        log_sarvam_request("navigation", prompt, raw, self.model, latency)

        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "segment_ids" in parsed:
                return [str(s) for s in parsed["segment_ids"]]
            if isinstance(parsed, list):
                return [str(s) for s in parsed]
        except Exception:
            # Try regex fallback for segment IDs (e.g. seg_01, seg_04)
            seg_matches = re.findall(r"seg_\d+", raw)
            if seg_matches:
                return list(dict.fromkeys(seg_matches))
            logger.warning("sarvam.navigate_category.parse_failed", raw=raw[:200])
        return []

    def extract_page_ontology(
        self,
        page_md: str,
        page_number: int,
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Phase C: Extract candidate ontology nodes and edges from a single page using LLM.
        """
        prompt = render_prompt(
            "page_ontology_extraction",
            page_md=page_md[:6000],
            page_number=page_number,
            extraction_focus=extraction_focus,
        )
        params = prompt_params("page_ontology_extraction")
        t0 = time.time()

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 3000),
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        latency = time.time() - t0
        choice = resp.choices[0]
        raw = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        if not raw and reasoning:
            raw = str(reasoning)
        log_sarvam_request(f"page_ontology_extraction p{page_number}", prompt, raw, self.model, latency)

        return self._parse_page_ontology_json(raw, page_number)

    def _parse_page_ontology_json(self, raw: str, page_number: int = 0) -> Dict[str, Any]:
        """
        Parse page ontology extraction JSON with resilience against model output truncation.
        """
        cleaned = _strip_fences(raw)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return {
                    "nodes": parsed.get("nodes", []),
                    "edges": parsed.get("edges", []),
                }
        except Exception:
            pass

        # Recovery Strategy 1: Truncation repair by cutting back to last complete '}'
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            truncated = cleaned[:last_brace + 1]
            for closing in ["]}", "\n  ]\n}", "}\n}", "\n}"]:
                try:
                    candidate = truncated + closing
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        nodes = parsed.get("nodes", [])
                        edges = parsed.get("edges", [])
                        if nodes or edges:
                            logger.info(
                                "sarvam.extract_page_ontology.recovered_truncated",
                                page=page_number,
                                recovered_nodes=len(nodes),
                                recovered_edges=len(edges),
                            )
                            return {"nodes": nodes, "edges": edges}
                except Exception:
                    pass

        # Recovery Strategy 2: Individual JSON object regex extraction
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        nodes_part = cleaned.split('"nodes"')[1] if '"nodes"' in cleaned else ""
        edges_part = nodes_part.split('"edges"')[1] if '"edges"' in nodes_part else ""
        if '"edges"' in nodes_part:
            nodes_part = nodes_part.split('"edges"')[0]

        for m in re.finditer(r'\{[^{}]*"category"[^{}]*\}', nodes_part, re.DOTALL):
            try:
                node = json.loads(m.group(0))
                if isinstance(node, dict) and "category" in node and "label" in node:
                    nodes.append(node)
            except Exception:
                pass

        for m in re.finditer(r'\{[^{}]*"source_label"[^{}]*\}', edges_part, re.DOTALL):
            try:
                edge = json.loads(m.group(0))
                if isinstance(edge, dict) and "source_label" in edge and "target_label" in edge:
                    edges.append(edge)
            except Exception:
                pass

        if nodes or edges:
            logger.info(
                "sarvam.extract_page_ontology.recovered_regex",
                page=page_number,
                recovered_nodes=len(nodes),
                recovered_edges=len(edges),
            )
            return {"nodes": nodes, "edges": edges}

        logger.warning("sarvam.extract_page_ontology.parse_failed", page=page_number, raw=cleaned[:200])
        return {"nodes": [], "edges": []}

    def extract_key_findings(
        self,
        segments: List[Dict[str, Any]],
        candidate_nodes: List[Dict[str, Any]],
        extraction_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize document-level key findings from segments and candidate nodes.
        """
        prompt = render_prompt(
            "document_key_findings",
            segments=segments,
            candidate_nodes=candidate_nodes,
            extraction_focus=extraction_focus,
        )
        params = prompt_params("document_key_findings")
        t0 = time.time()

        resp = self.raw_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 3000),
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        latency = time.time() - t0
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if not raw and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            raw = choice.message.reasoning_content
        log_sarvam_request("document_key_findings", prompt, raw, self.model, latency)

        raw = _strip_fences(raw)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {"key_findings": parsed.get("key_findings", [])}
        except Exception:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        return {"key_findings": parsed.get("key_findings", [])}
                except Exception:
                    pass
            logger.warning("sarvam.extract_key_findings.parse_failed", raw=raw[:200])
        return {"key_findings": []}


