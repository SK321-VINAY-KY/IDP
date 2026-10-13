"""
File: llm_metrics.py
Purpose: Phase-observability (audit finding #1) — MEASUREMENT-ONLY instrumentation
         for Bedrock Converse calls. Records call metadata (never content) so we can
         see stopReason distribution, max_tokens truncations, parse failures, empty
         results, throttling and output-token distribution per call purpose.

STRICT RULES this module obeys:
  1. METADATA ONLY. It never reads, stores, or logs prompt text, document text, or
     model output text. It touches only: stopReason, usage token counts, latency,
     maxTokens setting, call purpose, outcome enum, owner, doc id, page number.
  2. PASS-THROUGH. record_converse() returns nothing and mutates nothing; callers
     pass the already-received response in and continue using it unchanged. This
     module cannot alter extraction behavior.
  3. TENANT-SCOPED. Any summary file is written under the owner-partitioned path
     dataset_output/_users/<owner>/metrics/. owner is required for file writes.

This is deliberately a small recorder, NOT a gateway/wrapper around Converse — the
task is to measure first, not to refactor the call sites into a single client.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Outcome enums (metadata only — no content).
OUTCOME_OK = "ok"
OUTCOME_PARSE_FAILED = "parse_failed"
OUTCOME_EMPTY_RESULT = "empty_result"
OUTCOME_EXCEPTION = "exception"

# Call-purpose enums for the Layer-3 Bedrock sites.
PURPOSE_GRAPH_EXTRACTION = "graph_extraction"
PURPOSE_RESOLVER = "schema_resolution"
PURPOSE_CHECK_PAGE = "check_page_for_fields"
PURPOSE_NAVIGATE = "navigate"
PURPOSE_SUMMARIZE = "summarize_page"
PURPOSE_EXTRACT = "extract"
PURPOSE_VISION = "vision_transcribe"
PURPOSE_QUERY_ANSWER = "query_answer"
PURPOSE_GROUNDING = "grounding_check"
PURPOSE_OTHER = "other"


def stop_reason_from_response(response: Optional[Dict[str, Any]]) -> Optional[str]:
    """Read stopReason from a Converse response. Metadata only."""
    if not isinstance(response, dict):
        return None
    return response.get("stopReason")


def usage_from_response(response: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Read usage token counts. Handles a missing 'usage' field gracefully."""
    if not isinstance(response, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    usage = response.get("usage") or {}
    return {
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "total_tokens": usage.get("totalTokens"),
    }


@dataclass
class CallRecord:
    """One Converse call's metadata. NO content fields exist on this dataclass."""
    purpose: str
    model_id: str
    stop_reason: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: Optional[float]
    max_tokens: Optional[int]
    outcome: str
    owner: Optional[str] = None
    doc_id: Optional[str] = None
    page_number: Optional[int] = None


def _percentile(sorted_vals: List[int], pct: float) -> Optional[int]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return int(round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac))


class MetricsCollector:
    """In-process, thread-safe accumulator of CallRecords. One per process run; the
    per-document summary is sliced out by (owner, doc_id) when written."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[CallRecord] = []

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._records.append(rec)
        # Structured, metadata-only log line per call.
        logger.info(
            "llm.converse.call",
            purpose=rec.purpose,
            model_id=rec.model_id,
            stop_reason=rec.stop_reason,
            input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens,
            latency_ms=rec.latency_ms,
            max_tokens=rec.max_tokens,
            outcome=rec.outcome,
            owner=rec.owner,
            doc_id=rec.doc_id,
            page_number=rec.page_number,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def records_for(self, owner: Optional[str] = None, doc_id: Optional[str] = None) -> List[CallRecord]:
        with self._lock:
            recs = list(self._records)
        if owner is not None:
            recs = [r for r in recs if r.owner == owner]
        if doc_id is not None:
            recs = [r for r in recs if r.doc_id == doc_id]
        return recs

    def summarize(self, owner: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Build the per-document (or filtered) metadata summary. No content."""
        recs = self.records_for(owner=owner, doc_id=doc_id)
        total = len(recs)

        stop_dist: Dict[str, int] = {}
        outcome_dist: Dict[str, int] = {}
        for r in recs:
            sr = r.stop_reason or "unknown"
            stop_dist[sr] = stop_dist.get(sr, 0) + 1
            outcome_dist[r.outcome] = outcome_dist.get(r.outcome, 0) + 1

        max_token_stops = stop_dist.get("max_tokens", 0)
        parse_failures = outcome_dist.get(OUTCOME_PARSE_FAILED, 0)
        empty_results = outcome_dist.get(OUTCOME_EMPTY_RESULT, 0)
        throttles = sum(1 for r in recs if r.outcome == OUTCOME_EXCEPTION and (r.stop_reason == "ThrottlingException"))

        # Output-token distribution per purpose.
        by_purpose: Dict[str, Dict[str, Any]] = {}
        purposes = sorted({r.purpose for r in recs})
        for p in purposes:
            out_toks = sorted(r.output_tokens for r in recs if r.purpose == p and r.output_tokens is not None)
            p_recs = [r for r in recs if r.purpose == p]
            by_purpose[p] = {
                "calls": len(p_recs),
                "max_tokens_stops": sum(1 for r in p_recs if r.stop_reason == "max_tokens"),
                "p50_output_tokens": _percentile(out_toks, 0.50),
                "p95_output_tokens": _percentile(out_toks, 0.95),
                "max_output_tokens": out_toks[-1] if out_toks else None,
            }

        return {
            "owner": owner,
            "doc_id": doc_id,
            "total_calls": total,
            "stop_reason_distribution": stop_dist,
            "outcome_distribution": outcome_dist,
            "max_tokens_stops": max_token_stops,
            "parse_failures": parse_failures,
            "empty_result_calls": empty_results,
            "throttling_exceptions": throttles,
            "by_purpose": by_purpose,
        }

    def write_summary(self, owner: str, doc_id: str, output_root: Optional[Path] = None) -> Optional[Path]:
        """Write the per-document summary to the owner-partitioned metrics dir.
        owner is REQUIRED (tenant isolation). Returns the path written, or None."""
        if not owner or not str(owner).strip():
            raise ValueError("owner is required and cannot be empty")
        root = output_root or (Path("dataset_output") / "_users" / str(owner).strip().lower() / "metrics")
        root.mkdir(parents=True, exist_ok=True)
        safe_doc = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(doc_id))[:120]
        path = root / f"{safe_doc}.llm_metrics.json"
        summary = self.summarize(owner=owner, doc_id=doc_id)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("llm.metrics.summary_written", owner=owner, doc_id=doc_id, path=str(path),
                    total_calls=summary["total_calls"])
        return path


    def amend_last_outcome(self, purpose: str, outcome: str,
                           page_number: Optional[int] = None) -> None:
        """Refine the OUTCOME of the most recent record matching (purpose[, page]).
        Used by the Layer-3 methods to upgrade an 'ok' Converse record to
        'parse_failed' or 'empty_result' AFTER they parse the response — so the
        call isn't double-counted. Metadata only."""
        with self._lock:
            for rec in reversed(self._records):
                if rec.purpose == purpose and (page_number is None or rec.page_number == page_number):
                    rec.outcome = outcome
                    return


# Module-level singleton shared by all call sites + the summary writer.
_collector = MetricsCollector()

# Thread-local run context so the extractor can stamp owner/doc_id onto records
# WITHOUT changing the client method signatures (measurement-only, no behavior change).
# A call site's explicit owner/doc_id/page args always win over this context.
_ctx = threading.local()


def set_run_context(owner: Optional[str] = None, doc_id: Optional[str] = None) -> None:
    _ctx.owner = owner
    _ctx.doc_id = doc_id


def clear_run_context() -> None:
    _ctx.owner = None
    _ctx.doc_id = None


def _ctx_owner() -> Optional[str]:
    return getattr(_ctx, "owner", None)


def _ctx_doc_id() -> Optional[str]:
    return getattr(_ctx, "doc_id", None)


def get_collector() -> MetricsCollector:
    return _collector


def record_converse(
    *,
    purpose: str,
    model_id: str,
    max_tokens: Optional[int],
    response: Optional[Dict[str, Any]],
    latency_ms: Optional[float],
    outcome: str = OUTCOME_OK,
    owner: Optional[str] = None,
    doc_id: Optional[str] = None,
    page_number: Optional[int] = None,
    exception_type: Optional[str] = None,
) -> None:
    """Record ONE Converse call's metadata. Pure observer — returns None, mutates
    nothing the caller holds. `response` is read for stopReason + usage ONLY; its
    content blocks are never touched. On an exception path, pass response=None and
    exception_type; stop_reason carries the exception type so throttling is countable."""
    usage = usage_from_response(response)
    stop_reason = stop_reason_from_response(response)
    if outcome == OUTCOME_EXCEPTION and exception_type:
        stop_reason = exception_type  # e.g. "ThrottlingException" — metadata only
    # Fall back to the thread-local run context for owner/doc_id when the call
    # site didn't pass them explicitly.
    if owner is None:
        owner = _ctx_owner()
    if doc_id is None:
        doc_id = _ctx_doc_id()
    rec = CallRecord(
        purpose=purpose,
        model_id=model_id,
        stop_reason=stop_reason,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        max_tokens=max_tokens,
        outcome=outcome,
        owner=owner,
        doc_id=doc_id,
        page_number=page_number,
    )
    _collector.record(rec)


def amend_outcome(purpose: str, outcome: str, page_number: Optional[int] = None) -> None:
    """Refine the outcome of the most recent matching call record (metadata only)."""
    _collector.amend_last_outcome(purpose=purpose, outcome=outcome, page_number=page_number)
