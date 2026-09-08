"""
File: key_findings.py
Purpose: Layer 3 Document-Level Key Findings Extraction.
         Synthesizes overarching findings across the document's segments and
         extracted candidate nodes, optionally steered by extraction_focus hints.
Owner: engineer-a@idp-pilot
Updated: 2026-09-07
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.extractor import CandidateNode
from src.ai.layer3_extraction.segmentation import Segment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KeyFinding(BaseModel):
    label: str
    value: str
    importance: Literal["high", "medium", "low"] = "medium"
    reason: str = ""
    source_pages: List[int] = Field(default_factory=list)
    confidence: float = 1.0


def extract_document_key_findings(
    segments: List[Segment],
    candidate_nodes: List[CandidateNode],
    llm: Optional[ExtractionLLMClient] = None,
    extraction_focus: Optional[List[str]] = None,
) -> List[KeyFinding]:
    """
    Synthesize document-level key findings from segments and extracted candidate nodes.
    Calls LLM once if available, else uses deterministic heuristic fallback.

    Args:
        segments: List of document segments.
        candidate_nodes: List of candidate nodes extracted across pages.
        llm: Optional LLM client implementing ExtractionLLMClient.
        extraction_focus: Optional list of user-provided steering hints.

    Returns:
        List of KeyFinding objects.
    """
    if llm is not None and hasattr(llm, "extract_key_findings"):
        try:
            logger.info("key_findings.start_llm", has_focus=bool(extraction_focus))
            seg_dicts = [s.model_dump() for s in segments]
            # Prioritize high and medium importance candidate nodes to keep context concise
            top_nodes = [
                n.model_dump()
                for n in candidate_nodes
                if n.importance in ("high", "medium")
            ][:50]
            findings_data = llm.extract_key_findings(
                segments=seg_dicts,
                candidate_nodes=top_nodes,
                extraction_focus=extraction_focus,
            )
            raw_findings = findings_data.get("key_findings", []) if isinstance(findings_data, dict) else []
            findings: List[KeyFinding] = []
            for item in raw_findings:
                if not isinstance(item, dict):
                    continue
                lbl = str(item.get("label", "")).strip()
                val = str(item.get("value", "")).strip()
                raw_imp = str(item.get("importance", "medium")).lower().strip()
                imp: Literal["high", "medium", "low"] = raw_imp if raw_imp in ("high", "medium", "low") else "medium"
                rsn = str(item.get("reason", "")).strip()
                sp = item.get("source_pages", [])
                pages = [int(p) for p in sp if isinstance(p, (int, str)) and str(p).isdigit()]
                if not pages:
                    pages = [1]
                conf = float(item.get("confidence", 0.95))
                if lbl and val:
                    findings.append(KeyFinding(
                        label=lbl,
                        value=val,
                        importance=imp,
                        reason=rsn,
                        source_pages=sorted(list(set(pages))),
                        confidence=round(conf, 3),
                    ))
            if findings:
                logger.info("key_findings.llm_success", count=len(findings))
                return findings
        except Exception as exc:
            logger.warning("key_findings.llm_failed", error=str(exc))

    return _heuristic_key_findings(segments, candidate_nodes)


def _heuristic_key_findings(
    segments: List[Segment],
    candidate_nodes: List[CandidateNode],
) -> List[KeyFinding]:
    """
    Deterministic fallback when LLM is unavailable:
    Derives key findings from primary document segments and top candidate nodes.
    """
    findings: List[KeyFinding] = []
    seen_labels = set()

    # 1. Primary document types from segments
    for seg in segments:
        if seg.doc_type_hint and seg.doc_type_hint.lower() not in ("unknown", "generic"):
            key = f"Document Type: {seg.doc_type_hint}"
            if key not in seen_labels:
                seen_labels.add(key)
                pages = list(range(seg.page_range[0], seg.page_range[1] + 1))
                findings.append(KeyFinding(
                    label="Document Section",
                    value=f"{seg.doc_type_hint}: {seg.one_line_summary}",
                    importance="high" if len(findings) == 0 else "medium",
                    reason="Identified primary structural segment of the document",
                    source_pages=pages,
                    confidence=0.90,
                ))

    # 2. Key High-Importance entities
    high_imp_nodes = [n for n in candidate_nodes if n.importance == "high"]
    for node in high_imp_nodes:
        key = f"{node.type}:{node.label}"
        if key not in seen_labels:
            seen_labels.add(key)
            findings.append(KeyFinding(
                label=f"Key {node.type}" if not node.subtype else f"Key {node.subtype}",
                value=node.label,
                importance="high",
                reason=node.importance_reason or f"High-importance {node.type} extracted from record",
                source_pages=[node.source_page],
                confidence=node.confidence,
            ))
        if len(findings) >= 10:
            break

    return findings
