"""
File: pipeline.py
Purpose: Layer 3 Navigation-First Pipeline Orchestrator.
         Implements the end-to-end pipeline specified in:
         - docs/poc/layer3_pageindex_navigation_poc.md (Phases A, B, C, Section 9)
         - docs/poc/layer3_graph_memory_poc_v2.md (Phase 2 identity resolution)

Flow:
  1. Phase A (Steps 1 & 2): Heuristic boundary detection (0 LLM calls) + Segment summaries (S LLM calls).
  2. Phase B (Step 3): Navigation per ontology category (9 LLM calls).
  3. Phase C (Step 4): Targeted extraction on navigated pages only (N LLM calls).
  4. Step 5: Deterministic recall check on Identifier and Amount (0 LLM calls, segment-only re-runs on failure).
  5. Step 6: Scoped identity resolution on Person and Organization (bounded reasoning pass).
  6. Step 7: Deterministic identifier matching (strict kind-gated zero-LLM pass).
  7. Step 8: Document-level key findings synthesis (1 LLM call or heuristic fallback).
  8. Cost reporting: Logs S + 9 + N + 1 vs exhaustive scan baseline (P + 1).

Owner: engineer-a@idp-pilot
Created: 2026-09-07
"""
# ARCHITECTURAL SEAM: extraction_focus
# Extraction is intentionally schema-free and model-judged: the LLM determines what is important
# in each document based on the ontology baseline floor and document context. The optional
# extraction_focus parameter provides an architectural seam for future user-directed steering.
# When provided, free-text hints are forwarded end-to-end to navigation, page extraction, and
# key findings prompts without enforcing rigid field constraints or predefined Pydantic schemas.

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.expectation_check import ExpectationCheckResult, run_expectation_check
from src.ai.layer3_extraction.extractor import (
    CandidateEdge,
    CandidateNode,
    extract_candidate_graph,
)
from src.ai.layer3_extraction.identity_resolution import (
    DisambiguationResult,
    ResolvedEntity,
    run_scoped_identity_resolution,
)
from src.ai.layer3_extraction.identifier_matching import (
    IdentifierMatchResult,
    run_identifier_matching,
)
from src.ai.layer3_extraction.key_findings import (
    KeyFinding,
    extract_document_key_findings,
)
from src.ai.layer3_extraction.navigation import NavigationMap, navigate_categories
from src.ai.layer3_extraction.segmentation import Segment, build_document_segments
from src.ai.schemas.page import PageOutput
from src.config.ontology import OntologyConfig, load_ontology
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CostSummary(BaseModel):
    total_pages_P: int
    total_segments_S: int
    segmentation_llm_calls: int
    navigation_llm_calls: int
    navigated_pages_N: int
    extraction_llm_calls: int
    expectation_check_rerun_calls: int = 0
    key_findings_llm_calls: int = 0
    actual_llm_calls_total: int
    baseline_exhaustive_calls: int
    reduction_percentage: float


class NavigationPipelineResult(BaseModel):
    segments: List[Segment]
    navigation_map: NavigationMap
    candidate_nodes: List[CandidateNode]
    candidate_edges: List[CandidateEdge]
    resolved_entities: List[ResolvedEntity]
    expectation_check_results: List[ExpectationCheckResult]
    disambiguation_log: List[DisambiguationResult]
    identifier_match_log: List[IdentifierMatchResult] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    cost_summary: CostSummary
    elapsed_seconds: float

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Serialize the complete pipeline result to a JSON file on disk."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return p



def run_navigation_extraction_pipeline(
    pages: Union[List[Dict[str, Any]], List[PageOutput], List[str], Sequence[Any]],
    llm: Optional[ExtractionLLMClient] = None,
    ontology: Optional[OntologyConfig] = None,
    extraction_focus: Optional[List[str]] = None,
    target_pages: Optional[List[int]] = None,
) -> NavigationPipelineResult:
    """
    Execute the full Layer 3 navigation-first pipeline.
    If target_pages is supplied, Phase A (segmentation) and Phase B (navigation)
    are completely bypassed (0 LLM calls), and extraction executes strictly on target_pages.
    """
    start_time = time.time()
    if llm is None:
        llm = get_extraction_client()
    if ontology is None:
        ontology = load_ontology()

    # Normalize pages for extraction, preserving OCR confidence and engine metadata
    normalized_pages = []
    for idx, p in enumerate(pages):
        if isinstance(p, PageOutput):
            normalized_pages.append({
                "page_number": p.page_number,
                "markdown": p.markdown,
                "confidence": getattr(p, "confidence", 0.95),
                "chars": getattr(p, "chars", len(p.markdown)),
                "engines_used": getattr(p, "engines_used", []),
                "is_blank": getattr(p, "is_blank", False),
            })
        elif isinstance(p, dict):
            normalized_pages.append(p)
        else:
            p_str = str(p)
            p_clean = p_str.strip()
            normalized_pages.append({
                "page_number": idx + 1,
                "markdown": p_str,
                "confidence": 0.95,
                "chars": len(p_str),
                "engines_used": [],
                "is_blank": len(p_clean) < 30,
            })

    total_pages = len(normalized_pages)
    logger.info("layer3.pipeline.start", total_pages=total_pages)

    if target_pages is not None:
        target_set = set(target_pages)
        pages_to_process = [p for p in normalized_pages if p["page_number"] in target_set]
        logger.info(
            "layer3.pipeline.target_pages_bypass_active",
            target_pages=sorted(list(target_set)),
            matched_pages=len(pages_to_process),
        )
        # Bypass Phase A: 0 LLM calls. Synthetic 1-page segment stubs with empty hint (Step 5 passes open)
        segments = [
            Segment(
                segment_id=f"test_seg_{p['page_number']}",
                page_range=[p["page_number"], p["page_number"]],
                doc_type_hint="",
                one_line_summary=f"Targeted test page {p['page_number']}",
            )
            for p in pages_to_process
        ]
        s_count = len(segments)
        segmentation_calls = 0

        # Bypass Phase B: 0 LLM calls. Empty NavigationMap
        nav_map = NavigationMap(category_pages={}, total_llm_calls=0)
        navigation_calls = 0
        navigated_pages = sorted(list(target_set))
        n_count = len(navigated_pages)
    else:
        pages_to_process = normalized_pages
        # --- Phase A: Segmentation ---
        # Heuristic boundary detection (0 LLM calls) + S summary calls
        segments = build_document_segments(normalized_pages, llm)
        s_count = len(segments)
        segmentation_calls = s_count

        logger.info("layer3.pipeline.phase_a_complete", segments_count=s_count)

        # --- Phase B: Navigation ---
        # Exactly 1 LLM call per ontology category (9 calls total)
        nav_map = navigate_categories(segments, llm, ontology, extraction_focus=extraction_focus)
        navigation_calls = nav_map.total_llm_calls
        navigated_pages = nav_map.all_navigated_pages
        n_count = len(navigated_pages)

        logger.info(
            "layer3.pipeline.phase_b_complete",
            navigated_pages_count=n_count,
            navigated_pages=navigated_pages,
        )

    # --- Phase C: Targeted Extraction ---
    # Extract only from navigated pages (write-only, zero cross-page reads/merges)
    logger.info(
        "phase_c.llm_check",
        llm_type=type(llm).__name__ if llm is not None else "None",
        has_method=hasattr(llm, "extract_page_ontology") if llm is not None else False,
    )
    t_phase_c = time.time()
    nodes, edges = extract_candidate_graph(
        pages_md=pages_to_process,
        target_pages=navigated_pages,
        llm=llm,
        ontology=ontology,
        extraction_focus=extraction_focus,
    )
    phase_c_duration = round(time.time() - t_phase_c, 3)
    extraction_calls = n_count

    logger.info(
        "layer3.pipeline.phase_c_complete",
        extracted_nodes=len(nodes),
        extracted_edges=len(edges),
        duration_s=phase_c_duration,
    )

    # --- Step 5: Deterministic Recall Check ---
    # Checks doc_type_hint expectations on Identifier and Amount only
    updated_nodes, updated_edges, check_results = run_expectation_check(
        segments=segments,
        nodes=nodes,
        edges=edges,
        pages_md=normalized_pages,
        llm=llm,
    )
    reruns_count = sum(1 for r in check_results if r.rerun_triggered)

    # --- Step 6: Scoped Identity Resolution ---
    # Bounded disambiguation pass over Person and Organization candidates only
    resolved_entities, step6_edges, disambiguation_log = run_scoped_identity_resolution(
        nodes=updated_nodes,
        edges=updated_edges,
    )

    # --- Step 7: Deterministic Identifier Matching ---
    # Deterministic matching pass for Identifier candidates
    resolved_identifiers, final_edges, identifier_match_log = run_identifier_matching(
        nodes=updated_nodes,
        edges=step6_edges,
    )
    all_resolved_entities = resolved_entities + resolved_identifiers

    # --- Step 8: Document-Level Key Findings ---
    key_findings = extract_document_key_findings(
        segments=segments,
        candidate_nodes=updated_nodes,
        llm=llm,
        extraction_focus=extraction_focus,
    )
    key_findings_calls = 1 if (llm is not None and hasattr(llm, "extract_key_findings")) else 0

    elapsed = round(time.time() - start_time, 3)

    # Cost model per Section 9:
    # Navigation pipeline: S + 9 + N (+ reruns) + 1 (key findings)
    # Baseline exhaustive scan: P + 1
    actual_calls = segmentation_calls + navigation_calls + extraction_calls + reruns_count + key_findings_calls
    baseline_calls = total_pages + 1
    reduction_pct = round((1.0 - (actual_calls / max(1, baseline_calls))) * 100, 1)

    cost = CostSummary(
        total_pages_P=total_pages,
        total_segments_S=s_count,
        segmentation_llm_calls=segmentation_calls,
        navigation_llm_calls=navigation_calls,
        navigated_pages_N=n_count,
        extraction_llm_calls=extraction_calls,
        expectation_check_rerun_calls=reruns_count,
        key_findings_llm_calls=key_findings_calls,
        actual_llm_calls_total=actual_calls,
        baseline_exhaustive_calls=baseline_calls,
        reduction_percentage=reduction_pct,
    )

    logger.info(
        "layer3.pipeline.finished",
        actual_calls=actual_calls,
        baseline_calls=baseline_calls,
        reduction_pct=f"{reduction_pct}%",
        elapsed_seconds=elapsed,
    )

    return NavigationPipelineResult(
        segments=segments,
        navigation_map=nav_map,
        candidate_nodes=updated_nodes,
        candidate_edges=final_edges,
        resolved_entities=all_resolved_entities,
        expectation_check_results=check_results,
        disambiguation_log=disambiguation_log,
        identifier_match_log=identifier_match_log,
        key_findings=key_findings,
        cost_summary=cost,
        elapsed_seconds=elapsed,
    )


def run_pipeline(
    doc_path: Union[str, Path],
    llm: Optional[ExtractionLLMClient] = None,
    ontology: Optional[OntologyConfig] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    extraction_focus: Optional[List[str]] = None,
    target_pages: Optional[List[int]] = None,
) -> NavigationPipelineResult:
    """
    Top-level entry point to execute the Layer 3 navigation-first pipeline
    on a markdown document file.
    If target_pages is provided, Phase A/B are bypassed and extraction targets those pages.
    """
    p = Path(doc_path)
    if not p.exists():
        candidates = [
            Path("dataset_output") / p.name,
            Path("dataset_output") / p.name.replace("_", " "),
            Path(str(p).replace("_", " ")),
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    if not p.exists():
        raise FileNotFoundError(f"Document file not found: {doc_path}")

    logger.info("layer3.pipeline.run_pipeline_called", path=str(p), target_pages=target_pages)

    with open(p, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Split into individual page blocks using <!-- PAGE (\d+) ... -->
    page_marker = re.compile(r"<!-- PAGE (\d+) \| ([^>]+) -->")
    parts = page_marker.split(md_text)

    pages: List[Dict[str, Any]] = []
    if len(parts) > 1:
        for i in range(1, len(parts), 3):
            pnum = int(parts[i])
            pmeta = parts[i + 1]
            pbody = re.sub(r"<!-- /PAGE \d+ -->.*", "", parts[i + 2], flags=re.DOTALL).strip()
            conf_match = re.search(r"conf=([0-9.]+)", pmeta)
            conf = float(conf_match.group(1)) if conf_match else 0.95
            eng_match = re.search(r"engine=([a-zA-Z0-9_]+)", pmeta)
            engines = [eng_match.group(1)] if eng_match else []
            pages.append({
                "page_number": pnum,
                "markdown": pbody,
                "confidence": conf,
                "chars": len(pbody),
                "engines_used": engines,
                "is_blank": len(pbody) < 30,
            })
    else:
        splits = re.split(r"\n---\n", md_text)
        for idx, sp in enumerate(splits, 1):
            cleaned_sp = sp.strip()
            if cleaned_sp:
                pages.append({
                    "page_number": idx,
                    "markdown": cleaned_sp,
                    "confidence": 0.95,
                    "chars": len(cleaned_sp),
                    "engines_used": [],
                    "is_blank": len(cleaned_sp) < 30,
                })

    # Load companion schema_ref.json for accurate OCR metadata if present
    schema_ref_candidate = p.with_suffix(".schema_ref.json")
    if not schema_ref_candidate.exists():
        schema_ref_candidate = p.parent / (p.stem + ".schema_ref.json")
    if schema_ref_candidate.exists():
        try:
            with open(schema_ref_candidate, "r", encoding="utf-8") as f:
                schema_ref = json.load(f)
            ref_pages = {item["page_number"]: item for item in schema_ref.get("pages", [])}
            for page in pages:
                pnum = page.get("page_number")
                if pnum is not None and pnum in ref_pages:
                    ref = ref_pages[pnum]
                    page["confidence"] = ref.get("confidence", 1.0)
                    md_val = str(page.get("markdown", ""))
                    page["chars"] = ref.get("chars", len(md_val))
                    page["engines_used"] = ref.get("engines_used", [])
                    page["is_blank"] = ref.get("chars", 0) < 30 or not md_val.strip()
        except Exception as e:
            logger.warning("pipeline.schema_ref_load_failed", error=str(e))

    if llm is None:
        llm = get_extraction_client()
    if ontology is None:
        ontology = load_ontology()

    result = run_navigation_extraction_pipeline(
        pages=pages,
        llm=llm,
        ontology=ontology,
        extraction_focus=extraction_focus,
        target_pages=target_pages,
    )

    # Save to JSON file (defaults to <doc_stem>.layer3_result.json)
    target_out = output_json_path or p.with_suffix(".layer3_result.json")
    try:
        saved_path = result.save_json(target_out)
        logger.info("layer3.pipeline.result_saved", path=str(saved_path))
    except Exception as e:
        logger.warning("layer3.pipeline.save_json_failed", error=str(e))

    return result
