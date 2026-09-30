"""
File: pipeline.py
Purpose: Orchestrates inspect -> capabilities -> plan -> run -> merge -> PageOutput.
         Supports two routing modes via settings.routing_mode:
           "single_engine"    — legacy router.py path: one route per page,
                                route_from_profile() / resolve_route_with_classification().
                                Default — no behaviour change unless overridden.
           "capability_based" — opt-in Stage 1 path: capabilities_from_profile()
                                builds a multi-engine plan, results are merged
                                with line-level dedup. Activated by setting
                                IDP_ROUTING_MODE=capability_based.
         Escalation ladder is shared across both modes as a confidence fallback.
         Returns (PageOutput, PageMetadata) — PageOutput is the Engineer B contract,
         PageMetadata is the internal processing audit trail.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (routing_mode dispatch, PageMetadata wiring)
Deps: pydantic
"""
import time
from typing import Any

from src.adapters.llm.base import LLMClient
from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import (
    build_engine_plan,
    capabilities_from_profile,
    capabilities_from_vlm_analysis,
    next_escalation_route,
    route_from_profile,
)
from src.ai.layer2_conversion.digital import convert_digital_page
from src.ai.layer2_conversion.scanned import (
    convert_handwritten_via_paddle,
    convert_scanned_page,
)
from src.ai.output import write_document
from src.ai.schemas.page import EngineTask, PageCapabilities, PageOutput, VLMAnalysis
from src.ai.schemas.page_metadata import PageMetadata
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Route string → engine name mapping used by both single_engine plan wrapping
# and the escalation ladder's engine lookup.
_ROUTE_TO_ENGINE: dict[str, str] = {
    "digital":        "docling",
    "scanned":        "paddleocr_printed",
    "handwritten":    "paddleocr_handwritten",
    "vlm_transcribe": "vlm_transcribe",
    "skip":           "skip",
}
_ENGINE_TO_ROUTE: dict[str, str] = {v: k for k, v in _ROUTE_TO_ENGINE.items()}


# ---------------------------------------------------------------------------
# Internal: single-engine dispatch
# ---------------------------------------------------------------------------

def _run_engine_task(
    task: EngineTask,
    page_context: dict,
    llm_client: LLMClient,
) -> tuple[str, float, float]:
    """
    Execute one EngineTask and return (markdown, confidence, latency_ms).
    All exceptions are caught and surfaced as ("", 0.0, latency_ms) so a
    single engine failure never aborts the entire plan.
    """
    t0 = time.monotonic()
    try:
        if task.engine == "docling":
            text, conf = convert_digital_page(
                page_context["pdf_path"],
                page_context["page_number"],
            )

        elif task.engine == "paddleocr_printed":
            text, conf = convert_scanned_page(
                page_context.get("image_array"),
                page_context["page_number"],
                image_bytes=page_context.get("image_bytes"),
            )

        elif task.engine == "paddleocr_handwritten":
            text, conf = convert_handwritten_via_paddle(
                page_context.get("image_array"),
                page_context["page_number"],
                image_bytes=page_context.get("image_bytes"),
            )

        elif task.engine == "vlm_transcribe":
            logger.info(
                "pipeline.vlm_transcribe_invoked",
                page_number=page_context.get("page_number"),
            )
            text, conf = llm_client.transcribe_handwriting(page_context["image_bytes"])

        elif task.engine == "skip":
            text, conf = "", 1.0

        else:
            raise ValueError(f"Unknown engine: {task.engine!r}")

        latency_ms = (time.monotonic() - t0) * 1000
        return text, conf, latency_ms

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "pipeline.engine_task_failed",
            engine=task.engine,
            page_number=page_context.get("page_number"),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0, latency_ms


# ---------------------------------------------------------------------------
# Internal: multi-engine plan execution
# ---------------------------------------------------------------------------

def _run_plan(
    plan: list[EngineTask],
    page_context: dict,
    llm_client: LLMClient,
) -> list[tuple[EngineTask, str, float, float]]:
    """
    Execute the primary task in the plan.
    Only 1 engine runs initially. If it fails or confidence is below threshold,
    the pipeline escalates directly to VLM without chaining intermediate OCR engines.
    """
    results: list[tuple[EngineTask, str, float, float]] = []
    if plan:
        task = plan[0]
        markdown, confidence, latency_ms = _run_engine_task(task, page_context, llm_client)
        logger.info(
            "pipeline.engine_task_complete",
            engine=task.engine,
            page_number=page_context.get("page_number"),
            confidence=round(confidence, 4),
            chars_out=len(markdown.strip()),
            latency_ms=round(latency_ms, 1),
        )
        if markdown.strip():
            results.append((task, markdown, confidence, latency_ms))
        else:
            results.append((task, "", confidence, latency_ms))
    return results


def _analyze_with_vlm(
    llm_client: LLMClient, image_bytes: bytes, profile
) -> VLMAnalysis:
    """Use rich analysis when available; base clients retain old compatibility."""
    return llm_client.analyze_page(
        image_bytes, page_profile_hint=profile.model_dump()
    )


def _finalize_direct_vlm(
    page_number: int,
    metadata: PageMetadata,
    analysis: VLMAnalysis,
    caps: PageCapabilities,
    page_start: float,
) -> tuple[PageOutput, PageMetadata]:
    """Create the normal page contract for a terminal VLM extraction."""
    capabilities = sorted(analysis.detected_capabilities)
    logger.info(
        "page.routing_decision",
        page_number=page_number,
        route="vlm_direct",
        confidence=analysis.confidence,
    )
    metadata.capabilities = capabilities
    metadata.set_routing(
        engine_plan=["vlm_direct"],
        routing_mode="vlm_direct",
        selected_engine="vlm_direct",
        route_confidence=analysis.confidence,
    )
    metadata.add_engine_result(
        engine="vlm_direct",
        confidence=analysis.confidence,
        success=True,
        output_type="markdown",
    )
    metadata.set_final_result(
        engine="vlm_direct",
        confidence=analysis.confidence,
        success=True,
        total_latency_ms=round((time.monotonic() - page_start) * 1000, 1),
    )
    return PageOutput(
        page_number=page_number,
        markdown=analysis.extracted_markdown,
        engines_used=["vlm_direct"],
        confidence=analysis.confidence,
        capabilities=capabilities or caps.active_capabilities(),
        escalated=False,
        escalation_attempts=0,
        low_confidence=False,
    ), metadata


# ---------------------------------------------------------------------------
# Internal: merge + dedup
# ---------------------------------------------------------------------------

def _merge_results(
    results: list[tuple[EngineTask, str, float, float]],
) -> tuple[str, float, list[str]]:
    """
    Merge outputs from multiple engines into a single (markdown, confidence,
    engines_used) tuple.  The latency_ms field in each tuple is used by the
    caller for PageMetadata but not consumed here.

    Stage 1 merge strategy — full-page, no region bounding boxes:
      1. Split each engine's output into non-empty lines.
      2. Build a seen-set from the PRIMARY engine (lowest priority number)
         so its text is never dropped.
      3. For each subsequent engine, add lines that are NOT exact duplicates
         of anything already in the merged output.  Whitespace is normalised
         before comparison so minor formatting differences don't cause false
         non-duplicates.
      4. Confidence = weighted average by char-count contribution of each
         engine's non-duplicate lines.

    This is intentionally simple.  Stage 2 will replace the string-level
    dedup with spatial bbox dedup once region maps are available.
    """
    if not results:
        return "", 0.0, []

    if len(results) == 1:
        task, markdown, confidence, _latency = results[0]
        return markdown, confidence, [task.engine]

    # ── Multi-engine merge ──────────────────────────────────────────────────
    merged_lines: list[str] = []
    seen_normalised: dict[str, str] = {}
    total_weighted_conf = 0.0
    total_chars = 0
    engines_used: list[str] = []

    for task, markdown, confidence, _latency in results:
        engine_lines = [ln for ln in markdown.split("\n") if ln.strip()]
        added_chars = 0

        for line in engine_lines:
            norm = " ".join(line.lower().split())
            if norm not in seen_normalised:
                seen_normalised[norm] = line
                merged_lines.append(line)
                added_chars += len(line)

        if added_chars > 0:
            total_weighted_conf += confidence * added_chars
            total_chars += added_chars
            engines_used.append(task.engine)

    merged_markdown = "\n".join(merged_lines)
    avg_confidence = total_weighted_conf / total_chars if total_chars > 0 else 0.0

    logger.info(
        "pipeline.merge_complete",
        engines_merged=engines_used,
        total_lines=len(merged_lines),
        avg_confidence=round(avg_confidence, 4),
    )
    return merged_markdown, round(avg_confidence, 4), engines_used


# ---------------------------------------------------------------------------
# Public: process a single page
# ---------------------------------------------------------------------------

def process_page(
    page: Any,
    page_number: int,
    page_context: dict,
    llm_client: LLMClient,
    page_image_bytes: bytes,
    document_name: str = "",
    document_id: str = "",
    extraction_requirements: dict | None = None,
) -> tuple[PageOutput, PageMetadata]:
    """
    Full pipeline for one page. Returns (PageOutput, PageMetadata).

    PageOutput  — the contract handed to Engineer B's Layer 3 (unchanged).
    PageMetadata — the complete internal audit trail for this page.

    routing_mode="single_engine" (default):
      1. inspect_page() → PageProfile
      2. route_from_profile() → single route string (Step A)
         if None: llm_client.classify_page() → resolve_route_with_classification() (Step B)
      3. Wrap single route in a one-task plan
      4–6. _run_plan() + _merge_results()
      7. Escalation ladder fallback if confidence < threshold
      8. Return (PageOutput, PageMetadata)

    routing_mode="capability_based" (opt-in):
      1. inspect_page() → PageProfile
      2. capabilities_from_profile() → PageCapabilities
      3. If ambiguous: llm_client.classify_page() → capabilities_from_classification()
      4. build_engine_plan() → list[EngineTask]
      5. _run_plan() → per-engine (markdown, confidence, latency_ms)
      6. _merge_results() → merged markdown, weighted confidence, engines_used
      7. Escalation ladder fallback if merged confidence < threshold
      8. Return (PageOutput, PageMetadata)
    """
    page_start = time.monotonic()
    page_context = {**page_context, "image_bytes": page_image_bytes, "page_number": page_number}

    # Initialise metadata — will be populated incrementally throughout
    metadata = PageMetadata(
        document_id=document_id or None,
        document_name=document_name or None,
        page_number=page_number,
    )

    # ── Step 1: inspect ─────────────────────────────────────────────────────
    profile = inspect_page(page, page_number)

    metadata.char_count       = profile.char_count
    metadata.image_coverage   = profile.image_coverage
    metadata.is_scanned       = profile.is_scanned
    metadata.primary_script   = profile.primary_script
    metadata.complexity_score = float(profile.complexity_score)

    # ── Steps 2–4: route resolution + engine plan ────────────────────────────
    if settings.routing_mode == "capability_based":
        caps = capabilities_from_profile(profile)
        plan = build_engine_plan(caps)

        if not plan:
            logger.warning("pipeline.empty_plan", page_number=page_number)
            metadata.set_final_result(None, 0.0, success=False,
                                       total_latency_ms=(time.monotonic() - page_start) * 1000)
            output = PageOutput(
                page_number=page_number, markdown="", engines_used=[],
                confidence=0.0, capabilities=caps.active_capabilities(),
                escalated=False, escalation_attempts=0, low_confidence=True,
            )
            return output, metadata

    else:
        single_route = route_from_profile(profile)
        if single_route is None:
            # Default: route scanned, complex, or image-heavy pages directly to PaddleOCR first
            if (
                profile.is_scanned
                or profile.complexity_score >= 4
                or profile.image_coverage > settings.mixed_content_min_image_coverage
            ):
                single_route = "scanned"
            else:
                single_route = "digital"

        caps = capabilities_from_profile(profile)
        plan = [EngineTask(
            engine=_ROUTE_TO_ENGINE.get(single_route, single_route),
            priority=1,
            reason=f"single_engine:{single_route}",
        )]

        logger.info(
            "pipeline.single_engine_route",
            page_number=page_number,
            route=single_route,
            routing_mode="single_engine",
        )

    # Populate capability + routing metadata
    for cap in caps.active_capabilities():
        metadata.add_capability(cap)

    primary_task = plan[0] if plan else None
    planned_engines = [primary_task.engine] if primary_task else []
    metadata.set_routing(
        engine_plan=planned_engines,
        routing_mode=settings.routing_mode,
        selected_engine=primary_task.engine if primary_task else None,
    )

    # ── Step 5 + 6: run plan and merge ──────────────────────────────────────
    raw_results = _run_plan(plan, page_context, llm_client)

    # Record each engine result in metadata
    for task, _md, conf, latency_ms in raw_results:
        metadata.add_engine_result(
            engine=task.engine,
            confidence=round(conf, 4),
            success=True,
            latency_ms=round(latency_ms, 1),
            output_type="markdown",
        )

    markdown, confidence, engines_used = _merge_results(raw_results)

    # ── Step 7: escalation fallback ─────────────────────────────────────────
    # Only 1 engine runs initially; if confidence < threshold or text is empty,
    # route directly to VLM without chaining intermediate OCR rungs.
    escalation_attempts = 0
    escalated = False

    primary_engine = primary_task.engine if primary_task else "skip"
    current_route = _ENGINE_TO_ROUTE.get(primary_engine, "scanned")
    engines_attempted = {primary_engine} | set(engines_used)

    while (
        (confidence < settings.escalation_confidence_threshold or not markdown.strip())
        and escalation_attempts < settings.max_escalation_attempts
    ):
        next_route = next_escalation_route(
            current_route, reason=f"confidence={confidence:.2f} < threshold={settings.escalation_confidence_threshold}"
        )
        if next_route is None:
            logger.warning(
                "pipeline.escalation_terminal",
                page_number=page_number,
                current_route=current_route,
                confidence=confidence,
            )
            break

        escalation_engine = _ROUTE_TO_ENGINE.get(next_route, next_route)
        if escalation_engine in engines_attempted:
            break
        engines_attempted.add(escalation_engine)

        esc_task = EngineTask(
            engine=escalation_engine,
            priority=99,
            reason=f"escalation_from_{current_route}",
        )
        esc_markdown, esc_confidence, esc_latency = _run_engine_task(
            esc_task, page_context, llm_client
        )

        # Record escalation in metadata before re-merging
        metadata.add_escalation(
            from_engine=current_route,
            to_engine=escalation_engine,
            reason=f"confidence={confidence:.2f} < threshold={settings.escalation_confidence_threshold}",
            from_confidence=round(confidence, 4),
            threshold=settings.escalation_confidence_threshold,
            attempt=escalation_attempts + 1,
        )
        metadata.add_engine_result(
            engine=escalation_engine,
            confidence=round(esc_confidence, 4),
            success=bool(esc_markdown.strip()),
            latency_ms=round(esc_latency, 1),
            output_type="markdown",
        )

        if esc_markdown.strip():
            if escalation_engine == "vlm_transcribe":
                # VLM provides full-page transcription. When escalating because Paddle OCR failed
                # or had low confidence (< 0.75), VLM's markdown takes precedence.
                if esc_confidence >= confidence or not markdown.strip():
                    markdown = esc_markdown
                    confidence = esc_confidence
                    if escalation_engine not in engines_used:
                        engines_used.append(escalation_engine)
                else:
                    if escalation_engine not in engines_used:
                        engines_used.append(escalation_engine)
            else:
                all_results = raw_results + [(esc_task, esc_markdown, esc_confidence, esc_latency)]
                markdown, confidence, engines_used = _merge_results(all_results)

        escalation_attempts += 1
        escalated = True
        current_route = next_route

        logger.info(
            "pipeline.escalation_attempt",
            page_number=page_number,
            attempt=escalation_attempts,
            escalation_engine=escalation_engine,
            new_confidence=round(confidence, 4),
        )

    low_confidence = confidence < settings.capability_low_confidence_floor

    # ── Step 8: finalise metadata ────────────────────────────────────────────
    total_latency_ms = (time.monotonic() - page_start) * 1000
    metadata.set_final_result(
        engine=engines_used[0] if engines_used else None,
        confidence=round(confidence, 4),
        success=bool(markdown.strip()),
        total_latency_ms=round(total_latency_ms, 1),
    )

    output = PageOutput(
        page_number=page_number,
        markdown=markdown,
        engines_used=engines_used,
        confidence=round(confidence, 4),
        capabilities=caps.active_capabilities(),
        escalated=escalated,
        escalation_attempts=escalation_attempts,
        low_confidence=low_confidence,
    )

    logger.info(
        "pipeline.page_complete",
        page_number=page_number,
        engines_used=output.engines_used,
        capabilities=output.capabilities,
        confidence=output.confidence,
        escalated=output.escalated,
        low_confidence=output.low_confidence,
        total_latency_ms=round(total_latency_ms, 1),
    )
    return output, metadata


# ---------------------------------------------------------------------------
# Public: process a whole document
# ---------------------------------------------------------------------------

def process_document(
    pages: list[dict],
    llm_client: LLMClient,
    document_name: str = "",
    document_id: str = "",
    write_output: bool = False,
    output_dir: str | None = None,
    overwrite: bool = True,
    extraction_requirements: dict | None = None,
) -> list[tuple[PageOutput, PageMetadata]]:
    """
    Process every page in a document.
    Each entry in `pages` must supply:
      page          — PyMuPDF page object
      page_number   — int
      context       — dict with pdf_path, image_array, image_bytes
      image_bytes   — PNG bytes of the rendered page

    If write_output=True and output_dir is set, the merged per-document
    markdown is written to <output_dir>/<document_stem>.md via md_writer.

    Returns a list of (PageOutput, PageMetadata) pairs, one per page.
    """
    results: list[tuple[PageOutput, PageMetadata]] = []
    concurrency_limit = max(1, getattr(settings, "page_concurrency_limit", 25))

    if len(pages) <= 1 or concurrency_limit == 1:
        for page_data in pages:
            output, metadata = process_page(
                page=page_data["page"],
                page_number=page_data["page_number"],
                page_context=page_data["context"],
                llm_client=page_data.get("llm_client", llm_client),
                page_image_bytes=page_data["image_bytes"],
                document_name=document_name,
                document_id=document_id,
                extraction_requirements=page_data.get("extraction_requirements", extraction_requirements),
            )
            results.append((output, metadata))
    else:
        from concurrent.futures import ThreadPoolExecutor

        def _worker(page_data: dict) -> tuple[PageOutput, PageMetadata]:
            return process_page(
                page=page_data["page"],
                page_number=page_data["page_number"],
                page_context=page_data["context"],
                llm_client=page_data.get("llm_client", llm_client),
                page_image_bytes=page_data["image_bytes"],
                document_name=document_name,
                document_id=document_id,
                extraction_requirements=page_data.get("extraction_requirements", extraction_requirements),
            )

        max_workers = min(len(pages), concurrency_limit)
        logger.info(
            "pipeline.concurrent_pages_start",
            page_count=len(pages),
            workers=max_workers,
            document=document_name,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map preserves original page ordering (page 1, 2, 3...)
            results = list(executor.map(_worker, pages))

    outputs  = [r[0] for r in results]
    multi_engine_pages = sum(1 for o in outputs if len(o.engines_used) > 1)
    low_conf_pages     = sum(1 for o in outputs if o.low_confidence)
    escalated_pages    = sum(1 for o in outputs if o.escalated)

    logger.info(
        "pipeline.document_complete",
        page_count=len(outputs),
        multi_engine_pages=multi_engine_pages,
        low_confidence_pages=low_conf_pages,
        escalated_pages=escalated_pages,
    )

    if write_output and output_dir and document_name:
        out_path = write_document(
            document_name=document_name,
            pages=results,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        logger.info(
            "pipeline.document_written",
            document=document_name,
            output_path=out_path,
        )

    return results
