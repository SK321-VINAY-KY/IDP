"""
File: router.py
Purpose: Routing decision table, capability detection, engine plan builder,
         and escalation ladder. Stage 1 adds capability-based multi-engine
         routing alongside the existing single-route escalation path.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (Stage 1 capability-based routing)
Deps: pydantic
"""
from src.ai.schemas.page import (
    EngineTask,
    PageCapabilities,
    PageClassification,
    PageProfile,
)
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"digital", "scanned", "handwritten", "vlm_transcribe", "skip"}

# Engine priority constants — lower number runs first and wins merge tiebreaks.
# Rationale: digital text extraction is the highest-fidelity source when
# available; printed OCR is next; handwriting OCR is supplementary.
_PRI_DIGITAL = 1
_PRI_PRINTED = 2
_PRI_HANDWRITTEN = 3
_PRI_VLM = 4

# ---------------------------------------------------------------------------
# Indic script set (engine deferred — see build guide §9)
# ---------------------------------------------------------------------------
_INDIC_SCRIPTS = {
    "devanagari", "tamil", "bengali", "gujarati",
    "gurmukhi", "kannada", "malayalam", "odia", "telugu",
}


# ---------------------------------------------------------------------------
# Legacy helpers — kept for backward-compat and the escalation ladder path
# ---------------------------------------------------------------------------

def is_mixed_content(profile: PageProfile) -> bool:
    """
    Fingerprint for 'typed form + handwritten annotation' style pages:
    substantial embedded text AND meaningful-but-not-total image coverage.
    """
    return (
        profile.char_count > settings.mixed_content_min_char_count
        and settings.mixed_content_min_image_coverage
        < profile.image_coverage
        < settings.mixed_content_max_image_coverage
    )


def route_from_profile(profile: PageProfile) -> str | None:
    """
    Legacy Step A programmatic routing — returns a single route string or None.
    Still used by the escalation ladder fallback path.
    For new call sites prefer capabilities_from_profile() + build_engine_plan().
    """
    if (
        profile.char_count < settings.skip_char_count_threshold
        and profile.image_coverage < settings.skip_image_coverage_threshold
    ):
        return "skip"

    if profile.primary_script in _INDIC_SCRIPTS:
        logger.warning(
            "router.indic_engine_deferred",
            page_number=profile.page_number,
            primary_script=profile.primary_script,
        )
        return "scanned"

    if profile.has_text and profile.char_count > settings.digital_char_count_threshold:
        if is_mixed_content(profile):
            logger.info(
                "router.mixed_content_detected",
                page_number=profile.page_number,
                reason="digital_with_partial_image",
            )
            return None
        return "digital"

    if profile.complexity_score >= 4:
        return None

    return None


def resolve_route_with_classification(
    profile: PageProfile, classification: PageClassification
) -> str:
    """
    Legacy Step B — resolve using VLM classification when Step A was
    inconclusive.  Implements the handwriting_pct dead-zone fix.
    """
    if classification.handwriting_pct >= settings.handwriting_pct_handwritten_floor:
        route = "handwritten"
    elif classification.handwriting_pct <= settings.handwriting_pct_scanned_ceiling:
        route = "scanned"
    else:
        logger.info(
            "router.dead_zone_handwriting_pct",
            page_number=profile.page_number,
            handwriting_pct=classification.handwriting_pct,
            resolution="handwritten",
        )
        route = "handwritten"

    logger.info(
        "router.step_b_resolved",
        page_number=profile.page_number,
        route=route,
        handwriting_pct=classification.handwriting_pct,
        vlm_confidence=classification.confidence,
    )
    return route


def next_escalation_route(current_route: str, reason: str) -> str | None:
    """
    Escalation ladder (single-engine retry path).
        digital -> scanned -> handwritten -> vlm_transcribe -> None
    """
    ladder = {
        "digital":        "scanned",
        "scanned":        "handwritten",
        "handwritten":    "vlm_transcribe",
        "vlm_transcribe": None,
    }
    next_route = ladder.get(current_route)
    logger.info(
        "router.escalation_decision",
        from_route=current_route,
        to_route=next_route,
        reason=reason,
    )
    return next_route


# ---------------------------------------------------------------------------
# Stage 1: capability detection
# ---------------------------------------------------------------------------

def capabilities_from_profile(profile: PageProfile) -> PageCapabilities:
    """
    Derive a PageCapabilities set from a PageProfile using pure heuristics —
    no model calls, ~0ms on top of what inspect_page() already computed.

    Multiple capabilities can be True simultaneously:
      - A form with printed labels + handwritten fill-ins →
          has_digital_text=True AND has_handwriting=True
      - A scanned printed document →
          has_printed_scan=True
      - A mixed academic page with text, tables, and a scanned figure →
          has_digital_text=True AND has_tables=True AND has_figures=True

    Stage 1 note: all capabilities apply to the full page.  No region
    bounding boxes yet — Stage 2 will add spatial decomposition.
    """
    caps = PageCapabilities()

    # ── Blank / skip ────────────────────────────────────────────────────────
    if (
        profile.char_count < settings.skip_char_count_threshold
        and profile.image_coverage < settings.skip_image_coverage_threshold
    ):
        caps.is_blank = True
        return caps  # nothing else applies to a blank page

    # ── Indic script ────────────────────────────────────────────────────────
    if profile.primary_script in _INDIC_SCRIPTS:
        caps.has_indic_script = True
        # Indic pages still contain scanned content — flag both so the plan
        # can route to the best available engine (printed OCR placeholder).
        caps.has_printed_scan = True
        logger.warning(
            "router.indic_engine_deferred",
            page_number=profile.page_number,
            primary_script=profile.primary_script,
        )
        return caps

    # ── Digital embedded text ───────────────────────────────────────────────
    if profile.has_text and profile.char_count > settings.digital_char_count_threshold:
        caps.has_digital_text = True
        caps.digital_confidence_hint = 0.97  # Docling's typical confidence on clean text

    # ── Scanned / rasterised content ────────────────────────────────────────
    # is_scanned=True means near-zero char_count AND high image_coverage.
    # Also flag printed_scan if image_coverage is substantial even when some
    # text is present (mixed page: typed text + scanned figure insert).
    if profile.is_scanned:
        caps.has_printed_scan = True
    elif profile.image_coverage > settings.capability_scan_supplement_image_threshold:
        # Significant image area even on a digital page → there's a scanned
        # region (figure, stamp, photo) worth running printed OCR over.
        caps.has_printed_scan = True

    # ── Handwriting signal ──────────────────────────────────────────────────
    # Heuristic proxy at Stage 1 (no VLM yet at this point):
    #   • scanned page with image coverage at or above the dual-OCR threshold →
    #     run both printed and handwriting engines and let the merge decide.
    #   • mixed content (text + image): may have annotation layer.
    if profile.is_scanned and profile.image_coverage >= settings.capability_dual_ocr_scan_threshold:
        # Coverage is high enough that it is worth running both engines.
        caps.has_handwriting = True
    elif profile.is_scanned:
        # Low-coverage scanned page — conservative: still flag handwriting so
        # VLM classification can confirm or override in Step 3.
        caps.has_handwriting = True
    elif is_mixed_content(profile):
        # Form with annotations: digital text engine + handwriting engine.
        caps.has_handwriting = True
        logger.info(
            "router.mixed_content_detected",
            page_number=profile.page_number,
            reason="digital_with_partial_image",
        )

    # ── Tables ──────────────────────────────────────────────────────────────
    # Note: _detect_tables() geometry check is still a stub in inspect.py
    # (always returns True when drawings exist).  We only flag has_tables
    # when has_vector_drawings=True as a proxy, not the stub result, to
    # avoid false-positives on every page with any vector element.
    if profile.has_tables and profile.has_vector_drawings:
        caps.has_tables = True

    # ── Figures / embedded images ───────────────────────────────────────────
    # Flag as figures when there is substantial image area but the page is NOT
    # fully a scan (i.e. the image is embedded content, not the whole page).
    if (
        profile.image_coverage > settings.capability_scan_supplement_image_threshold
        and not profile.is_scanned
    ):
        caps.has_figures = True

    logger.info(
        "router.capabilities_detected",
        page_number=profile.page_number,
        capabilities=caps.active_capabilities(),
    )
    return caps


def capabilities_from_classification(
    caps: PageCapabilities,
    classification: PageClassification,
    profile: PageProfile,
) -> PageCapabilities:
    """
    Enrich a PageCapabilities set with VLM classification results (Step B).
    Called only when route_from_profile() returned None (ambiguous pages).
    Returns a new PageCapabilities — does not mutate the input.
    """
    updated = caps.model_copy()
    updated.handwriting_pct_hint = classification.handwriting_pct

    # Override / add capabilities based on VLM's handwriting_pct
    if classification.handwriting_pct >= settings.handwriting_pct_handwritten_floor:
        updated.has_handwriting = True
        updated.has_printed_scan = False  # VLM is confident it's handwriting, not print
        logger.info(
            "router.vlm_enriched_capabilities",
            page_number=profile.page_number,
            added="has_handwriting",
            handwriting_pct=classification.handwriting_pct,
        )
    elif classification.handwriting_pct <= settings.handwriting_pct_scanned_ceiling:
        updated.has_printed_scan = True
        updated.has_handwriting = False
        logger.info(
            "router.vlm_enriched_capabilities",
            page_number=profile.page_number,
            added="has_printed_scan",
            handwriting_pct=classification.handwriting_pct,
        )
    else:
        # Dead zone (0.10–0.30): run both engines conservatively
        updated.has_printed_scan = True
        updated.has_handwriting = True
        logger.info(
            "router.vlm_dead_zone",
            page_number=profile.page_number,
            handwriting_pct=classification.handwriting_pct,
            resolution="run_both",
        )

    return updated


# ---------------------------------------------------------------------------
# Stage 1: engine plan builder
# ---------------------------------------------------------------------------

def build_engine_plan(caps: PageCapabilities) -> list[EngineTask]:
    """
    Convert a PageCapabilities set into an ordered list of EngineTask objects.

    Rules (Stage 1 — full-page, no region bounding boxes):
      • One task per applicable engine — engines do not repeat.
      • Priority determines execution order AND merge tiebreak winner.
        Lower priority number = runs first = wins when two engines
        extract the same text (dedup keeps the version from the
        lower-priority/higher-confidence engine).
      • skip is terminal — if is_blank, return a single skip task and nothing else.
      • vlm_transcribe is only planned as a fallback via the escalation ladder,
        NOT added here — it fires in pipeline.py when all plan engines fall below
        the confidence threshold.

    Returns an empty list only if somehow no capability applies (shouldn't
    happen after capabilities_from_profile, but callers must handle it).
    """
    if caps.is_blank:
        return [EngineTask(engine="skip", priority=0, reason="is_blank")]

    tasks: list[EngineTask] = []

    if caps.has_digital_text:
        tasks.append(EngineTask(
            engine="docling",
            priority=_PRI_DIGITAL,
            reason="has_digital_text",
        ))

    if caps.has_printed_scan and not caps.has_indic_script:
        tasks.append(EngineTask(
            engine="paddleocr_printed",
            priority=_PRI_PRINTED,
            reason="has_printed_scan",
        ))

    if caps.has_handwriting:
        tasks.append(EngineTask(
            engine="paddleocr_handwritten",
            priority=_PRI_HANDWRITTEN,
            reason="has_handwriting",
        ))

    # Indic script: best available is printed OCR as placeholder
    if caps.has_indic_script:
        tasks.append(EngineTask(
            engine="paddleocr_printed",
            priority=_PRI_PRINTED,
            reason="has_indic_script (engine_deferred)",
        ))

    # Sort by priority so pipeline.py can iterate in order without sorting itself
    tasks.sort(key=lambda t: t.priority)

    # Deduplicate engine names — shouldn't happen with current logic but
    # guards against double-adding the same engine if capabilities overlap.
    seen: set[str] = set()
    unique_tasks: list[EngineTask] = []
    for t in tasks:
        if t.engine not in seen:
            seen.add(t.engine)
            unique_tasks.append(t)

    # Enforce the per-page engine cap — drop lowest-priority (highest number)
    # engines first so the most important engines always run.
    cap = settings.capability_max_engines_per_page
    if len(unique_tasks) > cap:
        dropped = [t.engine for t in unique_tasks[cap:]]
        logger.warning(
            "router.engine_plan_capped",
            original_count=len(unique_tasks),
            cap=cap,
            dropped_engines=dropped,
        )
        unique_tasks = unique_tasks[:cap]

    logger.info(
        "router.engine_plan_built",
        capabilities=caps.active_capabilities(),
        plan=[t.engine for t in unique_tasks],
    )
    return unique_tasks
