"""
File: router.py
Purpose: Routing decision table + escalation ladder. Implements the fixed
         thresholds, mixed-content detection, and capped-retry terminal state
         agreed in design review.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: pydantic
"""

from src.ai.schemas.page import PageProfile, PageClassification
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"digital", "scanned", "handwritten", "vlm_transcribe", "skip"}


def is_mixed_content(profile: PageProfile) -> bool:
    """
    Fingerprint for 'typed form + handwritten annotation' style pages:
    substantial embedded text AND meaningful-but-not-total image coverage.
    Per design discussion, these route conservatively to the handwritten
    tier rather than being silently mis-routed to digital/scanned.
    """
    return (
        profile.char_count > settings.mixed_content_min_char_count
        and settings.mixed_content_min_image_coverage
        < profile.image_coverage
        < settings.mixed_content_max_image_coverage
    )


def route_from_profile(profile: PageProfile) -> str | None:
    """
    Programmatic-only routing (Step A). Returns a route if the decision is
    conclusive without a VLM call, or None if Step B (VLM classification) is
    needed to resolve ambiguity.
    """
    # Row 6: blank/skip
    if (
        profile.char_count < settings.skip_char_count_threshold
        and profile.image_coverage < settings.skip_image_coverage_threshold
    ):
        return "skip"

    # Row 4: Indic script detected directly (deferred engine — see build guide §9,
    # routed to 'scanned' as a placeholder rather than silently dropped)
    indic_scripts = {
        "devanagari",
        "tamil",
        "bengali",
        "gujarati",
        "gurmukhi",
        "kannada",
        "malayalam",
        "odia",
        "telugu",
    }
    if profile.primary_script in indic_scripts:
        logger.warning(
            "router.indic_engine_deferred",
            page_number=profile.page_number,
            primary_script=profile.primary_script,
        )
        return "scanned"

    # Row 1: digital text
    if profile.has_text and profile.char_count > settings.digital_char_count_threshold:
        if is_mixed_content(profile):
            logger.info(
                "router.mixed_content_detected",
                page_number=profile.page_number,
                reason="digital_with_partial_image",
            )
            return None  # force VLM classification rather than trust pure-digital
        return "digital"

    # Complexity override: dense/complex layouts always get a VLM opinion,
    # regardless of what the other signals suggest (previously an unused field —
    # now wired into the decision as agreed in design review).
    if profile.complexity_score >= 4:
        return None

    # Everything else (is_scanned=True, ambiguous handwriting_pct) needs Step B.
    return None


def resolve_route_with_classification(
    profile: PageProfile, classification: PageClassification
) -> str:
    """
    Step B: resolve using VLM classification when Step A was inconclusive.
    Implements the handwriting_pct dead-zone fix — no gap between the
    scanned ceiling and handwritten floor; anything in between routes
    conservatively to handwritten.
    """
    if classification.handwriting_pct >= settings.handwriting_pct_handwritten_floor:
        route = "handwritten"
    elif classification.handwriting_pct <= settings.handwriting_pct_scanned_ceiling:
        route = "scanned"
    else:
        # dead zone (previously undefined) — route conservatively
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
    The escalation ladder. Returns the next route to try, or None if the
    current route is already the top tier (terminal — caller must set
    low_confidence=True and pass the page through as-is, capped by
    MAX_ESCALATION_ATTEMPTS).

    Ladder (low -> high quality):
        digital -> scanned -> handwritten -> vlm_transcribe -> None
    "handwritten" = PaddleOCR with handwriting-tuned config (det_db_thresh lowered).
    "vlm_transcribe" = Ollama VLM full-page transcription; top tier, no further step.
    """
    ladder = {
        "digital": "scanned",  # broken digital parse -> re-render + OCR
        "scanned": "handwritten",  # low printed-OCR confidence -> handwriting engine
        "handwritten": "vlm_transcribe",  # low PaddleOCR confidence -> VLM top tier
        "vlm_transcribe": None,  # already top tier, terminal
    }
    next_route = ladder.get(current_route)
    logger.info(
        "router.escalation_decision",
        from_route=current_route,
        to_route=next_route,
        reason=reason,
    )
    return next_route
