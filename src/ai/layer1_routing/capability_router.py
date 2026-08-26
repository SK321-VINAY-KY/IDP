"""
File: capability_router.py
Purpose: Two-stage capability-based routing:
           1. detect_required_capabilities*() — what does this page need?
           2. CapabilityRouter.route()        — who can do it, cheapest first?
         This is an OPT-IN layer underneath the existing router.py.
         route_from_profile(), resolve_route_with_classification(), and
         next_escalation_route() in router.py are NOT modified — this module
         is purely additive. Activated by settings.routing_mode="capability_based".
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: none (stdlib only)
"""
from typing import List, Optional

from src.ai.layer1_routing.capability_types import (
    Capability,
    CapabilityRequirement,
    PROCESSOR_CAPABILITIES,
    PROCESSOR_PRIORITY,
    RouteDecision,
)
from src.ai.schemas.page import PageClassification, PageProfile
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_mixed_content_signal(profile: PageProfile) -> bool:
    """
    Mirrors router.is_mixed_content() exactly. Duplicated rather than
    imported to keep this module self-contained and avoid any risk of
    circular imports when router.py later imports from capability_router.py.

    ANY threshold change to is_mixed_content() in router.py MUST be mirrored
    here. Two tests in test_capability_router.py assert both copies stay in
    agreement on the same inputs — run those if you touch either copy:
        test_precheck_agrees_with_router_on_skip
        test_precheck_agrees_with_router_on_indic
    """
    return (
        profile.char_count > settings.mixed_content_min_char_count
        and settings.mixed_content_min_image_coverage
        < profile.image_coverage
        < settings.mixed_content_max_image_coverage
    )


# ---------------------------------------------------------------------------
# Step A: deterministic capability detection (no model call)
# ---------------------------------------------------------------------------

def detect_required_capabilities(
    profile: PageProfile,
) -> Optional[List[CapabilityRequirement]]:
    """
    Deterministic (Step A only) capability detection — no model call.

    Returns a list of CapabilityRequirements if the decision is conclusive,
    or None when the page is too ambiguous without a VLM opinion (mirrors
    router.py's existing "None -> Step B" pattern exactly).

    Note: skip and Indic-script decisions are handled by pipeline.py's
    _precheck_skip_or_indic() BEFORE this is called — they sit outside the
    capability vocabulary (skip needs no processor; the Indic engine is
    deferred per build guide §9).
    """
    requirements: List[CapabilityRequirement] = []
    has_typed_text = (
        profile.has_text and profile.char_count > settings.digital_char_count_threshold
    )

    if has_typed_text:
        requirements.append(
            CapabilityRequirement(Capability.TEXT_EXTRACTION, mandatory=True)
        )

    if profile.has_tables:
        # Mandatory only alongside confirmed typed text (a real digital table).
        # A scanned table still needs OCR first regardless of grid structure,
        # so TABLE_STRUCTURE stays optional/desirable rather than blocking.
        requirements.append(
            CapabilityRequirement(
                Capability.TABLE_STRUCTURE, mandatory=has_typed_text
            )
        )

    if has_typed_text and _is_mixed_content_signal(profile):
        # Can't tell what's in the image region without a VLM look.
        return None

    if profile.complexity_score >= 4:
        # Dense/unusual layout — Step A heuristics are unreliable.
        return None

    if has_typed_text:
        # Clean digital page, fully characterized.
        return requirements

    # No confirmed typed text → scanned/image territory. Step A cannot
    # distinguish clean-printed-scan from handwritten — OCR is clearly
    # needed but HANDWRITING is unknown → ambiguous, defer to Step B.
    return None


# ---------------------------------------------------------------------------
# Step A + VLM: full capability detection after classification
# ---------------------------------------------------------------------------

def detect_required_capabilities_with_classification(
    profile: PageProfile,
    classification: PageClassification,
) -> List[CapabilityRequirement]:
    """
    Full capability detection using Step A + VLM classification.
    Resolves the ambiguous cases detect_required_capabilities() punted on.

    IMPORTANT — TEXT_EXTRACTION is always OPTIONAL here, never mandatory.
    Rationale: once we've reached Step B the page has meaningful image content
    (that's why Step A punted). No single processor in PROCESSOR_CAPABILITIES
    does both Docling-style text-layer extraction AND pixel OCR in one call,
    so marking TEXT_EXTRACTION mandatory alongside OCR/HANDWRITING produces an
    unsatisfiable requirement set — CapabilityRouter.route() would always fall
    back to vlm_transcribe for every mixed page, which is wrong.
    TEXT_EXTRACTION stays as an optional signal so callers can observe
    "typed text is present" for logging/labeling without blocking the match.
    (Covered by test_text_extraction_is_optional_not_mandatory_at_step_b.)
    """
    requirements: List[CapabilityRequirement] = []
    has_typed_text = (
        profile.has_text and profile.char_count > settings.digital_char_count_threshold
    )

    if has_typed_text:
        # Optional — see docstring above.
        requirements.append(
            CapabilityRequirement(Capability.TEXT_EXTRACTION, mandatory=False)
        )

    if profile.has_tables:
        requirements.append(
            CapabilityRequirement(Capability.TABLE_STRUCTURE, mandatory=False)
        )

    # OCR is always mandatory on image pages that reach Step B.
    requirements.append(CapabilityRequirement(Capability.OCR, mandatory=True))

    if classification.handwriting_pct > settings.handwriting_pct_scanned_ceiling:
        # VLM sees meaningful handwriting — add HANDWRITING as mandatory so
        # paddleocr_handwritten beats paddleocr_printed in the matcher.
        requirements.append(
            CapabilityRequirement(Capability.HANDWRITING, mandatory=True)
        )

    if profile.complexity_score >= 4 or classification.noise_level > 0.5:
        # Complex layout or noisy page — LAYOUT is desirable (not blocking).
        requirements.append(
            CapabilityRequirement(Capability.LAYOUT, mandatory=False)
        )

    return requirements


# ---------------------------------------------------------------------------
# Capability matcher
# ---------------------------------------------------------------------------

class CapabilityRouter:
    """
    Matches a set of CapabilityRequirements against PROCESSOR_CAPABILITIES
    and returns the cheapest processor that satisfies all mandatory requirements.
    """

    def route(self, requirements: List[CapabilityRequirement]) -> RouteDecision:
        mandatory = {r.capability for r in requirements if r.mandatory}
        optional = {r.capability for r in requirements if not r.mandatory}

        candidates: list[tuple] = []
        for processor, caps in PROCESSOR_CAPABILITIES.items():
            missing_mandatory = mandatory - caps
            if not missing_mandatory:
                missing_optional = optional - caps
                candidates.append((processor, missing_optional))

        if not candidates:
            # No processor satisfies every mandatory capability — fall back to
            # the strongest generalist rather than fail the page outright.
            logger.warning(
                "capability_router.no_full_match",
                mandatory=[c.value for c in mandatory],
                fallback="vlm_transcribe",
            )
            return RouteDecision(
                processor="vlm_transcribe",
                matched_mandatory=set(),
                missing_optional=optional,
                reason="no processor satisfies all mandatory capabilities",
            )

        # Prefer fewest missing optional capabilities, then cheapest priority.
        candidates.sort(
            key=lambda c: (len(c[1]), PROCESSOR_PRIORITY.get(c[0], 99))
        )
        selected_processor, missing_optional = candidates[0]

        decision = RouteDecision(
            processor=selected_processor,
            matched_mandatory=mandatory,
            missing_optional=missing_optional,
            reason=(
                "best capability match"
                if not missing_optional
                else "capability match with missing optional"
            ),
        )

        logger.info(
            "capability_router.decision",
            processor=decision.processor,
            mandatory=[c.value for c in mandatory],
            optional=[c.value for c in optional],
            missing_optional=[c.value for c in missing_optional],
            reason=decision.reason,
        )
        return decision


# ---------------------------------------------------------------------------
# Bridge: RouteDecision → existing pipeline route string
# ---------------------------------------------------------------------------

def decision_to_pipeline_route(decision: RouteDecision) -> str:
    """
    Maps a RouteDecision processor name back to the route strings that
    pipeline.py's _run_engine_task() and PageOutput already understand:
        "digital" | "scanned" | "handwritten" | "vlm_transcribe"

    This is the seam that keeps capability routing non-breaking — pipeline.py,
    the escalation ladder, and PageOutput.engines_used all need zero changes
    when routing_mode="capability_based" is active.

    Note: "mixed" is NOT produced here. Both "mixed" and "handwritten" labels
    invoke the same paddleocr_handwritten engine; the mixed/handwritten
    distinction is a labeling decision that lives in router.py alongside the
    classification data it needs. The bridge only concerns itself with which
    engine to call, not what to name the result.
    """
    mapping = {
        "docling":               "digital",
        "paddleocr_printed":     "scanned",
        "paddleocr_handwritten": "handwritten",
        "vlm_transcribe":        "vlm_transcribe",
    }
    route = mapping.get(decision.processor)
    if route is None:
        raise ValueError(
            f"Unrecognized processor in RouteDecision: {decision.processor!r}"
        )
    return route
