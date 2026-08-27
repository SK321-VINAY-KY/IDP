"""
File: pipeline.py
Purpose: Orchestrates inspect -> route -> convert -> escalate -> PageOutput.
         This is Engineer A's primary deliverable function — Engineer B's
         Layer 3 consumes its output directly, with no knowledge of what
         happened inside.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (handwritten -> PaddleOCR, vlm_transcribe tier)
Deps: pydantic
"""

from typing import Any

from src.adapters.llm.base import LLMClient
from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import (
    route_from_profile,
    resolve_route_with_classification,
    next_escalation_route,
)
from src.ai.layer2_conversion.digital import convert_digital_page
from src.ai.layer2_conversion.scanned import (
    convert_scanned_page,
    convert_handwritten_via_paddle,
)
from src.ai.schemas.page import PageOutput
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _run_engine(
    route: str, page_context: dict, llm_client: LLMClient
) -> tuple[str, float]:
    """
    Dispatch to the Layer 2 converter for a given route.

    Routes:
        digital        — extract embedded text directly from PDF
        scanned        — PaddleOCR printed config
        handwritten    — PaddleOCR handwriting-tuned config (lower det_db_thresh)
        vlm_transcribe — Ollama VLM full-page transcription (top tier)
        skip           — blank page, trivially certain
    """
    if route == "digital":
        return convert_digital_page(
            page_context["pdf_path"], page_context["page_number"]
        )

    if route == "scanned":
        return convert_scanned_page(
            page_context["image_array"], page_context["page_number"]
        )

    if route == "handwritten":
        # PaddleOCR's DBNet detector handles line segmentation internally —
        # no external segment_lines() call needed. Pass the numpy image array
        # directly; the engine was already loaded with handwriting-tuned settings.
        return convert_handwritten_via_paddle(
            page_context["image_array"],
            page_context["page_number"],
        )

    if route == "vlm_transcribe":
        # Top tier of the escalation ladder — Ollama VLM full-page transcription.
        # transcribe_handwriting() was previously unused (dead code); it is now
        # load-bearing as the ladder's ceiling.
        logger.info(
            "pipeline.vlm_transcribe_invoked",
            page_number=page_context.get("page_number"),
        )
        return llm_client.transcribe_handwriting(page_context["image_bytes"])

    if route == "skip":
        return "", 1.0  # blank page, confidence is trivially certain

    raise ValueError(f"Unknown route: {route}")


def process_page(
    page: Any,
    page_number: int,
    page_context: dict,
    llm_client: LLMClient,
    page_image_bytes: bytes,
) -> PageOutput:
    """
    Processes a single page through the full Layer 1 + Layer 2 flow, including
    the escalation ladder. Returns a PageOutput — the contract handed to
    Engineer B's Layer 3, regardless of how many engines this page bounced through.
    """
    # Ensure image_bytes is always available in page_context for vlm_transcribe
    page_context = {
        **page_context,
        "image_bytes": page_image_bytes,
        "page_number": page_number,
    }

    profile = inspect_page(page, page_number)

    # Step A: try to resolve route programmatically
    route = route_from_profile(profile)

    # Step B: fall back to VLM classification if Step A was inconclusive
    if route is None:
        classification = llm_client.classify_page(
            page_image_bytes,
            page_profile_hint=profile.model_dump(),
        )
        route = resolve_route_with_classification(profile, classification)

    escalation_attempts = 0
    escalated = False
    markdown, confidence = _run_engine(route, page_context, llm_client)

    # Escalation ladder: retry (capped by settings) if confidence is low
    while (
        confidence < settings.escalation_confidence_threshold
        and escalation_attempts < settings.max_escalation_attempts
    ):
        next_route = next_escalation_route(route, reason=f"confidence={confidence:.2f}")
        if next_route is None:
            logger.warning(
                "pipeline.escalation_terminal",
                page_number=page_number,
                route=route,
                confidence=confidence,
            )
            break

        escalation_attempts += 1
        escalated = True
        route = next_route
        markdown, confidence = _run_engine(route, page_context, llm_client)

        logger.info(
            "pipeline.escalation_attempt",
            page_number=page_number,
            attempt=escalation_attempts,
            new_route=route,
            new_confidence=round(confidence, 4),
        )

    low_confidence = confidence < settings.escalation_confidence_threshold

    output = PageOutput(
        page_number=page_number,
        markdown=markdown,
        engine_used=route,
        confidence=round(confidence, 4),
        escalated=escalated,
        escalation_attempts=escalation_attempts,
        low_confidence=low_confidence,
        primary_script=profile.primary_script,
        complexity_score=profile.complexity_score,
        has_images=profile.image_coverage > 0,
    )

    logger.info(
        "pipeline.page_complete",
        page_number=page_number,
        engine_used=output.engine_used,
        confidence=output.confidence,
        escalated=output.escalated,
        low_confidence=output.low_confidence,
    )
    return output


def process_document(pages: list[dict], llm_client: LLMClient) -> list[PageOutput]:
    """
    Processes every page in a document. Each entry in `pages` must supply
    the PyMuPDF page object, page_number, and the engine-specific context
    (pdf_path / image_array / image_bytes) needed for whichever route it ends up on.
    """
    outputs: list[PageOutput] = []
    for page_data in pages:
        output = process_page(
            page=page_data["page"],
            page_number=page_data["page_number"],
            page_context=page_data["context"],
            llm_client=page_data["llm_client"]
            if "llm_client" in page_data
            else llm_client,
            page_image_bytes=page_data["image_bytes"],
        )
        outputs.append(output)

    low_conf_count = sum(1 for o in outputs if o.low_confidence)
    logger.info(
        "pipeline.document_complete",
        page_count=len(outputs),
        low_confidence_pages=low_conf_count,
        escalated_pages=sum(1 for o in outputs if o.escalated),
    )
    return outputs
