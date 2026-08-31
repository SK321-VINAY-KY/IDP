"""
File: test_capability_router.py
Purpose: Tests for capability_types.py + capability_router.py — the
         detector, the matcher, and the bridge back to pipeline route
         strings. Also asserts skip/Indic pre-checks stay in agreement
         with router.py's originals (duplicated logic, see
         capability_router.py comments on precheck_skip_or_indic).
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: pytest
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.schemas.page import PageProfile, PageClassification
from src.ai.layer1_routing.capability_types import (
    Capability,
    CapabilityRequirement,
    PROCESSOR_CAPABILITIES,
)
from src.ai.layer1_routing.capability_router import (
    CapabilityRouter,
    decision_to_pipeline_route,
    detect_required_capabilities,
    detect_required_capabilities_with_classification,
    precheck_skip_or_indic,
)
from src.ai.layer1_routing.router import route_from_profile


def make_profile(**overrides) -> PageProfile:
    base = dict(
        page_number=1, has_text=False, char_count=0, image_coverage=0.0,
        has_tables=False, is_scanned=False, has_vector_drawings=False,
        primary_script="latin", complexity_score=0, dpi_estimate=0,
    )
    base.update(overrides)
    return PageProfile(**base)


def make_classification(**overrides) -> PageClassification:
    base = dict(
        route="scanned", confidence=0.8, language_hint="en",
        handwriting_pct=0.0, noise_level=0.1, needs_preprocessing=[],
    )
    base.update(overrides)
    return PageClassification(**base)


# ---------------------------------------------------------------------------
# Detector: Step A only
# ---------------------------------------------------------------------------

def test_clean_digital_page_resolves_without_vlm():
    p = make_profile(has_text=True, char_count=500, image_coverage=0.0)
    reqs = detect_required_capabilities(p)
    assert reqs is not None, "clean digital page should not need Step B"
    caps = {r.capability for r in reqs if r.mandatory}
    assert Capability.TEXT_EXTRACTION in caps
    print("PASS: clean digital page resolves capabilities without a VLM call")


def test_mixed_content_signal_forces_step_b():
    p = make_profile(has_text=True, char_count=500, image_coverage=0.3)
    reqs = detect_required_capabilities(p)
    assert reqs is None, "mixed-content signal should force Step B, same as router.py"
    print("PASS: mixed-content page forces capability Step B, matching router.py")


def test_scanned_page_always_needs_step_b():
    p = make_profile(is_scanned=True, char_count=0, image_coverage=0.9)
    reqs = detect_required_capabilities(p)
    assert reqs is None, "Step A alone can't tell handwriting from clean scan"
    print("PASS: scanned page defers to Step B (can't confirm handwriting from Step A alone)")


# ---------------------------------------------------------------------------
# Detector: Step A + VLM
# ---------------------------------------------------------------------------

def test_text_extraction_is_optional_not_mandatory_at_step_b():
    """
    If TEXT_EXTRACTION were mandatory alongside OCR/HANDWRITING at Step B,
    no processor could ever satisfy the requirement set — no engine does
    both text-layer extraction AND pixel OCR in one call. This test guards
    against that regression.
    """
    p = make_profile(has_text=True, char_count=300, image_coverage=0.2)
    classification = make_classification(handwriting_pct=0.25)
    reqs = detect_required_capabilities_with_classification(p, classification)
    text_extraction_reqs = [r for r in reqs if r.capability == Capability.TEXT_EXTRACTION]
    assert len(text_extraction_reqs) == 1
    assert text_extraction_reqs[0].mandatory is False
    print("PASS: TEXT_EXTRACTION is optional at Step B, never mandatory")


def test_handwriting_present_requires_ocr_and_handwriting():
    p = make_profile(has_text=True, char_count=300, image_coverage=0.2)
    classification = make_classification(handwriting_pct=0.25)
    reqs = detect_required_capabilities_with_classification(p, classification)
    mandatory = {r.capability for r in reqs if r.mandatory}
    assert Capability.OCR in mandatory
    assert Capability.HANDWRITING in mandatory
    print("PASS: handwriting_pct above ceiling requires both OCR and HANDWRITING, mandatory")


# ---------------------------------------------------------------------------
# Router: matching
# ---------------------------------------------------------------------------

def test_router_selects_docling_for_pure_text_extraction():
    router = CapabilityRouter()
    reqs = [CapabilityRequirement(Capability.TEXT_EXTRACTION, mandatory=True)]
    decision = router.route(reqs)
    assert decision.processor == "docling"
    print("PASS: pure TEXT_EXTRACTION requirement selects docling")


def test_router_selects_paddleocr_printed_for_ocr_only():
    router = CapabilityRouter()
    reqs = [CapabilityRequirement(Capability.OCR, mandatory=True)]
    decision = router.route(reqs)
    assert decision.processor == "paddleocr_printed"
    print("PASS: OCR-only requirement selects the cheaper paddleocr_printed")


def test_router_selects_paddleocr_handwritten_for_ocr_plus_handwriting():
    router = CapabilityRouter()
    reqs = [
        CapabilityRequirement(Capability.OCR, mandatory=True),
        CapabilityRequirement(Capability.HANDWRITING, mandatory=True),
    ]
    decision = router.route(reqs)
    assert decision.processor == "paddleocr_handwritten"
    print("PASS: OCR+HANDWRITING selects paddleocr_handwritten over the pricier vlm_transcribe")


def test_router_falls_back_to_vlm_when_nothing_satisfies_mandatory():
    router = CapabilityRouter()
    reqs = [CapabilityRequirement(Capability.VISION_REASONING, mandatory=True)]
    decision = router.route(reqs)
    assert decision.processor == "vlm_transcribe"
    print("PASS: requirement only vlm_transcribe can satisfy correctly selects it")


def test_router_prefers_cheaper_when_multiple_satisfy():
    router = CapabilityRouter()
    # OCR alone: paddleocr_printed, paddleocr_handwritten, AND vlm_transcribe
    # all have OCR. Priority must pick the cheapest.
    reqs = [CapabilityRequirement(Capability.OCR, mandatory=True)]
    decision = router.route(reqs)
    assert decision.processor == "paddleocr_printed", (
        f"expected cheapest OCR-capable processor, got {decision.processor}"
    )
    print("PASS: cheapest-first priority respected when multiple processors satisfy mandatory caps")


# ---------------------------------------------------------------------------
# Bridge: decision -> pipeline route string
# ---------------------------------------------------------------------------

def test_bridge_maps_docling_to_digital():
    from src.ai.layer1_routing.capability_types import RouteDecision
    decision = RouteDecision(
        processor="docling", matched_mandatory=set(),
        missing_optional=set(), reason="x"
    )
    assert decision_to_pipeline_route(decision) == "digital"
    print("PASS: docling bridges to 'digital' route string")


def test_bridge_maps_paddleocr_printed_to_scanned():
    from src.ai.layer1_routing.capability_types import RouteDecision
    decision = RouteDecision(
        processor="paddleocr_printed", matched_mandatory=set(),
        missing_optional=set(), reason="x"
    )
    assert decision_to_pipeline_route(decision) == "scanned"
    print("PASS: paddleocr_printed bridges to 'scanned'")


def test_bridge_maps_paddleocr_handwritten_to_handwritten():
    from src.ai.layer1_routing.capability_types import RouteDecision
    decision = RouteDecision(
        processor="paddleocr_handwritten", matched_mandatory=set(),
        missing_optional=set(), reason="x"
    )
    assert decision_to_pipeline_route(decision) == "handwritten"
    print("PASS: paddleocr_handwritten bridges to 'handwritten'")


def test_bridge_maps_vlm_transcribe_unchanged():
    from src.ai.layer1_routing.capability_types import RouteDecision
    decision = RouteDecision(
        processor="vlm_transcribe", matched_mandatory=set(),
        missing_optional=set(), reason="x"
    )
    assert decision_to_pipeline_route(decision) == "vlm_transcribe"
    print("PASS: vlm_transcribe bridges unchanged (same name both sides)")


# ---------------------------------------------------------------------------
# Cross-mode agreement: skip/Indic pre-check must match router.py
# ---------------------------------------------------------------------------

def test_precheck_agrees_with_router_on_skip():
    """
    precheck_skip_or_indic() in capability_router.py is a miniature
    duplicate of router.route_from_profile()'s skip/Indic branch. This
    test asserts both agree on skip pages. If you change the skip
    threshold in router.py, update capability_router.py too.
    """
    p = make_profile(char_count=5, image_coverage=0.0)
    assert precheck_skip_or_indic(p) == "skip"
    assert route_from_profile(p) == "skip"
    print("PASS: capability-mode skip pre-check agrees with router.route_from_profile()")


def test_precheck_agrees_with_router_on_indic():
    """
    Same agreement test for Indic script placeholder routing.
    """
    p = make_profile(primary_script="devanagari", char_count=100)
    assert precheck_skip_or_indic(p) == "scanned"
    assert route_from_profile(p) == "scanned"
    print("PASS: capability-mode Indic pre-check agrees with router.route_from_profile()")


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

def test_every_registered_processor_has_at_least_one_capability():
    for processor, caps in PROCESSOR_CAPABILITIES.items():
        assert len(caps) > 0, f"{processor} has no declared capabilities"
    print("PASS: every processor in the registry declares at least one capability")


def test_processor_names_use_handwritten_not_handwriting():
    """
    Guard against the paddleocr_handwriting vs paddleocr_handwritten naming
    mismatch — the rest of the codebase (pipeline.py, router.py, tests) uses
    the 'handwritten' (past-tense) form. This test will catch any regression.
    """
    assert "paddleocr_handwritten" in PROCESSOR_CAPABILITIES
    assert "paddleocr_handwriting" not in PROCESSOR_CAPABILITIES
    print("PASS: PROCESSOR_CAPABILITIES uses 'paddleocr_handwritten' (past-tense, consistent)")


if __name__ == "__main__":
    test_clean_digital_page_resolves_without_vlm()
    test_mixed_content_signal_forces_step_b()
    test_scanned_page_always_needs_step_b()
    test_text_extraction_is_optional_not_mandatory_at_step_b()
    test_handwriting_present_requires_ocr_and_handwriting()
    test_router_selects_docling_for_pure_text_extraction()
    test_router_selects_paddleocr_printed_for_ocr_only()
    test_router_selects_paddleocr_handwritten_for_ocr_plus_handwriting()
    test_router_falls_back_to_vlm_when_nothing_satisfies_mandatory()
    test_router_prefers_cheaper_when_multiple_satisfy()
    test_bridge_maps_docling_to_digital()
    test_bridge_maps_paddleocr_printed_to_scanned()
    test_bridge_maps_paddleocr_handwritten_to_handwritten()
    test_bridge_maps_vlm_transcribe_unchanged()
    test_precheck_agrees_with_router_on_skip()
    test_precheck_agrees_with_router_on_indic()
    test_every_registered_processor_has_at_least_one_capability()
    test_processor_names_use_handwritten_not_handwriting()
    print("\nAll capability router tests passed.")
