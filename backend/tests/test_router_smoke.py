"""
File: test_router_smoke.py
Purpose: Smoke test for routing table + escalation ladder logic, no real
         OCR/VLM calls (mocked). Validates the dead-zone fix, mixed-content
         detection, and escalation cap terminate correctly.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.schemas.page import PageProfile, PageClassification
from src.ai.layer1_routing.router import (
    route_from_profile,
    resolve_route_with_classification,
    next_escalation_route,
    is_mixed_content,
)


def make_profile(**overrides) -> PageProfile:
    base = dict(
        page_number=1,
        has_text=False,
        char_count=0,
        image_coverage=0.0,
        has_tables=False,
        is_scanned=False,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=0,
        dpi_estimate=0,
    )
    base.update(overrides)
    return PageProfile(**base)


def test_digital_route_clean():
    p = make_profile(has_text=True, char_count=500, image_coverage=0.0)
    assert route_from_profile(p) == "digital"
    print("PASS: clean digital page routes to 'digital'")


def test_blank_page_skip():
    p = make_profile(char_count=5, image_coverage=0.0)
    assert route_from_profile(p) == "skip"
    print("PASS: blank page routes to 'skip'")


def test_mixed_content_forces_step_b():
    # digital-looking char count but partial image coverage -> should NOT
    # trust pure digital route, must fall through to VLM (None)
    p = make_profile(has_text=True, char_count=500, image_coverage=0.3)
    assert is_mixed_content(p) is True
    assert route_from_profile(p) is None
    print("PASS: mixed-content page forces Step B instead of trusting digital")


def test_dead_zone_resolves_to_handwritten():
    p = make_profile(is_scanned=True, primary_script="latin")
    classification = PageClassification(
        route="scanned",
        confidence=0.8,
        language_hint="en",
        handwriting_pct=0.2,  # dead zone: between 0.1 ceiling and 0.3 floor
        noise_level=0.1,
        needs_preprocessing=[],
    )
    route = resolve_route_with_classification(p, classification)
    assert route == "handwritten", f"expected conservative routing, got {route}"
    print(
        "PASS: dead-zone handwriting_pct=0.2 resolves conservatively to 'handwritten'"
    )


def test_escalation_ladder_terminates():
    # digital -> scanned -> handwritten -> vlm_transcribe -> None (terminal, no infinite loop)
    assert next_escalation_route("digital", "broken_parse") == "scanned"
    assert next_escalation_route("scanned", "low_confidence") == "handwritten"
    assert next_escalation_route("handwritten", "still_low") == "vlm_transcribe"
    assert next_escalation_route("vlm_transcribe", "still_low") is None
    print("PASS: escalation ladder terminates at 'vlm_transcribe' -> None, no infinite bounce")



def test_indic_script_detected():
    p = make_profile(primary_script="devanagari", char_count=100)
    route = route_from_profile(p)
    assert route == "scanned"  # deferred engine, placeholder route per build guide
    print(
        "PASS: Indic script routes to placeholder 'scanned' (engine deferred, not silently dropped)"
    )


if __name__ == "__main__":
    test_digital_route_clean()
    test_blank_page_skip()
    test_mixed_content_forces_step_b()
    test_dead_zone_resolves_to_handwritten()
    test_escalation_ladder_terminates()
    test_indic_script_detected()
    print("\nAll smoke tests passed.")
