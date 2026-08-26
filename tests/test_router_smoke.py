"""
File: test_router_smoke.py
Purpose: Smoke tests for routing table, escalation ladder, and Stage 1
         capability detection + engine plan builder. No real OCR/VLM calls.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (Stage 1 capability routing tests)
Deps: pytest
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.schemas.page import PageCapabilities, PageClassification, PageProfile
from src.ai.layer1_routing.router import (
    build_engine_plan,
    capabilities_from_classification,
    capabilities_from_profile,
    is_mixed_content,
    next_escalation_route,
    resolve_route_with_classification,
    route_from_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        handwriting_pct=0.5, noise_level=0.1, needs_preprocessing=[],
    )
    base.update(overrides)
    return PageClassification(**base)


def engine_names(plan) -> list[str]:
    return [t.engine for t in plan]


# ===========================================================================
# Existing tests — legacy routing table + escalation ladder
# ===========================================================================

def test_digital_route_clean():
    p = make_profile(has_text=True, char_count=500, image_coverage=0.0)
    assert route_from_profile(p) == "digital"
    print("PASS: clean digital page routes to 'digital'")


def test_blank_page_skip():
    p = make_profile(char_count=5, image_coverage=0.0)
    assert route_from_profile(p) == "skip"
    print("PASS: blank page routes to 'skip'")


def test_mixed_content_forces_step_b():
    p = make_profile(has_text=True, char_count=500, image_coverage=0.3)
    assert is_mixed_content(p) is True
    assert route_from_profile(p) is None
    print("PASS: mixed-content page forces Step B (None) instead of trusting digital")


def test_dead_zone_resolves_to_handwritten():
    p = make_profile(is_scanned=True)
    cls = make_classification(handwriting_pct=0.2)  # dead zone: 0.10–0.30
    route = resolve_route_with_classification(p, cls)
    assert route == "handwritten", f"expected 'handwritten', got {route!r}"
    print("PASS: dead-zone handwriting_pct=0.2 resolves conservatively to 'handwritten'")


def test_escalation_ladder_full():
    """
    Walk the full updated ladder: digital→scanned→handwritten→vlm_transcribe→None.
    Previously this test asserted handwritten→None (stale); updated to reflect
    the vlm_transcribe tier added in the Stage 1 upgrade.
    """
    assert next_escalation_route("digital", "broken_parse") == "scanned"
    assert next_escalation_route("scanned", "low_confidence") == "handwritten"
    assert next_escalation_route("handwritten", "still_low") == "vlm_transcribe"
    assert next_escalation_route("vlm_transcribe", "still_low") is None
    print("PASS: escalation ladder full walk digital→scanned→handwritten→vlm_transcribe→None")


def test_indic_script_detected():
    p = make_profile(primary_script="devanagari", char_count=100)
    assert route_from_profile(p) == "scanned"
    print("PASS: Indic script routes to placeholder 'scanned'")


# ===========================================================================
# Stage 1: capabilities_from_profile
# ===========================================================================

def test_caps_blank_page():
    p = make_profile(char_count=5, image_coverage=0.0)
    caps = capabilities_from_profile(p)
    assert caps.is_blank is True
    assert caps.has_digital_text is False
    assert caps.has_printed_scan is False
    assert caps.has_handwriting is False
    print("PASS: blank page → is_blank only")


def test_caps_pure_digital():
    """Clean digital PDF page: only has_digital_text should fire."""
    p = make_profile(has_text=True, char_count=800, image_coverage=0.0)
    caps = capabilities_from_profile(p)
    assert caps.has_digital_text is True
    assert caps.has_printed_scan is False
    assert caps.has_handwriting is False
    assert caps.is_blank is False
    print("PASS: clean digital page → has_digital_text only")


def test_caps_fully_scanned():
    """Fully scanned page: printed_scan + handwriting both fire (dual-OCR)."""
    p = make_profile(is_scanned=True, image_coverage=0.84, char_count=0)
    caps = capabilities_from_profile(p)
    assert caps.has_printed_scan is True
    assert caps.has_handwriting is True
    assert caps.has_digital_text is False
    print("PASS: fully scanned page → has_printed_scan + has_handwriting (dual-OCR)")


def test_caps_mixed_content():
    """Form with text + embedded scan → digital + handwriting."""
    p = make_profile(has_text=True, char_count=300, image_coverage=0.35)
    caps = capabilities_from_profile(p)
    assert caps.has_digital_text is True
    assert caps.has_handwriting is True  # mixed content annotation path
    print("PASS: mixed-content page → has_digital_text + has_handwriting")


def test_caps_indic_script():
    """Indic page → has_indic_script + has_printed_scan (placeholder engine)."""
    p = make_profile(primary_script="tamil", char_count=200,
                     image_coverage=0.0, has_text=True)
    caps = capabilities_from_profile(p)
    assert caps.has_indic_script is True
    assert caps.has_printed_scan is True
    assert caps.has_digital_text is False   # Indic branch returns early
    print("PASS: Indic script → has_indic_script + has_printed_scan")


def test_caps_digital_with_figure():
    """Digital page with embedded image (e.g. diagram): digital + figures + printed_scan."""
    p = make_profile(has_text=True, char_count=400, image_coverage=0.30)
    caps = capabilities_from_profile(p)
    assert caps.has_digital_text is True
    # image_coverage=0.30 > capability_scan_supplement_image_threshold(0.10)
    assert caps.has_printed_scan is True
    assert caps.has_figures is True
    print("PASS: digital page with figure → has_digital_text + has_printed_scan + has_figures")


# ===========================================================================
# Stage 1: capabilities_from_classification (VLM enrichment)
# ===========================================================================

def test_vlm_enrichment_handwriting():
    """VLM says high handwriting_pct → override to handwriting only."""
    p = make_profile(is_scanned=True, image_coverage=0.80)
    caps = capabilities_from_profile(p)
    # before VLM: both flags True (dual-OCR heuristic)
    assert caps.has_printed_scan is True
    assert caps.has_handwriting is True

    cls = make_classification(handwriting_pct=0.85)  # above 0.30 floor
    enriched = capabilities_from_classification(caps, cls, p)
    assert enriched.has_handwriting is True
    assert enriched.has_printed_scan is False  # VLM confident — drop printed
    assert enriched.handwriting_pct_hint == 0.85
    print("PASS: VLM high handwriting_pct overrides to handwriting-only")


def test_vlm_enrichment_printed():
    """VLM says low handwriting_pct → printed scan only."""
    p = make_profile(is_scanned=True, image_coverage=0.80)
    caps = capabilities_from_profile(p)

    cls = make_classification(handwriting_pct=0.05)  # below 0.10 ceiling
    enriched = capabilities_from_classification(caps, cls, p)
    assert enriched.has_printed_scan is True
    assert enriched.has_handwriting is False
    print("PASS: VLM low handwriting_pct overrides to printed-only")


def test_vlm_enrichment_dead_zone():
    """VLM dead zone (0.10–0.30) keeps both engines."""
    p = make_profile(is_scanned=True, image_coverage=0.80)
    caps = capabilities_from_profile(p)

    cls = make_classification(handwriting_pct=0.20)  # dead zone
    enriched = capabilities_from_classification(caps, cls, p)
    assert enriched.has_printed_scan is True
    assert enriched.has_handwriting is True
    print("PASS: VLM dead-zone keeps both printed + handwriting engines")


# ===========================================================================
# Stage 1: build_engine_plan
# ===========================================================================

def test_plan_blank():
    caps = PageCapabilities(is_blank=True)
    plan = build_engine_plan(caps)
    assert engine_names(plan) == ["skip"]
    print("PASS: blank page plan = [skip]")


def test_plan_digital_only():
    caps = PageCapabilities(has_digital_text=True)
    plan = build_engine_plan(caps)
    assert engine_names(plan) == ["docling"]
    print("PASS: pure digital plan = [docling]")


def test_plan_scanned_only():
    caps = PageCapabilities(has_printed_scan=True)
    plan = build_engine_plan(caps)
    assert engine_names(plan) == ["paddleocr_printed"]
    print("PASS: printed scan plan = [paddleocr_printed]")


def test_plan_handwritten_only():
    caps = PageCapabilities(has_handwriting=True)
    plan = build_engine_plan(caps)
    assert engine_names(plan) == ["paddleocr_handwritten"]
    print("PASS: handwriting plan = [paddleocr_handwritten]")


def test_plan_dual_ocr_scanned():
    """Fully scanned page gets both OCR engines in priority order."""
    caps = PageCapabilities(has_printed_scan=True, has_handwriting=True)
    plan = build_engine_plan(caps)
    names = engine_names(plan)
    assert "paddleocr_printed" in names
    assert "paddleocr_handwritten" in names
    # printed (priority 2) must come before handwritten (priority 3)
    assert names.index("paddleocr_printed") < names.index("paddleocr_handwritten")
    print("PASS: dual-OCR scan plan = [paddleocr_printed, paddleocr_handwritten] in order")


def test_plan_mixed_form():
    """Form with digital text + handwriting annotations."""
    caps = PageCapabilities(has_digital_text=True, has_handwriting=True)
    plan = build_engine_plan(caps)
    names = engine_names(plan)
    assert names == ["docling", "paddleocr_handwritten"]
    # docling (priority 1) before paddleocr_handwritten (priority 3)
    print("PASS: mixed form plan = [docling, paddleocr_handwritten] in order")


def test_plan_full_mixed():
    """Digital text + printed scan + handwriting: all three in priority order."""
    caps = PageCapabilities(
        has_digital_text=True, has_printed_scan=True, has_handwriting=True
    )
    plan = build_engine_plan(caps)
    names = engine_names(plan)
    assert names == ["docling", "paddleocr_printed", "paddleocr_handwritten"]
    print("PASS: full-mixed plan = [docling, paddleocr_printed, paddleocr_handwritten]")


def test_plan_no_duplicate_engines():
    """Indic script adds paddleocr_printed; so does has_printed_scan — must dedup."""
    caps = PageCapabilities(has_indic_script=True, has_printed_scan=True)
    plan = build_engine_plan(caps)
    names = engine_names(plan)
    assert names.count("paddleocr_printed") == 1
    print("PASS: indic + printed_scan deduped to single paddleocr_printed task")


def test_plan_priority_order():
    """Plans must always be sorted by ascending priority."""
    caps = PageCapabilities(
        has_digital_text=True, has_printed_scan=True, has_handwriting=True
    )
    plan = build_engine_plan(caps)
    priorities = [t.priority for t in plan]
    assert priorities == sorted(priorities)
    print("PASS: engine plan is sorted by ascending priority")


def test_plan_engine_cap(monkeypatch):
    """
    When more engines are requested than capability_max_engines_per_page,
    the lowest-priority engines are dropped.
    """
    import src.config.settings as settings_module
    monkeypatch.setattr(settings_module.settings, "capability_max_engines_per_page", 2)

    caps = PageCapabilities(
        has_digital_text=True, has_printed_scan=True, has_handwriting=True
    )
    plan = build_engine_plan(caps)
    assert len(plan) == 2
    # Should keep the two highest-priority (lowest number) engines
    names = engine_names(plan)
    assert "docling" in names           # priority 1 — must survive
    assert "paddleocr_printed" in names # priority 2 — must survive
    assert "paddleocr_handwritten" not in names  # priority 3 — dropped
    print("PASS: engine cap enforced — highest-priority engines kept, lowest dropped")


# ===========================================================================
# Stage 1: _merge_results (unit test without real OCR)
# ===========================================================================

def test_merge_single_engine():
    from src.ai.layer1_routing.pipeline import _merge_results
    from src.ai.schemas.page import EngineTask

    task = EngineTask(engine="docling", priority=1, reason="test")
    markdown, confidence, engines = _merge_results([(task, "line one\nline two", 0.95, 0.0)])
    assert markdown == "line one\nline two"
    assert confidence == 0.95
    assert engines == ["docling"]
    print("PASS: single-engine merge returns input unchanged")


def test_merge_dedup_exact():
    """Lines identical across engines should appear only once."""
    from src.ai.layer1_routing.pipeline import _merge_results
    from src.ai.schemas.page import EngineTask

    t1 = EngineTask(engine="paddleocr_printed",     priority=2, reason="test")
    t2 = EngineTask(engine="paddleocr_handwritten", priority=3, reason="test")
    results = [
        (t1, "Max Z = 2x1 + 3x2\nSubject to", 0.88, 0.0),
        (t2, "Max Z = 2x1 + 3x2\nExtra line", 0.86, 0.0),  # first line is a duplicate
    ]
    markdown, confidence, engines = _merge_results(results)
    lines = [ln for ln in markdown.split("\n") if ln.strip()]
    # "Max Z = 2x1 + 3x2" should appear exactly once
    assert lines.count("Max Z = 2x1 + 3x2") == 1
    # "Subject to" and "Extra line" should both be present
    assert "Subject to" in markdown
    assert "Extra line" in markdown
    assert set(engines) == {"paddleocr_printed", "paddleocr_handwritten"}
    print("PASS: duplicate lines deduped across engines; unique lines from both kept")


def test_merge_empty_results():
    from src.ai.layer1_routing.pipeline import _merge_results
    markdown, confidence, engines = _merge_results([])
    assert markdown == ""
    assert confidence == 0.0
    assert engines == []
    print("PASS: empty results → empty merge output")


def test_merge_confidence_weighted():
    """
    Engine contributing more unique chars gets more weight in confidence.
    Engine A: 3 unique lines (~60 chars) at conf=0.90
    Engine B: 1 unique line (~20 chars) at conf=0.50
    Expected weighted avg ≈ (0.90*60 + 0.50*20) / 80 = 64/80 = 0.80
    """
    from src.ai.layer1_routing.pipeline import _merge_results
    from src.ai.schemas.page import EngineTask

    t1 = EngineTask(engine="docling",            priority=1, reason="test")
    t2 = EngineTask(engine="paddleocr_printed",  priority=2, reason="test")
    results = [
        (t1, "Alpha line here\nBeta line here\nGamma line here", 0.90, 0.0),
        (t2, "Delta line only",                                  0.50, 0.0),
    ]
    markdown, confidence, engines = _merge_results(results)
    # Confidence should be between 0.50 and 0.90, weighted towards 0.90
    assert 0.70 < confidence < 0.92, f"unexpected confidence={confidence}"
    assert "docling" in engines
    assert "paddleocr_printed" in engines
    print(f"PASS: weighted confidence={confidence:.3f} (expected ~0.80)")


if __name__ == "__main__":
    # Run without pytest for quick manual verification
    test_digital_route_clean()
    test_blank_page_skip()
    test_mixed_content_forces_step_b()
    test_dead_zone_resolves_to_handwritten()
    test_escalation_ladder_full()
    test_indic_script_detected()

    test_caps_blank_page()
    test_caps_pure_digital()
    test_caps_fully_scanned()
    test_caps_mixed_content()
    test_caps_indic_script()
    test_caps_digital_with_figure()

    test_vlm_enrichment_handwriting()
    test_vlm_enrichment_printed()
    test_vlm_enrichment_dead_zone()

    test_plan_blank()
    test_plan_digital_only()
    test_plan_scanned_only()
    test_plan_handwritten_only()
    test_plan_dual_ocr_scanned()
    test_plan_mixed_form()
    test_plan_full_mixed()
    test_plan_no_duplicate_engines()
    test_plan_priority_order()
    # test_plan_engine_cap requires monkeypatch — run via pytest only

    test_merge_single_engine()
    test_merge_dedup_exact()
    test_merge_empty_results()
    test_merge_confidence_weighted()

    print("\nAll smoke tests passed.")
