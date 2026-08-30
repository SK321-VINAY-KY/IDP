from src.ai.layer1_routing.router import (
    capabilities_from_classification,
    build_engine_plan,
)
from src.ai.schemas.page import PageCapabilities, PageClassification, PageProfile


def test_capabilities_from_classification_dead_zone():
    """handwriting_pct in the dead zone should enable both printed and handwriting"""
    caps = PageCapabilities()
    cls = PageClassification(
        route="scanned",
        confidence=0.9,
        language_hint="en",
        handwriting_pct=0.20,  # dead-zone (between 0.10 and 0.30)
        noise_level=0.1,
        needs_preprocessing=[],
    )
    profile = PageProfile(
        page_number=1,
        has_text=True,
        char_count=150,
        image_coverage=0.15,
        has_tables=False,
        is_scanned=True,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=1,
        dpi_estimate=300,
    )

    updated = capabilities_from_classification(caps, cls, profile)
    assert updated.has_printed_scan is True
    assert updated.has_handwriting is True
    assert updated.handwriting_pct_hint == 0.20


def test_build_engine_plan_respects_max_engines(monkeypatch):
    """When the per-page engine cap is lower than available engines, lowest-priority engines are dropped."""
    # Temporarily set the cap to 2 to force dropping the lowest-priority engine
    monkeypatch.setattr("src.config.settings.settings.capability_max_engines_per_page", 2)

    caps = PageCapabilities(
        has_digital_text=True,
        has_printed_scan=True,
        has_handwriting=True,
    )

    plan = build_engine_plan(caps)
    engines = [t.engine for t in plan]

    # With a cap of 2, the highest-priority engines should be kept
    assert engines == ["docling", "paddleocr_printed"]
    assert len(engines) == 2
