from src.ai.layer1_routing.router import (
    capabilities_from_profile,
    capabilities_from_classification,
    build_engine_plan,
)
from src.ai.schemas.page import PageProfile, PageClassification
from src.adapters.llm.factory import get_llm_client


llm_client = get_llm_client()
print(f"Using LLM provider: {type(llm_client).__name__}")


def show(profile, classification=None, name="sample"):
    print(f"--- {name} ---")
    caps = capabilities_from_profile(profile)
    print("From profile:", caps.model_dump())
    if classification is not None:
        enriched = capabilities_from_classification(caps, classification, profile)
        print("After VLM classification:", enriched.model_dump())
        plan = build_engine_plan(enriched)
    else:
        plan = build_engine_plan(caps)
    print("Engine plan:", [p.model_dump() for p in plan])
    print()


def main():
    # Blank page
    p_blank = PageProfile(
        page_number=1,
        has_text=False,
        char_count=0,
        image_coverage=0.0,
        has_tables=False,
        is_scanned=False,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=0,
        dpi_estimate=300,
    )

    # Clean digital page
    p_digital = PageProfile(
        page_number=2,
        has_text=True,
        char_count=600,
        image_coverage=0.0,
        has_tables=False,
        is_scanned=False,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=1,
        dpi_estimate=300,
    )

    # Scanned printed page
    p_scanned = PageProfile(
        page_number=3,
        has_text=False,
        char_count=10,
        image_coverage=0.95,
        has_tables=False,
        is_scanned=True,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=2,
        dpi_estimate=300,
    )

    # Mixed digital + figures (should add printed OCR supplement)
    p_mixed = PageProfile(
        page_number=4,
        has_text=True,
        char_count=200,
        image_coverage=0.2,
        has_tables=False,
        is_scanned=False,
        has_vector_drawings=True,
        primary_script="latin",
        complexity_score=2,
        dpi_estimate=300,
    )

    # Indic script (deferred to printed OCR)
    p_indic = PageProfile(
        page_number=5,
        has_text=True,
        char_count=50,
        image_coverage=0.9,
        has_tables=False,
        is_scanned=True,
        has_vector_drawings=False,
        primary_script="devanagari",
        complexity_score=1,
        dpi_estimate=300,
    )

    # Ambiguous scanned page — show dead-zone classification behavior
    p_amb = PageProfile(
        page_number=6,
        has_text=False,
        char_count=20,
        image_coverage=0.30,
        has_tables=False,
        is_scanned=True,
        has_vector_drawings=False,
        primary_script="latin",
        complexity_score=2,
        dpi_estimate=300,
    )
    cls_dead = PageClassification(
        route="scanned",
        confidence=0.85,
        language_hint="en",
        handwriting_pct=0.20,
        noise_level=0.1,
        needs_preprocessing=[],
    )

    show(p_blank, name="Blank page")
    show(p_digital, name="Clean digital page")
    show(p_scanned, name="Scanned printed page")
    show(p_mixed, name="Mixed digital+figures")
    show(p_indic, name="Indic script page")
    show(p_amb, classification=cls_dead, name="Ambiguous scanned (dead-zone VLM)")


if __name__ == "__main__":
    main()
