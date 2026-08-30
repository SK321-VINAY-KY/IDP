"""
File: settings.py
Purpose: Central Pydantic settings for Engineer A's pipeline (Layer 1 + Layer 2).
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (Stage 1 capability-based routing thresholds)
Deps: pydantic-settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IDP_")

    # --- LLM / VLM (provider selection) ---
    # Set `llm_provider` to the concrete provider you want to use.
    # Supported values: "ollama" (local), "gemini" (Google Gemini via REST/proxy).
    llm_provider: str = "ollama"

    # Ollama (local) settings (kept for backwards compatibility)
    ollama_base_url: str = "http://localhost:11434/v1"

    # Gemini settings
    # Example: set `IDP_GEMINI_BASE_URL` to a proxy or leave empty to use
    # the official Google Generative API client when available.
    gemini_base_url: str = ""
    gemini_api_key: str = ""

    # Default VLM model name (provider-specific). For Ollama this was
    # `qwen2-vl:2b`; for Gemini use your chosen model like `models/gemini-1.5-mini`.
    # Use a Google Gemini model name by default when `llm_provider` is "gemini".
    vlm_model_name: str = "qwen2.5vl:7b"

    # --- PaddleOCR engine settings ---
    # Handwriting mode: lower detection threshold so thinner/more irregular
    # handwriting strokes are not missed at the DBNet detection stage.
    # Default PaddleOCR det_db_thresh is 0.3; 0.2 catches more stroke fragments.
    # Tune against the eval set — lower values increase recall at cost of more
    # false-positive detections on noisy backgrounds.
    paddle_handwriting_det_db_thresh: float = 0.2

    # --- Core routing thresholds (tunable without redeploy) ---
    digital_char_count_threshold: int = 100
    scanned_char_count_threshold: int = 30
    scanned_image_coverage_threshold: float = 0.25
    skip_char_count_threshold: int = 20
    skip_image_coverage_threshold: float = 0.02
    handwriting_pct_scanned_ceiling: float = 0.10
    handwriting_pct_handwritten_floor: float = 0.30

    # --- Escalation ladder ---
    # NOTE: After replacing TrOCR with PaddleOCR the confidence distribution
    # shifted — PaddleOCR reports genuine per-word scores (0.7–0.99 on legible
    # text). For testing you can raise this threshold to force the escalation
    # ladder to run the VLM fallback. Production default should be ~0.70.
    # For demo/testing we'll set a high threshold so pages will escalate.
    escalation_confidence_threshold: float = 0.70
    max_escalation_attempts: int = 1

    # --- Mixed-content page detection ---
    mixed_content_min_char_count: int = 100
    # NOTE (2026-08-20): lowered from 0.10 to 0.02. A small signature box or
    # a few handwritten form fields is often well under 10% of page area —
    # the old floor silently let those pages skip the VLM check entirely and
    # go straight to Docling, which drops handwritten content since it never
    # OCRs embedded images. Tune against the eval set: too low risks
    # triggering unnecessary VLM calls on pages with small logos/stamps.
    mixed_content_min_image_coverage: float = 0.02
    mixed_content_max_image_coverage: float = 0.85

    # --- Routing architecture (opt-in capability-based mode) ---
    # "single_engine"    — existing router.py behavior: one route string per
    #                      page, chosen via route_from_profile() /
    #                      resolve_route_with_classification(). DEFAULT —
    #                      no behavior change unless explicitly overridden.
    # "capability_based" — opt-in: capability_router.py detects the SET of
    #                      capabilities a page needs (not a single label) and
    #                      matches against PROCESSOR_CAPABILITIES. Bridges
    #                      back to the same route strings via
    #                      decision_to_pipeline_route(), so PageOutput and
    #                      the escalation ladder are unaffected either way.
    #                      Activate with: IDP_ROUTING_MODE=capability_based
    routing_mode: str = "single_engine"

    # -------------------------------------------------------------------------
    # Stage 1: capability-based routing settings
    # -------------------------------------------------------------------------

    # Minimum char count for a page to be considered purely digital.
    # Pages above this threshold with no significant image area go straight to
    # Docling and do NOT also run PaddleOCR (avoids wasted CPU on clean PDFs).
    capability_digital_only_char_threshold: int = 500

    # Image coverage threshold above which a digital page also gets a printed
    # OCR pass (covers the scanned figure / stamp / embedded image use case).
    # Below this value on a digital page, has_printed_scan stays False.
    # Matches mixed_content_min_image_coverage by default — tune separately
    # if clean digital pages with small logos are triggering unnecessary OCR.
    capability_scan_supplement_image_threshold: float = 0.10

    # When a scanned page has image_coverage above this value we treat it as
    # "fully scanned" and run BOTH printed and handwriting OCR engines on it,
    # merging at the line level. Below this value we rely on VLM classification
    # to decide which single OCR engine to use.
    # Setting this to 0.0 effectively disables the dual-OCR-on-scan behaviour;
    # setting it to 1.0 always runs dual OCR on every scanned page.
    capability_dual_ocr_scan_threshold: float = 0.25

    # Minimum merged confidence below which a multi-engine result is still
    # considered low_confidence (terminal flag for Engineer B).
    # Separate from escalation_confidence_threshold so you can tune them
    # independently: escalation fires earlier, low_confidence is the final flag.
    capability_low_confidence_floor: float = 0.50

    # Maximum number of engine tasks allowed in a single page plan.
    # Guards against runaway plans on pathological pages (e.g. a page that
    # somehow triggers every capability). With 3 engines at ~30s each on CPU,
    # a plan of 3 is already ~90s — raise only if hardware warrants it.
    capability_max_engines_per_page: int = 3


settings = Settings()
