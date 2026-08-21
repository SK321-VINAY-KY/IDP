"""
File: settings.py
Purpose: Central Pydantic settings for Engineer A's pipeline (Layer 1 + Layer 2).
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (TrOCR removed, PaddleOCR handwriting threshold added)
Deps: pydantic
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IDP_")
    # --- LLM / VLM (Ollama, local, no API keys) ---
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    vlm_model_name: str = "qwen2-vl:2b"

    #extraction settings
    extraction_model_name: str = "qwen2.5:7b"
    summary_model_name: str = "qwen2.5:3b-instruct"
    short_doc_page_limit: int = 10
    page_summary_max_words: int = 25
    max_extraction_retries: int = 2
    # --- PaddleOCR engine settings ---
    # Handwriting mode: lower detection threshold so thinner/more irregular
    # handwriting strokes are not missed at the DBNet detection stage.
    # Default PaddleOCR det_db_thresh is 0.3; 0.2 catches more stroke fragments.
    # Tune against the eval set — lower values increase recall at cost of more
    # false-positive detections on noisy backgrounds.
    paddle_handwriting_det_db_thresh: float = 0.2

    # --- Routing thresholds (tunable without redeploy) ---
    digital_char_count_threshold: int = 100
    scanned_char_count_threshold: int = 30
    scanned_image_coverage_threshold: float = 0.25
    skip_char_count_threshold: int = 20
    skip_image_coverage_threshold: float = 0.02
    handwriting_pct_scanned_ceiling: float = 0.10
    handwriting_pct_handwritten_floor: float = 0.30

    # --- Escalation ladder ---
    # NOTE: After replacing TrOCR with PaddleOCR the confidence distribution
    # has shifted — PaddleOCR reports genuine per-word scores (typically 0.7–0.99
    # on legible text) rather than TrOCR's blob-level geometric-mean scores
    # (~0.2–0.6). Re-calibrate this threshold against the full eval set before
    # assuming 0.70 is still the right cutoff. Run the 13-page handwritten eval
    # and inspect the new distribution first.
    escalation_confidence_threshold: float = 0.70
    max_escalation_attempts: int = 1

    # --- Mixed-content page detection (see design discussion) ---
    mixed_content_min_char_count: int = 100
    mixed_content_min_image_coverage: float = 0.10
    mixed_content_max_image_coverage: float = 0.85


settings = Settings()
