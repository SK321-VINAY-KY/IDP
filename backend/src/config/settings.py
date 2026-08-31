"""
File: settings.py
Purpose: Central Pydantic settings for the full IDP pipeline.
         Layer 1+2 (routing + conversion) + Layer 3 (extraction) + API + Postgres.
Owner: engineer-a@idp-pilot, engineer-b@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-28 (merged Layer 1+2 and Layer 3 settings)
Deps: pydantic-settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_PATHS = (Path(".env"), _BACKEND_DIR / ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATHS,
        env_prefix="IDP_",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Layer 1+2 — VLM provider
    # -------------------------------------------------------------------------
    # Supported values: "ollama" (local), "gemini" (Google Gemini)
    llm_provider: str = "ollama"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434/v1"

    # Gemini
    gemini_base_url: str = ""
    gemini_api_key: str = ""

    # VLM model used for Layer 1 classification/transcription
    vlm_model_name: str = "qwen2.5vl:7b"

    # -------------------------------------------------------------------------
    # Layer 3 — Extraction LLM
    # -------------------------------------------------------------------------
    # "sarvam" or "ollama"
    extraction_backend: str = "sarvam"

    # Sarvam AI
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    sarvam_model_name: str = "sarvam-105b"
    sarvam_api_key: str = ""

    # Ollama extraction fallback
    extraction_model_name: str = "qwen2.5:7b"

    # Extraction retry
    max_extraction_retries: int = 2

    # -------------------------------------------------------------------------
    # PostgreSQL
    # -------------------------------------------------------------------------
    database_url: str = "postgresql://postgres:password@localhost:5432/idp"

    # -------------------------------------------------------------------------
    # PaddleOCR engine settings
    # -------------------------------------------------------------------------
    paddle_handwriting_det_db_thresh: float = 0.2

    # -------------------------------------------------------------------------
    # Routing thresholds (Layer 1)
    # -------------------------------------------------------------------------
    digital_char_count_threshold: int = 100
    scanned_char_count_threshold: int = 30
    scanned_image_coverage_threshold: float = 0.25
    skip_char_count_threshold: int = 20
    skip_image_coverage_threshold: float = 0.02
    handwriting_pct_scanned_ceiling: float = 0.10
    handwriting_pct_handwritten_floor: float = 0.30
    vlm_direct_extraction_confidence_threshold: float = 0.85

    # -------------------------------------------------------------------------
    # Escalation ladder (Layer 2)
    # -------------------------------------------------------------------------
    escalation_confidence_threshold: float = 0.70
    max_escalation_attempts: int = 1

    # -------------------------------------------------------------------------
    # Mixed-content page detection
    # -------------------------------------------------------------------------
    mixed_content_min_char_count: int = 100
    mixed_content_min_image_coverage: float = 0.02
    mixed_content_max_image_coverage: float = 0.85

    # -------------------------------------------------------------------------
    # Routing architecture
    # -------------------------------------------------------------------------
    # "single_engine" | "capability_based"
    routing_mode: str = "single_engine"

    # Capability-based routing thresholds
    capability_digital_only_char_threshold: int = 500
    capability_scan_supplement_image_threshold: float = 0.10
    capability_dual_ocr_scan_threshold: float = 0.25
    capability_low_confidence_floor: float = 0.50
    capability_max_engines_per_page: int = 3


settings = Settings()
