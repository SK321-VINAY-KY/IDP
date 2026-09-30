"""
File: settings.py
Purpose: Central Pydantic settings for Engineer A's pipeline (Layer 1 + Layer 2).
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (Stage 1 capability-based routing thresholds)
Deps: pydantic-settings
"""
import os
from pathlib import Path
from typing import Any, Optional
import dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_DIR = Path(__file__).resolve().parents[2]
_ROOT_ENV = _ROOT_DIR / ".env"
dotenv.load_dotenv(_ROOT_ENV, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[_ROOT_ENV, ".env"],
        env_prefix="IDP_",
        extra="ignore",
    )

    # --- Environment ---
    app_env: str = "development"

    # --- LLM / VLM (provider selection) ---
    # Set `llm_provider` to the concrete provider you want to use.
    # Supported values: "bedrock", "sarvam".
    llm_provider: str = "bedrock"

    # Bedrock settings
    bedrock_region: str = "ap-south-1"
    bedrock_model_id: str = "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
    bedrock_reasoning_effort: str = "low"
    bedrock_vlm_model_id: str = "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/lmbukv3mwnhm"

    # Ollama (local) settings (kept for backwards compatibility)
    ollama_base_url: str = "http://localhost:11434/v1"

    # Gemini settings
    # Example: set `IDP_GEMINI_BASE_URL` to a proxy or leave empty to use
    # the official Google Generative API client when available.
    gemini_base_url: str = ""
    gemini_api_key: str = ""

    # Default VLM model name (provider-specific).
    # For Bedrock, points to the Qwen3-VL-235B-A22B application inference profile.
    vlm_model_name: str = "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/lmbukv3mwnhm"

    # --- Layer 3 — Extraction LLM (Bedrock / Sarvam) ---
    extraction_backend: str = "bedrock"
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    sarvam_model_name: str = "sarvam-105b"
    sarvam_api_key: str = ""
    sarvam_timeout_s: float = 180.0
    sarvam_reasoning_effort: Optional[str] = None
    extraction_model_name: str = "qwen2.5:7b"
    summary_model_name: str = "qwen2.5:7b"
    max_extraction_retries: int = 2
    layer3_strategy: str = "graph_memory_concurrent"  # "graph_memory_concurrent" (default), "graph_memory", or "page_scan"
    graph_concurrency_limit: int = 8
    graph_anchor_max_k: int = 5
    graph_anchor_types: list[str] = []
    page_concurrency_limit: int = 25

    # --- PostgreSQL Storage ---
    database_url: str = "postgresql://postgres:password@localhost:5432/idp"

    @field_validator("page_concurrency_limit")
    @classmethod
    def _validate_page_concurrency_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page_concurrency_limit must be >= 1")
        return v

    @field_validator("graph_concurrency_limit")
    @classmethod
    def _validate_concurrency_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("graph_concurrency_limit must be >= 1")
        return v

    @field_validator("graph_anchor_max_k")
    @classmethod
    def _validate_anchor_max_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("graph_anchor_max_k must be >= 1")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fallback_unprefixed_env(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("page_concurrency_limit"):
                raw_page_limit = os.getenv("IDP_PAGE_CONCURRENCY_LIMIT") or os.getenv("PAGE_CONCURRENCY_LIMIT")
                if raw_page_limit:
                    data["page_concurrency_limit"] = int(raw_page_limit)
            if not data.get("escalation_confidence_threshold"):
                raw_thresh = os.getenv("IDP_ESCALATION_CONFIDENCE_THRESHOLD") or os.getenv("ESCALATION_CONFIDENCE_THRESHOLD")
                if raw_thresh:
                    data["escalation_confidence_threshold"] = float(raw_thresh)
            if not data.get("max_escalation_attempts"):
                raw_max_esc = os.getenv("IDP_MAX_ESCALATION_ATTEMPTS") or os.getenv("MAX_ESCALATION_ATTEMPTS")
                if raw_max_esc:
                    data["max_escalation_attempts"] = int(raw_max_esc)
            if not data.get("layer3_strategy"):
                data["layer3_strategy"] = os.getenv("IDP_LAYER3_STRATEGY") or os.getenv("LAYER3_STRATEGY") or "graph_memory_concurrent"
            if not data.get("graph_concurrency_limit"):
                raw_limit = os.getenv("IDP_GRAPH_CONCURRENCY_LIMIT") or os.getenv("GRAPH_CONCURRENCY_LIMIT")
                if raw_limit:
                    data["graph_concurrency_limit"] = int(raw_limit)
            if not data.get("graph_anchor_max_k"):
                raw_k = os.getenv("IDP_GRAPH_ANCHOR_MAX_K") or os.getenv("GRAPH_ANCHOR_MAX_K")
                if raw_k:
                    data["graph_anchor_max_k"] = int(raw_k)
            if not data.get("graph_anchor_types"):
                raw_anchors = os.getenv("IDP_GRAPH_ANCHOR_TYPES") or os.getenv("GRAPH_ANCHOR_TYPES")
                if raw_anchors:
                    if raw_anchors.startswith("["):
                        import json
                        data["graph_anchor_types"] = json.loads(raw_anchors)
                    else:
                        data["graph_anchor_types"] = [a.strip() for a in raw_anchors.split(",") if a.strip()]
            if not data.get("sarvam_api_key"):
                data["sarvam_api_key"] = os.getenv("IDP_SARVAM_API_KEY") or os.getenv("SARVAM_API_KEY") or ""
            if not data.get("sarvam_base_url"):
                data["sarvam_base_url"] = os.getenv("IDP_SARVAM_BASE_URL") or os.getenv("SARVAM_BASE_URL") or "https://api.sarvam.ai/v1"
            if not data.get("sarvam_model_name"):
                data["sarvam_model_name"] = os.getenv("IDP_SARVAM_MODEL_NAME") or os.getenv("SARVAM_MODEL") or "sarvam-105b"
            if not data.get("sarvam_timeout_s"):
                raw_timeout = os.getenv("IDP_SARVAM_TIMEOUT_S") or os.getenv("SARVAM_TIMEOUT_S")
                if raw_timeout:
                    data["sarvam_timeout_s"] = float(raw_timeout)
            if "sarvam_reasoning_effort" not in data or data.get("sarvam_reasoning_effort") is None:
                data["sarvam_reasoning_effort"] = os.getenv("IDP_SARVAM_REASONING_EFFORT") or os.getenv("SARVAM_REASONING_EFFORT") or None
            if not data.get("bedrock_region"):
                data["bedrock_region"] = os.getenv("IDP_BEDROCK_REGION") or os.getenv("BEDROCK_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"
            if not data.get("bedrock_model_id"):
                data["bedrock_model_id"] = os.getenv("IDP_BEDROCK_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID") or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
            if not data.get("bedrock_reasoning_effort"):
                data["bedrock_reasoning_effort"] = os.getenv("IDP_BEDROCK_REASONING_EFFORT") or os.getenv("BEDROCK_REASONING_EFFORT") or "low"
            if not data.get("bedrock_vlm_model_id"):
                data["bedrock_vlm_model_id"] = (
                    os.getenv("IDP_BEDROCK_VLM_MODEL_ID")
                    or os.getenv("BEDROCK_VLM_MODEL_ID")
                    or os.getenv("VLM_MODEL_ID")
                    or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/lmbukv3mwnhm"
                )
            if not data.get("vlm_model_name") or data.get("vlm_model_name") == "qwen2.5vl:7b":
                data["vlm_model_name"] = (
                    os.getenv("IDP_VLM_MODEL_NAME")
                    or os.getenv("VLM_MODEL_NAME")
                    or data["bedrock_vlm_model_id"]
                )
            if not data.get("extraction_backend"):
                data["extraction_backend"] = os.getenv("IDP_EXTRACTION_BACKEND") or os.getenv("EXTRACTION_BACKEND") or "bedrock"
            if not data.get("llm_provider"):
                data["llm_provider"] = os.getenv("IDP_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "bedrock"
            if not data.get("database_url") or data.get("database_url") == "postgresql://postgres:password@localhost:5432/idp":
                env_db = os.getenv("IDP_DATABASE_URL") or os.getenv("DATABASE_URL")
                if env_db:
                    data["database_url"] = env_db
            env_routing = os.getenv("IDP_ROUTING_MODE") or os.getenv("ROUTING_MODE")
            if env_routing:
                data["routing_mode"] = env_routing.strip()
            if not data.get("paddle_lambda_region"):
                data["paddle_lambda_region"] = (
                    os.getenv("IDP_PADDLE_LAMBDA_REGION")
                    or os.getenv("PADDLE_LAMBDA_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "ap-south-1"
                )
            if not data.get("paddle_printed_lambda_function"):
                data["paddle_printed_lambda_function"] = (
                    os.getenv("IDP_PADDLE_PRINTED_LAMBDA_FUNCTION")
                    or os.getenv("PADDLE_PRINTED_LAMBDA_FUNCTION")
                    or "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-printed"
                )
            if not data.get("paddle_handwritten_lambda_function"):
                data["paddle_handwritten_lambda_function"] = (
                    os.getenv("IDP_PADDLE_HANDWRITTEN_LAMBDA_FUNCTION")
                    or os.getenv("PADDLE_HANDWRITTEN_LAMBDA_FUNCTION")
                    or "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-handwritten"
                )
            if "paddle_prefer_lambda" not in data or data.get("paddle_prefer_lambda") is None:
                val = os.getenv("IDP_PADDLE_PREFER_LAMBDA") or os.getenv("PADDLE_PREFER_LAMBDA")
                if val is not None:
                    data["paddle_prefer_lambda"] = val.lower() in ("true", "1", "yes")
                else:
                    data["paddle_prefer_lambda"] = True
            if not data.get("app_env"):
                env = os.getenv("IDP_APP_ENV") or os.getenv("APP_ENV")
                if env and env.strip():
                    data["app_env"] = env.strip().lower()
                elif (
                    os.getenv("ECS_CONTAINER_METADATA_URI")
                    or os.getenv("ECS_CONTAINER_METADATA_URI_V4")
                    or os.getenv("AWS_EXECUTION_ENV")
                ):
                    data["app_env"] = "production"
                else:
                    data["app_env"] = "development"
        return data

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        if self.app_env in ("production", "staging") and not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            weak_passwords = {"password", "12345", "changeme", "admin", "postgres", "root", "secret"}
            import urllib.parse
            parsed = urllib.parse.urlparse(self.database_url)
            pwd = parsed.password
            if not pwd or pwd.lower() in weak_passwords:
                raise ValueError(f"Insecure/default database password in DATABASE_URL for {self.app_env}: '{pwd}'")
        return self

    # --- PaddleOCR engine settings ---
    # Handwriting mode: lower detection threshold so thinner/more irregular
    # handwriting strokes are not missed at the DBNet detection stage.
    # Default PaddleOCR det_db_thresh is 0.3; 0.2 catches more stroke fragments.
    # Tune against the eval set — lower values increase recall at cost of more
    # false-positive detections on noisy backgrounds.
    paddle_handwriting_det_db_thresh: float = 0.2

    # --- PaddleOCR AWS Lambda offload ---
    paddle_prefer_lambda: bool = True
    paddle_lambda_region: str = "ap-south-1"
    paddle_printed_lambda_function: str = "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-printed"
    paddle_handwritten_lambda_function: str = "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-handwritten"

    # --- Core routing thresholds (tunable without redeploy) ---
    digital_char_count_threshold: int = 100
    scanned_char_count_threshold: int = 30
    scanned_image_coverage_threshold: float = 0.25
    skip_char_count_threshold: int = 20
    skip_image_coverage_threshold: float = 0.02
    handwriting_pct_scanned_ceiling: float = 0.10
    handwriting_pct_handwritten_floor: float = 0.30
    vlm_direct_extraction_confidence_threshold: float = 0.85
    enable_vlm_direct: bool = False  # Disabled: pages route to Paddle first; VLM is only invoked on failure or confidence < 0.75

    # --- Escalation ladder ---
    # Pages first run OCR (PaddleOCR). If PaddleOCR fails or confidence < 0.75,
    # the escalation ladder invokes the VLM fallback (vlm_transcribe).
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


class LambdaSettings(Settings):
    """
    Dedicated configuration for stateless AWS Lambda handlers (e.g. Layer 1 inspection & Layer 2 engines).
    Stateless Lambda handlers do not use or require the PostgreSQL database, so database_url
    production checks are skipped specifically for Lambda handler configurations.
    """
    @model_validator(mode="after")
    def _validate_secrets(self) -> "LambdaSettings":
        # Stateless Lambda handler config: database_url is not required or validated
        return self


settings = Settings()

