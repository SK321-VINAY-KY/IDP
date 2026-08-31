"""
Central configuration. Everything is env-driven so the same code can run
against Ollama locally today and Bedrock in AWS later without code changes.
"""
import os
from dataclasses import dataclass

from pathlib import Path

from dotenv import load_dotenv

# Load from CWD first, then schema_chatbot_v2 root so env vars are found regardless of launch location
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    # Which LLM adapter to use: "ollama" | "bedrock" | "mock"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # --- Ollama ---
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_timeout_s: float = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

    # --- Bedrock ---
    bedrock_region: str = os.getenv("BEDROCK_REGION", "ap-south-1")
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # --- Sarvam AI ---
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_model: str = os.getenv("SARVAM_MODEL", "sarvam-105b")
    sarvam_base_url: str = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
    sarvam_timeout_s: float = float(os.getenv("SARVAM_TIMEOUT_S", "60"))
    # --- Sarvam Document AI (used only for document-upload schema intake) ---
    sarvam_doc_ai_language: str = os.getenv("SARVAM_DOC_AI_LANGUAGE", "en-IN")
    sarvam_doc_ai_poll_interval_s: float = float(os.getenv("SARVAM_DOC_AI_POLL_INTERVAL_S", "6"))
    sarvam_doc_ai_timeout_s: float = float(os.getenv("SARVAM_DOC_AI_TIMEOUT_S", "120"))

    # --- Storage ---
    session_store: str = os.getenv("SESSION_STORE", "memory")  # memory | dynamodb (future)

    # --- Auth ---
    jwt_secret: str = os.getenv("JWT_SECRET", "idp-schema-pipeline-dev-secret-key-change-me")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 hours default

    # --- App ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
