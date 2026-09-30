"""
Central configuration. Everything is env-driven so the same code can run
against Ollama locally today and Bedrock in AWS later without code changes.
"""
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import urllib.parse

from dotenv import load_dotenv

# Load from CWD first, then schema_chatbot_v2 root and project root so env vars are found regardless of launch location
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


logger = logging.getLogger(__name__)

KNOWN_DEFAULT_JWT_SECRETS = {
    "idp-schema-pipeline-dev-secret-key-change-me",
    "changeme",
    "secret",
}

KNOWN_WEAK_PASSWORDS = {
    "password",
    "12345",
    "changeme",
    "admin",
    "postgres",
    "root",
    "secret",
}


def _check_db_url_password(url: str) -> tuple[bool, str]:
    if not url:
        return True, "DATABASE_URL is not set"
    try:
        parsed = urllib.parse.urlparse(url)
        pwd = parsed.password
        if not pwd:
            return True, "No password found in DATABASE_URL"
        if pwd.lower() in KNOWN_WEAK_PASSWORDS:
            return True, f"Weak/default database password '{pwd}' in DATABASE_URL"
        return False, ""
    except Exception as exc:
        return False, f"Could not parse DATABASE_URL: {exc}"


def resolve_app_env() -> str:
    """
    Resolves APP_ENV explicitly.
    If unset and in an AWS runtime (ECS metadata or Lambda function name),
    defaults to production (fail-closed).
    If unset and NOT in an AWS runtime, raises ValueError (dev must opt in explicitly).
    """
    env = os.getenv("IDP_APP_ENV") or os.getenv("APP_ENV")
    if env and env.strip():
        return env.strip().lower()

    is_aws = bool(
        os.getenv("ECS_CONTAINER_METADATA_URI")
        or os.getenv("ECS_CONTAINER_METADATA_URI_V4")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("AWS_EXECUTION_ENV")
    )
    if is_aws:
        return "production"

    raise ValueError(
        "APP_ENV environment variable is unset. It must be explicitly configured "
        "(e.g., APP_ENV=development or APP_ENV=production). Dev must opt in explicitly."
    )


@dataclass(frozen=True)
class Settings:
    # --- Environment ---
    app_env: str = field(default_factory=resolve_app_env)

    # Which LLM adapter to use: "bedrock" | "sarvam" | "ollama" | "mock"
    llm_provider: str = field(default_factory=lambda: os.getenv("IDP_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "bedrock")

    # --- Ollama ---
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    ollama_timeout_s: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_S", "60")))

    # --- Bedrock ---
    bedrock_region: str = field(default_factory=lambda: os.getenv("IDP_BEDROCK_REGION") or os.getenv("BEDROCK_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1")
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv(
            "IDP_BEDROCK_MODEL_ID",
            os.getenv("BEDROCK_MODEL_ID", "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"),
        )
    )
    bedrock_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("IDP_BEDROCK_REASONING_EFFORT") or os.getenv("BEDROCK_REASONING_EFFORT") or "low"
    )

    # --- Sarvam AI ---
    sarvam_api_key: str = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    sarvam_model: str = field(default_factory=lambda: os.getenv("SARVAM_MODEL", "sarvam-105b"))
    sarvam_base_url: str = field(default_factory=lambda: os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"))
    sarvam_timeout_s: float = field(default_factory=lambda: float(os.getenv("SARVAM_TIMEOUT_S", "60")))
    # --- Sarvam Document AI (used only for document-upload schema intake) ---
    sarvam_doc_ai_language: str = field(default_factory=lambda: os.getenv("SARVAM_DOC_AI_LANGUAGE", "en-IN"))
    sarvam_doc_ai_poll_interval_s: float = field(default_factory=lambda: float(os.getenv("SARVAM_DOC_AI_POLL_INTERVAL_S", "6")))
    sarvam_doc_ai_timeout_s: float = field(default_factory=lambda: float(os.getenv("SARVAM_DOC_AI_TIMEOUT_S", "120")))

    # --- Storage ---
    session_store: str = field(default_factory=lambda: os.getenv("SESSION_STORE", "memory"))  # memory | dynamodb (future)
    database_url: str = field(
        default_factory=lambda: os.getenv("IDP_DATABASE_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:password@localhost:5432/idp"
    )

    # --- Auth ---
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "idp-schema-pipeline-dev-secret-key-change-me"))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expire_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_EXPIRE_MINUTES", "1440")))  # 24 hours default
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "changeme"))

    # --- App ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        is_prod = self.app_env in ("production", "staging")
        weak_db, db_reason = _check_db_url_password(self.database_url)

        if is_prod:
            if not self.jwt_secret or self.jwt_secret in KNOWN_DEFAULT_JWT_SECRETS:
                raise ValueError("JWT_SECRET must be explicitly set and cannot use default value in production/staging.")
            if len(self.jwt_secret) < 32:
                raise ValueError(f"JWT_SECRET must be at least 32 characters in production/staging (got {len(self.jwt_secret)}).")
            if not self.admin_password or self.admin_password in KNOWN_WEAK_PASSWORDS:
                raise ValueError("ADMIN_PASSWORD must be explicitly set and cannot use a weak default (e.g. 'changeme') in production/staging.")
            if weak_db:
                raise ValueError(f"DATABASE_URL is insecure in production/staging: {db_reason}.")
        else:
            if not self.jwt_secret or self.jwt_secret in KNOWN_DEFAULT_JWT_SECRETS:
                logger.warning("Insecure/default JWT_SECRET in use. Set a secure secret in production.")
            elif len(self.jwt_secret) < 32:
                logger.warning("JWT_SECRET is shorter than 32 characters. Use a 32+ char secret in production.")
            if not self.admin_password or self.admin_password in KNOWN_WEAK_PASSWORDS:
                logger.warning("Default admin password '%s' in use. Set ADMIN_PASSWORD in production.", self.admin_password)
            if weak_db:
                logger.warning("Insecure database configuration: %s", db_reason)


settings = Settings()
