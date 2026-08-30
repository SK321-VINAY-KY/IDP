"""
File: factory.py
Purpose: Instantiate the configured `LLMClient` implementation based on
`settings.llm_provider` so callers can obtain a provider without importing
provider-specific classes.

Usage: `from src.adapters.llm.factory import get_llm_client; client = get_llm_client()`
"""
from typing import Any

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_client() -> Any:
    """Return an `LLMClient` instance for the configured provider.

    Supported providers: 'ollama', 'gemini'.
    """
    provider = (settings.llm_provider or "").lower()
    if provider == "ollama":
        from src.adapters.llm.ollama_client import OllamaClient

        logger.info("llm.factory.selected", provider="ollama")
        return OllamaClient()

    if provider == "gemini":
        from src.adapters.llm.gemini_client import GeminiClient

        logger.info("llm.factory.selected", provider="gemini")
        return GeminiClient()

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider!r}")
