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

    Supported provider: 'bedrock'.
    """
    from src.adapters.llm.bedrock_client import BedrockLLMClient

    provider = (settings.llm_provider or "bedrock").lower()
    logger.info("llm.factory.selected", provider=provider)
    return BedrockLLMClient()

