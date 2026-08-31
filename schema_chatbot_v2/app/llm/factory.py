from __future__ import annotations

from app.config import settings
from app.llm.base import LLMAdapter


def get_llm_adapter(provider: str | None = None) -> LLMAdapter:
    provider = (provider or settings.llm_provider).lower()

    if provider == "ollama":
        from app.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter()

    if provider == "bedrock":
        from app.llm.bedrock_adapter import BedrockAdapter
        return BedrockAdapter()

    if provider == "sarvam":
        from app.llm.sarvam_adapter import SarvamAdapter
        return SarvamAdapter()

    if provider == "mock":
        from app.llm.mock_adapter import MockLLMAdapter
        return MockLLMAdapter()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
