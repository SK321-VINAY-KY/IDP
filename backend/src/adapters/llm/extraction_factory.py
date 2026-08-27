"""
File: extraction_factory.py
Purpose: Return the configured ExtractionLLMClient for Layer 3.
Owner: genai-platform@shellkode
Created: 2026-08-20
"""
from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.config.settings import settings


def get_extraction_client() -> ExtractionLLMClient:
    if settings.extraction_backend == "sarvam":
        from src.adapters.llm.sarvam_client import SarvamExtractionClient
        return SarvamExtractionClient()
    else:
        from src.adapters.llm.extraction_client import OllamaExtractionClient
        return OllamaExtractionClient()