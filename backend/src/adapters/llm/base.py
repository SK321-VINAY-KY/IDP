"""
File: base.py
Purpose: Abstract LLMClient interface. Local Ollama implementation today;
         Bedrock implementation later swaps in behind this same interface.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: pydantic
"""
from abc import ABC, abstractmethod

from src.ai.schemas.page import PageClassification, VLMAnalysis


class LLMClient(ABC):
    """Every provider (Ollama today, Bedrock later) implements this interface.
    Layer 1/2 code must depend only on this — never on a concrete provider."""

    @abstractmethod
    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        """Vision classification for ambiguous pages: route, handwriting_pct, noise_level."""
        raise NotImplementedError

    def analyze_page(self, image_bytes: bytes, page_profile_hint: dict) -> VLMAnalysis:
        """Backward-compatible analysis for clients that only classify pages."""
        classification = self.classify_page(image_bytes, page_profile_hint)
        capabilities = {"handwriting"} if classification.handwriting_pct > 0.10 else {"ocr"}
        return VLMAnalysis(
            confidence=classification.confidence,
            detected_capabilities=capabilities,
            required_capabilities=capabilities,
            reason="legacy classification client; direct extraction unavailable",
        )

    @abstractmethod
    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        """Returns (markdown_text, confidence). Only called for the handwritten route
        when handwriting_engine != 'trocr'."""
        raise NotImplementedError
