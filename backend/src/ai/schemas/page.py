"""
File: page.py
Purpose: Shared Pydantic schemas — the contract between Engineer A (produces
         PageOutput) and Engineer B (consumes PageOutput). Changes require
         sign-off from both engineers.
Owner: engineer-a@idp-pilot, engineer-b@idp-pilot
Created: 2026-08-19 | Deps: pydantic
"""

from pydantic import BaseModel, Field
from typing import List


class PageProfile(BaseModel):
    """Output of Layer 1 Step A — programmatic inspection (PyMuPDF, ~10ms)."""

    page_number: int
    has_text: bool
    char_count: int
    image_coverage: float = Field(ge=0.0, le=1.0)
    has_tables: bool
    is_scanned: bool
    has_vector_drawings: bool
    primary_script: str
    complexity_score: int = Field(ge=0, le=5)
    dpi_estimate: int


class PageClassification(BaseModel):
    """Output of Layer 1 Step B — light VLM classification (only for ambiguous pages)."""

    route: str  # "digital" | "scanned" | "handwritten" | "skip"
    confidence: float = Field(ge=0.0, le=1.0)
    language_hint: str
    handwriting_pct: float = Field(ge=0.0, le=1.0)
    noise_level: float = Field(ge=0.0, le=1.0)
    needs_preprocessing: List[str] = Field(default_factory=list)


class PageOutput(BaseModel):
    """
    The contract handed to Engineer B's Layer 3. Every page, regardless of
    which engine produced it, resolves to exactly this shape.
    """

    page_number: int
    markdown: str
    engine_used: str  # "docling" | "paddleocr" | "trocr" | "skip"
    confidence: float = Field(ge=0.0, le=1.0)
    escalated: bool
    escalation_attempts: int
    low_confidence: bool  # terminal flag: ladder exhausted without resolving
