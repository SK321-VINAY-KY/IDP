"""
File: page.py
Purpose: Shared Pydantic schemas — the contract between Engineer A (produces
         PageOutput) and Engineer B (consumes PageOutput). Changes require
         sign-off from both engineers.
Owner: engineer-a@idp-pilot, engineer-b@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (Stage 1 capability-based routing)
Deps: pydantic
"""
from typing import List

from pydantic import BaseModel, Field


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


# ---------------------------------------------------------------------------
# Stage 1: capability-based routing schemas
# ---------------------------------------------------------------------------


class PageCapabilities(BaseModel):
    """
    What a page *contains* — not a single label, but a set of detected
    capabilities. Multiple can be True simultaneously (e.g. a form with
    printed labels AND handwritten fill-ins).

    Produced by router.capabilities_from_profile() (Step A heuristics) and
    optionally enriched by router.capabilities_from_classification() (Step B VLM).

    Stage 1 note: capabilities operate on the full page — no region bounding
    boxes yet. Stage 2 will add spatial region maps per capability.
    """

    # Content-type capabilities
    has_digital_text: bool = False   # selectable embedded text → Docling
    has_printed_scan: bool = False   # rasterised printed text → PaddleOCR printed
    has_handwriting: bool = False    # cursive/manuscript → PaddleOCR handwriting
    has_tables: bool = False         # tabular grid structure (stub in Stage 1)
    has_figures: bool = False        # image/diagram regions (detected, not extracted)
    has_indic_script: bool = False   # Indic-family script (engine deferred)

    # Structural signals
    is_blank: bool = False           # page is effectively empty → skip

    # Per-capability confidence hints (0–1).  None = not assessed.
    digital_confidence_hint: float | None = None
    handwriting_pct_hint: float | None = None   # from VLM classification

    def active_capabilities(self) -> list[str]:
        """Return names of capabilities that are True — useful for logging."""
        caps = []
        for field in ("has_digital_text", "has_printed_scan", "has_handwriting",
                      "has_tables", "has_figures", "has_indic_script", "is_blank"):
            if getattr(self, field):
                caps.append(field)
        return caps


class EngineTask(BaseModel):
    """
    A single engine invocation in an engine plan.

    engine:   which converter to call
    priority: execution order (lower = runs first); also used as tiebreaker
              when merging overlapping text — lower-priority engine wins ties
              since it typically has higher intrinsic confidence on this page type
    reason:   the capability that triggered this task (for logging/audit)
    """

    engine: str   # "docling" | "paddleocr_printed" | "paddleocr_handwritten" | "vlm_transcribe" | "skip"
    priority: int
    reason: str


class PageOutput(BaseModel):
    """
    The contract handed to Engineer B's Layer 3. Every page, regardless of
    which engine(s) produced it, resolves to exactly this shape.

    Stage 1 change: engine_used (str) → engines_used (list[str]).
    Engineer B should treat this as an ordered list — first entry is the
    primary/highest-confidence engine; subsequent entries are supplementary.
    """

    page_number: int
    markdown: str
    engines_used: List[str]          # was: engine_used: str
    confidence: float = Field(ge=0.0, le=1.0)
    capabilities: List[str] = Field(default_factory=list)  # active capabilities that fired
    escalated: bool
    escalation_attempts: int
    low_confidence: bool             # terminal flag: all engines exhausted below threshold
