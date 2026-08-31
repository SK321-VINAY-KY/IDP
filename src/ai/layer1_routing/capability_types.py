"""
File: capability_types.py
Purpose: Core types for capability-based routing — the Capability vocabulary,
         per-processor capability registry (mirrors the real engines in
         layer2_conversion/*.py), cost priority, and requirement/decision
         records (mandatory vs optional capabilities).
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: none (stdlib only)
"""
from enum import Enum
from typing import NamedTuple, Set, Dict


class Capability(str, Enum):
    TEXT_EXTRACTION = "text_extraction"    # native/selectable text (Docling)
    OCR = "ocr"                            # recognize text from pixels
    HANDWRITING = "handwriting"            # recognize handwritten strokes specifically
    TABLE_STRUCTURE = "table_structure"    # preserve row/column structure
    LAYOUT = "layout"                      # multi-region / reading-order understanding
    VISION_REASONING = "vision_reasoning"  # holistic image understanding (VLM)


class CapabilityRequirement(NamedTuple):
    capability: Capability
    mandatory: bool  # False = "desirable but not blocking" (design doc §9)


# What each processor in THIS codebase can actually do today — mirrors the
# real engines already implemented in layer2_conversion/*.py. Not a
# hypothetical future set; every key here corresponds to a working function.
PROCESSOR_CAPABILITIES: Dict[str, Set[Capability]] = {
    "docling": {
        Capability.TEXT_EXTRACTION,
        Capability.TABLE_STRUCTURE,
        Capability.LAYOUT,
    },
    "paddleocr_printed": {
        Capability.OCR,
    },
    "paddleocr_handwritten": {
        Capability.OCR,
        Capability.HANDWRITING,
    },
    "vlm_transcribe": {
        Capability.OCR,
        Capability.HANDWRITING,
        Capability.LAYOUT,
        Capability.VISION_REASONING,
    },
}

# Lower = cheaper/preferred. Mirrors the "cheap deterministic first, expensive
# model only when necessary" principle the escalation ladder already follows.
PROCESSOR_PRIORITY: Dict[str, int] = {
    "docling": 1,
    "paddleocr_printed": 1,
    "paddleocr_handwritten": 2,
    "vlm_transcribe": 3,
}


class RouteDecision(NamedTuple):
    processor: str
    matched_mandatory: Set[Capability]
    missing_optional: Set[Capability]
    reason: str
