"""
File: types.py
Purpose: Shared vocabulary + data contracts for the Layer 1.5 probe
         (script / handwriting / table detection on scanned pages).
Deps: pydantic, stdlib
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Script vocabulary. The first nine mirror inspect.py's _SCRIPT_RANGES names so
# probe output stays compatible with PageProfile.primary_script.
SCRIPTS = {
    "latin", "devanagari", "tamil", "bengali", "gujarati", "gurmukhi",
    "kannada", "malayalam", "odia", "telugu",
    "arabic", "cyrillic", "greek", "thai", "hebrew",
    "chinese", "japanese", "korean",
    "other", "unknown",
}
INDIC = {"devanagari", "tamil", "bengali", "gujarati", "gurmukhi",
         "kannada", "malayalam", "odia", "telugu"}

# script -> PaddleOCR `lang` code. VERIFY against the PaddleOCR version you ship
# (scripts without a Paddle recognizer are simply absent -> routed to the VLM).
SCRIPT_TO_PADDLE_LANG: Dict[str, str] = {
    "latin": "en",
    "devanagari": "devanagari",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "ka",
    "arabic": "ar",
    "cyrillic": "cyrillic",
    "greek": "el",
    "thai": "th",
    "chinese": "ch",
    "japanese": "japan",
    "korean": "korean",
}

ROUTE_SKIP = "skip"
ROUTE_DOCLING = "docling"
ROUTE_PRINTED = "paddleocr_printed"
ROUTE_HANDWRITTEN = "paddleocr_handwritten"
ROUTE_VLM = "vlm_transcribe"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"IDP_PROBE_{name}", default)


@dataclass
class ProbeConfig:
    """All knobs, read from IDP_PROBE_* env vars so Lambda config changes need no redeploy."""

    # --- rendering ---
    page_image_dpi: int = 144            # matches both Paddle engines (Matrix 2.0)
    analysis_dpi: int = 100              # CV analysis resolution
    jpeg_quality: int = 82
    write_page_images: bool = True
    image_prefix: str = "pages"
    image_bucket: Optional[str] = None   # default: same bucket as the PDF

    # --- candidate selection ---
    digital_char_threshold: int = 100    # same meaning as settings.digital_char_count_threshold
    force_probe_all: bool = False

    # --- blank / table ---
    blank_ink_ratio: float = 0.0015
    table_min_h_lines: int = 3
    table_min_v_lines: int = 2
    table_min_intersections: int = 4

    # --- sampling ---
    n_samples: int = 3
    max_samples: int = 6                 # cap when samples disagree
    onnx_min_conf: float = 0.80
    vlm_enabled: bool = True
    min_script_conf: float = 0.60
    # Max sample observers (and thus VLM Bedrock calls) run in parallel during
    # Phase B. <=0 means "one thread per sampled page in the wave" (the prior
    # implicit behaviour). Applies to BOTH the initial sample wave and the
    # disagreement-widening second wave. Env: IDP_PROBE_VLM_CONCURRENCY.
    vlm_concurrency: int = 0
    # --- lipika line-wise observer gates (additional early-None path INSIDE the
    #     observer; do NOT change onnx_min_conf behaviour in probe.py) ---
    lipika_min_windows: int = 8          # fewer total windows -> observer returns None
    lipika_min_conf: float = 0.60        # mean-softmax top-1 below this -> None
    lipika_min_margin: float = 0.20      # top1-top2 margin below this -> None
    lipika_max_windows: int = 32         # cap; keep an even round-robin spread across lines

    # --- handwriting (flag-gated; re-added in Phase 2 item 4) ---
    vlm_for_handwriting: bool = True     # IDP_PROBE_VLM_FOR_HANDWRITING
    hw_threshold: float = 0.30           # latin page routes to handwritten engine at/above this

    # --- models ---
    onnx_model_path: Optional[str] = None
    onnx_labels_path: Optional[str] = None
    vlm_model_id: Optional[str] = None
    vlm_region: str = "ap-south-1"

    # --- cache / budget ---
    cache_enabled: bool = True
    cache_prefix: str = "probe-cache"
    cache_version: str = "v1"
    reserve_s: float = 45.0              # stop rendering when less time than this remains
    upload_workers: int = 8

    @classmethod
    def from_env(cls) -> "ProbeConfig":
        c = cls()
        c.page_image_dpi = int(_env("PAGE_IMAGE_DPI", str(c.page_image_dpi)))
        c.write_page_images = _env("WRITE_PAGE_IMAGES", "1") == "1"
        c.image_bucket = os.environ.get("IDP_PROBE_IMAGE_BUCKET") or None
        c.force_probe_all = _env("FORCE_PROBE_ALL", "0") == "1"
        c.n_samples = int(_env("N_SAMPLES", str(c.n_samples)))
        c.max_samples = int(_env("MAX_SAMPLES", str(c.max_samples)))
        c.vlm_enabled = _env("VLM_ENABLED", "1") == "1"
        c.min_script_conf = float(_env("MIN_SCRIPT_CONF", str(c.min_script_conf)))
        c.vlm_concurrency = int(_env("VLM_CONCURRENCY", str(c.vlm_concurrency)))
        c.lipika_min_windows = int(_env("LIPIKA_MIN_WINDOWS", str(c.lipika_min_windows)))
        c.lipika_min_conf = float(_env("LIPIKA_MIN_CONF", str(c.lipika_min_conf)))
        c.lipika_min_margin = float(_env("LIPIKA_MIN_MARGIN", str(c.lipika_min_margin)))
        c.lipika_max_windows = int(_env("LIPIKA_MAX_WINDOWS", str(c.lipika_max_windows)))
        c.vlm_for_handwriting = _env("VLM_FOR_HANDWRITING", "1") == "1"
        c.hw_threshold = float(_env("HW_THRESHOLD", str(c.hw_threshold)))
        c.onnx_model_path = os.environ.get("IDP_PROBE_ONNX_MODEL") or None
        c.onnx_labels_path = os.environ.get("IDP_PROBE_ONNX_LABELS") or None
        c.vlm_model_id = (
            os.environ.get("IDP_PROBE_VLM_MODEL_ID")
            or os.environ.get("IDP_BEDROCK_VLM_MODEL_ID")
            or os.environ.get("IDP_VLM_MODEL_NAME")
        )
        c.vlm_region = (
            os.environ.get("IDP_PROBE_VLM_REGION")
            or os.environ.get("IDP_BEDROCK_REGION")
            or os.environ.get("AWS_REGION", c.vlm_region)
        )
        c.cache_enabled = _env("CACHE", "1") == "1"
        c.reserve_s = float(_env("RESERVE_S", str(c.reserve_s)))
        return c


class SampleObs(BaseModel):
    """One script/handwriting observation from a sampled page."""
    page_number: int
    script: str = "unknown"
    script_confidence: float = 0.0
    mixed_scripts: bool = False
    has_table: Optional[bool] = None
    handwriting_prob: Optional[float] = None
    source: str = "none"          # onnx | vlm | none


class DocPrior(BaseModel):
    script: str = "unknown"
    script_confidence: float = 0.0
    mixed_scripts: bool = False
    handwriting_prob: Optional[float] = None
    heterogeneous: bool = False   # samples disagreed -> per-page nearest-sample inheritance matters
    source: str = "none"
    sampled_pages: List[int] = Field(default_factory=list)
    samples: List[SampleObs] = Field(default_factory=list)


class RouteChoice(BaseModel):
    route: str
    ocr_lang: Optional[str] = None
    use_structure: bool = False
    reason: str = ""


class PageProbe(BaseModel):
    page_number: int
    probed: bool = False
    status: str = "ok"            # ok | blank | not_needed | skipped_timeout | error
    route: str = ROUTE_PRINTED
    ocr_lang: Optional[str] = None
    use_structure: bool = False
    route_reason: str = ""
    script: str = "unknown"
    script_confidence: float = 0.0
    script_source: str = "none"   # sample | prior | embedded | none
    mixed_scripts: bool = False
    handwriting_prob: Optional[float] = None
    has_table: bool = False
    table_confidence: float = 0.0
    is_blank: bool = False
    ink_ratio: Optional[float] = None
    image_s3_bucket: Optional[str] = None
    image_s3_key: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
