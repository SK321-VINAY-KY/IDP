"""
File: inprocess.py
Purpose: Single integration surface for the live in-process pipeline hook.
         Runs the observer + CV + routing directly on one page's in-memory RGB
         image (no S3, no PyMuPDF render). Imports ONLY imaging, cv_features,
         routing, and the observer -- never fitz / probe.py -- so PyMuPDF stays
         off the hot path.
Deps: numpy, opencv (via imaging/cv_features), onnxruntime (via the observer)
"""
from __future__ import annotations

from typing import Any, Optional

from src.ai.layer1_probe import cv_features, imaging
from src.ai.layer1_probe.routing import decide_route
from src.ai.layer1_probe.script_classifier import LipikaOnnxScriptObserver
from src.ai.layer1_probe.types import ProbeConfig, RouteChoice

# Observer singleton: built ONCE and reused. The session= ctor param stays on the
# observer for tests; this module-level instance uses the lazy module singleton.
_OBSERVER: Optional[LipikaOnnxScriptObserver] = None


def _get_observer() -> LipikaOnnxScriptObserver:
    global _OBSERVER
    if _OBSERVER is None:
        _OBSERVER = LipikaOnnxScriptObserver()
    return _OBSERVER


def probe_page_route(page_context: dict, page_number: int, profile: Any, cfg: ProbeConfig,
                     *, render_dpi: Optional[int] = None) -> Optional[RouteChoice]:
    render_dpi = render_dpi or cfg.page_image_dpi
    rgb = page_context.get("image_array")
    if rgb is None:
        return None

    gray = imaging.analysis_gray(rgb, render_dpi, cfg.analysis_dpi)
    mask = imaging.ink_mask(gray)
    blank = cv_features.ink_ratio(mask) < cfg.blank_ink_ratio
    has_table, _ = cv_features.detect_table(
        mask, cfg.table_min_h_lines, cfg.table_min_v_lines, cfg.table_min_intersections)

    obs = _get_observer().observe(page_number, gray, mask, rgb=rgb)
    return decide_route(
        script=(obs.script if obs else "unknown"),
        script_conf=(obs.script_confidence if obs else 0.0),
        mixed=(obs.mixed_scripts if obs else False),
        is_blank=blank,
        has_table=has_table,
        cfg=cfg,
    )
