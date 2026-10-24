"""
File: imaging.py
Purpose: Render a PDF page ONCE; derive the analysis image from it.
Deps: pymupdf, numpy, opencv
"""
from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np


def render_rgb(page: Any, dpi: int) -> np.ndarray:
    """Render page to an RGB uint8 array (H, W, 3)."""
    import fitz
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def analysis_gray(rgb: np.ndarray, render_dpi: int, analysis_dpi: int) -> np.ndarray:
    """Grayscale downscaled to analysis_dpi (never upscaled)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if analysis_dpi < render_dpi:
        f = analysis_dpi / float(render_dpi)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground (ink) = 255. Adaptive threshold copes with uneven scan lighting."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)
    return mask


def image_size(rgb: np.ndarray) -> Tuple[int, int]:
    return rgb.shape[1], rgb.shape[0]
