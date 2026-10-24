"""
File: line_crops.py
Purpose: Extract tight text-line crops from an analysis grayscale page.
         Drops ruled lines, merges words of a line via horizontal dilation,
         and returns crops taken from the GRAY image in reading order.
Deps: numpy, opencv (reuses imaging.ink_mask). NO torch / onnxruntime / fitz.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.ai.layer1_probe import imaging


def extract_line_crops(
    gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    min_height: int = 10,
    max_height_frac: float = 0.5,
    pad: int = 3,
    min_width: int = 15,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Extract text-line crops from a grayscale page, in reading order.

    Algorithm:
      1. If ``mask`` is None, compute ``mask = imaging.ink_mask(gray)``.
      2. Detect long thin near-horizontal strokes (ruled lines) via a
         morphological open with a wide horizontal kernel (mirroring
         ``cv_features.detect_table``) and subtract them from the mask, so a
         page containing only rules yields no line boxes.
      3. Dilate the de-ruled ink mask horizontally with a wide, short kernel to
         merge the words of a single text line into one blob.
      4. Label connected components via ``cv2.connectedComponentsWithStats``.
      5. Keep boxes whose height is in ``[min_height, max_height_frac * H]`` and
         whose width is at least ``min_width`` (drops specks and page-tall blobs).
      6. Crop each kept box tight from ``gray`` (not the mask) with ``pad`` px,
         clamped to image bounds.
      7. Sort top-to-bottom, ties left-to-right (reading order).

    Args:
        gray: 2-D uint8 grayscale analysis image (H, W).
        mask: Optional precomputed ink mask (ink = 255). Computed if None.
        min_height: Minimum box height in px to keep (default 10).
        max_height_frac: Max box height as a fraction of page height (default 0.5).
        pad: Padding in px added around each kept box before cropping (default 3).
        min_width: Minimum box width in px to keep (default 15).

    Returns:
        List of ``(crop, (x, y, w, h))`` tuples where ``crop`` is a view-copy of
        ``gray`` for that box (padded, clamped) and ``(x, y, w, h)`` is the
        padded, clamped bounding box in page coordinates. Empty list if none.
    """
    if gray.ndim != 2:
        raise ValueError("extract_line_crops expects a 2-D grayscale image")

    H, W = gray.shape
    if mask is None:
        mask = imaging.ink_mask(gray)

    # (2) Detect and subtract long THIN horizontal ruled strokes. A wide
    # horizontal open isolates horizontal runs; we then keep only the thin ones
    # (ruled lines are a few px tall) so tall text bands are not mistaken for
    # rules and erased.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 25), 1))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    rule_max_h = max(2, min_height // 2)
    nh, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    thin = np.zeros(nh, dtype=bool)
    for i in range(1, nh):
        thin[i] = int(h_stats[i][3]) <= rule_max_h  # thin => ruled line
    rules = np.where(thin[h_labels], np.uint8(255), np.uint8(0))
    deruled = cv2.subtract(mask, rules)

    # (3) Dilate horizontally to merge words of one line into a single blob.
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 30), 3))
    merged = cv2.dilate(deruled, dk)

    # (4) Connected components.
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    max_h = max_height_frac * H
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh = (int(stats[i][0]), int(stats[i][1]),
                        int(stats[i][2]), int(stats[i][3]))
        # (5) Keep text-line-sized boxes only.
        if bh < min_height or bh > max_h or bw < min_width:
            continue
        boxes.append((x, y, bw, bh))

    # (7) Reading order: top-to-bottom, ties left-to-right.
    boxes.sort(key=lambda b: (b[1], b[0]))

    crops: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for x, y, bw, bh in boxes:
        # (6) Pad and clamp to image bounds, then crop from gray.
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + bw + pad)
        y1 = min(H, y + bh + pad)
        crop = gray[y0:y1, x0:x1].copy()
        crops.append((crop, (x0, y0, x1 - x0, y1 - y0)))

    return crops
