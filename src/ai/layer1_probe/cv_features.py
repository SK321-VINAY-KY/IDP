"""
File: cv_features.py
Purpose: Model-free, ~10-30ms/page signals computed on the analysis image:
           - ink ratio / blank detection
           - ruled-table detection (grid of long H/V lines with intersections)
Deps: numpy, opencv
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def ink_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0


def detect_table(mask: np.ndarray, min_h: int = 3, min_v: int = 2,
                 min_intersections: int = 4) -> Tuple[bool, float]:
    """
    Ruled-table detector: extract long horizontal and vertical strokes with
    morphological opening, then require several of each AND crossings.
    Catches scanned bordered/grid tables. It will NOT catch borderless tables
    (that needs a layout model) and a lined *form* with a grid will also fire.
    Returns (has_table, confidence 0-1).
    """
    h, w = mask.shape
    if h < 50 or w < 50:
        return False, 0.0

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 25)))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)

    def _count(lines: np.ndarray, min_len: int, axis: int) -> int:
        n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        # stats cols: x, y, w, h, area ; axis 0 -> width is length, 1 -> height is length
        return int(sum(1 for i in range(1, n) if stats[i][2 + axis] >= min_len))

    n_h = _count(horiz, int(w * 0.15), 0)
    n_v = _count(vert, int(h * 0.04), 1)

    crossings = cv2.bitwise_and(cv2.dilate(horiz, np.ones((3, 3), np.uint8)),
                                cv2.dilate(vert, np.ones((3, 3), np.uint8)))
    n_x = max(0, cv2.connectedComponents(crossings, connectivity=8)[0] - 1)

    has = n_h >= min_h and n_v >= min_v and n_x >= min_intersections
    conf = 0.0
    if n_h and n_v:
        conf = min(1.0, (min(n_h, 6) / 6.0) * 0.4 + (min(n_v, 4) / 4.0) * 0.3
                   + (min(n_x, 12) / 12.0) * 0.3)
    return bool(has), round(conf, 3)
