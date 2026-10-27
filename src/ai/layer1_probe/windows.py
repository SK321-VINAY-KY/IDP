"""
File: windows.py
Purpose: Shared square-window + letterbox preprocessing for the lipika script
         model, used by BOTH the line-wise observer (script_classifier.py) and
         the offline eval harness (scripts/eval_lipika_crops.py). Keeping the
         preprocessing in one place guarantees the observer runs the EXACT tensor
         the eval validated under "strategy B".

Deps: numpy + OpenCV only. NO torch, NO onnxruntime, NO fitz — import-safe on the
      probe hot path.

Strategy B (validated in scripts/eval_lipika_crops.py):
  square_windows(rgb)   -> cut square-ish windows (aspect 1:1..2:1) along the
                           longer axis of a line crop, ~50% overlap.
  letterbox_square(rgb) -> white-pad to a max(h,w) square, cv2.resize INTER_CUBIC,
                           /255, NCHW float32 [1,3,size,size].
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

# Model input side (square). The eval harness imports this so both paths agree.
_INPUT_SIZE = 224


def letterbox_square(rgb: np.ndarray, size: int = _INPUT_SIZE) -> np.ndarray:
    """White-pad to a max(h,w) square, cv2.resize INTER_CUBIC, /255, NCHW.

    Mirrors the lipika letterbox with mean 0 / std 1. Returns float32
    [1, 3, size, size].
    """
    h, w = rgb.shape[:2]
    side = max(h, w, 1)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    # NCHW
    return np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)


def square_windows(rgb: np.ndarray) -> List[np.ndarray]:
    """Cut square-ish windows along the longer axis of a line crop.

    Windows have aspect between 1:1 and 2:1 (long:short). Slides a window of
    2x the short-side length along the long axis with ~50% overlap. Crops that
    are already <= 2:1 return a single window (the whole crop). The caller
    letterboxes each window with ``letterbox_square`` and means the softmax
    across windows before argmax.
    """
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return []
    short = min(h, w)
    long_ = max(h, w)
    if long_ <= 2 * short:
        # already square-ish (<= 2:1): one window = the whole crop.
        return [rgb]
    win = 2 * short            # 2:1 window along the long axis
    step = max(1, short)       # ~50% overlap
    windows: List[np.ndarray] = []
    pos = 0
    horizontal = w >= h
    while pos < long_:
        end = min(pos + win, long_)
        start = max(0, end - win)
        if horizontal:
            windows.append(rgb[:, start:end].copy())
        else:
            windows.append(rgb[start:end, :].copy())
        if end >= long_:
            break
        pos += step
    return windows
