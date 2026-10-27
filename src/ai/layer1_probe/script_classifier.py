"""
File: script_classifier.py
Purpose: LINE-WISE lipika ONNX script observer for the Layer 1.5 probe (pilot,
         Strategy B, fp32).

MODEL CONTRACT (exported by scripts/export_lipika_script_onnx.py):
  input   float32 [N, 3, 224, 224]  RGB, 0..1, letterboxed to a 224 square,
                                     white (1.0) padding, no ImageNet mean/std.
  output  float32 [N, K]            script logits (softmax applied in the observer)
  labels  JSON metadata             {"labels":[...14 probe script names...], ...}

LINE-WISE PATH (replaces the old whole-page letterbox):
  1. gray (+ ink mask via imaging.ink_mask when not supplied)
  2. line crops via line_crops.extract_line_crops, capped at MAX 24 lines,
     LARGEST FIRST (sort by box area descending)
  3. per line crop, source pixels taken from the full-res `rgb` (bbox mapped from
     analysis-DPI gray coords) when `rgb` is given, else gray->RGB — mirrors the
     eval harness crops_from_rgb_page
  4. cut square-ish windows per line crop (windows.square_windows, Strategy B)
  5. letterbox each window (windows.letterbox_square) into ONE batch, single
     batched session.run
  6. page aggregation = MEAN softmax across ALL windows; script = argmax of the
     mean; script_confidence = top-1 share of the mean; margin = top1 - top2
  7. gate (ProbeConfig): too few windows / low conf / low margin -> return None
     (the page falls through to the VLM).

No torch at runtime: onnxruntime + numpy + OpenCV only. When the model file is
absent `_get_session()` returns None and `observe()` returns None, so the probe
falls back to the VLM (the designed degraded behavior).

`OnnxScriptClassifier` is kept as a module-level alias of `LipikaOnnxScriptObserver`
so lambda_probe_handler.py and existing imports keep working.
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import cv2
import numpy as np

from src.ai.layer1_probe import imaging
from src.ai.layer1_probe.line_crops import extract_line_crops
from src.ai.layer1_probe.types import ProbeConfig, SampleObs
from src.ai.layer1_probe.windows import letterbox_square, square_windows
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Provisional carry-over from the old per-crop averaged-vote model. With the
# line-wise mean softmax a large second-place share still means a genuinely
# mixed-script page, so this only pushes borderline pages to the VLM (the safe
# direction). Re-tune once real mean-softmax distributions are available.
_MIXED_MARGIN = 0.25

_HERE = os.path.dirname(os.path.abspath(__file__))

# Hard cap on line crops per page (largest first) to bound the batch size.
_MAX_LINES = 24

# Module-level singletons — built ONCE, reused across warm invocations.
_SESSION: Any = None          # onnxruntime.InferenceSession | _MISSING sentinel | None
_META: Optional[dict] = None
_INPUT_NAME: Optional[str] = None

# sentinel cached when construction fails so we do not retry per call
_MISSING = object()


def default_labels_path() -> str:
    return os.path.join(_HERE, "models", "lipika_script.labels.json")


def default_model_path() -> str:
    """int8 iff the sidecar metadata says shipped_default == 'int8', else fp32."""
    labels = default_labels_path()
    try:
        with open(labels, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("shipped_default") == "int8":
            return os.path.join(_HERE, "models", "lipika_script.int8.onnx")
    except Exception:
        pass
    return os.path.join(_HERE, "models", "lipika_script.onnx")


def _reset_for_tests() -> None:
    """Clear the module singletons so 'constructed once' assertions and per-test
    fake sessions do not leak across tests."""
    global _SESSION, _META, _INPUT_NAME
    _SESSION = None
    _META = None
    _INPUT_NAME = None


def _load_meta(labels_path: str) -> dict:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_session() -> Any:
    """Lazily build/cache the module singleton InferenceSession. Returns None when
    the model file is missing (logged once); caches a sentinel on construction
    failure so it is not retried per call."""
    global _SESSION, _META, _INPUT_NAME
    if _SESSION is _MISSING:
        return None
    if _SESSION is not None:
        return _SESSION

    model_path = os.environ.get("IDP_PROBE_ONNX_MODEL") or default_model_path()
    labels_path = os.environ.get("IDP_PROBE_ONNX_LABELS") or default_labels_path()
    if not os.path.exists(model_path):
        logger.warning("probe.onnx_model_missing", model_path=model_path)
        _SESSION = _MISSING
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("IDP_PROBE_ORT_THREADS", "2"))
        sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.error("probe.onnx_session_failed", model_path=model_path, error=str(exc))
        _SESSION = _MISSING
        return None
    _SESSION = sess
    _META = _load_meta(labels_path)
    _INPUT_NAME = sess.get_inputs()[0].name
    return _SESSION


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _even_spread(per_line_windows: List[List[np.ndarray]], cap: int) -> List[np.ndarray]:
    """Flatten per-line window lists, keeping at most ``cap`` windows spread evenly
    across the lines.

    When the total window count is <= ``cap`` (or the cap is non-positive) the
    simple flattened list is returned UNCHANGED: line 0's windows, then line 1's,
    ... preserving the input order.

    Otherwise exactly ``cap`` windows are selected by ROUND-ROBIN over the lines
    top-to-bottom: window 0 of line 0, window 0 of line 1, ... then window 1 of
    each line that still has one, wrapping until ``cap`` windows are collected.
    Lines that run out are skipped. Deterministic: no RNG, stable iteration over
    the input lists.
    """
    total = sum(len(w) for w in per_line_windows)
    flat: List[np.ndarray] = [w for line in per_line_windows for w in line]
    if cap <= 0 or total <= cap:
        return flat

    selected: List[np.ndarray] = []
    col = 0
    max_len = max((len(line) for line in per_line_windows), default=0)
    while len(selected) < cap and col < max_len:
        for line in per_line_windows:
            if col < len(line):
                selected.append(line[col])
                if len(selected) >= cap:
                    break
        col += 1
    return selected


def _scale_box(box, fx: float, fy: float, W: int, H: int):
    """Map an analysis-DPI (gray) bbox to full-res rgb coords. Mirrors the eval
    harness scripts/eval_lipika_crops.py::_scale_box."""
    x, y, w, h = box
    x0 = max(0, int(round(x * fx)))
    y0 = max(0, int(round(y * fy)))
    x1 = min(W, int(round((x + w) * fx)))
    y1 = min(H, int(round((y + h) * fy)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


class LipikaOnnxScriptObserver:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None,
                 session: Any = None, cfg: Optional[ProbeConfig] = None):
        self.cfg = cfg or ProbeConfig()
        # Non-routing telemetry set after each observe() (read by probe.py via
        # getattr so stub/VLM-only tests keep working). Not part of SampleObs.
        self.last_n_windows: Optional[int] = None
        self.last_margin: Optional[float] = None
        if session is not None:
            # injected (tests): use as-is, no onnxruntime import; load meta locally.
            self.session = session
            self.meta = _load_meta(labels_path or default_labels_path())
            self.input_name = session.get_inputs()[0].name
            self._injected = True
        else:
            self.session = None           # lazily resolved via the module singleton
            self.meta = None
            self.input_name = None
            self._injected = False

    def _resolve(self):
        if self._injected:
            return self.session, self.meta, self.input_name
        sess = _get_session()
        if sess is None:
            return None, None, None
        return sess, _META, _INPUT_NAME

    def _line_sources(self, gray: np.ndarray, mask: np.ndarray,
                      rgb: Optional[np.ndarray]) -> List[np.ndarray]:
        """Line crops (largest first, capped) mapped to RGB source sub-images."""
        boxes = extract_line_crops(gray, mask)
        if not boxes:
            return []
        # largest first by box area, then cap.
        boxes = sorted(boxes, key=lambda cb: cb[1][2] * cb[1][3], reverse=True)[:_MAX_LINES]
        sources: List[np.ndarray] = []
        if rgb is not None:
            gh, gw = gray.shape[:2]
            H, W = rgb.shape[:2]
            fx = W / float(gw)
            fy = H / float(gh)
            for crop_gray, box in boxes:
                x0, y0, bw, bh = _scale_box(box, fx, fy, W, H)
                if bw <= 0 or bh <= 0:
                    continue
                sources.append(rgb[y0:y0 + bh, x0:x0 + bw].copy())
        else:
            for crop_gray, _box in boxes:
                sources.append(cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB))
        return sources

    def observe(self, page_number: int, gray: np.ndarray, mask: np.ndarray,
                *, rgb: Optional[np.ndarray] = None) -> Optional[SampleObs]:
        # Reset per-page telemetry; set on every return path below.
        self.last_n_windows = 0
        self.last_margin = None

        session, meta, input_name = self._resolve()
        if session is None or meta is None:
            return None

        if mask is None:
            mask = imaging.ink_mask(gray)

        line_sources = self._line_sources(gray, mask, rgb)

        # (4) square-ish windows per line crop, kept grouped per line so the cap
        # can spread the kept windows EVENLY ACROSS the lines (round-robin).
        per_line_windows: List[List[np.ndarray]] = [square_windows(src) for src in line_sources]
        all_windows = _even_spread(per_line_windows, self.cfg.lipika_max_windows)

        n_windows = len(all_windows)
        self.last_n_windows = n_windows
        # GATE A: too few windows -> script unknown, fall through to the VLM.
        if n_windows < self.cfg.lipika_min_windows:
            return None

        # (5) letterbox each window -> single batched forward pass.
        try:
            batch = np.concatenate([letterbox_square(w) for w in all_windows], axis=0)
            logits = np.asarray(session.run(None, {input_name: batch})[0], dtype=np.float32)
        except Exception as exc:
            logger.warning("probe.onnx_run_failed", page_number=page_number, error=str(exc))
            return None

        # (6) page aggregation = MEAN softmax across all windows.
        probs = _softmax(logits)                 # [W, K]
        mean_probs = probs.mean(axis=0)          # [K]
        order = np.argsort(mean_probs)[::-1]
        top_idx = int(order[0])
        top = float(mean_probs[top_idx])
        second = float(mean_probs[int(order[1])]) if mean_probs.shape[0] > 1 else 0.0
        margin = top - second
        self.last_margin = round(margin, 3)

        # GATE B: low confidence OR low margin -> unresolved -> VLM.
        if top < self.cfg.lipika_min_conf or margin < self.cfg.lipika_min_margin:
            return None

        # 14 output slots; indices 7/9/11 == "other" (meetei_mayek/ol_chiki/
        # syloti_nagri collapse to "other", perso_arabic -> "arabic"). The
        # index->label mapping source is scripts/export_lipika_script_onnx.py::
        # _LIPIKA_TO_PROBE and the loopdesk-ai/lipika model card; the collapse is
        # already encoded in lipika_script.labels.json, so argmax -> labels[idx].
        labels = meta.get("labels", [])
        if top_idx >= len(labels):
            logger.warning("probe.onnx_label_oob", page_number=page_number, index=top_idx)
            return SampleObs(page_number=page_number, script="unknown",
                             script_confidence=0.0, source="onnx")
        return SampleObs(
            page_number=page_number,
            script=labels[top_idx],
            script_confidence=round(top, 3),
            mixed_scripts=second > _MIXED_MARGIN,
            source="onnx",
        )


# Backward-compatible alias: lambda_probe_handler.py + existing imports keep working.
OnnxScriptClassifier = LipikaOnnxScriptObserver
