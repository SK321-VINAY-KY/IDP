"""
File: script_classifier.py
Purpose: Whole-image lipika ONNX script observer for the Layer 1.5 probe.

MODEL CONTRACT (exported by scripts/export_lipika_script_onnx.py):
  input   float32 [N, 3, H, W]   RGB, 0..1, letterboxed to a square H==W (default
                                 224), white (1.0) padding, no ImageNet mean/std.
  output  float32 [N, K]         script logits (softmax applied in the observer)
  labels  JSON metadata          {"labels":[...probe script names...],
                                  "input_size":[H,W], "mean":[...], "std":[...],
                                  "preprocessing":"single_forward|patch_vote",
                                  "shipped_default":"fp32|int8", ...}

The observer classifies the WHOLE page image (downscaled to the fixed square), not
line crops. No torch at runtime: onnxruntime + numpy + OpenCV only. When the model
file is absent `_get_session()` returns None and `observe()` returns None, so the
probe falls back to the VLM (the designed degraded behavior).

`OnnxScriptClassifier` is kept as a module-level alias of `LipikaOnnxScriptObserver`
so lambda_probe_handler.py and existing imports keep working.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import cv2
import numpy as np

from src.ai.layer1_probe.types import SampleObs
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Provisional carry-over from the old per-crop averaged-vote model. A single
# whole-image softmax has a different second-place distribution, so this is NOT
# re-validated on real lipika logits; it only pushes borderline pages to the VLM
# (the safe direction) and should be re-tuned once real logits are available.
_MIXED_MARGIN = 0.25

_HERE = os.path.dirname(os.path.abspath(__file__))

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


def _letterbox(rgb: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Pad to square (white) then resize to size x size, scale to 0..1 and apply
    (x/255 - mean)/std. Mirrors lipika's `letterbox` (mean=0, std=1 by default)."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    x = (x - mean) / std
    # NCHW
    return np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max())
    return e / e.sum()


class LipikaOnnxScriptObserver:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None,
                 session: Any = None):
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

    def observe(self, page_number: int, gray: np.ndarray, mask: np.ndarray,
                *, rgb: Optional[np.ndarray] = None) -> Optional[SampleObs]:
        session, meta, input_name = self._resolve()
        if session is None or meta is None:
            return None

        src = rgb if rgb is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        h, w = meta.get("input_size", [224, 224])
        mean = np.asarray(meta.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.asarray(meta.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        # square model input: honour H (==W in practice) for the letterbox side.
        x = _letterbox(src, int(h), mean, std)

        try:
            logits = np.asarray(session.run(None, {input_name: x})[0], dtype=np.float32)
        except Exception as exc:
            logger.warning("probe.onnx_run_failed", page_number=page_number, error=str(exc))
            return None

        probs = _softmax(logits[0])
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top = float(probs[top_idx])
        second = float(probs[int(order[1])]) if probs.shape[0] > 1 else 0.0
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


# Backward-compatible alias: existing imports keep working.
OnnxScriptClassifier = LipikaOnnxScriptObserver
