"""
Fast, fully-mocked tests for the lipika whole-image ONNX script observer.
No torch, no real model load: every observer result comes from an injected fake
InferenceSession (or a patched _get_session) whose .run() returns canned logits.
"""
import json

import numpy as np
import pytest

from src.ai.layer1_probe import script_classifier as sc
from src.ai.layer1_probe.script_classifier import LipikaOnnxScriptObserver
from src.ai.layer1_probe.types import PageProbe, SampleObs

LABELS = ["latin", "devanagari", "telugu", "other"]
_META = {"labels": LABELS, "input_size": [224, 224], "mean": [0, 0, 0], "std": [1, 1, 1],
         "preprocessing": "single_forward", "shipped_default": "fp32"}


@pytest.fixture
def labels_file(tmp_path):
    p = tmp_path / "lipika_script.labels.json"
    p.write_text(json.dumps(_META))
    return str(p)


def _fake_session(peak_index, construct_counter=None):
    class _Input:
        name = "pixel_values"

    class FakeSession:
        def __init__(self):
            if construct_counter is not None:
                construct_counter.append(1)

        def get_inputs(self):
            return [_Input()]

        def run(self, _out, _feeds):
            logits = np.full((1, len(LABELS)), -5.0, dtype=np.float32)
            logits[0, peak_index] = 9.0
            return [logits]

    return FakeSession()


@pytest.mark.parametrize("peak,script", [(0, "latin"), (1, "devanagari"), (2, "telugu")])
def test_canned_logits_map_to_script(peak, script, labels_file):
    obs = LipikaOnnxScriptObserver(labels_path=labels_file, session=_fake_session(peak))
    gray = np.zeros((40, 80), dtype=np.uint8)
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    out = obs.observe(1, gray, np.zeros_like(gray), rgb=rgb)
    assert out is not None
    assert out.script == script
    assert out.source == "onnx"


def test_no_handwriting_field_on_contracts():
    assert "handwriting_prob" not in SampleObs.model_fields
    assert "handwriting_prob" not in PageProbe.model_fields
    assert "hw_source" not in PageProbe.model_fields


def test_module_session_constructed_once(monkeypatch, labels_file):
    """Through the module singleton (_get_session), the InferenceSession is built
    exactly once across multiple observe() calls."""
    sc._reset_for_tests()
    counter = []

    class _Input:
        name = "pixel_values"

    class FakeSession:
        def __init__(self, *a, **k):
            counter.append(1)

        def get_inputs(self):
            return [_Input()]

        def run(self, _out, _feeds):
            logits = np.full((1, len(LABELS)), -5.0, dtype=np.float32)
            logits[0, 2] = 9.0  # telugu
            return [logits]

    # fake onnxruntime so no real session/model is needed
    import types as _types
    fake_ort = _types.ModuleType("onnxruntime")

    class _Opts:
        intra_op_num_threads = 1

    fake_ort.SessionOptions = lambda: _Opts()
    fake_ort.InferenceSession = lambda *a, **k: FakeSession()
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    # model path exists + labels resolve to our meta
    monkeypatch.setenv("IDP_PROBE_ONNX_MODEL", labels_file)  # any existing file
    monkeypatch.setenv("IDP_PROBE_ONNX_LABELS", labels_file)

    obs = LipikaOnnxScriptObserver()  # no injected session -> uses module singleton
    gray = np.zeros((40, 80), dtype=np.uint8)
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    o1 = obs.observe(1, gray, np.zeros_like(gray), rgb=rgb)
    o2 = obs.observe(2, gray, np.zeros_like(gray), rgb=rgb)
    assert o1 is not None and o2 is not None
    assert o1.script == "telugu" and o2.script == "telugu"
    assert sum(counter) == 1  # constructed exactly once
    sc._reset_for_tests()


def test_missing_model_returns_none(monkeypatch):
    """No model file -> _get_session() returns None -> observe() returns None (VLM fallback)."""
    sc._reset_for_tests()
    monkeypatch.setenv("IDP_PROBE_ONNX_MODEL", r"c:\does\not\exist\lipika_script.onnx")
    obs = LipikaOnnxScriptObserver()
    gray = np.zeros((40, 80), dtype=np.uint8)
    assert obs.observe(1, gray, np.zeros_like(gray)) is None
    sc._reset_for_tests()
