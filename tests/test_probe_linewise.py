"""
Tests for the LINE-WISE lipika ONNX script observer (Strategy B) and the probe
VLM-override / telemetry seam. Fully mocked: ONNX via an injected fake session
(canned logits), VLM via a stub. No torch, no AWS, no real model load except the
opt-in, skippable label-mapping test against the real fp32 model.
"""
import glob
import json
import os

import cv2
import numpy as np
import pytest

from src.ai.layer1_probe import imaging
from src.ai.layer1_probe.probe import probe_document
from src.ai.layer1_probe.script_classifier import LipikaOnnxScriptObserver
from src.ai.layer1_probe.types import ProbeConfig, SampleObs

# Reuse the synthetic-PDF + stub helpers from the main probe test module.
from tests.test_probe import StubS3, StubVLM, _pdf_from_images, _text_page, LINES, _cfg

LABELS = ["latin", "devanagari", "telugu", "other"]
_META = {"labels": LABELS, "input_size": [224, 224], "mean": [0, 0, 0], "std": [1, 1, 1],
         "preprocessing": "single_forward", "shipped_default": "fp32"}


@pytest.fixture
def labels_file(tmp_path):
    p = tmp_path / "lipika_script.labels.json"
    p.write_text(json.dumps(_META))
    return str(p)


def _wide_text_gray(n_lines=8, H=600, W=1000):
    """White grayscale page with n_lines wide black text bands -> many windows."""
    gray = np.full((H, W), 255, dtype=np.uint8)
    step = (H - 120) // max(1, n_lines)
    for i in range(n_lines):
        y = 60 + i * step
        cv2.rectangle(gray, (40, y), (W - 40, y + 22), 0, -1)
    return gray


def _fake_session(logits_row):
    """Fake InferenceSession returning the SAME logits row for every window in
    the batch (shape [N, K])."""
    row = np.asarray(logits_row, dtype=np.float32)

    class _Input:
        name = "pixel_values"

    class FakeSession:
        def get_inputs(self):
            return [_Input()]

        def run(self, _out, feeds):
            n = next(iter(feeds.values())).shape[0]
            return [np.tile(row, (n, 1)).astype(np.float32)]

    return FakeSession()


# --------------------------------------------------------------- (a) aggregation
def test_observer_latin_aggregation_and_conf(labels_file):
    # logits peak the latin slot (index 0); same row for every window.
    logits = [9.0, 0.0, 0.0, 0.0]
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file,
                                       session=_fake_session(logits),
                                       cfg=ProbeConfig())
    gray = _wide_text_gray()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb)
    assert out is not None
    assert out.script == "latin" and out.source == "onnx"
    # every window has identical logits, so the mean softmax == softmax(row).
    expected = np.exp(np.asarray(logits) - max(logits))
    expected = expected / expected.sum()
    assert out.script_confidence == pytest.approx(round(float(expected[0]), 3), abs=1e-3)
    # the gate passed: enough windows were produced.
    assert obs_clf.last_n_windows >= ProbeConfig().lipika_min_windows


# --------------------------------------------------------------- (b) too few windows
def test_too_few_windows_returns_none(labels_file):
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file,
                                       session=_fake_session([9.0, 0.0, 0.0, 0.0]),
                                       cfg=ProbeConfig(lipika_min_windows=1000))
    gray = _wide_text_gray()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    assert obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb) is None


# --------------------------------------------------------------- (c) low conf / margin
def test_low_conf_returns_none(labels_file):
    # near-uniform logits -> top-1 share ~0.25 < default lipika_min_conf 0.60.
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file,
                                       session=_fake_session([0.1, 0.0, 0.0, 0.0]),
                                       cfg=ProbeConfig())
    gray = _wide_text_gray()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    assert obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb) is None


def test_low_margin_returns_none(labels_file):
    # two near-tied slots -> top-1 high enough but margin < lipika_min_margin.
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file,
                                       session=_fake_session([5.0, 4.95, -5.0, -5.0]),
                                       cfg=ProbeConfig(lipika_min_conf=0.0))
    gray = _wide_text_gray()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb)
    assert out is None
    assert obs_clf.last_margin is not None and obs_clf.last_margin < ProbeConfig().lipika_min_margin


# --------------------------------------------------------------- probe-level stubs
class _ObserverStub:
    """Stands in for the observer inside probe_document: returns a canned SampleObs
    and exposes last_n_windows / last_margin like the real line-wise observer."""

    def __init__(self, script, conf=0.95, mixed=False, n_windows=40, margin=0.5):
        self._script, self._conf, self._mixed = script, conf, mixed
        self.last_n_windows = n_windows
        self.last_margin = margin

    def observe(self, page_number, gray, mask, *, rgb=None):
        return SampleObs(page_number=page_number, script=self._script,
                         script_confidence=self._conf, mixed_scripts=self._mixed,
                         source="onnx")


class _LogSpy:
    def __init__(self):
        self.events = []

    def __call__(self, event, **kw):
        self.events.append((event, kw))


# --------------------------------------------------------------- (d) VLM contradicts
def test_vlm_contradicts_lipika_latin_routes_to_vlm(monkeypatch):
    import src.ai.layer1_probe.probe as probe_mod
    spy = _LogSpy()
    monkeypatch.setattr(probe_mod.logger, "info", spy)

    pdf = _pdf_from_images([_text_page(LINES)])
    onnx = _ObserverStub("latin", conf=0.95)
    vlm = StubVLM(script="devanagari", conf=0.95)
    out = probe_document(pdf, document_id="d.pdf", s3_bucket="b", s3_key="k.pdf",
                         pages_in=None, cfg=_cfg(vlm_for_handwriting=True),
                         s3=StubS3(), vlm=vlm, onnx=onnx)
    pg = out["pages"][0]
    # confident lipika 'latin' but VLM says devanagari -> VLM wins -> NOT paddle.
    assert pg["route"] != "paddleocr_printed"
    assert pg["route"] == "vlm_transcribe"
    assert out["stats"]["lipika_vlm_disagree"] >= 1
    # telemetry event emitted with agree=False.
    ev = [kw for (name, kw) in spy.events if name == "probe.lipika_vs_vlm"]
    assert ev and any(e.get("agree") is False and e.get("lipika_script") == "latin"
                      and e.get("vlm_script") == "devanagari" for e in ev)


# --------------------------------------------------------------- (e) agreement
def test_confident_latin_vlm_latin_routes_printed(monkeypatch):
    import src.ai.layer1_probe.probe as probe_mod
    spy = _LogSpy()
    monkeypatch.setattr(probe_mod.logger, "info", spy)

    pdf = _pdf_from_images([_text_page(LINES)])
    onnx = _ObserverStub("latin", conf=0.95)
    vlm = StubVLM(script="latin", conf=0.95)  # hw defaults None -> below threshold
    out = probe_document(pdf, document_id="d.pdf", s3_bucket="b", s3_key="k.pdf",
                         pages_in=None, cfg=_cfg(vlm_for_handwriting=True),
                         s3=StubS3(), vlm=vlm, onnx=onnx)
    pg = out["pages"][0]
    assert pg["route"] == "paddleocr_printed"
    assert out["stats"]["lipika_vlm_agree"] >= 1
    ev = [kw for (name, kw) in spy.events if name == "probe.lipika_vs_vlm"]
    assert ev and any(e.get("agree") is True and e.get("n_windows") == 40 for e in ev)


# --------------------------------------------------------------- (CHANGE 6) label map
# dataset language folder -> expected probe script (mirror of the eval harness map).
_LANGUAGE_TO_SCRIPT = {
    "hindi": "devanagari", "marathi": "devanagari", "assamese": "bengali",
    "bengali": "bengali", "punjabi": "gurmukhi", "gujarati": "gujarati",
    "kannada": "kannada", "malayalam": "malayalam", "odia": "odia",
    "tamil": "tamil", "telugu": "telugu", "urdu": "arabic",
}
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_REPO, "src", "ai", "layer1_probe", "models", "lipika_script.onnx")
_VERIFIED = os.path.join(_REPO, "dataset", "verified_twice")


def _one_crop(lang):
    folder = os.path.join(_VERIFIED, lang, "cropped_images")
    if not os.path.isdir(folder):
        return None
    jpegs = sorted(g for g in glob.glob(os.path.join(folder, "*.jpeg"))
                   if not os.path.basename(g).startswith("."))
    if not jpegs:
        return None
    bgr = cv2.imread(jpegs[0], cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------- (ITEM 2) window cap
def _recording_session(logits_row):
    """Fake InferenceSession that records the batch size (window count) of every
    run() call into .batch_sizes, returning the SAME logits row per window."""
    row = np.asarray(logits_row, dtype=np.float32)

    class _Input:
        name = "pixel_values"

    class FakeSession:
        def __init__(self):
            self.batch_sizes = []

        def get_inputs(self):
            return [_Input()]

        def run(self, _out, feeds):
            n = next(iter(feeds.values())).shape[0]
            self.batch_sizes.append(n)
            return [np.tile(row, (n, 1)).astype(np.float32)]

    return FakeSession()


def _tagged_per_line(counts):
    """Per-line window lists where each window is a 1-element array tagged with a
    (line_index, col_index) pair so round-robin order can be asserted exactly."""
    return [[np.array([(li, ci)]) for ci in range(n)] for li, n in enumerate(counts)]


def _tag(win):
    return tuple(win[0])


def test_cap_respected(labels_file):
    """More windows than the cap -> exactly lipika_max_windows forwarded."""
    from src.ai.layer1_probe.script_classifier import _even_spread
    sess = _recording_session([9.0, 0.0, 0.0, 0.0])
    cfg = ProbeConfig(lipika_max_windows=12, lipika_min_windows=1)
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file, session=sess, cfg=cfg)
    # wide page with many lines -> many windows generated.
    gray = _wide_text_gray(n_lines=20, H=1400, W=1600)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb)
    assert out is not None
    assert obs_clf.last_n_windows == cfg.lipika_max_windows
    assert sess.batch_sizes and sess.batch_sizes[0] == cfg.lipika_max_windows


def test_even_spread_across_lines():
    """Kept windows are round-robin across lines, NOT the first-N from line 0."""
    from src.ai.layer1_probe.script_classifier import _even_spread
    # 4 lines, 5 windows each (20 total), cap 8.
    per_line = _tagged_per_line([5, 5, 5, 5])
    kept = _even_spread(per_line, cap=8)
    assert len(kept) == 8
    tags = [_tag(w) for w in kept]
    # first 4 kept come one-per-line, top-to-bottom (col 0 of each line).
    assert tags[:4] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    # next 4 are col 1 of each line.
    assert tags[4:8] == [(0, 1), (1, 1), (2, 1), (3, 1)]
    # per-line share is near-equal (2 each), not all from line 0.
    per_line_kept = {}
    for li, _ci in tags:
        per_line_kept[li] = per_line_kept.get(li, 0) + 1
    assert per_line_kept == {0: 2, 1: 2, 2: 2, 3: 2}


def test_even_spread_uneven_lines_skips_exhausted():
    """Lines with fewer windows are skipped once exhausted; cap still met."""
    from src.ai.layer1_probe.script_classifier import _even_spread
    # line 0 has 1 window, line 1 has 4 (5 total), cap 4.
    per_line = _tagged_per_line([1, 4])
    kept = _even_spread(per_line, cap=4)
    tags = [_tag(w) for w in kept]
    assert len(kept) == 4
    # round-robin: (0,0),(1,0) then line0 exhausted -> (1,1),(1,2).
    assert tags == [(0, 0), (1, 0), (1, 1), (1, 2)]


def test_cap_deterministic():
    """Identical input -> identical selected set/order across two runs."""
    from src.ai.layer1_probe.script_classifier import _even_spread
    per_line = _tagged_per_line([5, 4, 6, 3])
    first = [_tag(w) for w in _even_spread(per_line, cap=9)]
    second = [_tag(w) for w in _even_spread(per_line, cap=9)]
    assert first == second
    assert len(first) == 9


def test_below_cap_unchanged():
    """<= cap windows -> all used in the original flatten order, no selection."""
    from src.ai.layer1_probe.script_classifier import _even_spread
    per_line = _tagged_per_line([2, 3, 1])  # 6 total
    kept = _even_spread(per_line, cap=32)
    tags = [_tag(w) for w in kept]
    # identical to a plain flatten: line 0's windows, then line 1's, then line 2's.
    assert tags == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)]


def test_below_cap_unchanged_observe(labels_file):
    """Through observe(): a page under the cap forwards every generated window."""
    sess = _recording_session([9.0, 0.0, 0.0, 0.0])
    cfg = ProbeConfig(lipika_max_windows=1000, lipika_min_windows=1)
    obs_clf = LipikaOnnxScriptObserver(labels_path=labels_file, session=sess, cfg=cfg)
    gray = _wide_text_gray()
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = obs_clf.observe(1, gray, imaging.ink_mask(gray), rgb=rgb)
    assert out is not None
    # no cap applied: forwarded count == observer's reported window count.
    assert sess.batch_sizes[0] == obs_clf.last_n_windows


@pytest.mark.slow
def test_label_mapping_real_fp32_optional():
    """Index->label sanity against the REAL fp32 model. Skips cleanly if the model
    or fixtures are absent. Hard-asserts the two high-accuracy scripts from the
    eval (bengali, tamil); logs the rest without failing."""
    if not os.path.exists(_MODEL):
        pytest.skip("fp32 model file absent")
    if not os.path.isdir(_VERIFIED):
        pytest.skip("verified_twice fixtures absent")

    from src.ai.layer1_probe import script_classifier as sc
    from src.ai.layer1_probe.windows import letterbox_square, square_windows

    sc._reset_for_tests()
    sess = sc._get_session()
    if sess is None:
        pytest.skip("fp32 session could not be constructed")
    meta, input_name = sc._META, sc._INPUT_NAME
    labels = meta["labels"]

    def _predict(rgb):
        windows = square_windows(rgb) or [rgb]
        batch = np.concatenate([letterbox_square(w) for w in windows], axis=0)
        logits = np.asarray(sess.run(None, {input_name: batch})[0], dtype=np.float32)
        probs = sc._softmax(logits).mean(axis=0)
        return labels[int(np.argmax(probs))]

    results = {}
    for lang, expected in sorted(_LANGUAGE_TO_SCRIPT.items()):
        rgb = _one_crop(lang)
        if rgb is None:
            continue
        results[lang] = (_predict(rgb), expected)

    if not results:
        pytest.skip("no fixture crops found")

    print("\nlabel-mapping sanity (predicted vs expected):")
    for lang, (pred, exp) in results.items():
        print(f"  {lang:10s} -> pred={pred:12s} expected={exp}")

    for lang in ("bengali", "tamil"):
        if lang in results:
            pred, exp = results[lang]
            assert pred == exp, f"{lang}: expected {exp}, got {pred}"

    sc._reset_for_tests()
