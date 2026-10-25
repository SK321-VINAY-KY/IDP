"""
Integration tests for the per-document probe wired into pipeline.process_document
(FEAT-002). Flag-guarded by settings.probe_enabled, default OFF.

Fully mocked: pipeline.inspect_page yields controlled PageProfiles, the LLM client
is a stub, the Paddle Lambda invoke is monkeypatched (payload captured, no AWS), and
pipeline._build_probe_deps is monkeypatched to inject a StubVLM + onnx=None so no
Bedrock/boto/onnx loads. Nothing here touches torch, a real model, or real AWS.
"""
import numpy as np
import pytest

import src.ai.layer1_routing.pipeline as pipeline
import src.ai.layer2_conversion.scanned as scanned
from src.ai.layer1_probe import inprocess
from src.ai.layer1_probe.types import SampleObs
from src.ai.schemas.page import PageProfile


# ----------------------------------------------------------------------------- stubs
class StubLLM:
    def __init__(self):
        self.transcribe_calls = 0

    def transcribe_handwriting(self, image_bytes):
        self.transcribe_calls += 1
        return "vlm markdown", 0.95

    def analyze_page(self, image_bytes, page_profile_hint=None):
        raise AssertionError("analyze_page should not be called")


class StubVLM:
    """Script observer stub. Returns a per-page SampleObs; records call pages."""
    def __init__(self, script="latin", conf=0.95, per_page=None, hw=None):
        self.script, self.conf, self.per_page, self.hw = script, conf, per_page or {}, hw
        self.calls = []

    def observe(self, pn, jpeg):
        self.calls.append(pn)
        s, c = self.per_page.get(pn, (self.script, self.conf))
        return SampleObs(page_number=pn, script=s, script_confidence=c,
                         handwriting_prob=self.hw, source="vlm")


# ----------------------------------------------------------------------------- profiles
def _scanned_profile(pn, script="unknown"):
    return PageProfile(
        page_number=pn, has_text=False, char_count=0, image_coverage=0.99,
        has_tables=False, is_scanned=True, has_vector_drawings=False,
        primary_script=script, complexity_score=3, dpi_estimate=150,
    )


def _digital_profile(pn):
    return PageProfile(
        page_number=pn, has_text=True, char_count=900, image_coverage=0.0,
        has_tables=False, is_scanned=False, has_vector_drawings=False,
        primary_script="latin", complexity_score=0, dpi_estimate=0,
    )


# ----------------------------------------------------------------------------- images
def _ink_image():
    rgb = np.full((200, 160, 3), 255, dtype=np.uint8)
    rgb[40:170, 20:140] = 10  # plenty of ink -> not blank
    return rgb


def _blank_image():
    return np.full((200, 160, 3), 255, dtype=np.uint8)


def _table_image():
    rgb = np.full((300, 300, 3), 255, dtype=np.uint8)
    # draw a ruled grid so cv_features.detect_table fires
    for y in range(40, 260, 30):
        rgb[y:y + 2, 30:270] = 0
    for x in range(30, 270, 40):
        rgb[40:260, x:x + 2] = 0
    return rgb


def _page_data(pn, image):
    return {
        "page": object(),
        "page_number": pn,
        "context": {"image_array": image, "image_bytes": b"\x89PNG_fake", "render_dpi": 150},
        "image_bytes": b"\x89PNG_fake",
    }


# ----------------------------------------------------------------------------- harness
@pytest.fixture(autouse=True)
def _reset_observer():
    inprocess._OBSERVER = None
    yield
    inprocess._OBSERVER = None


@pytest.fixture(autouse=True)
def _no_concurrency(monkeypatch):
    # Keep page order deterministic and single-threaded.
    monkeypatch.setattr(pipeline.settings, "page_concurrency_limit", 1, raising=False)


def _wire(monkeypatch, profiles_by_pn, vlm=None, captured=None, probe_enabled=True):
    """Install all mocks. profiles_by_pn: {page_number -> PageProfile}."""
    monkeypatch.setattr(pipeline.settings, "probe_enabled", probe_enabled, raising=False)
    monkeypatch.setattr(pipeline, "inspect_page", lambda page, pn: profiles_by_pn[pn])

    if vlm is not None:
        monkeypatch.setattr(pipeline, "_build_probe_deps", lambda cfg: (vlm, None))

    def _fake_invoke(image_array, page_number, mode, image_bytes=None, ocr_lang=None):
        if captured is not None:
            payload = {"image_base64": "x", "page_number": page_number, "mode": mode}
            if ocr_lang:
                payload["ocr_lang"] = ocr_lang
            captured.append(payload)
        return ("ocr text", 0.95, [])

    monkeypatch.setattr(scanned, "_invoke_paddle_lambda", _fake_invoke)
    monkeypatch.setattr(scanned.settings, "paddle_prefer_lambda", True, raising=False)


def _process(pages):
    return pipeline.process_document(pages=pages, llm_client=StubLLM())


# ============================================================================= tests

def test_flag_off_is_byte_identical(monkeypatch):
    """Flag OFF: no probe deps built, no 'probe' key, Layer-1 routing stands."""
    called = {"deps": 0}
    monkeypatch.setattr(pipeline, "_build_probe_deps",
                        lambda cfg: called.__setitem__("deps", called["deps"] + 1) or (None, None))
    captured = []
    _wire(monkeypatch, {1: _scanned_profile(1, "latin")}, captured=captured, probe_enabled=False)

    pages = [_page_data(1, _ink_image())]
    results = _process(pages)
    _out, meta = results[0]

    assert called["deps"] == 0                       # probe never engaged
    assert "probe" not in pages[0]["context"]
    assert "probe" not in meta.extra
    assert meta.selected_engine == "paddleocr_printed"


def test_flag_on_scanned_latin_printed_payload_has_ocr_lang(monkeypatch):
    vlm = StubVLM(script="latin", conf=0.95)
    captured = []
    _wire(monkeypatch, {1: _scanned_profile(1, "unknown")}, vlm=vlm, captured=captured)

    pages = [_page_data(1, _ink_image())]
    _out, meta = _process(pages)[0]

    assert meta.selected_engine == "paddleocr_printed"
    assert meta.extra["probe"]["route"] == "paddleocr_printed"
    assert captured and captured[0].get("ocr_lang") == "en"
    assert captured[0]["mode"] == "printed"


def test_flag_on_latin_handwriting_routes_handwritten(monkeypatch):
    vlm = StubVLM(script="latin", conf=0.95, hw=0.9)
    _wire(monkeypatch, {1: _scanned_profile(1, "unknown")}, vlm=vlm)

    pages = [_page_data(1, _ink_image())]
    _out, meta = _process(pages)[0]

    assert meta.selected_engine == "paddleocr_handwritten"
    assert meta.extra["probe"]["handwriting_prob"] == 0.9


def test_flag_on_non_latin_routes_vlm_no_escalation(monkeypatch):
    vlm = StubVLM(script="telugu", conf=0.95)
    _wire(monkeypatch, {1: _scanned_profile(1, "unknown")}, vlm=vlm)

    pages = [_page_data(1, _ink_image())]
    out, meta = _process(pages)[0]

    assert meta.selected_engine == "vlm_transcribe"
    assert out.escalation_attempts == 0              # ladder not re-entered
    assert out.escalated is False


def test_flag_on_blank_page_skipped(monkeypatch):
    vlm = StubVLM(script="latin", conf=0.95)
    _wire(monkeypatch, {1: _scanned_profile(1, "unknown")}, vlm=vlm)

    pages = [_page_data(1, _blank_image())]
    _out, meta = _process(pages)[0]

    assert meta.extra["probe"]["route"] == "skip"
    assert meta.extra["probe"]["is_blank"] is True
    assert meta.selected_engine == "skip"
    assert vlm.calls == []                            # blank pages are never sampled


def test_mixed_document_per_page_engines(monkeypatch):
    """digital + printed scan + blank + table scan -> docling, printed, skip, printed."""
    vlm = StubVLM(script="latin", conf=0.95)
    profiles = {
        1: _digital_profile(1),
        2: _scanned_profile(2, "unknown"),
        3: _scanned_profile(3, "unknown"),
        4: _scanned_profile(4, "unknown"),
    }
    _wire(monkeypatch, profiles, vlm=vlm)

    pages = [
        _page_data(1, _ink_image()),     # digital -> not probed
        _page_data(2, _ink_image()),     # printed scan
        _page_data(3, _blank_image()),   # blank
        _page_data(4, _table_image()),   # table scan
    ]
    results = _process(pages)
    engines = [m.selected_engine for _o, m in results]

    assert engines == ["docling", "paddleocr_printed", "skip", "paddleocr_printed"]
    # digital page never probed
    assert "probe" not in pages[0]["context"]
    assert "probe" not in results[0][1].extra
    # table scan: use_structure recorded, not acted on (engine stays printed)
    assert results[3][1].extra["probe"]["has_table"] is True
    # engines differ per page
    assert len(set(engines)) >= 3


def test_heterogeneous_document_widens_and_inherits(monkeypatch):
    """Sampled pages disagree -> sampling widens; non-sampled inherit nearest sample."""
    # 7 scanned pages; first-wave samples (1, 4, 7) disagree -> widen.
    per_page = {1: ("latin", 0.95), 4: ("telugu", 0.95), 7: ("latin", 0.95)}
    vlm = StubVLM(per_page=per_page, script="latin", conf=0.95)
    profiles = {pn: _scanned_profile(pn, "unknown") for pn in range(1, 8)}
    _wire(monkeypatch, profiles, vlm=vlm)

    pages = [_page_data(pn, _ink_image()) for pn in range(1, 8)]
    results = _process(pages)

    assert len(vlm.calls) > 3                         # widened beyond n_samples
    engines = [m.selected_engine for _o, m in results]
    # a mix of routes appears (telugu -> vlm, latin -> printed) via inheritance
    assert "vlm_transcribe" in engines
    assert "paddleocr_printed" in engines


def test_probe_raises_falls_back_and_document_completes(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("probe exploded")

    import src.ai.layer1_probe.probe as probe_mod
    monkeypatch.setattr(probe_mod, "probe_document_images", _boom)
    captured = []
    profiles = {1: _scanned_profile(1, "latin"), 2: _scanned_profile(2, "latin")}
    _wire(monkeypatch, profiles, vlm=StubVLM(), captured=captured)

    pages = [_page_data(1, _ink_image()), _page_data(2, _ink_image())]
    results = _process(pages)

    assert len(results) == 2                          # document completed
    for pd in pages:
        assert "probe" not in pd["context"]           # fell back to today's routing
    for _o, m in results:
        assert "probe" not in m.extra
        assert m.selected_engine == "paddleocr_printed"


def test_digital_pages_never_probed(monkeypatch):
    vlm = StubVLM(script="latin", conf=0.95)
    _wire(monkeypatch, {1: _digital_profile(1), 2: _digital_profile(2)}, vlm=vlm)

    pages = [_page_data(1, _ink_image()), _page_data(2, _ink_image())]
    results = _process(pages)

    assert vlm.calls == []                            # nothing needed probing
    for pd in pages:
        assert "probe" not in pd["context"]
    assert [m.selected_engine for _o, m in results] == ["docling", "docling"]


def test_both_flags_on_new_probe_wins_old_hook_not_called(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_PROBE", "1")       # dormant per-page hook enabled too
    spy = {"calls": 0}
    monkeypatch.setattr(pipeline, "_maybe_probe_route",
                        lambda *a, **k: spy.__setitem__("calls", spy["calls"] + 1) or None)
    vlm = StubVLM(script="latin", conf=0.95)
    _wire(monkeypatch, {1: _scanned_profile(1, "unknown")}, vlm=vlm)

    pages = [_page_data(1, _ink_image())]
    _out, meta = _process(pages)[0]

    assert meta.extra["probe"]["route"] == "paddleocr_printed"
    assert spy["calls"] == 0                           # old hook skipped for probed page


def test_vlm_probe_call_count_is_n_samples(monkeypatch):
    """Homogeneous latin doc, 3 nonblank pages -> VLM observes exactly 3 samples."""
    vlm = StubVLM(script="latin", conf=0.95)
    profiles = {pn: _scanned_profile(pn, "unknown") for pn in (1, 2, 3)}
    _wire(monkeypatch, profiles, vlm=vlm)

    pages = [_page_data(pn, _ink_image()) for pn in (1, 2, 3)]
    _process(pages)

    assert len(vlm.calls) == 3
