"""
Flag-gated in-process probe wiring tests for pipeline.process_page.

Fully mocked: inspect_page is patched to yield a controlled PageProfile, the LLM
client is a stub, and the Paddle Lambda invoke is monkeypatched so no real OCR/AWS
runs. The observer is injected (fake SampleObs) via inprocess._OBSERVER so no model
loads. Nothing here touches torch, a real model, or real AWS.
"""
import inspect as _inspect

import numpy as np
import pytest

import src.ai.layer1_routing.pipeline as pipeline
from src.ai.layer1_probe import inprocess
from src.ai.layer1_probe.types import PageProbe, SampleObs
from src.ai.schemas.page import PageProfile


# ----------------------------------------------------------------------------- stubs
class StubLLM:
    def __init__(self):
        self.transcribe_calls = 0

    def transcribe_handwriting(self, image_bytes):
        self.transcribe_calls += 1
        return "vlm markdown", 0.95

    def analyze_page(self, image_bytes, page_profile_hint=None):  # unused in these paths
        raise AssertionError("analyze_page should not be called")


class FakeObserver:
    """Stands in for LipikaOnnxScriptObserver; returns a canned SampleObs."""
    def __init__(self, script, conf=0.95, mixed=False):
        self._obs = SampleObs(page_number=1, script=script, script_confidence=conf,
                              mixed_scripts=mixed, source="onnx")
        self.calls = 0

    def observe(self, page_number, gray, mask, *, rgb=None):
        self.calls += 1
        return SampleObs(page_number=page_number, script=self._obs.script,
                         script_confidence=self._obs.script_confidence,
                         mixed_scripts=self._obs.mixed_scripts, source="onnx")


def _scanned_profile(script="unknown"):
    return PageProfile(
        page_number=1, has_text=False, char_count=0, image_coverage=0.99,
        has_tables=False, is_scanned=True, has_vector_drawings=False,
        primary_script=script, complexity_score=3, dpi_estimate=150,
    )


def _digital_profile():
    return PageProfile(
        page_number=1, has_text=True, char_count=900, image_coverage=0.0,
        has_tables=False, is_scanned=False, has_vector_drawings=False,
        primary_script="latin", complexity_score=0, dpi_estimate=0,
    )


def _page_context():
    rgb = np.full((200, 160, 3), 255, dtype=np.uint8)
    rgb[50:150, 30:130] = 10  # some ink so it is not blank
    return {"image_array": rgb, "image_bytes": b"\x89PNG_fake", "render_dpi": 150}


@pytest.fixture(autouse=True)
def _reset_observer():
    inprocess._OBSERVER = None
    yield
    inprocess._OBSERVER = None


def _run(monkeypatch, profile, observer=None, capture_payload=None):
    monkeypatch.setattr(pipeline, "inspect_page", lambda page, pn: profile)
    if observer is not None:
        inprocess._OBSERVER = observer
    llm = StubLLM()

    # Capture the Paddle Lambda payload without hitting AWS.
    import src.ai.layer2_conversion.scanned as scanned

    def _fake_invoke(image_array, page_number, mode, image_bytes=None, ocr_lang=None):
        if capture_payload is not None:
            payload = {"image_base64": "x", "page_number": page_number}
            if ocr_lang:
                payload["ocr_lang"] = ocr_lang
            capture_payload.append(payload)
        return ("ocr text", 0.95, [])

    monkeypatch.setattr(scanned, "_invoke_paddle_lambda", _fake_invoke)
    monkeypatch.setattr(scanned.settings, "paddle_prefer_lambda", True, raising=False)

    out, meta = pipeline.process_page(
        page=object(), page_number=1, page_context=_page_context(),
        llm_client=llm, page_image_bytes=b"\x89PNG_fake",
    )
    return out, meta, llm


# ----------------------------------------------------------------------------- tests
def test_probe_off_is_passthrough_and_no_probe_import(monkeypatch):
    monkeypatch.delenv("IDP_ENABLE_PROBE", raising=False)
    observer = FakeObserver("telugu")
    out, meta, _ = _run(monkeypatch, _scanned_profile("unknown"),
                        observer=observer, capture_payload=[])
    # probe never ran -> observer not called; engine stays Layer-1's choice (paddleocr_printed)
    assert observer.calls == 0
    assert meta.selected_engine == "paddleocr_printed"


def test_probe_on_non_latin_routes_to_vlm(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_PROBE", "1")
    observer = FakeObserver("telugu")
    out, meta, llm = _run(monkeypatch, _scanned_profile("unknown"), observer=observer)
    assert observer.calls == 1
    assert meta.selected_engine == "vlm_transcribe"
    assert llm.transcribe_calls >= 1


def test_probe_on_latin_routes_to_printed_with_ocr_lang(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_PROBE", "1")
    observer = FakeObserver("latin")
    captured = []
    out, meta, _ = _run(monkeypatch, _scanned_profile("unknown"),
                        observer=observer, capture_payload=captured)
    assert meta.selected_engine == "paddleocr_printed"
    assert captured and captured[0].get("ocr_lang") == "en"


def test_probe_off_payload_is_byte_identical(monkeypatch):
    monkeypatch.delenv("IDP_ENABLE_PROBE", raising=False)
    captured = []
    # Layer-1 routes a scanned page to paddleocr_printed; no probe -> no ocr_lang key.
    _run(monkeypatch, _scanned_profile("latin"), observer=FakeObserver("latin"),
         capture_payload=captured)
    assert captured
    assert captured[0] == {"image_base64": "x", "page_number": 1}  # no ocr_lang key


def test_digital_page_not_probed_with_flag_on(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_PROBE", "1")
    observer = FakeObserver("telugu")
    out, meta, _ = _run(monkeypatch, _digital_profile(), observer=observer)
    # digital -> docling primary -> _should_probe returns False
    assert observer.calls == 0
    assert meta.selected_engine == "docling"


def test_contract_fields_removed_and_signature_script_only():
    assert "handwriting_prob" in SampleObs.model_fields
    assert "handwriting_prob" in PageProbe.model_fields
    assert "hw_source" not in PageProbe.model_fields
    from src.ai.layer1_probe.routing import decide_route
    sig = _inspect.signature(decide_route)
    assert "hw_prob" not in sig.parameters


def test_no_route_yields_paddleocr_handwritten(monkeypatch):
    """Across the script-only router, no observed script maps to the handwritten engine."""
    from src.ai.layer1_probe.routing import decide_route
    from src.ai.layer1_probe.types import ProbeConfig
    cfg = ProbeConfig()
    for script in ("latin", "devanagari", "telugu", "unknown", "other"):
        for mixed in (False, True):
            c = decide_route(script=script, script_conf=0.9, mixed=mixed,
                             is_blank=False, has_table=False, cfg=cfg)
            assert c.route != "paddleocr_handwritten"
