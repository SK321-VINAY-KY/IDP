"""Tests for the in-memory probe entry point probe_document_images and the
BedrockScriptProbe magic-byte image-format detection. No AWS, no Paddle, no
torch: the VLM is a stub, onnx is None or a fake peaking session.
"""
import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ai.layer1_probe import imaging, probe as probe_mod
from src.ai.layer1_probe.probe import probe_document_images
from src.ai.layer1_probe.types import ProbeConfig, SampleObs
from src.ai.layer1_probe.vlm_probe import BedrockScriptProbe

W, H = 1240, 1754  # A4-ish


def _cfg(**kw):
    c = ProbeConfig()
    c.cache_enabled = kw.pop("cache_enabled", False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _text_rgb(lines=("The quick brown fox jumps over the lazy dog",) * 12,
              size=34) -> np.ndarray:
    """Synthetic text page as an RGB numpy array (no font dependency: draws with
    the PIL default bitmap font so it works on any machine)."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    y = 120
    for line in lines:
        d.text((100, y), line, font=font, fill="black")
        y += int(size * 1.9)
        if y > H - 150:
            break
    return np.asarray(img, dtype=np.uint8)


def _table_rgb() -> np.ndarray:
    base = _text_rgb(("Invoice summary", "Invoice summary"))
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 120, 500, 1120, 1100
    for i in range(7):
        yy = y0 + i * (y1 - y0) // 6
        d.line([(x0, yy), (x1, yy)], fill="black", width=3)
    for j in range(5):
        xx = x0 + j * (x1 - x0) // 4
        d.line([(xx, y0), (xx, y1)], fill="black", width=3)
    return np.asarray(img, dtype=np.uint8)


def _blank_rgb() -> np.ndarray:
    return np.full((H, W, 3), 255, np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _jpeg_bytes(rgb: np.ndarray) -> bytes:
    return imaging.encode_jpeg(rgb, 85)


class StubVLM:
    """Records the pages it was called on; returns a script per page (or default)."""

    def __init__(self, script="latin", conf=0.95, hw=None, per_page=None, table=False):
        self.script, self.conf, self.hw = script, conf, hw
        self.per_page = per_page or {}
        self.table = table
        self.calls = []

    def observe(self, pn, jpeg):
        self.calls.append(pn)
        s, c = self.per_page.get(pn, (self.script, self.conf))
        return SampleObs(page_number=pn, script=s, script_confidence=c,
                         has_table=self.table, handwriting_prob=self.hw, source="vlm")


# ---------------------------------------------------------------- numpy RGB routing
def test_images_latin_routes_printed_with_ocr_lang_en():
    images = {1: _text_rgb(), 2: _text_rgb(), 3: _text_rgb()}
    vlm = StubVLM(script="latin")
    out = probe_document_images(images, [1, 2, 3], cfg=_cfg(), vlm=vlm)
    assert out[1].route == "paddleocr_printed"
    assert out[1].ocr_lang == "en"
    assert all(out[pn].probed for pn in (1, 2, 3))


@pytest.mark.parametrize("script", ["devanagari", "telugu"])
def test_images_non_latin_routes_vlm_transcribe(script):
    images = {1: _text_rgb(), 2: _text_rgb(), 3: _text_rgb()}
    vlm = StubVLM(script=script)
    out = probe_document_images(images, [1, 2, 3], cfg=_cfg(), vlm=vlm)
    assert {out[pn].route for pn in (1, 2, 3)} == {"vlm_transcribe"}


def test_images_blank_page_skipped():
    images = {1: _text_rgb(), 2: _blank_rgb()}
    vlm = StubVLM(script="latin")
    out = probe_document_images(images, [1, 2], cfg=_cfg(), vlm=vlm)
    assert out[2].route == "skip"
    assert out[2].status == "blank" and out[2].is_blank
    # a blank page is never sampled by the VLM
    assert 2 not in vlm.calls


# ---------------------------------------------------------------- bytes inputs
def test_png_and_jpeg_bytes_decode_and_route_identically():
    rgb_pages = {1: _text_rgb(), 2: _text_rgb(), 3: _text_rgb()}
    png = {pn: _png_bytes(r) for pn, r in rgb_pages.items()}
    jpg = {pn: _jpeg_bytes(r) for pn, r in rgb_pages.items()}

    out_png = probe_document_images(png, [1, 2, 3], cfg=_cfg(), vlm=StubVLM(script="latin"))
    out_jpg = probe_document_images(jpg, [1, 2, 3], cfg=_cfg(), vlm=StubVLM(script="latin"))

    for pn in (1, 2, 3):
        assert out_png[pn].route == out_jpg[pn].route == "paddleocr_printed"
        assert out_png[pn].ocr_lang == out_jpg[pn].ocr_lang == "en"


# ---------------------------------------------------------------- image_s3_* absent
def test_images_entry_point_sets_no_s3_fields():
    out = probe_document_images({1: _text_rgb()}, [1], cfg=_cfg(), vlm=StubVLM())
    assert out[1].image_s3_bucket is None and out[1].image_s3_key is None


# ---------------------------------------------------------------- magic-byte format
class _RecordingClient:
    """Captures the Converse image 'format' field; returns a fixed latin JSON."""

    def __init__(self):
        self.formats = []

    def converse(self, **kw):
        content = kw["messages"][0]["content"]
        for part in content:
            if "image" in part:
                self.formats.append(part["image"]["format"])
        return {"output": {"message": {"content": [
            {"text": '{"script":"latin","script_confidence":0.95,"has_table":false}'}]}}}


def test_bedrock_magic_byte_png():
    client = _RecordingClient()
    probe = BedrockScriptProbe("m", client=client)
    probe.observe(1, _png_bytes(_text_rgb()))
    assert client.formats == ["png"]


def test_bedrock_magic_byte_jpeg():
    client = _RecordingClient()
    probe = BedrockScriptProbe("m", client=client)
    probe.observe(1, _jpeg_bytes(_text_rgb()))
    assert client.formats == ["jpeg"]


def test_bedrock_magic_byte_ambiguous_defaults_jpeg():
    client = _RecordingClient()
    probe = BedrockScriptProbe("m", client=client)
    probe.observe(1, b"not-an-image-header")
    assert client.formats == ["jpeg"]


def test_image_format_classifier_direct():
    assert BedrockScriptProbe._image_format(b"\x89PNG\r\n\x1a\n") == "png"
    assert BedrockScriptProbe._image_format(b"\xff\xd8\xff\xe0") == "jpeg"
    assert BedrockScriptProbe._image_format(b"GIF89a") == "jpeg"


# ---------------------------------------------------------------- sampling count
def test_vlm_call_count_equals_n_samples_for_multipage_latin():
    images = {pn: _text_rgb() for pn in range(1, 9)}  # 8 latin pages
    vlm = StubVLM(script="latin")
    probe_document_images(images, list(range(1, 9)), cfg=_cfg(n_samples=3), vlm=vlm)
    assert len(vlm.calls) == 3


def test_samples_disagree_widens_sampling_and_nearest_inherits():
    pages = list(range(1, 13))  # 12 pages
    images = {pn: _text_rgb() for pn in pages}
    # Early pages latin, later pages devanagari -> the first-wave samples (1/6/12)
    # disagree, so sampling widens.
    per_page = {pn: (("latin", 0.95) if pn <= 3 else ("devanagari", 0.95)) for pn in pages}
    vlm = StubVLM(per_page=per_page)
    out = probe_document_images(images, pages, cfg=_cfg(n_samples=3, max_samples=6), vlm=vlm)
    # widened: more than the first n_samples=3 VLM calls were made
    assert len(vlm.calls) > 3
    assert out[1].script == "latin" and out[1].route == "paddleocr_printed"
    # heterogeneous doc -> every page inherits from its nearest sample, not a
    # single doc-wide script; a late page lands on devanagari -> vlm_transcribe.
    assert out[12].script == "devanagari" and out[12].route == "vlm_transcribe"


# ---------------------------------------------------------------- per-lang engine cache
def test_get_paddle_engine_caches_per_mode_and_lang(monkeypatch):
    import src.ai.layer2_conversion.scanned as scanned

    created = []

    class FakePaddleOCR:
        def __init__(self, **kw):
            created.append((kw.get("lang"),))

    fake_mod = type("M", (), {"PaddleOCR": FakePaddleOCR})
    monkeypatch.setitem(__import__("sys").modules, "paddleocr", fake_mod)
    monkeypatch.setattr(scanned, "_paddle_engines", {})

    e_en = scanned._get_paddle_engine(mode="printed", lang="en")
    e_en2 = scanned._get_paddle_engine(mode="printed", lang="en")
    e_te = scanned._get_paddle_engine(mode="printed", lang="te")
    e_hw = scanned._get_paddle_engine(mode="handwritten", lang="en")

    assert e_en is e_en2                       # same (mode, lang) -> cached
    assert e_te is not e_en                     # different lang -> new engine
    assert e_hw is not e_en                      # different mode -> new engine
    assert [c[0] for c in created] == ["en", "te", "en"]
    assert set(scanned._paddle_engines) == {("printed", "en"), ("printed", "te"),
                                            ("handwritten", "en")}
