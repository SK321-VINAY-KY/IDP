"""Tests for the Layer 1.5 probe. Synthetic scanned PDFs + stub S3/VLM (no AWS, no models)."""
import io
import json
import os
import random

import fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ai.layer1_probe import cv_features, imaging
from src.ai.layer1_probe.aggregate import (
    aggregate_doc, nearest_sample, pick_sample_pages, samples_disagree,
)
from src.ai.layer1_probe.probe import needs_probe, probe_document
from src.ai.layer1_probe.routing import decide_route
from src.ai.layer1_probe.script_verify import verify_ocr_output
from src.ai.layer1_probe.types import ProbeConfig, SampleObs
from src.ai.layer1_probe.vlm_probe import BedrockScriptProbe, parse_vlm_json

def _find_font(candidates):
    """Return the first font path that exists on this machine, else None."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


# Latin font: used as the default for every synthetic page. If none is found,
# the whole module can't render text, so it is skipped at module level.
LATIN = _find_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux (CI / container)
    "C:/Windows/Fonts/arial.ttf",                         # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS
])

# Devanagari font: used ONLY by the Devanagari e2e test. If none is found, just
# that one test is skipped (see @pytest.mark.skipif below).
DEVA = _find_font([
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",   # Linux (CI / container)
    "C:/Windows/Fonts/mangal.ttf",                        # Windows (Devanagari)
    "C:/Windows/Fonts/Nirmala.ttf",                       # Windows 10+ (Indic)
    "/System/Library/Fonts/Kohinoor.ttc",                 # macOS (Devanagari)
])

if LATIN is None:
    pytest.skip(
        "No Latin TrueType font available to render synthetic test pages.",
        allow_module_level=True,
    )

W, H = 1240, 1754  # A4 @150dpi


# ---------------------------------------------------------------- synthetic pages
def _text_page(lines, font_path=LATIN, size=34, wobble=False, seed=0):
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    y = 120
    for line in lines:
        x = 100
        if not wobble:
            d.text((x, y), line, font=font, fill="black")
        else:
            for ch in line:
                f = ImageFont.truetype(font_path, int(size * rnd.uniform(0.75, 1.3)))
                d.text((x, y + rnd.randint(-9, 9)), ch, font=f, fill=(20, 20, 60))
                x += int(size * rnd.uniform(0.45, 0.8))
        y += int(size * 1.9) + (rnd.randint(-6, 6) if wobble else 0)
        if y > H - 150:
            break
    return img


def _table_page():
    img = _text_page(["Invoice summary"] * 2)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 120, 500, 1120, 1100
    for i in range(7):
        y = y0 + i * (y1 - y0) // 6
        d.line([(x0, y), (x1, y)], fill="black", width=3)
    for j in range(5):
        x = x0 + j * (x1 - x0) // 4
        d.line([(x, y0), (x, y1)], fill="black", width=3)
    return img


def _pdf_from_images(images) -> bytes:
    doc = fitz.open()
    for im in images:
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buf.getvalue())
    out = doc.tobytes(); doc.close()
    return out


LINES = ["The quick brown fox jumps over the lazy dog", "Pack my box with five dozen liquor jugs",
         "How vexingly quick daft zebras jump today", "Sphinx of black quartz judge my vow"] * 5


class StubS3:
    def __init__(self): self.objs = {}
    def put_object(self, Bucket, Key, Body, ContentType=None): self.objs[(Bucket, Key)] = Body
    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objs: raise KeyError(Key)
        return {"Body": io.BytesIO(self.objs[(Bucket, Key)])}


class StubVLM:
    def __init__(self, script="latin", conf=0.95, table=False, per_page=None, hw=None):
        self.script, self.conf, self.table, self.per_page = script, conf, table, per_page or {}
        self.hw = hw
        self.calls = []
    def observe(self, pn, jpeg):
        self.calls.append(pn)
        s, c = self.per_page.get(pn, (self.script, self.conf))
        return SampleObs(page_number=pn, script=s, script_confidence=c,
                         has_table=self.table, handwriting_prob=self.hw, source="vlm")


def _cfg(**kw):
    c = ProbeConfig(); c.cache_enabled = kw.pop("cache_enabled", False)
    for k, v in kw.items(): setattr(c, k, v)
    return c


# ---------------------------------------------------------------- cv features
def test_blank_page_has_negligible_ink():
    rgb = np.full((H, W, 3), 255, np.uint8)
    g = imaging.analysis_gray(rgb, 144, 100)
    assert cv_features.ink_ratio(imaging.ink_mask(g)) < 0.0015


def _mask(img):
    rgb = np.array(img); g = imaging.analysis_gray(rgb, 150, 100)
    return imaging.ink_mask(g)


def test_ruled_table_detected_and_plain_text_not():
    assert cv_features.detect_table(_mask(_table_page()))[0] is True
    assert cv_features.detect_table(_mask(_text_page(LINES)))[0] is False


def test_underlined_form_without_verticals_is_not_a_table():
    img = _text_page(LINES[:6]); d = ImageDraw.Draw(img)
    for y in range(300, 1300, 90):
        d.line([(100, y), (1100, y)], fill="black", width=2)
    assert cv_features.detect_table(_mask(img))[0] is False


# ---------------------------------------------------------------- routing table
@pytest.mark.parametrize("kw,route,lang", [
    (dict(script="latin", script_conf=.9), "paddleocr_printed", "en"),
    (dict(script="devanagari", script_conf=.9), "vlm_transcribe", None),   # no dedicated OCR; -> VLM
    (dict(script="latin", script_conf=.3), "vlm_transcribe", None),        # low confidence -> VLM
    (dict(script="unknown", script_conf=0.0), "vlm_transcribe", None),
])
def test_routing_table(kw, route, lang):
    c = decide_route(mixed=False, is_blank=False, has_table=False, cfg=_cfg(), **kw)
    assert (c.route, c.ocr_lang) == (route, lang)
    # no script-only routing path may yield the handwritten engine
    assert c.route != "paddleocr_handwritten"


def test_devanagari_goes_to_vlm_and_mixed_latin_goes_to_vlm():
    cfg = _cfg()
    c = decide_route(script="devanagari", script_conf=.9, mixed=False, is_blank=False, has_table=False, cfg=cfg)
    assert c.route == "vlm_transcribe"
    m = decide_route(script="latin", script_conf=.9, mixed=True, is_blank=False, has_table=False, cfg=cfg)
    assert m.route == "vlm_transcribe"


def test_table_sets_structure_hint_only_for_printed():
    c = decide_route(script="latin", script_conf=.9, mixed=False, is_blank=False, has_table=True, cfg=_cfg())
    assert c.use_structure is True
    assert decide_route(script="latin", script_conf=.9, mixed=False, is_blank=False,
                        has_table=False, cfg=_cfg()).use_structure is False


def test_blank_routes_to_skip():
    assert decide_route(script="latin", script_conf=.9, mixed=False, is_blank=True,
                        has_table=False, cfg=_cfg()).route == "skip"


# ---------------------------------------------------------------- aggregate
def test_pick_sample_pages_spread():
    assert pick_sample_pages(list(range(1, 11)), 3) == [1, 5, 10] or pick_sample_pages(list(range(1, 11)), 3) == [1, 6, 10]
    assert pick_sample_pages([4, 9], 3) == [4, 9]


def test_aggregate_and_disagreement_and_nearest():
    a = SampleObs(page_number=1, script="latin", script_confidence=.9, source="vlm")
    b = SampleObs(page_number=20, script="devanagari", script_confidence=.9, source="vlm")
    assert samples_disagree([a, b])
    prior = aggregate_doc([a, b])
    assert prior.heterogeneous
    assert nearest_sample(3, prior.samples).page_number == 1
    assert nearest_sample(18, prior.samples).page_number == 20


# ---------------------------------------------------------------- VLM parsing
def test_parse_vlm_json_handles_fences_and_prose():
    t = 'Sure!\n```json\n{"script":"Hindi","script_confidence":0.9,"handwriting_fraction":0.1,"mixed_scripts":false,"has_table":true}\n```'
    assert parse_vlm_json(t)["script"] == "Hindi"
    assert parse_vlm_json("no json here") is None


def test_bedrock_probe_normalises_and_survives_garbage():
    class C:
        def __init__(self, txt): self.txt = txt
        def converse(self, **kw): return {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": self.txt}]}}}
    ok = BedrockScriptProbe("m", client=C('{"script":"Hindi","script_confidence":1.4,"has_table":true}'))
    o = ok.observe(3, b"x")
    assert (o.script, o.script_confidence, o.has_table) == ("devanagari", 1.0, True)
    assert BedrockScriptProbe("m", client=C("garbage")).observe(1, b"x") is None

    class Boom:
        def converse(self, **kw): raise RuntimeError("throttled")
    assert BedrockScriptProbe("m", client=Boom()).observe(1, b"x") is None


# ---------------------------------------------------------------- script verify
def test_script_verify():
    assert verify_ocr_output("The quick brown fox jumps over the lazy dog", "latin").ok
    bad = verify_ocr_output("यह एक परीक्षण दस्तावेज़ है जो हिंदी में लिखा गया है", "latin")
    assert not bad.ok and bad.observed_script == "devanagari"
    assert verify_ocr_output("abc", "latin").reason == "too_little_text"
    assert verify_ocr_output("Invoice 2024 Total ₹ 5,000 नमस्ते धन्यवाद सर", "devanagari").ok


# ---------------------------------------------------------------- needs_probe
def test_needs_probe_rules():
    c = _cfg()
    assert not needs_probe(None, "docling", c) and not needs_probe(None, "skip", c)
    assert needs_probe(None, "paddleocr_printed", c)
    assert needs_probe({"has_text": True, "char_count": 900, "primary_script": "latin", "is_scanned": False}, None, c) is False
    assert needs_probe({"has_text": False, "char_count": 0, "primary_script": "unknown", "is_scanned": True}, None, c)


# ---------------------------------------------------------------- end to end
def _run(pdf, pages_in=None, vlm=None, cfg=None, s3=None, etag=None, time_left=None):
    s3 = s3 or StubS3()
    return probe_document(pdf, document_id="doc 1.pdf", s3_bucket="b", s3_key="k.pdf", pages_in=pages_in,
                          cfg=cfg or _cfg(), s3=s3, vlm=vlm, onnx=None, etag=etag, time_left=time_left), s3


def test_e2e_mixed_doc_latin_print_table_blank():
    pdf = _pdf_from_images([_text_page(LINES), _table_page(), Image.new("RGB", (W, H), "white"), _text_page(LINES)])
    vlm = StubVLM()
    out, s3 = _run(pdf, vlm=vlm)
    pg = {p["page_number"]: p for p in out["pages"]}
    assert pg[1]["route"] == "paddleocr_printed" and pg[1]["ocr_lang"] == "en"
    assert pg[2]["has_table"] and pg[2]["use_structure"]
    assert pg[3]["route"] == "skip" and pg[3]["status"] == "blank"
    assert out["doc_prior"]["script"] == "latin"
    # page images written once per non-blank page, keyed by sanitised doc id
    assert ("b", "pages/doc_1.pdf/1.jpg") in s3.objs and ("b", "pages/doc_1.pdf/3.jpg") not in s3.objs
    assert pg[1]["image_s3_key"] == "pages/doc_1.pdf/1.jpg"
    # doc-level sampling: far fewer VLM calls than pages would need (<= n_samples)
    assert len(vlm.calls) <= 3


def test_e2e_doc_level_sampling_limits_vlm_calls():
    pdf = _pdf_from_images([_text_page(LINES, seed=i) for i in range(12)])
    vlm = StubVLM()
    _run(pdf, vlm=vlm)
    assert len(vlm.calls) == 3


@pytest.mark.skipif(DEVA is None, reason="No Devanagari-capable font available to render the test page.")
def test_e2e_devanagari_goes_to_vlm_and_latin_to_printed():
    pdf = _pdf_from_images([_text_page(["यह एक परीक्षण दस्तावेज़ है"] * 6, font_path=DEVA) for _ in range(3)])
    out, _ = _run(pdf, vlm=StubVLM(script="devanagari"))
    assert {p["route"] for p in out["pages"]} == {"vlm_transcribe"}
    out, _ = _run(pdf, vlm=StubVLM(script="latin"))
    assert {p["route"] for p in out["pages"]} == {"paddleocr_printed"}


def test_e2e_digital_and_l1_skip_pages_pass_through_unprobed():
    pdf = _pdf_from_images([_text_page(LINES), _text_page(LINES)])
    pages_in = [{"page_number": 1, "route": "docling"}, {"page_number": 2, "route": "paddleocr_printed"}]
    vlm = StubVLM()
    out, s3 = _run(pdf, pages_in=pages_in, vlm=vlm)
    pg = {p["page_number"]: p for p in out["pages"]}
    assert pg[1]["probed"] is False and pg[1]["route"] == "docling"
    assert ("b", "pages/doc_1.pdf/1.jpg") not in s3.objs
    assert pg[2]["probed"] is True


def test_e2e_cache_hit_skips_vlm():
    pdf = _pdf_from_images([_text_page(LINES), _text_page(LINES)])
    s3 = StubS3(); cfg = _cfg(cache_enabled=True)
    v1 = StubVLM(); _run(pdf, vlm=v1, cfg=cfg, s3=s3, etag="abc-123")
    v2 = StubVLM(); out, _ = _run(pdf, vlm=v2, cfg=cfg, s3=s3, etag="abc-123")
    assert v1.calls and not v2.calls and out["stats"]["cache_hit"]
    assert {p["route"] for p in out["pages"]} == {"paddleocr_printed"}


def test_e2e_no_models_available_routes_to_vlm_not_garbage_ocr():
    pdf = _pdf_from_images([_text_page(LINES)])
    out, _ = _run(pdf, vlm=None)
    assert out["pages"][0]["route"] == "vlm_transcribe"
    assert out["pages"][0]["route_reason"].startswith("script_unresolved")


def test_e2e_vlm_failure_degrades_safely():
    class Dead:
        def observe(self, pn, jpeg): return None
    out, _ = _run(_pdf_from_images([_text_page(LINES)] * 3), vlm=Dead())
    assert {p["route"] for p in out["pages"]} == {"vlm_transcribe"}


def test_e2e_time_budget_stops_rendering_and_routes_by_prior():
    pdf = _pdf_from_images([_text_page(LINES, seed=i) for i in range(6)])
    calls = {"n": 0}
    def left():
        calls["n"] += 1
        return 100.0 if calls["n"] <= 3 else 1.0
    out, _ = _run(pdf, vlm=StubVLM(), time_left=left)
    pg = {p["page_number"]: p for p in out["pages"]}
    assert pg[1]["status"] == "ok" and pg[6]["status"] == "skipped_timeout"
    assert pg[6]["route"] == "paddleocr_printed" and pg[6]["image_s3_key"] is None


def test_e2e_upload_failure_leaves_image_key_none_but_still_routes():
    class BadS3(StubS3):
        def put_object(self, **kw): raise RuntimeError("denied")
    out, _ = _run(_pdf_from_images([_text_page(LINES)]), vlm=StubVLM(), s3=BadS3())
    assert out["pages"][0]["route"] == "paddleocr_printed" and out["pages"][0]["image_s3_key"] is None


# ---------------------------------------------------------------- ONNX observer contract
def test_onnx_classifier_contract(tmp_path):
    from src.ai.layer1_probe.script_classifier import OnnxScriptClassifier
    labels = ["latin", "devanagari", "telugu", "other"]
    meta = {"labels": labels, "input_size": [224, 224], "mean": [0, 0, 0], "std": [1, 1, 1],
            "preprocessing": "single_forward", "shipped_default": "fp32"}
    (tmp_path / "l.json").write_text(json.dumps(meta))

    class FakeSession:
        def get_inputs(self):
            class _I:  # noqa: E301
                name = "pixel_values"
            return [_I()]
        def run(self, _out, _feeds):
            # strongly peak on devanagari (index 1)
            return [np.array([[0.0, 9.0, 0.0, 0.0]], dtype=np.float32)]

    clf = OnnxScriptClassifier(str(tmp_path / "x.onnx"), str(tmp_path / "l.json"),
                               session=FakeSession())
    rgb = np.array(_text_page(LINES)); g_ = imaging.analysis_gray(rgb, 150, 100)
    obs = clf.observe(1, g_, imaging.ink_mask(g_), rgb=rgb)
    assert obs.script == "devanagari" and obs.source == "onnx"
    assert "hw" + "_prob" not in SampleObs.model_fields  # no abbreviated hw field on the contract


# ---------------------------------------------------------------- handwriting flag (ITEM 4)
def _onnx_peaking(tmp_path, labels, peak_index):
    """Injected OnnxScriptClassifier whose fake session peaks on labels[peak_index]."""
    from src.ai.layer1_probe.script_classifier import OnnxScriptClassifier
    meta = {"labels": labels, "input_size": [224, 224], "mean": [0, 0, 0], "std": [1, 1, 1],
            "preprocessing": "single_forward", "shipped_default": "fp32"}
    lp = tmp_path / "hw_labels.json"
    lp.write_text(json.dumps(meta))

    class FakeSession:
        def get_inputs(self):
            class _I:
                name = "pixel_values"
            return [_I()]
        def run(self, _out, _feeds):
            logits = np.zeros((1, len(labels)), dtype=np.float32)
            logits[0, peak_index] = 9.0
            return [logits]

    return OnnxScriptClassifier(str(tmp_path / "x.onnx"), str(lp), session=FakeSession())


_HW_LABELS = ["latin", "devanagari", "telugu", "other"]


def test_hw_flag_on_confident_latin_handwritten_routes_to_handwritten(tmp_path):
    pdf = _pdf_from_images([_text_page(LINES)])
    onnx = _onnx_peaking(tmp_path, _HW_LABELS, 0)      # confident latin
    vlm = StubVLM(hw=0.9)                               # VLM reports handwriting
    out = probe_document(pdf, document_id="d.pdf", s3_bucket="b", s3_key="k.pdf", pages_in=None,
                         cfg=_cfg(vlm_for_handwriting=True), s3=StubS3(), vlm=vlm, onnx=onnx)
    pg = out["pages"][0]
    assert pg["route"] == "paddleocr_handwritten"
    assert vlm.calls                                   # the extra VLM call happened


def test_hw_flag_off_confident_latin_skips_vlm_and_routes_printed(tmp_path):
    pdf = _pdf_from_images([_text_page(LINES)])
    onnx = _onnx_peaking(tmp_path, _HW_LABELS, 0)
    vlm = StubVLM(hw=0.9)
    out = probe_document(pdf, document_id="d.pdf", s3_bucket="b", s3_key="k.pdf", pages_in=None,
                         cfg=_cfg(vlm_for_handwriting=False), s3=StubS3(), vlm=vlm, onnx=onnx)
    pg = out["pages"][0]
    assert pg["route"] == "paddleocr_printed"
    assert vlm.calls == []                              # no extra VLM call for confident latin


def test_hw_flag_non_latin_confident_never_triggers_vlm_for_handwriting(tmp_path):
    pdf = _pdf_from_images([_text_page(LINES)])
    onnx = _onnx_peaking(tmp_path, _HW_LABELS, 1)       # confident devanagari
    vlm = StubVLM(script="devanagari", hw=0.9)
    out = probe_document(pdf, document_id="d.pdf", s3_bucket="b", s3_key="k.pdf", pages_in=None,
                         cfg=_cfg(vlm_for_handwriting=True), s3=StubS3(), vlm=vlm, onnx=onnx)
    pg = out["pages"][0]
    assert pg["route"] == "vlm_transcribe"
    assert vlm.calls == []                              # confident non-latin skips the VLM


def test_decide_route_handwriting_param_optional_and_gated():
    c = _cfg()
    # no handwriting_prob -> byte-identical to today
    base = decide_route(script="latin", script_conf=.9, mixed=False, is_blank=False,
                        has_table=False, cfg=c)
    assert (base.route, base.ocr_lang) == ("paddleocr_printed", "en")
    # latin + handwriting above threshold -> handwritten engine
    hw = decide_route(script="latin", script_conf=.9, mixed=False, is_blank=False,
                      has_table=False, cfg=c, handwriting_prob=0.9)
    assert hw.route == "paddleocr_handwritten" and hw.ocr_lang == "en"
    # below threshold stays printed
    lo = decide_route(script="latin", script_conf=.9, mixed=False, is_blank=False,
                      has_table=False, cfg=c, handwriting_prob=0.1)
    assert lo.route == "paddleocr_printed"


# ---------------------------------------------------------------- handler
def test_handler_end_to_end_slim_output_and_report(monkeypatch):
    import src.api.lambda_probe_handler as h
    pdf = _pdf_from_images([_text_page(LINES), _table_page()])

    class S3(StubS3):
        def head_object(self, Bucket, Key): return {"ETag": '"deadbeef"', "ContentLength": len(pdf)}
    s3 = S3(); s3.objs[("bkt", "docs/a.pdf")] = pdf
    monkeypatch.setattr(h, "_get_s3", lambda: s3)
    monkeypatch.setattr(h, "_vlm", StubVLM()); monkeypatch.setattr(h, "_onnx", None)
    monkeypatch.setattr(h, "_models_loaded", True); monkeypatch.setattr(h, "_cfg", _cfg(cache_enabled=True))
    out = h.lambda_handler({"document_id": "a.pdf", "s3_bucket": "bkt", "s3_key": "docs/a.pdf",
                            "pages": [{"page_number": 1, "route": "paddleocr_printed"},
                                      {"page_number": 2, "route": "paddleocr_printed"}]}, None)
    body = out["body"]
    assert out["statusCode"] == 200 and body["report_s3_key"] == "probe-reports/a.pdf.json"
    assert set(body["pages"][0]) == set(h._SLIM_FIELDS)
    assert body["pages"][1]["use_structure"] is True
    assert ("bkt", "probe-cache/v1/deadbeef-%d.json" % len(pdf)) in s3.objs
    json.dumps(out)                                   # must be JSON-serialisable for Step Functions
    with pytest.raises(ValueError):
        h.lambda_handler({"document_id": "x"}, None)  # errors raise -> SFN Catch fallback
