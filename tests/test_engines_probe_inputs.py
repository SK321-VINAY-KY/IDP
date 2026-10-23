"""
Fast, fully-mocked tests for the Layer 1.5 probe-input integration in the two
PaddleOCR engines/handlers (printed + handwritten).

These tests MUST stay fast: they never load a real PaddleOCR model and never read
a real PDF/image through the engine. S3 and the engine functions are mocked in the
same style as tests/test_layer2_engines_local.py.
"""
import io
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import docker.paddle_handwritten.engine as paddle_hand_engine
import docker.paddle_handwritten.handler as paddle_hand_handler
import docker.paddle_printed.engine as paddle_printed_engine
import docker.paddle_printed.handler as paddle_printed_handler


class DummyContext:
    aws_request_id = "test-req-probe-12345"


# (handler module, engine module, engine name)
ENGINE_PAIRS = [
    pytest.param(paddle_printed_handler, paddle_printed_engine, "paddleocr_printed", id="printed"),
    pytest.param(paddle_hand_handler, paddle_hand_engine, "paddleocr_handwritten", id="handwritten"),
]

ORIGINAL_RESULT_KEYS = {
    "page_number",
    "engine",
    "markdown",
    "confidence",
    "lines_extracted",
    "blocks",
    "latency_ms",
    "status",
}


def _tiny_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def _canned_result(engine_name: str, image_source: str, ocr_lang: str = "en") -> dict:
    return {
        "page_number": 1,
        "engine": engine_name,
        "markdown": "hello",
        "confidence": 0.9,
        "lines_extracted": 1,
        "blocks": [],
        "latency_ms": 1.0,
        "status": "SUCCESS",
        "ocr_lang": ocr_lang,
        "image_source": image_source,
    }


# ==============================================================================
# (a) image-key path used, PDF NOT downloaded
# ==============================================================================

@pytest.mark.parametrize("handler, engine, engine_name", ENGINE_PAIRS)
def test_image_key_path_used_pdf_not_downloaded(handler, engine, engine_name):
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: _tiny_jpeg_bytes())}

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "pdf-bucket",
        "s3_key": "documents/165.pdf",
        "image_s3_bucket": "image-bucket",
        "image_s3_key": "pages/165/1.jpg",
        "ocr_lang": "fr",
        "page_number": 1,
    }

    canned = _canned_result(engine_name, image_source="s3_image", ocr_lang="fr")
    with patch.object(handler, "get_s3_client", return_value=mock_s3), \
            patch.object(handler.engine, "process_image", return_value=canned) as mock_proc_image, \
            patch.object(handler.engine, "process_pdf_pages") as mock_proc_pdf:
        resp = handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert isinstance(body, dict)  # success body stays a dict
    assert body["engine"] == engine_name
    assert body["results"] == [canned]

    # PDF path must not run
    mock_proc_pdf.assert_not_called()

    # S3 get_object called once, for the IMAGE key only (never the PDF key)
    assert mock_s3.get_object.call_count == 1
    _, kwargs = mock_s3.get_object.call_args
    assert kwargs["Key"] == "pages/165/1.jpg"
    assert kwargs["Bucket"] == "image-bucket"

    # engine.process_image invoked with the image source + requested lang
    _, img_kwargs = mock_proc_image.call_args
    assert img_kwargs["image_source"] == "s3_image"
    assert img_kwargs["lang"] == "fr"


# ==============================================================================
# (b) keys nested under event["page"] behave identically
# ==============================================================================

@pytest.mark.parametrize("handler, engine, engine_name", ENGINE_PAIRS)
def test_probe_keys_nested_under_page(handler, engine, engine_name):
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: _tiny_jpeg_bytes())}

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "pdf-bucket",
        "s3_key": "documents/165.pdf",
        "page_number": 1,
        "page": {
            "image_s3_bucket": "image-bucket",
            "image_s3_key": "pages/165/1.jpg",
            "ocr_lang": "fr",
        },
    }

    canned = _canned_result(engine_name, image_source="s3_image", ocr_lang="fr")
    with patch.object(handler, "get_s3_client", return_value=mock_s3), \
            patch.object(handler.engine, "process_image", return_value=canned) as mock_proc_image, \
            patch.object(handler.engine, "process_pdf_pages") as mock_proc_pdf:
        resp = handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    mock_proc_pdf.assert_not_called()
    assert mock_s3.get_object.call_count == 1
    _, kwargs = mock_s3.get_object.call_args
    assert kwargs["Key"] == "pages/165/1.jpg"
    assert kwargs["Bucket"] == "image-bucket"
    _, img_kwargs = mock_proc_image.call_args
    assert img_kwargs["image_source"] == "s3_image"
    assert img_kwargs["lang"] == "fr"


# ==============================================================================
# (c) missing new keys -> PDF path, schema is original keys + two new ones
# ==============================================================================

@pytest.mark.parametrize("handler, engine, engine_name", ENGINE_PAIRS)
def test_missing_keys_uses_pdf_path_with_new_keys(handler, engine, engine_name):
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"%PDF-1.4 fake")}

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "pdf-bucket",
        "s3_key": "documents/165.pdf",
        "page_number": 1,
    }

    canned = _canned_result(engine_name, image_source="pdf", ocr_lang="en")
    with patch.object(handler, "get_s3_client", return_value=mock_s3), \
            patch.object(handler.engine, "process_pdf_pages", return_value=[canned]) as mock_proc_pdf, \
            patch.object(handler.engine, "process_image") as mock_proc_image:
        resp = handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert isinstance(body, dict)
    mock_proc_image.assert_not_called()
    mock_proc_pdf.assert_called_once()

    # PDF key was fetched (not an image key)
    _, kwargs = mock_s3.get_object.call_args
    assert kwargs["Key"] == "documents/165.pdf"

    result = body["results"][0]
    # original keys preserved, plus exactly the two new keys
    assert ORIGINAL_RESULT_KEYS.issubset(result.keys())
    assert "ocr_lang" in result
    assert "image_source" in result
    assert result["image_source"] == "pdf"


# ==============================================================================
# (d) S3 image fetch error -> falls back to PDF path
# ==============================================================================

@pytest.mark.parametrize("handler, engine, engine_name", ENGINE_PAIRS)
def test_image_fetch_error_falls_back_to_pdf(handler, engine, engine_name):
    def _get_object(Bucket, Key):
        if Key == "pages/165/1.jpg":
            raise RuntimeError("image not found")
        return {"Body": MagicMock(read=lambda: b"%PDF-1.4 fake")}

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = _get_object

    event = {
        "document_id": "165.pdf",
        "s3_bucket": "pdf-bucket",
        "s3_key": "documents/165.pdf",
        "image_s3_bucket": "image-bucket",
        "image_s3_key": "pages/165/1.jpg",
        "page_number": 1,
    }

    canned = _canned_result(engine_name, image_source="pdf", ocr_lang="en")
    with patch.object(handler, "get_s3_client", return_value=mock_s3), \
            patch.object(handler.engine, "process_pdf_pages", return_value=[canned]) as mock_proc_pdf, \
            patch.object(handler.engine, "process_image") as mock_proc_image:
        resp = handler.lambda_handler(event, DummyContext())

    assert resp["statusCode"] == 200
    body = resp["body"]
    assert isinstance(body, dict)
    mock_proc_image.assert_not_called()
    mock_proc_pdf.assert_called_once()
    assert body["results"][0]["image_source"] == "pdf"


# ==============================================================================
# (e) lang cache at the engine level
# ==============================================================================

@pytest.mark.parametrize("engine", [paddle_printed_engine, paddle_hand_engine],
                         ids=["printed", "handwritten"])
def test_lang_cache_constructs_once_per_lang(engine, monkeypatch):
    # Fake paddleocr module so no real package/model is needed.
    fake_module = types.ModuleType("paddleocr")
    constructed = []

    class FakePaddleOCR:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs.get("lang"))

        def ocr(self, image_input):
            return []

    fake_module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    # Clear the module-level cache so the test is deterministic.
    engine._paddle_engines.clear()

    img = np.zeros((1, 1, 3), dtype=np.uint8)

    engine.process_image(img, 1, lang="en")
    engine.process_image(img, 1, lang="en")
    assert constructed.count("en") == 1  # same lang -> constructed once

    engine.process_image(img, 1, lang="fr")
    assert constructed.count("fr") == 1  # different lang -> a second construction
    assert len(constructed) == 2

    engine._paddle_engines.clear()
