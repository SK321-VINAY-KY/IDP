"""
File: engine.py
Purpose: Isolated PaddleOCR engine tuned for handwritten / cursive text.
Owner: engineer-a@idp-pilot
"""
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure pre-baked models from /opt/paddlex are available in Lambda's writable /tmp/.paddlex
if os.path.exists("/opt/paddlex") and not os.path.exists("/tmp/.paddlex"):
    try:
        os.symlink("/opt/paddlex", "/tmp/.paddlex")
    except Exception:
        shutil.copytree("/opt/paddlex", "/tmp/.paddlex", dirs_exist_ok=True)

if os.path.exists("/opt/paddleocr") and not os.path.exists("/tmp/.paddleocr"):
    try:
        os.symlink("/opt/paddleocr", "/tmp/.paddleocr")
    except Exception:
        shutil.copytree("/opt/paddleocr", "/tmp/.paddleocr", dirs_exist_ok=True)

_paddle_engine: Any = None


def get_paddle_engine() -> Any:
    """Lazy singleton initializer for PaddleOCR handwritten engine (det_db_thresh=0.20)."""
    global _paddle_engine
    if _paddle_engine is None:
        from paddleocr import PaddleOCR
        _paddle_engine = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            use_textline_orientation=True,
            text_det_thresh=0.2,  # Lower threshold for irregular handwriting stroke fragments
        )
    return _paddle_engine


def _rollup_confidence(result: Any) -> Tuple[List[str], List[float]]:
    """
    Parse raw PaddleOCR v2 or v3 result into (lines, word_confidences).
    """
    lines: List[str] = []
    word_confidences: List[float] = []

    for page_result in result or []:
        if page_result is None:
            continue

        # PaddleOCR v3 dict format: {"rec_texts": [...], "rec_scores": [...]}
        if isinstance(page_result, dict):
            texts = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            for text, conf in zip(texts, scores):
                if text and str(text).strip():
                    lines.append(str(text))
                    word_confidences.append(float(conf))

        # PaddleOCR v2 list format: [[bbox, (text, conf)], ...]
        elif isinstance(page_result, list):
            for line_item in page_result:
                if not line_item or len(line_item) < 2:
                    continue
                text_conf = line_item[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    t, c = text_conf[0], text_conf[1]
                    if t and str(t).strip():
                        lines.append(str(t))
                        word_confidences.append(float(c))

    return lines, word_confidences


def process_image(image_input: Any, page_number: int) -> Dict[str, Any]:
    """
    Executes PaddleOCR handwritten extraction on a numpy image array or image file path.
    """
    t0 = time.monotonic()
    try:
        engine = get_paddle_engine()
        ocr_result = engine.ocr(image_input)
        lines, confs = _rollup_confidence(ocr_result)
        markdown = "\n".join(lines)
        confidence = float(np.mean(confs)) if confs else 0.0
        latency_ms = (time.monotonic() - t0) * 1000

        return {
            "page_number": page_number,
            "engine": "paddleocr_handwritten",
            "markdown": markdown,
            "confidence": round(confidence, 4),
            "lines_extracted": len(lines),
            "latency_ms": round(latency_ms, 2),
            "status": "SUCCESS",
        }
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "page_number": page_number,
            "engine": "paddleocr_handwritten",
            "markdown": "",
            "confidence": 0.0,
            "lines_extracted": 0,
            "latency_ms": round(latency_ms, 2),
            "status": "ERROR",
            "error": str(exc),
        }


def process_pdf_pages(pdf_stream_or_path: Any, page_numbers: List[int]) -> List[Dict[str, Any]]:
    """
    Renders specified PDF pages into images and runs PaddleOCR handwritten extraction.
    """
    import fitz

    if isinstance(pdf_stream_or_path, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_stream_or_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_stream_or_path)

    mat = fitz.Matrix(2.0, 2.0)  # 144 DPI
    results = []

    try:
        for p in page_numbers:
            if p < 1 or p > len(doc):
                results.append({
                    "page_number": p,
                    "engine": "paddleocr_handwritten",
                    "markdown": "",
                    "confidence": 0.0,
                    "lines_extracted": 0,
                    "latency_ms": 0.0,
                    "status": "ERROR",
                    "error": f"Page number {p} out of range (document has {len(doc)} pages)",
                })
                continue

            page = doc[p - 1]
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            # If RGBA, slice to RGB
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            res = process_image(img_array, p)
            results.append(res)
    finally:
        doc.close()

    return results
