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

_paddle_engines: Dict[str, Any] = {}


def _normalize_lang(lang: Optional[str]) -> str:
    """Normalize an optional lang code: None or empty -> 'en'."""
    if lang is None:
        return "en"
    lang = str(lang).strip()
    return lang or "en"


def get_paddle_engine(lang: Optional[str] = None) -> Any:
    """Lazy, lang-keyed cache initializer for PaddleOCR handwritten engine (text_det_thresh=0.20).

    Keeps the existing constructor kwargs unchanged; only `lang` varies.
    """
    norm_lang = _normalize_lang(lang)
    engine = _paddle_engines.get(norm_lang)
    if engine is None:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang=norm_lang,
            device="cpu",
            enable_mkldnn=False,
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            text_recognition_batch_size=16,
            text_det_thresh=0.2,  # Lower threshold for irregular handwriting stroke fragments
        )
        _paddle_engines[norm_lang] = engine
    return engine


try:
    from src.ai.layer2_conversion.paddle_block_rollup import (
        _bbox_from_quad,
        rollup_confidence_with_blocks as _rollup_confidence_with_blocks,
    )
    from src.ai.schemas.document_ir import Block, BlockType, BBox
except ImportError:
    # Packaged at container root alongside document_ir.py
    from paddle_block_rollup import (  # type: ignore[no-redef]
        _bbox_from_quad,
        rollup_confidence_with_blocks as _rollup_confidence_with_blocks,
    )
    from document_ir import Block, BlockType, BBox  # type: ignore[no-redef]


def _rollup_confidence(result: Any) -> Tuple[List[str], List[float]]:
    """Backward compatibility shim returning (lines, word_confidences)."""
    lines, confs, _ = _rollup_confidence_with_blocks(result, page_number=1, engine_name="paddleocr_handwritten")
    return lines, confs


def process_image(
    image_input: Any,
    page_number: int,
    lang: Optional[str] = None,
    image_source: str = "pdf",
) -> Dict[str, Any]:
    """
    Executes PaddleOCR handwritten extraction on a numpy image array or image file path.
    """
    norm_lang = _normalize_lang(lang)
    t0 = time.monotonic()
    try:
        engine = get_paddle_engine(norm_lang)
        ocr_result = engine.ocr(image_input)
        lines, confs, blocks = _rollup_confidence_with_blocks(
            ocr_result, page_number, "paddleocr_handwritten"
        )
        markdown = "\n".join(lines)
        confidence = float(np.mean(confs)) if confs else 0.0
        latency_ms = (time.monotonic() - t0) * 1000

        return {
            "page_number": page_number,
            "engine": "paddleocr_handwritten",
            "markdown": markdown,
            "confidence": round(confidence, 4),
            "lines_extracted": len(lines),
            "blocks": [b.model_dump() for b in blocks],
            "latency_ms": round(latency_ms, 2),
            "status": "SUCCESS",
            "ocr_lang": norm_lang,
            "image_source": image_source,
        }
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "page_number": page_number,
            "engine": "paddleocr_handwritten",
            "markdown": "",
            "confidence": 0.0,
            "lines_extracted": 0,
            "blocks": [],
            "latency_ms": round(latency_ms, 2),
            "status": "ERROR",
            "error": str(exc),
            "ocr_lang": norm_lang,
            "image_source": image_source,
        }


def process_pdf_pages(
    pdf_stream_or_path: Any,
    page_numbers: List[int],
    lang: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Renders specified PDF pages into images and runs PaddleOCR handwritten extraction.
    """
    import fitz

    norm_lang = _normalize_lang(lang)

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
                    "blocks": [],
                    "latency_ms": 0.0,
                    "status": "ERROR",
                    "error": f"Page number {p} out of range (document has {len(doc)} pages)",
                    "ocr_lang": norm_lang,
                    "image_source": "pdf",
                })
                continue

            page = doc[p - 1]
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            # If RGBA, slice to RGB
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            res = process_image(img_array, p, lang=norm_lang, image_source="pdf")
            results.append(res)
    finally:
        doc.close()

    return results


# Unified interface alias
process_pages = process_pdf_pages
