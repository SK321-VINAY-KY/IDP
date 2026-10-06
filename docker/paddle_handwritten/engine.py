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
            ocr_version="PP-OCRv6",
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            text_recognition_batch_size=16,
            text_det_thresh=0.2,  # Lower threshold for irregular handwriting stroke fragments
        )
    return _paddle_engine


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


def process_image(image_input: Any, page_number: int) -> Dict[str, Any]:
    """
    Executes PaddleOCR handwritten extraction on a numpy image array or image file path.
    """
    t0 = time.monotonic()
    try:
        engine = get_paddle_engine()
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
                    "blocks": [],
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


# Unified interface alias
process_pages = process_pdf_pages
