"""
File: paddle_block_rollup.py
Purpose: Shared PaddleOCR result-parsing utilities used by:
           - docker/paddle_printed/engine.py   (Lambda container)
           - docker/paddle_handwritten/engine.py (Lambda container)
           - src/ai/layer2_conversion/scanned.py (in-process local fallback)

         Previously each of those three files carried byte-identical copies
         of _bbox_from_quad and _rollup_confidence_with_blocks. This module
         is the single source of truth. Any bug fix or key-name change here
         propagates to all three consumers automatically.

Owner: engineer-a@idp-pilot
Created: 2026-10-06 (extracted from scanned.py / docker engine.py copies)
Deps: pydantic (via document_ir), no heavy runtime deps
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Dual-import pattern: works whether this module is imported from the src
# package root (in-process / tests) or packaged at the container root (Lambda).
try:
    from src.ai.schemas.document_ir import Block, BlockType, BBox
except ImportError:
    from document_ir import Block, BlockType, BBox  # type: ignore[no-redef]


def _bbox_from_quad(quad: Any) -> Optional[BBox]:
    """Collapse a PaddleOCR polygon/box into an axis-aligned BBox.

    Accepts two formats:
      * 4-point polygon  – [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] (PaddleOCR v2/v3)
      * Flat 4-coord box – [x0, y0, x1, y1]

    Returns None when the input is None or cannot be parsed.
    """
    if quad is None:
        return None
    try:
        # Flat [x0, y0, x1, y1] — elements are scalars, not sequences
        if hasattr(quad, "__len__") and len(quad) == 4 and not hasattr(quad[0], "__len__"):
            return BBox(
                x0=float(quad[0]),
                y0=float(quad[1]),
                x1=float(quad[2]),
                y1=float(quad[3]),
            )
        # 4-point polygon [[x, y], …]
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    except (TypeError, IndexError, KeyError, ValueError):
        return None


def rollup_confidence_with_blocks(
    result: Any,
    page_number: int,
    engine_name: str,
) -> Tuple[List[str], List[float], List[Block]]:
    """Parse a raw PaddleOCR ocr() result into (lines, word_confidences, blocks).

    Handles both PaddleOCR v2 and v3 output formats:

    v2 (list-of-lists):
        result = [ [quad, [text, conf]], … ]

    v3 (list-of-dicts):
        result = [ {"rec_texts": […], "rec_scores": […],
                    "rec_polys"|"dt_polys"|"rec_boxes": […]} ]

    Key fallback order for polygon coordinates (v3 dict format):
        rec_polys  →  dt_polys  →  rec_boxes  →  no bbox (None)

    Lines/blocks with empty or whitespace-only text are silently skipped.
    """
    lines: List[str] = []
    word_confidences: List[float] = []
    blocks: List[Block] = []
    order = 0

    for page_result in result or []:
        if page_result is None:
            continue

        # ── v3 dict format ──────────────────────────────────────────────
        if isinstance(page_result, dict):
            texts = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []

            # Polygon key resolution: verified against PP-OCRv4 / PP-OCRv6
            polys = page_result.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = page_result.get("dt_polys")
            if polys is None or len(polys) == 0:
                polys = page_result.get("rec_boxes") or []

            for i, (text, conf) in enumerate(zip(texts, scores)):
                if text and str(text).strip():
                    lines.append(str(text))
                    word_confidences.append(float(conf))
                    bbox = _bbox_from_quad(polys[i]) if i < len(polys) else None
                    blocks.append(Block(
                        block_id=f"p{page_number}-{engine_name}-{order}",
                        type=BlockType.LINE,
                        text=str(text),
                        bbox=bbox,
                        reading_order=order,
                        confidence=float(conf),
                        source_engine=engine_name,
                    ))
                    order += 1

        # ── v2 list-of-lists format ─────────────────────────────────────
        elif isinstance(page_result, list):
            for line_item in page_result:
                if not line_item or len(line_item) < 2:
                    continue
                quad, text_conf = line_item[0], line_item[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    t, c = text_conf[0], text_conf[1]
                    if t and str(t).strip():
                        lines.append(str(t))
                        word_confidences.append(float(c))
                        blocks.append(Block(
                            block_id=f"p{page_number}-{engine_name}-{order}",
                            type=BlockType.LINE,
                            text=str(t),
                            bbox=_bbox_from_quad(quad),
                            reading_order=order,
                            confidence=float(c),
                            source_engine=engine_name,
                        ))
                        order += 1

    return lines, word_confidences, blocks
