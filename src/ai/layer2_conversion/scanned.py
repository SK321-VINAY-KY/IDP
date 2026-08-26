"""
File: scanned.py
Purpose: Layer 2 "scanned" and "handwritten" routes — PaddleOCR v6 for both
         printed and handwriting-tuned modes. CPU-viable, fast, cheap.
         Confidence rolls up from per-word scores via a shared helper.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (mode-aware engine, handwritten route)
Deps: paddleocr, numpy
"""
from typing import Any

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Per-mode singleton cache — each mode may carry distinct PaddleOCR settings,
# so we cannot share a single instance between printed and handwritten pages.
_paddle_engines: dict[str, Any] = {}


def _get_paddle_engine(mode: str = "printed") -> Any:
    """
    Return a lazily-loaded PaddleOCR engine for the given mode.

    mode="printed"     — default settings, det_db_thresh=0.3 (PaddleOCR default)
    mode="handwritten" — lower det_db_thresh (from settings) so thinner/more
                         irregular handwriting strokes are not missed at detection;
                         use_angle_cls=True kept because handwriting line angles
                         are less predictable than printed scans.
    """
    global _paddle_engines
    if mode not in _paddle_engines:
        from paddleocr import PaddleOCR

        if mode == "handwritten":
            # PaddleOCR v3 API: use_textline_orientation replaces use_angle_cls,
            # text_det_thresh replaces det_db_thresh, device="cpu" replaces use_gpu.
            # enable_mkldnn=False avoids an oneDNN ConvertPirAttribute crash on
            # CPUs without full oneDNN support.
            engine = PaddleOCR(
                lang="en",
                device="cpu",
                enable_mkldnn=False,
                use_textline_orientation=True,
                text_det_thresh=settings.paddle_handwriting_det_db_thresh,
            )
            logger.info(
                "layer2.paddle_engine_loaded",
                mode=mode,
                text_det_thresh=settings.paddle_handwriting_det_db_thresh,
                device="cpu",
            )
        else:  # "printed" — keep all PaddleOCR defaults
            engine = PaddleOCR(
                lang="en",
                device="cpu",
                enable_mkldnn=False,
                use_textline_orientation=True,
            )
            logger.info("layer2.paddle_engine_loaded", mode=mode, device="cpu")

        _paddle_engines[mode] = engine

    return _paddle_engines[mode]


# ---------------------------------------------------------------------------
# Shared confidence rollup helper
# ---------------------------------------------------------------------------

def _rollup_confidence(result: Any) -> tuple[list[str], list[float]]:
    """
    Parse raw PaddleOCR v3 result into (lines, word_confidences).

    PaddleOCR v3 changed the result format from the v2 nested list
    [[bbox, (text, conf)], ...] to a list of page dicts, each with
    'rec_texts' (list[str]) and 'rec_scores' (list[float]) at the top level.

    Shared by both the scanned and handwritten converters so the rollup
    logic is never duplicated.
    """
    lines: list[str] = []
    word_confidences: list[float] = []

    for page_result in result or []:
        texts = page_result.get("rec_texts") or []
        scores = page_result.get("rec_scores") or []
        for text, conf in zip(texts, scores):
            if text and text.strip():
                lines.append(text)
                word_confidences.append(float(conf))

    return lines, word_confidences


# ---------------------------------------------------------------------------
# Scanned (printed) converter — unchanged contract
# ---------------------------------------------------------------------------

def convert_scanned_page(image_array: Any, page_number: int) -> tuple[str, float]:
    """
    Converts a single scanned/printed page image (numpy array) to markdown-ish
    text via PaddleOCR. Returns (text, avg_confidence).
    """
    engine = _get_paddle_engine(mode="printed")

    try:
        result = engine.ocr(image_array)
        lines, word_confidences = _rollup_confidence(result)

        markdown = "\n".join(lines)
        avg_confidence = (
            sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
        )

        logger.info(
            "layer2.scanned.converted",
            page_number=page_number,
            line_count=len(lines),
            avg_confidence=round(avg_confidence, 4),
        )
        return markdown, avg_confidence

    except Exception as exc:
        logger.error(
            "layer2.scanned.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0


# ---------------------------------------------------------------------------
# Handwritten converter — PaddleOCR with handwriting-tuned config
# ---------------------------------------------------------------------------

def convert_handwritten_via_paddle(image_array: Any, page_number: int) -> tuple[str, float]:
    """
    Converts a handwritten page image (numpy array) to text via PaddleOCR
    using the "handwritten" mode engine (lower det_db_thresh, angle cls on).

    PaddleOCR's built-in DBNet text detector segments lines automatically,
    so no external line_segmentation step is needed — this is the root-cause
    fix for the TrOCR whole-page collapse that produced ~0.34 avg confidence.

    Returns (text, avg_confidence) with the same contract as convert_scanned_page().
    """
    engine = _get_paddle_engine(mode="handwritten")

    try:
        result = engine.ocr(image_array)
        lines, word_confidences = _rollup_confidence(result)

        markdown = "\n".join(lines)
        avg_confidence = (
            sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
        )

        logger.info(
            "layer2.handwritten.converted",
            page_number=page_number,
            line_count=len(lines),
            avg_confidence=round(avg_confidence, 4),
        )
        return markdown, avg_confidence

    except Exception as exc:
        logger.error(
            "layer2.handwritten.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0


# ---------------------------------------------------------------------------
# Secondary quality signal (unchanged)
# ---------------------------------------------------------------------------

def low_confidence_word_ratio(image_array: Any, threshold: float = 0.6) -> float:
    """
    What fraction of words fell below a per-word confidence threshold.
    A page can have a deceptively OK average while having a large garbled
    minority — this catches that case for Engineer B's quality-scoring module.
    Uses the printed engine since this is a diagnostic helper, not a converter.
    """
    engine = _get_paddle_engine(mode="printed")
    result = engine.ocr(image_array)

    total, low = 0, 0
    for page_result in result or []:
        texts = page_result.get("rec_texts") or []
        scores = page_result.get("rec_scores") or []
        for _text, conf in zip(texts, scores):
            total += 1
            if float(conf) < threshold:
                low += 1

    return (low / total) if total > 0 else 0.0
