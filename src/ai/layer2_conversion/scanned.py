"""
File: scanned.py
Purpose: Layer 2 "scanned" and "handwritten" routes — PaddleOCR for both
         printed and handwriting-tuned modes. Supports seamless offloading to
         dedicated AWS Lambda PaddleOCR engines (idp-engine-paddle-printed,
         idp-engine-paddle-handwritten) with local in-process fallback.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-09-29 (dual-resilience AWS Lambda + local PaddleOCR)
Deps: boto3, paddleocr, numpy, pillow
"""
import base64
import io
import json
from typing import Any, Optional
import numpy as np
from PIL import Image

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Per-mode singleton cache for local PaddleOCR engines
_paddle_engines: dict[str, Any] = {}
_lambda_client: Any = None


def _get_lambda_client() -> Any:
    """Return a lazily-cached boto3 Lambda client with extended timeout for OCR."""
    global _lambda_client
    if _lambda_client is None:
        try:
            import boto3
            from botocore.config import Config
            region = (
                getattr(settings, "paddle_lambda_region", None)
                or getattr(settings, "bedrock_region", None)
                or "ap-south-1"
            )
            lambda_cfg = Config(
                read_timeout=300,
                connect_timeout=15,
                retries={"max_attempts": 0},
            )
            _lambda_client = boto3.client("lambda", region_name=region, config=lambda_cfg)
        except Exception as exc:
            logger.warning("layer2.paddle_lambda.client_init_failed", error=str(exc))
            _lambda_client = False
    return _lambda_client if _lambda_client is not False else None


def _invoke_paddle_lambda(
    image_array: Any,
    page_number: int,
    mode: str,
    image_bytes: Optional[bytes] = None,
) -> Optional[tuple[str, float]]:
    """
    Invokes the dedicated AWS Lambda PaddleOCR engine (printed or handwritten).
    Returns (markdown, avg_confidence) if successful, or None if unavailable/failed.
    """
    client = _get_lambda_client()
    if client is None:
        return None

    try:
        # Convert image input to base64 string
        if image_bytes:
            pil_img = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_array, np.ndarray):
            pil_img = Image.fromarray(image_array)
        elif isinstance(image_array, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_array))
        elif isinstance(image_array, Image.Image):
            pil_img = image_array
        else:
            logger.warning("layer2.paddle_lambda.unsupported_image_type", img_type=type(image_array).__name__)
            return None

        # Resize if max dimension exceeds 1200px to maintain high OCR fidelity while optimizing inference speed
        max_dim = max(pil_img.size)
        if max_dim > 1200:
            scale = 1200.0 / max_dim
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        fn_name = (
            getattr(settings, "paddle_handwritten_lambda_function", "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-handwritten")
            if mode == "handwritten"
            else getattr(settings, "paddle_printed_lambda_function", "arn:aws:lambda:ap-south-1:106611079163:function:idp-engine-paddle-printed")
        )

        payload = {
            "image_base64": b64_str,
            "page_number": page_number,
        }

        resp = client.invoke(
            FunctionName=fn_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )

        status_code = resp.get("StatusCode")
        if status_code not in (200, 202):
            logger.warning("layer2.paddle_lambda.http_error", status_code=status_code, mode=mode)
            return None

        raw_payload = resp["Payload"].read()
        res_data = json.loads(raw_payload)

        # Handle API gateway / Lambda proxy wrapper or direct dict
        body = res_data.get("body", res_data)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                pass

        if isinstance(body, dict):
            if "results" in body and body["results"]:
                r0 = body["results"][0]
                if r0.get("status") == "SUCCESS":
                    markdown = r0.get("markdown", "")
                    confidence = float(r0.get("confidence", 0.0))
                    logger.info(
                        f"layer2.{mode}.lambda_converted",
                        page_number=page_number,
                        lines_extracted=r0.get("lines_extracted", 0),
                        confidence=round(confidence, 4),
                        latency_ms=r0.get("latency_ms"),
                    )
                    return markdown, confidence
                else:
                    logger.warning("layer2.paddle_lambda.result_error", result=r0)
                    return None
            elif "error" in body:
                logger.warning("layer2.paddle_lambda.error_reported", error=body.get("error"))
                return None

        logger.warning("layer2.paddle_lambda.unrecognized_response", payload=res_data)
        return None

    except Exception as exc:
        logger.warning(
            "layer2.paddle_lambda.invoke_exception",
            mode=mode,
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


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
            for item in page_result:
                if item is None:
                    continue
                try:
                    text, conf = item[1]
                    if text and str(text).strip():
                        lines.append(str(text))
                        word_confidences.append(float(conf))
                except (IndexError, TypeError, ValueError):
                    continue

    return lines, word_confidences


# ---------------------------------------------------------------------------
# Internal shared OCR runner
# ---------------------------------------------------------------------------

def _ocr_convert(
    image_array: Any,
    page_number: int,
    mode: str,
    image_bytes: Optional[bytes] = None,
) -> tuple[str, float]:
    """
    Common execution path for PaddleOCR conversion (printed or handwritten).
    Prioritizes dedicated AWS Lambda PaddleOCR engine when available/preferred;
    falls back seamlessly to local in-process PaddleOCR engine.
    """
    # 1. Try AWS Lambda PaddleOCR offload first if preferred
    if getattr(settings, "paddle_prefer_lambda", True):
        lambda_result = _invoke_paddle_lambda(image_array, page_number, mode, image_bytes)
        if lambda_result is not None:
            return lambda_result

        # In production/staging, do not run heavy local OCR in the web container to prevent OOM
        if getattr(settings, "app_env", "development") in ("production", "staging"):
            logger.warning(
                f"layer2.{mode}.lambda_failed_skipping_local_in_prod",
                page_number=page_number,
            )
            return "", 0.0

    # 2. In-process PaddleOCR engine (local execution for dev/test)
    try:
        engine = _get_paddle_engine(mode=mode)
        result = engine.ocr(image_array)
        lines, word_confidences = _rollup_confidence(result)

        markdown = "\n".join(lines)
        avg_confidence = (
            sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
        )

        logger.info(
            f"layer2.{mode}.converted",
            page_number=page_number,
            line_count=len(lines),
            avg_confidence=round(avg_confidence, 4),
        )
        return markdown, avg_confidence

    except (ImportError, ModuleNotFoundError) as imp_err:
        logger.warning(
            f"layer2.{mode}.local_module_not_found",
            page_number=page_number,
            error=str(imp_err),
            fallback="lambda",
        )
        # If lambda was not preferred or failed initially, attempt Lambda invocation now
        lambda_result = _invoke_paddle_lambda(image_array, page_number, mode, image_bytes)
        if lambda_result is not None:
            return lambda_result

        logger.error(
            f"layer2.{mode}.failed",
            page_number=page_number,
            error=str(imp_err),
            error_type=type(imp_err).__name__,
        )
        return "", 0.0

    except Exception as exc:
        # If in-process execution failed with another error, try Lambda as a fallback
        logger.warning(
            f"layer2.{mode}.local_failed_trying_lambda",
            page_number=page_number,
            error=str(exc),
        )
        lambda_result = _invoke_paddle_lambda(image_array, page_number, mode, image_bytes)
        if lambda_result is not None:
            return lambda_result

        logger.error(
            f"layer2.{mode}.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0


# ---------------------------------------------------------------------------
# Scanned (printed) converter
# ---------------------------------------------------------------------------

def convert_scanned_page(
    image_array: Any,
    page_number: int,
    image_bytes: Optional[bytes] = None,
) -> tuple[str, float]:
    """
    Converts a single scanned/printed page image (numpy array or bytes) to markdown-ish
    text via PaddleOCR. Returns (text, avg_confidence).
    """
    return _ocr_convert(image_array, page_number, mode="printed", image_bytes=image_bytes)


# ---------------------------------------------------------------------------
# Handwritten converter — PaddleOCR with handwriting-tuned config
# ---------------------------------------------------------------------------

def convert_handwritten_via_paddle(
    image_array: Any,
    page_number: int,
    image_bytes: Optional[bytes] = None,
) -> tuple[str, float]:
    """
    Converts a handwritten page image (numpy array or bytes) to text via PaddleOCR
    using the "handwritten" mode engine (lower det_db_thresh, angle cls on).

    PaddleOCR's built-in DBNet text detector segments lines automatically,
    so no external line_segmentation step is needed.

    Returns (text, avg_confidence) with the same contract as convert_scanned_page().
    """
    return _ocr_convert(image_array, page_number, mode="handwritten", image_bytes=image_bytes)


# ---------------------------------------------------------------------------
# Secondary quality signal
# ---------------------------------------------------------------------------

def low_confidence_word_ratio(image_array: Any, threshold: float = 0.6) -> float:
    """
    What fraction of words fell below a per-word confidence threshold.
    A page can have a deceptively OK average while having a large garbled
    minority — this catches that case for Engineer B's quality-scoring module.
    Uses the printed engine since this is a diagnostic helper, not a converter.
    """
    try:
        engine = _get_paddle_engine(mode="printed")
        result = engine.ocr(image_array)
        _, word_confidences = _rollup_confidence(result)
        if not word_confidences:
            return 0.0
        low = sum(1 for conf in word_confidences if conf < threshold)
        return low / len(word_confidences)
    except Exception as exc:
        logger.warning("layer2.scanned.low_confidence_ratio_failed", error=str(exc))
        return 0.0
