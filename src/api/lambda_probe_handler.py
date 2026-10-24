"""
File: lambda_probe_handler.py
Purpose: AWS Lambda entry point for the Layer 1.5 probe (script / handwriting / table
         detection on scanned pages + render-once page images to S3).
Invoked by Step Functions AFTER Layer 1 and BEFORE the page Map.

Input  (execution input shape already used by scripts/test_step_functions.py):
  {"document_id": "...", "s3_bucket": "...", "s3_key": "...",
   "pages": [{"page_number": 1, "route": "paddleocr_printed", "profile": {...optional...}}, ...]}
  `pages` is optional (probe then inspects the whole PDF itself). A Layer 1 response
  wrapped as {"body": {...}} is also accepted.
Output : {"statusCode": 200, "body": {document_id, s3_bucket, s3_key, doc_prior, pages[], stats}}
Errors RAISE (unlike the engines) so Step Functions Retry/Catch can fall back to Layer 1 routes.
"""
import os
import urllib.parse
from typing import Any, Dict

import boto3

from src.ai.layer1_probe.probe import probe_document
from src.ai.layer1_probe.types import ProbeConfig
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger("layer1_probe_lambda")

_s3: Any = None
_vlm: Any = None
_onnx: Any = None
_cfg: Any = None
_models_loaded = False


def _get_s3() -> Any:
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _load_models(cfg: ProbeConfig) -> None:
    """Module-level singletons: paid once per warm container."""
    global _vlm, _onnx, _models_loaded
    if _models_loaded:
        return
    from src.ai.layer1_probe import script_classifier
    model_path = cfg.onnx_model_path or script_classifier.default_model_path()
    labels_path = cfg.onnx_labels_path or script_classifier.default_labels_path()
    if os.path.exists(model_path):
        from src.ai.layer1_probe.script_classifier import OnnxScriptClassifier
        _onnx = OnnxScriptClassifier(model_path, labels_path)
        logger.info("probe.onnx_loaded", path=model_path)
    if cfg.vlm_enabled and cfg.vlm_model_id:
        from src.ai.layer1_probe.vlm_probe import BedrockScriptProbe
        _vlm = BedrockScriptProbe(cfg.vlm_model_id, cfg.vlm_region)
    elif cfg.vlm_enabled:
        logger.warning("probe.vlm_enabled_but_no_model_id")
    _models_loaded = True


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    global _cfg
    set_correlation_id(getattr(context, "aws_request_id", "local-probe"))
    if "body" in event and isinstance(event["body"], dict):
        event = event["body"]

    bucket = event.get("s3_bucket") or event.get("bucket")
    key = event.get("s3_key") or event.get("key")
    if not (bucket and key):
        raise ValueError("probe: 's3_bucket' and 's3_key' are required")
    key = urllib.parse.unquote_plus(key)
    document_id = event.get("document_id") or os.path.basename(key)

    _cfg = _cfg or ProbeConfig.from_env()
    _load_models(_cfg)

    s3 = _get_s3()
    head = s3.head_object(Bucket=bucket, Key=key)
    etag = f'{head.get("ETag", "").strip(chr(34))}-{head.get("ContentLength", 0)}'
    pdf_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    time_left = None
    if hasattr(context, "get_remaining_time_in_millis"):
        time_left = lambda: context.get_remaining_time_in_millis() / 1000.0  # noqa: E731

    logger.info("probe.start", document_id=document_id, bucket=bucket, key=key)
    body = probe_document(
        pdf_bytes, document_id=document_id, s3_bucket=bucket, s3_key=key,
        pages_in=event.get("pages"), cfg=_cfg, s3=s3, vlm=_vlm, onnx=_onnx,
        etag=etag, time_left=time_left,
    )
    return {"statusCode": 200, "body": _slim(body, s3, bucket, document_id, _cfg)}


_SLIM_FIELDS = ("page_number", "route", "ocr_lang", "use_structure", "status", "script",
                "has_table", "image_s3_bucket", "image_s3_key")


def _slim(body: Dict[str, Any], s3: Any, bucket: str, document_id: str, cfg: ProbeConfig) -> Dict[str, Any]:
    """Step Functions caps state payloads at 256KB. Keep the Map items small and park the
    full per-page report in S3 (probe-reports/{doc}.json)."""
    if os.environ.get("IDP_PROBE_SLIM_OUTPUT", "1") != "1":
        return body
    import json
    import re
    report_key = f"probe-reports/{re.sub(r'[^A-Za-z0-9._-]', '_', document_id)}.json"
    try:
        s3.put_object(Bucket=cfg.image_bucket or bucket, Key=report_key,
                      Body=json.dumps(body).encode(), ContentType="application/json")
    except Exception as exc:
        logger.warning("probe.report_write_failed", error=str(exc))
        report_key = None
    slim = dict(body)
    slim["pages"] = [{k: p.get(k) for k in _SLIM_FIELDS} for p in body["pages"]]
    slim["report_s3_key"] = report_key
    return slim


handler = lambda_handler
