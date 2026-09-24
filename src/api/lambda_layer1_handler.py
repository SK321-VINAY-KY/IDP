"""
File: lambda_layer1_handler.py
Purpose: AWS Lambda entry point for Layer 1 (Page Inspection & Capability Routing).
Owner: engineer-a@idp-pilot
Dependencies: pymupdf, pydantic, pydantic-settings, boto3 (standard AWS Lambda SDK)
"""
import base64
import io
import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import boto3
import fitz  # PyMuPDF

from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import (
    build_engine_plan,
    capabilities_from_profile,
    route_from_profile,
)
from src.config.settings import settings
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger("layer1_lambda")
_s3_client: Optional[Any] = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _parse_event_target(event: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[bytes]]:
    """
    Extracts bucket, object key, or raw PDF bytes from various Lambda invocation payloads:
    1. S3 ObjectCreated event: event['Records'][0]['s3']
    2. Direct JSON invocation: {'s3_bucket': ..., 's3_key': ...} or {'bucket': ..., 'key': ...}
    3. Direct base64 PDF: {'pdf_base64': ...}
    """
    if "Records" in event and len(event["Records"]) > 0:
        record = event["Records"][0]
        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            raw_key = record["s3"]["object"]["key"]
            key = urllib.parse.unquote_plus(raw_key)
            return bucket, key, None

    bucket = event.get("s3_bucket") or event.get("bucket")
    key = event.get("s3_key") or event.get("key")
    if bucket and key:
        return bucket, urllib.parse.unquote_plus(key), None

    if "pdf_base64" in event:
        pdf_bytes = base64.b64decode(event["pdf_base64"])
        doc_name = event.get("document_name", "inline_document.pdf")
        return None, doc_name, pdf_bytes

    return None, None, None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for Layer 1 document inspection and routing.
    """
    request_id = getattr(context, "aws_request_id", "local-test-req")
    set_correlation_id(request_id)

    bucket, key, pdf_bytes = _parse_event_target(event)

    if not key:
        err_msg = "Invalid invocation event. Must supply 's3_bucket' and 's3_key', or 'pdf_base64', or S3 Records."
        logger.error("layer1.invalid_event", error=err_msg, event_keys=list(event.keys()))
        return {
            "statusCode": 400,
            "body": json.dumps({"error": err_msg}),
        }

    logger.info("layer1.start", bucket=bucket, key=key)

    # Fetch PDF stream from S3 or use provided bytes
    if pdf_bytes is None:
        try:
            s3 = get_s3_client()
            s3_response = s3.get_object(Bucket=bucket, Key=key)
            pdf_bytes = s3_response["Body"].read()
        except Exception as exc:
            logger.error("layer1.s3_read_failed", bucket=bucket, key=key, error=str(exc))
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": f"Failed to read s3://{bucket}/{key}: {str(exc)}",
                    "bucket": bucket,
                    "key": key,
                }),
            }

    # Open PDF with PyMuPDF in memory
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.error("layer1.pdf_open_failed", key=key, error=str(exc))
        return {
            "statusCode": 422,
            "body": json.dumps({
                "error": f"Failed to parse PDF document with PyMuPDF: {str(exc)}",
                "key": key,
            }),
        }

    total_pages = len(doc)
    page_reports: List[Dict[str, Any]] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Step 1: Programmatic inspection (~10ms)
        profile = inspect_page(page, page_num)

        # Step 2: Routing decision
        if settings.routing_mode == "capability_based":
            caps = capabilities_from_profile(profile)
            plan = build_engine_plan(caps)
            tasks = [
                {"engine": task.engine, "priority": task.priority, "reason": task.reason}
                for task in plan
            ]
            active_caps = caps.active_capabilities()
            primary_route = plan[0].engine if plan else "skip"
            needs_vlm = (
                not caps.is_blank
                and not caps.has_indic_script
                and (caps.has_printed_scan or caps.has_handwriting)
                and (profile.is_scanned or profile.complexity_score >= 4 or
                     profile.image_coverage > settings.mixed_content_min_image_coverage)
            )
        else:
            single_route = route_from_profile(profile)
            primary_route = single_route or "vlm_transcribe"
            tasks = [
                {"engine": primary_route, "priority": 1, "reason": "single_engine_route"}
            ]
            active_caps = ["has_digital_text"] if profile.has_text else ["has_printed_scan"]
            needs_vlm = (single_route is None or profile.complexity_score >= 4)

        logger.info(
            "layer1.page_routed",
            page_number=page_num,
            primary_route=primary_route,
            tasks=[t["engine"] for t in tasks],
            needs_vlm=needs_vlm,
            complexity=profile.complexity_score,
        )

        page_reports.append({
            "page_number": page_num,
            "profile": profile.model_dump(),
            "route": primary_route,
            "capabilities": active_caps,
            "tasks": tasks,
            "needs_vlm": needs_vlm,
        })

    doc.close()

    result_payload = {
        "document_name": os.path.basename(key),
        "s3_bucket": bucket,
        "s3_key": key,
        "total_pages": total_pages,
        "routing_mode": settings.routing_mode,
        "pages": page_reports,
    }

    logger.info("layer1.complete", document_name=os.path.basename(key), total_pages=total_pages)

    return {
        "statusCode": 200,
        "body": result_payload,
    }
