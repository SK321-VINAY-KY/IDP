"""
File: handler.py
Purpose: AWS Lambda handler for isolated PaddleOCR printed conversion service.
Owner: engineer-a@idp-pilot
"""
import base64
import json
import os
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional

import boto3

try:
    import engine
except ImportError:
    from . import engine

_s3_client: Optional[Any] = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point for PaddleOCR printed text extraction.
    """
    request_id = getattr(context, "aws_request_id", str(uuid.uuid4()))

    # 1. Parse Event parameters
    s3_bucket = event.get("s3_bucket") or event.get("bucket")
    s3_key = event.get("s3_key") or event.get("key")
    if s3_key:
        s3_key = urllib.parse.unquote_plus(s3_key)

    document_id = event.get("document_id") or (os.path.basename(s3_key) if s3_key else f"doc_{request_id[:8]}")

    pages: List[int] = []
    if "pages" in event and isinstance(event["pages"], list):
        pages = [int(p) for p in event["pages"]]
    elif "page_number" in event:
        pages = [int(event["page_number"])]

    if not pages and "image_base64" not in event:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'pages' or 'page_number' in request payload."}),
        }

    try:
        # 2. Handle direct base64 image
        if "image_base64" in event and event["image_base64"]:
            import io
            from PIL import Image
            import numpy as np

            img_bytes = base64.b64decode(event["image_base64"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(img)
            target_page = pages[0] if pages else 1
            result = engine.process_image(img_array, target_page)
            results = [result]

        # 3. Handle PDF from S3 or base64
        elif s3_bucket and s3_key:
            s3 = get_s3_client()
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            pdf_bytes = response["Body"].read()
            results = engine.process_pdf_pages(pdf_bytes, pages)

        elif "pdf_base64" in event and event["pdf_base64"]:
            pdf_bytes = base64.b64decode(event["pdf_base64"])
            results = engine.process_pdf_pages(pdf_bytes, pages)

        elif "local_path" in event and os.path.exists(event["local_path"]):
            results = engine.process_pdf_pages(event["local_path"], pages)

        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Must supply 's3_bucket' and 's3_key', or 'pdf_base64', or 'image_base64'."}),
            }

        response_payload = {
            "document_id": document_id,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "engine": "paddleocr_printed",
            "results": results,
        }

        return {
            "statusCode": 200,
            "body": response_payload,
        }

    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "document_id": document_id,
                "engine": "paddleocr_printed",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }),
        }
