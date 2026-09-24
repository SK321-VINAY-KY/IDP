"""
File: handler.py
Purpose: AWS Lambda handler for isolated Docling digital conversion service.
Owner: engineer-a@idp-pilot
"""
import base64
import json
import os
import tempfile
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
    AWS Lambda entry point for Docling digital conversion.
    Supports both batch pages (`pages: [1, 2]`) and single page (`page_number: 1`).
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

    if not pages:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'pages' or 'page_number' in request payload."}),
        }

    # 2. Acquire PDF file on local disk (/tmp)
    temp_pdf_path = None
    try:
        if "pdf_base64" in event and event["pdf_base64"]:
            pdf_bytes = base64.b64decode(event["pdf_base64"])
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(pdf_bytes)
                temp_pdf_path = tf.name
        elif s3_bucket and s3_key:
            s3 = get_s3_client()
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            pdf_bytes = response["Body"].read()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(pdf_bytes)
                temp_pdf_path = tf.name
        elif "local_path" in event and os.path.exists(event["local_path"]):
            temp_pdf_path = event["local_path"]
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Must supply 's3_bucket' and 's3_key', or 'pdf_base64', or 'local_path'."}),
            }

        # 3. Execute Engine Conversion
        results = engine.process_pages(temp_pdf_path, pages)

        response_payload = {
            "document_id": document_id,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "engine": "docling",
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
                "engine": "docling",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }),
        }
    finally:
        # Cleanup temporary PDF file
        if temp_pdf_path and temp_pdf_path != event.get("local_path"):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass
