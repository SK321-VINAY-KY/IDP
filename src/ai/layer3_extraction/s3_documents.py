"""
File: s3_documents.py
Purpose: S3-backed storage for source PDF documents.

Source PDFs live in S3 (bucket from settings.s3_bucket) under a documents
prefix, instead of the local `dataset/` folder. Shared (admin) documents use:

    <prefix>/<filename>.pdf

Per-user documents are isolated under:

    <prefix>/_users/<username>/<filename>.pdf

All functions degrade gracefully (return empty / None / False) when the bucket
is not configured or an S3 error occurs, so callers can treat "no S3" the same
as "no documents".

Owner: engineer-a@idp-pilot
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import List, Optional

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_USERS_SEGMENT = "_users"


def is_configured() -> bool:
    """True if an S3 bucket is configured for document storage."""
    return bool(getattr(settings, "s3_bucket", "") or "")


def _client():
    import boto3  # imported lazily so environments without boto3 still import this module
    region = getattr(settings, "s3_region", None) or "ap-south-1"
    return boto3.client("s3", region_name=region)


def _prefix() -> str:
    p = (getattr(settings, "s3_documents_prefix", "documents") or "documents").strip("/")
    return p


def _shared_prefix() -> str:
    return f"{_prefix()}/"


def _user_prefix(username: str) -> str:
    uname = (username or "").strip().lower()
    return f"{_prefix()}/{_USERS_SEGMENT}/{uname}/"


def _shared_key(filename: str) -> str:
    return f"{_shared_prefix()}{filename}"


def _user_key(username: str, filename: str) -> str:
    return f"{_user_prefix(username)}{filename}"


def _is_pdf_key(key: str) -> bool:
    return key.lower().endswith(".pdf")


def _list_objects(prefix: str, recursive: bool = False) -> List[dict]:
    """
    List object summaries under a prefix. When recursive is False, only objects
    directly under the prefix (no deeper "folders") are returned.
    """
    if not is_configured():
        return []
    bucket = settings.s3_bucket
    out: List[dict] = []
    try:
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key == prefix or key.endswith("/"):
                    continue
                out.append({
                    "key": key,
                    "name": key.rsplit("/", 1)[-1],
                    "size": obj.get("Size", 0),
                    "modified": (obj.get("LastModified") or datetime.now(timezone.utc)).isoformat(),
                })
    except Exception as exc:
        logger.warning("s3_documents.list_failed", prefix=prefix, error=str(exc))
    return out


def list_shared_documents() -> List[dict]:
    """List shared (admin dataset) PDFs directly under the documents prefix."""
    return [d for d in _list_objects(_shared_prefix(), recursive=False) if _is_pdf_key(d["key"])]


def list_user_documents(username: str) -> List[dict]:
    """List a single user's PDFs under their isolated prefix."""
    return [d for d in _list_objects(_user_prefix(username), recursive=False) if _is_pdf_key(d["key"])]


def upload_shared_document(filename: str, data: bytes, content_type: str = "application/pdf") -> Optional[str]:
    """Upload a shared document to S3. Returns the object key, or None on failure."""
    return _put(_shared_key(filename), data, content_type)


def upload_user_document(username: str, filename: str, data: bytes, content_type: str = "application/pdf") -> Optional[str]:
    """Upload a per-user document to S3. Returns the object key, or None on failure."""
    return _put(_user_key(username, filename), data, content_type)


def _put(key: str, data: bytes, content_type: str) -> Optional[str]:
    if not is_configured():
        logger.warning("s3_documents.put_skipped_no_bucket", key=key)
        return None
    try:
        _client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type or "application/pdf",
        )
        logger.info("s3_documents.uploaded", key=key, size=len(data))
        return key
    except Exception as exc:
        logger.error("s3_documents.put_failed", key=key, error=str(exc))
        return None


def download_shared_document(filename: str) -> Optional[bytes]:
    return _get(_shared_key(filename))


def download_user_document(username: str, filename: str) -> Optional[bytes]:
    return _get(_user_key(username, filename))


def download_bytes_by_key(key: str) -> Optional[bytes]:
    return _get(key)


def _get(key: str) -> Optional[bytes]:
    if not is_configured():
        return None
    try:
        resp = _client().get_object(Bucket=settings.s3_bucket, Key=key)
        return resp["Body"].read()
    except Exception as exc:
        logger.warning("s3_documents.get_failed", key=key, error=str(exc))
        return None


def shared_document_exists(filename: str) -> bool:
    return _exists(_shared_key(filename))


def user_document_exists(username: str, filename: str) -> bool:
    return _exists(_user_key(username, filename))


def _exists(key: str) -> bool:
    if not is_configured():
        return False
    try:
        _client().head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except Exception:
        return False


def delete_shared_document(filename: str) -> bool:
    return _delete(_shared_key(filename))


def delete_user_document(username: str, filename: str) -> bool:
    return _delete(_user_key(username, filename))


def _delete(key: str) -> bool:
    if not is_configured():
        return False
    try:
        _client().delete_object(Bucket=settings.s3_bucket, Key=key)
        logger.info("s3_documents.deleted", key=key)
        return True
    except Exception as exc:
        logger.error("s3_documents.delete_failed", key=key, error=str(exc))
        return False
