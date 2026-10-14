"""
File: ir_store.py
Purpose: Phase 4, commit 1 — per-owner persistence of DocumentIR PageIR objects,
         so downstream chunking/retrieval can carry block_ids + bbox and the
         evidence resolver can run against the real persisted blocks.

TENANT ISOLATION:
  Files are written under the SAME owner-partitioned convention already used by
  llm_metrics (dataset_output/_users/<owner>/...). `owner` is REQUIRED (no default)
  on every public function. doc_id is sanitized with the SAME rule llm_metrics uses
  (alnum + "._-" kept, everything else -> "_", capped length), so a doc_id like
  "../../evil" cannot escape the owner directory. A read with the wrong owner looks
  in that owner's directory only and returns None — it cannot reach another owner's,
  an ownerless, or a `_legacy` file (there is no cross-owner or sentinel path here).

FILE FORMAT (deterministic, UTF-8):
  {
    "schema_version": 1,
    "doc_id": "<original doc_id>",
    "pages": [ <PageIR.model_dump()>, ... ]   # sorted by page_number
  }
  Atomic write: a temp file in the same dir is fully written+flushed, then
  os.replace()'d over the target (atomic on Windows and POSIX).

CONTENT SAFETY: IR files contain document text. This module NEVER logs, prints, or
returns that text in warnings — only counts, ids, categories, and the path.

NO MODEL / NO AWS: pure local disk + json. owner is required; nothing here calls
Bedrock, S3, or any network service.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from src.ai.schemas.document_ir import PageIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Bump when the on-disk shape changes; a file with a different version loads as None.
IR_SCHEMA_VERSION = 1

# Root mirrors the existing owner-partitioned convention (see llm_metrics.write_summary:
# dataset_output/_users/<owner>/metrics/). We use a sibling "ir" subdir.
_DATASET_ROOT = Path("dataset_output")
_USERS_SEGMENT = "_users"


def _safe_owner(owner: str) -> str:
    """Normalize owner the same way the metrics path does: strip + lowercase.
    Raises on empty owner (owner is required, no default)."""
    if not owner or not str(owner).strip():
        raise ValueError("owner is required and cannot be empty")
    return str(owner).strip().lower()


def _safe_doc_id(doc_id: str) -> str:
    """Sanitize a doc_id into a single path-safe filename segment, IDENTICAL to the
    rule in llm_metrics.write_summary. Keeps alnum and '._-', maps everything else
    (including '/', '\\', and '..' separators) to '_', and caps length. This makes
    traversal doc_ids like '../../evil' collapse to a harmless flat name."""
    if doc_id is None or not str(doc_id).strip():
        raise ValueError("doc_id is required and cannot be empty")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(doc_id))[:120]


def _ir_dir(owner: str, output_root: Optional[Path] = None) -> Path:
    root = output_root or (_DATASET_ROOT / _USERS_SEGMENT / _safe_owner(owner) / "ir")
    return root


def ir_path(owner: str, doc_id: str, output_root: Optional[Path] = None) -> Path:
    """Return the owner-scoped path for a document's IR file. owner required."""
    # _safe_owner is applied inside _ir_dir when output_root is None; apply here too
    # so an explicit output_root still gets a sanitized, flat doc filename.
    safe_doc = _safe_doc_id(doc_id)
    base = output_root if output_root is not None else _ir_dir(owner)
    return Path(base) / f"{safe_doc}.ir.json"


def save_page_irs(
    owner: str,
    doc_id: str,
    pages: List[PageIR],
    output_root: Optional[Path] = None,
) -> Path:
    """Persist a document's PageIR list under the owner partition. owner REQUIRED.

    Deterministic serialization (pages sorted by page_number, json indent=2,
    sort_keys), atomic write (temp file -> os.replace). Returns the path written.
    Never logs document text."""
    _safe_owner(owner)  # validate (raises on empty)
    path = ir_path(owner, doc_id, output_root=output_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(pages, key=lambda p: p.page_number)
    payload = {
        "schema_version": IR_SCHEMA_VERSION,
        "doc_id": str(doc_id),
        "pages": [p.model_dump() for p in ordered],
    }
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    # Atomic write: temp file in the SAME directory, then os.replace.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".ir_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        raise

    logger.info(
        "ir_store.saved",
        owner=_safe_owner(owner),
        doc_id=_safe_doc_id(doc_id),
        pages=len(ordered),
        blocks=sum(len(p.blocks) for p in ordered),
        path=str(path),
    )
    return path


def load_page_irs(
    owner: str,
    doc_id: str,
    output_root: Optional[Path] = None,
) -> Optional[List[PageIR]]:
    """Load a document's PageIR list for `owner`. Returns None (never raises) when:
      - the file is missing,
      - the file is corrupted / not valid JSON,
      - the schema_version doesn't match IR_SCHEMA_VERSION,
      - the payload shape is invalid.
    Warnings are metadata-only (owner, doc_id, reason) — never the file content."""
    _safe_owner(owner)
    path = ir_path(owner, doc_id, output_root=output_root)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        logger.warning("ir_store.load_corrupted",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="unreadable_or_bad_json")
        return None
    if not isinstance(payload, dict):
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="not_an_object")
        return None
    if payload.get("schema_version") != IR_SCHEMA_VERSION:
        logger.warning("ir_store.load_version_mismatch",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id),
                       reason="schema_version_mismatch", found=payload.get("schema_version"))
        return None
    pages_raw = payload.get("pages")
    if not isinstance(pages_raw, list):
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="pages_not_a_list")
        return None
    try:
        return [PageIR.model_validate(p) for p in pages_raw]
    except Exception:
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="pageir_validation_failed")
        return None
