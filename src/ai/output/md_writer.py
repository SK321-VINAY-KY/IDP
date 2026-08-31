"""
File: md_writer.py
Purpose: Writes pipeline PageOutput results to structured Markdown files.
         One .md file per source document — pages are concatenated in order
         with a standard per-page header block carrying provenance metadata.

         Architectural position:
             pipeline.process_document() → list[(PageOutput, PageMetadata)]
                                                         │
                                                         ▼
                                               md_writer.write_document()
                                                         │
                                                         ▼
                                           output/{document_stem}.md

         The writer is intentionally dumb — it never modifies, re-ranks, or
         summarises content. Raw extracted markdown in, readable .md file out.
         Any downstream enrichment (field extraction, schema application,
         LLM-based structuring) is Layer 3 (Engineer B) territory.

Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: stdlib only
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.ai.schemas.page import PageOutput
from src.ai.schemas.page_metadata import PageMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_document(
    document_name: str,
    pages: List[Tuple[PageOutput, PageMetadata]],
    output_dir: str,
    overwrite: bool = True,
) -> str:
    """
    Write all pages of one document to a single Markdown file.

    Args:
        document_name:  Original PDF filename (e.g. "MY_resume.pdf").
                        Used in the file header and to derive the output name.
        pages:          Ordered list of (PageOutput, PageMetadata) pairs
                        as returned by pipeline.process_document().
        output_dir:     Directory to write the .md file into.
                        Created automatically if it does not exist.
        overwrite:      If False and the file already exists, raise FileExistsError.

    Returns:
        Absolute path to the written .md file.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem      = _stem(document_name)
    out_path  = os.path.join(output_dir, f"{stem}.md")

    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"Output file already exists: {out_path}")

    lines: List[str] = []

    # ── Document header ────────────────────────────────────────────────
    lines += _document_header(document_name, pages)

    # ── Per-page content ───────────────────────────────────────────────
    for output, metadata in pages:
        lines.append("")
        lines += _page_block(output, metadata)

    # ── Document footer ────────────────────────────────────────────────
    lines += _document_footer(pages)

    content = "\n".join(lines) + "\n"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    total_chars = sum(len(o.markdown.strip()) for o, _ in pages)
    logger.info(
        "md_writer.document_written",
        document=document_name,
        output_path=out_path,
        pages=len(pages),
        total_chars=total_chars,
    )
    return os.path.abspath(out_path)


def write_documents(
    documents: List[Tuple[str, List[Tuple[PageOutput, PageMetadata]]]],
    output_dir: str,
    overwrite: bool = True,
) -> List[str]:
    """
    Convenience wrapper — write multiple documents in one call.

    Args:
        documents:  List of (document_name, pages) pairs.
        output_dir: Shared output directory for all files.
        overwrite:  Passed through to write_document().

    Returns:
        List of absolute paths written, in the same order as input.
    """
    paths = []
    for doc_name, pages in documents:
        path = write_document(doc_name, pages, output_dir, overwrite=overwrite)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Markdown block builders
# ---------------------------------------------------------------------------

def _document_header(
    document_name: str,
    pages: List[Tuple[PageOutput, PageMetadata]],
) -> List[str]:
    """Top-of-file header — document identity + pipeline run summary."""
    now     = datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S %Z")
    n_pages = len(pages)
    engines = sorted({e for o, _ in pages for e in o.engines_used})
    avg_conf = (
        sum(o.confidence for o, _ in pages) / n_pages if n_pages else 0.0
    )
    escalated = sum(1 for o, _ in pages if o.escalated)
    low_conf  = sum(1 for o, _ in pages if o.low_confidence)

    return [
        f"# {document_name}",
        "",
        "<!-- IDP Pipeline Output",
        f"     Generated   : {now}",
        f"     Pages       : {n_pages}",
        f"     Engines     : {', '.join(engines)}",
        f"     Avg conf    : {avg_conf:.3f}",
        f"     Escalated   : {escalated}",
        f"     Low-conf    : {low_conf}",
        "-->",
        "",
        "---",
    ]


def _page_block(output: PageOutput, metadata: PageMetadata) -> List[str]:
    """
    One page's extracted content wrapped in a provenance header and footer.

    Format:
        <!-- PAGE n | engine | confidence | capabilities -->
        <extracted markdown>
        <!-- /PAGE n -->
    """
    pg       = output.page_number
    engines  = ", ".join(output.engines_used) if output.engines_used else "skip"
    conf     = f"{output.confidence:.3f}"
    caps     = ", ".join(output.capabilities) if output.capabilities else "—"
    lat_ms   = (
        f"{metadata.total_latency_ms:.0f}ms"
        if metadata.total_latency_ms is not None
        else "—"
    )
    esc_flag = " | escalated" if output.escalated else ""
    lc_flag  = " | low_confidence" if output.low_confidence else ""

    lines = [
        f"<!-- PAGE {pg}"
        f" | engine={engines}"
        f" | conf={conf}"
        f" | caps=[{caps}]"
        f" | latency={lat_ms}"
        f"{esc_flag}{lc_flag}"
        " -->",
    ]

    # Page content — use a horizontal rule as visual separator
    content = output.markdown.strip()
    if content:
        lines.append(content)
    else:
        lines.append("_(no content extracted from this page)_")

    lines.append(f"<!-- /PAGE {pg} -->")
    lines.append("")
    lines.append("---")
    return lines


def _document_footer(pages: List[Tuple[PageOutput, PageMetadata]]) -> List[str]:
    """
    End-of-file metadata block — compact JSON summary of every page's
    processing result. Engineer B's Layer 3 can parse this block to
    understand what was extracted and with what confidence, without having
    to re-run the pipeline.
    """
    summary = []
    for output, metadata in pages:
        summary.append({
            "page":        output.page_number,
            "engines":     output.engines_used,
            "confidence":  round(output.confidence, 4),
            "capabilities": output.capabilities,
            "chars":       len(output.markdown.strip()),
            "escalated":   output.escalated,
            "low_confidence": output.low_confidence,
            "latency_ms":  metadata.total_latency_ms,
        })

    return [
        "",
        "---",
        "",
        "<!-- PIPELINE_SUMMARY",
        json.dumps(summary, indent=2),
        "-->",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem(filename: str) -> str:
    """'MY_resume.pdf' → 'MY_resume'"""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return name
