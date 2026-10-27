"""
File: vlm_block_rollup.py
Purpose: Build DocumentIR Block objects from a VLM full-page transcription.

         The VLM transcribe path (llm_client.transcribe_handwriting) returns the
         whole page as Markdown text with NO per-region coordinates. Before this,
         the pipeline attached an empty block list for VLM pages, so scanned docs
         that escalated to the VLM produced IR with zero blocks — the chunker,
         evidence resolver and citation chain then had nothing to anchor on.

         This module splits that Markdown into logical segments (headings,
         paragraphs, list items, table rows) and emits one Block per segment, in
         reading order. Blocks carry bbox=None (the VLM gives no coordinates — a
         case the IR store and chunker already handle), so downstream retrieval
         works on scanned docs even without coordinates.

PURE / DETERMINISTIC / OFFLINE: stdlib only (re). No model, no AWS.
Owner: Phase 4 scanned-IR fix.
"""
from __future__ import annotations

import re
from typing import List

from src.ai.schemas.document_ir import Block, BlockType

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")  # |---|---| separator row
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\(?\d+[.)]|[a-zA-Z][.)])\s+\S")


def _block_type(segment: str) -> BlockType:
    first = segment.lstrip().splitlines()[0] if segment.strip() else ""
    if _HEADING.match(first):
        return BlockType.HEADING
    if _TABLE_ROW.match(first):
        return BlockType.TABLE
    if _LIST_ITEM.match(first):
        return BlockType.LINE
    return BlockType.PARAGRAPH


def blocks_from_vlm_markdown(
    markdown: str,
    page_number: int,
    confidence: float = 1.0,
    engine_name: str = "vlm_transcribe",
) -> List[Block]:
    """Split a VLM full-page Markdown transcription into ordered Blocks.

    Segmentation (deterministic, line-based):
      - Markdown headings (#..######) become their own HEADING block.
      - Contiguous table rows (| ... |) are grouped into one TABLE block;
        the |---| separator row is dropped from the grouped text.
      - Everything else is split on blank lines into PARAGRAPH/LINE blocks.
    Empty/whitespace-only segments are skipped. bbox is None (no coordinates).
    """
    if not markdown or not markdown.strip():
        return []

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: List[str] = []
    buf: List[str] = []
    mode = "text"  # "text" | "table"

    def _flush():
        if buf:
            seg = "\n".join(buf).strip()
            if seg:
                segments.append(seg)
        buf.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if _HEADING.match(line):
            _flush(); mode = "text"
            segments.append(stripped)
            continue

        if _TABLE_ROW.match(line):
            if mode != "table":
                _flush(); mode = "table"
            if not _TABLE_SEP.match(line):  # drop |---|---| separators
                buf.append(stripped)
            continue

        if mode == "table":
            _flush(); mode = "text"

        if stripped == "":
            _flush()  # blank line ends a paragraph
            continue
        buf.append(line)

    _flush()

    blocks: List[Block] = []
    order = 0
    conf = max(0.0, min(1.0, float(confidence)))
    for seg in segments:
        blocks.append(Block(
            block_id=f"p{page_number}-{engine_name}-{order}",
            type=_block_type(seg),
            text=seg,
            bbox=None,                 # VLM provides no coordinates
            reading_order=order,
            confidence=conf,
            source_engine=engine_name,
        ))
        order += 1
    return blocks
