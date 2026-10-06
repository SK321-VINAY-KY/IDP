"""
File: document_ir.py
Purpose: DocumentIR — the structured, position-aware intermediate
         representation that sits between Layer 2 (OCR/conversion) and
         everything downstream (Markdown generation, Layer 3 extraction,
         merge logic, Query Bot evidence highlighting).

         Phase 1 of the DocumentIR rollout. Markdown stays the contract
         Layer 3 consumes for now — blocks are additive, not a replacement.
         PageOutput.blocks defaults to an empty list, so this is backward
         compatible with every existing consumer of PageOutput.

Owner: TBD (touches the Engineer A / Engineer B PageOutput contract —
       see src/ai/schemas/page.py header; needs sign-off from both before
       PageOutput itself is edited).
Deps: pydantic
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    SIGNATURE = "signature"
    STAMP = "stamp"
    LINE = "line"          # single OCR line — used by PaddleOCR engines
    CAPTION = "caption"
    OTHER = "other"


class BBox(BaseModel):
    """Pixel-space bounding box, top-left origin. Matches the coordinate
    space of the rendered page image used for OCR (see render DPI in the
    engine that produced this block)."""

    x0: float
    y0: float
    x1: float
    y1: float


class TableCell(BaseModel):
    row: int
    column: int
    text: str
    bbox: Optional[BBox] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    row_span: int = 1
    col_span: int = 1


class Block(BaseModel):
    """One structural unit on a page: a paragraph, heading, table, figure,
    or raw OCR line. This is the thing field-level evidence (Phase 2) will
    point at via block_id, and the thing region-level routing (Phase 5)
    will dispatch per-block instead of per-page."""

    block_id: str                      # unique within the page, e.g. "p4-b17"
    type: BlockType
    text: str = ""
    bbox: Optional[BBox] = None        # None only when the source engine truly has no coordinates
    reading_order: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_engine: str = ""            # "docling" | "paddleocr_printed" | "paddleocr_handwritten" | "vlm_transcribe"
    cells: Optional[List[TableCell]] = None   # populated only when type == TABLE

    def to_markdown_line(self) -> str:
        """Phase 1 fallback renderer — keeps existing Markdown-based
        consumers (Layer 3, merge logic) working unchanged while blocks
        are being adopted. A richer table/heading renderer can replace
        this per block type without touching callers."""
        if self.type == BlockType.TABLE and self.cells:
            return _table_cells_to_markdown(self.cells)
        return self.text


class PageIR(BaseModel):
    """DocumentIR for a single page. One of these is produced per page by
    each Layer 2 engine invocation and attached to PageOutput.blocks."""

    page_number: int
    blocks: List[Block] = Field(default_factory=list)
    page_image_uri: Optional[str] = None   # set once pages are persisted to S3/disk
    source_engine: str = ""
    engine_version: Optional[str] = None


def markdown_from_blocks(blocks: List[Block]) -> str:
    """Derive Markdown from blocks, in reading order where known. This is
    what lets Layer 3 keep consuming PageOutput.markdown unchanged during
    the Phase 1-2 transition — Markdown becomes a *view* of the IR, not
    the source of truth."""
    ordered = sorted(
        blocks,
        key=lambda b: (b.reading_order if b.reading_order is not None else 1_000_000),
    )
    lines = [b.to_markdown_line() for b in ordered if b.to_markdown_line().strip()]
    return "\n".join(lines)


def _table_cells_to_markdown(cells: List[TableCell]) -> str:
    if not cells:
        return ""
    n_rows = max(c.row for c in cells) + 1
    n_cols = max(c.column for c in cells) + 1
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        grid[c.row][c.column] = c.text
    header = "| " + " | ".join(grid[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(n_cols)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in grid[1:])
    return "\n".join([header, sep, body]) if n_rows > 1 else header
