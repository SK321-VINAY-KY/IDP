"""
File: chunker.py
Purpose: Phase 4, commit 2 — turn persisted DocumentIR (PageIR + Block) into
         retrieval Chunks. A chunk is the unit the embedder/lexical index
         (later commits) will carry, and the unit a citation resolves to.

PURE / DETERMINISTIC / OFFLINE:
  No model, no AWS, no network, stdlib + pydantic only. The same input
  produces byte-identical output across runs and across PYTHONHASHSEED
  values: blocks are ordered by a stable (reading_order, block_id) key,
  there is no iteration over sets of strings, and ids are positional.

TENANT ISOLATION:
  Every public function that takes `owner` requires it with NO default, and
  the owner is carried on every Chunk. Chunk ids are positional and content
  free (c{page}-{seq}), so an id never leaks document text.

CONTENT SAFETY:
  This module copies block text into Chunk.text (that is its job), but it
  never logs or prints that text. Callers that build reports must emit counts
  and ids only.

Owner: Phase 4 retrieval. Deps: pydantic, stdlib (re).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR, TableCell

# ---------------------------------------------------------------------------
# Tunables (documented defaults; all overridable via chunk_page_irs(..., config=))
# ---------------------------------------------------------------------------

# Target max characters for a coalesced text chunk. Paragraph/heading/line
# blocks are packed in reading order until adding the next block would exceed
# this, then a new chunk starts. ~400 chars ≈ a few sentences — small enough
# to localise a citation, large enough to keep a label with its value.
DEFAULT_TARGET_CHARS = 400

# A block shorter than this (after strip) is a candidate "label" that should be
# glued to the following value block so e.g. "Sanctioned Amount" and its number
# land in one chunk. See _is_label_like for the full rule.
LABEL_MAX_CHARS = 40


class ChunkConfig(BaseModel):
    """Chunker tunables. Kept explicit (not kwargs) so a config is reproducible
    and can be logged by id/values without any document content."""

    target_chars: int = DEFAULT_TARGET_CHARS
    label_max_chars: int = LABEL_MAX_CHARS
    prepend_table_header: bool = True


class Chunk(BaseModel):
    """One retrieval unit. Never spans a page boundary. block_ids is the ordered
    list of source Block.block_id values this chunk was built from, so a citation
    can map back to exact blocks + bbox."""

    chunk_id: str                       # positional, content-free: "c{page}-{seq}"
    owner: str
    doc_id: str
    page: int
    block_ids: List[str]
    bbox: Optional[BBox] = None         # union of source block bboxes; None if none had one
    text: str = ""
    source_engine: str = ""
    kind: str = "other"                 # paragraph | heading | table_row | list | other
    row_index: Optional[int] = None     # set only when kind == "table_row"


# ---------------------------------------------------------------------------
# Ordering & small helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•·]|\(?\d+[.)]|[a-z]\))\s+", re.IGNORECASE)


def _order_key(block: Block) -> Tuple[int, str]:
    """Stable ordering: reading_order first (blocks with none sort last), then
    block_id as a deterministic tie-break. No set iteration, no hash reliance."""
    ro = block.reading_order if block.reading_order is not None else 1_000_000
    return (ro, block.block_id or "")


def _union_bbox(boxes: List[Optional[BBox]]) -> Optional[BBox]:
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    return BBox(
        x0=min(b.x0 for b in present),
        y0=min(b.y0 for b in present),
        x1=max(b.x1 for b in present),
        y1=max(b.y1 for b in present),
    )


def _is_label_like(block: Block) -> bool:
    """Heuristic (no model) for a block that is a field label whose value is the
    next block in reading order. The LENGTH gate is applied by the caller (it
    owns the configurable cap); this tests only the SHAPE:
      - it does NOT end in sentence punctuation (. ! ?), and
      - EITHER it ends with ':' (classic "Label:")
        OR it has no internal '.' and <= 6 words (e.g. "Sanctioned Amount").
    Headings are never labels. Deterministic string tests only."""
    t = (block.text or "").strip()
    if not t:
        return False
    if block.type == BlockType.HEADING:
        return False  # headings get their own chunk, not glued to a value
    if t[-1] in ".!?":
        return False
    if t.endswith(":"):
        return True
    return ("." not in t) and (len(t.split()) <= 6)


def _kind_for(block: Block) -> str:
    if block.type == BlockType.HEADING:
        return "heading"
    if _LIST_PREFIX.match(block.text or ""):
        return "list"
    if block.type in (BlockType.PARAGRAPH, BlockType.LINE, BlockType.CAPTION):
        return "paragraph"
    return "other"


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------

def _table_grid(cells: List[TableCell]) -> Dict[int, Dict[int, TableCell]]:
    """Map (row -> col -> cell). Later cells at the same (row,col) win, which is
    deterministic given the input order; row/col spans are preserved on the cell."""
    grid: Dict[int, Dict[int, TableCell]] = {}
    for c in cells:
        grid.setdefault(c.row, {})[c.column] = c
    return grid


def _row_text(grid_row: Dict[int, TableCell]) -> str:
    """Join a row's cells left-to-right by column index. Empty cells become
    empty strings so column alignment is preserved; separator is ' | '."""
    if not grid_row:
        return ""
    max_col = max(grid_row.keys())
    parts = [(grid_row.get(c).text.strip() if grid_row.get(c) else "") for c in range(max_col + 1)]
    return " | ".join(parts)


def _has_header(grid: Dict[int, Dict[int, TableCell]]) -> bool:
    """A table has a usable header when row 0 exists and has >=2 non-empty cells
    and there is at least one data row below it. Conservative on purpose: a bad
    prepend corrupts every row chunk."""
    if 0 not in grid:
        return False
    nonempty = sum(1 for cell in grid[0].values() if (cell.text or "").strip())
    data_rows = [r for r in grid.keys() if r > 0]
    return nonempty >= 2 and len(data_rows) >= 1


def _table_row_chunks(
    block: Block,
    page: int,
    owner: str,
    doc_id: str,
    seq_start: int,
    config: ChunkConfig,
) -> Tuple[List[Chunk], int]:
    """One chunk per data row; header row text prepended for context when a
    header is detected. When the TABLE block has no `cells`, fall back to a
    single chunk from block.text (so a table still produces something)."""
    seq = seq_start
    if not block.cells:
        # No structured cells: treat the whole table as one text chunk.
        text = (block.text or "").strip()
        if not text:
            return [], seq
        chunk = Chunk(
            chunk_id=f"c{page}-{seq}",
            owner=owner, doc_id=doc_id, page=page,
            block_ids=[block.block_id],
            bbox=block.bbox,
            text=text,
            source_engine=block.source_engine,
            kind="table_row",
            row_index=None,
        )
        return [chunk], seq + 1

    grid = _table_grid(block.cells)
    header_text = ""
    if config.prepend_table_header and _has_header(grid):
        header_text = _row_text(grid[0])

    chunks: List[Chunk] = []
    data_rows = sorted(r for r in grid.keys() if (not header_text or r > 0))
    for r in data_rows:
        body = _row_text(grid[r])
        if not body.strip():
            continue  # skip fully empty rows
        text = f"{header_text}\n{body}" if header_text else body
        cell_boxes = [c.bbox for c in grid[r].values()]
        # include header cell boxes in the union so the citation covers context too
        if header_text:
            cell_boxes += [c.bbox for c in grid[0].values()]
        chunks.append(Chunk(
            chunk_id=f"c{page}-{seq}",
            owner=owner, doc_id=doc_id, page=page,
            block_ids=[block.block_id],
            bbox=_union_bbox(cell_boxes),
            text=text,
            source_engine=block.source_engine,
            kind="table_row",
            row_index=r,
        ))
        seq += 1
    return chunks, seq


# ---------------------------------------------------------------------------
# Long-block splitting (never mid-word)
# ---------------------------------------------------------------------------

def _split_long_text(text: str, cap: int) -> List[str]:
    """Split an over-cap single block at sentence boundaries, falling back to
    whitespace. Never splits inside a word. Returns >=1 pieces, each <= cap
    where possible (a single word longer than cap is emitted whole)."""
    text = text.strip()
    if len(text) <= cap:
        return [text]
    pieces: List[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) <= cap:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(sentence) <= cap:
            buf = sentence
        else:
            # sentence itself too long: split on whitespace, never mid-word
            word_buf = ""
            for word in sentence.split():
                cand = f"{word_buf} {word}".strip() if word_buf else word
                if len(cand) <= cap:
                    word_buf = cand
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    word_buf = word  # a lone over-cap word is kept whole
            if word_buf:
                buf = word_buf
    if buf:
        pieces.append(buf)
    return pieces or [text]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_page_irs(
    owner: str,
    doc_id: str,
    pages: List[PageIR],
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """Turn persisted PageIR objects into deterministic Chunks.

    owner is REQUIRED (no default). A chunk never crosses a page boundary.
    Text blocks are coalesced in reading order up to config.target_chars;
    label-like blocks are kept with the following value block; tables produce
    one chunk per data row with the header prepended. Empty-text blocks are
    skipped. Blocks with no bbox still chunk (null bbox)."""
    if not owner or not str(owner).strip():
        raise ValueError("owner is required and cannot be empty")
    cfg = config or ChunkConfig()
    all_chunks: List[Chunk] = []

    for page_ir in sorted(pages, key=lambda p: p.page_number):
        page = page_ir.page_number
        ordered = sorted(page_ir.blocks, key=_order_key)
        seq = 0
        i = 0
        # pending coalescing buffer for text/heading/list blocks
        buf_ids: List[str] = []
        buf_boxes: List[Optional[BBox]] = []
        buf_text_parts: List[str] = []
        buf_engine = ""
        buf_kind = "paragraph"

        def _flush() -> None:
            nonlocal seq, buf_ids, buf_boxes, buf_text_parts, buf_engine, buf_kind
            if not buf_ids:
                return
            text = "\n".join(tp for tp in buf_text_parts if tp)
            all_chunks.append(Chunk(
                chunk_id=f"c{page}-{seq}",
                owner=owner, doc_id=doc_id, page=page,
                block_ids=list(buf_ids),
                bbox=_union_bbox(buf_boxes),
                text=text,
                source_engine=buf_engine,
                kind=buf_kind,
            ))
            seq += 1
            buf_ids = []
            buf_boxes = []
            buf_text_parts = []
            buf_engine = ""
            buf_kind = "paragraph"

        while i < len(ordered):
            block = ordered[i]
            btext = (block.text or "").strip()

            # Tables: flush any pending text, then emit per-row chunks.
            if block.type == BlockType.TABLE:
                _flush()
                row_chunks, seq = _table_row_chunks(block, page, owner, doc_id, seq, cfg)
                all_chunks.extend(row_chunks)
                i += 1
                continue

            if not btext:  # skip empty-text blocks
                i += 1
                continue

            # Headings get their own chunk (flush pending, emit, continue).
            if block.type == BlockType.HEADING:
                _flush()
                all_chunks.append(Chunk(
                    chunk_id=f"c{page}-{seq}",
                    owner=owner, doc_id=doc_id, page=page,
                    block_ids=[block.block_id],
                    bbox=block.bbox,
                    text=btext,
                    source_engine=block.source_engine,
                    kind="heading",
                ))
                seq += 1
                i += 1
                continue

            # Label+value gluing: a short label-like block binds to the NEXT
            # block in reading order (whatever its length) in the same chunk.
            is_label = len(btext) <= cfg.label_max_chars and _is_label_like(block)
            if is_label and (i + 1) < len(ordered):
                nxt = ordered[i + 1]
                nxt_text = (nxt.text or "").strip()
                if nxt_text and nxt.type not in (BlockType.TABLE, BlockType.HEADING):
                    _flush()  # don't bleed previous paragraph into the label pair
                    all_chunks.append(Chunk(
                        chunk_id=f"c{page}-{seq}",
                        owner=owner, doc_id=doc_id, page=page,
                        block_ids=[block.block_id, nxt.block_id],
                        bbox=_union_bbox([block.bbox, nxt.bbox]),
                        text=f"{btext} {nxt_text}",
                        source_engine=block.source_engine or nxt.source_engine,
                        kind=_kind_for(nxt),
                    ))
                    seq += 1
                    i += 2
                    continue

            # Over-cap single block: emit as its own chunk(s), same block_ids.
            if len(btext) > cfg.target_chars:
                _flush()
                for piece in _split_long_text(btext, cfg.target_chars):
                    all_chunks.append(Chunk(
                        chunk_id=f"c{page}-{seq}",
                        owner=owner, doc_id=doc_id, page=page,
                        block_ids=[block.block_id],
                        bbox=block.bbox,
                        text=piece,
                        source_engine=block.source_engine,
                        kind=_kind_for(block),
                    ))
                    seq += 1
                i += 1
                continue

            # Normal coalescing: would adding this block exceed the cap?
            current_len = sum(len(tp) for tp in buf_text_parts) + len(buf_text_parts)
            if buf_ids and (current_len + len(btext) > cfg.target_chars):
                _flush()
            if not buf_ids:
                buf_engine = block.source_engine
                buf_kind = _kind_for(block)
            buf_ids.append(block.block_id)
            buf_boxes.append(block.bbox)
            buf_text_parts.append(btext)
            i += 1

        _flush()

    return all_chunks


def chunk_id_to_blocks(
    chunk_id: str,
    chunks: List[Chunk],
) -> Tuple[List[str], Optional[BBox]]:
    """Map a chunk_id back to its (block_ids, union bbox) for citations.
    Returns ([], None) when the id is not present (never raises)."""
    for c in chunks:
        if c.chunk_id == chunk_id:
            return list(c.block_ids), c.bbox
    return [], None
