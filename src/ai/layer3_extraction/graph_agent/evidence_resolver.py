"""
File: evidence_resolver.py
Purpose: Phase 2 of DocumentIR — resolve an LLM-provided evidence text
         snippet (from PageDelta extraction) back to the DocumentIR
         block(s) it came from, so GraphNode/Evidence can carry block_ids
         + bbox instead of just free text.

         Deliberately NOT a single-block match. The extraction prompt
         asks the LLM for "exact verbatim text snippet" at SENTENCE
         granularity, but PaddleOCR blocks are per OCR LINE (see
         paddle_block_rollup.py) and Docling blocks are per paragraph.
         A sentence will often span 2+ line-blocks. This resolves a
         CONTIGUOUS SPAN of blocks, not one block_id.

         Deterministic, no LLM call, no new dependency beyond stdlib
         difflib. Accuracy here is exactly the kind of thing the Phase 3
         eval harness should measure — this is a best-effort heuristic,
         not guaranteed-correct, and degrades gracefully (empty
         block_ids, not an exception) when no good match is found.

Owner: TBD
Deps: stdlib only (difflib, re)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

try:
    from src.ai.schemas.document_ir import Block
except ImportError:  # pragma: no cover - import path varies by deployment context
    Block = None  # type: ignore


# Minimum match ratio (0-1) to accept a resolution at all. Below this,
# we'd rather return no block_ids than a wrong one — a wrong citation is
# worse than an absent one for anything downstream (highlighting, review).
MIN_MATCH_RATIO = 0.55


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def resolve_evidence_to_blocks(
    evidence_text: str,
    page_blocks: List["Block"],
) -> Tuple[List[str], Optional[dict]]:
    """
    Find the contiguous span of page_blocks (in reading order) whose
    concatenated text best matches evidence_text.

    Returns (block_ids, union_bbox_dict). block_ids is [] and
    union_bbox_dict is None when no match clears MIN_MATCH_RATIO —
    callers should treat that as "no resolution", not an error.
    """
    norm_evidence = _normalize(evidence_text)
    if not norm_evidence or not page_blocks:
        return [], None

    ordered = sorted(
        page_blocks,
        key=lambda b: (b.reading_order if b.reading_order is not None else 1_000_000),
    )

    # Build one normalized concatenated string for the page, tracking which
    # block each character range belongs to, so a match in the concatenated
    # string can be mapped back to the specific blocks it spans.
    concatenated = ""
    char_to_block_idx: List[int] = []  # parallel array: char_to_block_idx[i] = index into `ordered`
    for idx, block in enumerate(ordered):
        block_text = _normalize(block.text)
        if not block_text:
            continue
        if concatenated:
            concatenated += " "
            char_to_block_idx.append(-1)  # separator belongs to no block
        start = len(concatenated)
        concatenated += block_text
        char_to_block_idx.extend([idx] * (len(concatenated) - start))

    if not concatenated:
        return [], None

    matcher = SequenceMatcher(None, concatenated, norm_evidence, autojunk=False)
    match = matcher.find_longest_match(0, len(concatenated), 0, len(norm_evidence))

    if match.size == 0:
        return [], None

    match_ratio = match.size / max(len(norm_evidence), 1)
    if match_ratio < MIN_MATCH_RATIO:
        return [], None

    matched_block_indices = sorted({
        char_to_block_idx[i]
        for i in range(match.a, match.a + match.size)
        if 0 <= i < len(char_to_block_idx) and char_to_block_idx[i] != -1
    })
    if not matched_block_indices:
        return [], None

    matched_blocks = [ordered[i] for i in matched_block_indices]
    block_ids = [b.block_id for b in matched_blocks]

    boxed = [b.bbox for b in matched_blocks if b.bbox is not None]
    union_bbox = None
    if boxed:
        union_bbox = {
            "x0": min(b.x0 for b in boxed),
            "y0": min(b.y0 for b in boxed),
            "x1": max(b.x1 for b in boxed),
            "y1": max(b.y1 for b in boxed),
        }

    return block_ids, union_bbox
