"""
Phase 4, commit 2 — chunker tests. SYNTHETIC PageIR only, fully offline:
no model, no AWS, no disk. Covers coalescing within/not-across pages, long
block handling, label+value gluing, table rows (header prepend / spans / empty
cells / no header / no cells), null bbox, empty-block skipping, reading-order
normalisation, determinism across PYTHONHASHSEED, content-free ids, and the
owner-required signature.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR, TableCell
from src.ai.layer3_extraction.chunker import (
    Chunk,
    ChunkConfig,
    chunk_id_to_blocks,
    chunk_page_irs,
)

OWNER = "tester"
DOC = "synthetic_doc"


def _blk(bid, text, ro, type_=BlockType.PARAGRAPH, bbox=(0, 0, 10, 10), engine="docling", cells=None):
    return Block(
        block_id=bid,
        type=type_,
        text=text,
        reading_order=ro,
        bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]) if bbox else None,
        source_engine=engine,
        cells=cells,
    )


def _page(n, blocks):
    return PageIR(page_number=n, blocks=blocks, source_engine="docling")


# --------------------------------------------------------------------------- #
# Coalescing
# --------------------------------------------------------------------------- #

def test_coalesce_within_page_under_cap():
    page = _page(1, [_blk("b1", "Alpha one.", 0), _blk("b2", "Beta two.", 1), _blk("b3", "Gamma.", 2)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b1", "b2", "b3"]
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == "c1-0"


def test_never_coalesce_across_pages():
    p1 = _page(1, [_blk("a", "Page one text.", 0)])
    p2 = _page(2, [_blk("b", "Page two text.", 0)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[p1, p2])
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert all(bid in {"a"} for bid in c.block_ids) if c.page == 1 else True
    # each page's chunk ids restart at seq 0
    assert "c1-0" in {c.chunk_id for c in chunks}
    assert "c2-0" in {c.chunk_id for c in chunks}


def test_cap_starts_new_chunk():
    # Use clearly sentence-like text (ends with '.') so these are NOT treated as
    # label blocks; this exercises the coalescing cap, not the label path.
    cfg = ChunkConfig(target_chars=20)
    page = _page(1, [_blk("b1", "First sentence here.", 0), _blk("b2", "Second sentence now.", 1)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page], config=cfg)
    assert len(chunks) == 2
    assert chunks[0].block_ids == ["b1"]
    assert chunks[1].block_ids == ["b2"]


# --------------------------------------------------------------------------- #
# Long block
# --------------------------------------------------------------------------- #

def test_long_block_split_never_mid_word():
    cfg = ChunkConfig(target_chars=30)
    long = "The quick brown fox jumps over the lazy dog near the river bank today."
    page = _page(1, [_blk("b1", long, 0)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page], config=cfg)
    assert len(chunks) >= 2
    # same block_ids on every piece
    assert all(c.block_ids == ["b1"] for c in chunks)
    # no piece splits a word: rejoining preserves whole words
    words_in = long.split()
    words_out = " ".join(c.text for c in chunks).split()
    assert words_out == words_in


def test_single_word_longer_than_cap_kept_whole():
    cfg = ChunkConfig(target_chars=5)
    page = _page(1, [_blk("b1", "supercalifragilistic", 0)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page], config=cfg)
    assert len(chunks) == 1
    assert chunks[0].text == "supercalifragilistic"


# --------------------------------------------------------------------------- #
# Label + value
# --------------------------------------------------------------------------- #

def test_label_colon_glued_to_value():
    page = _page(1, [_blk("b1", "Sanctioned Amount:", 0), _blk("b2", "500000", 1)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b1", "b2"]
    assert "Sanctioned Amount" in chunks[0].text and "500000" in chunks[0].text


def test_label_no_colon_short_phrase_glued():
    page = _page(1, [_blk("b1", "Account Holder Name", 0), _blk("b2", "Jane Doe", 1)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert chunks[0].block_ids == ["b1", "b2"]


def test_sentence_not_treated_as_label():
    page = _page(1, [_blk("b1", "This is a full sentence.", 0), _blk("b2", "Another one.", 1)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    # coalesced normally (not a label pair), both in one chunk under default cap
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b1", "b2"]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def _cell(r, c, text, bbox=(0, 0, 5, 5), row_span=1, col_span=1):
    return TableCell(
        row=r, column=c, text=text,
        bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]) if bbox else None,
        row_span=row_span, col_span=col_span,
    )


def test_table_rows_header_prepended():
    cells = [
        _cell(0, 0, "Field"), _cell(0, 1, "Value"),
        _cell(1, 0, "Name"), _cell(1, 1, "Jane"),
        _cell(2, 0, "City"), _cell(2, 1, "Pune"),
    ]
    page = _page(1, [_blk("t1", "", 0, type_=BlockType.TABLE, cells=cells)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 2  # two data rows
    assert all(c.kind == "table_row" for c in chunks)
    assert [c.row_index for c in chunks] == [1, 2]
    for c in chunks:
        assert "Field | Value" in c.text  # header prepended
        assert c.block_ids == ["t1"]


def test_table_spans_and_empty_cells():
    cells = [
        _cell(0, 0, "H1"), _cell(0, 1, "H2"), _cell(0, 2, "H3"),
        _cell(1, 0, "merged", col_span=2), _cell(1, 2, ""),  # empty cell + span
    ]
    page = _page(1, [_blk("t1", "", 0, type_=BlockType.TABLE, cells=cells)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    assert chunks[0].row_index == 1
    # empty cell renders as empty between separators; row still produced
    assert "merged" in chunks[0].text


def test_table_without_header_no_prepend():
    # single row only -> _has_header is False (no data row below), no prepend
    cells = [_cell(0, 0, "lonely"), _cell(0, 1, "row")]
    page = _page(1, [_blk("t1", "", 0, type_=BlockType.TABLE, cells=cells)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    # row 0 emitted as data (no header prepend), text has no duplicated header
    assert chunks[0].text.count("lonely") == 1


def test_table_block_without_cells_falls_back_to_text():
    page = _page(1, [_blk("t1", "raw table text", 0, type_=BlockType.TABLE, cells=None)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    assert chunks[0].kind == "table_row"
    assert chunks[0].row_index is None
    assert chunks[0].block_ids == ["t1"]


# --------------------------------------------------------------------------- #
# bbox / empty / ordering
# --------------------------------------------------------------------------- #

def test_block_without_bbox_still_chunks_null_bbox():
    page = _page(1, [_blk("b1", "no box here", 0, bbox=None)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert len(chunks) == 1
    assert chunks[0].bbox is None
    assert chunks[0].block_ids == ["b1"]


def test_union_bbox_covers_all_source_blocks():
    page = _page(1, [
        _blk("b1", "left", 0, bbox=(0, 0, 10, 10)),
        _blk("b2", "right", 1, bbox=(20, 5, 30, 25)),
    ])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    bb = chunks[0].bbox
    assert (bb.x0, bb.y0, bb.x1, bb.y1) == (0, 0, 30, 25)


def test_empty_text_blocks_skipped():
    page = _page(1, [_blk("b1", "   ", 0), _blk("b2", "real", 1), _blk("b3", "", 2)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    ids = [bid for c in chunks for bid in c.block_ids]
    assert "b1" not in ids and "b3" not in ids
    assert "b2" in ids


def test_reading_order_respected_when_out_of_order():
    # blocks arrive shuffled; chunker must order by reading_order
    page = _page(1, [
        _blk("b3", "third.", 2),
        _blk("b1", "first.", 0),
        _blk("b2", "second.", 1),
    ])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    assert chunks[0].block_ids == ["b1", "b2", "b3"]


# --------------------------------------------------------------------------- #
# ids & determinism
# --------------------------------------------------------------------------- #

def test_chunk_ids_contain_no_document_text():
    secret = "CONFIDENTIAL_PATIENT_NAME"
    page = _page(1, [_blk("b1", secret, 0)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    for c in chunks:
        assert secret not in c.chunk_id
        assert secret.lower() not in c.chunk_id.lower()
    assert chunks[0].chunk_id == "c1-0"


def test_chunk_id_to_blocks_roundtrip():
    page = _page(1, [_blk("b1", "Label:", 0), _blk("b2", "value", 1)])
    chunks = chunk_page_irs(owner=OWNER, doc_id=DOC, pages=[page])
    cid = chunks[0].chunk_id
    block_ids, bbox = chunk_id_to_blocks(cid, chunks)
    assert block_ids == ["b1", "b2"]
    assert bbox is not None
    # missing id -> ([], None), never raises
    assert chunk_id_to_blocks("c9-9", chunks) == ([], None)


_DETERMINISM_DRIVER = textwrap.dedent(
    """
    import sys, json
    sys.path.insert(0, ".")
    from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR, TableCell
    from src.ai.layer3_extraction.chunker import chunk_page_irs

    def blk(bid, text, ro, type_=BlockType.PARAGRAPH):
        return Block(block_id=bid, type=type_, text=text, reading_order=ro,
                     bbox=BBox(x0=0,y0=0,x1=5,y1=5), source_engine="docling")

    cells = [TableCell(row=0,column=0,text="H1"), TableCell(row=0,column=1,text="H2"),
             TableCell(row=1,column=0,text="a"), TableCell(row=1,column=1,text="b"),
             TableCell(row=2,column=0,text="c"), TableCell(row=2,column=1,text="d")]
    pages = [
        PageIR(page_number=1, blocks=[
            blk("b3","third sentence here.",2), blk("b1","Amount:",0), blk("b2","42",1),
            Block(block_id="t1", type=BlockType.TABLE, text="", reading_order=3, cells=cells,
                  bbox=BBox(x0=0,y0=0,x1=9,y1=9), source_engine="docling"),
        ]),
        PageIR(page_number=2, blocks=[blk("c1","page two.",0)]),
    ]
    chunks = chunk_page_irs(owner="tester", doc_id="d", pages=pages)
    print(json.dumps([c.model_dump() for c in chunks], sort_keys=True, ensure_ascii=False))
    """
)


def test_determinism_across_pythonhashseed():
    outputs = []
    for seed in ("0", "1", "42", "123", "99991"):
        env = {"PYTHONHASHSEED": seed, "PATH": _env_path()}
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_DRIVER],
            capture_output=True, text=True, env=env, cwd=".",
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())
    assert len(set(outputs)) == 1, "chunk output not byte-identical across PYTHONHASHSEED"


def _env_path():
    import os
    return os.environ.get("PATH", "")


# --------------------------------------------------------------------------- #
# Signature guard (owner required, no default)
# --------------------------------------------------------------------------- #

def test_owner_required_no_default():
    import inspect
    from src.ai.layer3_extraction.chunker import chunk_page_irs as fn
    sig = inspect.signature(fn)
    assert "owner" in sig.parameters
    assert sig.parameters["owner"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        fn(doc_id="d", pages=[])  # type: ignore
    with pytest.raises(ValueError):
        fn(owner="", doc_id="d", pages=[])
