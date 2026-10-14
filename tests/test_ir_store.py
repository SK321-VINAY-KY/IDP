"""
Tests for per-owner DocumentIR persistence (Phase 4, commit 1).

Synthetic PageIR only — NO document content. Offline (disk + json only), no model
or AWS calls. Uses a tmp output_root so nothing touches the real dataset_output.
"""
import inspect
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR, TableCell
from src.ai.layer3_extraction import ir_store
from src.ai.layer3_extraction.ir_store import (
    IR_SCHEMA_VERSION,
    load_page_irs,
    save_page_irs,
    ir_path,
)


# --------------------------------------------------------------------------- #
# synthetic fixtures (no document content)
# --------------------------------------------------------------------------- #

def _page(page_number: int) -> PageIR:
    return PageIR(
        page_number=page_number,
        source_engine="paddleocr_printed",
        blocks=[
            Block(
                block_id=f"p{page_number}-b0",
                type=BlockType.PARAGRAPH,
                text="synthetic-text-A",
                bbox=BBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0),
                reading_order=0,
                confidence=0.9,
                source_engine="paddleocr_printed",
            ),
            Block(
                block_id=f"p{page_number}-b1",
                type=BlockType.TABLE,
                text="",
                reading_order=1,
                cells=[
                    TableCell(row=0, column=0, text="h0"),
                    TableCell(row=0, column=1, text="h1"),
                    TableCell(row=1, column=0, text="v0", bbox=BBox(x0=5, y0=6, x1=7, y1=8)),
                ],
                source_engine="paddleocr_printed",
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #

def test_round_trip_equal(tmp_path):
    pages = [_page(2), _page(1)]  # unsorted on input
    root = tmp_path / "ir"
    save_page_irs(owner="alice", doc_id="doc1.pdf", pages=pages, output_root=root)
    loaded = load_page_irs(owner="alice", doc_id="doc1.pdf", output_root=root)
    assert loaded is not None
    # saved sorted by page_number
    assert [p.page_number for p in loaded] == [1, 2]
    # blocks, bbox, table cells, reading order preserved
    original_sorted = sorted(pages, key=lambda p: p.page_number)
    assert loaded == original_sorted
    p1 = loaded[0]
    assert p1.blocks[0].bbox == BBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    assert p1.blocks[1].type == BlockType.TABLE
    assert p1.blocks[1].cells[2].bbox == BBox(x0=5, y0=6, x1=7, y1=8)
    assert p1.blocks[0].reading_order == 0


def test_file_has_schema_version(tmp_path):
    root = tmp_path / "ir"
    path = save_page_irs(owner="alice", doc_id="d", pages=[_page(1)], output_root=root)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == IR_SCHEMA_VERSION
    assert payload["doc_id"] == "d"


def test_deterministic_serialization(tmp_path):
    root = tmp_path / "ir"
    p1 = save_page_irs(owner="a", doc_id="d", pages=[_page(1), _page(2)], output_root=root)
    bytes1 = Path(p1).read_bytes()
    p2 = save_page_irs(owner="a", doc_id="d", pages=[_page(2), _page(1)], output_root=root)
    bytes2 = Path(p2).read_bytes()
    assert bytes1 == bytes2, "serialization must be deterministic regardless of input order"


# --------------------------------------------------------------------------- #
# owner isolation
# --------------------------------------------------------------------------- #

def test_two_owners_same_docname_separate_files():
    # default (real convention) paths, but under distinct owners -> distinct files
    pa = ir_path(owner="owner_a", doc_id="shared.pdf")
    pb = ir_path(owner="owner_b", doc_id="shared.pdf")
    assert pa != pb
    assert "owner_a" in str(pa).replace("\\", "/")
    assert "owner_b" in str(pb).replace("\\", "/")


def test_wrong_owner_returns_none(tmp_path):
    # Each owner gets its own output_root (mirrors per-owner dir separation).
    root_a = tmp_path / "_users" / "alice" / "ir"
    root_b = tmp_path / "_users" / "bob" / "ir"
    save_page_irs(owner="alice", doc_id="x.pdf", pages=[_page(1)], output_root=root_a)
    # bob has no such file in his own partition
    assert load_page_irs(owner="bob", doc_id="x.pdf", output_root=root_b) is None
    # alice still reads her own
    assert load_page_irs(owner="alice", doc_id="x.pdf", output_root=root_a) is not None


def test_default_path_is_owner_partitioned():
    # The default (no output_root) path must sit under _users/<owner>/ir/
    p = ir_path(owner="Alice", doc_id="d.pdf")
    norm = str(p).replace("\\", "/")
    assert "/_users/alice/ir/" in norm  # owner lowercased + partitioned


# --------------------------------------------------------------------------- #
# path traversal / sanitization
# --------------------------------------------------------------------------- #

def test_doc_id_traversal_sanitized(tmp_path):
    root = tmp_path / "ir"
    path = ir_path(owner="alice", doc_id="../../evil", output_root=root)
    # the doc_id must collapse to a flat filename inside root — no escape.
    # Path separators are mapped to '_', so the name cannot traverse even though
    # the literal characters '.' may survive (a flat name like '.._.._evil' is safe).
    assert Path(path).parent == Path(root)
    assert "/" not in Path(path).name and "\\" not in Path(path).name
    # resolving the full path stays directly inside root (no directory escape)
    assert Path(path).resolve().parent == Path(root).resolve()
    # and it round-trips as a harmless flat name
    save_page_irs(owner="alice", doc_id="../../evil", pages=[_page(1)], output_root=root)
    assert load_page_irs(owner="alice", doc_id="../../evil", output_root=root) is not None


def test_empty_owner_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_page_irs(owner="", doc_id="d", pages=[_page(1)], output_root=tmp_path)
    with pytest.raises(ValueError):
        load_page_irs(owner="   ", doc_id="d", output_root=tmp_path)


def test_empty_doc_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_page_irs(owner="a", doc_id="", pages=[_page(1)], output_root=tmp_path)


# --------------------------------------------------------------------------- #
# missing / corrupted / wrong-version => None (never raises)
# --------------------------------------------------------------------------- #

def test_missing_file_returns_none(tmp_path):
    assert load_page_irs(owner="a", doc_id="nope", output_root=tmp_path / "ir") is None


def test_corrupted_file_returns_none(tmp_path):
    root = tmp_path / "ir"
    root.mkdir(parents=True)
    bad = ir_path(owner="a", doc_id="d", output_root=root)
    bad.write_text("{ this is not json ::::", encoding="utf-8")
    assert load_page_irs(owner="a", doc_id="d", output_root=root) is None


def test_wrong_version_returns_none(tmp_path):
    root = tmp_path / "ir"
    root.mkdir(parents=True)
    p = ir_path(owner="a", doc_id="d", output_root=root)
    p.write_text(json.dumps({"schema_version": 999, "doc_id": "d", "pages": []}), encoding="utf-8")
    assert load_page_irs(owner="a", doc_id="d", output_root=root) is None


def test_invalid_shape_returns_none(tmp_path):
    root = tmp_path / "ir"
    root.mkdir(parents=True)
    p = ir_path(owner="a", doc_id="d", output_root=root)
    p.write_text(json.dumps({"schema_version": IR_SCHEMA_VERSION, "pages": "notalist"}), encoding="utf-8")
    assert load_page_irs(owner="a", doc_id="d", output_root=root) is None


# --------------------------------------------------------------------------- #
# no document content in warning logs
# --------------------------------------------------------------------------- #

SECRET = "CONFIDENTIAL_IR_" + "BODY_TEXT_42"

def test_no_content_in_corruption_warning(tmp_path, caplog):
    root = tmp_path / "ir"
    root.mkdir(parents=True)
    p = ir_path(owner="a", doc_id="d", output_root=root)
    # corrupted file whose bytes contain a secret marker
    p.write_text('{"schema_version":1,"pages":[ ' + SECRET + ' broken', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert load_page_irs(owner="a", doc_id="d", output_root=root) is None
    blob = "\n".join(r.getMessage() for r in caplog.records) + str([r.__dict__ for r in caplog.records])
    assert SECRET not in blob, "file content leaked into a warning log"


# --------------------------------------------------------------------------- #
# signature guard: owner required with no default on the new functions
# --------------------------------------------------------------------------- #

def test_ir_store_functions_require_owner_without_default():
    for fn in (save_page_irs, load_page_irs, ir_path):
        sig = inspect.signature(fn)
        assert "owner" in sig.parameters, f"{fn.__name__} missing 'owner'"
        assert sig.parameters["owner"].default is inspect.Parameter.empty, (
            f"{fn.__name__} 'owner' must not have a default"
        )
    # calling without owner is a TypeError
    with pytest.raises(TypeError):
        save_page_irs(doc_id="d", pages=[])  # type: ignore
    with pytest.raises(TypeError):
        load_page_irs(doc_id="d")  # type: ignore
