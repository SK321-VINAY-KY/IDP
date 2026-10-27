"""
Tests for vlm_block_rollup: turning a VLM full-page Markdown transcription into
ordered DocumentIR Blocks. Synthetic strings only; offline, no model, no AWS.
"""
from src.ai.layer2_conversion.vlm_block_rollup import blocks_from_vlm_markdown
from src.ai.schemas.document_ir import BlockType


def test_empty_markdown_yields_no_blocks():
    assert blocks_from_vlm_markdown("", page_number=1) == []
    assert blocks_from_vlm_markdown("   \n  \n", page_number=1) == []


def test_paragraphs_split_on_blank_lines():
    md = "First paragraph line one.\nstill first.\n\nSecond paragraph."
    blocks = blocks_from_vlm_markdown(md, page_number=3, confidence=0.9)
    assert len(blocks) == 2
    assert blocks[0].type == BlockType.PARAGRAPH
    assert blocks[0].block_id == "p3-vlm_transcribe-0"
    assert blocks[1].block_id == "p3-vlm_transcribe-1"
    # reading order is sequential and bbox is None (no coords from VLM)
    assert [b.reading_order for b in blocks] == [0, 1]
    assert all(b.bbox is None for b in blocks)
    assert all(b.source_engine == "vlm_transcribe" for b in blocks)
    assert all(abs(b.confidence - 0.9) < 1e-6 for b in blocks)


def test_heading_is_its_own_block():
    md = "# Patient Report\n\nSome body text here."
    blocks = blocks_from_vlm_markdown(md, page_number=1)
    assert blocks[0].type == BlockType.HEADING
    assert "Patient Report" in blocks[0].text
    assert blocks[1].type == BlockType.PARAGRAPH


def test_table_rows_grouped_and_separator_dropped():
    md = "| Field | Value |\n|-------|-------|\n| Name | Jane |\n| City | Pune |"
    blocks = blocks_from_vlm_markdown(md, page_number=2)
    assert len(blocks) == 1
    tbl = blocks[0]
    assert tbl.type == BlockType.TABLE
    # the |---|---| separator row must be dropped
    assert "---" not in tbl.text
    assert "Field | Value" in tbl.text
    assert "Jane" in tbl.text and "Pune" in tbl.text


def test_list_items_become_line_blocks():
    md = "- first item\n- second item"
    blocks = blocks_from_vlm_markdown(md, page_number=1)
    # blank-line-free consecutive list lines group into one segment here
    assert blocks
    assert blocks[0].type == BlockType.LINE


def test_confidence_clamped():
    b = blocks_from_vlm_markdown("text", page_number=1, confidence=5.0)[0]
    assert b.confidence == 1.0
    b2 = blocks_from_vlm_markdown("text", page_number=1, confidence=-1.0)[0]
    assert b2.confidence == 0.0


def test_mixed_document_order_preserved():
    md = "# Title\n\nIntro paragraph.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nClosing note."
    blocks = blocks_from_vlm_markdown(md, page_number=4)
    types = [b.type for b in blocks]
    assert types == [BlockType.HEADING, BlockType.PARAGRAPH, BlockType.TABLE, BlockType.PARAGRAPH]
    assert [b.reading_order for b in blocks] == [0, 1, 2, 3]
    assert [b.block_id for b in blocks] == [f"p4-vlm_transcribe-{i}" for i in range(4)]
