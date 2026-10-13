"""
File: test_evidence_resolver.py
Purpose: Unit and integration tests for DocumentIR Phase 2 evidence_resolver.py.
         Tests deterministic resolution of LLM evidence snippets to DocumentIR Block spans
         and union bounding boxes across synthetic and real extraction output.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.ai.schemas.document_ir import Block, BlockType, BBox
from src.ai.layer3_extraction.graph_agent.evidence_resolver import (
    resolve_evidence_to_blocks,
    _normalize,
    MIN_MATCH_RATIO,
)
from src.ai.layer3_extraction.graph_agent.models import Evidence, GraphNode, GraphEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(block_id: str, text: str, x0: float, y0: float, x1: float, y1: float, reading_order: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        reading_order=reading_order,
        confidence=0.98,
        source_engine="docling",
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestEvidenceResolverBasics:
    """Core algorithmic and boundary behavior of resolve_evidence_to_blocks."""

    def test_empty_evidence_or_empty_blocks_returns_empty(self):
        blocks = [_make_block("p1-b1", "Sample text", 10, 10, 100, 30, 0)]
        assert resolve_evidence_to_blocks("", blocks) == ([], None)
        assert resolve_evidence_to_blocks("Sample", []) == ([], None)
        assert resolve_evidence_to_blocks("   ", blocks) == ([], None)

    def test_exact_single_block_match(self):
        blocks = [
            _make_block("p1-b0", "HEADER TITLE", 10, 10, 100, 25, 0),
            _make_block("p1-b1", "Invoice Total: $450.00 USD", 10, 30, 200, 50, 1),
            _make_block("p1-b2", "Thank you for your business.", 10, 60, 250, 80, 2),
        ]
        b_ids, bbox = resolve_evidence_to_blocks("Invoice Total: $450.00 USD", blocks)
        assert b_ids == ["p1-b1"]
        assert bbox == {"x0": 10.0, "y0": 30.0, "x1": 200.0, "y1": 50.0}

    def test_multi_block_contiguous_span_match(self):
        """Sentence that spans two line-blocks (common in OCR)."""
        blocks = [
            _make_block("line-1", "This agreement is entered into by and between", 10, 10, 300, 25, 0),
            _make_block("line-2", "Acme Corporation and Beta LLC on March 15.", 10, 30, 280, 45, 1),
            _make_block("line-3", "1. Definitions and Scope.", 10, 60, 180, 75, 2),
        ]
        evidence = "This agreement is entered into by and between Acme Corporation and Beta LLC on March 15."
        b_ids, bbox = resolve_evidence_to_blocks(evidence, blocks)
        assert b_ids == ["line-1", "line-2"]
        assert bbox == {"x0": 10.0, "y0": 10.0, "x1": 300.0, "y1": 45.0}

    def test_partial_match_clearing_threshold(self):
        """LLM slightly rephrased or omitted leading punctuation."""
        blocks = [
            _make_block("p1-b1", "Machine Learning (LLM/SLM) Intern - Bangalore, India", 45, 200, 280, 215, 0),
        ]
        evidence = "Machine Learning Intern - Bangalore, India"
        b_ids, bbox = resolve_evidence_to_blocks(evidence, blocks)
        assert b_ids == ["p1-b1"]
        assert bbox is not None

    def test_low_match_ratio_rejected(self):
        """Unrelated evidence snippet must return empty block list."""
        blocks = [
            _make_block("p1-b1", "Rashi Peripherals Limited Bangalore", 45, 180, 170, 200, 0),
        ]
        evidence = "Total Claim Amount under Section 80D is Rupees Fifty Thousand Only"
        b_ids, bbox = resolve_evidence_to_blocks(evidence, blocks)
        assert b_ids == []
        assert bbox is None

    def test_block_without_bbox_handled_gracefully(self):
        """When matching blocks have no bbox (e.g. bbox=None), union_bbox is None."""
        block_no_bbox = Block(
            block_id="b-nobbox",
            type=BlockType.PARAGRAPH,
            text="Text without coordinates",
            bbox=None,
            reading_order=0,
        )
        b_ids, bbox = resolve_evidence_to_blocks("Text without coordinates", [block_no_bbox])
        assert b_ids == ["b-nobbox"]
        assert bbox is None


# ---------------------------------------------------------------------------
# Real Data Integration Tests
# ---------------------------------------------------------------------------

class TestEvidenceResolverRealData:
    """Tests resolver against actual document extraction output and IR blocks."""

    def test_real_evidence_resolutions_from_fixture(self):
        """Extract blocks from Vinay_Resume.pdf and match against real graph nodes."""
        pdf_path = Path("tests/fixtures/Vinay_Resume.pdf")
        graph_path = Path("dataset_output/Vinay_Resume.graph.json")
        if not pdf_path.exists() or not graph_path.exists():
            pytest.skip("Real data fixture files not available")

        from src.ai.layer2_conversion.digital import convert_digital_page

        _, _, page1_blocks = convert_digital_page(str(pdf_path), 1)
        _, _, page2_blocks = convert_digital_page(str(pdf_path), 2)

        with open(graph_path, encoding="utf-8") as f:
            graph_data = json.load(f)

        resolved_count = 0
        total_samples = 0

        for node in graph_data.get("nodes", []):
            for ev in node.get("evidence", []):
                p_num = ev.get("page_number", 2)
                text = ev.get("text", "")
                if not text or len(text) < 10:
                    continue

                target_blocks = page1_blocks if p_num == 1 else page2_blocks
                b_ids, bbox = resolve_evidence_to_blocks(text, target_blocks)

                total_samples += 1
                if b_ids:
                    resolved_count += 1
                    assert bbox is not None
                    assert "x0" in bbox and "y0" in bbox and "x1" in bbox and "y1" in bbox
                    assert bbox["x1"] >= bbox["x0"]
                    assert bbox["y1"] >= bbox["y0"]

        assert total_samples > 5, "Expected at least 5 real evidence samples"
        resolution_rate = resolved_count / total_samples
        assert resolution_rate >= 0.85, f"Expected >= 85% resolution rate on real data, got {resolution_rate:.2%}"


# ---------------------------------------------------------------------------
# Models Serialization Round-Trip
# ---------------------------------------------------------------------------

class TestEvidenceModelSerialization:
    """Verify Evidence dataclass backward and forward compatibility."""

    def test_evidence_to_dict_and_from_dict_round_trip(self):
        ev = Evidence(
            page_number=2,
            text="Sample verified snippet",
            confidence=0.95,
            block_ids=["p2-docling-3", "p2-docling-4"],
            bbox={"x0": 45.0, "y0": 100.0, "x1": 500.0, "y1": 180.0},
        )
        d = ev.to_dict()
        assert d["block_ids"] == ["p2-docling-3", "p2-docling-4"]
        assert d["bbox"] == {"x0": 45.0, "y0": 100.0, "x1": 500.0, "y1": 180.0}

        reconstructed = Evidence.from_dict(d)
        assert reconstructed.page_number == 2
        assert reconstructed.text == "Sample verified snippet"
        assert reconstructed.confidence == 0.95
        assert reconstructed.block_ids == ["p2-docling-3", "p2-docling-4"]
        assert reconstructed.bbox == {"x0": 45.0, "y0": 100.0, "x1": 500.0, "y1": 180.0}

    def test_legacy_evidence_dict_without_new_fields(self):
        """Old persisted graphs with no block_ids/bbox keys load cleanly."""
        legacy_dict = {
            "page_number": 1,
            "text": "Legacy evidence snippet",
            "confidence": 0.9,
        }
        ev = Evidence.from_dict(legacy_dict)
        assert ev.page_number == 1
        assert ev.text == "Legacy evidence snippet"
        assert ev.confidence == 0.9
        assert ev.block_ids == []
        assert ev.bbox is None
