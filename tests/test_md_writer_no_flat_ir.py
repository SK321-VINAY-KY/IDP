"""
Phase 4 commit 3 regression: md_writer.write_document must NO LONGER emit a flat
<stem>.ir.json next to the .md. That file wrote document text to a shared,
non-owner-scoped location and nothing read it. The .md output must still exist.

Synthetic PageOutput only; offline; no model, no AWS.
"""
from pathlib import Path

from src.ai.output.md_writer import write_document
from src.ai.schemas.page import PageOutput
from src.ai.schemas.page_metadata import PageMetadata


def _page(n, md):
    return PageOutput(
        page_number=n, markdown=md, engines_used=["docling"], confidence=0.9,
        capabilities=["digital_pdf"], escalated=False, escalation_attempts=0,
        low_confidence=False, blocks=[],
    )


def test_write_document_does_not_write_flat_ir_json(tmp_path):
    pages = [(_page(1, "hello page one"), PageMetadata(document_name="doc.pdf", page_number=1))]
    out = write_document("doc.pdf", pages, output_dir=str(tmp_path), overwrite=True)

    md_file = Path(out)
    assert md_file.is_file() and md_file.suffix == ".md"
    # the old sibling flat IR file must NOT be produced
    assert not (tmp_path / "doc.ir.json").exists()
    # and no .ir.json anywhere in the output dir
    assert list(tmp_path.glob("*.ir.json")) == []
