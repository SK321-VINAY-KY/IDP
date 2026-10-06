"""
File: engine.py
Purpose: Isolated Docling extraction engine for digital PDF pages.
Owner: engineer-a@idp-pilot
"""
import time
from typing import Any, Dict, List, Optional, Tuple

_docling_converter: Any = None


def get_docling_converter() -> Any:
    """Lazy singleton initializer for Docling DocumentConverter."""
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Digital PDFs do not need OCR; prevents RapidOCR read-only filesystem crash
            pipeline_options.do_table_structure = True

            _docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except Exception:
            _docling_converter = False  # Mark unavailable
    return _docling_converter


try:
    from src.ai.schemas.document_ir import Block, BlockType, BBox, TableCell
except ImportError:
    from document_ir import Block, BlockType, BBox, TableCell  # fallback if packaged at root in container


def _docling_bbox_to_ir(prov: Any, page_height: Optional[float] = None) -> Optional[BBox]:
    try:
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None
        # In Docling-Core, bbox coordinates are bottom-left origin by default
        if getattr(bbox, "coord_origin", None) is not None:
            from docling_core.types.doc import CoordOrigin
            if bbox.coord_origin == CoordOrigin.BOTTOMLEFT and page_height:
                tl = bbox.to_top_left_origin(page_height)
                return BBox(x0=float(tl.l), y0=float(tl.t), x1=float(tl.r), y1=float(tl.b))
        # Fallback if already top-left or page_height unavailable
        l = float(getattr(bbox, "l", getattr(bbox, "x0", 0.0)))
        t = float(getattr(bbox, "t", getattr(bbox, "y0", 0.0)))
        r = float(getattr(bbox, "r", getattr(bbox, "x1", 0.0)))
        b = float(getattr(bbox, "b", getattr(bbox, "y1", 0.0)))
        return BBox(x0=l, y0=min(t, b), x1=r, y1=max(t, b))
    except Exception:
        return None


def extract_blocks_from_docling(document: Any, page_number: int) -> List[Block]:
    blocks: List[Block] = []
    order = 0

    page_height = None
    if hasattr(document, "pages") and isinstance(document.pages, dict) and page_number in document.pages:
        page_obj = document.pages[page_number]
        if hasattr(page_obj, "size") and hasattr(page_obj.size, "height"):
            page_height = float(page_obj.size.height)

    for text_item in getattr(document, "texts", []):
        prov = text_item.prov[0] if getattr(text_item, "prov", None) else None
        if prov and getattr(prov, "page_no", page_number) != page_number:
            continue
        label = getattr(text_item, "label", "")
        block_type = BlockType.HEADING if label in ("section_header", "title") else BlockType.PARAGRAPH
        blocks.append(Block(
            block_id=f"p{page_number}-docling-{order}",
            type=block_type,
            text=getattr(text_item, "text", ""),
            bbox=_docling_bbox_to_ir(prov, page_height) if prov else None,
            reading_order=order,
            confidence=0.97,
            source_engine="docling",
        ))
        order += 1

    for table_item in getattr(document, "tables", []):
        prov = table_item.prov[0] if getattr(table_item, "prov", None) else None
        if prov and getattr(prov, "page_no", page_number) != page_number:
            continue
        cells = []
        try:
            grid = table_item.export_to_dataframe()
            for r, row in enumerate(grid.itertuples(index=False)):
                for c, val in enumerate(row):
                    cells.append(TableCell(row=r, column=c, text=str(val)))
        except Exception:
            pass
        blocks.append(Block(
            block_id=f"p{page_number}-docling-{order}",
            type=BlockType.TABLE,
            bbox=_docling_bbox_to_ir(prov, page_height) if prov else None,
            reading_order=order,
            confidence=0.95,
            source_engine="docling",
            cells=cells or None,
        ))
        order += 1

    return blocks


def process_single_page(pdf_path: str, page_number: int) -> Tuple[str, float, List[Block]]:
    """
    Extracts structured markdown from a digital page using Docling.
    Falls back to PyMuPDF raw text extraction if Docling is unavailable or fails.
    Returns (markdown, confidence, blocks).
    """
    try:
        converter = get_docling_converter()
        if converter and converter is not False:
            result = converter.convert(pdf_path, page_range=(page_number, page_number))
            markdown = result.document.export_to_markdown()
            blocks = extract_blocks_from_docling(result.document, page_number)
            if markdown and len(markdown.strip()) > 0:
                confidence = 0.97 if len(markdown.strip()) > 20 else 0.30
                return markdown, confidence, blocks
    except Exception:
        pass

    # High-speed fallback to PyMuPDF text extraction
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        text = page.get_text("text").strip()
        confidence = 0.95 if len(text) > 20 else 0.30
        fallback_blocks = []
        if text:
            fallback_blocks.append(Block(
                block_id=f"p{page_number}-docling-0",
                type=BlockType.PARAGRAPH,
                text=text,
                reading_order=0,
                confidence=confidence,
                source_engine="docling_fallback",
            ))
        return text, confidence, fallback_blocks
    finally:
        doc.close()


def process_pages(pdf_path: str, page_numbers: List[int]) -> List[Dict[str, Any]]:
    """
    Processes requested page numbers and returns standardized result dictionaries.
    """
    results = []
    for p in page_numbers:
        t0 = time.monotonic()
        try:
            markdown, confidence, blocks = process_single_page(pdf_path, p)
            latency_ms = (time.monotonic() - t0) * 1000
            non_empty_lines = [l for l in markdown.splitlines() if l.strip()]
            results.append({
                "page_number": p,
                "engine": "docling",
                "markdown": markdown,
                "confidence": round(confidence, 4),
                "lines_extracted": len(non_empty_lines),
                "blocks": [b.model_dump() for b in blocks],
                "latency_ms": round(latency_ms, 2),
                "status": "SUCCESS",
            })
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            results.append({
                "page_number": p,
                "engine": "docling",
                "markdown": "",
                "confidence": 0.0,
                "lines_extracted": 0,
                "blocks": [],
                "latency_ms": round(latency_ms, 2),
                "status": "ERROR",
                "error": str(exc),
            })
    return results

