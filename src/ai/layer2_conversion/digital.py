"""
File: digital.py
Purpose: Layer 2 "digital" route — native text extraction via Docling for
         PDFs with embedded, selectable text. Free, instant, zero OCR error.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: docling
"""
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Module-level singleton cache for DocumentConverter — mirrors the caching
# pattern used by _paddle_engines in scanned.py to avoid repeatedly paying
# Docling's initialization cost on every digital page conversion in warm runtimes.
_docling_converter: Any = None

# Optional module-level symbol so tests can mock/patch DocumentConverter directly
DocumentConverter: Any = None


def _get_docling_converter() -> Any:
    """
    Return a lazily-loaded DocumentConverter singleton instance.
    Configured with do_ocr=False to keep digital text extraction fast and lightweight.
    """
    global _docling_converter
    if _docling_converter is None:
        if DocumentConverter is not None:
            converter_cls = DocumentConverter
            _docling_converter = converter_cls()
        else:
            try:
                from docling.document_converter import DocumentConverter as _DocConverter, PdfFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = True

                _docling_converter = _DocConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            except Exception:
                from docling.document_converter import DocumentConverter as _DocConverter
                _docling_converter = _DocConverter()

        logger.info("layer2.digital.docling_converter_loaded")

    return _docling_converter


# Alias for compatibility if referenced as _get_document_converter
_get_document_converter = _get_docling_converter


from src.ai.schemas.document_ir import Block, BlockType, BBox, TableCell


def _docling_bbox_to_ir(prov: Any, page_height: float | None = None) -> BBox | None:
    try:
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None
        if getattr(bbox, "coord_origin", None) is not None:
            from docling_core.types.doc import CoordOrigin
            if bbox.coord_origin == CoordOrigin.BOTTOMLEFT and page_height:
                tl = bbox.to_top_left_origin(page_height)
                return BBox(x0=float(tl.l), y0=float(tl.t), x1=float(tl.r), y1=float(tl.b))
        l = float(getattr(bbox, "l", getattr(bbox, "x0", 0.0)))
        t = float(getattr(bbox, "t", getattr(bbox, "y0", 0.0)))
        r = float(getattr(bbox, "r", getattr(bbox, "x1", 0.0)))
        b = float(getattr(bbox, "b", getattr(bbox, "y1", 0.0)))
        return BBox(x0=l, y0=min(t, b), x1=r, y1=max(t, b))
    except Exception:
        return None


def extract_blocks_from_docling(document: Any, page_number: int) -> list[Block]:
    blocks: list[Block] = []
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


def _extract_pymupdf_fallback(pdf_path: str, page_number: int) -> tuple[str, float, list[Block]]:
    """Fallback text extraction via PyMuPDF when Docling is unavailable or fails."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return "", 0.0, []

    try:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            raw_text = page.get_text("text")
            text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()
            confidence = 0.95 if len(text) > 20 else 0.30
            fallback_blocks: list[Block] = []
            if text:
                fallback_blocks.append(Block(
                    block_id=f"p{page_number}-docling-0",
                    type=BlockType.PARAGRAPH,
                    text=text,
                    reading_order=0,
                    confidence=confidence,
                    source_engine="docling_fallback",
                ))
            logger.info(
                "layer2.digital.converted",
                page_number=page_number,
                engine="pymupdf_fallback",
                markdown_length=len(text),
                confidence=confidence,
            )
            return text, confidence, fallback_blocks
        finally:
            doc.close()
    except Exception as exc:
        logger.error(
            "layer2.digital.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0, []


def convert_digital_page(pdf_path: str, page_number: int) -> tuple[str, float, list[Block]]:
    """
    Converts a single digital-text page to markdown using Docling.
    Returns (markdown, confidence, blocks). Confidence is near-1.0 for digital text
    since there's no OCR uncertainty — the text is extracted, not recognized.
    Falls back to raw PyMuPDF text extraction if Docling is not installed or fails.
    """
    try:
        converter = _get_docling_converter()
        result = converter.convert(pdf_path, page_range=(page_number, page_number))
        markdown = result.document.export_to_markdown()
        blocks = extract_blocks_from_docling(result.document, page_number)
        if not markdown or not markdown.strip():
            # If docling produced empty output on a page with digital text, fall back to PyMuPDF
            return _extract_pymupdf_fallback(pdf_path, page_number)

        confidence = 0.97 if len(markdown.strip()) > 20 else 0.30
        logger.info(
            "layer2.digital.converted",
            page_number=page_number,
            engine="docling",
            markdown_length=len(markdown),
            confidence=confidence,
            blocks_count=len(blocks),
        )
        return markdown, confidence, blocks

    except (ModuleNotFoundError, ImportError):
        logger.warning(
            "layer2.digital.docling_unavailable",
            page_number=page_number,
            fallback="pymupdf_raw_text",
        )
        return _extract_pymupdf_fallback(pdf_path, page_number)

    except Exception as exc:
        logger.warning(
            "layer2.digital.docling_failed_fallback_pymupdf",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return _extract_pymupdf_fallback(pdf_path, page_number)

