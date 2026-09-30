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


def _extract_pymupdf_fallback(pdf_path: str, page_number: int) -> tuple[str, float]:
    """Fallback text extraction via PyMuPDF when Docling is unavailable or fails."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return "", 0.0

    try:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            text = page.get_text("text").strip()
            confidence = 0.95 if len(text) > 20 else 0.30
            logger.info(
                "layer2.digital.converted",
                page_number=page_number,
                engine="pymupdf_fallback",
                markdown_length=len(text),
                confidence=confidence,
            )
            return text, confidence
        finally:
            doc.close()
    except Exception as exc:
        logger.error(
            "layer2.digital.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0


def convert_digital_page(pdf_path: str, page_number: int) -> tuple[str, float]:
    """
    Converts a single digital-text page to markdown using Docling.
    Returns (markdown, confidence). Confidence is near-1.0 for digital text
    since there's no OCR uncertainty — the text is extracted, not recognized.
    Falls back to raw PyMuPDF text extraction if Docling is not installed or fails.
    """
    try:
        converter = _get_docling_converter()
        result = converter.convert(pdf_path, page_range=(page_number, page_number))
        markdown = result.document.export_to_markdown()
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
        )
        return markdown, confidence

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
