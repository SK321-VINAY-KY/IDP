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
    Mirrors the _get_paddle_engine pattern from scanned.py.
    """
    global _docling_converter
    if _docling_converter is None:
        if DocumentConverter is not None:
            converter_cls = DocumentConverter
        else:
            from docling.document_converter import DocumentConverter as _DocConverter

            converter_cls = _DocConverter

        _docling_converter = converter_cls()
        logger.info("layer2.digital.docling_converter_loaded")

    return _docling_converter


# Alias for compatibility if referenced as _get_document_converter
_get_document_converter = _get_docling_converter


def convert_digital_page(pdf_path: str, page_number: int) -> tuple[str, float]:
    """
    Converts a single digital-text page to markdown using Docling.
    Returns (markdown, confidence). Confidence is near-1.0 for digital text
    since there's no OCR uncertainty — the text is extracted, not recognized.
    Falls back to raw PyMuPDF text extraction if Docling is not installed.
    """
    try:
        converter = _get_docling_converter()
        result = converter.convert(pdf_path, page_range=(page_number, page_number))
        markdown = result.document.export_to_markdown()
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
        # Docling not installed — fall back to PyMuPDF direct text extraction.
        # Functionally equivalent for plain digital PDFs; no table/heading structure.
        logger.warning(
            "layer2.digital.docling_unavailable",
            page_number=page_number,
            fallback="pymupdf_raw_text",
        )
        try:
            import pymupdf
            doc  = pymupdf.open(pdf_path)
            page = doc[page_number - 1]
            text = page.get_text("text").strip()
            doc.close()
            confidence = 0.95 if len(text) > 20 else 0.30
            logger.info(
                "layer2.digital.converted",
                page_number=page_number,
                engine="pymupdf_fallback",
                markdown_length=len(text),
                confidence=confidence,
            )
            return text, confidence
        except Exception as exc:
            logger.error("layer2.digital.failed", page_number=page_number,
                         error=str(exc), error_type=type(exc).__name__)
            return "", 0.0

    except Exception as exc:
        logger.error(
            "layer2.digital.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "", 0.0
