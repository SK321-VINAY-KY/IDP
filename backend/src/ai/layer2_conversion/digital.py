"""
File: digital.py
Purpose: Layer 2 "digital" route — native text extraction via Docling for
         PDFs with embedded, selectable text. Free, instant, zero OCR error.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: docling
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_digital_page(pdf_path: str, page_number: int) -> tuple[str, float]:
    """
    Converts a single digital-text page to markdown using Docling.
    Returns (markdown, confidence). Confidence is near-1.0 for digital text
    since there's no OCR uncertainty — the text is extracted, not recognized.
    """
    from docling.document_converter import DocumentConverter

    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path, page_range=(page_number, page_number))
        markdown = result.document.export_to_markdown()

        # Structural sanity check: a digital page that Docling parses into
        # near-empty markdown likely means the page wasn't actually digital
        # text (mis-routed) — flag it for escalation rather than trust it blindly.
        confidence = 0.97 if len(markdown.strip()) > 20 else 0.30

        logger.info(
            "layer2.digital.converted",
            page_number=page_number,
            markdown_length=len(markdown),
            confidence=confidence,
        )
        return markdown, confidence

    except Exception as exc:
        logger.error(
            "layer2.digital.failed",
            page_number=page_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        # Signal downstream escalation rather than raising — a broken digital
        # parse should re-render as an image and go through the scanned route.
        return "", 0.0
