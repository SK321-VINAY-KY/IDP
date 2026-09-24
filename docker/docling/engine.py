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


def process_single_page(pdf_path: str, page_number: int) -> Tuple[str, float]:
    """
    Extracts structured markdown from a digital page using Docling.
    Falls back to PyMuPDF raw text extraction if Docling is unavailable or fails.
    """
    try:
        converter = get_docling_converter()
        if converter and converter is not False:
            result = converter.convert(pdf_path, page_range=(page_number, page_number))
            markdown = result.document.export_to_markdown()
            if markdown and len(markdown.strip()) > 0:
                confidence = 0.97 if len(markdown.strip()) > 20 else 0.30
                return markdown, confidence
    except Exception:
        pass

    # High-speed fallback to PyMuPDF text extraction
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        text = page.get_text("text").strip()
        confidence = 0.95 if len(text) > 20 else 0.30
        return text, confidence
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
            markdown, confidence = process_single_page(pdf_path, p)
            latency_ms = (time.monotonic() - t0) * 1000
            non_empty_lines = [l for l in markdown.splitlines() if l.strip()]
            results.append({
                "page_number": p,
                "engine": "docling",
                "markdown": markdown,
                "confidence": round(confidence, 4),
                "lines_extracted": len(non_empty_lines),
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
                "latency_ms": round(latency_ms, 2),
                "status": "ERROR",
                "error": str(exc),
            })
    return results
