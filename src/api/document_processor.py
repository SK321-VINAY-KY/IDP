"""
File: document_processor.py
Purpose: Bridge between an uploaded PDF on disk and the list[dict] contract
         that src.ai.layer1_routing.pipeline.process_document() expects.
         Not part of the Layer 1/2 contract itself — this is API-side glue.
Owner: api@idp-pilot
Created: 2026-08-26
"""
from typing import Any

import fitz  # PyMuPDF
import numpy as np


def build_pages_for_document(pdf_path: str) -> tuple[Any, list[dict]]:
    """
    Opens the PDF and returns (doc, pages) where `doc` must be kept open
    (and closed by the caller) for the lifetime of pipeline processing,
    since the page objects referenced in `pages` are bound to it.
    """
    doc = fitz.open(pdf_path)
    pages: list[dict] = []

    for index in range(len(doc)):
        page = doc[index]
        page_number = index + 1

        # Render once at a reasonable resolution for both the classifier
        # image and the OCR engines (scanned/handwritten routes).
        pixmap = page.get_pixmap(dpi=200)
        image_bytes = pixmap.tobytes("png")
        image_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )

        pages.append(
            {
                "page": page,
                "page_number": page_number,
                "context": {
                    "pdf_path": pdf_path,
                    "image_array": image_array,
                    "image_bytes": image_bytes,
                },
                "image_bytes": image_bytes,
            }
        )

    return doc, pages