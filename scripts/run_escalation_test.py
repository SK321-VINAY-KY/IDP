"""Run a single-page end-to-end test that forces escalation to the VLM.

This renders page 1 of the CamScanner handwritten PDF, runs the normal
plan (PaddleOCR printed + handwritten), and because `escalation_confidence_threshold`
is high the pipeline should invoke `vlm_transcribe` as a fallback. Use this
to verify your Gemini key is accepted by the selected provider.
"""
import os
import io
import pymupdf
from PIL import Image
import numpy as np

from src.adapters.llm.factory import get_llm_client
from src.ai.layer1_routing.pipeline import process_page
from src.config.settings import settings

PDF = os.path.join("tests", "fixtures", "camscanner_handwritten.pdf")


def render_page_bytes(page, dpi=150) -> tuple[bytes, np.ndarray]:
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    img_bytes = pix.tobytes("png")
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img_bytes, np.array(pil)


def main():
    client = get_llm_client()
    print(f"Using LLM provider: {type(client).__name__}")

    doc = pymupdf.open(PDF)
    page = doc[0]
    page_number = 1
    img_bytes, image_array = render_page_bytes(page)

    page_context = {
        "pdf_path": PDF,
        "image_array": image_array,
    }

    print("Processing page 1 (this may take a while)...")
    output = process_page(page, page_number, page_context, client, img_bytes)

    print("\n--- Output ---")
    print("Page:", output.page_number)
    print("Engines used:", output.engines_used)
    print("Escalated:", output.escalated)
    print("Escalation attempts:", output.escalation_attempts)
    print("Confidence:", output.confidence)
    print("--- Markdown (truncated) ---")
    print(output.markdown[:2000])

    doc.close()


if __name__ == "__main__":
    main()
