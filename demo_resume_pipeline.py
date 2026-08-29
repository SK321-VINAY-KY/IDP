"""
File: demo_resume_pipeline.py
Purpose: End-to-end demo of Engineer A's pipeline on the two resume PDFs.
         Runs both resumes through process_document() with the new
         write_output option so .md files are created automatically.

Flow:
    1. Load MY_resume.pdf + Vinay_Resume.pdf from tests/fixtures/
    2. For each PDF: render every page to numpy + PNG bytes (context shape
       expected by process_document)
    3. Call pipeline.process_document(write_output=True, output_dir=tests/fixtures)
    4. Report per-page: route, engines used, confidence, chars extracted
    5. Write .md outputs: tests/fixtures/MY_resume.md
                          tests/fixtures/Vinay_Resume.md

Note: extraction_requirements=None is INTENTIONAL.
      The confirmed schema from the chatbot is cached by Engineer B's Layer 3
      and applied to the .md files post-hoc via LLM-based field extraction.
      Engineer A never sees field-level requirements — that separation of
      concerns is deliberate.

Deps: pymupdf, PIL, numpy, pydantic-settings
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pymupdf
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, VLMAnalysis
from src.ai.layer1_routing.pipeline import process_document
from src.config.settings import settings
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures")
RESUME_PDFS = [
    "MY_resume.pdf",
    "Vinay_Resume.pdf",
]
RENDER_DPI = 150
MAT = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)


# ---------------------------------------------------------------------------
# Mock LLM client — digital resumes never hit the VLM classification step,
# so this client is just a compliant stand-in that loudly fails if called.
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Stand-in LLM client that raises if any VLM method is actually invoked.

    Digital resumes (char_count >> digital_char_count_threshold=100) route
    directly to Docling in Step A without a Step B VLM call, so for clean
    digital PDFs none of these methods should ever fire.
    """

    def classify_page(
        self, image_bytes: bytes, page_profile_hint: dict
    ) -> PageClassification:
        raise RuntimeError(
            "MockLLMClient.classify_page called — routing decision should "
            "have been conclusive in Step A for a digital resume. Check "
            "inspect_page() thresholds."
        )

    def analyze_page(
        self, image_bytes: bytes, page_profile_hint: dict
    ) -> VLMAnalysis:
        raise RuntimeError(
            "MockLLMClient.analyze_page called — see classify_page()."
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        raise RuntimeError(
            "MockLLMClient.transcribe_handwriting called — resume should "
            "be digital, no VLM transcription needed."
        )


# ---------------------------------------------------------------------------
# PDF -> page context builders
# ---------------------------------------------------------------------------

def _render_page(page) -> tuple[np.ndarray, bytes]:
    """Render one PyMuPDF page to (numpy_array, png_bytes) at RENDER_DPI."""
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    png_bytes = pix.tobytes("png")
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil)
    return arr, png_bytes


def _build_page_list(pdf_path: str) -> list[dict]:
    """Build the pages[] structure required by process_document().

    Each entry must have:
        page          — PyMuPDF page object
        page_number   — int (1-based)
        context       — dict with pdf_path, image_array, image_bytes
        image_bytes   — PNG bytes
        llm_client    — optional override (we use global client)
    """
    doc = pymupdf.open(pdf_path)
    pages: list[dict] = []
    for i, page in enumerate(doc):
        image_array, image_bytes = _render_page(page)
        pages.append({
            "page":        page,
            "page_number": i + 1,
            "context": {
                "pdf_path":    pdf_path,
                "image_array": image_array,
                "image_bytes": image_bytes,
            },
            "image_bytes": image_bytes,
        })
    # NOTE: we intentionally do NOT close `doc` here — process_document()
    # still holds references to the PyMuPDF page objects and will call
    # get_text() / get_drawings() etc. on them. The GC will close the doc
    # after pages[] goes out of scope.
    return pages


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_page_report(doc_name: str, results) -> None:
    """Pretty-print a per-page processing report for one document."""
    print(f"\n  {'Pg':>2}  {'Route/Engine':<22}  {'Conf':>6}  "
          f"{'Chars':>6}  {'Esc':>3}  {'LowC':>4}  {'Caps'}")
    print(f"  {'--':>2}  {'------------':<22}  {'----':>6}  "
          f"{'-----':>6}  {'---':>3}  {'----':>4}  {'----'}")
    for output, metadata in results:
        engine = (output.engines_used[0] if output.engines_used else "-")
        extras = output.engines_used[1:]
        engine_col = engine + (f" +{len(extras)}" if extras else "")
        esc = " !" if output.escalated else ""
        lc  = " !" if output.low_confidence else ""
        caps = ",".join(output.capabilities) if output.capabilities else "-"
        chars = len(output.markdown.strip())
        print(f"  {output.page_number:2d}  {engine_col:<22}  "
              f"{output.confidence:6.3f}  {chars:6d}  "
              f"{esc:>3}  {lc:>4}  {caps}")

    outputs = [r[0] for r in results]
    avg_conf = (
        sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0
    )
    total_chars = sum(len(o.markdown.strip()) for o in outputs)
    print(f"\n  Avg confidence : {avg_conf:.3f}")
    print(f"  Total chars    : {total_chars}")
    print(f"  Escalated pgs  : {sum(1 for o in outputs if o.escalated)}")
    print(f"  Low-conf pgs   : {sum(1 for o in outputs if o.low_confidence)}")
    print(f"  Multi-engine   : {sum(1 for o in outputs if len(o.engines_used) > 1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    set_correlation_id(f"demo-resume-{int(time.time())}")
    llm_client = MockLLMClient()

    print("\n" + "=" * 72)
    print("  IDP Engineer A - Resume Pipeline Demo")
    print("=" * 72)
    print(f"  Routing mode           : {settings.routing_mode}")
    print(f"  Digital char threshold : {settings.digital_char_count_threshold}")
    print(f"  Escalation threshold   : {settings.escalation_confidence_threshold}")
    print(f"  Output dir             : {FIXTURES_DIR}")
    print(f"  LLM client             : Mock (digital pages should not need VLM)")
    print("=" * 72)

    all_ok = True
    written_paths: list[str] = []

    for pdf_name in RESUME_PDFS:
        pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"\n[SKIP] {pdf_name} - not found at {pdf_path}")
            all_ok = False
            continue

        stem = os.path.splitext(pdf_name)[0]
        t0 = time.monotonic()
        print(f"\n{'-' * 72}")
        print(f">> {pdf_name}")
        print(f"{'-' * 72}")

        try:
            pages = _build_page_list(pdf_path)
            print(f"  Rendered {len(pages)} pages at {RENDER_DPI} DPI")

            results = process_document(
                pages=pages,
                llm_client=llm_client,
                document_name=pdf_name,
                document_id=f"demo-{stem}",
                write_output=True,
                output_dir=FIXTURES_DIR,
                overwrite=True,
            )

            elapsed = time.monotonic() - t0
            print(f"  process_document() finished in {elapsed:.1f}s")
            _print_page_report(pdf_name, results)

            out_path = os.path.join(FIXTURES_DIR, f"{stem}.md")
            if os.path.exists(out_path):
                written_paths.append(out_path)
                sz = os.path.getsize(out_path)
                print(f"  [OK] Markdown written -> {out_path} ({sz:,} bytes)")
            else:
                print(f"  [WARN] No .md output found at expected path {out_path}")
                all_ok = False

        except Exception as exc:
            logger.error("demo.document_failed", document=pdf_name,
                         error=str(exc), error_type=type(exc).__name__)
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            all_ok = False

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    if written_paths:
        print(f"  Output files written: {len(written_paths)}")
        for p in written_paths:
            print(f"    - {p}")
    else:
        print("  [WARN] No output files were written.")

    print(f"\n  Result: {'ALL OK' if all_ok else 'SOME STEPS FAILED - check logs/pipeline.log'}")
    print("=" * 72 + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
