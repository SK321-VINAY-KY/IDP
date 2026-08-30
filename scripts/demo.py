"""
Script: demo.py
Purpose: End-to-end pipeline demo using the real pipeline.process_page() so
         PageMetadata is populated and displayed alongside PageOutput.
         Runs both test PDFs through Layer 1 + Layer 2 and prints:
           - Per-page capability detection + engine plan
           - Confidence, chars, latency per engine
           - Full PageMetadata JSON for the last processed page
           - Tail of logs/pipeline.log

Usage:
    py -3.11 scripts/demo.py
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-20 (PageMetadata end-to-end)
Deps: pymupdf, paddleocr, docling, pydantic-settings
"""
import io
import json
import logging
import os
import sys
import time
import uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pymupdf
from PIL import Image

# Suppress PaddleOCR / Paddle framework noise
logging.getLogger("paddleocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)

from src.ai.layer1_routing.pipeline import process_page as pipeline_process_page
from src.ai.schemas.page_metadata import PageMetadata
from src.config.settings import settings
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger("demo")

DPI = 150
MAT = pymupdf.Matrix(DPI / 72, DPI / 72)

# ---------------------------------------------------------------------------
# Stub LLM client — VLM calls are slow on CPU so we skip Step B in this demo.
# Pages that would normally go to Step B fall back to the scanned route.
# ---------------------------------------------------------------------------

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification


class StubLLMClient(LLMClient):
    """Returns a conservative 'scanned' classification without calling Ollama."""

    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        return PageClassification(
            route="scanned",
            confidence=0.60,
            language_hint="en",
            handwriting_pct=0.50,   # dead-zone → both OCR engines will run
            noise_level=0.20,
            needs_preprocessing=[],
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        return "", 0.0


# ---------------------------------------------------------------------------
# Demo config
# ---------------------------------------------------------------------------

DEMOS = [
    {
        "label": "Handwritten OR/LPP Notes (CamScanner)",
        "pdf":   "tests/fixtures/camscanner_handwritten.pdf",
        "pages": [0, 6, 12],   # pages 1, 7, 13
    },
    {
        "label": "SDG Goals (clean digital PDF)",
        "pdf":   "tests/fixtures/sdg_goals.pdf",
        "pages": [0, 17],      # pages 1, 18
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_page(page) -> tuple[np.ndarray, bytes]:
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    png = pix.tobytes("png")
    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return arr, png


def bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


# ---------------------------------------------------------------------------
# Run one document
# ---------------------------------------------------------------------------

def run_demo(demo: dict, llm_client: LLMClient) -> list[tuple]:
    """
    Process the selected pages from one PDF using pipeline.process_page().
    Returns a list of (output, metadata) tuples — one per page.
    """
    pdf_path  = os.path.join(os.path.dirname(__file__), "..", demo["pdf"])
    label     = demo["label"]
    page_idxs = demo["pages"]

    if not os.path.exists(pdf_path):
        print(f"  [SKIP] PDF not found: {pdf_path}")
        return []

    doc        = pymupdf.open(pdf_path)
    total_pages = len(doc)
    run_id     = str(uuid.uuid4())[:8]
    set_correlation_id(run_id)

    logger.info(
        "document.started",
        document=os.path.basename(pdf_path),
        total_pages=total_pages,
        demo_pages=[p + 1 for p in page_idxs],
        routing_mode=settings.routing_mode,
        run_id=run_id,
    )

    print(f"\n{'═'*95}")
    print(f"  {label}")
    print(f"  File        : {os.path.basename(pdf_path)}  ({total_pages} total pages)")
    print(f"  Pages       : {[p+1 for p in page_idxs]}")
    print(f"  routing_mode: {settings.routing_mode}   run_id: {run_id}")
    print(f"{'═'*95}")
    print()
    print(f"  {'Pg':>3}  {'Capabilities':<38}  {'Plan':<38}  {'Conf':>5}  {'Chars':>6}  {'Time':>6}")
    print("  " + "─" * 108)

    page_results = []  # list of (output, metadata)
    summaries = []

    for idx in page_idxs:
        page       = doc[idx]
        page_num   = idx + 1
        arr, png   = render_page(page)

        page_context = {
            "pdf_path":    os.path.abspath(pdf_path),
            "page_number": page_num,
            "image_array": arr,
            "image_bytes": png,
        }

        t0 = time.monotonic()
        output, metadata = pipeline_process_page(
            page=page,
            page_number=page_num,
            page_context=page_context,
            llm_client=llm_client,
            page_image_bytes=png,
            document_name=os.path.basename(pdf_path),
            document_id=run_id,
        )
        elapsed = time.monotonic() - t0
        page_results.append((output, metadata))

        caps_str = ", ".join(output.capabilities) if output.capabilities else "none"
        plan_str = " + ".join(output.engines_used) if output.engines_used else "skip"
        low_flag = " !" if output.low_confidence else ""

        print(f"  {page_num:>3}  {caps_str:<38}  {plan_str:<38}"
              f"  {output.confidence:>5.3f}{low_flag}  {len(output.markdown.strip()):>6}  {elapsed:>5.1f}s")

        # Per-engine breakdown from metadata
        for er in metadata.engine_results:
            conf_s = f"{er.confidence:.3f}" if er.confidence is not None else "—"
            lat_s  = fmt_ms(er.latency_ms)
            print(f"       {'':38}  ↳ {er.engine:<34}  {conf_s}  {lat_s}")

        # Escalation flag
        if metadata.escalated:
            for esc in metadata.escalation_history:
                print(f"       ⬆ ESCALATED  {esc.from_engine} → {esc.to_engine}"
                      f"  (conf={esc.from_confidence:.3f} < threshold={esc.threshold})")

        # Text sample
        sample_text = output.markdown.strip().replace("\n", " ")[:90]
        if sample_text:
            print(f"       Sample: {sample_text}")
        print()

        summaries.append({
            "page":       page_num,
            "confidence": output.confidence,
            "chars":      len(output.markdown.strip()),
            "engines":    output.engines_used,
            "escalated":  output.escalated,
            "low_conf":   output.low_confidence,
            "latency_ms": metadata.total_latency_ms,
        })

    doc.close()

    avg_conf     = sum(s["confidence"] for s in summaries) / len(summaries)
    multi_engine = sum(1 for s in summaries if len(s["engines"]) > 1)
    low_conf_cnt = sum(1 for s in summaries if s["low_conf"])
    total_time   = sum(s["latency_ms"] or 0 for s in summaries)

    logger.info(
        "document.completed",
        document=os.path.basename(pdf_path),
        pages_processed=len(summaries),
        avg_confidence=round(avg_conf, 3),
        multi_engine_pages=multi_engine,
        low_confidence_pages=low_conf_cnt,
        total_elapsed_ms=round(total_time, 1),
        run_id=run_id,
    )

    print(f"  {'─'*108}")
    print(f"  Pages processed   : {len(summaries)} of {total_pages}")
    print(f"  Avg confidence    : {avg_conf:.3f}  {bar(avg_conf)}")
    print(f"  Multi-engine pages: {multi_engine}")
    print(f"  Low-conf pages    : {low_conf_cnt}")
    print(f"  Total time        : {total_time/1000:.1f}s")

    return page_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔" + "═"*93 + "╗")
    print("║  IDP PIPELINE — END-TO-END DEMO  (Layer 1 routing + Layer 2 conversion + PageMetadata)" + " "*2 + "║")
    print(f"║  routing_mode : {settings.routing_mode:<28}  vlm : {settings.vlm_model_name} (stub in demo)" + " "*6 + "║")
    print("╚" + "═"*93 + "╝")
    print(f"\n  Log file  : logs/pipeline.log")
    print(f"  Engines   : Docling · PaddleOCR-printed · PaddleOCR-handwritten")
    print(f"  VLM Step B: stubbed out (Ollama/7B too slow on CPU for a demo)")

    llm_client = StubLLMClient()

    # Collect ALL (output, metadata) pairs across both documents
    all_results: list[tuple] = []
    for demo in DEMOS:
        pairs = run_demo(demo, llm_client)
        all_results.extend(pairs)

    # ── Compact metadata table for EVERY processed page ──────────────────────
    print()
    print("═" * 95)
    print("  PAGE METADATA SUMMARY — all processed pages")
    print("═" * 95)
    print(f"  {'Doc':<28}  {'Pg':>3}  {'Caps':<32}  {'Engine(s)':<32}  "
          f"{'Conf':>5}  {'Chars':>6}  {'Latency':>8}  {'Esc':>3}  {'OK':>3}")
    print("  " + "─" * 140)

    for output, meta in all_results:
        doc_s  = (meta.document_name or "")[:27]
        caps_s = ", ".join(meta.capabilities)[:31] if meta.capabilities else "—"
        eng_s  = ", ".join(
            [er.engine for er in meta.engine_results]
        )[:31] if meta.engine_results else "—"
        conf_s = f"{meta.final_confidence:.3f}" if meta.final_confidence is not None else "—"
        lat_s  = fmt_ms(meta.total_latency_ms)
        esc_s  = str(meta.escalation_count) if meta.escalated else "—"
        ok_s   = "✓" if meta.processing_success else "✗"

        print(f"  {doc_s:<28}  {meta.page_number:>3}  {caps_s:<32}  {eng_s:<32}  "
              f"{conf_s:>5}  {output.confidence * len(output.markdown.strip()):>6.0f}  "
              f"{lat_s:>8}  {esc_s:>3}  {ok_s:>3}")

    # ── Full PageMetadata JSON for the first handwritten page ─────────────────
    handwritten_pair = next(
        ((o, m) for o, m in all_results
         if "has_handwriting" in (m.capabilities or []) and
            "has_digital_text" not in (m.capabilities or [])),
        None,
    )
    if handwritten_pair:
        _, hw_meta = handwritten_pair
        print()
        print("═" * 95)
        print(f"  FULL PageMetadata JSON — handwritten page "
              f"(doc: {hw_meta.document_name}  pg: {hw_meta.page_number})")
        print("═" * 95)
        print(hw_meta.to_json())

    # ── Full PageMetadata JSON for first digital-only page ────────────────────
    digital_pair = next(
        ((o, m) for o, m in all_results
         if m.capabilities == ["has_digital_text"]),
        None,
    )
    if digital_pair:
        _, dig_meta = digital_pair
        print()
        print("═" * 95)
        print(f"  FULL PageMetadata JSON — digital-only page "
              f"(doc: {dig_meta.document_name}  pg: {dig_meta.page_number})")
        print("═" * 95)
        print(dig_meta.to_json())

    # ── Tail the log file ────────────────────────────────────────────────────
    print()
    print("═" * 95)
    print("  LAST 20 LOG LINES  (logs/pipeline.log)")
    print("═" * 95)
    try:
        with open("logs/pipeline.log", encoding="utf-8") as f:
            raw_lines = f.readlines()
        for raw in raw_lines[-20:]:
            try:
                entry    = json.loads(raw.strip())
                ts       = entry.get("timestamp", "")[11:23]
                lvl      = entry.get("level", "")
                evt      = entry.get("event", "")
                skip     = {"timestamp","level","service","logger","event","correlation_id"}
                extras   = {k: v for k, v in entry.items() if k not in skip and v is not None}
                ext_str  = "  " + "  ".join(f"{k}={v}" for k, v in list(extras.items())[:5])
                print(f"  {ts}  {lvl:<7} {evt:<38}{ext_str}")
            except Exception:
                print(f"  {raw.strip()[:115]}")
    except FileNotFoundError:
        print("  (log file not found)")
    print()


if __name__ == "__main__":
    main()
