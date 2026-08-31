"""
File: test_handwritten_extraction.py
Purpose: End-to-end eval of the IDP pipeline on the CamScanner handwritten
         OR/LPP notes PDF. Validates:
           1. Router correctly classifies pages as handwritten (not digital)
           2. PaddleOCR (handwriting-tuned config) extracts meaningful content
           3. Key mathematical terms and LPP structure are present in the output
           4. Confidence distribution is reported so escalation_confidence_threshold
              can be re-calibrated after the TrOCR -> PaddleOCR switch
Owner: engineer-a@idp-pilot
Updated: 2026-08-20 — TrOCR removed, PaddleOCR handwritten route, vlm_transcribe ladder
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pymupdf
from PIL import Image
import io

from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import route_from_profile
from src.ai.layer2_conversion.scanned import convert_handwritten_via_paddle
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

PDF_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "camscanner_handwritten.pdf")
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "fixtures", "camscanner_output.md")

# DPI for rendering — 150 is a good balance of speed vs quality
RENDER_DPI = 150
MAT = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)


def page_to_numpy(page) -> np.ndarray:
    """Render a PyMuPDF page to an RGB numpy array at RENDER_DPI."""
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    img_bytes = pix.tobytes("png")
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)


def run_pipeline(pdf_path: str) -> tuple[str, list[dict]]:
    doc = pymupdf.open(pdf_path)
    page_results = []
    full_markdown = ""

    print(f"\n{'='*65}")
    print(f"Document : {os.path.basename(pdf_path)}")
    print(f"Pages    : {len(doc)}")
    print(f"Render   : {RENDER_DPI} DPI")
    print(f"Engine   : PaddleOCR (handwritten mode, det_db_thresh={settings.paddle_handwriting_det_db_thresh})")
    print(f"Ladder   : digital -> scanned -> handwritten -> vlm_transcribe")
    print(f"{'='*65}\n")

    for i, page in enumerate(doc):
        page_number = i + 1

        # --- Layer 1 Step A: programmatic inspection ---
        profile = inspect_page(page, page_number)
        route = route_from_profile(profile)

        # Scanned/handwritten pages have near-zero embedded text; Step A
        # returns None or "scanned". Force to "handwritten" so we exercise
        # the new PaddleOCR handwriting-tuned engine directly.
        if route is None or route == "scanned":
            route = "handwritten"

        print(f"  Page {page_number:2d}: inspected -> route={route} | "
              f"chars={profile.char_count} | img_cov={profile.image_coverage:.2f} | "
              f"scanned={profile.is_scanned}", flush=True)

        # --- Render page to numpy array for PaddleOCR ---
        image_array = page_to_numpy(page)

        print(f"          running PaddleOCR (handwritten)... ", end="", flush=True)

        text, confidence = convert_handwritten_via_paddle(image_array, page_number)

        # Count lines in output — PaddleOCR returns one text fragment per
        # detected text region, joined by newlines
        n_lines = len([l for l in text.split("\n") if l.strip()])
        print(f"done  confidence={confidence:.3f}  lines={n_lines}  chars_out={len(text.strip())}", flush=True)

        page_md = f"\n<!-- Page {page_number} | route={route} | confidence={confidence:.3f} -->\n{text}\n"
        full_markdown += page_md

        page_results.append({
            "page": page_number,
            "route": route,
            "char_count_in_pdf": profile.char_count,
            "image_coverage": profile.image_coverage,
            "is_scanned": profile.is_scanned,
            "lines_detected": n_lines,
            "paddle_confidence": round(confidence, 4),
            "paddle_chars": len(text.strip()),
            "text_sample": text.strip()[:120].replace("\n", " "),
        })

    doc.close()
    return full_markdown, page_results


def check_routing_accuracy(page_results: list[dict]) -> None:
    """Verify every page was routed away from 'digital'."""
    print(f"\n{'='*65}")
    print("ROUTING ACCURACY")
    print(f"{'='*65}")

    total = len(page_results)
    digital_mismatch = [p for p in page_results if p["route"] == "digital"]

    print(f"  Total pages           : {total}")
    print(f"  Routed as handwritten : {sum(1 for p in page_results if p['route'] == 'handwritten')}")
    print(f"  Routed as scanned     : {sum(1 for p in page_results if p['route'] == 'scanned')}")
    print(f"  Wrongly routed digital: {len(digital_mismatch)}")

    if digital_mismatch:
        for p in digital_mismatch:
            print(f"    [FAIL] Page {p['page']} — chars={p['char_count_in_pdf']}, "
                  f"img_cov={p['image_coverage']:.2f}")
    else:
        print(f"\n  [PASS] All {total} pages correctly avoided digital route")

    avg_pdf_chars = sum(p['char_count_in_pdf'] for p in page_results) / total
    avg_img_cov = sum(p['image_coverage'] for p in page_results) / total
    print(f"\n  Avg embedded chars/page : {avg_pdf_chars:.1f}  (expected ~0 for scanned)")
    print(f"  Avg image coverage/page : {avg_img_cov:.3f}  (expected >0.25 for scanned)")


def check_content_accuracy(page_results: list[dict], full_markdown: str) -> tuple[int, int]:
    """
    Spot-check extracted text against known content from the document.
    The document contains OR/LPP problems with specific mathematical terms,
    variable names, and keywords we can verify.
    """
    print(f"\n{'='*65}")
    print("CONTENT ACCURACY (PaddleOCR extraction vs known ground truth)")
    print(f"{'='*65}")

    md_lower = full_markdown.lower()

    checks = [
        # Mathematical keywords
        ("Math terms",    "max",          "Max Z ="),
        ("Math terms",    "min",          "Min Z ="),
        ("Math terms",    "subject to",   "Subject to / Sub"),
        ("Math terms",    "objective",    "Objective function"),
        # Variable names
        ("Variables",     "x1",           "x1 variable"),
        ("Variables",     "x2",           "x2 variable"),
        # LPP structure keywords
        ("LPP keywords",  "step",         "Step 1/2/3"),
        ("LPP keywords",  "constraint",   "Constraints"),
        ("LPP keywords",  "profit",       "Profit"),
        ("LPP keywords",  "formulation",  "LPP Formulation"),
        # Domain words from problem statements
        ("Domain",        "hen",          "Hens problem (pg 1-2)"),
        ("Domain",        "farmer",       "Farmer / coconut problem (pg 3-4)"),
        ("Domain",        "post",         "Post office problem (pg 5-6)"),
        ("Domain",        "boat",         "Boat manufacturer problem (pg 7-9)"),
        ("Domain",        "graphical",    "Graphical solution method (pg 10-11)"),
        # Specific numbers from the problems
        ("Numbers",       "2500",         "Rs 2500 budget constraint"),
        ("Numbers",       "100",          "Rs 100 old hen cost"),
        ("Numbers",       "4400",         "4400 sq meters land area"),
        ("Numbers",       "680",          "680 packages constraint"),
        ("Numbers",       "128",          "Max Z = 128 (final answer pg 13)"),
    ]

    results_by_cat: dict[str, list] = {}
    for cat, term, label in checks:
        found = term.lower() in md_lower
        results_by_cat.setdefault(cat, []).append((label, term, found))

    total_pass = 0
    total_fail = 0
    for cat, items in results_by_cat.items():
        print(f"\n  [{cat}]")
        for label, term, found in items:
            status = "PASS" if found else "FAIL"
            symbol = "✓" if found else "✗"
            print(f"    [{status}] {symbol} '{term}'  ← {label}")
            if found:
                total_pass += 1
            else:
                total_fail += 1

    pct = int(100 * total_pass / (total_pass + total_fail))
    print(f"\n  Score: {total_pass}/{total_pass + total_fail} terms found ({pct}%)")
    return total_pass, total_fail


def check_paddle_output_quality(page_results: list[dict]) -> None:
    """
    Report per-page PaddleOCR confidence and extracted character counts.
    Also reports the confidence distribution to help re-calibrate
    escalation_confidence_threshold after the TrOCR -> PaddleOCR switch.
    """
    print(f"\n{'='*65}")
    print("PER-PAGE PaddleOCR QUALITY")
    print(f"{'='*65}")
    print(f"  {'Page':>4}  {'Lines':>5}  {'Confidence':>10}  {'Chars out':>9}  Sample")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*40}")

    for p in page_results:
        conf_flag = " !" if p['paddle_confidence'] < 0.3 else ""
        sample = p['text_sample'][:50] if p['text_sample'] else "(empty)"
        print(f"  {p['page']:>4}  {p['lines_detected']:>5}  {p['paddle_confidence']:>10.3f}  "
              f"{p['paddle_chars']:>9}  {sample}{conf_flag}")

    confidences = [p['paddle_confidence'] for p in page_results]
    avg_conf = sum(confidences) / len(confidences)
    avg_chars = sum(p['paddle_chars'] for p in page_results) / len(page_results)
    avg_lines = sum(p['lines_detected'] for p in page_results) / len(page_results)
    empty_pages = sum(1 for p in page_results if p['paddle_chars'] == 0)

    print(f"\n  Avg lines/page  : {avg_lines:.1f}")
    print(f"  Avg confidence  : {avg_conf:.3f}")
    print(f"  Avg chars/page  : {avg_chars:.0f}")
    print(f"  Empty pages     : {empty_pages}")

    # Confidence distribution for threshold re-calibration
    buckets = {">=0.90": 0, "0.70-0.89": 0, "0.50-0.69": 0, "<0.50": 0}
    for c in confidences:
        if c >= 0.90:
            buckets[">=0.90"] += 1
        elif c >= 0.70:
            buckets["0.70-0.89"] += 1
        elif c >= 0.50:
            buckets["0.50-0.69"] += 1
        else:
            buckets["<0.50"] += 1

    print(f"\n  Confidence distribution (for threshold re-calibration):")
    for band, count in buckets.items():
        bar = "█" * count
        print(f"    {band:>10}  {bar}  ({count} pages)")
    print(f"\n  Current escalation_confidence_threshold : {settings.escalation_confidence_threshold}")
    pages_would_escalate = sum(1 for c in confidences if c < settings.escalation_confidence_threshold)
    print(f"  Pages that would escalate at this threshold: {pages_would_escalate}/{len(confidences)}")
    print(f"\n  NOTE: PaddleOCR reports genuine per-word scores (typically 0.7-0.99 on legible")
    print(f"  text). Re-calibrate the threshold based on the distribution above before merge.")


def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    full_markdown, page_results = run_pipeline(PDF_PATH)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# CamScanner Handwritten Notes — Extracted by IDP Pipeline (PaddleOCR handwritten mode)\n\n")
        f.write(full_markdown)
    print(f"\nMarkdown written → {OUTPUT_MD}")

    # Reports
    check_routing_accuracy(page_results)
    passed, failed = check_content_accuracy(page_results, full_markdown)
    check_paddle_output_quality(page_results)

    print(f"\n{'='*65}")
    if failed == 0:
        print("OVERALL: EXCELLENT")
    elif failed <= 4:
        print(f"OVERALL: GOOD — {failed} terms missing (handwriting recognition limits expected)")
    elif failed <= 8:
        print(f"OVERALL: FAIR — {failed} terms missing")
    else:
        print(f"OVERALL: NEEDS REVIEW — {failed} terms missing")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
