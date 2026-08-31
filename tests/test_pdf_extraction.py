"""
File: test_pdf_extraction.py
Purpose: End-to-end extraction test using the SDG 17 Goals PDF.
         Downloads/saves the PDF, runs it through the Layer 1 router and
         a lightweight text extractor (no OCR models needed for a clean
         digital PDF), writes output.md, then scores accuracy.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pymupdf  # PyMuPDF
from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import route_from_profile, is_mixed_content

PDF_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sdg_goals.pdf")
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "fixtures", "sdg_goals_output.md")


def extract_page_to_markdown(page, page_number: int, route: str) -> str:
    """
    Lightweight markdown extractor for digital pages.
    Preserves heading structure by detecting font-size differences.
    """
    if route == "skip":
        return f"<!-- Page {page_number}: skipped (blank) -->\n"

    blocks = page.get_text("dict")["blocks"]
    lines_out = []

    for block in blocks:
        if block["type"] != 0:  # 0 = text block
            continue
        for line in block["lines"]:
            line_text = ""
            max_size = 0
            for span in line["spans"]:
                line_text += span["text"]
                if span["size"] > max_size:
                    max_size = span["size"]

            line_text = line_text.strip()
            if not line_text:
                continue

            # Rough heading detection by font size
            if max_size >= 14:
                lines_out.append(f"## {line_text}")
            elif max_size >= 11 and line_text.startswith("Goal"):
                lines_out.append(f"### {line_text}")
            elif line_text.startswith("Target"):
                lines_out.append(f"- **{line_text}**")
            else:
                lines_out.append(line_text)

    return "\n".join(lines_out) + "\n\n"


def run_extraction(pdf_path: str) -> tuple[str, list[dict]]:
    """Run the full extraction pipeline on a PDF file."""
    doc = pymupdf.open(pdf_path)
    page_results = []
    full_markdown = ""

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    print(f"Total pages: {len(doc)}")
    print(f"{'='*60}\n")

    for i, page in enumerate(doc):
        page_number = i + 1
        profile = inspect_page(page, page_number)
        route = route_from_profile(profile)

        # For this clean digital PDF, None means mixed-content -> still extract as digital
        if route is None:
            route = "digital"

        md_content = extract_page_to_markdown(page, page_number, route)
        full_markdown += f"<!-- Page {page_number} | route={route} | chars={profile.char_count} -->\n"
        full_markdown += md_content

        page_results.append({
            "page": page_number,
            "route": route,
            "char_count": profile.char_count,
            "image_coverage": profile.image_coverage,
            "is_scanned": profile.is_scanned,
            "has_text": profile.has_text,
            "chars_extracted": len(md_content.strip()),
        })

        status = "OK" if profile.char_count > 50 else "LOW"
        print(f"  Page {page_number:2d}: route={route:8s} | chars={profile.char_count:4d} | "
              f"img_cov={profile.image_coverage:.2f} | [{status}]")

    doc.close()
    return full_markdown, page_results


def score_accuracy(page_results: list[dict], full_markdown: str) -> None:
    """
    Accuracy check against known content from the SDG document.
    Verifies that key terms from each goal are present in the output.
    """
    print(f"\n{'='*60}")
    print("ACCURACY CHECK")
    print(f"{'='*60}")

    # Known ground-truth strings that must appear in a correct extraction
    required_terms = [
        ("Goal 1", "End poverty in all its forms"),
        ("Goal 2", "End hunger"),
        ("Goal 3", "healthy lives"),
        ("Goal 4", "quality education"),
        ("Goal 5", "gender equality"),
        ("Goal 6", "water and sanitation"),
        ("Goal 7", "energy"),
        ("Goal 8", "economic growth"),
        ("Goal 9", "infrastructure"),
        ("Goal 10", "inequality"),
        ("Goal 11", "cities"),
        ("Goal 12", "consumption"),
        ("Goal 13", "climate change"),
        ("Goal 14", "oceans"),
        ("Goal 15", "terrestrial ecosystems"),
        ("Goal 16", "peaceful"),
        ("Goal 17", "partnership"),
        ("Target 1.1", "extreme poverty"),
        ("Target 3.1", "maternal mortality"),
        ("Target 13.1", "climate-related hazards"),
        ("1.25", "$1.25"),  # specific data point
    ]

    md_lower = full_markdown.lower()
    passed = 0
    failed = 0

    for goal_label, term in required_terms:
        found = term.lower() in md_lower
        status = "PASS" if found else "FAIL"
        if found:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {goal_label}: '{term}'")

    print(f"\nRouting accuracy:")
    total_pages = len(page_results)
    digital_pages = sum(1 for p in page_results if p["route"] == "digital")
    skip_pages = sum(1 for p in page_results if p["route"] == "skip")
    pages_with_text = sum(1 for p in page_results if p["char_count"] > 50)

    print(f"  Total pages     : {total_pages}")
    print(f"  Routed digital  : {digital_pages}")
    print(f"  Routed skip     : {skip_pages}")
    print(f"  Pages with text : {pages_with_text}")

    print(f"\nContent accuracy : {passed}/{len(required_terms)} terms found "
          f"({100*passed//len(required_terms)}%)")

    total_chars = sum(p["char_count"] for p in page_results)
    extracted_chars = sum(p["chars_extracted"] for p in page_results)
    print(f"Chars in PDF     : {total_chars}")
    print(f"Chars in .md     : {extracted_chars}")

    if failed == 0:
        print("\nResult: EXCELLENT — all expected terms extracted correctly")
    elif failed <= 2:
        print(f"\nResult: GOOD — {failed} term(s) missing (minor formatting variation likely)")
    else:
        print(f"\nResult: NEEDS REVIEW — {failed} terms missing")


def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Please save the SDG PDF as: tests/fixtures/sdg_goals.pdf")
        sys.exit(1)

    full_markdown, page_results = run_extraction(PDF_PATH)

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# SDG 17 Goals — Extracted by IDP Pipeline\n\n")
        f.write(full_markdown)

    print(f"\nMarkdown written to: {OUTPUT_MD}")
    score_accuracy(page_results, full_markdown)


if __name__ == "__main__":
    main()
