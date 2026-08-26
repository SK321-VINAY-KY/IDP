"""
Script: show_capabilities.py
Purpose: Print per-page capability detection and engine plan for any PDF.
         Shows the difference between legacy single-engine routing and the
         capability-based multi-engine plan for every page.
Usage:
    py -3.11 scripts/show_capabilities.py [path/to/file.pdf]
    Defaults to tests/fixtures/sdg_goals.pdf if no argument given.
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: pymupdf, pydantic-settings
"""
import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pymupdf

from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import (
    capabilities_from_profile,
    build_engine_plan,
    route_from_profile,
    is_mixed_content,
)
from src.config.settings import settings

logging.disable(logging.CRITICAL)

PDF = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "sdg_goals.pdf")
)

if not os.path.exists(PDF):
    print(f"ERROR: PDF not found: {PDF}")
    sys.exit(1)

doc = pymupdf.open(PDF)

print()
print("=" * 100)
print(f"  Document      : {os.path.basename(PDF)}  ({len(doc)} pages)")
print(f"  routing_mode  : {settings.routing_mode}")
print(f"  mixed_content_min_image_coverage : {settings.mixed_content_min_image_coverage}")
print(f"  dual_ocr_scan_threshold          : {settings.capability_dual_ocr_scan_threshold}")
print("=" * 100)
print()
print(f"  {'Pg':>2}  {'chars':>6}  {'img_cov':>7}  {'scanned':>7}  {'mixed':>5}  "
      f"{'legacy_route':<15}  {'capabilities':<44}  engine_plan")
print("  " + "-" * 120)

plan_summary: dict[str, int] = {}

for i, page in enumerate(doc):
    pn = i + 1
    profile = inspect_page(page, pn)

    legacy = route_from_profile(profile) or "→ Step B"
    mixed  = is_mixed_content(profile)

    caps    = capabilities_from_profile(profile)
    plan    = build_engine_plan(caps)
    active  = caps.active_capabilities()
    engines = [t.engine for t in plan]

    # tally plan types for summary
    key = "+".join(engines) if engines else "empty"
    plan_summary[key] = plan_summary.get(key, 0) + 1

    print(f"  {pn:>2}  {profile.char_count:>6}  {profile.image_coverage:>7.3f}"
          f"  {str(profile.is_scanned):>7}  {str(mixed):>5}  {legacy:<15}  "
          f"{str(active):<44}  {engines}")

doc.close()

# Summary
print()
print("=" * 100)
print("  ENGINE PLAN SUMMARY")
print("  " + "-" * 50)
for plan_type, count in sorted(plan_summary.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"  {plan_type:<55}  {bar}  ({count} pages)")
print("=" * 100)
print()
