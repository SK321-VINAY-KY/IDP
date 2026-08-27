"""
File: handwritten.py
Purpose: Layer 2 "handwritten" route — thin delegation shim.
         TrOCR has been removed (2026-08-20). The engine is now PaddleOCR
         with a handwriting-tuned config (lower det_db_thresh, angle cls on).
         PaddleOCR's built-in DBNet detector handles line segmentation
         automatically, eliminating the whole-page collapse that caused
         ~0.34 avg confidence with TrOCR.

         All implementation lives in scanned.py alongside convert_scanned_page()
         so the two PaddleOCR engine instances share the same rollup helper
         and cache infrastructure without duplication.
Owner: engineer-a@idp-pilot
Updated: 2026-08-20 | Deps: paddleocr (via scanned.py)
"""

from src.ai.layer2_conversion.scanned import convert_handwritten_via_paddle

# Re-export under the original name so any direct callers outside pipeline.py
# continue to work without a change — though pipeline.py now calls
# convert_handwritten_via_paddle() directly.
__all__ = ["convert_handwritten_via_paddle"]
