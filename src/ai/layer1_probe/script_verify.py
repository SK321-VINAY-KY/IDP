"""
File: script_verify.py
Purpose: Post-OCR verification. Count Unicode scripts in the OCR OUTPUT and compare
         with the script the page was routed under. Used by engines / the pipeline
         to trigger ONE re-route (different lang or VLM) when OCR garbled the page.
Deps: stdlib
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, NamedTuple, Optional

_RANGES = {
    "latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)],
    "devanagari": [(0x0900, 0x097F)], "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)], "gujarati": [(0x0A80, 0x0AFF)],
    "odia": [(0x0B00, 0x0B7F)], "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)], "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)], "thai": [(0x0E00, 0x0E7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)], "hebrew": [(0x0590, 0x05FF)],
    "cyrillic": [(0x0400, 0x04FF)], "greek": [(0x0370, 0x03FF)],
    "chinese": [(0x4E00, 0x9FFF)], "japanese": [(0x3040, 0x30FF)],
    "korean": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
}
# Han characters appear in Japanese text too; kana presence decides.
_SHARED_OK = {"chinese": {"japanese"}, "japanese": {"chinese"}}


class VerifyResult(NamedTuple):
    ok: bool
    observed_script: str
    observed_share: float
    foreign_ratio: float
    reason: str


def script_histogram(text: str) -> Dict[str, int]:
    c: Counter = Counter()
    for ch in text:
        o = ord(ch)
        for name, rs in _RANGES.items():
            if any(lo <= o <= hi for lo, hi in rs):
                c[name] += 1
                break
    return dict(c)


def verify_ocr_output(
    text: str, expected_script: str, confidence: Optional[float] = None,
    min_chars: int = 15, min_expected_share: float = 0.6, low_conf: float = 0.6,
) -> VerifyResult:
    """
    ok=False when the output's dominant script contradicts `expected_script`, or when
    confidence is low AND a large share of letters belong to other scripts (the
    signature of a recognizer forced onto the wrong language).
    Too little text -> ok=True (cannot judge; let the confidence gate decide).
    """
    hist = script_histogram(text)
    total = sum(hist.values())
    if total < min_chars:
        return VerifyResult(True, "unknown", 0.0, 0.0, "too_little_text")
    observed = max(hist, key=hist.__getitem__)
    share = hist[observed] / total
    exp_share = hist.get(expected_script, 0) / total
    foreign = 1.0 - exp_share - sum(hist.get(s, 0) for s in _SHARED_OK.get(expected_script, ())) / total
    foreign = max(0.0, foreign)

    # Latin digits/acronyms legitimately appear in every script -> only flag when
    # the expected script is clearly NOT the dominant one.
    if observed != expected_script and observed not in _SHARED_OK.get(expected_script, ()) \
            and exp_share < (1 - min_expected_share):
        return VerifyResult(False, observed, round(share, 3), round(foreign, 3),
                            f"dominant_script_{observed}_not_{expected_script}")
    if confidence is not None and confidence < low_conf and foreign > 0.5:
        return VerifyResult(False, observed, round(share, 3), round(foreign, 3),
                            "low_confidence_with_foreign_script_mix")
    return VerifyResult(True, observed, round(share, 3), round(foreign, 3), "ok")
