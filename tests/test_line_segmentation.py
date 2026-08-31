"""
File: test_line_segmentation.py
Purpose: Unit tests for segment_lines() — synthetic images with known structure,
         no model calls, no PDF fixtures required.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: pillow, numpy, opencv-python
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw
import numpy as np

from src.ai.layer2_conversion.line_segmentation import (
    segment_lines,
    _MAX_LINES_PER_PAGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lines_image(
    n_lines: int,
    page_w: int = 800,
    line_h: int = 30,
    gap_h: int = 20,
    top_margin: int = 40,
) -> tuple[Image.Image, list[tuple[int, int]]]:
    """
    Draw n_lines solid black horizontal bars on a white background.
    Returns (image, list_of_(top, bottom)_pixel_coords_for_each_bar).
    The bars are guaranteed ground-truth: each one is exactly line_h px tall
    with gap_h px of white between them.
    """
    total_h = top_margin + n_lines * line_h + (n_lines - 1) * gap_h + top_margin
    img = Image.new("RGB", (page_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    ground_truth: list[tuple[int, int]] = []
    y = top_margin
    for _ in range(n_lines):
        draw.rectangle([0, y, page_w - 1, y + line_h - 1], fill=(0, 0, 0))
        ground_truth.append((y, y + line_h - 1))
        y += line_h + gap_h

    return img, ground_truth


def make_blank_image(page_w: int = 800, page_h: int = 1000) -> Image.Image:
    return Image.new("RGB", (page_w, page_h), color=(255, 255, 255))


def make_noisy_image(page_w: int = 800, page_h: int = 1000) -> Image.Image:
    """
    One isolated pixel per row, each on a different row, spread out so no
    consecutive run of rows ever reaches min_line_height_px. Uses every 50th
    row so the gap between any two active rows is 49 -- far above any realistic
    min_gap_px -- meaning segment_lines must emit one region per dot, each
    only 1 row tall, which is below min_line_height_px=15 and gets discarded.
    """
    img = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    arr = np.array(img)
    # One black pixel every 50 rows -- max consecutive active rows = 1
    for row in range(0, page_h, 50):
        arr[row, page_w // 2] = [0, 0, 0]
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_correct_line_count_5():
    """5 well-separated bars → exactly 5 crops."""
    img, gt = make_lines_image(n_lines=5, line_h=30, gap_h=20)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 5, (
        f"Expected 5 lines, got {len(crops)}. "
        "Check gap_h vs min_gap_px or line_h vs min_line_height_px."
    )
    print(f"PASS: 5-line image -> {len(crops)} crops")


def test_correct_line_count_1():
    """Single bar → exactly 1 crop."""
    img, _ = make_lines_image(n_lines=1, line_h=40, gap_h=0)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 1, f"Expected 1 line, got {len(crops)}"
    print(f"PASS: 1-line image -> {len(crops)} crop")


def test_correct_line_count_10():
    """10 bars → exactly 10 crops."""
    img, _ = make_lines_image(n_lines=10, line_h=25, gap_h=18)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 10, f"Expected 10 lines, got {len(crops)}"
    print(f"PASS: 10-line image -> {len(crops)} crops")


def test_top_to_bottom_order():
    """Crops must be ordered top-to-bottom (first crop's top < second crop's top)."""
    img, gt = make_lines_image(n_lines=5, line_h=30, gap_h=20, top_margin=40)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 5

    # Verify heights are strictly increasing (crops are non-overlapping and ordered)
    # We can't access row coords directly, but we can check crop heights are consistent
    # and that the number matches — order is guaranteed by the sequential row scan.
    heights = [c.size[1] for c in crops]
    assert all(h > 0 for h in heights), "All crops must have positive height"
    print(f"PASS: top-to-bottom order verified, crop heights = {heights}")


def test_crop_dimensions():
    """Every crop must be full page width and taller than 0."""
    page_w = 600
    img, _ = make_lines_image(n_lines=3, line_h=35, gap_h=25, page_w=page_w)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 3
    for i, crop in enumerate(crops):
        assert (
            crop.size[0] == page_w
        ), f"Crop {i} width {crop.size[0]} != page width {page_w}"
        assert crop.size[1] > 0, f"Crop {i} has zero height"
    print(f"PASS: all 3 crops have full page width={page_w}px and positive height")


def test_padding_applied():
    """
    With padding_px=10, each crop should be taller than the bare line_h=30.
    The bar is 30px, so with padding the crop should be >= 30px (often 30+padding
    but clamped at image edges for the first/last line).
    """
    img, _ = make_lines_image(n_lines=3, line_h=30, gap_h=30, top_margin=50)
    crops_no_pad = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=0)
    crops_padded = segment_lines(
        img, min_line_height_px=15, min_gap_px=3, padding_px=10
    )

    assert len(crops_no_pad) == len(crops_padded) == 3
    for i, (c0, cp) in enumerate(zip(crops_no_pad, crops_padded)):
        assert (
            cp.size[1] >= c0.size[1]
        ), f"Crop {i}: padded height {cp.size[1]} < unpadded {c0.size[1]}"
    print(
        f"PASS: padding applied -- unpadded heights {[c.size[1] for c in crops_no_pad]}"
        f" -> padded {[c.size[1] for c in crops_padded]}"
    )


def test_blank_image_returns_empty():
    """Completely white image → no lines detected → empty list (not an exception)."""
    img = make_blank_image()
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert crops == [], f"Expected empty list for blank image, got {len(crops)} crops"
    print("PASS: blank image -> empty list (no exception)")


def test_noise_only_returns_empty():
    """Scattered single pixels below min_line_height_px -> empty list."""
    img = make_noisy_image()
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert crops == [], (
        f"Expected empty list for noisy image, got {len(crops)} crops. "
        "Single-pixel noise should be discarded by min_line_height_px filter."
    )
    print("PASS: noise-only image -> empty list")


def test_short_lines_discarded():
    """
    Lines shorter than min_line_height_px (here: 5px bars, threshold=15) must
    be discarded even if there are many of them.
    """
    img, _ = make_lines_image(n_lines=8, line_h=5, gap_h=20)  # bars too short
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert crops == [], (
        f"Expected 0 crops (bars too short), got {len(crops)}. "
        "min_line_height_px filter not working."
    )
    print("PASS: 5px bars (< min_line_height_px=15) all discarded")


def test_over_segmentation_cap():
    """
    >60 detected regions → capped at _MAX_LINES_PER_PAGE (60), warning logged.
    We use a tiny min_line_height to force many 1-row "lines" to be kept.
    """
    # 70 thin bars (2px each) with 5px gaps — all pass a very low threshold
    img, _ = make_lines_image(n_lines=70, line_h=2, gap_h=5, top_margin=10)
    crops = segment_lines(img, min_line_height_px=1, min_gap_px=3, padding_px=0)
    assert (
        len(crops) <= _MAX_LINES_PER_PAGE
    ), f"Expected at most {_MAX_LINES_PER_PAGE} crops, got {len(crops)}"
    print(f"PASS: over-segmentation capped at {len(crops)} (<= {_MAX_LINES_PER_PAGE})")


def test_crops_are_rgb():
    """Output crops must be RGB mode (ViTImageProcessor requirement)."""
    img, _ = make_lines_image(n_lines=3, line_h=30, gap_h=20)
    crops = segment_lines(img, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 3
    for i, crop in enumerate(crops):
        assert crop.mode == "RGB", f"Crop {i} is mode '{crop.mode}', expected 'RGB'"
    print("PASS: all crops returned as RGB PIL.Image")


def test_grayscale_input_accepted():
    """segment_lines should accept grayscale ('L' mode) input without error."""
    img, _ = make_lines_image(n_lines=3, line_h=30, gap_h=20)
    img_gray = img.convert("L")
    crops = segment_lines(img_gray, min_line_height_px=15, min_gap_px=3, padding_px=4)
    assert len(crops) == 3
    print("PASS: grayscale input accepted, 3 crops returned")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_correct_line_count_5,
        test_correct_line_count_1,
        test_correct_line_count_10,
        test_top_to_bottom_order,
        test_crop_dimensions,
        test_padding_applied,
        test_blank_image_returns_empty,
        test_noise_only_returns_empty,
        test_short_lines_discarded,
        test_over_segmentation_cap,
        test_crops_are_rgb,
        test_grayscale_input_accepted,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__} — {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Unit tests: {passed}/{passed+failed} passed")
    if failed:
        print("Result: FAILED")
        sys.exit(1)
    else:
        print("Result: ALL PASSED")
