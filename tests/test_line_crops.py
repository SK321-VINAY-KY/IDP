"""Tests for line_crops.extract_line_crops. Synthetic numpy/cv2 pages, no PDF/fonts."""
import cv2
import numpy as np

from src.ai.layer1_probe.line_crops import extract_line_crops


def _blank_page(h=300, w=400):
    """White grayscale page (255 = paper)."""
    return np.full((h, w), 255, dtype=np.uint8)


def test_ruled_lines_only_yields_no_crops():
    """A page of only long thin horizontal rules must produce 0 line crops."""
    gray = _blank_page()
    for y in (60, 110, 160, 210, 260):
        cv2.line(gray, (20, y), (380, y), 0, 1)  # thin long horizontal rule
    crops = extract_line_crops(gray)
    assert len(crops) == 0


def test_single_text_line_yields_one_crop_in_band():
    """One text-height horizontal blob in the middle -> exactly 1 crop within the band."""
    gray = _blank_page()
    y0, y1, x0, x1 = 140, 160, 40, 360  # ~20px tall text band
    cv2.rectangle(gray, (x0, y0), (x1, y1), 0, thickness=-1)
    crops = extract_line_crops(gray)
    assert len(crops) == 1
    _, (bx, by, bw, bh) = crops[0]
    # The returned bbox must correspond to the drawn band. The crop includes a
    # pad margin; vertically only pad + ~1px of mask slack applies, while
    # horizontally the word-merging dilation (wide short kernel) also widens the
    # box by roughly half the kernel width, so x gets a larger slack.
    v_slack = 3 + 2
    h_slack = 3 + max(15, 400 // 30)  # pad + horizontal dilation kernel width
    assert by >= y0 - v_slack
    assert by + bh <= y1 + v_slack
    assert bx >= x0 - h_slack
    assert bx + bw <= x1 + h_slack


def test_blank_page_yields_no_crops():
    """A blank white page must produce 0 crops."""
    crops = extract_line_crops(_blank_page())
    assert len(crops) == 0
