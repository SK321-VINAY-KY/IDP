"""
File: line_segmentation.py
Purpose: Horizontal-projection-profile line segmentation for handwritten pages.
         Splits a full-page PIL Image into individual line-region crops that
         TrOCR (a line-level model) can actually process correctly.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: opencv-python, numpy, pillow
"""

import numpy as np
from PIL import Image
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Hard cap: more than this many detected regions on one page is almost certainly
# over-segmentation on noise — abort and fall back to whole-page.
_MAX_LINES_PER_PAGE = 60


def _deskew(
    gray: np.ndarray, binary: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate and correct skew angle using minAreaRect on the text mask.

    Returns corrected (gray, binary, angle_degrees). If the estimated angle
    is negligible (<0.5°) or the rotation fails for any reason, the originals
    are returned unchanged with angle=0.0 — deskew is best-effort.
    """
    coords = cv2_coords_from_binary(binary)
    if coords.shape[0] < 50:
        # Too few text pixels to get a reliable angle estimate
        return gray, binary, 0.0

    try:
        import cv2

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # minAreaRect returns angles in [-90, 0); map to [-45, 45)
        if angle < -45:
            angle += 90

        if abs(angle) < 0.5:
            return gray, binary, 0.0  # negligible — skip rotation

        h, w = gray.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_gray = cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotated_binary = cv2.warpAffine(
            binary,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        logger.info(
            "line_segmentation.deskew_applied",
            angle_deg=round(angle, 2),
            image_shape=(h, w),
        )
        return rotated_gray, rotated_binary, angle

    except Exception as exc:
        logger.warning(
            "line_segmentation.deskew_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return gray, binary, 0.0


def cv2_coords_from_binary(binary: np.ndarray) -> np.ndarray:
    """Return (N, 1, 2) int32 array of white-pixel coordinates for minAreaRect."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)
    coords = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.int32)
    return coords


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple uniform moving average — smooths single-pixel noise gaps."""
    if window <= 1:
        return arr.astype(float)
    kernel = np.ones(window) / window
    # 'same' mode keeps the output length equal to input
    return np.convolve(arr.astype(float), kernel, mode="same")


def segment_lines(
    image: Image.Image,
    min_line_height_px: int = 15,
    min_gap_px: int = 3,
    padding_px: int = 4,
    needs_deskew: bool = False,
) -> List[Image.Image]:
    """
    Split a full-page PIL Image into individual line-region crops using
    horizontal projection profiling. Returns crops in top-to-bottom order.

    Args:
        image:              Full-page PIL.Image (any mode; converted internally).
        min_line_height_px: Minimum row-span for a candidate to be kept as a
                            real line. Shorter regions are discarded as noise.
                            Should come from Settings.trocr_min_line_height_px.
        min_gap_px:         Minimum number of consecutive below-threshold rows
                            to be treated as a line boundary. Prevents splitting
                            within a single line at a momentary thin gap.
        padding_px:         Rows added above and below each crop. Ascenders and
                            descenders are clipped without this — do not set to 0.
        needs_deskew:       If True, attempt angle correction before profiling.
                            Pass True when PageClassification.needs_preprocessing
                            contains "deskew".

    Returns:
        List of PIL.Image crops, or an empty list if no lines are detected
        (caller should fall back to whole-page).
    """
    import cv2

    # --- 1. Grayscale ---
    gray = np.array(image.convert("L"))

    # --- 2. Binarize (Otsu, THRESH_BINARY_INV so text pixels = 255/white) ---
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 3. Optional deskew before profiling ---
    if needs_deskew:
        gray, binary, _ = _deskew(gray, binary)

    # --- 4. Horizontal projection profile (row-wise sum of white pixels) ---
    profile = binary.sum(axis=1).astype(float)  # shape: (height,)

    # --- 5. Smooth to absorb single-pixel noise within a line ---
    smoothed = _moving_average(profile, window=3)

    # Threshold: any row with at least one white pixel is "active".
    # Using a small epsilon rather than strict >0 to survive float smoothing.
    threshold = 0.5
    active = smoothed > threshold

    # --- 6. Find contiguous active regions with gap enforcement ---
    height = gray.shape[0]
    regions: list[tuple[int, int]] = []  # (row_start, row_end) inclusive
    in_line = False
    line_start = 0
    gap_count = 0

    for row in range(height):
        if active[row]:
            if not in_line:
                in_line = True
                line_start = row
            gap_count = 0
        else:
            if in_line:
                gap_count += 1
                if gap_count >= min_gap_px:
                    # Gap is wide enough — close the current line
                    line_end = row - gap_count  # last active row
                    regions.append((line_start, line_end))
                    in_line = False
                    gap_count = 0

    # Close any still-open line at the bottom of the page
    if in_line:
        regions.append((line_start, height - 1))

    # --- 7. Discard regions shorter than min_line_height_px ---
    regions = [(s, e) for s, e in regions if (e - s + 1) >= min_line_height_px]

    if not regions:
        logger.warning(
            "line_segmentation.no_lines_found",
            image_size=image.size,
            profile_max=float(profile.max()),
        )
        return []

    if len(regions) > _MAX_LINES_PER_PAGE:
        logger.warning(
            "line_segmentation.over_segmentation",
            detected=len(regions),
            cap=_MAX_LINES_PER_PAGE,
            image_size=image.size,
        )
        regions = regions[:_MAX_LINES_PER_PAGE]

    # --- 8. Crop with padding, clamp to image bounds ---
    img_h, img_w = gray.shape
    crops: List[Image.Image] = []

    # Work from the original (possibly deskewed) grayscale to build crops,
    # but return RGB crops since TrOCR's ViTImageProcessor expects colour input.
    source_rgb = Image.fromarray(gray).convert("RGB")

    for row_start, row_end in regions:
        top = max(0, row_start - padding_px)
        bottom = min(img_h, row_end + padding_px + 1)  # PIL crop end is exclusive
        crop = source_rgb.crop((0, top, img_w, bottom))
        crops.append(crop)

    logger.info(
        "line_segmentation.complete",
        image_size=image.size,
        lines_detected=len(crops),
        needs_deskew=needs_deskew,
    )
    return crops
