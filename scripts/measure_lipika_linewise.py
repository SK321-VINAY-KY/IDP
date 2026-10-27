"""
File: scripts/measure_lipika_linewise.py
Purpose: DEVELOPER-ONLY one-off measurement (NOT a pytest test, NOT committed as
         a test). Reports:
           (1) fp32 model LOAD time (onnxruntime.InferenceSession construction).
           (2) EXTRA per-page latency of the line-wise path on 5 REAL dataset
               pages: render -> analysis gray -> ink mask -> line crops ->
               square windows -> batched forward -> mean-softmax aggregate.

Run:  .venv\\Scripts\\python.exe scripts/measure_lipika_linewise.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.ai.layer1_probe import imaging
from src.ai.layer1_probe.line_crops import extract_line_crops
from src.ai.layer1_probe.windows import letterbox_square, square_windows
from src.ai.layer1_probe.script_classifier import default_model_path, default_labels_path, _softmax

_DATASET = os.path.join(_REPO_ROOT, "dataset")
_RENDER_DPI = 144
_ANALYSIS_DPI = 100
_MAX_LINES = 24

# The five English PDFs supply Latin pages; verified_twice supplies scanned crops
# but we want full PAGES for the line-wise path, so we render dataset PDFs.
_LATIN_PDFS = [
    "MY_resume.pdf", "Vinay_Resume.pdf", "bert vs sas.pdf",
    "missouri_water_sample.pdf", "Chander Kochhar 01_compressed 2.pdf",
]


def _collect_pages(n: int = 5) -> List[np.ndarray]:
    """Return up to n real RGB page arrays rendered from dataset PDFs."""
    import fitz
    pages: List[np.ndarray] = []
    pdfs = [os.path.join(_DATASET, p) for p in _LATIN_PDFS]
    pdfs += sorted(glob.glob(os.path.join(_DATASET, "*.pdf")))
    seen = set()
    for p in pdfs:
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        try:
            doc = fitz.open(p)
        except Exception:
            continue
        try:
            for pn in range(min(2, doc.page_count)):
                pages.append(imaging.render_rgb(doc[pn], _RENDER_DPI))
                if len(pages) >= n:
                    return pages
        finally:
            doc.close()
    return pages


def _linewise_once(sess, input_name: str, rgb: np.ndarray) -> Tuple[int, str]:
    """Run the full line-wise path on one page; return (n_windows, script_idx)."""
    gray = imaging.analysis_gray(rgb, _RENDER_DPI, _ANALYSIS_DPI)
    mask = imaging.ink_mask(gray)
    boxes = extract_line_crops(gray, mask)
    boxes = sorted(boxes, key=lambda cb: cb[1][2] * cb[1][3], reverse=True)[:_MAX_LINES]
    gh, gw = gray.shape[:2]
    H, W = rgb.shape[:2]
    fx, fy = W / float(gw), H / float(gh)
    windows: List[np.ndarray] = []
    for _crop, (x, y, w, h) in boxes:
        x0 = max(0, int(round(x * fx)))
        y0 = max(0, int(round(y * fy)))
        x1 = min(W, int(round((x + w) * fx)))
        y1 = min(H, int(round((y + h) * fy)))
        if x1 <= x0 or y1 <= y0:
            continue
        windows.extend(square_windows(rgb[y0:y1, x0:x1].copy()))
    if not windows:
        return 0, "none"
    batch = np.concatenate([letterbox_square(w) for w in windows], axis=0)
    logits = np.asarray(sess.run(None, {input_name: batch})[0], dtype=np.float32)
    mean_probs = _softmax(logits).mean(axis=0)
    return len(windows), str(int(np.argmax(mean_probs)))


def main() -> int:
    import onnxruntime as ort
    model_path = default_model_path()
    if not os.path.exists(model_path):
        print(f"FATAL: fp32 model missing: {model_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("IDP_PROBE_ORT_THREADS", "2"))
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000.0
    input_name = sess.get_inputs()[0].name
    print(f"fp32 model load time: {load_ms:.1f} ms  ({os.path.basename(model_path)})")

    pages = _collect_pages(5)
    if not pages:
        print("No dataset pages found to measure.", file=sys.stderr)
        return 1

    # warm up (first run pays allocator/JIT costs that are not per-page latency).
    _linewise_once(sess, input_name, pages[0])

    per_page: List[float] = []
    print("\nper-page line-wise latency:")
    for i, rgb in enumerate(pages, 1):
        t = time.perf_counter()
        n_windows, idx = _linewise_once(sess, input_name, rgb)
        ms = (time.perf_counter() - t) * 1000.0
        per_page.append(ms)
        print(f"  page {i}: {ms:8.1f} ms  windows={n_windows:4d}  argmax_idx={idx}")

    mean_ms = sum(per_page) / len(per_page)
    print(f"\nmean per-page line-wise latency: {mean_ms:.1f} ms over {len(per_page)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
