"""
File: scripts/measure_lipika_cap.py
Purpose: DEVELOPER-ONLY, OFFLINE, LOCAL measurement sweep for the lipika line-wise
         observer window cap (IDP_PROBE_LIPIKA_MAX_WINDOWS). One-off measurement
         tool — NOT a committed test, NOT imported at runtime, NOT part of CI.

What it measures (original request items 3a/3b/3c):
  3a. Per-page OBSERVER-ONLY latency on 5 real dataset pages for caps {16,32,64}:
      seconds/sampled page (the observe() call = windows + even-spread cap +
      batched forward + aggregate, EXCLUDING PDF render), windows actually used,
      and page-level predicted script vs the known script.
  3b. Full labelled page-level eval, split BY SOURCE/PAGE (page-level vote =
      argmax of mean softmax over that page's windows), for caps {16,32,64} on
      the full labelled set the harness builds (dataset/verified_twice -> scripts,
      English PDFs -> latin). Reuses scripts/eval_lipika_crops.py machinery with a
      cap applied via the SAME even-spread selector the observer uses
      (script_classifier._even_spread). Reports per-script page accuracy, the
      confusion matrix (same collapsed-'other' columns as the prior eval report),
      and the count/rate of NON-LATIN pages predicted as LATIN (the harmful
      misroute direction, since only latin -> Paddle).
  3c. If NO cap reaches ~3 s/sampled page (observer-only) with page accuracy
      within ~2 points of the uncapped Strategy-B result, additionally measure
      int8 at that cap (load lipika_script.int8.onnx, same cap).

Loads the real fp32 model ONCE and reuses the session. NEVER invents numbers:
any cell that cannot be produced is marked 'not measured' with a reason.

Writes a machine-readable JSON blob to stdout (consumed by the reporting step)
and prints a human summary. Does NOT change routing, defaults, flags, or the
shipped fp32 default model.

Run (repo venv):
  .venv\\Scripts\\python.exe scripts/measure_lipika_cap.py --caps 16,32,64 \
      --per-script-cap 40 --json-out .agents/tasks/_lipika_cap_measure.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from src.ai.layer1_probe import imaging  # noqa: E402
from src.ai.layer1_probe.script_classifier import _even_spread, _softmax  # noqa: E402
from src.ai.layer1_probe.types import ProbeConfig  # noqa: E402
from src.ai.layer1_probe.windows import letterbox_square, square_windows  # noqa: E402

# Reuse the eval harness manifest + crop production + label machinery.
from scripts.eval_lipika_crops import (  # noqa: E402
    LATIN_PDFS,
    _DATASET_DIR,
    _FP32_PATH,
    _INT8_PATH,
    _LABELS_PATH,
    _RENDER_DPI,
    balance_manifest,
    build_auto_manifest,
    collapse_label,
    crops_from_rgb_page,
    load_labels,
    produce_crops,
)

# Observer's per-page line cap (largest first). Mirrors script_classifier._MAX_LINES.
_MAX_LINES = 24


# ---------------------------------------------------------------------------
# Session (loaded ONCE, reused)
# ---------------------------------------------------------------------------
def load_session(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("IDP_PROBE_ORT_THREADS", "2"))
    t0 = time.perf_counter()
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    return sess, sess.get_inputs()[0].name, load_s


# ---------------------------------------------------------------------------
# Observer replica (fp32, Strategy B + even-spread cap), session reused.
# This mirrors LipikaOnnxScriptObserver.observe EXACTLY for the windowing,
# even-spread cap, batched forward, and mean-softmax aggregation, but takes a
# shared session + explicit cap so we can time observe-only cost per cap without
# reloading the model. No gates are applied here (we want the raw prediction and
# window count regardless of gate thresholds) — the gates are a routing concern,
# not a measurement concern.
# ---------------------------------------------------------------------------
from src.ai.layer1_probe.line_crops import extract_line_crops  # noqa: E402
from src.ai.layer1_probe.script_classifier import _scale_box  # noqa: E402


def _line_sources(gray: np.ndarray, mask: np.ndarray,
                  rgb: Optional[np.ndarray]) -> List[np.ndarray]:
    boxes = extract_line_crops(gray, mask)
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda cb: cb[1][2] * cb[1][3], reverse=True)[:_MAX_LINES]
    sources: List[np.ndarray] = []
    if rgb is not None:
        gh, gw = gray.shape[:2]
        H, W = rgb.shape[:2]
        fx = W / float(gw)
        fy = H / float(gh)
        for _crop_gray, box in boxes:
            x0, y0, bw, bh = _scale_box(box, fx, fy, W, H)
            if bw <= 0 or bh <= 0:
                continue
            sources.append(rgb[y0:y0 + bh, x0:x0 + bw].copy())
    else:
        for crop_gray, _box in boxes:
            sources.append(cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB))
    return sources


def observe_windows_for_rgb(rgb: np.ndarray, cap: int) -> List[np.ndarray]:
    """Build the exact even-spread capped window list the observer would forward
    for a full-res RGB page. Mirrors observe() windowing + _even_spread."""
    gray = imaging.analysis_gray(rgb, _RENDER_DPI, 100)
    mask = imaging.ink_mask(gray)
    line_sources = _line_sources(gray, mask, rgb)
    per_line_windows = [square_windows(src) for src in line_sources]
    return _even_spread(per_line_windows, cap)


def forward_mean_softmax(sess, input_name: str,
                         windows: List[np.ndarray]) -> Optional[np.ndarray]:
    if not windows:
        return None
    batch = np.concatenate([letterbox_square(w) for w in windows], axis=0)
    logits = np.asarray(sess.run(None, {input_name: batch})[0], dtype=np.float32)
    probs = _softmax(logits)
    return probs.mean(axis=0)


# ---------------------------------------------------------------------------
# 3a — per-page observer latency on 5 real pages
# ---------------------------------------------------------------------------
def pick_5_pages() -> List[Tuple[str, str]]:
    """The 5 representative real pages: page 0 of each of the 5 English/Latin
    PDFs in dataset/. These are the Latin PDFs the prior uncapped Strategy-B
    baseline (~403 windows/page) was measured on. Returns [(label, pdf_path)]."""
    out: List[Tuple[str, str]] = []
    for pdf in LATIN_PDFS:
        p = os.path.join(_DATASET_DIR, pdf)
        if os.path.exists(p):
            out.append((pdf, p))
    return out


def render_pdf_page0_rgb(path: str) -> Optional[np.ndarray]:
    import fitz
    try:
        doc = fitz.open(path)
    except Exception:
        return None
    try:
        if doc.page_count == 0:
            return None
        return imaging.render_rgb(doc[0], _RENDER_DPI)
    finally:
        doc.close()


def measure_3a(sess, input_name: str, raw_labels: List[str],
               caps: List[int], repeats: int, pages_limit: int = 0) -> List[dict]:
    pages = pick_5_pages()
    if pages_limit and pages_limit > 0:
        pages = pages[:pages_limit]
    rows: List[dict] = []
    # Pre-render each page once (render excluded from the observer timing).
    rendered: List[Tuple[str, np.ndarray]] = []
    for label, path in pages:
        rgb = render_pdf_page0_rgb(path)
        if rgb is not None:
            rendered.append((label, rgb))
    for cap in caps:
        for label, rgb in rendered:
            # Time observe-only: windowing + cap + forward + aggregate.
            best = None
            n_windows = 0
            pred = "not measured"
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                windows = observe_windows_for_rgb(rgb, cap)
                mean_probs = forward_mean_softmax(sess, input_name, windows)
                dt = time.perf_counter() - t0
                n_windows = len(windows)
                if mean_probs is not None:
                    pred = collapse_label(raw_labels, int(np.argmax(mean_probs)))
                best = dt if best is None else min(best, dt)
            rows.append({
                "cap": cap,
                "page": label,
                "true_script": "latin",
                "seconds": round(best, 3) if best is not None else None,
                "n_windows": n_windows,
                "pred_script": pred,
                "correct": pred == "latin",
            })
    return rows


# ---------------------------------------------------------------------------
# 3b — full labelled page-level eval per cap (reuses eval harness manifest)
# ---------------------------------------------------------------------------
class PageEval:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.index = {n: i for i, n in enumerate(vocab)}
        k = len(vocab)
        self.confusion = np.zeros((k, k), dtype=np.int64)
        self.total: Dict[str, int] = {}
        self.correct: Dict[str, int] = {}
        self.nonlatin_pages = 0
        self.nonlatin_as_latin = 0

    def add(self, true_script: str, pred: str):
        self.total[true_script] = self.total.get(true_script, 0) + 1
        if pred == true_script:
            self.correct[true_script] = self.correct.get(true_script, 0) + 1
        ti = self.index.get(true_script)
        pi = self.index.get(pred)
        if ti is not None and pi is not None:
            self.confusion[ti, pi] += 1
        if true_script != "latin":
            self.nonlatin_pages += 1
            if pred == "latin":
                self.nonlatin_as_latin += 1

    def per_script(self) -> Dict[str, Tuple[int, int, float]]:
        out: Dict[str, Tuple[int, int, float]] = {}
        for s, t in sorted(self.total.items()):
            c = self.correct.get(s, 0)
            out[s] = (c, t, (c / t) if t else 0.0)
        return out


def _page_windows_from_crops(crops: List[np.ndarray], cap: int) -> List[np.ndarray]:
    """For a source (page) that yields multiple crops, treat EACH crop as a line
    source, build Strategy-B square windows per crop, then apply the SAME
    even-spread cap across crops. For a single tight word crop (verified_twice
    jpeg) this is one line -> its windows capped. For a PDF page it is many line
    crops -> windows spread across lines. Mirrors the observer aggregation unit
    (page-level mean softmax over the capped window set)."""
    per_line_windows = [square_windows(c) for c in crops]
    return _even_spread(per_line_windows, cap)


def measure_3b(sess, input_name: str, raw_labels: List[str], vocab: List[str],
               caps: List[int], per_script_cap: int, max_pages: int,
               max_crops_per_source: int, seed: int) -> Dict[int, PageEval]:
    rows, skipped = build_auto_manifest()
    sampled, counts, low = balance_manifest(rows, 0, per_script_cap, seed=seed)

    evals: Dict[int, PageEval] = {cap: PageEval(vocab) for cap in caps}

    # Produce crops ONCE per source; reuse across caps. For verified_twice jpegs
    # each source is one crop (one "page"); for PDFs each PAGE is its own source
    # unit here — we split the PDF into per-page crop groups so the eval is truly
    # "split by document/page".
    import fitz
    n_sources = len(sampled)
    for si, row in enumerate(sampled):
        if si % 25 == 0:
            print(f"[3b] source {si}/{n_sources} ({row.true_script})...", file=sys.stderr)
        page_crop_groups: List[List[np.ndarray]] = []
        if row.kind == "crop":
            got = produce_crops(row, max_pages=max_pages)
            if got:
                page_crop_groups.append(got[:max_crops_per_source] if max_crops_per_source > 0 else got)
        elif row.kind == "pdf":
            try:
                doc = fitz.open(row.source_path)
            except Exception:
                doc = None
            if doc is not None:
                try:
                    n = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
                    for pn in range(n):
                        rgb = imaging.render_rgb(doc[pn], _RENDER_DPI)
                        pc = crops_from_rgb_page(rgb)
                        if pc:
                            if max_crops_per_source > 0:
                                pc = pc[:max_crops_per_source]
                            page_crop_groups.append(pc)
                finally:
                    doc.close()
        else:
            got = produce_crops(row, max_pages=max_pages)
            if got:
                page_crop_groups.append(got[:max_crops_per_source] if max_crops_per_source > 0 else got)

        for crops in page_crop_groups:
            if not crops:
                continue
            for cap in caps:
                windows = _page_windows_from_crops(crops, cap)
                mean_probs = forward_mean_softmax(sess, input_name, windows)
                if mean_probs is None:
                    continue
                pred = collapse_label(raw_labels, int(np.argmax(mean_probs)))
                evals[cap].add(row.true_script, pred)

    return evals


# ---------------------------------------------------------------------------
# Confusion-matrix formatting (same collapsed-'other' column style as prior)
# ---------------------------------------------------------------------------
def fmt_confusion(ev: PageEval) -> List[str]:
    vocab = ev.vocab
    present = [i for i, _ in enumerate(vocab)
               if ev.confusion[i].sum() > 0 or ev.confusion[:, i].sum() > 0]
    if not present:
        return ["(no data)"]
    short = [vocab[i][:4] for i in present]
    header = "true\\pred | " + " | ".join(f"{s:>5}" for s in short)
    lines = [header, "-" * len(header)]
    for i in present:
        cells = " | ".join(f"{ev.confusion[i, j]:>5}" for j in present)
        lines.append(f"{vocab[i][:8]:>8} | {cells}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Measure lipika window-cap latency + page accuracy.")
    ap.add_argument("--caps", default="16,32,64", help="comma list of caps to sweep")
    ap.add_argument("--per-script-cap", type=int, default=40,
                    help="jpeg sources per script for the 3b sweep (runtime control)")
    ap.add_argument("--max-pages", type=int, default=4, help="max PDF pages per PDF source")
    ap.add_argument("--max-crops-per-source", type=int, default=40,
                    help="cap crops per source/page (<=0 = all)")
    ap.add_argument("--repeats", type=int, default=3, help="3a timing repeats (min kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int8-cap", type=int, default=0,
                    help="if >0, also run 3a+3b with int8 at this cap (3c)")
    ap.add_argument("--skip-3b", action="store_true",
                    help="skip the full page-level sweep (3a only)")
    ap.add_argument("--pages-3a", type=int, default=0,
                    help="limit 3a to the first N of the 5 pages (0 = all 5)")
    ap.add_argument("--json-out", default=None, help="write machine-readable JSON here")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    raw_labels, vocab = load_labels(_LABELS_PATH)

    result: dict = {"fp32": {}, "int8": {}, "notes": []}

    if not os.path.exists(_FP32_PATH):
        result["notes"].append(f"fp32 model missing: {_FP32_PATH}")
        print(json.dumps(result))
        return 2

    sess, input_name, load_s = load_session(_FP32_PATH)
    result["fp32"]["model_load_seconds"] = round(load_s, 3)
    print(f"[fp32] model load = {load_s:.3f}s", file=sys.stderr)

    # 3a
    print("[3a] measuring per-page observer latency (fp32)...", file=sys.stderr)
    rows_3a = measure_3a(sess, input_name, raw_labels, caps, args.repeats,
                         pages_limit=args.pages_3a)
    result["fp32"]["per_page_latency"] = rows_3a

    if args.skip_3b:
        out = json.dumps(result, indent=2)
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"\nJSON written to {args.json_out}", file=sys.stderr)
        print(out)
        return 0

    # 3b
    print("[3b] full labelled page-level sweep (fp32)...", file=sys.stderr)
    evals = measure_3b(sess, input_name, raw_labels, vocab, caps,
                       args.per_script_cap, args.max_pages,
                       args.max_crops_per_source, args.seed)
    result["fp32"]["page_eval"] = {}
    for cap in caps:
        ev = evals[cap]
        ps = ev.per_script()
        overall_c = sum(c for c, _, _ in ps.values())
        overall_t = sum(t for _, t, _ in ps.values())
        result["fp32"]["page_eval"][str(cap)] = {
            "per_script": {s: {"correct": c, "total": t, "acc": round(a, 4)}
                           for s, (c, t, a) in ps.items()},
            "overall_acc": round(overall_c / overall_t, 4) if overall_t else None,
            "overall_correct": overall_c,
            "overall_total": overall_t,
            "confusion_lines": fmt_confusion(ev),
            "nonlatin_pages": ev.nonlatin_pages,
            "nonlatin_as_latin": ev.nonlatin_as_latin,
            "nonlatin_as_latin_rate": round(ev.nonlatin_as_latin / ev.nonlatin_pages, 4)
                if ev.nonlatin_pages else None,
        }
        print(f"[3b] cap={cap} overall page acc="
              f"{(overall_c / overall_t) if overall_t else 0:.4f} "
              f"({overall_c}/{overall_t}); nonlatin->latin="
              f"{ev.nonlatin_as_latin}/{ev.nonlatin_pages}", file=sys.stderr)

    # 3c — int8 at a chosen cap (only if requested)
    if args.int8_cap > 0:
        if not os.path.exists(_INT8_PATH):
            result["notes"].append(f"int8 model missing, 3c not measured: {_INT8_PATH}")
        else:
            cap = args.int8_cap
            isess, iinput, iload = load_session(_INT8_PATH)
            result["int8"]["model_load_seconds"] = round(iload, 3)
            print(f"[3c] int8 model load = {iload:.3f}s; cap={cap}", file=sys.stderr)
            rows_3a_int8 = measure_3a(isess, iinput, raw_labels, [cap], args.repeats)
            ievals = measure_3b(isess, iinput, raw_labels, vocab, [cap],
                                args.per_script_cap, args.max_pages,
                                args.max_crops_per_source, args.seed)
            iev = ievals[cap]
            ips = iev.per_script()
            ic = sum(c for c, _, _ in ips.values())
            it = sum(t for _, t, _ in ips.values())
            result["int8"]["cap"] = cap
            result["int8"]["per_page_latency"] = rows_3a_int8
            result["int8"]["page_eval"] = {
                "per_script": {s: {"correct": c, "total": t, "acc": round(a, 4)}
                               for s, (c, t, a) in ips.items()},
                "overall_acc": round(ic / it, 4) if it else None,
                "overall_correct": ic,
                "overall_total": it,
                "confusion_lines": fmt_confusion(iev),
                "nonlatin_pages": iev.nonlatin_pages,
                "nonlatin_as_latin": iev.nonlatin_as_latin,
                "nonlatin_as_latin_rate": round(iev.nonlatin_as_latin / iev.nonlatin_pages, 4)
                    if iev.nonlatin_pages else None,
            }

    out = json.dumps(result, indent=2)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\nJSON written to {args.json_out}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
