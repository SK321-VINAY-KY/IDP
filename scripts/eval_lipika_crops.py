"""
File: scripts/eval_lipika_crops.py
Purpose: DEVELOPER-ONLY, OFFLINE eval harness for the lipika script ONNX model.
         Compares fp32 (lipika_script.onnx) vs int8 (lipika_script.int8.onnx)
         under three input strategies on real dataset word crops + Latin line
         crops rendered from the English PDFs in dataset/.

           A  full-crop letterbox = the shared observer preprocessing
              (src.ai.layer1_probe.windows.letterbox_square:
               white pad to square, cv2.resize INTER_CUBIC, /255, NCHW).
           B  square-ish windows cut along each crop (aspect 1:1..2:1),
              letterbox each window as in A, mean the softmax over windows.
           C  lipika's OWN documented preprocessing, a numpy/PIL replica of
              fontrecog.infer.predictor.letterbox (PIL RGB, pad to square
              fill=255, Image.BICUBIC resize to 224, /255, CHW).

IMPORTANT: this script is NOT imported at runtime by the probe, and it imports
NO torch and NO fontrecog. fontrecog.infer.predictor.letterbox returns a
torch.Tensor and importing it pulls torch, so strategy C is a faithful numpy/PIL
REPLICA of that letterbox (source read: convert RGB, pad to max(w,h) square with
fill=255, resize to 224 BICUBIC, arr/255, permute to CHW). Deps: onnxruntime,
numpy, OpenCV, PIL, fitz, stdlib. All heavy execution is guarded under
`if __name__ == "__main__":`.

The full run is a multi-minute CPU sweep over thousands of crops. For CI/smoke,
use --min-per-script 2 --per-script-cap 3.

Run:  .venv\\Scripts\\python.exe scripts/eval_lipika_crops.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Allow running as a plain script (`python scripts/eval_lipika_crops.py`): put the
# repo root on sys.path so the absolute `src....` imports resolve. Harmless when
# the module is imported with the repo root already on the path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.ai.layer1_probe import imaging
from src.ai.layer1_probe.line_crops import extract_line_crops
# Strategy A/B preprocessing now lives in the library module so the line-wise
# observer and this eval harness run byte-identical tensors (no duplication).
from src.ai.layer1_probe.windows import (
    _INPUT_SIZE,
    letterbox_square as _letterbox_A,
    square_windows as _windows_B,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_REPO, "src", "ai", "layer1_probe", "models")
_DATASET_DIR = os.path.join(_REPO, "dataset")
_VERIFIED_DIR = os.path.join(_DATASET_DIR, "verified_twice")
_LABELS_PATH = os.path.join(_MODELS_DIR, "lipika_script.labels.json")
_FP32_PATH = os.path.join(_MODELS_DIR, "lipika_script.onnx")
_INT8_PATH = os.path.join(_MODELS_DIR, "lipika_script.int8.onnx")
_REPORT_PATH = os.path.join(_REPO, ".agents", "tasks", "lipika-eval-results.md")

_OTHER = "other"

# Authoritative dataset language folder -> probe script label (context.json).
LANGUAGE_TO_SCRIPT: Dict[str, str] = {
    "hindi": "devanagari",
    "marathi": "devanagari",
    "assamese": "bengali",
    "bengali": "bengali",
    "punjabi": "gurmukhi",
    "gujarati": "gujarati",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "odia": "odia",
    "tamil": "tamil",
    "telugu": "telugu",
    "urdu": "arabic",
}

# The five English PDFs in dataset/ that supply Latin line crops.
LATIN_PDFS: List[str] = [
    "MY_resume.pdf",
    "Vinay_Resume.pdf",
    "bert vs sas.pdf",
    "missouri_water_sample.pdf",
    "Chander Kochhar 01_compressed 2.pdf",
]

# Render / analysis DPI for PDF/image line extraction (ProbeConfig defaults).
_RENDER_DPI = 144
_ANALYSIS_DPI = 100

_IGNORE_DIR = {"__MACOSX"}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def load_labels(labels_path: str) -> Tuple[List[str], List[str]]:
    """Return (raw_labels, collapsed_vocab).

    raw_labels: the model's 14 output slots verbatim (three are 'other').
    collapsed_vocab: ordered unique script names used for the confusion matrix,
                     with the three 'other' slots collapsed to a single 'other'.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    raw = list(meta.get("labels", []))
    vocab: List[str] = []
    for name in raw:
        if name not in vocab:
            vocab.append(name)
    return raw, vocab


def collapse_label(raw_labels: List[str], idx: int) -> str:
    """Map a model output index to its collapsed script name ('other' unified)."""
    if 0 <= idx < len(raw_labels):
        return raw_labels[idx]
    return "unknown"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
class ManifestRow:
    __slots__ = ("source_path", "kind", "true_script", "page")

    def __init__(self, source_path: str, kind: str, true_script: str,
                 page: Optional[int] = None):
        self.source_path = source_path
        self.kind = kind            # crop | pdf | image
        self.true_script = true_script
        self.page = page


def read_manifest_csv(path: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            page = r.get("page")
            rows.append(ManifestRow(
                source_path=r["source_path"],
                kind=(r.get("kind") or "crop").strip(),
                true_script=r["true_script"].strip(),
                page=int(page) if page not in (None, "") else None,
            ))
    return rows


def build_auto_manifest() -> Tuple[List[ManifestRow], List[str]]:
    """Build the balanced-source manifest from the dataset layout.

    Returns (rows, skipped) where `skipped` lists folders/PDFs that could not be
    mapped (and were therefore skipped). NEVER uses lipika/VLM to label.
    """
    rows: List[ManifestRow] = []
    skipped: List[str] = []

    # crop rows: dataset/verified_twice/<lang>/cropped_images/*.jpeg
    if os.path.isdir(_VERIFIED_DIR):
        for name in sorted(os.listdir(_VERIFIED_DIR)):
            if name in _IGNORE_DIR or name.startswith("."):
                continue
            folder = os.path.join(_VERIFIED_DIR, name)
            if not os.path.isdir(folder):
                continue
            script = LANGUAGE_TO_SCRIPT.get(name.lower())
            if script is None:
                skipped.append(f"folder:{name} (no LANGUAGE_TO_SCRIPT mapping)")
                continue
            crop_dir = os.path.join(folder, "cropped_images")
            if not os.path.isdir(crop_dir):
                skipped.append(f"folder:{name} (no cropped_images/)")
                continue
            jpegs = sorted(glob.glob(os.path.join(crop_dir, "*.jpeg")))
            for jp in jpegs:
                base = os.path.basename(jp)
                if base.startswith(".") or base == ".DS_Store":
                    continue
                rows.append(ManifestRow(jp, "crop", script))
    else:
        skipped.append(f"dataset dir missing: {_VERIFIED_DIR}")

    # pdf rows: the five English PDFs -> latin
    for pdf in LATIN_PDFS:
        p = os.path.join(_DATASET_DIR, pdf)
        if os.path.exists(p):
            rows.append(ManifestRow(p, "pdf", "latin"))
        else:
            skipped.append(f"pdf:{pdf} (missing)")

    return rows, skipped


def balance_manifest(rows: List[ManifestRow], min_per_script: int,
                     per_script_cap: int, seed: int = 0
                     ) -> Tuple[List[ManifestRow], Dict[str, int], List[str]]:
    """Sample a balanced subset of SOURCE rows per true_script.

    PDFs expand to many line crops downstream, so for `pdf`/`image` sources we
    keep all mapped sources (there are only five Latin PDFs) and rely on the
    crop cap during production. For `crop` sources we cap the number of jpeg
    sources per script at `per_script_cap`. Returns the sampled rows, the actual
    per-script SOURCE counts, and a list of scripts that fall below
    `min_per_script` sources (a caveat, not a hard failure).
    """
    rng = np.random.default_rng(seed)
    by_script: Dict[str, List[ManifestRow]] = {}
    for r in rows:
        by_script.setdefault(r.true_script, []).append(r)

    sampled: List[ManifestRow] = []
    counts: Dict[str, int] = {}
    low: List[str] = []
    for script, group in sorted(by_script.items()):
        crops = [r for r in group if r.kind == "crop"]
        others = [r for r in group if r.kind != "crop"]
        if crops:
            idx = np.arange(len(crops))
            rng.shuffle(idx)
            keep = idx[:per_script_cap]
            chosen = [crops[i] for i in keep]
        else:
            chosen = []
        chosen += others  # keep all pdf/image sources
        sampled += chosen
        counts[script] = len(chosen)
        if len(chosen) < min_per_script:
            low.append(script)
    return sampled, counts, low


# ---------------------------------------------------------------------------
# Crop production
# ---------------------------------------------------------------------------
def _scale_box(box: Tuple[int, int, int, int], fx: float, fy: float,
               W: int, H: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x0 = max(0, int(round(x * fx)))
    y0 = max(0, int(round(y * fy)))
    x1 = min(W, int(round((x + w) * fx)))
    y1 = min(H, int(round((y + h) * fy)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crops_from_rgb_page(rgb: np.ndarray) -> List[np.ndarray]:
    """Render -> analysis gray -> extract line crops -> map each box back to the
    full-res RGB page and return the RGB sub-images."""
    gray = imaging.analysis_gray(rgb, _RENDER_DPI, _ANALYSIS_DPI)
    boxes = extract_line_crops(gray)
    if not boxes:
        return []
    gh, gw = gray.shape[:2]
    H, W = rgb.shape[:2]
    fx = W / float(gw)
    fy = H / float(gh)
    out: List[np.ndarray] = []
    for _crop, box in boxes:
        x0, y0, bw, bh = _scale_box(box, fx, fy, W, H)
        if bw <= 0 or bh <= 0:
            continue
        out.append(rgb[y0:y0 + bh, x0:x0 + bw].copy())
    return out


def produce_crops(row: ManifestRow, max_pages: int = 4) -> List[np.ndarray]:
    """Produce a list of RGB crops (H, W, 3 uint8) for one manifest source.

    crop  -> load the jpeg directly (already a tight word crop).
    pdf   -> fitz render each page, cut line crops, map to full-res rgb.
    image -> cv2 decode, same line-crop path.
    """
    kind = row.kind
    if kind == "crop":
        bgr = cv2.imread(row.source_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return []
        return [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]

    if kind == "pdf":
        import fitz  # local import: only PDFs need it
        try:
            doc = fitz.open(row.source_path)
        except Exception:
            return []
        out: List[np.ndarray] = []
        try:
            n = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
            pages = [row.page] if row.page is not None else range(n)
            for pn in pages:
                if pn < 0 or pn >= doc.page_count:
                    continue
                rgb = imaging.render_rgb(doc[pn], _RENDER_DPI)
                out += crops_from_rgb_page(rgb)
        finally:
            doc.close()
        return out

    if kind == "image":
        bgr = cv2.imread(row.source_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return crops_from_rgb_page(rgb)

    return []


# ---------------------------------------------------------------------------
# Strategies (produce model input tensors [N,3,224,224] float32)
# ---------------------------------------------------------------------------
# Strategy A (`_letterbox_A`) and Strategy B (`_windows_B`) are imported from
# src.ai.layer1_probe.windows (letterbox_square / square_windows) so the eval and
# the observer share ONE implementation. Only Strategy C lives here (it is the
# fontrecog-replica path, not used by the runtime observer).


def _letterbox_C(rgb: np.ndarray, size: int = _INPUT_SIZE) -> np.ndarray:
    """Strategy C: faithful numpy/PIL REPLICA of fontrecog.infer.predictor.letterbox.

    fontrecog is NOT imported (its letterbox returns a torch.Tensor and importing
    it pulls torch). Source read: PIL convert RGB, pad to max(w,h) square with
    fill=255, resize to 224 with Image.BICUBIC, np.array/255, transpose to CHW.
    """
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    w, h = img.size
    side = max(w, h, 1)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, :, :, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_session(sess, input_name: str, x: np.ndarray) -> np.ndarray:
    """Run the model on a batch [N,3,224,224] and return softmax probs [N,K]."""
    logits = np.asarray(sess.run(None, {input_name: x})[0], dtype=np.float32)
    return _softmax(logits)


def crop_softmax(sess, input_name: str, rgb: np.ndarray, strategy: str) -> np.ndarray:
    """Return a single [K] softmax vector for one crop under a strategy."""
    if strategy == "A":
        x = _letterbox_A(rgb)
        return run_session(sess, input_name, x)[0]
    if strategy == "C":
        x = _letterbox_C(rgb)
        return run_session(sess, input_name, x)[0]
    if strategy == "B":
        windows = _windows_B(rgb)
        if not windows:
            return run_session(sess, input_name, _letterbox_A(rgb))[0]
        batch = np.concatenate([_letterbox_A(w) for w in windows], axis=0)
        probs = run_session(sess, input_name, batch)  # [W, K]
        return probs.mean(axis=0)
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class StratResult:
    """Accumulates per-crop predictions + per-source (page-level) votes."""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.index = {name: i for i, name in enumerate(vocab)}
        k = len(vocab)
        self.confusion = np.zeros((k, k), dtype=np.int64)  # rows=true, cols=pred
        self.per_script_total: Dict[str, int] = {}
        self.per_script_correct: Dict[str, int] = {}
        self.margins: List[float] = []
        # source-level vote accuracy
        self.vote_total: Dict[str, int] = {}
        self.vote_correct: Dict[str, int] = {}
        self.min_crops_per_source: Optional[int] = None

    def add_crop(self, true_script: str, probs: np.ndarray, raw_labels: List[str]):
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        pred = collapse_label(raw_labels, top_idx)
        top = float(probs[top_idx])
        second = float(probs[int(order[1])]) if probs.shape[0] > 1 else 0.0
        self.margins.append(top - second)
        self.per_script_total[true_script] = self.per_script_total.get(true_script, 0) + 1
        if pred == true_script:
            self.per_script_correct[true_script] = self.per_script_correct.get(true_script, 0) + 1
        ti = self.index.get(true_script)
        pi = self.index.get(pred)
        if ti is not None and pi is not None:
            self.confusion[ti, pi] += 1

    def add_source_vote(self, true_script: str, mean_probs: np.ndarray,
                        raw_labels: List[str], n_crops: int):
        pred = collapse_label(raw_labels, int(np.argmax(mean_probs)))
        self.vote_total[true_script] = self.vote_total.get(true_script, 0) + 1
        if pred == true_script:
            self.vote_correct[true_script] = self.vote_correct.get(true_script, 0) + 1
        if self.min_crops_per_source is None or n_crops < self.min_crops_per_source:
            self.min_crops_per_source = n_crops

    def per_script_accuracy(self) -> Dict[str, Tuple[int, int, float]]:
        out: Dict[str, Tuple[int, int, float]] = {}
        for s, total in sorted(self.per_script_total.items()):
            correct = self.per_script_correct.get(s, 0)
            out[s] = (correct, total, (correct / total) if total else 0.0)
        return out

    def vote_accuracy(self) -> Dict[str, Tuple[int, int, float]]:
        out: Dict[str, Tuple[int, int, float]] = {}
        for s, total in sorted(self.vote_total.items()):
            correct = self.vote_correct.get(s, 0)
            out[s] = (correct, total, (correct / total) if total else 0.0)
        return out

    def margin_stats(self) -> Dict[str, float]:
        if not self.margins:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
        m = np.asarray(self.margins, dtype=np.float64)
        return {
            "min": float(m.min()),
            "median": float(np.median(m)),
            "mean": float(m.mean()),
            "max": float(m.max()),
            "n": int(m.size),
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_confusion(res: StratResult) -> List[str]:
    vocab = res.vocab
    # restrict to rows/cols that appear to keep the table readable
    present = [i for i, _ in enumerate(vocab)
               if res.confusion[i].sum() > 0 or res.confusion[:, i].sum() > 0]
    if not present:
        return ["(no data)"]
    short = [vocab[i][:4] for i in present]
    header = "true\\pred | " + " | ".join(f"{s:>5}" for s in short)
    lines = [header, "-" * len(header)]
    for i in present:
        cells = " | ".join(f"{res.confusion[i, j]:>5}" for j in present)
        lines.append(f"{vocab[i][:8]:>8} | {cells}")
    return lines


def write_report(path: str, results: Dict[Tuple[str, str], StratResult],
                 counts: Dict[str, int], crop_counts: Dict[str, int],
                 skipped: List[str], low: List[str], models_run: List[str],
                 strategies: List[str], smoke: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# Lipika script ONNX eval — fp32 vs int8 under strategies A/B/C")
    lines.append("")
    if smoke:
        lines.append("> SMOKE RUN (reduced sampling) — not the full eval. "
                     "Re-run without --min-per-script/--per-script-cap limits for final numbers.")
        lines.append("")
    lines.append("## Strategies")
    lines.append("- **A** — full-crop letterbox = shared observer preprocessing "
                 "(`windows.letterbox_square`: white pad to square, "
                 "cv2.resize INTER_CUBIC, /255, NCHW).")
    lines.append("- **B** — square-ish windows (aspect 1:1..2:1) cut along the "
                 "longer axis, each letterboxed as A; softmax averaged over windows.")
    lines.append("- **C** — faithful numpy/PIL REPLICA of "
                 "`fontrecog.infer.predictor.letterbox` (PIL RGB, pad to square "
                 "fill=255, Image.BICUBIC resize to 224, /255, CHW). "
                 "**fontrecog is NOT imported** (its letterbox returns a "
                 "torch.Tensor and pulls torch).")
    lines.append("")
    lines.append("## Sources & crops")
    lines.append("Per-script SOURCE counts (crops=jpeg files, pdfs=PDF files):")
    lines.append("")
    lines.append("| script | sources | crops produced |")
    lines.append("| --- | ---: | ---: |")
    for s in sorted(set(list(counts.keys()) + list(crop_counts.keys()))):
        lines.append(f"| {s} | {counts.get(s, 0)} | {crop_counts.get(s, 0)} |")
    lines.append("")
    if low:
        lines.append(f"**Caveat:** scripts below --min-per-script sources: {', '.join(low)}")
        lines.append("")
    if skipped:
        lines.append("**Skipped (unmappable, never labelled by lipika/VLM):**")
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")
    lines.append(f"Models run: {', '.join(models_run) if models_run else '(none)'}")
    lines.append("")

    for model in models_run:
        for strat in strategies:
            res = results.get((model, strat))
            if res is None:
                continue
            lines.append(f"## {model} — strategy {strat}")
            lines.append("")
            lines.append("### Per-script top-1 accuracy")
            lines.append("| script | correct | total | acc |")
            lines.append("| --- | ---: | ---: | ---: |")
            for s, (c, t, a) in res.per_script_accuracy().items():
                lines.append(f"| {s} | {c} | {t} | {a:.4f} |")
            lines.append("")
            lines.append("### Confusion matrix (rows=true, cols=pred, 'other' collapsed)")
            lines.append("```")
            lines += _fmt_confusion(res)
            lines.append("```")
            lines.append("")
            ms = res.margin_stats()
            lines.append("### Top1-top2 softmax margin")
            lines.append(f"- n={ms['n']} min={ms['min']:.4f} median={ms['median']:.4f} "
                         f"mean={ms['mean']:.4f} max={ms['max']:.4f}")
            lines.append("")
            lines.append("### Page/source-level vote accuracy (argmax of mean softmax per source)")
            lines.append(f"- minimum crop count per source N = {res.min_crops_per_source}")
            lines.append("| script | correct | total | acc |")
            lines.append("| --- | ---: | ---: | ---: |")
            for s, (c, t, a) in res.vote_accuracy().items():
                lines.append(f"| {s} | {c} | {t} | {a:.4f} |")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(results: Dict[Tuple[str, str], StratResult],
                  counts: Dict[str, int], crop_counts: Dict[str, int],
                  models_run: List[str], strategies: List[str]) -> None:
    print("\n=== lipika eval summary ===")
    print("per-script SOURCE counts:", {k: counts[k] for k in sorted(counts)})
    print("per-script CROP counts  :", {k: crop_counts[k] for k in sorted(crop_counts)})
    for model in models_run:
        for strat in strategies:
            res = results.get((model, strat))
            if res is None:
                continue
            accs = res.per_script_accuracy()
            overall_c = sum(c for c, _, _ in accs.values())
            overall_t = sum(t for _, t, _ in accs.values())
            overall = (overall_c / overall_t) if overall_t else 0.0
            print(f"[{model} / {strat}] overall top-1 = {overall:.4f} "
                  f"({overall_c}/{overall_t} crops)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_session(path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def run_eval(args) -> int:
    raw_labels, vocab = load_labels(_LABELS_PATH)

    # --- manifest ---
    if args.manifest:
        rows = read_manifest_csv(args.manifest)
        skipped: List[str] = []
    else:
        rows, skipped = build_auto_manifest()

    sampled, counts, low = balance_manifest(
        rows, args.min_per_script, args.per_script_cap, seed=args.seed)

    if skipped:
        print("Skipped (unmappable):")
        for s in skipped:
            print(f"  - {s}")
    if low:
        print(f"Below --min-per-script sources: {', '.join(low)}")

    # --- models ---
    models: List[Tuple[str, str]] = []
    if os.path.exists(_FP32_PATH):
        models.append(("fp32", _FP32_PATH))
    else:
        print(f"WARNING: fp32 model missing, skipping: {_FP32_PATH}", file=sys.stderr)
    if os.path.exists(_INT8_PATH):
        models.append(("int8", _INT8_PATH))
    else:
        print(f"WARNING: int8 model missing, skipping: {_INT8_PATH}", file=sys.stderr)
    if not models:
        print("FATAL: both fp32 and int8 model files are missing.", file=sys.stderr)
        return 2

    strategies = ["A", "B", "C"]
    results: Dict[Tuple[str, str], StratResult] = {}
    for model_name, _ in models:
        for strat in strategies:
            results[(model_name, strat)] = StratResult(vocab)

    # Produce crops once per source; reuse across all model x strategy combos.
    # Reuse ONE session per model across all strategies.
    sessions: Dict[str, Tuple[object, str]] = {}
    for model_name, path in models:
        sessions[model_name] = _load_session(path)

    crop_counts: Dict[str, int] = {}
    cap = args.per_script_cap
    for row in sampled:
        crops = produce_crops(row, max_pages=args.max_pages)
        if not crops:
            continue
        # Cap crops per source so a long PDF does not dominate; also respects the
        # per_script_cap loosely at the source level.
        if args.max_crops_per_source > 0 and len(crops) > args.max_crops_per_source:
            crops = crops[:args.max_crops_per_source]
        crop_counts[row.true_script] = crop_counts.get(row.true_script, 0) + len(crops)

        for model_name, _ in models:
            sess, input_name = sessions[model_name]
            for strat in strategies:
                res = results[(model_name, strat)]
                mean_acc = np.zeros(len(raw_labels), dtype=np.float64)
                for rgb in crops:
                    probs = crop_softmax(sess, input_name, rgb, strat)
                    res.add_crop(row.true_script, probs, raw_labels)
                    mean_acc += probs
                mean_acc /= max(1, len(crops))
                res.add_source_vote(row.true_script, mean_acc, raw_labels, len(crops))

    models_run = [m for m, _ in models]
    write_report(_REPORT_PATH, results, counts, crop_counts, skipped, low,
                 models_run, strategies,
                 smoke=(args.min_per_script < 30 or args.per_script_cap < 150))
    print_summary(results, counts, crop_counts, models_run, strategies)
    print(f"\nReport written to {_REPORT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline lipika script ONNX eval (fp32 vs int8, strategies A/B/C).")
    p.add_argument("--manifest", default=None,
                   help="optional CSV (source_path, kind, true_script[, page]); "
                        "auto-built from dataset/ when omitted.")
    p.add_argument("--min-per-script", type=int, default=30,
                   help="minimum sources per script (reporting threshold).")
    p.add_argument("--per-script-cap", type=int, default=150,
                   help="cap on crop (jpeg) sources per script.")
    p.add_argument("--max-pages", type=int, default=4,
                   help="max PDF pages to read per PDF source (<=0 = all).")
    p.add_argument("--max-crops-per-source", type=int, default=40,
                   help="cap on crops produced per source (<=0 = all).")
    p.add_argument("--seed", type=int, default=0, help="sampling RNG seed.")
    return p


if __name__ == "__main__":
    _args = build_arg_parser().parse_args()
    raise SystemExit(run_eval(_args))
