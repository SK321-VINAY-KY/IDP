# Lipika Script-Detection Integration — Read-Only Audit

Scope: `c:\Users\Dell\Desktop\IDP\engineer_a`, commit `4d3af94` (HEAD of `feat/llm-observability`) plus two
unstaged model binaries on disk. No code, git, or AWS state was modified. No `.onnx` file was staged,
committed, or deleted. Every claim is tagged **[confirmed by reading]** or **[inferred]**.

---

## Summary answer (the crux first)

**The lipika observer is fed the WHOLE page image, not line crops, and `extract_line_crops()` does not exist
anywhere in the repo at `4d3af94`.** lipika (loopdesk-ai/lipika) is trained on tight *text-line / font crops*
("upload a crop of text" per its own PyPI page), so handing it a full downscaled scanned page is off-distribution —
this is the mechanical reason the coder observed misclassification even in native torch. The four `docs/Screenshot*.png`
validation images compound the problem: they are full-page **screen captures** (not scanner/camera crops), an even
worse proxy for the training distribution. The handwriting signal is fully removed from the script contract. fp32 is
the shipped default because int8 flipped at least one of those four (already-unreliable) predictions.

---

## 1. Changed files for the lipika work

Commands run:
```
git -C c:\Users\Dell\Desktop\IDP\engineer_a status --short
git -C c:\Users\Dell\Desktop\IDP\engineer_a diff --stat HEAD~1 HEAD
git -C c:\Users\Dell\Desktop\IDP\engineer_a show --stat 4d3af94
```
`HEAD`, `HEAD~1 HEAD`, and `show 4d3af94` all describe the same commit (`4d3af94` IS HEAD). **[confirmed by reading]**

Commit `4d3af94` — *"feat(probe): script-only lipika ONNX probe + in-process pipeline wiring"* — touches 25 files,
+2842/-19. The lipika-relevant additions/modifications:

| File | Change |
|------|--------|
| `scripts/export_lipika_script_onnx.py` | **added** (+224) — offline exporter |
| `src/ai/layer1_probe/script_classifier.py` | **added** (+198) — new whole-image observer |
| `src/ai/layer1_probe/models/lipika_script.labels.json` | **added** (+36) — model metadata |
| `src/ai/layer1_probe/probe.py` | **added** (+261) — Phase A/B/C orchestrator |
| `src/ai/layer1_probe/routing.py` | **added** (+28) — script×table → engine |
| `src/ai/layer1_probe/inprocess.py` | **added** (+52) — in-process pipeline hook |
| `src/ai/layer1_probe/{types,aggregate,cv_features,imaging,vlm_probe,script_verify}.py` | **added** |
| `src/api/lambda_probe_handler.py` | **added** (+119) |
| `tests/test_probe*.py`, `tests/test_pipeline_probe_integration.py` | **added** |
| `src/ai/layer1_routing/pipeline.py`, `src/ai/layer2_conversion/scanned.py`, `schema_chatbot_v2/...` | modified — wiring |

### Was `src/ai/layer1_probe/script_classifier.py` modified?
**It was ADDED, not modified from a prior tracked version.** `git show HEAD~1:src/ai/layer1_probe/script_classifier.py`
returns exit code 1 (path absent at `HEAD~1`). **[confirmed by reading]** So the entire `src/ai/layer1_probe/` package,
including `script_classifier.py`, is new in this commit.

Because the file is new, there is no tracked "old `OnnxScriptClassifier` line-crop contract" to diff against in git.
What exists in the file is the **new** contract and a note that `OnnxScriptClassifier` is retained only as a
module-level alias for backward-compatible imports:

- `script_classifier.py:198` — `OnnxScriptClassifier = LipikaOnnxScriptObserver` **[confirmed by reading]**
- `script_classifier.py:3-21` (module docstring) documents the whole-image contract: input `float32 [N,3,H,W]`, RGB,
  0..1, letterboxed to a square, no ImageNet mean/std; "classifies the WHOLE page image (downscaled to the fixed
  square), not line crops." **[confirmed by reading]**

Old-vs-new, as far as the code reveals it: the name `OnnxScriptClassifier` (implying a per-line ONNX classifier with
averaged-vote logits) is now just an alias; the `_MIXED_MARGIN = 0.25` comment at `script_classifier.py:36-40`
explicitly calls itself a "provisional carry-over from the old per-crop averaged-vote model" that is "NOT re-validated
on real lipika logits." **[confirmed by reading]** This is the only textual trace of the previous line-crop design.

### Untracked model artifacts (the `.onnx` files)
`git status --short` lists under `??`:
- `src/ai/layer1_probe/models/lipika_script.onnx` — **111,642,305 bytes (~112 MB, fp32)** **[confirmed by reading]**
- `src/ai/layer1_probe/models/lipika_script.int8.onnx` — **28,494,953 bytes (~28 MB, int8)** **[confirmed by reading]**

Both are untracked (unstaged), not gitignored — `.gitignore` has no `onnx` entry. `.gitattributes` marks `*.onnx binary`
(line-ending handling only), so they would commit as raw binary, not LFS, if staged. **[confirmed by reading]**
The commit message itself says the binaries were "left unstaged pending a repo-hygiene decision (LFS vs commit)."

---

## 2. Model input spec and label list

Sources: `scripts/export_lipika_script_onnx.py`, `src/ai/layer1_probe/models/lipika_script.labels.json`,
`src/ai/layer1_probe/script_classifier.py`.

**Tensor shape / dtype** — `float32 [N, 3, 224, 224]`, dynamic batch axis.
- Export: `dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)` with `input_names=["pixel_values"]`,
  `output_names=["script_logits"]`, `dynamic_axes={"pixel_values": {0: "batch"}, ...}`, `opset_version=17`,
  `dynamo=False` — `export_lipika_script_onnx.py` (export block). **[confirmed by reading]**
- `_INPUT_SIZE = 224`; metadata `"input_size": [224, 224]`. **[confirmed by reading]**

**Color order** — RGB. Export opens images with `Image.open(path).convert("RGB")`; the runtime observer uses
`rgb` when supplied or `cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)` otherwise (`script_classifier.py:166`). **[confirmed by reading]**

**Normalization** — scale to 0..1 then `(x - mean)/std` with `mean=[0,0,0]`, `std=[1,1,1]`, i.e. **plain /255, no
ImageNet mean/std**. `labels.json` `"mean":[0,0,0] "std":[1,1,1]`; `_letterbox` does `resized.astype(float32)/255.0`
then `(x-mean)/std` (`script_classifier.py:_letterbox`). Export docstring: "No ImageNet mean/std is applied by lipika."
**[confirmed by reading]**

**Letterbox / resize** — pad to a square with **white (255)** padding, center the image, then **BICUBIC** resize to
224×224, transpose to NCHW. `_letterbox` builds `canvas = np.full((side,side,3),255)`, centers via
`y0=(side-h)//2, x0=(side-w)//2`, `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`, then
`np.transpose(x,(2,0,1))[None]`. The exporter mirrors lipika's own `fontrecog.infer.predictor.letterbox`
(`_letterbox_batch` → `letterbox(img, size=224)`), so runtime preprocessing is a numpy/OpenCV re-implementation of
lipika's torch letterbox. `preprocessing: "single_forward"` (Option A — one forward on the whole image). **[confirmed by reading]**

**Label list (index order, from `labels.json`)**:
```
0 bengali   1 devanagari  2 gujarati  3 gurmukhi  4 kannada  5 latin  6 malayalam
7 other     8 odia        9 other    10 arabic   11 other   12 tamil 13 telugu
```
These are **script-only** labels. The exporter maps lipika's 14 raw scripts → probe vocab via `_LIPIKA_TO_PROBE`:
`perso_arabic→arabic`; `meetei_mayek`, `ol_chiki`, `syloti_nagri → other` (indices 7, 9, 11); all others are direct
lowercase matches. **[confirmed by reading]**

**No print/handwriting labels.** The label set contains only scripts (plus `other`). The export isolates the FontNet
*script* head alone (`class ScriptHead` wraps `backbone + drop + script`, `forward → self.script(...)`), explicitly
dropping the family/weight/italic heads. There are no `_print` / `_hw` entries, and `SampleObs` carries no
handwriting field — `tests/test_probe.py:358` asserts `"hw_prob" not in SampleObs.model_fields`. Handwriting is gone.
**[confirmed by reading]**

---

## 3. What image the observer actually receives (the crux)

**The full page image is fed to the model. There are no line crops. `extract_line_crops()` does not exist in the repo.**
`grep` for `extract_line_crops` across `**/*.py` → **no matches**. **[confirmed by reading]**

Two call paths both pass whole-page pixels:

**Path A — Lambda / `probe.py` Phase B** (`probe.py:160-165`): **[confirmed by reading]**
```python
if onnx is not None:
    gray, mask = _gray_mask_from_jpeg(jpeg, cfg)   # decodes the FULL-PAGE JPEG
    obs = onnx.observe(pn, gray, mask)
```
`jpeg` is the full rendered page: Phase A does `rgb = imaging.render_rgb(doc[pn-1], cfg.page_image_dpi)` then
`jpeg = imaging.encode_jpeg(rgb, ...)` with no cropping (`probe.py` Phase A loop). `_gray_mask_from_jpeg` decodes that
whole page and derives `gray` via `imaging.analysis_gray` (a downscale of the full page, never a crop). So the model
sees the **entire downscaled page letterboxed into 224×224**. **[confirmed by reading]**

**Path B — in-process pipeline** (`inprocess.py:44`): `obs = _get_observer().observe(page_number, gray, mask, rgb=rgb)`
where `rgb = page_context.get("image_array")` is the full page array (`inprocess.py:35`). **[confirmed by reading]**

Inside the observer, `observe(...)` takes whatever page array it is given and squeezes it whole into 224×224 via
`_letterbox` (`script_classifier.py:173-176`): `src = rgb if rgb is not None else gray→RGB`; `x = _letterbox(src, 224, ...)`.
No tiling, no line segmentation, no crop loop. `preprocessing:"single_forward"` confirms a single whole-image forward
pass. **[confirmed by reading]**

**Is `extract_line_crops()` dead code?** It is not dead code — it is *absent*. The old line-crop contract referenced in
the brief left no function behind in this commit; the only residue is the stale name `OnnxScriptClassifier` (alias) and
the `_MIXED_MARGIN` "carry-over" comment. **[confirmed by reading]**

**Why this explains the misclassification** — lipika's FontNet was trained on tight single-line / font crops. A full
A4 scan letterboxed to 224×224 reduces body text to a few pixels tall, surrounded by whitespace, headers, logos, rules,
and tables — a global texture lipika never saw in training. Feeding that to a line-crop model yields unreliable logits,
which is exactly what was observed even in native torch (so it is not an ONNX-export artifact). **[inferred]** — mechanism
inferred from the confirmed whole-image code path plus lipika's documented crop-level training input (see §4 citations).

---

## 4. Validation-image provenance

**The four `docs/Screenshot*.png` are full-page screen captures, and `docs/` is an untracked new directory.**
`git log --oneline -- "docs/*.png"` returns nothing — the files are not tracked. **[confirmed by reading]**
`.gitignore` has no `docs`/`Screenshot` entry and `.gitattributes` only marks `*.png binary` (line-ending policy), so
`docs/` is simply uncommitted, not deliberately ignored. (No `.dockerignore` governs them for this audit; they are
dev-only validation assets, not shipped in the image.) **[confirmed by reading]**

Filenames and on-disk timestamps line up as screen-capture artifacts: **[confirmed by reading]**
| File | Size | LastWrite | Brief's label |
|------|------|-----------|---------------|
| Screenshot 2026-10-23 140614.png | 647 KB | 14:06:14 | latin (hospital tariff) |
| Screenshot 2026-10-23 140632.png | 674 KB | 14:06:32 | latin (food/discharge) |
| Screenshot 2026-10-23 142509.png | 390 KB | 14:25:10 | telugu passage |
| Screenshot 2026-10-23 142632.png | 547 KB | 14:26:33 | devanagari (handwritten Hindi notebook) |

The embedded timestamp equals the file mtime to the second ("Screenshot YYYY-MM-DD HHMMSS") — the OS screenshot naming
convention, confirming screen captures rather than scanner output. The exporter hard-codes them as its only validation
set: `_VALIDATION = [("Screenshot 2026-10-23 140614.png","latin"), ...]`. **[confirmed by reading]**

**Why screenshots are a poor proxy for lipika's distribution:** **[inferred, grounded in cited sources]**
1. *Whole page vs line crop.* lipika expects a crop of text (PyPI: "Upload a crop of text in any of 13 Indic scripts"
   — https://pypi.org/project/lipika/2.4.0/; HF model card describes uploading "an image of Indian-script text" —
   https://huggingface.co/loopdesk-ai/lipika/blob/main/README.md). A full page is the wrong granularity regardless of capture method. *Content was rephrased for compliance with licensing restrictions.*
2. *Screenshot rendering vs scan noise.* Screenshots are crisp, anti-aliased, uniformly lit, pure-digital rasterizations.
   Real scans/photos carry JPEG artifacts, skew, bleed-through, uneven lighting, paper texture, and 144-DPI resampling —
   the domain the probe actually runs on (per `probe.py` docstring: the probe runs on *scanned* pages). Validating on
   screenshots tests neither the real input granularity nor the real noise distribution.
3. *The Telugu/Devanagari cases are additionally a page passage and a handwritten notebook* — multi-line layouts, and in
   the handwritten case a modality lipika (a printed-font recognizer) is not built for.

---

## Conclusions

- The integration is mechanically whole-image; lipika is a line/font-crop model. This granularity mismatch — not the
  ONNX export, not int8 quantization — is the root cause of the off-distribution misclassification. **[inferred from confirmed code + sources]**
- The only "validation" wired in is four screenshots embedded in a dev-only exporter; it is not a representative test
  and cannot justify any routing decision. **[confirmed by reading]**
- Safe-by-default behavior holds today: `routing.decide_route` sends every non-Latin detected script, plus
  `unknown`/`other`/low-confidence/mixed, to `vlm_transcribe` (`routing.py`), so a bad script guess currently costs
  latency (VLM), not garbled OCR. Only `latin` routes to `paddleocr_printed(en)`. **[confirmed by reading]**
- fp32 is shipped because int8 flipped at least one of the four screenshot predictions
  (`shipped_default = "int8" if ... int8_preserves else "fp32"`, with the "INT8 flipped at least one prediction" log).
  Since the screenshots themselves are unreliable, that fp32-vs-int8 decision rests on an unreliable comparison. **[confirmed by reading]**

---

## PROPOSED validation plan (NOT implemented)

### Dataset — ≥30 REAL line crops per script
Target scripts (all must have a `SCRIPT_TO_PADDLE_LANG` entry so routing-without-VLM is even meaningful —
`types.py:29-44`): **Latin, Telugu, Devanagari, plus Tamil (`ta`) or Kannada (`ka`)** as the fourth. **[confirmed by reading]**

Sources, in priority order:
1. **The repo's own `dataset/` PDFs** — `c:\...\engineer_a\dataset\*.pdf` (e.g. `165.pdf`..`170.pdf`, plus per-user
   trees under `dataset/_users/...`). These are genuine documents already in-domain for the pipeline. Render each page
   to RGB with the production `imaging.render_rgb(page, cfg.page_image_dpi)` so crops carry real scan characteristics.
   **[confirmed by reading — dataset dir enumerated]**
2. **Public scanned-text corpora** for scripts under-represented in `dataset/`: IndicSTR12 / Indic scene-text and
   Devanagari OCR line-image sets (the family of Indic STR datasets, e.g. the ~27k word-image Indic STR dataset —
   https://arxiv.org/html/2403.08007). lipika benchmarks on this family, so crops from it are maximally in-distribution.
   *Content was rephrased for compliance with licensing restrictions.* **[inferred]**
3. Hand-crop a small gold set from a handful of real scans per script to anchor ground truth.

Balance: ≥30 line crops/script × 4 scripts = ≥120 crops minimum; stratify across fonts, sizes, and scan quality.

### How to generate line crops so input matches lipika's distribution
There is currently **no** `extract_line_crops()` to reuse (confirmed absent in §3), so the plan must add a
line-segmentation helper rather than "reuse" one. Concretely:
1. Render each `dataset/` page: `rgb = imaging.render_rgb(page, 144)`; `gray = imaging.analysis_gray(rgb, 144, 100)`.
2. Build the ink mask with the existing `imaging.ink_mask(gray)` (already used by the probe — `imaging.py:ink_mask`),
   then horizontal-projection / connected-component line segmentation to cut individual text-line bounding boxes.
3. Crop each line from the full-resolution `rgb` (not the analysis gray) with a few px padding, so each crop is a tight
   single text line — the granularity lipika expects. Letterbox exactly as `script_classifier._letterbox` does at
   inference, so the validation preprocessing is byte-identical to production.
4. Label each crop by the source document's known script. **[plan — grounded in confirmed `imaging.py` API]**

### Metric
- **Per-script accuracy** (top-1) on the line-crop set, measured on **both** `lipika_script.onnx` (fp32) and
  `lipika_script.int8.onnx` (int8), loading each via the observer's `IDP_PROBE_ONNX_MODEL` override so the exact runtime
  path is exercised (`script_classifier.default_model_path` / env override — `script_classifier.py:_get_session`). **[confirmed by reading]**
- **Full confusion matrix** across the probe script vocabulary (the 14 label slots, collapsing the three `other`
  indices) — rows = true script, cols = predicted — so cross-script leakage (e.g. Devanagari↔Gujarati, Telugu↔Kannada)
  is visible, not just aggregate accuracy.
- Also report the **top1−top2 softmax margin** distribution, since `_MIXED_MARGIN`/`onnx_min_conf` gate VLM escalation
  and are currently un-validated (`script_classifier.py:36-40`, `types.py` `onnx_min_conf=0.80`). **[confirmed by reading]**

### Routing threshold to bypass the VLM safety net
Today every non-Latin script escalates to the VLM (`routing.py`). To let a non-Latin script route straight to its
PaddleOCR engine without the VLM, require **per-script top-1 accuracy ≥ 0.98 AND a per-script confidence gate (top-1
softmax) chosen so that, among crops at/above the gate, precision for that script is ≥ 0.99** on the real line-crop set,
for the fp32 model. Rationale tied to misroute cost: a wrong-script route sends the page to the wrong OCR recognizer and
**silently garbles the transcription** (hard to detect downstream, corrupts extracted fields), whereas the VLM is slow
but safe. Because a misroute is far costlier than an extra VLM call, the bar must be high — a ≤1% misroute rate at the
gate is the most leniency the cost asymmetry justifies, and only for scripts that clear it. Scripts below the bar keep
the current always-VLM policy. **[inferred — proposal]**

**Fallback policy below threshold:** if top-1 confidence < the per-script gate, or the script did not clear the ≥0.98
bar, or `mixed_scripts` is set, or the label is `other`/`unknown` → route to `vlm_transcribe` (today's behavior). This
preserves safe-by-default and makes direct-to-OCR an *opt-in per script* earned by evidence. **[inferred — proposal]**

### Effect on the fp32-vs-int8 shipped default
The current default is **fp32** solely because int8 flipped ≥1 of the four screenshot predictions
(`export_lipika_script_onnx.py` shipped-default logic). On a proper line-crop set, compute per-script accuracy and the
confusion matrix for both models; if int8 matches fp32 within a small tolerance (e.g. ≤0.5% absolute per-script
accuracy drop and no new confusion that pushes any script below its routing gate), **switch the shipped default to int8**
(4× smaller: 28 MB vs 112 MB — confirmed sizes in §1) by setting `shipped_default` in `labels.json`. If int8 degrades
any gated script, keep fp32. The decision should rest on the real line-crop metric, never on the four screenshots. **[inferred — proposal]**

---

### Evidence index (file:line / commands)
- `git -C <repo> status --short` → untracked `.onnx` files, `docs/`, modified wiring files.
- `git -C <repo> show --stat 4d3af94` / `diff --stat HEAD~1 HEAD` → 25 files, lipika additions.
- `git -C <repo> show HEAD~1:src/ai/layer1_probe/script_classifier.py` → exit 1 (file new in `4d3af94`).
- `git -C <repo> log --oneline -- "docs/*.png"` → empty (untracked).
- `grep extract_line_crops **/*.py` → no matches.
- `script_classifier.py:3-21,36-40,166,173-176,198`; `probe.py:160-165` + Phase A; `inprocess.py:35,44`;
  `routing.py` (decide_route); `types.py:29-44`; `imaging.py` (render_rgb/analysis_gray/ink_mask);
  `models/lipika_script.labels.json`; `scripts/export_lipika_script_onnx.py` (`_LIPIKA_TO_PROBE`, `_VALIDATION`,
  `ScriptHead`, export/quant/validation blocks).
- File sizes: `lipika_script.onnx` 111,642,305 B; `lipika_script.int8.onnx` 28,494,953 B.
- External: lipika PyPI 2.4.0 (crop of text) https://pypi.org/project/lipika/2.4.0/ ; HF model card
  https://huggingface.co/loopdesk-ai/lipika/blob/main/README.md ; Indic STR dataset https://arxiv.org/html/2403.08007 .
