# Implementation Plan — Probe Phase 2 (line-crop extractor + lipika eval + flag-gated handwriting)

LOCAL build only. NO AWS of any kind. Repo root `c:\Users\Dell\Desktop\IDP\engineer_a`, current branch, NO worktree.
All python/pytest run via the repo venv `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`
(confirmed: Python 3.11.9; onnxruntime 1.31.0; opencv 5.0.0; pymupdf/fitz, numpy, PIL present).

## Context discovered during exploration (do not re-decide)
- The handwriting teardown from the earlier `probe-plan.md` ALREADY landed: `SampleObs` has no `handwriting_prob`;
  `routing.decide_route(*, script, script_conf, mixed, is_blank, has_table, cfg)` has NO handwriting branch;
  `ProbeConfig` has no `hw_threshold`/`printed_scripts`; `vlm_probe` prompt has no `handwriting_fraction`;
  `cv_features` has no handwriting heuristic. **Phase 2 item 4 RE-ADDS handwriting, flag-gated** — it does not
  remove anything. Baseline `tests/test_probe.py` = 27 passed (confirmed).
- Model contract (confirmed from `scripts/export_lipika_script_onnx.py` + `models/lipika_script.labels.json` +
  `script_classifier._letterbox`): input float32 `[N,3,224,224]`, RGB, `/255` only (mean `[0,0,0]`, std `[1,1,1]`),
  white (255) pad-to-square letterbox + resize. **Observer uses `cv2.resize(..., INTER_CUBIC)`**; lipika's own
  `fontrecog.infer.predictor.letterbox` uses **PIL `Image.BICUBIC`** and returns a torch tensor. This resize-backend
  difference is exactly what strategy A vs strategy C must expose.
- `fontrecog` and `torch` ARE importable in the venv, BUT `fontrecog.infer.predictor.letterbox` returns a
  `torch.Tensor` and importing it pulls torch. Item 2 forbids torch, so strategy C **replicates** lipika's letterbox
  in numpy/PIL (PIL pad-to-square + `Image.BICUBIC` resize + `/255`, CHW) and the report notes it is a faithful
  numpy replica of `fontrecog...letterbox` (source read: convert RGB, pad to `max(w,h)` square with fill=255, resize
  to 224 BICUBIC, `arr/255`, `permute(2,0,1)`). Do NOT import fontrecog/torch in the eval script.
- Labels (14 slots, index order): `bengali, devanagari, gujarati, gurmukhi, kannada, latin, malayalam, other, odia,
  other, arabic, other, tamil, telugu`. Three slots are `other` (indices 7, 9, 11) — collapse all three into a single
  `other` column in the confusion matrix.
- Dataset ground truth: `dataset/verified_twice/<lang>/cropped_images/*.jpeg` are tight WORD crops named
  `{imageid}_{idx}_{quadcoords}.jpeg`. Per-folder crop counts (confirmed): assamese 2097, bengali 3068, gujarati
  1853, hindi 1082, kannada 1078, malayalam 1187, marathi 1629, odia 3569, punjabi 3534, tamil 2462, telugu 1183,
  urdu 1177 — every folder has far more than 30. NO `latin` folder. Ignore `__MACOSX/` and `.DS_Store`.
  LANGUAGE→SCRIPT (authoritative probe label): hindi→devanagari, marathi→devanagari, assamese→bengali,
  bengali→bengali, punjabi→gurmukhi, gujarati→gujarati, kannada→kannada, malayalam→malayalam, odia→odia,
  tamil→tamil, telugu→telugu, urdu→arabic. LATIN crops are generated from the English PDFs in `dataset/`
  (`MY_resume.pdf`, `Vinay_Resume.pdf`, `bert vs sas.pdf`, `missouri_water_sample.pdf`,
  `Chander Kochhar 01_compressed 2.pdf`) via `imaging.render_rgb` + the NEW extractor, labelled script=latin.
- Test layout: `tests/conftest.py` only sets `APP_ENV`; imports use `src....` absolute (works because pytest rootdir
  = repo root). No `test_line_crops.py` and no `scripts/eval_lipika_crops.py` exist yet.

## ABSOLUTE RULES (apply to EVERY item; violating any is a failure)
- NO git commits, NO staging, NO git-LFS, NO deletion/tracking of ANY `.onnx` file. Leave
  `src/ai/layer1_probe/models/lipika_script.onnx` (~112MB fp32) and `lipika_script.int8.onnx` (~28MB) exactly as-is,
  UNSTAGED, untouched.
- Do NOT edit anything under `docker/`, `step_functions/`, or `src/ai/layer1_routing/`. The new extractor lives in
  `src/ai/layer1_probe/`.
- Do NOT restore PP-Structure to the printed engine.
- Do NOT change routing, confidence thresholds, or shipped defaults BEYOND the single flag
  `IDP_PROBE_VLM_FOR_HANDWRITING` (item 4). `decide_route` with NO handwriting signal supplied must stay
  byte-identical to today (latin→paddleocr_printed).
- Ground-truth labels come ONLY from dataset folder structure / PDF origin — NEVER from lipika or VLM output.
- The eval script (item 2) is NOT imported at runtime and uses NO torch and NO fontrecog and NO AWS.
- AGENTS.md AWS-tagging rule does NOT apply: no AWS resource is created/configured/provisioned by any item here.

---

## ITEM 1 — line-crop extractor module + unit tests

- [ ] 1. Create `src/ai/layer1_probe/line_crops.py` with `extract_line_crops`.
      Reuse `imaging.ink_mask`. Algorithm: (1) if `mask is None`, compute `mask = imaging.ink_mask(gray)`; (2) DROP
      ruled lines — detect long thin near-horizontal strokes (morphological open with a wide horizontal kernel, e.g.
      `cv2.getStructuringElement(MORPH_RECT, (max(25, w//25), 1))`, as `cv_features.detect_table` does) and subtract
      them from the mask so a page of only rules yields no line boxes; (3) dilate the de-ruled ink mask HORIZONTALLY
      (wide, short kernel, e.g. width ~ `max(15, w//30)`, height 1..3) to merge words of one text line into one blob;
      (4) find connected-component boxes via `cv2.connectedComponentsWithStats`; (5) keep boxes whose height is in
      `[min_height, max_height_frac*H]` and whose width exceeds a small floor (drop specks and page-tall blobs);
      (6) crop each kept box TIGHT from `gray` (NOT the mask) with `pad` px clamped to image bounds; (7) sort boxes
      top-to-bottom (ties left-to-right) = reading order; return `list[(crop_ndarray, (x,y,w,h))]`.
      Signature: `def extract_line_crops(gray: np.ndarray, mask: Optional[np.ndarray] = None, *, min_height: int = 10,
      max_height_frac: float = 0.5, pad: int = 3, min_width: int = 15) -> list[tuple[np.ndarray, tuple[int,int,int,int]]]`.
      Document the defaults in the docstring. Pure numpy/OpenCV, no torch, no onnxruntime, no fitz import.
      Files: `src/ai/layer1_probe/line_crops.py`
      Verify: `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -c "import src.ai.layer1_probe.line_crops as m; assert hasattr(m,'extract_line_crops')"` prints nothing and exits 0.

- [ ] 2. Create `tests/test_line_crops.py` — synthetic OpenCV/numpy pages, fast.
      Build small synthetic grayscale pages with numpy/`cv2` only (no PDF, no fonts): (a) ruled-line-only page —
      draw several long thin horizontal black lines on white → assert `len(extract_line_crops(gray)) == 0`
      (no line-sized text crops); (b) single-text-line page — draw one thick horizontal bar/blob of text-like height
      in the middle → assert `len(...) == 1` and the returned bbox is within the drawn band; (c) blank white page →
      assert `len(...) == 0`. Keep images small (e.g. 400x300) so it runs in well under a second.
      Files: `tests/test_line_crops.py`
      Verify: `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -m pytest tests/test_line_crops.py -q` — all new tests pass.

## ITEM 2 — offline lipika eval harness (fp32 vs int8 under strategies A/B/C)

- [ ] 3. Create `scripts/eval_lipika_crops.py` (onnxruntime + numpy + OpenCV + PIL + fitz; NO torch, NO fontrecog,
      NOT imported at runtime; guard any `if __name__ == "__main__"`).
      Inputs & manifest: accept an optional `--manifest` CSV (columns `source_path, kind, true_script[, page]`,
      `kind` in `{crop, pdf, image}`); when absent, BUILD the manifest automatically:
      `crop` rows from every `dataset/verified_twice/<lang>/cropped_images/*.jpeg` mapped folder→script via the
      authoritative LANGUAGE→SCRIPT table above (skip `__MACOSX`, `.DS_Store`); `latin` rows from the five English
      PDFs in `dataset/` as `kind=pdf, true_script=latin`. Sample BALANCED: `>=30` crops/script, cap `~150`/script for
      runtime (`--per-script-cap`, default 150; `--min-per-script`, default 30); report ACTUAL counts; list+skip any
      folder/PDF that cannot be mapped. Do NOT use lipika/VLM to label.
      Crop production per kind: `crop` → load the jpeg directly (already a tight word crop); `pdf`/`image` → render
      pages with `imaging.render_rgb(page, 144)` (fitz for pdf; cv2 decode for image), derive
      `gray = imaging.analysis_gray(rgb, 144, cfg.analysis_dpi)` and cut line crops with
      `line_crops.extract_line_crops(gray)` (ITEM 1), taking the corresponding region from the full-res `rgb`.
      Strategies (each produces the model input, letterboxed to 224 square):
        (A) full-line/word crop letterboxed to square via the CURRENT observer path — replicate
            `script_classifier._letterbox` EXACTLY (white pad to `max(h,w)` square, `cv2.resize INTER_CUBIC`, `/255`,
            mean 0 std 1, NCHW).
        (B) square-ish windows cut along each line (slide/tile the crop into windows of aspect 1:1..2:1), then
            letterbox each window as in A; aggregate the windows' softmax by mean before argmax for that crop.
        (C) lipika's OWN documented preprocessing — numpy/PIL replica of `fontrecog.infer.predictor.letterbox`
            (PIL convert RGB, pad to square fill=255, `Image.BICUBIC` resize to 224, `np.array/255`, transpose to
            CHW). Note in the report that fontrecog is NOT imported (torch dep) and this is a faithful replica.
      Models: load BOTH `src/ai/layer1_probe/models/lipika_script.onnx` (fp32) and `lipika_script.int8.onnx` (int8)
      via `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])`; reuse ONE session per model across
      all three strategies. Read labels from `models/lipika_script.labels.json`; COLLAPSE the three `other` slots
      (indices 7,9,11) into a single `other` class for the confusion matrix and accuracy.
      Output per (strategy × model): per-script top-1 accuracy; FULL confusion matrix over the probe script
      vocabulary (rows=true, cols=pred, `other` collapsed); top1−top2 softmax margin distribution (min/median/mean/max
      or a small histogram); page/source-level vote accuracy = argmax of MEAN softmax over that source's crops, with
      the minimum crop count N reported. If a model file is MISSING, warn and skip that model (do not crash);
      if BOTH are missing, exit non-zero with a clear message. Write a markdown report to
      `c:\Users\Dell\Desktop\IDP\engineer_a\.agents\tasks\lipika-eval-results.md` AND print a concise summary to
      stdout (actual per-script counts + headline accuracies).
      Files: `scripts/eval_lipika_crops.py`
      Verify (syntax/import only — the full eval is a long CPU run executed by the human later):
      `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -c "import ast; ast.parse(open(r'scripts/eval_lipika_crops.py',encoding='utf-8').read())"` parses clean AND
      `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -c "import importlib,sys; importlib.import_module('scripts.eval_lipika_crops'); assert 'torch' not in sys.modules and 'fontrecog' not in sys.modules"`
      imports without pulling torch/fontrecog (confirms runtime-safe deps). Optionally a smoke run with
      `--min-per-script 2 --per-script-cap 3` if time permits.

## ITEM 4 — handwriting regression fix behind `IDP_PROBE_VLM_FOR_HANDWRITING` (default '1')

> Items 4–9 RE-ADD a flag-gated handwriting signal end-to-end. Implement them together so the probe tree stays
> importable before running the suite. When the flag is OFF or `handwriting_prob is None`, behavior is byte-identical
> to today.

- [ ] 4. Add config to `src/ai/layer1_probe/types.py`.
      On `ProbeConfig` add two fields: `vlm_for_handwriting: bool = True` and `hw_threshold: float = 0.30`. In
      `from_env` add parsing mirroring existing flags: `c.vlm_for_handwriting = _env("VLM_FOR_HANDWRITING", "1") == "1"`
      and `c.hw_threshold = float(_env("HW_THRESHOLD", str(c.hw_threshold)))`. Re-add `handwriting_prob: Optional[float]
      = None` to `SampleObs`, to `DocPrior`, and to `PageProbe` (default None, Optional — so existing constructions
      stay valid). Keep `ROUTE_HANDWRITTEN` (already present).
      Files: `src/ai/layer1_probe/types.py`

- [ ] 5. Add the flag-gated handwriting branch to `src/ai/layer1_probe/routing.py::decide_route`.
      Add `handwriting_prob: Optional[float] = None` as a new keyword-only param with default None (so all existing
      call sites that omit it are unchanged). Import `ROUTE_HANDWRITTEN` from types. Insert the handwriting branch
      ONLY inside the existing `if script == "latin":` case, BEFORE returning `ROUTE_PRINTED`: if
      `handwriting_prob is not None and handwriting_prob >= cfg.hw_threshold` → return
      `RouteChoice(route=ROUTE_HANDWRITTEN, ocr_lang=SCRIPT_TO_PADDLE_LANG["latin"], use_structure=has_table,
      reason="latin_handwritten")`; otherwise return the current `ROUTE_PRINTED` choice unchanged. No non-Latin path
      gains a handwriting branch. When `handwriting_prob is None` the function is byte-identical to today.
      Files: `src/ai/layer1_probe/routing.py`

- [ ] 6. Re-add handwriting production to `src/ai/layer1_probe/vlm_probe.py`.
      Add `"handwriting_fraction": <0-1, fraction of text that is handwritten>` back into the `_PROMPT` JSON schema.
      In `BedrockScriptProbe.observe`, set `handwriting_prob=_clip01(data.get("handwriting_fraction"), 0.0)` on the
      returned `SampleObs`. `_clip01`/`_norm_script`/`parse_vlm_json` unchanged.
      Files: `src/ai/layer1_probe/vlm_probe.py`

- [ ] 7. Carry handwriting through `src/ai/layer1_probe/aggregate.py`.
      In `aggregate_doc`, compute a document handwriting value from the observations that carry one (e.g. mean of
      `o.handwriting_prob` over `o for o in obs if o.handwriting_prob is not None`, else None) and pass it as
      `DocPrior(handwriting_prob=...)`. Leave `samples_disagree` and `nearest_sample` script-based (do NOT reintroduce
      a handwriting-spread disagreement branch — out of scope).
      Files: `src/ai/layer1_probe/aggregate.py`

- [ ] 8. Wire the flag + handwriting through `src/ai/layer1_probe/probe.py` (Phase B observe + Phase C).
      Phase B `_observe(pn)`: today it calls the VLM only when `obs is None or conf < onnx_min_conf or mixed`. Add the
      flag-gated extra-VLM call: when `cfg.vlm_for_handwriting` AND the ONNX `obs` resolved `script == "latin"` with
      `obs.script_confidence >= cfg.onnx_min_conf` and `not obs.mixed_scripts` (the case that currently SKIPS the VLM)
      AND `vlm is not None and cfg.vlm_enabled`, make ONE VLM call and copy its `handwriting_prob` onto the ONNX obs
      (keep the ONNX script/confidence; only adopt `handwriting_prob` from the VLM). Non-Latin confident scripts and
      the flag-OFF path keep skipping the VLM exactly as today. Guard so flag=0 makes ZERO extra VLM calls.
      Phase C `_page_values` currently returns a 5-tuple `(script, script_confidence, mixed_scripts, source, samp)`;
      extend handwriting carry: resolve `hw = samp.handwriting_prob if (heterogeneous and samp) else prior.handwriting_prob`
      (fall back to `samp.handwriting_prob` when prior has none) and thread it to the `decide_route` calls and the
      `PageProbe(...)` construction. Pass `handwriting_prob=hw` into BOTH `decide_route(...)` call sites (the
      `for pn in nonblank` loop and the `for pn in timed_out` loop) and set `handwriting_prob=hw` on the `PageProbe`.
      Do the minimal plumbing; keep every non-handwriting behavior identical.
      Files: `src/ai/layer1_probe/probe.py`

- [ ] 9. (Handler slim fields) In `src/api/lambda_probe_handler.py`, if `_SLIM_FIELDS` is an explicit allow-list and
      you want `handwriting_prob` surfaced, OPTIONALLY add `"handwriting_prob"`; otherwise leave `_SLIM_FIELDS`
      untouched (not required by the four work items — the routing effect is internal). Only touch this file if the
      slim-output test needs it; do NOT change the four engine-contract keys. (Default: leave unchanged.)
      Files: `src/api/lambda_probe_handler.py` (optional)

- [ ] 10. Extend `tests/test_probe.py` with the flag tests (mock ONNX via injected `session=`, VLM via stub — NO real
      model, NO torch, NO AWS). Re-add an `hw` capability to `StubVLM` so it can return a chosen `handwriting_prob`
      (add an `hw=None` ctor arg and set `handwriting_prob=hw` on the returned `SampleObs`; keep `per_page` working).
      Cases:
      (i) flag=1 (`_cfg(vlm_for_handwriting=True)`), a SAMPLED page whose ONNX obs is confident-latin + a stub VLM
          returning `handwriting_prob=0.9` ⇒ page routes to `paddleocr_handwritten`. Drive this through
          `probe_document` with an injected `onnx` observer (fake session peaking on latin) and the stub VLM, OR a
          focused unit test of `_observe`/`decide_route` — prefer the end-to-end `probe_document` path for fidelity.
      (ii) flag=0 (`_cfg(vlm_for_handwriting=False)`), same inputs ⇒ current behavior: NO extra VLM call for the
          confident-latin page and route is `paddleocr_printed`. Assert the stub VLM was NOT called for that page.
      (iii) a non-Latin confident script (e.g. devanagari) ⇒ never triggers the VLM-for-handwriting path and still
          routes to `vlm_transcribe`.
      (iv) `decide_route(...)` called with NO `handwriting_prob` supplied is byte-identical to today
          (latin, conf .9 ⇒ `paddleocr_printed`, `ocr_lang="en"`); and `decide_route(..., script="latin",
          script_conf=.9, handwriting_prob=0.9, cfg=_cfg())` ⇒ `paddleocr_handwritten`.
      Keep the existing 27 tests green (adjust `StubVLM` additions so current `(script, conf)` tuple usage still
      works — default `hw=None`).
      Files: `tests/test_probe.py`

---

## VERIFICATION (run from repo root; absolute venv python)

1. Unit + integration tests (NOT the slow real-PaddleOCR tests):
   `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -m pytest tests/test_probe.py tests/test_line_crops.py -q`
   Expect: all pass (27 existing + new line-crop tests + new flag tests). Report pass/fail counts.
   Do NOT run `tests/test_layer2_engines_local.py::*_engine_local_path`.

2. Eval script is runtime-safe and parses:
   `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -c "import ast; ast.parse(open(r'scripts/eval_lipika_crops.py',encoding='utf-8').read())"`
   `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -c "import importlib,sys; importlib.import_module('scripts.eval_lipika_crops'); assert 'torch' not in sys.modules and 'fontrecog' not in sys.modules"`

3. No .onnx staged/committed and models untouched:
   `git -C c:\Users\Dell\Desktop\IDP\engineer_a status --short` — the two `.onnx` files remain `??` (untracked) and
   nothing is staged (no `A `/`M ` on any `.onnx`).

4. Forbidden paths untouched:
   `git -C c:\Users\Dell\Desktop\IDP\engineer_a status --short -- docker/ step_functions/ src/ai/layer1_routing/`
   — expect NO output.

5. Routing/thresholds/defaults unchanged except the item-4 flag — confirm by reading the diff of `routing.py`/
   `types.py`: the only new routing branch is the latin+handwriting one gated on `handwriting_prob is not None`.

6. Non-Latin never routes to a handwriting/printed engine — grep the probe routing source:
   `& "c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe" -m pytest tests/test_probe.py -q -k "routing or handwriting or devanagari"` passes, and read `routing.py` to confirm `ROUTE_HANDWRITTEN`/`ROUTE_PRINTED`
   appear ONLY inside the `script == "latin"` branch.

## Gaps / assumptions
- Item 2's full eval is a multi-minute CPU run over thousands of crops; the implementer VERIFIES by parse+import+safe
  smoke (`--min-per-script 2 --per-script-cap 3`), and leaves the full run and the accuracy/strategy recommendation to
  the human (Phase 3 STOP per the user). The report file path is fixed at
  `.agents/tasks/lipika-eval-results.md`.
- Item 3 (labeling policy) has no code task; it is realized by item 2's folder→script mapping and the NO-lipika/VLM
  labeling rule.
- There is no `latin` dataset folder; Latin ground truth comes exclusively from the five English PDFs via the new
  extractor.
