# Implementation Plan — wire lipika into the probe as a LINE-WISE observer (pilot, Strategy B, fp32)

Design decisions grounded in the code I read (`script_classifier.py`, `probe.py`, `inprocess.py`,
`line_crops.py`, `scripts/eval_lipika_crops.py`, `types.py`, `routing.py`, `labels.json`, the two
test files, `imaging.py`, `lambda_probe_handler.py`, `export_lipika_script_onnx.py`,
`.agents/tasks/lipika-eval-results.md`).

## ABSOLUTE RULES (plan constraints — every item must honour these)

- NO `git commit` / `git add` / staging / LFS. Do NOT delete or track ANY `.onnx` file.
  `src/ai/layer1_probe/models/lipika_script.onnx` (fp32, ~112 MB) and `lipika_script.int8.onnx`
  (~28 MB) stay on disk, UNSTAGED. Use the **fp32** model (`labels.json` already says
  `shipped_default: "fp32"`, so `default_model_path()` resolves to the fp32 file — do not change it).
- Do NOT edit anything under `docker/`, `step_functions/`, or `src/ai/layer1_routing/`.
- Routing policy UNCHANGED: non-Latin detected scripts still → `vlm_transcribe`; only `latin` →
  a Paddle engine. Do NOT change `IDP_PROBE_PRINTED_SCRIPTS` defaults, confidence thresholds, or
  `routing.py` decision logic. `routing.py` is READ-ONLY (reading it is fine; it was read).
- Keep `IDP_PROBE_VLM_FOR_HANDWRITING` flag behavior exactly as it is now (the `elif` branch in
  `probe.py::_observe` that adopts only `handwriting_prob` for confident-latin pages).
- Do NOT restore PP-Structure.
- `onnx_min_conf` (default 0.80) default and its use in `probe.py::_observe` are untouched; the new
  lipika gates are an ADDITIONAL early-None path INSIDE the observer, before it ever returns a SampleObs.

## Design decisions (one line of rationale each)

1. **Shared-module home = `src/ai/layer1_probe/windows.py` (NEW).** `line_crops.py` is the ink-mask /
   connected-components line extractor; the letterbox+window preprocessing is a distinct concern and the
   observer (which already imports `line_crops`) plus the eval script both need it. A new `windows.py`
   keeps `line_crops.py` focused and avoids a circular feel. It depends only on `numpy` + `cv2` (NO torch,
   NO onnxruntime, NO fitz) so it is import-safe on the probe hot path.
2. **`_letterbox_A` and `_windows_B` move verbatim** (same names, same math: white-pad to `max(h,w)` square,
   `cv2.resize INTER_CUBIC`, `/255`, NCHW; windows aspect 1:1..2:1 cut along the long axis with ~50% overlap)
   into `windows.py`. The eval script imports them from there — identical results, zero duplication.
3. **Observer keeps its public API** `observe(page_number, gray, mask, *, rgb=None) -> Optional[SampleObs]`
   so both call sites (`probe.py::_observe` → `onnx.observe(pn, gray, mask)`; `inprocess.py` →
   `observe(page_number, gray, mask, rgb=rgb)`) keep working unchanged. Gates are read from a `ProbeConfig`
   the observer holds (new optional `cfg=` ctor param, default `ProbeConfig()`), NOT by changing `observe()`'s
   positional signature — simplest way to thread the 3 gate values without touching call sites.
4. **Gates live on `ProbeConfig`** (`lipika_min_windows`, `lipika_min_conf`, `lipika_min_margin`) with
   `IDP_PROBE_*` overrides in `from_env`, mirroring the existing flags, so Lambda config changes need no redeploy.
5. **Label collapse reuses the existing JSON** (`labels.json`: 14 slots, indices 7 / 9 / 11 = `"other"`),
   argmax index → `labels[idx]`. Mapping source cited inline: `scripts/export_lipika_script_onnx.py::_LIPIKA_TO_PROBE`
   (meetei_mayek / ol_chiki / syloti_nagri → "other"; perso_arabic → "arabic"). No remap table needed —
   the JSON already encodes the collapse.
6. **VLM precedence (CHANGE 4) is realised in `probe.py::_observe`, which is editable** (it is NOT routing.py).
   The existing code already replaces `obs` with the VLM result on the None/low-conf/mixed path; the new work
   is (a) a telemetry event every sampled page and (b) confirming/closing the seam where a confident lipika
   "latin" could survive a contradicting VLM via the handwriting `elif`. See item 7.

---

## Ordered items

- [ ] 1. Create `src/ai/layer1_probe/windows.py` and move `_letterbox_A` + `_windows_B` into it.
      Cut both functions verbatim from `scripts/eval_lipika_crops.py` into a new module
      `src/ai/layer1_probe/windows.py` (module docstring: numpy+cv2 only, no torch/onnxruntime/fitz;
      `letterbox_square(rgb, size=224)` = old `_letterbox_A`, `square_windows(rgb)` = old `_windows_B`).
      Keep a thin backwards-compatible alias inside the eval script: `from src.ai.layer1_probe.windows import
      letterbox_square as _letterbox_A, square_windows as _windows_B` so the rest of
      `eval_lipika_crops.py` (e.g. `crop_softmax`) is unchanged and produces identical tensors.
      Export `_INPUT_SIZE = 224` from `windows.py` and import it in the eval script (it currently defines
      its own; keep the eval value identical). Do NOT touch `_letterbox_C` (strategy C stays in the script).
      Files: `src/ai/layer1_probe/windows.py` (new), `scripts/eval_lipika_crops.py` (import swap only).
      Verify: `.venv\Scripts\python.exe -c "import numpy as np; from src.ai.layer1_probe.windows import letterbox_square, square_windows; a=letterbox_square(np.zeros((40,300,3),np.uint8)); print(a.shape, a.dtype); print(len(square_windows(np.zeros((40,300,3),np.uint8))))"`
      → prints `(1, 3, 224, 224) float32` and a window count `> 1` for a 300x40 (wide) crop.

- [ ] 2. Add the three gate fields to `ProbeConfig` and wire `from_env`.
      In `src/ai/layer1_probe/types.py` add, in the `--- sampling ---` block next to `onnx_min_conf`:
      `lipika_min_windows: int = 8`, `lipika_min_conf: float = 0.60`, `lipika_min_margin: float = 0.20`.
      In `from_env` add: `c.lipika_min_windows = int(_env("LIPIKA_MIN_WINDOWS", str(c.lipika_min_windows)))`,
      `c.lipika_min_conf = float(_env("LIPIKA_MIN_CONF", str(c.lipika_min_conf)))`,
      `c.lipika_min_margin = float(_env("LIPIKA_MIN_MARGIN", str(c.lipika_min_margin)))`.
      Do NOT touch `onnx_min_conf` / `min_script_conf` / `hw_threshold` defaults or any other field.
      Files: `src/ai/layer1_probe/types.py`.
      Verify: `.venv\Scripts\python.exe -c "import os; os.environ['IDP_PROBE_LIPIKA_MIN_WINDOWS']='5'; from src.ai.layer1_probe.types import ProbeConfig; c=ProbeConfig(); print(c.lipika_min_windows,c.lipika_min_conf,c.lipika_min_margin); print(ProbeConfig.from_env().lipika_min_windows)"`
      → prints defaults `8 0.6 0.2` then `5` from the env override.

- [ ] 3. Rewrite the observer in `src/ai/layer1_probe/script_classifier.py` to the LINE-WISE Strategy-B path.
      Keep module singletons (`_get_session`, `_reset_for_tests`, `default_model_path`,
      `default_labels_path`, `_load_meta`, `_softmax`) and the injected-`session=` ctor path (NO onnxruntime
      import when a session is injected). Changes to `LipikaOnnxScriptObserver`:
      (a) ctor gains `cfg: Optional[ProbeConfig] = None` → store `self.cfg = cfg or ProbeConfig()` (import
      `ProbeConfig` from `.types`); preserve `model_path`, `labels_path`, `session` params and `_injected`.
      (b) Rewrite `observe(self, page_number, gray, mask, *, rgb=None)`:
        1. `session, meta, input_name = self._resolve()`; if `session is None or meta is None` → return None.
        2. `if mask is None: mask = imaging.ink_mask(gray)` (import `imaging` from `.`).
        3. `boxes = extract_line_crops(gray, mask)` (import from `.line_crops`); sort by area (`w*h`)
           descending and take at most 24: `boxes = sorted(boxes, key=lambda cb: cb[1][2]*cb[1][3], reverse=True)[:24]`.
        4. Build per-line source RGB exactly like eval `crops_from_rgb_page`: if `rgb` is not None, map each
           box (which is in `gray`/analysis-dpi coords) to full-res rgb using `fx = W_rgb/gw`, `fy = H_rgb/gh`
           (reuse the `_scale_box` math — add a small private helper in the observer or import a shared one;
           prefer a 6-line private `_scale_box` to avoid importing from `scripts/`); else use
           `cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)` per box.
        5. For each line-source RGB call `square_windows(src)` (from `.windows`); flatten into one
           `all_windows` list.
        6. GATE A: `if len(all_windows) < self.cfg.lipika_min_windows: return None`.
        7. `batch = np.concatenate([letterbox_square(w) for w in all_windows], axis=0)` →
           single `session.run(None, {input_name: batch})[0]` → `_softmax` per row → `mean_probs = probs.mean(0)`.
        8. `order = argsort(mean_probs)[::-1]`; `top_idx, top, second` → `margin = top - second`.
        9. GATE B: `if top < self.cfg.lipika_min_conf or margin < self.cfg.lipika_min_margin: return None`.
        10. `labels = meta["labels"]`; OOB guard as today; return
            `SampleObs(page_number=page_number, script=labels[top_idx], script_confidence=round(top,3),
            mixed_scripts=(second > _MIXED_MARGIN), source="onnx")`.
      Keep `_MIXED_MARGIN` for the `mixed_scripts` flag (routing.py still reads `mixed`). Wrap `session.run`
      in the existing try/except → `logger.warning("probe.onnx_run_failed", ...)` → return None.
      Files: `src/ai/layer1_probe/script_classifier.py`.
      Verify: covered by item 8 tests (`.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py -q`).

- [ ] 4. Update the module docstring + add the label-mapping citation comment in the observer.
      Rewrite the top-of-file `MODEL CONTRACT` / "classifies the WHOLE page" prose to describe the line-wise
      Strategy-B path (line crops → square-ish windows → batched forward → page-level mean-softmax). Add,
      next to where `labels = meta["labels"]` is used, a comment: `# 14 output slots; indices 7/9/11 == "other"
      (meetei_mayek/ol_chiki/syloti_nagri). Index→label mapping source: scripts/export_lipika_script_onnx.py::_LIPIKA_TO_PROBE
      and the loopdesk-ai/lipika model card.` Keep the `OnnxScriptClassifier = LipikaOnnxScriptObserver` alias
      (see item 6 — it is still imported).
      Files: `src/ai/layer1_probe/script_classifier.py`.
      Verify: `.venv\Scripts\python.exe -c "import src.ai.layer1_probe.script_classifier"` imports clean (no syntax error).

- [ ] 5. Remove the now-dead whole-page letterbox path from the observer.
      Delete the module-level `_letterbox(rgb, size, mean, std)` function in `script_classifier.py` (the
      whole-page path) — it is no longer called after item 3. Grep first to confirm no other importer:
      `.venv\Scripts\python.exe -c "import ast,sys"` is not needed — use the grep in the Verify line. The
      observer no longer reads `meta["mean"]/["std"]/["input_size"]` for a whole-page letterbox (windows are
      letterboxed via `windows.letterbox_square` at fixed 224); leave the keys in `labels.json` untouched
      (do not edit the JSON). Keep `_softmax` and `_MIXED_MARGIN`.
      Files: `src/ai/layer1_probe/script_classifier.py`.
      Verify: `grep -rn "script_classifier._letterbox\b" src tests scripts` (ripgrep) returns NO matches, and
      item 8's test run is green.

- [ ] 6. Confirm the `OnnxScriptClassifier` alias stays (it is still imported) — no deletion.
      Grep shows live importers: `src/api/lambda_probe_handler.py` (`from ...script_classifier import
      OnnxScriptClassifier`), `tests/test_probe.py`, `tests/test_probe_script_onnx.py` (via
      `LipikaOnnxScriptObserver`). Therefore KEEP `OnnxScriptClassifier = LipikaOnnxScriptObserver`. No code
      change here beyond confirming the alias line survives item 3/4's rewrite. `lambda_probe_handler.py`
      constructs `OnnxScriptClassifier(model_path, labels_path)` with no `cfg=` — the new `cfg` default
      `ProbeConfig()` keeps that call valid (it will use default gates; the handler path is out of pilot
      scope but must not break).
      Files: none (verification-only).
      Verify: `.venv\Scripts\python.exe -c "from src.ai.layer1_probe.script_classifier import OnnxScriptClassifier, LipikaOnnxScriptObserver; print(OnnxScriptClassifier is LipikaOnnxScriptObserver)"`
      → `True`; `.venv\Scripts\python.exe -c "import src.api.lambda_probe_handler"` imports clean.

- [ ] 7. CHANGE 4 — VLM override + `probe.lipika_vs_vlm` telemetry + stats counters in `probe.py::_observe`.
      `probe.py` is editable (NOT routing.py, NOT layer1_routing). In `probe_document`, extend the `stats`
      dict literal with `lipika_resolved: 0, lipika_vlm_agree: 0, lipika_vlm_disagree: 0`. Rework `_observe(pn)`:
      - Capture the lipika guess BEFORE any VLM call: `lip = obs` right after `onnx.observe(...)` (obs may be
        None). Record `lip_script = lip.script if lip else None`, `lip_conf`, `lip_margin` (see note below),
        `n_windows` (see note).
      - Keep the EXISTING control flow verbatim: the `if (obs is None or conf<onnx_min_conf or mixed) and vlm...`
        branch that sets `obs = v`, and the `elif vlm_for_handwriting ... obs.script=="latin" ...` branch that
        adopts ONLY `v.handwriting_prob`. **Do NOT change `onnx_min_conf` usage or the handwriting flag.**
      - CRITICAL SEAM (document + test, item 8d): when the handwriting `elif` fires for a confident lipika
        "latin" and the VLM returns a *non-latin* script, today's code keeps lipika's `script="latin"` (it only
        copies `handwriting_prob`), so the page would route to Paddle — contradicting CHANGE 4. Fix INSIDE
        `_observe` (allowed) with the MINIMAL change: in that `elif` branch, after the VLM call, if
        `v is not None and v.script and v.script != obs.script` then adopt the VLM result for routing
        (`obs = v`) — i.e. a contradicting VLM wins; a corroborating/`latin` VLM keeps lipika's script and the
        handwriting-only behavior is preserved (copy `handwriting_prob` as today). This touches only `_observe`,
        leaves `routing.py`/thresholds/defaults alone, and keeps the `vlm_for_handwriting` flag semantics for
        the agreeing case.
      - Telemetry: for EVERY sampled page, after resolution, emit
        `logger.info("probe.lipika_vs_vlm", page_number=pn, lipika_script=lip_script, lipika_conf=lip_conf,
        margin=lip_margin, n_windows=n_windows, vlm_script=<v.script or None, only if the VLM was called>,
        agree=<bool>)`. Define `agree` = (VLM was called) AND (`vlm_script == lip_script`); if the VLM was not
        called, log `vlm_script=None, agree=None` (document this rule in a comment). No image bytes in the event.
      - Stats: increment `lipika_resolved` when `lip is not None`; when the VLM was called and a `lip_script`
        exists, increment `lipika_vlm_agree` / `lipika_vlm_disagree` by the `agree` bool.
      - `n_windows` / `lip_margin` source: the current `SampleObs` carries neither window count nor margin. Keep
        the SampleObs contract unchanged; instead have the observer expose the last-page window count and margin
        for telemetry WITHOUT changing routing — simplest: add two optional, non-routing attributes set as
        plain instance state on the observer after each `observe()` call (`self.last_n_windows`,
        `self.last_margin`), read by `_observe` via `getattr(onnx, "last_n_windows", None)` /
        `getattr(onnx, "last_margin", None)` so stub/VLM-only tests still work. (Do NOT add fields to the pydantic
        `SampleObs`/`PageProbe` — tests assert those contracts stay minimal.)
      Files: `src/ai/layer1_probe/probe.py`, and `src/ai/layer1_probe/script_classifier.py` (set
      `self.last_n_windows` / `self.last_margin` inside `observe`, including on the early-None gate paths).
      Verify: items 8a–8e (`.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py tests/test_probe.py -q`).

- [ ] 8. Add `tests/test_probe_linewise.py` (new) covering the observer aggregation, gates, and VLM precedence.
      Mock ONNX via an injected fake session (canned logits), mock the VLM via a `StubVLM` (reuse the pattern in
      `tests/test_probe.py`). NO torch, NO AWS, NO real model load. Build a synthetic Latin-ish page with the
      `_text_page`/`_pdf_from_images` helpers (import or copy the minimal helpers from `test_probe.py`; prefer a
      small local `conftest`-free copy to keep the file self-contained, or import from `tests.test_probe`).
      Cases:
      (a) Observer-unit: inject a fake session returning fixed logits peaking the `latin` slot (index 5 in the
          real 14-slot order, or use a 4-label meta like the existing tests). Feed a synthetic wide `gray`+`rgb`
          that yields ≥ `lipika_min_windows` windows → assert `obs.script=="latin"`, and assert the aggregation
          math: `script_confidence` == top share of the MEAN softmax (compute expected from the canned logits),
          and the gate passes. Use a labels JSON written to `tmp_path` (mirror `test_probe_script_onnx.py`).
      (b) Fewer than `lipika_min_windows` total windows → `observe()` returns `None` (e.g. a page producing 1–2
          tiny line boxes, or set `cfg.lipika_min_windows` high via an injected `ProbeConfig`).
      (c) Low margin → None, and low conf → None: inject logits whose mean softmax top-1 is below
          `lipika_min_conf`, and separately a near-tie so `margin < lipika_min_margin`.
      (d) VLM contradicts lipika on a sampled page: run `probe_document` with an injected observer stub that
          returns a confident `latin` SampleObs and a `StubVLM(script="devanagari", conf=0.95)`; assert the
          page `route != "paddleocr_printed"` (it is `vlm_transcribe` via the Phase C routing of the VLM script),
          assert the `probe.lipika_vs_vlm` event fired with `agree=False` (capture via a caplog-style logger
          spy or by monkeypatching `probe.logger.info`), and assert `stats["lipika_vlm_disagree"] >= 1`.
      (e) Confident lipika latin + VLM latin: observer stub `latin` + `StubVLM(script="latin")` →
          page route `paddleocr_printed` (handwriting flag path stays as-is; with `vlm_for_handwriting=True`
          and `hw=None`/below threshold the route is still printed). Assert `stats["lipika_vlm_agree"] >= 1`.
      For (d)/(e) reuse the real `probe_document` + a `FakeObserver`-style stub (like
      `tests/test_pipeline_probe_integration.py::FakeObserver`) that also sets `last_n_windows`/`last_margin`.
      Files: `tests/test_probe_linewise.py` (new).
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py -q` — all new tests pass.

- [ ] 9. Add the opt-in, skippable label-mapping test against the REAL fp32 model (CHANGE 6).
      In `tests/test_probe_linewise.py` (or a clearly separated section), add
      `test_label_mapping_real_fp32_optional`: locate `src/ai/layer1_probe/models/lipika_script.onnx` and
      the fixture crops under `dataset/verified_twice/<lang>/cropped_images/`. `pytest.skip(...)` CLEANLY if the
      model file OR the fixtures are absent. Otherwise build a `LipikaOnnxScriptObserver()` (real module
      singleton, fp32 via `default_model_path()`), and for one known crop per script run it through the
      observer's window+letterbox+batched path (or directly via `windows.letterbox_square` on the crop →
      `session.run`). Map language folder → expected script via the eval's `LANGUAGE_TO_SCRIPT`
      (hindi/marathi→devanagari, assamese→bengali, punjabi→gurmukhi, urdu→arabic, else same name). HARD-assert
      the two high-accuracy scripts from the eval (`bengali` 1.00, `tamil` ~0.92 → assert argmax label equals
      expected); for the rest, log predicted-vs-expected without failing (wiring/label-order sanity, not an
      accuracy gate; small tolerance). Mark `@pytest.mark.slow` or guard behind an env flag so the default
      `pytest` run skips it when the model is absent.
      Files: `tests/test_probe_linewise.py`.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py -k label_mapping -q` →
      either PASSES (model+fixtures present) or reports `skipped` (never errors/fails on absence).

- [ ] 10. Full regression run of the affected probe test suite via the repo venv.
      Run the probe/observer/line-crop tests plus the new file. Do NOT run
      `tests/test_layer2_engines_local.py::*_engine_local_path`.
      Files: none.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_script_onnx.py tests/test_line_crops.py tests/test_pipeline_probe_integration.py tests/test_probe_linewise.py -q`
      → all pass (the pre-existing `OnnxScriptClassifier` contract tests in `test_probe.py` /
      `test_probe_script_onnx.py` still pass against the rewritten observer; if `test_probe_script_onnx.py`'s
      canned-logit tests assumed the whole-page letterbox, update those tests to the line-wise path as part of
      this item — they are tests, editable, and must reflect the new observer).

- [ ] 11. Write the one-off measurement script (NOT a committed test) and run it on 5 real dataset pages.
      Create `scripts/measure_lipika_linewise.py` (developer-only, guarded by `if __name__ == "__main__":`,
      same sys.path bootstrap as `eval_lipika_crops.py`). It: (1) times fp32
      `onnxruntime.InferenceSession(default_model_path())` construction (LOAD time); (2) picks 5 REAL pages
      from `dataset/` (reuse `LATIN_PDFS` + a couple of `verified_twice` PDFs/images, or render the first pages
      of the dataset PDFs via `imaging.render_rgb`), and for each page measures the EXTRA line-wise latency:
      render → `analysis_gray` → `ink_mask` → `extract_line_crops` → `square_windows` → batched
      `session.run` → mean-softmax aggregate. Print per-page timings and the mean, plus the load time.
      This script is a deliverable artifact for the report; it is NOT added to the pytest suite and NOT committed.
      Files: `scripts/measure_lipika_linewise.py` (new, uncommitted).
      Verify: `.venv\Scripts\python.exe scripts/measure_lipika_linewise.py` prints the fp32 load time and
      5 per-page line-wise latencies + mean without error.

## Gaps / assumptions noted (no stall)

- **Telemetry needs `n_windows` and `margin`, which `SampleObs` does not carry.** Assumption (item 7): expose
  them as plain non-pydantic instance attributes on the observer (`last_n_windows`, `last_margin`), read
  defensively via `getattr` in `probe.py`, so the `SampleObs`/`PageProbe` contract-minimality tests in
  `test_probe.py` / `test_probe_script_onnx.py` / `test_pipeline_probe_integration.py` keep passing.
- **The VLM-contradicts-latin seam** currently lets a confident lipika "latin" survive the handwriting `elif`.
  CHANGE 4 requires the VLM to win there; the minimal fix is confined to `probe.py::_observe` (item 7) and does
  not touch `routing.py`, thresholds, defaults, or the flag's agreeing-case behavior.
- **Observer gate threading:** chosen via a `cfg=` ctor arg (default `ProbeConfig()`) rather than extending
  `observe()`'s signature, so neither `probe.py` nor `inprocess.py` call sites change. For the pilot, `probe.py`
  does not currently pass a cfg into the injected `onnx` observer (the handler builds it); the module-singleton
  path in `inprocess.py` can pass `ProbeConfig.from_env()` — add that one-line `cfg=` at
  `inprocess._get_observer()` construction so live runs honour `IDP_PROBE_LIPIKA_*`. (Editing `inprocess.py` is
  allowed; it is not under the forbidden paths.)
- **fp32 selection** is automatic: `labels.json` has `shipped_default: "fp32"`, so `default_model_path()`
  already returns the fp32 `.onnx`. No model-path edit, no `.onnx` touch.

---

## VERIFICATION NOTE (first iteration — implemented + measured)

### Per-file summary of what changed
- `src/ai/layer1_probe/windows.py` (NEW) — shared Strategy-B preprocessing:
  `letterbox_square` (= old `_letterbox_A`) and `square_windows` (= old `_windows_B`), plus
  `_INPUT_SIZE = 224`. numpy + cv2 only (no torch/onnxruntime/fitz). CHANGE 1.
- `scripts/eval_lipika_crops.py` — removed the in-script `_letterbox_A` / `_windows_B` / `_INPUT_SIZE`;
  now imports them from `windows.py` (identical tensors, zero duplication). Strategy C stays local.
  Stale `script_classifier._letterbox` doc references updated to `windows.letterbox_square`. CHANGE 1.
- `src/ai/layer1_probe/types.py` — added `ProbeConfig.lipika_min_windows=8`, `lipika_min_conf=0.60`,
  `lipika_min_margin=0.20` with `IDP_PROBE_LIPIKA_*` overrides in `from_env`. `onnx_min_conf` untouched.
  CHANGE 3.
- `src/ai/layer1_probe/script_classifier.py` — REWRITTEN to the line-wise Strategy-B path: line crops
  (largest first, cap 24) -> per-line RGB (bbox->rgb when rgb given, else gray->RGB) -> `square_windows`
  -> single batched `session.run` -> MEAN softmax -> argmax/conf/margin -> gates (min windows / min conf /
  min margin) -> `SampleObs`. Injected-session ctor path and module singletons preserved. New `cfg=` ctor
  arg (default `ProbeConfig()`). Non-routing telemetry attrs `last_n_windows` / `last_margin` set every
  call. Old whole-page `_letterbox` removed. Label-mapping citation comment added. `OnnxScriptClassifier`
  alias kept (still imported by `lambda_probe_handler.py`). CHANGES 2, 5, 6.
- `src/ai/layer1_probe/probe.py` — `_observe` now captures lipika's pre-VLM guess, keeps the existing
  `onnx_min_conf` / handwriting-flag control flow, adds the CRITICAL SEAM (a contradicting VLM wins for a
  confident-latin page so it does NOT route to Paddle), emits `probe.lipika_vs_vlm` per sampled page, and
  increments `lipika_resolved` / `lipika_vlm_agree` / `lipika_vlm_disagree` stats. CHANGE 4.
- `src/ai/layer1_probe/inprocess.py` — one-line `cfg=ProbeConfig.from_env()` so live runs honour the gates.
- `tests/test_probe_linewise.py` (NEW) — observer aggregation, gate (too-few-windows / low-conf /
  low-margin), VLM-contradicts-latin (route = vlm_transcribe, agree=False, disagree counter), confident
  agreement (route = paddleocr_printed, agree counter), and the opt-in skippable real-fp32 label-mapping test.
- `tests/test_probe_script_onnx.py`, `tests/test_probe.py` — updated the canned-logit fake sessions to
  return one row PER window in the batch, and feed a wide synthetic text page so the line-wise path produces
  windows (the tests previously fed 40x80 whole-page inputs for the removed whole-page path).
- `scripts/measure_lipika_linewise.py` (NEW, uncommitted artifact) — load-time + per-page line-wise latency.

### Test counts (repo venv, `-p no:cacheprovider`)
Command: `pytest tests/test_probe.py tests/test_probe_script_onnx.py tests/test_line_crops.py
tests/test_pipeline_probe_integration.py tests/test_probe_linewise.py`
Result: **52 passed, 2 failed**.
- The 2 failures are PRE-EXISTING on the committed baseline (verified by stashing my edits and re-running):
  `test_probe_script_onnx.py::test_no_handwriting_field_on_contracts` and
  `test_pipeline_probe_integration.py::test_contract_fields_removed_and_signature_script_only`. Both assert
  `"handwriting_prob" not in SampleObs.model_fields`, but `SampleObs.handwriting_prob` was ALREADY added by
  the earlier handwriting-flag commit (705fe37) — unrelated to this line-wise change and outside its scope
  (the absolute rules forbid changing the handwriting-flag behavior/contract). Not touched.
- Opt-in label test `tests/test_probe_linewise.py -k label_mapping` against the REAL fp32 model: **PASSES**.
  Index->label order is correct — 11/12 dataset languages predict the expected script from a single crop
  (bengali/tamil hard-asserted; telugu's single crop mispredicts, logged under the soft tolerance as designed).

### Measurements (this CPU; fp32 `lipika_script.onnx`, ORT threads=2)
- **Model LOAD time:** ~0.33–0.37 s (InferenceSession construction, measured twice: 370 ms / 328 ms).
- **Per-window inference throughput:** ~105–133 ms/window (batches 8/16/32/64 -> 112/105/114/133 ms/window).
- **Per-page preprocessing** (render@144dpi -> analysis gray -> ink mask -> line crops -> square windows ->
  letterbox), 5 real dataset pages:

  | page | windows | prep ms |
  | --- | ---: | ---: |
  | MY_resume.pdf | 593 | 382 |
  | Vinay_Resume.pdf | 567 | 297 |
  | bert vs sas.pdf | 278 | 199 |
  | missouri_water_sample.pdf | 365 | 233 |
  | Chander Kochhar 01…pdf | 210 | 238 |
  | **mean** | **403** | **270** |

- **Extra per-page line-wise latency** = prep + batched forward. At ~110 ms/window and a mean of 403
  windows/page, the batched forward dominates at roughly **22–66 s/page** (≈44 s at the mean), i.e. total
  extra latency ≈ 0.27 s prep + ~44 s inference per page on this CPU.

### Flags / things that look awkward (NO behavior changed beyond the task)
- **Window explosion / latency.** At the probe's render DPI (144) each line crop maps to a large full-res
  RGB strip, so `square_windows` emits 200–600 windows/page. On fp32 CPU that is ~20–66 s/page of inference —
  far above a practical per-page budget. This is a direct consequence of Strategy B as validated; I did NOT
  change the window strategy, DPI, or model (int8 and/or a window cap would cut this, but both are
  product/threshold decisions outside this task). Recorded here for the owner's go/no-go.
- The 2 pre-existing `handwriting_prob`-contract failures (above) predate this work and were left untouched.
