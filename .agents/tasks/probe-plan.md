# Implementation Plan — Probe/lipika In-Process Integration (LOCAL build, NO AWS)

Authoritative design: `.agents/tasks/probe-lipika-design.md` (+ review `.agents/tasks/probe-lipika-design-review.md`).
This plan sequences that design; it does NOT re-decide architecture. Every file path is absolute under
`c:\Users\Dell\Desktop\IDP\engineer_a`. All python/pytest run via the repo venv
`c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe` (confirmed present, Python 3.11.9).

## Hard constraints (apply to every item)
- **NO AWS actions of any kind.** No deploy, no ECR push, no `boto3` calls to live services, no
  Lambda/ECS/state-machine create or update. Do NOT run any `deploy_*.py`. Tests mock boto3.
- Work on the CURRENT branch in the repo. NO git worktree.
- Keep the `docker/paddle_handwritten` engine code and the `paddleocr_handwritten` engine — only the
  probe's *routing to* handwriting is removed; the engine becomes unreachable but is NOT deleted.
- Keep `detect_table` / `ink_ratio` / blank logic in `cv_features.py`.
- Keep the probe OUTPUT dict engine-key contract on `PageProbe`:
  `image_s3_bucket`, `image_s3_key`, `ocr_lang`, `use_structure` — only handwriting fields are removed.
- **AGENTS.md AWS tagging rule is NOT relevant here.** No AWS resource is created, configured, or
  provisioned by any step in this plan, so the three mandatory tags (`createdby`/`purpose`/`customer`)
  do not apply to any action. If a *separate, later* step ever provisions AWS, the tags apply then.

## Blocked-vs-unblocked under the STEP-1 STOP-AND-ASK fallback
- **STEP 1 (offline ONNX export)** is the ONLY step that can be *blocked* — if lipika/fontrecog cannot
  be installed/imported or the script head cannot be isolated/traced. In that case: send a
  `send_message` severity `warning` describing exactly what failed, **do NOT fabricate or substitute a
  model**, write nothing to `src/ai/layer1_probe/models/` (or record `"shipped_default":"none"`), and
  continue.
- **STEPS 2, 3, 4, 5, 6 remain UNBLOCKED even if STEP 1 is blocked.** Every observer-dependent test
  injects a fake `InferenceSession` via the preserved `session=` constructor arg, never the runtime
  `_get_session()` path, so routing + wiring + tests ship regardless. With no model file at runtime the
  observer's `_get_session()` returns `None`, `observe()` returns `None`, and the in-process hook routes
  the page to `vlm_transcribe` via `decide_route(script="unknown", ...)` — the designed degraded
  behavior. The runtime observer module, singleton structure, and in-process hook are all implemented
  in STEP 2/STEP 4 whether or not the real model exists.

---

## STEP 1 — Offline ONNX export (dev-only; torch/fontrecog OFFLINE ONLY)

- [ ] 1. Create the offline export script `scripts/export_lipika_script_onnx.py`.
      Developer-run only; NOT imported at runtime and NOT run by CI. Guard ALL heavy imports
      (`import torch`, `import fontrecog`, `transformers`, `onnx`, quantization) INSIDE `main()` so
      merely importing the module never imports torch. The script:
      (a) loads `loopdesk-ai/lipika` via `from fontrecog.infer.predictor import Predictor;
      Predictor.from_pretrained("loopdesk-ai/lipika")`;
      (b) introspects the loaded `torch.nn.Module` to isolate the **ConvNeXtV2-Tiny backbone + the
      SCRIPT head only** (drop family/weight/italic heads + retrieval index) and wraps them in a thin
      `nn.Module` whose `forward(pixel_values)` returns only `script_logits`;
      (c) reads lipika's real input size + RGB mean/std from its processor/config (do NOT hard-code;
      ConvNeXtV2-Tiny is typically 224×224 but read the actual value);
      (d) exports FP32 via `torch.onnx.export` with input name `pixel_values` (dynamic batch axis 0),
      output name `script_logits`, **opset 17**;
      (e) runs `quantize_dynamic` to INT8 and saves the INT8 variant **regardless** of the validation
      outcome;
      (f) uses **Option A single-forward** preprocessing (design §2 decision); if Option A fails
      validation vs torch, fall back to patch-voting in the observer and record
      `"preprocessing":"patch_vote"`.
      Files: `scripts/export_lipika_script_onnx.py`
      Verify: `.venv\Scripts\python.exe -c "import ast,sys; ast.parse(open(r'scripts/export_lipika_script_onnx.py',encoding='utf-8').read())"` parses clean; `.venv\Scripts\python.exe -c "import importlib,sys; sys.modules.get('torch') is None"` (sanity: the script's heavy imports are not module-level — manual inspection that `import torch` sits inside `main()`).

- [ ] 2. Run the export + validation + quantization report via the repo venv.
      Validate on the four images (confirmed present):
      `docs/Screenshot 2026-10-23 140614.png`→latin, `...140632.png`→latin, `...142509.png`→telugu,
      `...142632.png`→devanagari. For each image print `predicted_script, confidence` for
      **(1) torch `predictor.predict`, (2) FP32 ONNX, (3) INT8 ONNX** side by side. A missing image →
      warn and skip it; exit non-zero only if zero images could be validated.
      Write the three model artifacts to `src/ai/layer1_probe/models/`:
      `lipika_script.onnx` (FP32), `lipika_script.int8.onnx` (INT8), and `lipika_script.labels.json`.
      The `labels.json` metadata is:
      `{"labels":[...probe-vocab names...],"input_size":[H,W],"mean":[...],"std":[...],
      "preprocessing":"single_forward|patch_vote","shipped_default":"fp32|int8","source_model":
      "loopdesk-ai/lipika","exported_at":"..."}`. The `labels` array maps **script-index → probe name,
      already remapped per design §4**: Perso-Arabic→`arabic`; Meetei Mayek→`other`; Ol Chiki→`other`;
      everything else = direct lowercase match (latin, devanagari, bengali, gujarati, gurmukhi,
      kannada, malayalam, odia, tamil, telugu). Set `shipped_default="int8"` ONLY if INT8 preserves all
      4 predictions within tolerance; **if INT8 flips any of the 4, keep `fp32` as the shipped default,
      say so in stdout, and still save the INT8 file.**
      **STOP-AND-ASK (send_message `warning`, do NOT fabricate a model) if lipika/fontrecog can't be
      installed/imported or the head export is infeasible.** In that case write nothing (or record
      `"shipped_default":"none"`) and proceed to STEP 2+ via the injectable fake-session mechanism.
      Files: `src/ai/layer1_probe/models/lipika_script.onnx`, `.int8.onnx`, `.labels.json`
      Verify: `.venv\Scripts\python.exe scripts/export_lipika_script_onnx.py` exits 0 and prints the
      torch/FP32/INT8 table with the 4 expected labels; the three artifacts exist (or, under
      STOP-AND-ASK, the warning is sent and the models dir is intentionally empty).

---

## STEP 2 — lipika-ONNX observer (`script_classifier.py`; onnxruntime+numpy+OpenCV, NO torch)

- [ ] 3. Rewrite `src/ai/layer1_probe/script_classifier.py` as the whole-image lipika observer.
      Replace `OnnxScriptClassifier` with class `LipikaOnnxScriptObserver`, and keep a module-level
      alias `OnnxScriptClassifier = LipikaOnnxScriptObserver` (so `lambda_probe_handler.py` and existing
      imports keep working). **Remove** the old line-crop code: `extract_line_crops`, `_prep`, the
      `IN_H, IN_W = 48, 320` contract, `_parse` of `"<script>_<mode>"` labels, and `predict`.
      Implementation details (design §3):
      - Module-level singletons built ONCE: `_SESSION`, `_META`, `_INPUT_NAME` (all `None` initially).
      - Named module constant `_MIXED_MARGIN = 0.25` with a comment that it is a provisional carry-over
        to be re-tuned on real lipika logits (only pushes borderline pages to the VLM — safe direction).
      - `_HERE = os.path.dirname(os.path.abspath(__file__))`; module-level helpers
        `default_model_path()` (returns `models/lipika_script.int8.onnx` iff the sidecar metadata
        `shipped_default == "int8"`, else `models/lipika_script.onnx`, resolved module-relative) and
        `default_labels_path()` (returns `models/lipika_script.labels.json`).
      - `_get_session()` lazily builds/caches the singleton from `os.environ.get("IDP_PROBE_ONNX_MODEL")`
        else `default_model_path()` (labels: `IDP_PROBE_ONNX_LABELS` else `default_labels_path()`);
        returns `None` when the model file is missing (log once WARNING `probe.onnx_model_missing`);
        on `InferenceSession` construction failure log once ERROR and cache a sentinel so it is not
        retried per call.
      - `__init__(self, model_path, labels_path, session=None)` — **keep the `session=None` kwarg
        verbatim** (tests inject a fake session). When `session` is provided, use it as-is and do NOT
        import onnxruntime; load labels/meta from `labels_path`.
      - Add a `_reset_for_tests()` hook that clears `_SESSION`/`_META`/`_INPUT_NAME` so the
        "constructed once" assertion and per-test fake sessions do not leak across tests.
      - `observe(self, page_number, gray, mask, *, rgb=None) -> Optional[SampleObs]`:
        choose a 3-channel image — use `rgb` when provided, else `cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)`;
        resize to metadata `input_size` with lipika mean/std normalization (downscaled fixed square,
        NOT full-res); `logits = session.run(None, {input_name: x})[0]`; softmax → top-1 index + prob;
        map index→name via `_META["labels"]`; `mixed_scripts = second_prob > _MIXED_MARGIN`; return
        `SampleObs(page_number=pn, script=<mapped>, script_confidence=<top>, mixed_scripts=..., source="onnx")`
        with **NO `handwriting_prob`**. On out-of-range index → `script="unknown", confidence=0.0`
        (WARNING). On `session.run` error for a page → WARNING with `page_number`, return `None`.
      Depends on: item 3 is independent of STEP 1's artifacts at import time; it only loads a model at
      runtime via `_get_session()` (absent-model path is the §9 fallback).
      Files: `src/ai/layer1_probe/script_classifier.py`
      Verify: `.venv\Scripts\python.exe -c "from src.ai.layer1_probe.script_classifier import LipikaOnnxScriptObserver, OnnxScriptClassifier, default_model_path, default_labels_path; assert OnnxScriptClassifier is LipikaOnnxScriptObserver"` (and full suite later in VERIFY).

- [ ] 4. Make the Step-Functions handler fall back to the observer's module-relative defaults.
      In `src/api/lambda_probe_handler.py::_load_models`, before the existence check, compute
      `model_path = cfg.onnx_model_path or script_classifier.default_model_path()` and
      `labels_path = cfg.onnx_labels_path or script_classifier.default_labels_path()`
      (import `from src.ai.layer1_probe import script_classifier`), then guard on
      `if os.path.exists(model_path):` and construct `OnnxScriptClassifier(model_path, labels_path)`.
      This loads the baked model with zero env-var config on both the SFN and in-process paths; under
      the §9 no-model fallback `os.path.exists` is False so `_onnx` stays `None` (VLM-only, unchanged).
      Also part of STEP 3: remove `"handwriting_prob"` from `_SLIM_FIELDS` (item 11).
      Files: `src/api/lambda_probe_handler.py`
      Verify: covered by `tests/test_probe.py::test_handler_end_to_end_slim_output_and_report` in VERIFY.

---

## STEP 3 — Script-only routing + schema teardown (must land atomically)

> Items 5–11 remove `handwriting_prob` and friends across the probe module. They are ordered so the
> tree is importable after the group completes; implement them together before running the suite.

- [ ] 5. Rewrite `src/ai/layer1_probe/routing.py::decide_route` to the script-only signature.
      New top-of-file import block (drop `ROUTE_HANDWRITTEN`, keep the rest):
      `from src.ai.layer1_probe.types import (ProbeConfig, RouteChoice, SCRIPT_TO_PADDLE_LANG,
      ROUTE_PRINTED, ROUTE_SKIP, ROUTE_VLM,)`.
      New signature and body (design §5, verbatim):
      `def decide_route(*, script, script_conf, mixed, is_blank, has_table, cfg) -> RouteChoice:`
      order: (1) `is_blank`→`ROUTE_SKIP` reason `"blank_page"`; (2)
      `script in ("unknown","other") or script_conf < cfg.min_script_conf`→`ROUTE_VLM` reason
      `f"script_unresolved({script},{script_conf:.2f})"`; (3) `mixed`→`ROUTE_VLM` reason
      `"mixed_scripts"`; (4) `script == "latin"`→`ROUTE_PRINTED`, `ocr_lang=SCRIPT_TO_PADDLE_LANG["latin"]`,
      `use_structure=has_table`, reason `"latin_print"`; (5) else→`ROUTE_VLM` reason
      `f"{script}_to_vlm_no_dedicated_ocr"` with a code comment that per-script OCR is a future goal.
      No handwriting branch, no `ROUTE_HANDWRITTEN`, no `cfg.printed_scripts`/`cfg.hw_threshold`.
      Files: `src/ai/layer1_probe/routing.py`

- [ ] 6. Strip handwriting + retired config from `src/ai/layer1_probe/types.py`.
      `SampleObs`: remove `handwriting_prob`. `DocPrior`: remove `handwriting_prob`. `PageProbe`:
      remove `handwriting_prob` and `hw_source`. `ProbeConfig`: remove the `hw_threshold` field and the
      `printed_scripts` field, and remove their `from_env` parsing lines (`HW_THRESHOLD`,
      `PRINTED_SCRIPTS`); drop the now-unused `Set`/`field` imports only if they become unused. Keep
      `min_script_conf`, `onnx_min_conf`, `onnx_model_path`, `onnx_labels_path`, VLM knobs, rendering/
      blank/table knobs, `SCRIPT_TO_PADDLE_LANG`, `SCRIPTS`, `RouteChoice`, and **keep the
      `ROUTE_HANDWRITTEN` constant defined** (engine + `_ROUTE_TO_ENGINE` still reference the engine).
      Files: `src/ai/layer1_probe/types.py`

- [ ] 7. Update `src/ai/layer1_probe/probe.py` for the script-only contract (Phase A/C).
      Phase A: remove `hw_h = ... cv_features.handwriting_heuristic(mask)` and drop `hw_h` from the
      `feats` dict. `_observe`: the final fallback becomes
      `obs = SampleObs(page_number=pn, source="none")` (no `handwriting_prob=`). `_page_values`: return
      the 5-tuple `(script, script_confidence, mixed_scripts, source, samp)` (drop the handwriting
      element and both `samp.handwriting_prob`/`prior.handwriting_prob` reads). Update BOTH
      `decide_route(...)` call sites to drop `hw_prob=hw` and match the new keyword-only signature —
      (1) the Phase C `for pn in nonblank:` loop, (2) the `for pn in timed_out:` loop. In both loops
      also drop `handwriting_prob=`/`hw_source=` from the `PageProbe(...)` construction and remove the
      `hw`/`hw_source`/`hw_h` plumbing. Update the unpacking of `_page_values` at both call sites to
      the new 5-tuple.
      Files: `src/ai/layer1_probe/probe.py`

- [ ] 8. Update `src/ai/layer1_probe/aggregate.py` to script-only disagreement/prior.
      `samples_disagree`: drop the `hw_gap` param + the handwriting-spread branch (return True only when
      more than one distinct confident script). `aggregate_doc`: drop the `hws` averaging and the
      `DocPrior(handwriting_prob=...)` kwarg. `nearest_sample`: `usable` filter becomes
      `[s for s in samples if s.script != "unknown"]` (drop the `or s.handwriting_prob is not None`).
      Files: `src/ai/layer1_probe/aggregate.py`

- [ ] 9. Remove the handwriting heuristic from `src/ai/layer1_probe/cv_features.py`.
      Delete `handwriting_heuristic`, `line_periodicity`, and the module constants
      `HW_PERIODICITY_TYPED`/`HW_PERIODICITY_RANGE`. Keep `ink_ratio`, `detect_table`, and the blank
      logic. Update the module docstring to drop the handwriting-heuristic bullet.
      Files: `src/ai/layer1_probe/cv_features.py`

- [ ] 10. Stop populating handwriting in `src/ai/layer1_probe/vlm_probe.py`.
      In `BedrockScriptProbe.observe`, drop the `handwriting_prob=_clip01(data.get("handwriting_fraction"),0.0)`
      kwarg from the `SampleObs(...)` construction. Remove the `handwriting_fraction` key from the
      `_PROMPT` JSON schema. `_norm_script` and `_clip01` stay. (Per review NIT 4, no `_norm_script`
      alias change is needed: a VLM returning `arabic` already maps; `perso-arabic`→`other`→VLM is
      route-equivalent. The Perso-Arabic→arabic remap lives in the ONNX `labels.json` only.)
      Files: `src/ai/layer1_probe/vlm_probe.py`

- [ ] 11. Remove `"handwriting_prob"` from `_SLIM_FIELDS` in `src/api/lambda_probe_handler.py`.
      (The two-line default-path fallback in the same file was done in item 4.) `_SLIM_FIELDS` keeps
      the four engine-contract keys and the rest; only `"handwriting_prob"` is dropped.
      Files: `src/api/lambda_probe_handler.py`

- [ ] 12. Delete the offline calibrator `scripts/calibrate_probe_heuristic.py`.
      It calls the now-removed `cv_features.line_periodicity`; it is non-runtime, not imported by
      runtime or tests. **Leave `.scratch_probe/run_probe_signals.py` and `run_probe_signals2.py`
      UNTOUCHED** (scratch dir, not collected/imported/shipped — they are the documented expected grep
      hits for `handwriting_heuristic`/`line_periodicity`).
      Files: delete `scripts/calibrate_probe_heuristic.py`

- [ ] 13. RE-GREP the probe source path and confirm the teardown is complete.
      Run the grep check (see VERIFY) for `handwriting_prob`, `hw_prob`, `hw_source`,
      `handwriting_heuristic`, `line_periodicity`, `hw_threshold`, `HW_THRESHOLD`, `printed_scripts`,
      `PRINTED_SCRIPTS` over `src/`. Expected remaining hits: NONE in the probe runtime path
      (`.scratch_probe/*` is outside `src/` and is intentionally untouched). Fix any stray hit before
      proceeding.
      Verify: see the grep block in VERIFY — zero matches under `src/ai/layer1_probe/` and
      `src/api/lambda_probe_handler.py`.

---

## STEP 4 — Wire the probe into the live in-process pipeline (default-OFF flag)

- [ ] 14. Create `src/ai/layer1_probe/inprocess.py` — the single integration surface.
      Imports ONLY `imaging`, `cv_features`, `routing`, and the observer (`script_classifier`); it must
      NOT import `fitz`/`probe.py` (keep PyMuPDF off the hot path). Export
      `probe_page_route(page_context, page_number, profile, cfg, *, render_dpi=None) -> Optional[RouteChoice]`:
      `render_dpi = render_dpi or cfg.page_image_dpi`; `rgb = page_context.get("image_array")`; if
      `rgb is None` return `None`. Derive `gray = imaging.analysis_gray(rgb, render_dpi, cfg.analysis_dpi)`,
      `mask = imaging.ink_mask(gray)`. `blank = cv_features.ink_ratio(mask) < cfg.blank_ink_ratio`;
      `has_table,_ = cv_features.detect_table(mask, cfg.table_min_h_lines, cfg.table_min_v_lines,
      cfg.table_min_intersections)`. Build/reuse the observer singleton and call
      `obs = observer.observe(page_number, gray, mask, rgb=rgb)` (pass real RGB). **No in-process VLM
      observer call** — when `obs is None`/low-conf/mixed, let `decide_route(script="unknown", ...)`
      send the page to `vlm_transcribe`. Return
      `decide_route(script=(obs.script if obs else "unknown"),
      script_conf=(obs.script_confidence if obs else 0.0),
      mixed=(obs.mixed_scripts if obs else False), is_blank=blank, has_table=has_table, cfg=cfg)`.
      Depends on: items 3 (observer), 5 (routing), 6 (types).
      Files: `src/ai/layer1_probe/inprocess.py`
      Verify: `.venv\Scripts\python.exe -c "import src.ai.layer1_probe.inprocess as m; assert not any('fitz' in n for n in __import__('sys').modules)"` imports clean without pulling fitz.

- [ ] 15. Add the probe hook to `src/ai/layer1_routing/pipeline.py::process_page`.
      Add module-level helpers:
      `def _probe_enabled() -> bool: return (os.getenv("IDP_ENABLE_PROBE") or "").strip().lower() in ("1","true","yes","on")`
      (add `import os` at top);
      `def _should_probe(profile, primary_engine) -> bool:` — `return False` if
      `primary_engine in ("docling","skip")`, else
      `return (profile.is_scanned or not profile.has_text or profile.primary_script == "unknown")`
      using **direct attribute access** (NOT `getattr`); and
      `def _maybe_probe_route(profile, page_context, primary_engine, page_number) -> Optional[RouteChoice]:`
      that returns `None` immediately if `not _probe_enabled()` (BEFORE importing `layer1_probe`),
      returns `None` if `not _should_probe(profile, primary_engine)`, else lazily
      `from src.ai.layer1_probe.inprocess import probe_page_route` and
      `from src.ai.layer1_probe.types import ProbeConfig`, computes
      `cfg = ProbeConfig.from_env()`, `render_dpi = page_context.get("render_dpi", cfg.page_image_dpi)`,
      and returns `probe_page_route(page_context, page_number, profile, cfg, render_dpi=render_dpi)`.
      Wrap the body in `try/except Exception` → log WARNING `pipeline.probe_hook_failed` + return `None`.
      Hook call: immediately after `primary_task = plan[0] if plan else None` and BEFORE
      `metadata.set_routing(...)` / `_run_plan(...)`:
      ```
      if primary_task is not None:
          _choice = _maybe_probe_route(profile, page_context, primary_task.engine, page_number)
          if _choice is not None:
              primary_task = EngineTask(engine=_choice.route, priority=1, reason=f"probe:{_choice.reason}")
              plan = [primary_task]
              if _choice.ocr_lang:
                  page_context["ocr_lang"] = _choice.ocr_lang
      ```
      Assign `_choice.route` **directly** to the engine (it is already an engine name —
      `paddleocr_printed`/`vlm_transcribe`/`skip`); do NOT route through `_ROUTE_TO_ENGINE`.
      In `_run_engine_task`, on the `paddleocr_printed` branch ONLY, pass
      `ocr_lang=page_context.get("ocr_lang")` into `convert_scanned_page(...)`.
      Depends on: items 14, 16.
      Files: `src/ai/layer1_routing/pipeline.py`

- [ ] 16. Thread `ocr_lang` through `src/ai/layer2_conversion/scanned.py`.
      Add `ocr_lang: Optional[str] = None` to `_invoke_paddle_lambda`, `_ocr_convert`,
      `convert_scanned_page`, and `convert_handwritten_via_paddle`. In `_invoke_paddle_lambda` build
      `payload = {"image_base64": b64_str, "page_number": page_number}` then
      `if ocr_lang: payload["ocr_lang"] = ocr_lang` (so the payload is **byte-identical** to today when
      `ocr_lang` is falsy). `_ocr_convert` must forward `ocr_lang=ocr_lang` to **all THREE**
      `_invoke_paddle_lambda(...)` call sites: the prefer-lambda primary, the
      `except (ImportError, ModuleNotFoundError)` fallback, and the broad `except Exception` fallback.
      Files: `src/ai/layer2_conversion/scanned.py`

- [ ] 17. Stash the real render DPI in the web page context.
      In `schema_chatbot_v2/app/api/pipeline_routes.py::_build_pages_for_pdf`, add
      `"render_dpi": RENDER_DPI` to the per-page `"context"` dict (alongside `pdf_path`/`image_array`/
      `image_bytes`). `RENDER_DPI = 150` already lives in this file — single source of truth; do NOT
      import it into core `pipeline.py` (that back-import would be a cycle). `pipeline.py` reads
      `page_context.get("render_dpi", cfg.page_image_dpi)` (item 15).
      Files: `schema_chatbot_v2/app/api/pipeline_routes.py`

---

## STEP 5 — Packaging (EDIT ONLY — no AWS build/push)

- [ ] 18. Edit `Dockerfile.probe`.
      Remove the `IDP_PROBE_PRINTED_SCRIPTS=latin` line from the `ENV` block (field retired in item 6);
      keep `IDP_PROBE_PAGE_IMAGE_DPI=144` and `PYTHONPATH=/var/task`. Replace the stale
      `RUN mkdir -p /opt/probe_models` + commented `# COPY models/probe/ /opt/probe_models/` lines with
      a short comment: the lipika ONNX ships via the existing `COPY src/ /var/task/src/` at the
      module-relative path `src/ai/layer1_probe/models/`, and `.dockerignore` passes `*.onnx`/`*.json`.
      Add a comment that `IDP_PROBE_ONNX_MODEL`/`IDP_PROBE_ONNX_LABELS` are optional overrides, not set
      by default. Do NOT add a second `COPY` for the models dir.
      Files: `Dockerfile.probe`

- [ ] 19. Reword the `onnxruntime` comment in `requirements-probe.txt` (NIT 6; branch-accurate).
      Change the `onnxruntime>=1.18.0  # only used if IDP_PROBE_ONNX_MODEL is set; ...` comment to
      something like `# lipika script observer; model baked at src/ai/layer1_probe/models/ when present,
      VLM-only fallback when absent`. Add NO new package (onnxruntime/numpy/opencv-python-headless
      already present; Pillow NOT needed on the probe runtime path). Ensure no torch/transformers/
      fontrecog appear.
      Files: `requirements-probe.txt`

- [ ] 20. Add in-process runtime deps to the Layer-3 app image.
      In `schema_chatbot_v2/requirements.txt` add `onnxruntime>=1.18.0` and
      `opencv-python-headless>=4.9.0` (both currently absent; the probe hook imports `cv2` via
      `imaging`/`cv_features`). `pillow`/`numpy` already present. No torch/transformers/fontrecog.
      In `schema_chatbot_v2/Dockerfile` add a comment only: the probe model resolves to
      `/app/src/ai/layer1_probe/models/lipika_script.onnx` (shipped by the existing `COPY . /app`;
      `.dockerignore` passes `*.onnx`/`*.json`), and `IDP_ENABLE_PROBE` is OFF by default so the new
      deps sit idle until the flag is turned on. No new `COPY`.
      Files: `schema_chatbot_v2/requirements.txt`, `schema_chatbot_v2/Dockerfile`

- [ ] 21. (Optional) Local docker build smoke-test — ONLY if docker is available.
      If `docker` is present locally, optionally run `docker build -f Dockerfile.probe .` and
      `docker build -f schema_chatbot_v2/Dockerfile .` to confirm the images still build. **If docker is
      absent, skip and note it.** Do NOT push to ECR, do NOT run any `deploy_*.py`, do NOT touch any
      Lambda/ECS/state machine.
      Verify: `docker --version` — if it errors, record "docker not present, smoke-test skipped".

---

## STEP 6 — Tests (fully mocked; NO real model, NO torch, NO real AWS)

- [ ] 22. RE-GREP `tests/` before editing and update/remove every probe-handwriting hit.
      Grep `tests/` for `handwriting_prob`, `hw_prob`, `hw=`, `printed_scripts`, `paddleocr_handwritten`,
      `handwriting_heuristic`, `line_periodicity`, `handwriting_fraction`. Confirm the probe hits are
      confined to `tests/test_probe.py` (the `paddleocr_handwritten` hits in `test_capability_router.py`,
      `test_router_smoke.py`, `test_document_ir_blocks.py`, `test_engines_probe_inputs.py`,
      `test_layer2_engines_local.py` are about the ENGINE itself and the router/capability layer — all
      out of scope, leave them). Then apply items 23–24 to `tests/test_probe.py`.

- [ ] 23. Edit `tests/test_probe.py` — teardown the handwriting-coupled tests.
      1. **DELETE** `test_e2e_heterogeneous_doc_uses_nearest_sample_and_widens_sampling` entirely
         (user decision: delete; no replacement needed even if that leaves nearest-sample/widening
         uncovered).
      2. `StubVLM` (~L114-122): remove the `hw=` ctor param and the `handwriting_prob=h` kwarg in
         `observe`; the `per_page` tuple shrinks from `(script,conf,hw)` to `(script,conf)` →
         `s, c = self.per_page.get(pn, (self.script, self.conf))`.
      3. `test_aggregate_and_disagreement_and_nearest` (~L206): drop the `handwriting_prob=` kwargs from
         both `SampleObs(...)` constructions; keep the script-based `samples_disagree`/`nearest_sample`
         assertions.
      4. `test_bedrock_probe_normalises_and_survives_garbage` / VLM-parse assertion (~L228): change the
         tuple to `assert (o.script, o.script_confidence, o.has_table) == ("devanagari", 1.0, True)`
         (drop `handwriting_prob`). (The fence-parse test at ~L217 can keep its raw JSON string with
         `handwriting_fraction` since `parse_vlm_json` is schema-agnostic; it only asserts `script`.)
      5. `test_routing_table` (~L167): drop `hw_prob=` from every param row; **remove the
         `latin + hw_prob=.8 → paddleocr_handwritten` row**; call becomes
         `decide_route(mixed=False, is_blank=False, has_table=False, cfg=_cfg(), **kw)` with the new
         signature. Rows: latin/.9→`paddleocr_printed`/`en`; devanagari/.9→`vlm_transcribe`/None;
         latin/.3→`vlm_transcribe`/None; unknown/0.0→`vlm_transcribe`/None. Add an assertion that no
         row yields `paddleocr_handwritten`.
      6. `test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm` (~L180): drop
         `_cfg(printed_scripts=...)` and `hw_prob=`; assert `devanagari`→`vlm_transcribe` and the
         `mixed=True` latin case→`vlm_transcribe`. (Rename is optional.)
      7. `test_table_sets_structure_hint_only_for_printed` (~L188): drop both `hw_prob=` args and the
         `hw_prob=.9→use_structure False` assertion; assert `script="latin", has_table=True →
         use_structure is True` and `has_table=False → use_structure is False`.
      8. `test_blank_routes_to_skip` (~L195): drop `hw_prob=0`; keep `is_blank=True → skip`.
      9. `test_e2e_devanagari_handwritten_goes_to_vlm_and_latin_hw_to_handwritten_engine` (~L286):
         rewrite to the script-only expectation — drop the `hw=0.9` StubVLM arg; Devanagari→
         `{"vlm_transcribe"}`; Latin→`{"paddleocr_printed"}` (NO case may expect `paddleocr_handwritten`).
      10. `test_onnx_classifier_contract` (~L364): rewrite for the whole-image lipika observer via the
          injectable fake session — construct `OnnxScriptClassifier(model_path, labels_path, session=<fake>)`
          where `<fake>.run(...)` returns canned logits peaking on a chosen script; assert
          `obs.script == <that script>` and `obs.source == "onnx"` only, **NO `handwriting_prob`
          assertion**. Remove the old `[N,1,48,320]` onnx graph construction.
      11. **REMOVE** `test_handwriting_heuristic_orders_wobbly_above_typed` and
          `test_heuristic_returns_none_when_too_little_text` (the heuristic is deleted in item 9).
      Files: `tests/test_probe.py`

- [ ] 24. Add `tests/test_probe_script_onnx.py` (observer mapping + singleton).
      Use a fake `InferenceSession` (NOT a real model): construct the observer with `session=<fake>`
      and/or patch `_get_session`/`InferenceSession` with a `mock.call_count` counter. Assert canned
      logits peaking on latin/telugu/devanagari indices map to the right `SampleObs.script`; assert
      `source == "onnx"`; assert `"handwriting_prob" not in SampleObs.model_fields`; assert the
      module-level session is constructed exactly ONCE across multiple `observe` calls (use
      `_reset_for_tests()` between tests to avoid singleton leakage). No torch, no real model load.
      Files: `tests/test_probe_script_onnx.py`

- [ ] 25. Add `tests/test_pipeline_probe_integration.py` (flag-gated in-process wiring).
      Mock the `llm_client`, monkeypatch the layer-2 converters, and mock boto3 `lambda.invoke` so no
      real OCR/AWS runs. Build a `page_context` with an `image_array` (synthetic RGB) + `render_dpi`.
      Assert:
      - with `IDP_ENABLE_PROBE` unset/off → `process_document`/`process_page` on a scanned page chooses
        the same engine as Layer-1 (probe hook returns `None`; the probe module is never imported);
      - with the flag on + observer injected to return `telugu` → a scanned non-latin page routes to
        `vlm_transcribe`;
      - with the flag on + observer returns `latin` → a scanned latin page routes to `paddleocr_printed`
        and `ocr_lang="en"` is threaded into the **mocked** `_invoke_paddle_lambda` payload
        (`payload["ocr_lang"] == "en"`);
      - the `_invoke_paddle_lambda` payload is **byte-identical** to today
        (`{"image_base64": ..., "page_number": n}`, no `ocr_lang` key) when no `ocr_lang` is set;
      - a digital/Docling page is never probed with the flag on (observer not called);
      - `"handwriting_prob" not in SampleObs.model_fields` and `not in PageProbe.model_fields`;
        `"hw_source" not in PageProbe.model_fields`; `decide_route` has no `hw_prob` parameter
        (`inspect.signature`);
      - no routed path can yield `paddleocr_handwritten`.
      Use `monkeypatch.setenv`/`delenv` for the flag and `_reset_for_tests()` for the observer
      singleton. All observer results come from the injected fake session, so this passes under the §9
      no-model fallback.
      Files: `tests/test_pipeline_probe_integration.py`

---

## VERIFY (run via the repo venv; do NOT run the slow real-PaddleOCR tests)

- [ ] 26. Grep checks (teardown is complete).
      PowerShell (from repo root). Expect ZERO matches in the probe runtime path and tests:
      ```
      .venv\Scripts\python.exe -m pip --version   # sanity: venv active
      ```
      Source path (expect no matches under src/ai/layer1_probe or lambda_probe_handler.py):
      - `grep` tool: query
        `handwriting_prob|hw_prob|hw_source|handwriting_heuristic|line_periodicity|hw_threshold|HW_THRESHOLD|printed_scripts|PRINTED_SCRIPTS`,
        includePattern `src/ai/layer1_probe/**/*.py` → expect **0 matches**.
      - same query, includePattern `src/api/lambda_probe_handler.py` → expect **0 matches**.
      Tests path (expect no probe-handwriting matches in test_probe.py):
      - query `handwriting_prob|hw_prob=|printed_scripts|handwriting_heuristic|line_periodicity`,
        includePattern `tests/test_probe.py` → expect **0 matches**.
      (The `.scratch_probe/*` hits for `handwriting_heuristic`/`line_periodicity` are expected and
      intentionally left; they are outside `src/`.)

- [ ] 27. Run the FAST probe + integration suite via the repo venv (deselect the ~8-min local Paddle
      tests). From repo root:
      ```
      .venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_script_onnx.py tests/test_pipeline_probe_integration.py -k "not engine_local_path" -q
      ```
      Expected: all collected tests pass (0 failures). If any observer test needs a real model it is a
      bug — every observer result must come from an injected fake session.

- [ ] 28. Run the broader fast suite to confirm nothing else regressed (still deselecting the slow
      local Paddle tests). From repo root:
      ```
      .venv\Scripts\python.exe -m pytest tests -k "not engine_local_path" -q
      ```
      Expected: green. Investigate and fix any failure introduced by the schema/routing teardown before
      finishing. (Do NOT run `tests/test_layer2_engines_local.py::*_engine_local_path` — ~8 min each.)

- [ ] 29. Clean up any temporary files created during verification. Leave the three model artifacts
      under `src/ai/layer1_probe/models/` (STEP 1) in place; they are deliverables, not temp files.

---

## Verification (recorded — reviewer can read WITHOUT re-running)

Environment: repo venv `.venv\Scripts\python.exe` (Python 3.11.9). All runs set
`PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` so the cp1252 console does not choke on model output.

### STEP 1 — offline lipika ONNX export (dev-only; torch/fontrecog OFFLINE via venv)
- Installed OFFLINE into the venv only (NOT in any runtime requirements file): `lipika==2.5.0`
  (provides `fontrecog`), `timm`, `onnxscript==0.7.2` (needed only by torch's exporter), `onnx_ir`.
  torch/transformers were already present. None added to `requirements-probe.txt` /
  `schema_chatbot_v2/requirements.txt`.
- `fontrecog.infer.predictor.Predictor.from_pretrained("loopdesk-ai/lipika")` loaded successfully
  (FontNet = ConvNeXt backbone + family/script/weight/italic heads). Introspection confirmed the
  script head (`model.script`) and 14 script labels via `space.script_to_idx`:
  bengali, devanagari, gujarati, gurmukhi, kannada, latin, malayalam, meetei_mayek, odia, ol_chiki,
  perso_arabic, syloti_nagri, tamil, telugu.
- Preprocessing is Option A single-forward reproducing lipika's own `letterbox` (white pad to square,
  BICUBIC resize to 224, /255, no ImageNet mean/std) -> metadata `mean=[0,0,0] std=[1,1,1]
  input_size=[224,224] preprocessing="single_forward"`.
- `.venv\Scripts\python.exe scripts/export_lipika_script_onnx.py` wrote all three artifacts to
  `src/ai/layer1_probe/models/`: `lipika_script.onnx` (FP32, ~112MB), `lipika_script.int8.onnx`
  (~28MB), `lipika_script.labels.json`. Label remap applied per design §4:
  perso_arabic->arabic, meetei_mayek/ol_chiki/syloti_nagri->other, rest direct.
- **Export fidelity (the criterion that matters): FP32 ONNX == torch logits on all 4 validation
  images** (devanagari/gurmukhi/gurmukhi/odia, identical label+confidence). So the ONNX graph is a
  faithful export of the script head.
- INT8 flipped at least one prediction vs torch -> `shipped_default="fp32"` kept (INT8 file still
  saved for the record), exactly as the design prescribes.
- NOTE (not an export bug, recorded for the reviewer): lipika's own single-forward (and even its
  NATIVE patch-voting `predict`) misclassifies these full-UI-screenshot validation images (only the
  telugu one, which lipika tiles into 5 patches, is correct natively). These are off-distribution for
  lipika (it expects cropped text). Because Option A == torch and the design chose Option A, this is
  within spec; low-confidence/non-latin pages route to the VLM (the safe direction), so a
  misclassification does not produce garbage OCR. STEP 1 was NOT blocked (fontrecog installed, model
  loaded, export faithful), so no STOP-AND-ASK was needed and STEPS 2-6 proceeded.

### Grep teardown checks (expect 0 — all clean)
- `grep handwriting_prob|hw_prob|hw_source|handwriting_heuristic|line_periodicity|hw_threshold|HW_THRESHOLD|printed_scripts|PRINTED_SCRIPTS`
  over `src/ai/layer1_probe/**/*.py` -> **0 matches**.
- same query over `src/api/lambda_probe_handler.py` -> **0 matches**.
- `grep handwriting_prob|hw_prob=|printed_scripts|handwriting_heuristic|line_periodicity`
  over `tests/test_probe.py` -> **0 matches**.
  (`.scratch_probe/*` left untouched by design — outside `src/`, not collected/shipped.)
- `decide_route` signature confirmed to have NO `hw_prob` parameter (inspect.signature).

### Test runs (FAST suite; slow real-PaddleOCR `*_engine_local_path` deselected)
- `pytest tests/test_probe.py tests/test_probe_script_onnx.py tests/test_pipeline_probe_integration.py -k "not engine_local_path"`
  -> **40 passed in 9.61s**. (New mocked observer + flag-gated in-process wiring tests all pass;
  every observer result comes from an injected fake session — no model load.)
- `pytest tests/test_capability_router.py tests/test_router_smoke.py tests/test_router_edgecases.py
  tests/test_document_ir_blocks.py tests/test_vlm_routing.py tests/test_engines_probe_inputs.py`
  -> **85 passed in 2.45s**.
- `pytest tests/test_layer3_hardening.py tests/test_layer3_pipeline.py
  tests/test_layer3_truncation_parse.py tests/test_digital_conversion.py tests/test_pdf_extraction.py
  tests/test_merge_determinism.py tests/test_merge_label_gate.py tests/test_chunker.py
  tests/test_crosspage_linking.py tests/test_ir_store.py tests/test_line_segmentation.py`
  -> **133 passed, 1 failed**. The single failure is
  `test_layer3_hardening.py::test_schema_resolution_fuzzy_candidate_must_not_populate_empty_field`,
  in the Layer-3 graph-agent schema resolver (`src/ai/layer3_extraction/graph_agent/resolver.py`,
  unchanged from HEAD) — a PRE-EXISTING failure from parallel working-tree changes
  (`extractor.py`/`test_layer3_hardening.py` were already modified `M` before this task). NOT touched
  by and NOT related to the probe work.
- `pytest tests/test_handwritten_extraction.py tests/test_lambda_layer1_local.py` -> **7 passed**.
- Running the WHOLE `tests` directory in one process timed out locally (network/model-backed suites:
  sarvam/query-bot/phase4 LLM/graph-memory). Ran the probe-affected + core subsets above instead;
  all green except the one pre-existing unrelated layer3 failure.

### STEP 5 packaging / docker
- `docker --version` -> Docker 29.6.2 present. The optional `docker build` smoke test (Dockerfile.probe
  and schema_chatbot_v2/Dockerfile) was NOT run: both images install heavy deps (paddleocr, etc.) and
  a full build is long; the task marks it optional and the priority was the test suite. Dockerfile
  edits (ENV `IDP_PROBE_PRINTED_SCRIPTS` removed; models-COPY comment; app-image comment) and the
  `.dockerignore` check (does NOT exclude `*.onnx`/`*.json`, so the baked model ships via the existing
  `COPY src/`/`COPY . /app`) were verified by inspection.

### Invariant spot-checks
- Flag OFF (`IDP_ENABLE_PROBE` unset) -> `_maybe_probe_route` returns `None` BEFORE importing
  `layer1_probe`; test `test_probe_off_is_passthrough_and_no_probe_import` asserts the observer is
  never called and the Layer-1 engine is kept.
- `_invoke_paddle_lambda` payload is byte-identical (`{"image_base64","page_number"}`, no `ocr_lang`)
  when no `ocr_lang` is set; asserted by `test_probe_off_payload_is_byte_identical`.
- No routed path yields `paddleocr_handwritten`; the engine code + `ROUTE_HANDWRITTEN` constant are
  retained (not deleted). Asserted by `test_no_route_yields_paddleocr_handwritten` and the routing table.
- `import src.ai.layer1_probe.inprocess` does NOT pull `fitz` onto the hot path (asserted at import).
- Observer singleton built ONCE across multiple `observe` calls; injectable `session=` ctor param kept.
