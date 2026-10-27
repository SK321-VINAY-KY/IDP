# Line-wise lipika observer (pilot, Strategy B, fp32)

The whole-page letterbox classifier in `script_classifier.py` is replaced by a line-wise path: ink-mask line crops (largest first, capped at 24) are cut into square-ish windows, letterboxed at 224, run through one batched `session.run`, and aggregated by mean softmax to produce `script` / `script_confidence` / `margin`. Three gates (`lipika_min_windows=8`, `lipika_min_conf=0.60`, `lipika_min_margin=0.20`) short-circuit to `None` so unresolved pages fall through to the VLM. `probe.py::_observe` gains a VLM-precedence seam (a VLM that contradicts a confident lipika "latin" now wins, so the page is not sent to Paddle), a `probe.lipika_vs_vlm` telemetry event, and three stats counters. The Strategy-B preprocessing is lifted into a shared `windows.py` so the observer and the eval harness run byte-identical tensors.

Watch for: the measured ~44 s/page fp32 inference cost at 200–600 windows/page (confirmed, flagged by the author as a go/no-go item — not a correctness bug). The working tree also carries several unrelated changes bundled alongside this work (likely, see Issues). The two reported test failures are pre-existing baseline failures unrelated to this change (confirmed).

**Verdict**: APPROVED

## High-level view

The six required behaviors are all present and correct. The whole-page `_letterbox` function is gone and `observe()` now walks line crops → `square_windows` → `letterbox_square` → single batched forward → `probs.mean(axis=0)` → argmax/conf/margin, which is Strategy B on the fp32 model that `default_model_path()` resolves by default. `_letterbox_A`/`_windows_B` physically moved to `src/ai/layer1_probe/windows.py`; `scripts/eval_lipika_crops.py` deletes its own copies and imports them (`crop_softmax` still calls the same names via the import alias), so there is one implementation, not two.

The three gates are real `ProbeConfig` fields with `IDP_PROBE_LIPIKA_*` overrides wired through `from_env`, and each maps to an early `return None` inside `observe()` (too few windows before the forward pass; low conf or low margin after aggregation). `onnx_min_conf` is untouched and still governs the separate VLM-fallback decision in `probe.py`.

VLM precedence is enforced inside `_observe` (not routing.py): on the handwriting-flag branch, a VLM script that differs from lipika's replaces `obs` entirely, so a disputed "latin" routes to `vlm_transcribe` rather than Paddle; an agreeing VLM keeps the handwriting-only behavior. The `probe.lipika_vs_vlm` event and the `lipika_resolved` / `lipika_vlm_agree` / `lipika_vlm_disagree` counters are emitted for every sampled page with no image bytes.

The 14-slot label map (indices 7/9/11 = "other") is confirmed against `lipika_script.labels.json`, cited inline to `export_lipika_script_onnx.py::_LIPIKA_TO_PROBE` and the model card, and exercised by a skippable real-fp32 test. The injected-session ctor path and the `_SESSION`/`_META` module singletons are preserved, and the `OnnxScriptClassifier` alias survives for `lambda_probe_handler.py`.

All absolute rules hold: nothing is staged, no `.onnx` is tracked, no tracked edits exist under `docker/`/`step_functions/`/`src/ai/layer1_routing/`, routing policy / thresholds / `IDP_PROBE_PRINTED_SCRIPTS` defaults / `onnx_min_conf` / the handwriting flag are all unchanged, and PP-Structure is not restored.

<details>
<summary>Issues (5)</summary>

1. **Window-count latency (informational)** — Strategy B emits 200–600 windows/page at the probe's 144 DPI, giving ~44 s/page fp32 inference on CPU. Flagged by the author for the owner's go/no-go; not a correctness defect and out of scope to change here (int8 / window cap are threshold decisions).
2. **`.onnx` not gitignored (possible)** — the fp32/int8 models are untracked but not matched by `.gitignore`, so a future `git add .` could stage them against the rule. Add a `*.onnx` ignore entry to remove the footgun.
3. **Unrelated bundled changes (likely)** — the working tree mixes in a layer3 `save_document_graph(owner=...)` fix + test, new S3 settings in `src/config/settings.py`, a `sentence-transformers==3.3.1` pin, a `schema_chatbot_v2` pipeline test, and `AGENTS.md`/`.gitignore` edits. None violate the path rules, but they should land as separate commits, not with the line-wise change.
4. **Untracked file under `step_functions/` (possible)** — `step_functions/state_machine_with_probe.asl.json` appears as a new untracked file. It is not a tracked edit and not referenced by this task, but since `step_functions/` is a forbidden path it should be confirmed it was not produced by this work before any commit.
5. **Two pre-existing test failures (confirmed, not blocking)** — `test_no_handwriting_field_on_contracts` and `test_contract_fields_removed_and_signature_script_only` fail because `SampleObs.handwriting_prob` was added by an earlier commit (705fe37); the author verified they fail on the clean baseline too. Out of scope for this change.

</details>

<details>
<summary>Details</summary>

### Whole-page path removed; line-wise Strategy B in place

`script_classifier.py` no longer contains the module-level `_letterbox(rgb, size, mean, std)` whole-page function — it is deleted, and nothing imports it (the eval script's doc references were repointed to `windows.letterbox_square`). `observe()` now builds per-line RGB sources via `_line_sources` (bbox mapped from analysis-DPI gray to full-res rgb when `rgb` is supplied, else gray→RGB), flattens `square_windows` over all lines, gates on window count, runs one `session.run` over the concatenated batch, and aggregates with `probs.mean(axis=0)`. `top`/`second`/`margin` come from the mean vector, which is the correct Strategy-B semantics. The `_softmax` was also corrected to operate row-wise (`axis=-1`) so it handles the `[W, K]` batch rather than a single row.

### Shared preprocessing, no duplication

`windows.py` holds `letterbox_square` (= old `_letterbox_A`), `square_windows` (= old `_windows_B`), and `_INPUT_SIZE = 224`, numpy+cv2 only — import-safe on the hot path. `eval_lipika_crops.py` removed its own three definitions and imports the two functions under their old names plus `_INPUT_SIZE`, leaving only `_letterbox_C` (strategy C) local. `crop_softmax` and the strategy dispatch still call `_letterbox_A`/`_windows_B`, so the eval numbers are unchanged while the observer and harness share one code path. This is exactly the "move, don't reimplement" the task asked for.

### Gates and their None paths

`ProbeConfig` gains `lipika_min_windows:int=8`, `lipika_min_conf:float=0.60`, `lipika_min_margin:float=0.20`, each read in `from_env` via `_env(...)` which prefixes `IDP_PROBE_`, yielding `IDP_PROBE_LIPIKA_MIN_WINDOWS/CONF/MARGIN`. In `observe()`, `n_windows < lipika_min_windows` returns `None` before the forward pass; after aggregation, `top < lipika_min_conf or margin < lipika_min_margin` returns `None`. These are additive — `onnx_min_conf` (0.80) and its use in `probe.py` are untouched, so the gate is a stricter pre-filter inside the observer, not a replacement for the existing VLM-fallback threshold. The tests drive each path with an injected `ProbeConfig` (min_windows=1000 for the window gate; near-uniform logits for low conf; a 5.0/4.95 near-tie for low margin).

### VLM precedence seam

The change is confined to the handwriting-flag `elif` in `_observe` (routing.py and thresholds untouched). Previously that branch only copied `handwriting_prob`, so a confident lipika "latin" survived even when the VLM returned a non-Latin script, which would route to Paddle. Now:

```
if v.script and v.script != obs.script:
    obs = v          # contradicting VLM wins -> non-latin -> vlm_transcribe
else:
    obs.handwriting_prob = v.handwriting_prob   # agree/latin: handwriting-only, unchanged
```

Test (d) confirms a `latin` lipika + `devanagari` VLM routes to `vlm_transcribe` (not `paddleocr_printed`), emits `agree=False`, and bumps `lipika_vlm_disagree`. Test (e) confirms `latin`+`latin` keeps `paddleocr_printed` and bumps `lipika_vlm_agree`. Telemetry defines `agree=None` when the VLM was not called and logs no image bytes; `n_windows`/`margin` are read off the observer via `getattr`, so VLM-only/stub tests that lack those attributes still work.

### Label mapping

`lipika_script.labels.json` lists 14 labels with "other" at indices 7, 9, 11 and "arabic" at 10 — matching the inline citation (`meetei_mayek`/`ol_chiki`/`syloti_nagri` → "other", `perso_arabic` → "arabic", source `export_lipika_script_onnx.py::_LIPIKA_TO_PROBE` + the loopdesk-ai/lipika card). The argmax→`labels[idx]` lookup keeps the OOB guard. `test_label_mapping_real_fp32_optional` is `@pytest.mark.slow`, skips cleanly when the model or `dataset/verified_twice` fixtures are absent, hard-asserts bengali/tamil, and logs the rest — a wiring/order sanity check, not an accuracy gate.

### Injected-session and singleton preservation

The ctor still accepts `session=` (no onnxruntime import on that path) and now also `cfg=` defaulting to `ProbeConfig()`, so `lambda_probe_handler.py`'s two-arg construction stays valid. `_get_session`/`_reset_for_tests`/`default_model_path`/`default_labels_path`/`_load_meta` are intact; `test_module_session_constructed_once` asserts the module singleton is built exactly once across two `observe()` calls. `inprocess._get_observer()` now passes `cfg=ProbeConfig.from_env()` so live runs honor the gate overrides.

### Test evidence (read, not re-run)

Per the plan's verification note: `pytest tests/test_probe.py tests/test_probe_script_onnx.py tests/test_line_crops.py tests/test_pipeline_probe_integration.py tests/test_probe_linewise.py` → **52 passed, 2 failed**. The two failures (`test_no_handwriting_field_on_contracts`, `test_contract_fields_removed_and_signature_script_only`) assert `handwriting_prob` is absent from the `SampleObs` contract; the author verified by stash that both fail on the committed baseline because `handwriting_prob` was added by the earlier handwriting-flag commit (705fe37), i.e. pre-existing and out of scope. The opt-in label-mapping test passes against the real fp32 model (11/12 languages correct from a single crop; telugu's single crop mispredicts under the soft tolerance by design). Measurements: fp32 load ~0.33–0.37 s; ~105–133 ms/window; prep ~0.27 s/page; total extra ~44 s/page at the mean 403 windows/page. Evidence accepted as recorded.

### Scope / rule check

`git diff --cached` is empty (nothing staged). `git ls-files "*.onnx"` is empty (no tracked model). `git diff --name-only -- docker/ step_functions/ src/ai/layer1_routing/` is empty (no tracked edits in forbidden paths). The diff contains no PP-Structure references. `IDP_PROBE_PRINTED_SCRIPTS`, `onnx_min_conf`, `min_script_conf`, `hw_threshold`, and `vlm_for_handwriting` defaults are unchanged. The bundled unrelated edits (layer3 owner fix, S3 settings, sentence-transformers, schema_chatbot test, AGENTS.md, .gitignore) and the untracked `step_functions/state_machine_with_probe.asl.json` are noted in Issues; none are tracked edits to forbidden paths and none affect the correctness of the line-wise change.

</details>

<details>
<summary>File map</summary>

- `src/ai/layer1_probe/windows.py` (new) — shared `letterbox_square` / `square_windows` / `_INPUT_SIZE`, numpy+cv2 only.
- `src/ai/layer1_probe/script_classifier.py` — rewritten to the line-wise Strategy-B path; whole-page `_letterbox` deleted; `cfg=` ctor arg; `last_n_windows`/`last_margin` telemetry attrs; label citation; alias kept.
- `src/ai/layer1_probe/types.py` — three `lipika_*` gate fields + `IDP_PROBE_LIPIKA_*` env overrides.
- `src/ai/layer1_probe/probe.py` — VLM-precedence seam, `probe.lipika_vs_vlm` event, three stats counters.
- `src/ai/layer1_probe/inprocess.py` — observer built with `cfg=ProbeConfig.from_env()`.
- `scripts/eval_lipika_crops.py` — imports `_letterbox_A`/`_windows_B`/`_INPUT_SIZE` from `windows.py`; local defs removed; strategy C kept.
- `tests/test_probe_linewise.py` (new) — aggregation, three gate paths, VLM contradict/agree, skippable real-fp32 label map.
- `tests/test_probe.py`, `tests/test_probe_script_onnx.py` — fake sessions return one row per window; wide synthetic page feeds the line-wise path.
- Bundled/unrelated (flagged): `src/ai/layer3_extraction/extractor.py`, `tests/test_layer3_hardening.py`, `src/config/settings.py`, `requirements.txt`, `schema_chatbot_v2/tests/test_pipeline_jobs.py`, `AGENTS.md`, `.agents/rules/aws_tagging.md`, `.gitignore`.

Full diff: `git -C c:\Users\Dell\Desktop\IDP\engineer_a diff`

</details>
