# Implementation Plan — lipika window cap + contract-test housekeeping

Scope: LOCAL-ONLY. Two code items (code + tests). NO git commit/stage/LFS, NO
AWS, NO deploy. Do NOT edit `docker/`, `step_functions/`, or
`src/ai/layer1_routing/`. Routing policy and all defaults otherwise UNCHANGED
(do not touch `routing.py` decision logic, `IDP_PROBE_PRINTED_SCRIPTS`,
`onnx_min_conf`, `min_script_conf`, `hw_threshold`, the
`IDP_PROBE_VLM_FOR_HANDWRITING` flag, or the existing
`lipika_min_windows`/`lipika_min_conf`/`lipika_min_margin` gate defaults). Keep
the fp32 model as the shipped default.

Python for ALL invocations: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`.

## Out of scope (do NOT do here)
The measurement/report phase (original request items 3 and 4 — per-cap latency
on 5 real dataset pages, the harness page-level eval split by document, the
confusion matrices, the recommended default cap) is handled by a LATER dedicated
step. It is NOT part of this implement-and-review loop. This plan covers ONLY
the two code items below (code + their unit tests). Do not run the eval harness
or produce report tables here.

## Context verified during exploration
- `src/ai/layer1_probe/types.py` — `ProbeConfig` dataclass already has the three
  lipika gate fields (`lipika_min_windows=8`, `lipika_min_conf=0.60`,
  `lipika_min_margin=0.20`) and wires each in `from_env` via the `_env(...)`
  helper using the `IDP_PROBE_` prefix. `SampleObs` and `PageProbe` BOTH already
  declare `handwriting_prob: Optional[float] = None`.
- `src/ai/layer1_probe/script_classifier.py` — `LipikaOnnxScriptObserver.observe`
  builds `line_sources` via `self._line_sources(...)` (largest-first, capped at
  `_MAX_LINES=24`), then flattens windows with
  `for src in line_sources: all_windows.extend(square_windows(src))`, sets
  `self.last_n_windows`, applies GATE A (`n_windows < cfg.lipika_min_windows ->
  None`), then letterboxes into one batch and runs the session. The cap must be
  inserted AFTER `all_windows` is built but BEFORE `n_windows`/GATE A and the
  batch concatenation. The observer reads `self.cfg` (a `ProbeConfig`).
- `square_windows` (in `src/ai/layer1_probe/windows.py`) returns a list of RGB
  window arrays per line crop; it is deterministic and stable in order.
- Two stale contract tests assert `handwriting_prob` is absent; they are the
  ONLY failures in the target suite (verified: 2 failed, 52 passed). In BOTH
  tests the `hw_source`/`hw_prob` assertions still PASS (those fields really are
  absent — `decide_route`'s handwriting param is named `handwriting_prob`, not
  `hw_prob`), so ONLY the two `handwriting_prob` `model_fields` assertions are
  wrong and must change.
- `.gitignore` does NOT currently contain `*.onnx`.

---

## Item 1 — Housekeeping

- [ ] 1. Add `*.onnx` to `.gitignore`.
      Append a single new line `*.onnx` (with a brief comment is fine, e.g. under
      a `# Model binaries (never commit)` header) so untracked `.onnx` binaries
      can never be accidentally staged. Change nothing else in the file; do not
      reorder or remove existing lines. Do NOT stage, commit, or `git add` the
      model files.
      Files: `.gitignore`
      Verify: read the file back and confirm `*.onnx` appears exactly once and no
      prior line was altered. (No test runs this; the suite run in item 4 must
      still pass.)

- [ ] 2. Fix the stale assertion in `test_no_handwriting_field_on_contracts`
      (`tests/test_probe_script_onnx.py`, ~line 74). Change ONLY the now-wrong
      `handwriting_prob` assertions to assert the INTENDED contract:
      `handwriting_prob` IS present on `SampleObs.model_fields` and on
      `PageProbe.model_fields` (it is Optional, VLM-sourced, re-added by the
      `IDP_PROBE_VLM_FOR_HANDWRITING` flag, commit 705fe37). Leave the
      `assert "hw_source" not in PageProbe.model_fields` line unchanged (still
      correct). Do not rename the test, add new tests, or touch any other
      assertion or test in the file. Concretely, replace the two
      `assert "handwriting_prob" not in ...` lines with
      `assert "handwriting_prob" in SampleObs.model_fields` and
      `assert "handwriting_prob" in PageProbe.model_fields`.
      Files: `tests/test_probe_script_onnx.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_script_onnx.py::test_no_handwriting_field_on_contracts -q` passes.

- [ ] 3. Fix the stale assertion in
      `test_contract_fields_removed_and_signature_script_only`
      (`tests/test_pipeline_probe_integration.py`, ~line 152). Change ONLY the two
      now-wrong `handwriting_prob` assertions the same way as item 2:
      `assert "handwriting_prob" in SampleObs.model_fields` and
      `assert "handwriting_prob" in PageProbe.model_fields`. Leave the
      `assert "hw_source" not in PageProbe.model_fields` line and the
      `decide_route` signature check (`assert "hw_prob" not in sig.parameters`)
      UNCHANGED — both are still correct (the handwriting routing exists behind
      the flag and the param is named `handwriting_prob`, so `hw_prob` is still
      absent). Do not rename the test or touch any other assertion or test.
      Files: `tests/test_pipeline_probe_integration.py`
      Verify: `.venv\Scripts\python.exe -m pytest "tests/test_pipeline_probe_integration.py::test_contract_fields_removed_and_signature_script_only" -q` passes.

---

## Item 2 — Latency fix: even-spread window cap

- [ ] 4. Add the `lipika_max_windows` config field.
      In `src/ai/layer1_probe/types.py`, add `lipika_max_windows: int = 32` to the
      `ProbeConfig` dataclass, placed alongside the other `lipika_*` gate fields
      (directly after `lipika_min_margin`). In `ProbeConfig.from_env`, add
      `c.lipika_max_windows = int(_env("LIPIKA_MAX_WINDOWS", str(c.lipika_max_windows)))`
      next to the other `c.lipika_min_* = int(_env("LIPIKA_MIN_*", ...))` lines,
      mirroring their exact style (env var `IDP_PROBE_LIPIKA_MAX_WINDOWS`). Do not
      change any existing default.
      Files: `src/ai/layer1_probe/types.py`
      Verify: `.venv\Scripts\python.exe -c "import os; os.environ['IDP_PROBE_LIPIKA_MAX_WINDOWS']='16'; from src.ai.layer1_probe.types import ProbeConfig; assert ProbeConfig().lipika_max_windows==32; assert ProbeConfig.from_env().lipika_max_windows==16; print('ok')"`
      prints `ok`.

- [ ] 5. Add a deterministic, even-across-lines window selector and apply it in
      the observer. In `src/ai/layer1_probe/script_classifier.py`:
      (a) Build windows grouped PER LINE instead of pre-flattening — replace the
      current flatten
      `for src in line_sources: all_windows.extend(square_windows(src))`
      with a per-line list, e.g.
      `per_line_windows = [square_windows(src) for src in line_sources]`
      (preserving the existing largest-first line order from `_line_sources`).
      (b) Add a small module-level pure helper, e.g.
      `def _even_spread(per_line_windows, cap)`, that returns a flat list of
      windows. When the total window count across all lines is `<= cap`, return
      the simple flattened list UNCHANGED (preserve current order: line 0's
      windows, then line 1's, ...). When the total `> cap`, select exactly `cap`
      windows by ROUND-ROBIN over the lines top-to-bottom: take window 0 of line
      0, window 0 of line 1, ... then window 1 of each line that still has one,
      wrapping until `cap` windows are collected. Deterministic — no RNG, stable
      iteration over the input lists; skip lines that have been exhausted. Keep
      the selection traceable to source lines so the spread is genuinely across
      lines (the round-robin index order itself proves this).
      (c) In `observe`, compute `all_windows = _even_spread(per_line_windows,
      self.cfg.lipika_max_windows)` AFTER windows are generated and BEFORE
      `n_windows = len(all_windows)`. GATE A (`lipika_min_windows`) then evaluates
      on the CAPPED set (unchanged code). Everything downstream (letterbox, batch,
      session.run, mean softmax, gates B) is unchanged. fp32 unchanged.
      Files: `src/ai/layer1_probe/script_classifier.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py tests/test_probe_script_onnx.py -q` — existing observer tests still pass (the cap default 32 must not break the existing `_wide_text_gray` aggregation tests; if those pages produce <=32 windows the capped path is a no-op, confirm by running).

- [ ] 6. Add cap unit tests to `tests/test_probe_linewise.py`.
      Use an injected fake session (reuse the existing `_fake_session` /
      `_wide_text_gray` helpers in that module) and a synthetic multi-line page so
      window counts are controllable. NO real model, NO AWS, NO torch. Make the
      fake session RECORD the batch size it is asked to run (capture
      `feeds[input_name].shape[0]` into a list) so tests can assert how many
      windows were forwarded. Add these four tests:
        - `test_cap_respected`: build a page / set `lipika_max_windows` so the
          generated window count exceeds the cap; assert the number of windows
          fed to the fake session (recorded batch size) `== cfg.lipika_max_windows`,
          and `observer.last_n_windows == cfg.lipika_max_windows`.
        - `test_even_spread_across_lines`: construct a controlled multi-line page
          (several distinct line bands) with a cap smaller than the total; assert
          the kept windows are distributed round-robin across lines (near-equal
          per-line share; the first `len(lines)` kept come one-per-line
          top-to-bottom), NOT the first-N windows all drawn from line 0. Prefer
          asserting against the `_even_spread` helper directly with synthetic
          per-line window lists (e.g. tagged arrays/ids) so the round-robin order
          is checked exactly and deterministically, independent of OpenCV
          windowing.
        - `test_cap_deterministic`: call `_even_spread` (or run `observe`) twice on
          identical input and assert the selected window set/order is identical
          across both runs.
        - `test_below_cap_unchanged`: generate `<= cap` windows and assert all are
          used (recorded batch size == total generated), identical to the pre-cap
          flatten order (no selection applied).
      Files: `tests/test_probe_linewise.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_linewise.py -q` — all new tests pass.

---

## Final verification (coder must run before marking done)

Run the full target suite with the repo venv and confirm 0 failures:

```
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest ^
  tests/test_probe.py ^
  tests/test_probe_linewise.py ^
  tests/test_line_crops.py ^
  tests/test_probe_script_onnx.py ^
  tests/test_pipeline_probe_integration.py ^
  --deselect "tests/test_layer2_engines_local.py::*_engine_local_path"
```

Note: `tests/test_layer2_engines_local.py` is not in the list above, so the
`--deselect` is a safety belt per the step instructions; the explicit file list
already excludes those `*_engine_local_path` tests. Baseline before changes was
`2 failed, 52 passed`; after the changes expect `0 failed` with the four new cap
tests added (54+ passed). The `@pytest.mark.slow` real-model label-mapping test
skips cleanly when the model/fixtures are absent — that is expected, not a
failure.

## Verification (first iteration)

Config field sanity:
```
.venv\Scripts\python.exe -c "import os; os.environ['IDP_PROBE_LIPIKA_MAX_WINDOWS']='16'; from src.ai.layer1_probe.types import ProbeConfig; assert ProbeConfig().lipika_max_windows==32; assert ProbeConfig.from_env().lipika_max_windows==16; print('ok')"
```
-> printed `ok` (default 32, env `IDP_PROBE_LIPIKA_MAX_WINDOWS` override works).

Full target suite:
```
.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_linewise.py tests/test_line_crops.py tests/test_probe_script_onnx.py tests/test_pipeline_probe_integration.py -q
```
-> **60 passed, 0 failed, 1 warning in 22.81s**.
- The two previously-failing contract tests now PASS:
  `tests/test_probe_script_onnx.py::test_no_handwriting_field_on_contracts` and
  `tests/test_pipeline_probe_integration.py::test_contract_fields_removed_and_signature_script_only`.
- Baseline before changes was `2 failed, 52 passed`; after changes `0 failed`
  (52 originals + the two fixed contract tests counted in + 6 new cap tests in
  `tests/test_probe_linewise.py`).
- `tests/test_layer2_engines_local.py` is NOT in the file list, so no
  `*_engine_local_path` test runs (the `--deselect` safety belt is unnecessary
  here).
- The `@pytest.mark.slow` real-model label-mapping test skips cleanly (model /
  fixtures absent); the `PytestUnknownMarkWarning` for the `slow` mark is
  pre-existing and unrelated to this change.

New cap tests added to `tests/test_probe_linewise.py`:
`test_cap_respected`, `test_even_spread_across_lines`,
`test_even_spread_uneven_lines_skips_exhausted`, `test_cap_deterministic`,
`test_below_cap_unchanged`, `test_below_cap_unchanged_observe`.

## Notes / assumptions flagged
- The cap (default 32) is >= `lipika_min_windows` (8), so the realistic path is:
  generate N, if N > 32 keep 32; GATE A then checks the capped count against 8.
  This ordering is explicit in item 5(c).
- `_even_spread` is written as a pure helper taking `per_line_windows` and `cap`
  so it is unit-testable in isolation (deterministic/even-spread tests target it
  directly), independent of the ONNX session and OpenCV window generation.
- No change to `square_windows`, `letterbox_square`, `_MAX_LINES` (24-line cap),
  the fp32 default model, or any routing/gate default.
