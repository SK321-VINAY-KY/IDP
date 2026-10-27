# Even-spread window cap + contract-test housekeeping for the lipika observer

This change adds a deterministic, even-across-lines cap on the number of window crops the line-wise lipika observer forwards per page, fixes two stale contract tests that assumed `handwriting_prob` was absent, and adds `*.onnx` to `.gitignore`. The cap is a new `ProbeConfig.lipika_max_windows` field (default 32, env `IDP_PROBE_LIPIKA_MAX_WINDOWS`) applied by a pure `_even_spread` helper that round-robins over the source lines top-to-bottom before the batched forward pass. Routing policy, gate defaults, the handwriting flag, and the fp32 default model are untouched. Verification evidence recorded by the coder reports 60 passed / 0 failed on the target suite, including the four required cap tests plus two extras.

Watch for: a second, unrelated `.gitignore` line (`repomix-layer12.*`) rode along with the `*.onnx` line — the scope asked for the `*.onnx` line and nothing else (confirmed). This is a minor scope deviation, not a behavioral or safety issue.

**Verdict**: APPROVED

## High-level view

The cap lives in `_even_spread(per_line_windows, cap)`, a pure module-level helper. The observer now builds windows grouped per line (`per_line_windows = [square_windows(src) for src in line_sources]`) instead of pre-flattening, then calls `_even_spread` with `self.cfg.lipika_max_windows`. When the total is at or under the cap, it returns the plain flatten in the original line order; when over, it selects exactly `cap` windows by round-robin across lines, skipping exhausted lines. The cap is applied after window generation and before `n_windows`/GATE A, so `lipika_min_windows` evaluates on the capped set and the batched `session.run` only ever sees the kept windows. fp32 and all downstream aggregation/gates are unchanged.

The config addition mirrors the existing `lipika_*` fields exactly: `lipika_max_windows: int = 32` sits directly after `lipika_min_margin`, and `from_env` reads `IDP_PROBE_LIPIKA_MAX_WINDOWS` with the same `int(_env(...))` pattern as the sibling gates. No existing default (`onnx_min_conf` 0.80, `hw_threshold` 0.30, `lipika_min_windows` 8, `lipika_min_conf` 0.60, `lipika_min_margin` 0.20) moved.

The two contract tests were flipped from asserting `handwriting_prob` is absent to asserting it is present on both `SampleObs` and `PageProbe`, matching the intended contract where the field is Optional and VLM-sourced behind `IDP_PROBE_VLM_FOR_HANDWRITING`. The adjacent `hw_source not in PageProbe.model_fields` and the `decide_route` `hw_prob not in sig.parameters` checks were left untouched, which is correct since those fields really are absent.

The new tests exercise the cap through a pure-helper path (exact round-robin ordering on tagged synthetic windows) and through `observe()` with an injected recording fake session that captures the forwarded batch size. No real model, torch, or AWS is involved in any non-skipped test; the only real-model test is `@pytest.mark.slow` and skips cleanly when the model or fixtures are absent.

The one deviation is cosmetic: `.gitignore` gained `repomix-layer12.*` alongside the `*.onnx` line. It does not touch any protected path, routing, model, or git state.

<details>
<summary>Issues (1)</summary>

1. **Unrelated .gitignore line** — `.gitignore` gained `repomix-layer12.*` in addition to the required `*.onnx` line; the scope called for the `*.onnx` line only. Non-blocking; drop the extra line if strict scope adherence matters.

</details>

<details>
<summary>Details</summary>

### Even-spread cap: placement and determinism

`_even_spread` is a pure helper taking the per-line window lists and the cap. For totals at or below the cap it returns `[w for line in per_line_windows for w in line]` — the plain flatten in largest-first line order, so the below-cap path is a genuine no-op on ordering. Over the cap, it walks columns outward (`col` 0, 1, 2 …) and for each column appends one window from each line that still has that column, breaking as soon as `cap` is reached. No RNG, stable list iteration, so identical input yields identical output — the determinism test asserts this directly.

The observer calls `_even_spread` after `per_line_windows` is built and before `n_windows = len(all_windows)`, so GATE A (`n_windows < lipika_min_windows -> None`) sees the capped count, and the batched `np.concatenate([letterbox_square(w) for w in all_windows])` only forwards kept windows. Window-to-line traceability holds because selection is by `(line, col)` position over the preserved line order; the `test_even_spread_across_lines` test confirms the first `len(lines)` kept are one-per-line top-to-bottom and the per-line share stays near-equal rather than front-loading line 0. `test_even_spread_uneven_lines_skips_exhausted` confirms a short line drops out of the rotation once consumed while the cap is still met.

### Config field parity

```python
lipika_min_windows: int = 8
lipika_min_conf: float = 0.60
lipika_min_margin: float = 0.20
lipika_max_windows: int = 32         # new
```

and in `from_env`:

```python
c.lipika_max_windows = int(_env("LIPIKA_MAX_WINDOWS", str(c.lipika_max_windows)))
```

Placement, type, and the `IDP_PROBE_` prefix match the sibling gates. The default 32 sits well above `lipika_min_windows` 8, so the realistic path is "generate N, cap to 32 if over, then check >= 8" — the cap cannot starve GATE A below its own threshold unless `lipika_max_windows` is explicitly set below `lipika_min_windows`, which is an operator choice, not a regression.

### Contract-test correction

Both edited tests now assert `handwriting_prob` is present on `SampleObs` and `PageProbe` and leave the `hw_source`/`hw_prob`-absence assertions in place. The field is re-added behind the handwriting flag (commit 705fe37 is HEAD), so the present-assertions match the shipped contract. No test was renamed, and no third test or unrelated assertion was touched — the diff is exactly the two-line flip in each file.

### Test seams: no real model / torch / AWS

The non-slow cap tests use `_recording_session`, a fake `InferenceSession` whose `run()` records `feeds[...].shape[0]` into `batch_sizes` and returns tiled canned logits. `test_cap_respected` asserts both `last_n_windows` and the recorded batch size equal `lipika_max_windows`; `test_below_cap_unchanged_observe` asserts the forwarded count equals the observer's window count (no selection). The helper-level tests operate on tagged 1-element arrays so ordering is checked exactly, independent of OpenCV windowing. The real-model label-mapping test is `@pytest.mark.slow` and guards on `os.path.exists(_MODEL)` / fixture presence, skipping cleanly otherwise.

### Protected paths, model, and git state

No tracked or staged changes exist under `docker/`, `step_functions/`, or `src/ai/layer1_routing/` (confirmed: `git diff HEAD --stat` over those paths is empty; the sole untracked `step_functions/*.asl.json` predates this change and is not part of the lipika-cap diff). `routing.py` is unchanged. Nothing is staged, no `.onnx` is staged/committed/deleted, and the fp32 default model is unchanged. All HARD CHECKS pass.

### Verification evidence (not re-run)

The plan file's Verification section records the config sanity one-liner printing `ok` (default 32, env override to 16) and the target suite at `60 passed, 0 failed, 1 warning`, with the two contract tests now passing and six cap-related tests added to `tests/test_probe_linewise.py` (the four required plus `test_even_spread_uneven_lines_skips_exhausted` and `test_below_cap_unchanged_observe`). The `slow` label-mapping test skips cleanly. Evidence is present and specific; no suite re-run was warranted.

</details>

<details>
<summary>File map</summary>

- `.gitignore` — added `*.onnx` under a "Model binaries" comment; also added an unrelated `repomix-layer12.*` line (out of stated scope).
- `src/ai/layer1_probe/types.py` — added `lipika_max_windows: int = 32` to `ProbeConfig` and its `IDP_PROBE_LIPIKA_MAX_WINDOWS` override in `from_env`.
- `src/ai/layer1_probe/script_classifier.py` — added the `_even_spread` round-robin cap helper; observer builds per-line window lists and applies the cap before GATE A and the batched forward pass; fp32 and aggregation unchanged.
- `tests/test_probe_script_onnx.py` — flipped `handwriting_prob` assertion in `test_no_handwriting_field_on_contracts` to present (plus the line-wise test-harness updates already on HEAD's feature line).
- `tests/test_pipeline_probe_integration.py` — flipped `handwriting_prob` assertion in `test_contract_fields_removed_and_signature_script_only` to present.
- `tests/test_probe_linewise.py` — added cap tests (`test_cap_respected`, `test_even_spread_across_lines`, `test_even_spread_uneven_lines_skips_exhausted`, `test_cap_deterministic`, `test_below_cap_unchanged`, `test_below_cap_unchanged_observe`).

Full diff: `git diff HEAD -- .gitignore src/ai/layer1_probe/types.py src/ai/layer1_probe/script_classifier.py tests/test_probe_script_onnx.py tests/test_pipeline_probe_integration.py tests/test_probe_linewise.py`

</details>
