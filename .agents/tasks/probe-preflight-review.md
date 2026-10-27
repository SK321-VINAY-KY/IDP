# Extend probe.page_decision with engines_used/final_engine + preflight

The change adds two fields to the `probe.page_decision` log event in `process_page` — `engines_used` (the full post-ladder engine list) and `final_engine` (the engine whose output was actually adopted after the escalation ladder) — while leaving `engine_used` as the unchanged primary. The summarizer is updated to key the HARMFUL count off `final_engine` (with a documented 3-level fallback), and tests plus fixtures exercise the printed→VLM escalation case. The rest of the delivery is read-only preflight: test counts, model-in-build-context verification, a dry-run of the image builder, and a command list derived from the real argparse. Nothing was committed, pushed, built, or deployed, and no `.onnx` was staged.

Watch for: nothing blocking. One cosmetic note (confirmed) — the HARMFUL report line still reads "routed to a Paddle engine" while the number is now computed from the FINAL engine; the code comments this deliberately to keep wording stable for existing tooling. (likely-intended) The escalation summarizer fixture packs HARMFUL=1 and near-miss=1 into one doc rather than a HARMFUL=0/near-miss=1 split, which is a stronger assertion than the brief's phrasing but satisfies its intent.

**Verdict**: APPROVED

## High-level view

The `final_engine` value is derived, not guessed. The coder confirmed no single existing post-ladder field holds the adopted engine (`engines_used[0]` is the primary, `engines_used[-1]` misreports the VLM-loses case, `PageMetadata.final_engine` equals the primary), so a new local tracks the adopted engine at the three ladder branch points. This is additive observability state that touches no routing, output, or metadata — the correct outcome, not a STOP-and-report.

Flag-off stays byte-identical: every new field lives inside the existing `if "probe" in page_context:` guard, and the `final_engine` local only feeds that event. With the probe flag off, `page_context` carries no `probe` key and the block is skipped.

The summarizer now computes HARMFUL from the final adopted engine via a `final_engine → engines_used[-1] → engine_used` fallback, preserving back-compat with log lines that predate the new fields. The near-miss metric (probe routed to Paddle but escalated away) is unchanged and reported separately without affecting the exit code; cannot-compute messaging is intact.

The API surface change is confined to two additive event keys. Item 4's command list matches the real argparse: `deploy_probe_image.py` takes only `--dry-run`/`--region` and emits the URI, `--image`/`--flag-off` belong to `deploy_probe_service.py`, and `rollback_probe_service.py` requires exactly one of `--rollback-file`/`--task-definition-arn`. The note flags that `.dockerignore` excludes `.env*` but not `*.onnx`, `*.json`, `rollback_*.json`, or `.secrets.baseline`.

<details>
<summary>Issues (2)</summary>

1. **HARMFUL line wording** — the report line says "routed to a Paddle engine" but the count is the FINAL post-escalation engine; non-blocking, already commented in code. Optionally align the wording with the final-engine semantics.
2. **Fixture packs harmful+near-miss in one doc** — `logs_final_engine.ndjson` asserts HARMFUL=1 and near-miss=1 together rather than the brief's HARMFUL=0/near-miss=1 split; this is a superset assertion (proves both the non-zero exit and the near-miss), acceptable as-is.

</details>

<details>
<summary>Details</summary>

### Additive-only event extension in process_page

The diff against HEAD shows the `probe.page_decision` emit unchanged in shape except for two appended kwargs. `engine_used` remains `output.engines_used[0] if output.engines_used else None` — the primary, byte-for-byte as before. The new `engines_used=output.engines_used` carries the full list, and `final_engine=final_engine` carries the adopted-winner local. No existing field was removed or renamed. `document_id` is a real parameter of `process_page` (`document_id: str = ""`), so the `document_id=document_id or None` field resolves in scope.

The `final_engine` local is initialized to `engines_used[0]` right after the initial merge and updated at exactly the ladder adoption points: `final_engine = escalation_engine` in the VLM-wins branch, left unchanged in the VLM-loses branch (so the prior OCR engine stays the adopted one even though `vlm_transcribe` is appended to `engines_used`), and `final_engine = engines_used[0]` after the OCR-to-OCR re-merge. This matches the plan's CRITICAL FINDING: the adopted engine is deterministic at those branch points and is not recoverable from any single existing field, so tracking it in a local is the additive, behavior-preserving fix rather than a STOP.

### Flag-off byte-identical guarantee

The entire new block is wrapped in `if "probe" in page_context:`, identical to how the event was already guarded. The `final_engine` local is computed unconditionally but is read only inside that guard and never feeds routing, `PageOutput`, `set_final_result`, or the `pipeline.page_complete` event. The flag-off test (`test_probe_page_decision_not_emitted_flag_off`) asserts no `probe` key, no event, and unchanged `selected_engine`.

### Summarizer HARMFUL keyed off final_engine

In `summarize()`, the `probe.page_decision` branch stores the adopted engine into `pd_final_engine[key]` using `rec["final_engine"]`, falling back to `engines_used[-1]` for a non-empty list, then to the legacy `engine_used`. A comment states explicitly that `engine_used` is the primary under the current pipeline and is used here only as the last-resort back-compat path. `page_engine` (consumed by `harmful_rows`) is `final_engine_by_page` with `pd_final_engine` winning on overlap, so HARMFUL = labelled non-Latin page whose final engine is in `_PADDLE_ENGINES`. `render()` sets `exit_code = 1` and prints each offending `doc=/page=/label=/engine=` when rows exist.

`near_miss_rows` is preserved verbatim: non-Latin page whose probe route was Paddle, escalated, with a non-Paddle final engine. It is reported in its own block and does not touch the exit code. The cannot-compute paths (`_NOT_AVAILABLE`, `script_accuracy` returning None, and the "near-miss requires probe.page_decision" message) are intact.

One cosmetic mismatch: the HARMFUL header still reads "labelled non-Latin page routed to a Paddle engine" while the number is now the final post-escalation engine. The code carries a comment that the wording is kept stable for existing tooling and that the semantics are the final engine. Not blocking.

### Test coverage

Event tests (`tests/test_pipeline_probe_document.py`): the no-escalation test asserts `engines_used == ["paddleocr_printed"]` and `final_engine == "paddleocr_printed"` alongside the existing `engine_used == engines_used[0]` and no-leak checks. The post-ladder test asserts `escalated is True`, `final_engine == "vlm_transcribe"`, and `engine_used == engines_used[0]` (still the paddle primary). The new `test_probe_page_decision_printed_escalates_to_vlm_final_engine` is the requested escalation case: primary `paddleocr_printed` returns empty text, ladder adopts VLM, and it asserts `engines_used == ["paddleocr_printed", "vlm_transcribe"]`, `final_engine == "vlm_transcribe"`, `engine_used == engines_used[0] == "paddleocr_printed"`, `escalated is True`.

Summarizer tests: `test_final_engine_harmful_and_near_miss` uses `logs_final_engine.ndjson` (docF=devanagari) where p1 has `final_engine=paddleocr_printed` (HARMFUL) and p2 has `engine_used=paddleocr_printed` but `final_engine=vlm_transcribe` (near-miss, not harmful) — p2 is the key record proving the summarizer keys off `final_engine`, not `engine_used`. `test_final_engine_fallback_to_engines_used_last` and `test_final_engine_fallback_to_engine_used` cover the two fallback rungs. `test_cli_final_engine_harmful_exits_non_zero` asserts the non-zero exit with HARMFUL=1 and "escalated away: 1".

The brief phrased the escalation fixture as "HARMFUL==0 near-miss==1 AND a final_engine-is-Paddle non-Latin record proves HARMFUL>0". The delivered fixture instead carries both a harmful page and a near-miss page in the same doc, so a single run proves HARMFUL=1 (non-zero exit) and near-miss=1. That is a stronger combined assertion than the split phrasing and meets the intent.

Not re-run: per instruction, the four probe suites and the summarizer suite were not re-executed. The note records 91 passed (31 + 13 + 15 + 7 + 25), collection succeeded. Trusted as recorded; nothing in the diff gives articulable reason to doubt it.

### Constraints and preflight

Confirmed against the working tree: nothing staged (`git diff --cached` empty); `src/config/settings.py` and `schema_chatbot_v2/Dockerfile` show no diff vs HEAD; `git ls-files "*.onnx"` is empty. The only files changed by this task are `src/ai/layer1_routing/pipeline.py`, `scripts/summarize_probe_results.py`, `tests/test_pipeline_probe_document.py`, `tests/test_summarize_probe_results.py`, and the two new fixtures. Deploy scripts and `deploy_idp_fargate.py` are untracked artifacts from prior tasks, not edited here. `.dockerignore` contains `.env*` but no `*.onnx`/`*.json`/`rollback_*.json` pattern; `.gitignore` contains `rollback_*.json`.

Item 4 command names match the real argparse per the plan/note: `deploy_probe_image.py` (`--dry-run`, `--region`, emits URI; no `--image`), `deploy_probe_service.py` (`--image` required, `--flag-off`, `--cluster`/`--service`/`--region`/`--dry-run`), `rollback_probe_service.py` (mutually-exclusive required `--rollback-file` XOR `--task-definition-arn`), `summarize_probe_results.py` (`--logs` required, `--labels`). Item 3 preflight (3a counts, 3b model sizes+sha256 and dockerignore status, 3c dry-run determination, 3d Dockerfile `COPY . /app`) and item 5 secret-file list (names only, with dockerignore status for `.env*`, `.secrets.baseline`, `rollback_*.json`) are all present in the note.

</details>

<details>
<summary>File map</summary>

- `src/ai/layer1_routing/pipeline.py` — added `final_engine` tracking local through the escalation ladder; extended `probe.page_decision` with `engines_used` and `final_engine` (additive, inside the probe guard).
- `scripts/summarize_probe_results.py` — HARMFUL/per-final-route now keyed off `final_engine` with a `final_engine → engines_used[-1] → engine_used` fallback; near-miss and cannot-compute messaging preserved.
- `tests/test_pipeline_probe_document.py` — updated two event tests, added `test_probe_page_decision_printed_escalates_to_vlm_final_engine`.
- `tests/test_summarize_probe_results.py` — added final_engine harmful/near-miss, two fallback tests, and a CLI non-zero-exit test.
- `tests/fixtures/probe/logs_final_engine.ndjson`, `tests/fixtures/probe/labels_final_engine.csv` — new escalation fixture (docF=devanagari; p1 harmful, p2 near-miss).

Full diff: `git diff HEAD` plus the untracked `scripts/summarize_probe_results.py`, `tests/test_summarize_probe_results.py`, and `tests/fixtures/probe/*_final_engine.*`.

</details>
