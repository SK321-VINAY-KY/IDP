# probe.page_decision — verification note

Branch: `feat/llm-observability`  Venv: `.venv\Scripts\python.exe`
First iteration (no `probe-page-decision-review.json` present).

## What changed (allowed files only)
- `.gitignore` — added one line `rollback_*.json` under the secrets block (item 1).
- `src/ai/layer1_routing/pipeline.py` — added ONE additive `probe.page_decision`
  `logger.info` in `process_page`, immediately AFTER `pipeline.page_complete`,
  guarded by `if "probe" in page_context:` (item 2). No other prod source edited.
  `scripts/deploy_idp_fargate.py` NOT touched.
- `scripts/summarize_probe_results.py` — `probe.page_decision` is now the PRIMARY
  source; `pipeline.page_complete` kept as fallback; added near-miss metric (item 3).
- Tests: `tests/test_pipeline_probe_document.py` (+3), `tests/test_summarize_probe_results.py` (+10),
  new fixtures `tests/fixtures/probe/logs_page_decision.ndjson` + `labels_page_decision.csv`.

## document_id identifier + plumbing
- Used the EXISTING `process_page(document_id: str = "")` parameter, which
  `process_document` already threads on every call site (serial loop + `_worker`).
- NO plumbing was added. `page_context["document_id"]` was NOT introduced; it was
  not needed. `correlation_id` rides along automatically via the logger ContextVar.
- Event logs `document_id or None`.

## Item 1 — rollback_*.json checks (commands + results)
- `git ls-files "rollback_*.json" "**/rollback_*.json"` -> EMPTY (none tracked).
- `git log --all --oneline --name-only -- "rollback_*.json" "**/rollback_*.json"` -> EMPTY (none in history).
- `git check-ignore rollback_20260101.json` -> prints the path (now ignored).
- `git status --porcelain .gitignore` -> ` M .gitignore`.
- No rollback snapshot is tracked/in history, so no STOP was needed.

## CONTRADICTION found vs the plan (recorded, non-blocking, no behavior change)
- The plan's item-2b case (b) expected the event's `engine_used == "vlm_transcribe"`
  after an escalation. The ACTUAL pipeline keeps the primary engine at
  `engines_used[0]` and APPENDS the adopted escalation engine, so after a
  paddle->VLM escalation `engines_used == ["paddleocr_printed", "vlm_transcribe"]`
  and `engines_used[0]` stays `paddleocr_printed`. This is the SAME convention the
  existing `pipeline.page_complete` and `PageMetadata.set_final_result` already use
  (`engines_used[0]`), and the plan's own field mapping is
  `engine_used = output.engines_used[0]`. So the IMPLEMENTATION follows the
  specified field mapping; only the plan's test EXPECTATION was based on a wrong
  belief about list ordering.
- Resolution: the required pre!=post escalation test asserts the POST-ladder truth
  that IS observable — `escalated` flips True and `confidence` becomes the adopted
  VLM confidence (0.95), while `engine_used` follows the existing
  `engines_used[0]` convention. All three event values come from `output` AFTER the
  ladder, satisfying "engine_used/confidence/escalated are FINAL post-ladder values".
  No routing/ladder/threshold/existing-event change was made.

## Summarizer metrics that CANNOT be computed from logs (reported, not invented)
- Near-miss requires `probe.page_decision` (the probe's PRE-ladder route). With only
  `pipeline.page_complete`, the pre-ladder route is unknown, so the script prints
  "not available: near-miss requires probe.page_decision" instead of a number.
- VLM calls per document + lipika-vs-VLM agreement come ONLY from
  `probe.lipika_vs_vlm`, joined to a document by `correlation_id` (that event has no
  `document_id`). This join is stated inline in the report output.
- Per-script accuracy returns the explicit "cannot join labels" message when no
  labelled doc appears in the logs.
- When neither `probe.page_decision` nor `pipeline.page_complete` is present,
  pages-per-route and escalation print the `_NOT_AVAILABLE` message.

## Test counts (MAIN tree, `.venv\Scripts\python.exe -m pytest`)
- `tests/test_pipeline_probe_document.py`: BEFORE 11 passed -> AFTER 14 passed (+3).
- `tests/test_summarize_probe_results.py`: BEFORE 11 passed -> AFTER 21 passed (+10).
- Combined run `test_pipeline_probe_document.py + test_pipeline_probe_integration.py + test_summarize_probe_results.py`: 42 passed.

## Item 4 — clean-worktree SIMULATION (the only git write action)
- `git worktree add c:\Users\Dell\Desktop\IDP\idp-clean HEAD` (detached @ acd90d1).
- Copied into the worktree (post-(a)+(b)+(c) mirror): `windows.py`,
  `vlm_block_rollup.py`, observer edits (`inprocess.py`, `script_classifier.py`,
  `types.py`), item-2/3 edits (`pipeline.py`, `summarize_probe_results.py`,
  `test_pipeline_probe_document.py`, `test_pipeline_probe_integration.py`,
  `test_summarize_probe_results.py`, `test_probe.py`) and `tests/fixtures/probe/*`.
  No `.onnx` copied.
- Ran in the worktree:
  `pytest tests/test_probe.py tests/test_probe_inprocess_images.py tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py`
  -> **65 passed**, including COLLECTION (no `ModuleNotFoundError:
  src.ai.layer2_conversion.vlm_block_rollup`).
- `git worktree remove --force` -> removed; worktree dir gone (`Test-Path` False).
- Main-tree `git status --porcelain` after sim shows ONLY the intended edits
  (` M .gitignore`, ` M src/ai/layer1_routing/pipeline.py`,
  ` M tests/test_pipeline_probe_document.py`, `?? scripts/summarize_probe_results.py`,
  `?? tests/fixtures/`, `?? tests/test_summarize_probe_results.py`). No sim leakage.

## Git actions performed
- `git status --porcelain`, `git ls-files`, `git log` (read-only), `git check-ignore`,
  and the item-4 worktree add/remove ONLY. NO commit, NO push, NO staging. No `.onnx`
  staged/committed/tracked/deleted.
