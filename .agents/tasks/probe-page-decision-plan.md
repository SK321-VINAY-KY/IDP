# Implementation Plan — probe.page_decision event + summarizer primary source

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a`  Branch: `feat/llm-observability`
Venv python (ALL pytest): `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`

## Absolute rules carried into this plan
- NO AWS, NO docker, NO `git push`, NO `git commit`. Commit plan (item 4) is PROPOSED only.
- NO `.onnx` staged/committed/tracked/deleted ever.
- The ONLY git action performed is the item-4 clean-worktree SIMULATION (`git worktree add` → pytest → `git worktree remove`). No commit, no push.
- Flag-off behavior stays BYTE-IDENTICAL: the new event fires ONLY when `page_context` contains a `"probe"` key.
- Allowed edits ONLY: `src/ai/layer1_routing/pipeline.py`, `.gitignore`, `scripts/summarize_probe_results.py`, and test files + fixtures. No other prod source. Do NOT touch `scripts/deploy_idp_fargate.py`.

## Findings confirmed against the actual code (grounding)
- `process_page(...)` already has `document_id: str = ""` as a parameter (pipeline.py ~line 215) and `process_document` passes it through on every call site (serial loop and `_worker`). **No document-id plumbing is needed.** The identifier to use for the event is the `document_id` parameter already in scope.
- Every event emitted through `logger.info(...)` automatically carries `correlation_id` (from a `ContextVar` in `src/utils/logger.py`) and `service`. So `probe.page_decision` will automatically include `correlation_id` too; we only add `document_id` explicitly (the existing `probe.lipika_vs_vlm` carries `correlation_id` but no `document_id`, which is why the summarizer joins VLM calls by `correlation_id`).
- The FINAL post-ladder values are the local vars present immediately before the existing `pipeline.page_complete` log (pipeline.py ~line 597-628): `engines_used` (final list; `engines_used[0]` is the final selected engine per `set_final_result`), `confidence` (float, rounded to 4 in `output.confidence`), and `escalated` (bool). These are POST-ladder because the `while` escalation loop has already completed. The `probe.page_decision` event MUST be emitted right after `pipeline.page_complete`, using `output.engines_used`, `output.confidence`, `output.escalated` (identical source to `page_complete`).
- `page_context["probe"]` dict shape (set in `_run_document_probe`, pipeline.py ~line 690): keys `route, ocr_lang, script, script_confidence, handwriting_prob, has_table, is_blank, route_reason`. The event's `probe_*` fields map from these (route→probe_route, ocr_lang→probe_ocr_lang, script→probe_script, script_confidence→probe_script_confidence, handwriting_prob→probe_handwriting_prob, has_table→probe_has_table, is_blank→probe_is_blank). `route_reason` is NOT one of the 12 required fields and is omitted.
- `engine_used` (singular, per the task spec field list) = the final selected engine = `output.engines_used[0] if output.engines_used else None` (same value `set_final_result` records). The `engines_used` list distribution is a separate summarizer concern, not an event field.
- The summarizer `scripts/summarize_probe_results.py` currently derives pages-per-route and escalation from `pipeline.page_complete` (final engine → route via `_ENGINE_TO_ROUTE`), VLM calls + agreement from `probe.lipika_vs_vlm`, and accuracy/harmful from the detected script joined to labels by `(doc, page)`. It groups by `correlation_id` (`_doc_key`). Item 3 makes `probe.page_decision` the PRIMARY source while keeping the `probe.lipika_vs_vlm` joins.

## CONTRADICTIONS vs the description (none blocking — recorded)
- None. The description allows adding `document_id` to `page_context` "if a document_id is genuinely NOT in scope"; it IS in scope as a `process_page` parameter, so no plumbing is added. This is noted so the implementer does not add unnecessary plumbing.
- Minor naming note (not a contradiction): the summarizer's existing `page_script` uses the probe's DETECTED script from `probe.lipika_vs_vlm` (vlm_script preferred, else lipika_script). Item 3 adds `probe_script` from `probe.page_decision` as the authoritative probe script for per-script accuracy, but the `probe.lipika_vs_vlm`-based path is kept as a fallback and its existing tests must keep passing. Where both exist for a `(doc,page)`, `probe.page_decision.probe_script` wins.

---

# Items

- [ ] 1. Add `rollback_*.json` to `.gitignore` (deploy rollback snapshots hold UNMASKED task env secrets). First confirm none is tracked or in history (DONE during exploration: `git ls-files "rollback_*.json"` → empty; `git log --all --name-only -- "rollback_*.json" "**/rollback_*.json"` → empty; on-disk glob → none; the only `rollback` match is the Python script `scripts/rollback_probe_service.py`, which is unrelated). Add exactly one line `rollback_*.json` under the existing "Virtual environments & secrets" block (after the `.env*` lines) in `.gitignore`. Change nothing else. If a `rollback_*.json` ever turns up tracked/in history, STOP and report — no history rewrite.
      Files: `.gitignore`
      Verify: `git -C c:\Users\Dell\Desktop\IDP\engineer_a check-ignore rollback_20260101.json` prints the path (ignored); `git -C ... status --porcelain .gitignore` shows ` M .gitignore`; `git ls-files "rollback_*.json"` stays empty.

- [ ] 2. Add exactly ONE additive structured log event `probe.page_decision` in `process_page`, emitted ONLY when `"probe" in page_context`, placed IMMEDIATELY AFTER the existing `logger.info("pipeline.page_complete", ...)` call (pipeline.py ~line 618-628), before `return output, metadata`. Guard: `if "probe" in page_context:`. Build the payload from `page_context["probe"]` for the `probe_*` fields and from `output` for the final post-ladder values. Use `document_id` (the parameter already in scope; log `document_id or None`). Use the module `logger` (already `get_logger(__name__)`), same kwargs call style as neighboring events. Exact fields, in order: `document_id`, `page_number`, `probe_route`, `probe_ocr_lang`, `probe_script`, `probe_script_confidence`, `probe_handwriting_prob`, `probe_has_table`, `probe_is_blank`, `engine_used`, `confidence`, `escalated`. Mapping: `probe_route=_p["route"]`, `probe_ocr_lang=_p.get("ocr_lang")`, `probe_script=_p.get("script")`, `probe_script_confidence=_p.get("script_confidence")`, `probe_handwriting_prob=_p.get("handwriting_prob")`, `probe_has_table=_p.get("has_table")`, `probe_is_blank=_p.get("is_blank")`, `engine_used=output.engines_used[0] if output.engines_used else None`, `confidence=output.confidence`, `escalated=output.escalated`, `page_number=page_number`, where `_p = page_context["probe"]`. NO image bytes, NO extracted text. Do NOT add `route_reason`. Do NOT alter routing, the ladder, thresholds, or any existing event. Do NOT add document-id plumbing (it is already a parameter). Report in the plan/PR that the identifier used is the `document_id` parameter and that NO plumbing was added.
      Files: `src/ai/layer1_routing/pipeline.py`
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py -q` — all pass (existing + the new item-2 tests from item 2b).

- [ ] 2b. Add item-2 tests to `tests/test_pipeline_probe_document.py` (reuses its existing stubs/`_wire`/`_process` harness — do NOT create a new test file; this keeps the StubVLM/`_build_probe_deps` mocking that avoids real Bedrock/onnx). Capture emitted events by monkeypatching `pipeline.logger` with a recorder that appends `(event, kwargs)` for each `info` call (the harness already monkeypatches `pipeline` attributes). Add these cases:
      (a) `test_probe_page_decision_emitted_on_probed_page`: flag ON, a scanned→printed probed page; assert exactly one `probe.page_decision` event recorded; assert its fields equal the probe dict values (`probe_route=="paddleocr_printed"`, `probe_script=="latin"`, etc.), `document_id` matches the id passed to `process_document`, and `engine_used`/`confidence`/`escalated` equal the returned `output` values.
      (b) `test_probe_page_decision_shows_post_ladder_values`: flag ON, a probed page whose primary engine yields LOW confidence so the escalation ladder fires and escalates to `vlm_transcribe` (drive via StubLLM returning high-confidence VLM markdown and the primary engine returning empty/low-confidence text; mirror the escalation setup already used by `test_flag_on_non_latin_routes_vlm_no_escalation` but force a low-confidence primary so pre!=post). Assert the event's `engine_used=="vlm_transcribe"` and `escalated is True` — i.e. POST-ladder, NOT the pre-ladder probe route. This is the REQUIRED pre!=post escalation case.
      (c) `test_probe_page_decision_not_emitted_flag_off`: flag OFF (reuse `test_flag_off_is_byte_identical` wiring); assert NO `probe.page_decision` event recorded AND `"probe" not in page_context` AND `meta.selected_engine` unchanged — flag-off byte-identical.
      Files: `tests/test_pipeline_probe_document.py`
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_pipeline_probe_document.py -q` — all pass including the 3 new cases.

- [ ] 3. Make `probe.page_decision` the PRIMARY source in `scripts/summarize_probe_results.py` while keeping the existing `probe.lipika_vs_vlm` joins. Changes to the `summarize()` function:
      (1) Parse `probe.page_decision` records: for each, record `(doc,page) -> {probe_route, engine_used, escalated, probe_script}` keyed by `_doc_key` (correlation_id). Track `saw_page_decision`.
      (2) pages-per-route: prefer `probe_route` from `probe.page_decision`; also keep/compute the FINAL `engine_used` distribution (route via `_ENGINE_TO_ROUTE`) and report both ("pages per PROBE route" and "pages per FINAL engine"). Fall back to the `pipeline.page_complete` final-engine computation when no `probe.page_decision` records exist (so the existing fixture/tests still pass).
      (3) escalation rate per route: compute directly from `probe.page_decision` as `probe_route -> fraction with escalated==true`; keep the `pipeline.page_complete`-based path as fallback when `probe.page_decision` is absent.
      (4) VLM probe calls per document: UNCHANGED — still from `probe.lipika_vs_vlm` (count of records with `vlm_script` set), grouped by `correlation_id`. Add an inline note in `render()` explaining the correlation_id join (probe.lipika_vs_vlm has no document_id).
      (5) lipika-vs-VLM agreement: UNCHANGED — from `probe.lipika_vs_vlm.agree`.
      (6) `--labels` per-script accuracy: use `probe_script` from `probe.page_decision` as the authoritative probe script when present for a `(doc,page)`, else fall back to the existing detected-script-from-lipika_vs_vlm. Keep `script_accuracy()` returning `None` when no labelled doc joins (the "cannot compute" path).
      (7) HARMFUL count: redefine to use the FINAL `engine_used` from `probe.page_decision` (preferred) OR the `pipeline.page_complete` final engine (fallback) — labelled NON-Latin pages whose FINAL engine is a Paddle engine (`paddleocr_printed`/`paddleocr_handwritten`) EVEN AFTER escalation. Exit NON-ZERO (code 1) if >0, listing offending `(doc,page)` rows with label+engine (keep `harmful_rows()` behavior, sourcing the final engine from page_decision when available).
      (8) NEW near-miss metric: count non-Latin pages the PROBE ROUTED to Paddle (`probe_route` in `{paddleocr_printed, paddleocr_handwritten}`) that ESCALATED AWAY (`escalated==true` AND final `engine_used` NOT a Paddle engine). Report SEPARATELY as "near-misses (reported, not harmful)". Add a helper `near_miss_rows(summary, labels)` and render it after the HARMFUL block; it does NOT affect the exit code.
      (9) "cannot compute" messaging: for any metric with no backing records (e.g. no `probe.page_decision` and no `pipeline.page_complete`), print the explicit `_NOT_AVAILABLE`/"cannot compute" message instead of a number — never invent. Update the module docstring to list `probe.page_decision` as the primary event and its fields.
      Files: `scripts/summarize_probe_results.py`
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_summarize_probe_results.py -q` — existing + new item-3b tests pass.

- [ ] 3b. Update `tests/test_summarize_probe_results.py` and `tests/fixtures/probe/` for the new primary source. Add a NEW fixture `tests/fixtures/probe/logs_page_decision.ndjson` containing `probe.page_decision` records (plus matching `pipeline.page_complete` and `probe.lipika_vs_vlm` for the joins) that exercise: multiple `probe_route` values (pages-per-route), at least one route with mixed escalated true/false (per-route escalation rate), at least one labelled non-Latin page whose final `engine_used` is `paddleocr_printed` even after escalation (HARMFUL>0 → non-zero exit), at least one non-Latin page with `probe_route` Paddle + `escalated=true` + final `engine_used` non-Paddle (near-miss count = reported, not harmful), and a doc whose labels never join (cannot-compute path). Add matching label CSVs if needed (e.g. `labels_page_decision.csv`). Keep the EXISTING fixtures (`logs_clean.ndjson`, `labels_clean.csv`, `labels_harmful.csv`) and their tests UNCHANGED so the fallback paths stay covered. Add tests:
      - `test_pages_per_probe_route` (from probe_route), `test_escalation_rate_per_probe_route`, `test_harmful_after_escalation_still_paddle` (asserts a page that escalated but stayed Paddle is harmful), `test_near_miss_escalated_away_not_harmful` (asserts near-miss counted, exit still governed only by harmful), `test_cli_page_decision_harmful_exits_non_zero`, and `test_cannot_compute_when_no_page_decision_and_no_page_complete`.
      Files: `tests/test_summarize_probe_results.py`, `tests/fixtures/probe/logs_page_decision.ndjson`, `tests/fixtures/probe/labels_page_decision.csv`
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_summarize_probe_results.py -q` — all pass.

- [ ] 4. Produce the ORDERED commit plan below (PROPOSE ONLY — do NOT run any `git add`/`git commit`). Grounded in `git status --porcelain` captured during exploration (XY codes shown). NO `.onnx` in any commit. Then run the clean-worktree SIMULATION (item 4-sim). This item's deliverable is the written plan + the passing simulation; it changes no tracked files.
      Files: (none committed — reporting only)
      Verify: the simulation in item 4-sim passes (all four suites collect and pass in the temp worktree).

- [ ] 4-sim. Clean-worktree simulation (the ONLY git action allowed).
      Steps:
      1. `git -C c:\Users\Dell\Desktop\IDP\engineer_a worktree add c:\Users\Dell\Desktop\IDP\idp-clean HEAD`
      2. Copy into the temp worktree (mirroring post-item-(c) state, i.e. items a+b+c): the UNTRACKED runtime files `src/ai/layer1_probe/windows.py` and `src/ai/layer2_conversion/vlm_block_rollup.py` (item 4a), the lipika observer working-tree EDITS (item 4b: `src/ai/layer1_probe/inprocess.py`, `src/ai/layer1_probe/script_classifier.py`, `src/ai/layer1_probe/types.py`), and the item-2/3 changes (item 4c: `src/ai/layer1_routing/pipeline.py`, `scripts/summarize_probe_results.py`, `tests/test_pipeline_probe_document.py` [post item 2b], `tests/test_pipeline_probe_integration.py` [its working-tree edit], `tests/test_summarize_probe_results.py`, and the `tests/fixtures/probe/` fixtures). Do NOT copy any `.onnx`; the pipeline probe suites stub the observer so no model is needed. Copy `tests/test_probe.py` working-tree edit as well since `test_probe.py` is in the target suite and has a working-tree edit (` M tests/test_probe.py`).
      3. Run: `c:\Users\Dell\Desktop\IDP\idp-clean\... ` is the worktree; run from there with the MAIN venv python:
         `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_inprocess_images.py tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py -q` with cwd = the temp worktree.
      4. Confirm ALL pass INCLUDING COLLECTION — specifically that `tests/test_pipeline_probe_document.py` and `tests/test_pipeline_probe_integration.py` no longer fail collection on `ModuleNotFoundError: src.ai.layer2_conversion.vlm_block_rollup` (that failure in the earlier clean-HEAD run is fixed once item-4a `vlm_block_rollup.py` + `windows.py` are copied in).
      5. `git -C c:\Users\Dell\Desktop\IDP\engineer_a worktree remove c:\Users\Dell\Desktop\IDP\idp-clean --force`; confirm `git -C c:\Users\Dell\Desktop\IDP\engineer_a status --porcelain` is unchanged from the pre-sim snapshot (main tree untouched — the new event/summarizer edits in the MAIN tree remain, nothing from the sim leaked back).
      Files: (temp worktree only; removed at end)
      Verify: all four suites pass incl. collection in the temp worktree; main-tree `git status --porcelain` identical before/after except for the intended item-1/2/3 edits.

---

# Item 4 — ORDERED COMMIT PLAN (PROPOSED ONLY; do NOT execute)

Grounded in `git status --porcelain` (XY codes). Every file named exists in that output. NO `.onnx` appears (the two `.onnx` models remain gitignored by `.gitignore:*.onnx` and ship only via docker COPY).

### Commit (a) — untracked runtime files a fresh HEAD needs to import the probe/pipeline path
Rationale: `pipeline.py:37` imports `src.ai.layer2_conversion.vlm_block_rollup`; the probe path imports `src.ai.layer1_probe.windows`. A clean HEAD fails collection without these.
- `?? src/ai/layer1_probe/windows.py`
- `?? src/ai/layer2_conversion/vlm_block_rollup.py`
- (NOTE: `line_crops.py` is already TRACKED and CLEAN — `git ls-files` returns `src/ai/layer1_probe/line_crops.py`, and it is absent from `git status --porcelain`. So it is NOT an untracked runtime file and needs no commit here. Nothing to add.)
- (Verify no other untracked `src/` import is needed: `vlm_block_rollup.py` ships its own test `?? tests/test_vlm_block_rollup.py` — group that test here or in commit (c); recommend here to keep the runtime file + its test together.)
- `?? tests/test_vlm_block_rollup.py`

### Commit (b) — lipika observer working-tree EDITS (modified, already tracked)
- ` M src/ai/layer1_probe/inprocess.py`
- ` M src/ai/layer1_probe/script_classifier.py`
- ` M src/ai/layer1_probe/types.py`
- ` M src/ai/layer1_probe/` ... plus the observer-related tracked test edits that pair with these:
- ` M tests/test_probe.py`
- ` M tests/test_probe_script_onnx.py`
- ` M scripts/eval_lipika_crops.py` (lipika eval helper tied to observer changes)
- `?? tests/test_probe_linewise.py`, `?? scripts/measure_lipika_linewise.py`, `?? scripts/measure_lipika_cap.py` (new lipika measurement helpers + tests — group with observer work)

### Commit (c) — probe log event + summarizer + tests (items 2/3 of THIS task)
- ` M src/ai/layer1_routing/pipeline.py` (item 2: `probe.page_decision` event)
- ` M .gitignore` (item 1: `rollback_*.json`)
- ` M tests/test_pipeline_probe_integration.py` (existing working-tree edit; plus if item 2b touches it)
- ` M tests/test_pipeline_probe_document.py` is UNTRACKED? No — it is NOT in porcelain as ` M`; it is not listed, meaning it is already clean/tracked. Item 2b edits it → it will become ` M tests/test_pipeline_probe_document.py` after item 2b; include here.
- `scripts/summarize_probe_results.py` — see note: it is `?? scripts/summarize_probe_results.py` (untracked, from the prior deploy-scripts task). Per the task description item (c)/(d) it belongs with the deploy scripts (commit d), BUT item-3 MODIFIES it. Recommendation: commit it ONCE in commit (d) with the deploy scripts, including the item-3 changes (since it is a single new untracked file, not a tracked edit). Note explicitly in commit (c) that the summarizer lives in commit (d).
- `?? tests/test_summarize_probe_results.py` (untracked) + `?? tests/fixtures/` (the whole fixtures dir incl. `probe/*.ndjson`, `*.csv`, and the new `logs_page_decision.ndjson`) — group with the summarizer in commit (d) OR here; recommend (d) to keep summarizer + its tests/fixtures together. NO `.pdf`/`.md` fixtures (gitignored).

### Commit (d) — deploy + measurement scripts (new untracked)
- `?? scripts/deploy_probe_image.py`
- `?? scripts/deploy_probe_service.py`
- `?? scripts/rollback_probe_service.py`
- `?? scripts/summarize_probe_results.py` (INCLUDING item-3 edits)
- `?? tests/test_summarize_probe_results.py`
- `?? tests/test_deploy_probe_image.py`
- `?? tests/test_deploy_probe_service.py`
- `?? tests/fixtures/probe/` (NDJSON + CSV fixtures only; NO `.pdf`/`.md`/`.onnx`)
- the untracked `deploy_idp_*.py` helpers the probe scripts import: `?? scripts/deploy_idp_ecr_images.py`, `?? scripts/deploy_idp_fargate.py`, `?? scripts/deploy_idp_lambdas.py`, `?? scripts/deploy_idp_s3_rds.py` (do NOT EDIT `deploy_idp_fargate.py`; committing the untracked file as-is is allowed and required so the probe scripts can import it in a fresh checkout)
- `?? step_functions/state_machine_with_probe.asl.json` (deploy artifact — group here)

### Commit (e) — EACH unrelated change as its OWN commit
- (e1) layer3 fix: ` M src/ai/layer3_extraction/extractor.py` + its test ` M tests/test_layer3_hardening.py`
- (e2) S3 settings: (the brief says `settings.py`) — NOT present in porcelain. CONTRADICTION note: there is NO `src/config/settings.py` change in `git status`. Do NOT invent one. If an S3 settings change was expected, it is absent; report that in the PR and skip.
- (e3) sentence-transformers pin: ` M requirements.txt` (+ `?? requirements-dev.txt` if related)
- (e4) schema_chatbot pipeline test: ` M schema_chatbot_v2/tests/test_pipeline_jobs.py`
- (e5) AGENTS.md / .agents docs: ` M AGENTS.md`, ` M .agents/rules/aws_tagging.md`, ` M .agents/tasks/lipika-eval-results.md`, ` M .agents/tasks/task-probe-inprocess-routing/features/FEAT-001.json`, ` M .agents/tasks/task-probe-inprocess-routing/features/FEAT-002.json`, and the many `?? .agents/tasks/*.md` / `*.json` planning docs (group docs/planning artifacts here; these are notes, not code).

### Explicitly EXCLUDED from all commits
- `*.onnx` models (gitignored; ship via docker only).
- `rollback_*.json` (now gitignored by item 1; none exists anyway).
- Scratch/output dirs already gitignored: `?? .scratch_probe/`, `?? dataset/__MACOSX/`, `?? dataset/verified_twice/`, `?? docs/` — review before committing; `.scratch_probe/` and `dataset/` artifacts should NOT be committed (likely large/scratch). `?? docs/` — decide per content; default EXCLUDE unless the user wants docs tracked.

### XY-code contradictions found (reported, non-blocking)
- `settings.py` S3 change named in the brief is ABSENT from `git status` — do not fabricate a commit for it.
- `line_crops.py` named in the earlier brief is already TRACKED and CLEAN (`git ls-files` shows `src/ai/layer1_probe/line_crops.py`; absent from porcelain) — it is NOT an untracked runtime file, nothing to commit.
- `scripts/summarize_probe_results.py` is UNTRACKED (`??`), not a tracked edit; item 3 modifies the untracked file, so it is committed once (in commit d) with the item-3 changes rather than as a separate "modify" commit.

---

# Report (per-file summary)
- `.gitignore` — add one line `rollback_*.json`. No existing entry; no rollback snapshot tracked/in history (confirmed).
- `src/ai/layer1_routing/pipeline.py` — add ONE `probe.page_decision` `logger.info` in `process_page` right after `pipeline.page_complete`, guarded by `"probe" in page_context`, using the in-scope `document_id` param (NO plumbing added) and the final post-ladder `output.*` values. Purely additive; flag-off path untouched.
- `scripts/summarize_probe_results.py` — `probe.page_decision` becomes primary for pages-per-route, per-route escalation, per-script accuracy, and harmful (post-escalation); adds near-miss metric; keeps `probe.lipika_vs_vlm` for VLM-calls/agreement joined by correlation_id; keeps fallbacks + cannot-compute messaging.
- Tests: `tests/test_pipeline_probe_document.py` (+3 cases incl. escalation pre!=post and flag-off no-event), `tests/test_summarize_probe_results.py` (+6 cases), new fixtures `tests/fixtures/probe/logs_page_decision.ndjson` (+ label CSV).
- Identifier used for `document_id`: the existing `process_page(document_id=...)` parameter. NO plumbing added. `correlation_id` rides along automatically via the logger ContextVar.

# Test counts (expected after implementation)
- `tests/test_pipeline_probe_document.py`: existing 12 + 3 new = 15.
- `tests/test_summarize_probe_results.py`: existing 10 + ~6 new = ~16.
- Clean-worktree sim suite: `test_probe.py` + `test_probe_inprocess_images.py` (44 stub tests) + the two pipeline-probe suites collect and pass (collection fix via item-4a files).
