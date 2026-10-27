# probe.page_decision observability event + summarizer primary-source rewire

Adds a single additive structured log event (`probe.page_decision`) to `process_page`, gated so it fires only when a probe decision is attached to the page, rewires `summarize_probe_results.py` to treat that event as the primary metrics source (keeping the older `pipeline.page_complete` / `probe.lipika_vs_vlm` paths as fallbacks), ignores `rollback_*.json` so credential-bearing rollback snapshots never get committed, and ships the matching tests, fixtures, and a PROPOSED-only commit plan. No AWS/docker/git-write actions were run beyond the sanctioned clean-worktree simulation. The event carries the FINAL post-ladder engine/confidence/escalated values and no image bytes or extracted text.

Watch for: the `.gitignore` change exceeds the plan's "one line" scope — it adds `*.onnx` and `repomix-layer12.*` in addition to `rollback_*.json` (confirmed; benign, both are additive ignore rules, one of which strengthens the never-stage-.onnx rule). The item-2b escalation test does not assert `engine_used == "vlm_transcribe"` as the plan text expected; the coder documented why (the real `engines_used[0]` convention keeps the primary engine at index 0) and instead proves POST-ladder state via `escalated=True` and `confidence=0.95` (confirmed, and consistent with the plan's own field mapping). Neither is blocking.

**Verdict**: APPROVED

## High-level view

The event is emitted immediately after the existing `pipeline.page_complete` log inside `process_page`, inside an `if "probe" in page_context:` guard. With the probe flag off, `page_context` never gains a `"probe"` key, so the block is skipped and the flag-off path is byte-identical. The event reuses the `document_id` parameter already present on `process_page` (no new plumbing added) and sources `engine_used`/`confidence`/`escalated` from the same `output` object `page_complete` uses, so they are the final post-ladder values rather than the pre-ladder probe plan.

The summarizer makes `probe.page_decision` the primary source for pages-per-route (probe route), per-route escalation, per-script accuracy, the HARMFUL count, and the new near-miss count, while VLM-calls-per-document and lipika-vs-VLM agreement continue to come from `probe.lipika_vs_vlm` joined by `correlation_id` (that event has no `document_id`, and the join is noted inline in the output). The HARMFUL count keys on the FINAL engine after escalation and exits non-zero when any labelled non-Latin page ends on a Paddle engine; the near-miss count (non-Latin pages the probe routed to Paddle but that escalated away) is reported separately and does not affect the exit code. Every metric with no backing records prints an explicit "not available" message instead of a fabricated number.

Scope holds: the only prod source touched by this task is `pipeline.py`; `.gitignore`, `summarize_probe_results.py`, and the tests/fixtures are the remaining edits. `deploy_idp_fargate.py` is untouched (still untracked, never modified). No `.onnx` is staged or tracked, nothing was committed or pushed, and no AWS/docker command ran. The commit plan is proposed only and grounded in the real `git status` XY codes, and the clean-worktree simulation reports all four named suites passing including collection.

<details>
<summary>Issues (3)</summary>

1. **.gitignore scope beyond plan** (confirmed) — the diff adds `*.onnx` and `repomix-layer12.*` on top of the planned `rollback_*.json`; the plan said "change nothing else" for this item and claimed HEAD already ignored `*.onnx` (HEAD does not). Benign and arguably protective, but note it; trim to just `rollback_*.json` if strict plan adherence matters.
2. **Escalation test asserts a weaker POST signal than the plan text** (confirmed) — `test_probe_page_decision_shows_post_ladder_values` proves post-ladder via `escalated=True` and `confidence=0.95` rather than `engine_used=="vlm_transcribe"`, because `engine_used` follows the real `engines_used[0]` convention. Documented and correct; no change required.
3. **Non-probe test edit bundled in** (confirmed) — `tests/test_pipeline_probe_integration.py` flips two `handwriting_prob` contract assertions, which is lipika-observer work, not the page_decision event. It is a test file (within the allowed category) and the commit plan groups it, but it is unrelated to this task's event and rides along.

</details>

<details>
<summary>Details</summary>

### Flag-off byte-identical: the guard and the document_id source

The new block sits right after the existing `pipeline.page_complete` call and before `return output, metadata`, wrapped in `if "probe" in page_context:`. The probe decision dict is written into `page_context["probe"]` only on the probe-enabled path (`_run_document_probe`), so with the flag off the key is absent and the block never runs. `test_probe_page_decision_not_emitted_flag_off` asserts all three of: no `probe.page_decision` event recorded, `"probe" not in pages[0]["context"]`, and `meta.selected_engine == "paddleocr_printed"` unchanged — covering the emission, the context, and the routing output together.

No document-id plumbing was added. The event uses the `document_id` parameter already declared on `process_page` (logged as `document_id or None`), which `process_document` threads on every call site. This keeps the change additive and flag-agnostic: nothing is written into `page_context` on any path, so the flag-off output cannot shift.

### Event correctness and the post-ladder values

Exactly one new event is emitted, named `probe.page_decision`, with all twelve required fields: `document_id`, `page_number`, `probe_route`, `probe_ocr_lang`, `probe_script`, `probe_script_confidence`, `probe_handwriting_prob`, `probe_has_table`, `probe_is_blank`, `engine_used`, `confidence`, `escalated`. The `probe_*` fields map from `page_context["probe"]`; `engine_used`/`confidence`/`escalated` come from `output` — the same object `pipeline.page_complete` reads — so they reflect the state after the escalation `while` loop completes, not the pre-ladder probe plan. No image bytes or text are included, and `test_probe_page_decision_emitted_on_probed_page` explicitly asserts `"image_bytes"`, `"markdown"`, and `"text"` are absent from the event payload.

The required escalating case is `test_probe_page_decision_shows_post_ladder_values`: it forces the primary Paddle engine to return empty text so the ladder escalates and adopts the VLM. The pre-ladder probe route is `paddleocr_printed`, but the event's `escalated` is `True` and `confidence` is `0.95` (the adopted VLM confidence) — values that would be `False`/`0.0` if read pre-ladder. The one deviation from the plan's wording is that the test does not assert `engine_used == "vlm_transcribe"`; the pipeline's `set_final_result` keeps the primary engine at `engines_used[0]` and appends the escalation engine, so `engine_used = output.engines_used[0]` stays `paddleocr_printed`. The event's field mapping matches the plan's own spec line (`engine_used = output.engines_used[0]`), and `escalated`/`confidence` carry the post-ladder proof. The coder recorded this in the verification note rather than silently diverging. No routing, ladder, threshold, or existing event was altered — the diff is purely the additive logging block.

### Summarizer rewire

`probe.page_decision` is parsed per `(doc, page)` into pre-ladder route, final engine, escalated flag, and probe script, keyed by `correlation_id` via `_doc_key`. Pages-per-route prefers the probe route and falls back to the `pipeline.page_complete` final-engine route when no decision records exist; a separate pages-per-FINAL-engine distribution is always shown. Per-route escalation is computed from the decision records (fallback to `page_complete`'s `escalated`). VLM-calls-per-document and lipika-vs-VLM agreement still come from `probe.lipika_vs_vlm`, joined by `correlation_id` because that event lacks a `document_id`; the output states this join inline. Per-script accuracy prefers `probe_script` and falls back to the detected script from `lipika_vs_vlm`, returning `None` (and printing "cannot join labels") when no labelled doc appears.

The HARMFUL count uses the FINAL engine (decision preferred, else `page_complete`) so a page that escalated but stayed on a Paddle engine is still counted; `>0` sets exit code 1 and lists the offending rows. The near-miss count is a distinct helper: non-Latin pages the probe routed to a Paddle engine (`pd_route`) that escalated (`pd_escalated`) to a non-Paddle final engine. It is rendered in its own block, explicitly described as "reported, NOT harmful", and does not touch the exit code. When neither event with the needed field is present, the render path prints `_NOT_AVAILABLE` and, for near-miss specifically, "near-miss requires probe.page_decision".

The fixture `logs_page_decision.ndjson` exercises exactly the required mix: p1 printed non-escalated (harmful, label devanagari), p2 printed escalated-but-still-Paddle (harmful after escalation), p3 printed escalated-away-to-VLM (near-miss, not harmful), p4 routed-to-VLM (clean). The CLI tests assert `HARMFUL ...: 2`, non-zero exit, and `escalated away: 1`. `test_cannot_compute_when_no_page_decision_and_no_page_complete` and `test_render_cannot_compute_message_when_no_page_complete` cover the not-available path. The test file adds ten cases on top of the retained clean-fixture tests, matching the verification note's +10.

### Scope, .gitignore, and git safety

The `.gitignore` diff adds three entries: `rollback_*.json` (the planned secret-snapshot ignore), `*.onnx` with a "never commit" comment, and `repomix-layer12.*`. The plan's item-1 scoped this to a single line and asserted HEAD already carried `*.onnx` — HEAD does not (`git show HEAD:.gitignore` has no onnx/rollback/repomix-layer12 match), so the working tree is adding it. Both extra lines are additive ignore rules; `*.onnx` reinforces the hard rule that `.onnx` models are never staged, and nothing is thereby un-ignored or deleted. Confirmed no `rollback_*.json` is tracked or in history, and `check-ignore` now matches it. No `.onnx` is tracked (`git ls-files "*.onnx"` empty) or staged (`git diff --cached` empty). `deploy_idp_fargate.py` remains untracked and unmodified.

The commit plan is proposed only, ordered a–e, and each named file maps to a real `git status --porcelain` XY code; it correctly flags the absent `settings.py` S3 change rather than fabricating one and notes `line_crops.py` is already tracked-clean. The clean-worktree simulation (worktree add → copy post-(a)+(b)+(c) files → pytest → worktree remove) is the only git-write action and the verification note records 65 passed across the four named suites including collection, with the main tree's porcelain unchanged afterward except the intended edits. Per instructions these recorded pass counts are read, not re-run; the evidence is internally consistent and left no articulable doubt warranting a re-run.

</details>

<details>
<summary>File map</summary>

- `src/ai/layer1_routing/pipeline.py` — one additive `probe.page_decision` `logger.info` block in `process_page`, guarded by `"probe" in page_context`, after `pipeline.page_complete`.
- `.gitignore` — adds `rollback_*.json` (planned) plus `*.onnx` and `repomix-layer12.*` (beyond plan scope, benign).
- `scripts/summarize_probe_results.py` — `probe.page_decision` primary source; pages-per-route, per-route escalation, per-script accuracy, HARMFUL (final engine, non-zero exit) and separate near-miss; fallbacks + explicit not-available messaging.
- `tests/test_pipeline_probe_document.py` — +3 cases: emitted-on-probed-page, post-ladder escalation, flag-off no-event.
- `tests/test_summarize_probe_results.py` — +10 cases incl. harmful>0 exit, near-miss, cannot-compute.
- `tests/fixtures/probe/logs_page_decision.ndjson`, `tests/fixtures/probe/labels_page_decision.csv` — new fixtures exercising harmful, near-miss, and clean routes.
- `tests/test_pipeline_probe_integration.py` — two `handwriting_prob` contract assertions flipped (lipika-observer concern, unrelated to this event).

Full diff: `git diff` on branch `feat/llm-observability`.

</details>
