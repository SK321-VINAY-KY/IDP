# Probe Preflight + Additive-Change Implementation Plan

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a` · branch `feat/llm-observability`
Venv python for ALL pytest: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`

## Absolute rules (restated — any violation is a failure)
- NO `git commit`, NO `git push`, NO AWS command (no aws cli, no boto3 to live services),
  NO `docker build`/`docker push`, NO staging/commit/track of any `.onnx` file.
- Edit ONLY: `src/ai/layer1_routing/pipeline.py` (additive only),
  `scripts/summarize_probe_results.py`, and test files + fixtures under `tests/`.
- Do NOT edit `deploy_idp_fargate.py`, `settings.py`, any deploy script, the Dockerfile, or `.dockerignore`.
- Flag-off behaviour MUST stay byte-identical: the `probe.page_decision` emit is already
  guarded by `if "probe" in page_context:`. KEEP that guard; add nothing outside it.
- Item-4 command lines are derived from the REAL argparse in the scripts (recorded in §D below),
  not from memory.

---

## CRITICAL FINDING — `final_engine` recoverability (investigation item 1)

### What the code does today (read from `process_page`, pipeline.py)
- The plan runs ONE primary engine. `engines_used` after the initial merge is `[primary_engine]`
  (single-result branch of `_merge_results`).
- The escalation ladder (`while confidence < threshold or not markdown.strip()`) may run once.
  When it runs, `engines_used` is mutated in one of three ways:
  1. **VLM escalation, VLM wins** (`esc_confidence >= confidence or not markdown.strip()`):
     `markdown/confidence/page_blocks` are replaced by the VLM result and `"vlm_transcribe"` is
     **appended** to `engines_used`. Adopted output engine = `vlm_transcribe`.
  2. **VLM escalation, VLM loses**: prior OCR markdown/confidence are KEPT, but `"vlm_transcribe"`
     is **still appended** to `engines_used`. Adopted output engine = the PRIOR engine
     (`engines_used[0]`), NOT `vlm_transcribe` — even though `vlm_transcribe` is now `engines_used[-1]`.
  3. **OCR-to-OCR escalation**: `markdown, confidence, engines_used = _merge_results(all_results)`.
     `_merge_results` rebuilds `engines_used` from engines that contributed non-duplicate lines,
     primary first. Adopted dominant engine = `engines_used[0]` of the re-merged list.
- `metadata.set_final_result(engine=engines_used[0] ...)` → `PageMetadata.final_engine` is
  ALSO just `engines_used[0]` (the primary), confirmed in `src/ai/schemas/page_metadata.py`.
- The existing event emits `engine_used = output.engines_used[0]` (the primary).

### Conclusion: is the adopted engine recoverable from EXISTING post-ladder state?
**Partially — and NOT reliably from any single existing field.**
- `engines_used[0]` = primary (correct adopted engine in cases 2 and 3, WRONG in case 1).
- `engines_used[-1]` = last-appended engine (correct adopted engine in case 1, WRONG in case 2 —
  VLM-loses appends `vlm_transcribe` but does not adopt it).
- `PageMetadata.final_engine` == `engines_used[0]` (same as primary; no extra information).

So `engines_used[-1]` alone is NOT a correct `final_engine` (it misreports the VLM-loses case),
and no existing field already holds the adopted engine. **This is NOT a STOP-and-report situation:**
the adopted engine IS deterministically known at the exact points in the ladder where the
win/lose decision is made. The additive, behaviour-preserving fix is to record the adopted engine
in a NEW LOCAL variable inside `process_page` as the ladder runs, then emit it. This changes NO
routing, NO output object, NO metadata, and nothing outside the `if "probe" in page_context` guard.

### Precise source expressions to add (local var `final_engine`, additive only)
Add, right after the initial merge and BEFORE the escalation `while` loop:
```python
final_engine = engines_used[0] if engines_used else None   # adopted so far = primary/merged
```
Update it at the adoption points INSIDE the ladder:
- VLM-wins branch (after `page_blocks = list(esc_blocks)`): `final_engine = escalation_engine`
- VLM-loses branch: leave `final_engine` unchanged (prior engine stays adopted).
- OCR-to-OCR branch (after the re-merge): `final_engine = engines_used[0] if engines_used else final_engine`
After the loop, also guard for the empty-result edge: if `final_engine is None` keep it `None`.

### Precise source expressions for the two NEW event fields
- `engines_used` (full list): `output.engines_used` (the list already on the returned `PageOutput`).
- `final_engine` (adopted engine): the local `final_engine` computed above.
- `engine_used` (UNCHANGED): stays `output.engines_used[0] if output.engines_used else None`.

Consistency note for later steps: in VLM-loses, `engines_used[-1]` == `vlm_transcribe` but
`final_engine` == the paddle primary. The summariser (§C) therefore prefers `final_engine` and
only falls back to `engines_used[-1]` for OLD log lines that predate this field.

---

## A. Code change — `src/ai/layer1_routing/pipeline.py` (additive only)

- [ ] 1. Track the adopted engine through the escalation ladder in `process_page`.
      Add local `final_engine = engines_used[0] if engines_used else None` immediately after the
      `markdown, confidence, engines_used = _merge_results(mergeable_results)` line (before the
      `while` loop). In the VLM-wins branch set `final_engine = escalation_engine`; in the
      OCR-to-OCR branch set `final_engine = engines_used[0] if engines_used else final_engine`
      after the re-merge; leave it unchanged in the VLM-loses branch. Do not touch `set_final_result`,
      `PageOutput`, or the `pipeline.page_complete` event.
      Files: `src/ai/layer1_routing/pipeline.py`
      Verify: build/collect + targeted tests in step 3 below; flag-off test (A3) proves no behaviour change.

- [ ] 2. Extend ONLY the existing `probe.page_decision` `logger.info(...)` call (inside the
      `if "probe" in page_context:` block) with two additive kwargs, keeping `engine_used` exactly
      as-is: `engines_used=output.engines_used` and `final_engine=final_engine`. Add NO image bytes,
      NO text/markdown. Do not move or duplicate the emit; keep it after `pipeline.page_complete`
      (post-ladder) so all values are final.
      Files: `src/ai/layer1_routing/pipeline.py`
      Verify: step 3 tests below.

## B. Tests — event content (tests/test_pipeline_probe_document.py)

- [ ] 3. Update `test_probe_page_decision_emitted_on_probed_page` to additionally assert the two new
      fields on the no-escalation printed page: `d["engines_used"] == out.engines_used` (i.e.
      `["paddleocr_printed"]`) and `d["final_engine"] == "paddleocr_printed"`, while keeping all
      existing assertions (every current field + the `engine_used == out.engines_used[0]` check +
      the "no image_bytes/markdown/text" leak check). Also extend
      `test_probe_page_decision_shows_post_ladder_values` (primary paddle returns empty → VLM adopted)
      to assert `d["final_engine"] == "vlm_transcribe"`, `"vlm_transcribe" in d["engines_used"]`,
      and that `d["engine_used"] == out.engines_used[0]` (still the primary, unchanged).
      Add ONE new test `test_probe_page_decision_printed_escalates_to_vlm_final_engine`: primary
      `paddleocr_printed` on a probed latin page, ladder escalates and ADOPTS `vlm_transcribe` (reuse
      the `_empty_invoke` pattern that forces empty paddle text so VLM wins); assert
      `d["probe_route"] == "paddleocr_printed"`, `d["engine_used"] == out.engines_used[0]`,
      `d["final_engine"] == "vlm_transcribe"`, `d["engines_used"] == out.engines_used`, and
      `d["escalated"] is True`. Keep `test_probe_page_decision_not_emitted_flag_off` unchanged
      (it already proves flag-off emits nothing / byte-identical routing).
      Files: `tests/test_pipeline_probe_document.py`
      Verify (run exactly, from repo root):
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_pipeline_probe_document.py -q`
      Expected: all pass, including the new escalation-to-VLM test and collection.

## C. Code change — scripts/summarize_probe_results.py

- [ ] 4. In `summarize()`, when reading `probe.page_decision`, record the FINAL adopted engine with a
      3-level fallback: prefer `rec["final_engine"]`, else `rec["engines_used"][-1]` (if a non-empty
      list is present), else the existing `rec["engine_used"]`. Store that value into
      `pd_final_engine[key]` (the dict already used for harmful + per-final-route). The existing
      `pipeline.page_complete` fallback path (uses `engines_used[-1]`) stays unchanged. Keep
      `engine_used` as the source for nothing else that changes; `probe_route`/`escalated`/
      `probe_script` handling is unchanged. The near-miss metric (`pd_route` + `pd_escalated` +
      final engine) is PRESERVED verbatim — it already keys off `pd_final_engine` via
      `summary["page_engine"]`.
      Files: `scripts/summarize_probe_results.py`
      Verify: step 6 below.

- [ ] 5. Confirm (no code change expected) that `harmful_rows`, `near_miss_rows`, per-route and
      per-final-route stats all consume `pd_final_engine`/`page_engine` and therefore now reflect
      `final_engine`. `_PADDLE_ENGINES`, `_ENGINE_TO_ROUTE`, and the non-zero exit on `HARMFUL>0`
      are unchanged. If any of these still read `engine_used` directly, redirect them to the
      final-engine dict; otherwise leave as-is.
      Files: `scripts/summarize_probe_results.py` (only if a direct `engine_used` read is found)
      Verify: step 6 below.

## D (tests). Summarizer tests + fixtures (tests/test_summarize_probe_results.py, tests/fixtures/probe/)

- [ ] 6. Add a new fixture `tests/fixtures/probe/logs_final_engine.ndjson` plus labels
      `tests/fixtures/probe/labels_final_engine.csv` (`doc,script` / `docF,devanagari`) exercising the
      escalation case with the NEW fields: one non-Latin page where the probe routed to
      `paddleocr_printed`, the ladder escalated, and `final_engine` = `paddleocr_printed` even after
      escalation (HARMFUL=1 contribution) and one non-Latin page routed to `paddleocr_printed` that
      escalated AWAY with `final_engine` = `vlm_transcribe` (near-miss=1, HARMFUL=0 contribution).
      Each `probe.page_decision` row carries `engine_used` (primary), `engines_used` (full list), and
      `final_engine`. Add a test `test_final_engine_harmful_and_near_miss`: `harmful_rows` counts the
      still-paddle final-engine page, `near_miss_rows` has exactly 1 row with
      `final_engine == "vlm_transcribe"`, and the escalated-away page is NOT harmful (asserting
      harmful=0, near-miss=1 for a doc where the only harmful candidate escaped). Also add
      `test_final_engine_fallback_to_engines_used_last`: a record with NO `final_engine` but with
      `engines_used` present → summariser uses `engines_used[-1]`; and
      `test_final_engine_fallback_to_engine_used`: a record with neither `final_engine` nor
      `engines_used` → summariser uses `engine_used` (back-compat with existing
      `logs_page_decision.ndjson` rows, which must still pass unchanged).
      Files: `tests/fixtures/probe/logs_final_engine.ndjson`,
      `tests/fixtures/probe/labels_final_engine.csv`, `tests/test_summarize_probe_results.py`
      Verify (run exactly):
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_summarize_probe_results.py -q`
      Expected: all existing + new tests pass; existing `logs_page_decision.ndjson` /
      `logs_clean.ndjson` tests still pass via the `engine_used` fallback.

---

## PREFLIGHT (read-only — item 3 of the brief; RUN, do not modify)

- [ ] 7. **3a — run the four probe suites + the summarizer suite; report counts.** Run exactly:
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_inprocess_images.py tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py tests/test_summarize_probe_results.py -q`
      Report pass/fail counts per file (and total), including that collection succeeded. No file edits.

- [ ] 8. **3b — confirm the lipika model + labels are in the Docker build context and NOT
      `.dockerignore`'d; print sizes + sha256.** Build context root = repo ROOT (confirmed:
      `deploy_probe_image.py` `ROOT = parents[1]`). Files (confirmed present on disk):
      `src/ai/layer1_probe/models/lipika_script.onnx` (111,642,305 B),
      `lipika_script.int8.onnx` (28,494,953 B), `lipika_script.labels.json` (541 B). Root
      `.dockerignore` excludes `*.bin`, `*.safetensors`, `.git/`, `tests/`, `.env*`, `.venv/`,
      `data/`, `logs/` but NOT `*.onnx` / `*.json` (confirmed — no `.onnx`/`.json` pattern present;
      only one `.dockerignore` exists, no `schema_chatbot_v2/.dockerignore`). Compute sha256 with the
      venv python (read-only), e.g. per file via `hashlib`, and print `name size sha256`. Note for the
      report: the `.onnx` files are gitignored (ship via `COPY . /app`), so they must NOT be staged.

- [ ] 9. **3c — `deploy_probe_image.py --dry-run` AWS-call determination.** Investigated: the
      `--dry-run` path in `main()` does NOT touch AWS — it uses the mirrored `EXPECTED_ACCOUNT`
      constant, builds the URI, and prints login/build/push commands with the password placeholder;
      `import boto3`, `sts.get_caller_identity()`, and all `ecr` calls live only in the real-run
      branch AFTER the `if args.dry_run:` return. HOWEVER, before that branch `main()` calls
      `git_short_sha()` and `git_porcelain()` (local git only, no AWS) and
      `verify_models_in_context(ROOT)`, which reads the ~112MB onnx to sha256 it (local only).
      Therefore `--dry-run` needs NO AWS credentials and MAY be run:
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts/deploy_probe_image.py --dry-run`
      It prints the image URI on stdout and the masked commands on stderr; it builds/pushes nothing.
      (If it unexpectedly errors for any AWS reason, skip and say so — but the code path shows it will not.)

- [ ] 10. **3d — confirm the Dockerfile ships the probe runtime.** `schema_chatbot_v2/Dockerfile`
      uses a single `COPY . /app` from the build-context root (confirmed), so `src/ai/layer1_probe/`,
      `src/ai/layer1_routing/windows.py` (if present under src), and
      `src/ai/layer2_conversion/vlm_block_rollup.py` are all copied from the WORKING TREE regardless of
      git-tracked status. No per-file COPY is needed. Report this as confirmed; note the Dockerfile
      resolves the probe model at `/app/src/ai/layer1_probe/models/lipika_script.onnx`.

---

## COMMAND LIST (item 4 — flags taken from the REAL argparse; see per-script notes)

Placeholders: `<REGION>` (scripts default `ap-south-1`), `<CLUSTER>` (default `idp-cluster`),
`<SERVICE>` (default `idp-layer3-service`), `<IMAGE_URI>` (output of the build step),
`<ROLLBACK_FILE>` a `rollback_<ts>.json`, `<LOGS_NDJSON>`, `<LABELS_CSV>`.
PY = `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`

### Real argparse flags per script (recorded — note deviations from the brief's guesses)
- `deploy_probe_image.py`: `--dry-run`, `--region` (default `ap-south-1`). There is NO `--image`
  here — this script PRODUCES the image URI (prints it to stdout). (brief assumed `--image`/`--flag-off`
  belong here; they do NOT.)
- `deploy_probe_service.py`: `--image` (REQUIRED), `--cluster`, `--service`, `--container`,
  `--flag-off` (store_true → sets `IDP_PROBE_ENABLED=false`; absence → `true`), `--region`, `--dry-run`.
- `rollback_probe_service.py`: mutually-exclusive REQUIRED group `--rollback-file` XOR
  `--task-definition-arn`; plus `--cluster`, `--service`, `--region`, `--dry-run`.
  (brief's `--rollback-file`/`--task-definition-arn` names are correct; note they are mutually exclusive.)
- `summarize_probe_results.py`: `--logs` (REQUIRED), `--labels` (optional). (brief's `--logs`/`--labels`
  correct; there is NO `--dry-run` here.)

### Commands (copy/paste; nothing here is executed by the plan/implementer)
1. Login check / build dry-run (NO AWS creds needed; prints URI + masked cmds):
   `PY scripts/deploy_probe_image.py --dry-run --region <REGION>`
2. Real build + push (REQUIRES AWS creds + docker; DO NOT run in this workflow):
   `PY scripts/deploy_probe_image.py --region <REGION>`
   → capture the printed `<IMAGE_URI>` from stdout.
3. Deploy with probe FLAG OFF — dry-run first, then real:
   `PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --flag-off --dry-run`
   `PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --flag-off`
4. Deploy with probe FLAG ON — dry-run first, then real (omit `--flag-off` → `IDP_PROBE_ENABLED=true`):
   `PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --dry-run`
   `PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION>`
5. Rollback to `idp-layer3-task:3` (ARN form; `--task-definition-arn` XOR `--rollback-file`):
   `PY scripts/rollback_probe_service.py --task-definition-arn arn:aws:ecs:<REGION>:<ACCOUNT>:task-definition/idp-layer3-task:3 --cluster <CLUSTER> --service <SERVICE> --region <REGION> --dry-run`
   then drop `--dry-run` for the real rollback. (Or `--rollback-file <ROLLBACK_FILE>` to restore from a snapshot.)
6. CloudWatch log export + summarizer (export step is an AWS CLI the USER runs; not part of this plan):
   after exporting the `/ecs/idp-layer3-app` stream to `<LOGS_NDJSON>`:
   `PY scripts/summarize_probe_results.py --logs <LOGS_NDJSON> --labels <LABELS_CSV>`
   (exits non-zero if HARMFUL>0.)

---

## SECRET / DO-NOT-COMMIT WARNINGS (item 5 — by NAME only, never contents)
- `rollback_<ts>.json` (written by `deploy_probe_service.py`) contains UNMASKED task-def env secrets —
  must NOT be committed and should be in `.gitignore` (add `rollback_*.json`).
- `.env`, `.env2` (present in the repo/open editors) and any `.env*` — secrets; `.dockerignore`
  already excludes `.env*` so they will NOT enter the image, but must never be committed/printed.
- `*.pem` / key files — none observed, but exclude by name if present in the build context.
- `*.onnx` model files — NOT secrets, but are gitignored and must NOT be staged/committed
  (ship via the Dockerfile `COPY . /app`).

---

## Contradictions vs the task description (report)
1. The brief groups `--image` / `--flag-off` under image-build assumptions; in reality `--image`
   and `--flag-off` belong to `deploy_probe_service.py`, and `deploy_probe_image.py` only takes
   `--dry-run`/`--region` and OUTPUTS the image URI. Commands above use the real flags.
2. `final_engine` is NOT already stored in any single post-ladder field: `engines_used[0]`,
   `engines_used[-1]`, and `PageMetadata.final_engine` each misreport at least one escalation case.
   It is still recoverable additively by tracking the adopted engine in a local var as the ladder
   runs (see CRITICAL FINDING). This is NOT a STOP-and-report; the adopted engine is deterministic.
3. `rollback_probe_service.py` requires EXACTLY ONE of `--rollback-file` / `--task-definition-arn`
   (mutually exclusive, required) — the brief lists both as if independently optional.
