# Probe deploy/measurement scripts — gate review

Four new scripts (`deploy_probe_image.py`, `deploy_probe_service.py`, `rollback_probe_service.py`, `summarize_probe_results.py`) plus three test files build and ship the Layer-3 app image carrying the in-process probe, flip `IDP_PROBE_ENABLED` on the live ECS task def with a rollback snapshot, and summarise probe behaviour from the structured JSON logs. All four are import-safe (side effects live under `main()`/functions), tag with the AGENTS.md mandatory tags, refuse `:latest`, and mask secrets. The coder ran the suite (38 new unit tests passing, plus the Step-1b clean-worktree check) and ran only the single permitted git worktree action — no AWS/docker/git-commit/git-push.

Watch for: (confirmed) `deploy_probe_service.py --dry-run` still issues live AWS `describe_services`/`describe_task_definition` read calls because the boto3 session and describes run before the dry-run branch — contrary to the plan's "fall back to a sample taskdef if no creds" intent. Read-only, no mutation, no secret print, so non-blocking. (confirmed) the dry-run image URI embeds the mirrored `EXPECTED_ACCOUNT` constant — sanctioned by the plan as a source-cited mirror, not a new hard-coded id, and the real run derives account from STS.

**Verdict**: APPROVED

## High-level view

No hard-coded account/region/subnet/role is introduced. Account, region, cluster, service and task-family come either from importing `deploy_idp_fargate` (import-safe — it builds no boto3 client at import) or, in a stripped checkout, from literals explicitly commented as mirrors of specific `deploy_idp_fargate.py` / `deploy_idp_ecr_images.py` source lines. The real run resolves the account from `sts.get_caller_identity()` and asserts it against the expected value; subnets and roles are never referenced because the service script clones the *live* task definition rather than constructing one.

The `:latest` guard is a single `assert_not_latest` that rejects a bare `latest` or any `…:latest` suffix. The image script builds its own `probe-<UTC>-<sha>[-dirty]` tag and calls the guard on both the tag and the final URI; the service and rollback scripts import the same guard (with a cited fallback copy) and reject a `:latest` `--image`.

No `.onnx` is staged or committed by any script. `deploy_probe_image.py` only *reads* the model files to assert they exist in the build context, confirms `.dockerignore` does not exclude them, and prints size + streamed sha256. The weights ship via the docker `COPY`, never via git.

Existing production source is untouched. The probe scripts import `deploy_idp_fargate` for constants and mirror `deploy_idp_ecr_images.py` build parameters with per-line source citations; neither file is edited.

The task-def clone flips exactly two things — the container image and `IDP_PROBE_ENABLED` — leaves `IDP_PROBE_N_SAMPLES` untouched, and strips the register-rejected read-only fields. The rollback snapshot is written before any register/update, with a printed warning that it holds unmasked secrets. `--flag-off` sets `false`. AGENTS.md tags are applied on `register_task_definition` and on the ECR repo (best-effort, non-fatal).

The summariser consumes only the JSON log stream (the honest source per the F5 investigation), computes pages-per-route / escalation / VLM-calls / agree-disagree / per-script accuracy / HARMFUL count, exits non-zero on HARMFUL>0, and prints a "not available"/"cannot join" message instead of inventing any metric it cannot derive.

<details>
<summary>Issues (3)</summary>

1. **Dry-run hits AWS (service script)** — confirmed. `deploy_probe_service.py --dry-run` performs live `describe_services`/`describe_task_definition` read calls before the dry-run branch, unlike the plan's no-creds fallback. Read-only and non-blocking; optionally guard the describes behind a creds check so dry-run is fully offline.
2. **Account constant in dry-run URI** — confirmed. The dry-run URI uses the mirrored `EXPECTED_ACCOUNT`; sanctioned by the plan and source-cited, real run uses STS. No action required; keep the "placeholder" label.
3. **ECR tag_resource failure is swallowed** — confirmed. `deploy_probe_image.py` wraps repo tagging in a broad `except Exception` that only warns. Acceptable (push must not depend on tagging) but means a tagging-permission gap is silent; note in hand-off IAM list.

</details>

<details>
<summary>Details</summary>

### No hard-coded account / subnet / role (check 1 — pass)

Every account/region/repo value in `deploy_probe_image.py` comes from `import deploy_idp_fargate` with a cited-line fallback (`REGION` ← fargate:21, `EXPECTED_ACCOUNT` ← fargate:22, `REPO` ← fargate:35). The real run overwrites `account` with `sts.get_caller_identity()["Account"]` and raises if it differs from `EXPECTED_ACCOUNT`. `deploy_probe_service.py` reads `REGION`/`CLUSTER`/`SERVICE`/`TASK_FAMILY` the same way and otherwise clones the live task def, so no subnet or role literal is ever written — those ride through from the described task def untouched. No subnet id, VPC id, or role ARN appears in any new file. The one account literal that does appear (`EXPECTED_ACCOUNT = "530084039784"`) is the fallback mirror, commented against `deploy_idp_fargate.py:22`, and is used only to assert the live account matches.

### Secret masking (check 2 — pass)

`deploy_probe_service.py` masks via `mask_secrets` (known secret keys `JWT_SECRET`/`ADMIN_PASSWORD`/`IDP_*` → `****`, and a URL-password regex for `*_URL`/`DATABASE_URL`). The dry-run prints the register request through `mask_taskdef_for_print`, which deep-copies and masks every container's environment — so the printed JSON never carries a live secret. The test suite asserts `JWT_SECRET=****`, `ADMIN_PASSWORD=****`, and that `p4ss` is absent from the masked `DATABASE_URL` while the `postgresql://u:` prefix survives. The image script masks the ECR login password in the emitted `docker login` command (`mask_login_command`), and dry-run uses a literal `<ECR_TOKEN>` placeholder with no real token fetched. The rollback snapshot file is deliberately unmasked (it must be faithful to re-register) and the script prints a WARNING that it holds secrets and must not be committed — the honest trade-off the plan called for.

### :latest guard (check 3 — pass)

`assert_not_latest` splits on the last `:` and rejects case-insensitively, so `latest`, `repo:latest`, and `…amazonaws.com/idp-layer3-app:latest` all raise. The image script generates its tag from `build_tag` (which can only produce `probe-<12 digits>-<sha>[-dirty]`) and guards both the tag and the final URI. The service and rollback scripts import the guard and duplicate it as a cited fallback, and the service script guards `--image` before building any boto3 session. Tests cover bare, suffix, and the allowed probe tag.

### No .onnx staged or committed (check 4 — pass)

No script runs `git add`/`git commit`. `deploy_probe_image.py` only reads the model files: `verify_models_in_context` resolves the default onnx (via `default_model_relpath`, which mirrors `script_classifier.py` by reading `shipped_default` from the labels sidecar) plus `labels.json`, asserts they exist, asserts `.dockerignore` does not exclude them, and prints size + streamed sha256 (1 MiB chunks, so the 111 MB fp32 model is not loaded whole). The weights reach the image through the Dockerfile `COPY`, matching the F3/F4 finding that `*.onnx` is gitignored by design but not dockerignored.

### Production source untouched; helpers imported/mirrored (check 5 — pass)

`deploy_idp_fargate.py` is imported for constants, never edited. `deploy_idp_ecr_images.py` build parameters are mirrored as module constants, each with a source-line citation in the header docstring and inline comments (`DOCKERFILE` ← ecr:37, `PLATFORM`/`BUILD_FLAGS` ← ecr:90, login flow ← ecr:79-82). The header explicitly calls out the F1 contradiction (build logic lives in `deploy_idp_ecr_images.py`, not `deploy_idp_fargate.py`). The mirrored constants match the real `deploy_idp_fargate.py` as read (`REGION`, `ACCOUNT`, `REPO`, `CLUSTER`, `SERVICE`, `TASK_FAMILY` all line up).

### Task-def clone: image + flag only, fields stripped, rollback first (check 6 — pass)

`apply_probe_changes` deep-copies the live def, sets `container["image"]`, upserts `IDP_PROBE_ENABLED`, and returns `strip_for_register(...)`. `upsert_env` replaces in place or appends — it does not touch any other key, and `IDP_PROBE_N_SAMPLES` is never written (asserted by `test_apply_probe_changes_flips_only_image_and_flag`). `_select_container` targets `idp-web-console` (cited fargate:200) or the sole container. `strip_for_register` removes exactly the eight register-rejected read-only fields (`taskDefinitionArn, revision, status, requiresAttributes, compatibilities, registeredAt, registeredBy, deregisteredAt`) and keeps `family`, roles, `cpu`/`memory`, `networkMode`, etc. In the real path, `write_rollback_file` runs before `register_task_definition`/`update_service`, writing `{taskDefinitionArn, taskDefinition}` with best-effort `0o600` perms; the dry-run path writes no file and the test asserts `rollback_*.json` is absent and that `register`/`update` were never called on the fake client.

### --flag-off (check 7 — pass)

`flag_value = "false" if args.flag_off else "true"`, threaded into `apply_probe_changes`. `test_dry_run_flag_off` asserts the printed env shows `IDP_PROBE_ENABLED=false`.

### AGENTS.md mandatory tags (check 8 — pass)

Both tagging-capable APIs get the three mandatory tags with the exact AGENTS.md values (`createdby=vinay.k@shellkode`, `purpose=internal`, `customer=idp-internal-35385`). The image script uses ECR `Key`/`Value` capitalisation on `tag_resource`; the service/rollback scripts use ECS lowercase `key`/`value` on `register_task_definition`. Note: `deploy_idp_fargate.py`'s own `TAGS` uses `vinay.k@shellkode.com` (with `.com`); the probe scripts correctly follow AGENTS.md's `.com`-less value rather than the divergent source constant. `test_ecs_tags_are_mandatory_agents_tags` locks the ECS values. ECR tagging is best-effort (broad `except` → warn), which is fine for a push path but means a missing `ecr:TagResource` permission fails silently — worth listing in the hand-off IAM notes.

### Summariser: HARMFUL exit, no invented metrics (check 9 — pass)

`render` sets `exit_code = 1` when `harmful_rows` is non-empty and prints each offending `(doc, page, label, engine)` row; `main` returns that code and the CLI test confirms a non-zero exit with `paddleocr_printed`/`page=2` on the harmful fixture and exit 0 with `HARMFUL … : 0` on the clean one. HARMFUL is computed from `pipeline.page_complete` final engines joined to the label CSV, independent of whether script accuracy can be joined. Every uncomputable metric prints a message instead of a number: pages/escalation print `_NOT_AVAILABLE` when `pipeline.page_complete` is absent, escalation prints a no-`escalated`-field message, VLM calls and agree/disagree print their own "no events" lines, and `script_accuracy` returns `None` (→ "cannot join labels") when no labelled doc appears in the logs. The grouping caveat (`probe.lipika_vs_vlm` has no document_id, so grouping is by `correlation_id`) is documented in the header and implemented in `_doc_key`.

### Import-safety and mocking (check 10 — pass)

All four scripts do real work only inside functions or under `if __name__ == "__main__"`; `boto3` is imported inside `main()`, so importing the modules in tests triggers no AWS/docker/git. `test_import_has_no_side_effects` asserts the module-level constants resolve without network. The service tests inject a `_FakeEcs` by monkeypatching `boto3.Session` and redirect `ROOT` to `tmp_path`; the image tests exercise pure helpers against fake tmp contexts; the summariser tests use NDJSON/CSV fixtures and a subprocess for the exit-code assertion. Git and docker are never invoked by the unit tests (the git wrappers in the image script are only reached via `main`, which the unit tests do not call).

### No AWS/docker/git-push/git-commit performed in authoring (check 11 — pass, per verification note)

The verification note records that the only git mutation was the permitted Step-1b `worktree add`/`remove`, that the clean worktree was removed and `git status` left unchanged, and that no `rollback_*.json` was left behind. Dry-run smoke runs printed the image URI/commands and the rollback `update_service` call with no AWS or docker execution. I did not re-run the suite per the judge instruction; the evidence is present and internally consistent (per-file counts 16/12/10 = 38, matching the reported total). One documented fixup (a summariser test expectation corrected from 3 to 4 for the page-4 join) is consistent with the fixture, where page 4 carries `lipika_script=latin`/`vlm_script=null` and is therefore counted.

### Non-blocking: service dry-run is not fully offline

`deploy_probe_service.main` builds the boto3 session and calls `describe_services`/`describe_task_definition` before checking `args.dry_run`, so a real `--dry-run` invocation with credentials present makes those two read-only calls against the live account. The plan preferred a no-creds fallback to a sample taskdef so masking/strip logic is demonstrable offline. This is read-only (no register/update, no secret printed) and the unit tests already exercise the masking path offline via the fake client, so it does not block. If offline dry-run matters operationally, guard the describes behind a credentials check and fall back to a labelled sample def.

</details>

<details>
<summary>File map</summary>

- `scripts/deploy_probe_image.py` — build+push the layer3 app image with a `probe-*` tag; model verification, label provenance, `:latest` guard, masked login.
- `scripts/deploy_probe_service.py` — clone live task def, flip image + `IDP_PROBE_ENABLED`, rollback snapshot first, register+update; `--flag-off`, `--dry-run`, masking helpers.
- `scripts/rollback_probe_service.py` — roll service back to a saved ARN or rollback-file snapshot; re-register if the revision is gone; `--dry-run`.
- `scripts/summarize_probe_results.py` — summarise probe metrics from JSON logs; HARMFUL check exits non-zero; prints "not available" for uncomputable metrics.
- `tests/test_deploy_probe_image.py` — 16 tests: tag/guard/mask/porcelain/labels/dockerignore/model-verify.
- `tests/test_deploy_probe_service.py` — 12 tests: strip/upsert/mask/flag/dry-run-no-rollback/rollback-roundtrip.
- `tests/test_summarize_probe_results.py` — 10 tests: route/escalation/vlm/agree/harmful/accuracy/cannot-join/CLI exit codes.
- `tests/fixtures/probe/{logs_clean.ndjson,labels_clean.csv,labels_harmful.csv}` — synthetic measurement fixtures (text, not gitignored).

</details>
