# Implementation Plan — Probe Deployment & Measurement Scripts

Scope: LOCAL authoring of deployment/measurement SCRIPTS and TESTS only.
Hard rules (enforced in every item below): do NOT run any AWS command (no aws cli, no live
boto3 calls), no `docker build/push`, no `git push`, no `git commit`. The ONE allowed git action
is the Step‑1b worktree test. Do NOT stage/commit any `.onnx`. NEVER use `:latest`. No hard‑coded
account ids / subnet ids / secrets — everything account/region/subnet/role specific comes from
runtime lookup or CLI arg/env. MASK secrets in printed output. Do NOT edit existing production
source files. Do NOT edit `scripts/deploy_idp_fargate.py`.

All new scripts live in `scripts/`. Repo venv python:
`c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`.

---

## Investigation findings that shape this plan (read first — these correct the task brief)

These were confirmed by reading the code. They change how Steps 2 and 4 must be built, so they
are stated up front and must be honored by the implementer.

**F1 — The image build/push logic is NOT in `deploy_idp_fargate.py`. It is in
`scripts/deploy_idp_ecr_images.py`.** `deploy_idp_fargate.py` only registers the task def and
service and points at `L3_IMAGE = f"{ECR}/idp-layer3-app:latest"` (line 37). The actual
`docker build … && docker push` for `idp-layer3-app` is in `deploy_idp_ecr_images.py`:
  - build context = repo ROOT (`ROOT = Path(__file__).resolve().parents[1]`, line ~23; `run()`
    uses `cwd=str(ROOT)`, line ~41).
  - Dockerfile = `schema_chatbot_v2/Dockerfile` (IMAGES dict, line ~37).
  - platform/build flags = `docker build --provenance=false --platform linux/amd64 -f {dockerfile} -t {tag} .`
    (line ~90).
  - ECR repo = `idp-layer3-app`; registry host derived from `ecr.get_authorization_token()` →
    `proxyEndpoint` (lines ~80–83), login via `docker login -u {user} -p {pwd} {endpoint}`.
  - region = `ap-south-1`; account asserted `== "530084039784"` via `sts.get_caller_identity`.
  So **Step 2 (`deploy_probe_image.py`) must mirror `deploy_idp_ecr_images.py`, not
  `deploy_idp_fargate.py`**, for context/Dockerfile/platform/repo. This is the single biggest
  deviation from the task description and MUST be called out in the final report as a
  CONTRADICTION (the brief says "mirror deploy_idp_fargate.py helpers" but that file has no build
  logic). `deploy_idp_fargate.py` is still the authority for Step 3 (task‑def/service/role) and
  for the Step‑5 IAM/role confirmation.

**F2 — `scripts/deploy_idp_fargate.py` is itself UNTRACKED** (`git status` shows `??`), as are
`deploy_idp_ecr_images.py`, `deploy_idp_lambdas.py`, `deploy_idp_s3_rds.py`. The new probe scripts
import from `deploy_idp_fargate.py` as a module. Because it is a plain script with **top‑level
constants and a `main()` guarded by `if __name__ == "__main__"`**, importing it is side‑effect
free (no boto3 client is built at import). Importing is therefore safe and preferred over
mirroring for its constants (REGION, CLUSTER, SERVICE, TASK_FAMILY, TAGS/ECS_TAGS, REPO, ECR).
If import fails in the clean checkout (the file may be absent from a committed baseline — see
F2b), fall back to mirroring the handful of constants with a comment citing the source lines.

**F2b — `deploy_idp_fargate.py` not being committed means it will NOT be in the probe image and
NOT in the `../idp-clean` worktree.** The probe scripts are tooling run from the working tree, so
this is fine for them, but the commit plan (Step 1d) must decide whether these deploy scripts get
committed. They are listed as untracked; the plan proposes committing them in a dedicated
`chore(deploy)` commit (see Step 1d), separate from probe runtime code.

**F3 — `*.onnx` is gitignored by design; `labels.json` is already TRACKED.**
`.gitignore:50` = `*.onnx`. `git check-ignore` confirms both
`src/ai/layer1_probe/models/lipika_script.onnx` (111,642,305 bytes) and
`lipika_script.int8.onnx` (28,494,953 bytes) are ignored. But `git ls-files` shows
`src/ai/layer1_probe/models/lipika_script.labels.json` **is tracked** (committed). So the probe's
label map ships via git; the `.onnx` weights ship ONLY via the docker build COPY. The model file
the observer loads by default must be confirmed in Step 1a (both `.onnx` and `.int8.onnx` exist;
`ProbeConfig.onnx_model_path` defaults to `None` → the classifier resolves a default path — the
implementer must read `script_classifier.py` to record which filename is the runtime default).

**F4 — `.dockerignore` does NOT exclude `*.onnx` or `*.json`, so the model ships in the image.**
Root `.dockerignore` excludes `*.bin`, `*.safetensors`, `.git/`, `tests/`, `docs/`, `.env*`,
`__pycache__/`, `.venv/`, `data/`, `logs/` — but NOT `*.onnx`. There is **no**
`schema_chatbot_v2/.dockerignore` (confirmed: file search found none), so the root `.dockerignore`
is the only one and it applies because the build context is the repo root (F1). The
`schema_chatbot_v2/Dockerfile` does `COPY . /app`, so `src/ai/layer1_probe/models/*.onnx` +
`lipika_script.labels.json` are copied into the image. This is the fact Step 2 must *verify at
build time* (print size+sha256, assert not excluded).

**F5 — THE MEASUREMENT DATA PATH (Step 4a) — per‑page probe results are NOT persisted to disk or
DB; they exist only as structured LOG events.** Traced end to end:
  - `process_page` sets `metadata.extra["probe"] = dict(_probe)` (pipeline.py ~line 459) with
    keys `route, ocr_lang, script, script_confidence, handwriting_prob, has_table, is_blank,
    route_reason` (the dict built in `_run_document_probe`, pipeline.py ~lines 700–712).
  - BUT `PageMetadata.extra` is **never written out**: `md_writer._document_footer` emits only
    `page, engines, confidence, capabilities, chars, escalated, low_confidence, latency_ms`
    (md_writer.py ~lines 220–231) — no `extra`/`probe`. The job result in
    `schema_chatbot_v2/app/api/pipeline_routes.py::_run_pipeline_job` builds `pages_for_layer3`
    from `r[0].markdown`/`page_number` only and persists DocumentIR (blocks) — never `extra`.
    `src/ai/layer3_extraction/storage.py` persists `page_details` with `engines_used, confidence,
    escalated` per page — again NOT the probe script/route decision.
  - Therefore the ONLY runtime‑reachable probe signal is the **structured JSON log stream**
    (`src/utils/logger.py` → `JSONFormatter` → `StreamHandler` to stdout; in Fargate the awslogs
    driver ships stdout to CloudWatch log group `/ecs/idp-layer3-app`, stream prefix `web` —
    from `deploy_idp_fargate.py::register_task` logConfiguration, lines ~205). Relevant events:
    - `probe.lipika_vs_vlm` (probe.py ~line 154): `page_number, lipika_script, lipika_conf,
      margin, n_windows, vlm_script, agree`. → lipika‑vs‑VLM agree/disagree per page/sample.
    - `probe.failed` (pipeline.py ~line 685): document‑level fallback marker.
    - `pipeline.page_complete` (pipeline.py ~line 619): `page_number, engines_used, confidence,
      capabilities`. → the FINAL engine each page ran (route proxy: `paddleocr_printed`,
      `paddleocr_handwritten`, `docling`, `vlm_direct`/`vlm_transcribe`, `skip`).
    - `pipeline.concurrent_pages_start`, `pipeline.merge_complete` (context only).
    - Probe doc‑level `stats` (`vlm_calls, onnx_calls, lipika_resolved, lipika_vlm_agree,
      lipika_vlm_disagree`) are returned by `probe_document`/`_sample_and_aggregate` but the
      in‑process path (`probe_document_images` via `_run_document_probe`) **does not log the stats
      dict** — only the per‑page `probe.lipika_vs_vlm` lines are emitted. So "VLM probe calls per
      document" must be DERIVED by counting `probe.lipika_vs_vlm` lines where `vlm_script` is
      non‑null, per `document_id`/`correlation_id`.
  - **CONTRADICTION to flag:** the task brief lists `PageMetadata.extra["probe"]` and
    "DB rows for per‑page routing/engine/metadata" as retrieval sources. Neither persists the
    probe decision today. The honest answer (and what `summarize_probe_results.py` must consume)
    is: **CloudWatch Logs export (JSON lines) is the authoritative source**; the `.md`
    PIPELINE_SUMMARY footer gives final engine per page as a secondary/local source. The script
    must say which metrics it cannot compute from logs rather than invent them.
  - `document_id` is `f"{job_id}-{pdf.stem}"` (pipeline_routes.py ~line 686); log correlation is
    via `correlation_id` (ContextVar) and the `document_id` field present on `probe.failed` /
    absent on `probe.lipika_vs_vlm` (which carries only `page_number`). **Grouping caveat:**
    `probe.lipika_vs_vlm` does NOT carry `document_id`; it only has `page_number`. Pages are tied
    to a document via `correlation_id` (set per pipeline execution). The summarizer must group by
    `correlation_id` (falling back to a single‑document assumption if absent) — record this
    limitation in the script output.

**F6 — Route vocabulary for the HARMFUL check.** `_ROUTE_TO_ENGINE` (pipeline.py lines 48–55):
`digital→docling`, `scanned→paddleocr_printed`, `handwritten→paddleocr_handwritten`,
`vlm_transcribe→vlm_transcribe`, `skip→skip`. The Paddle engines a non‑Latin page must NOT be
routed to are exactly `paddleocr_printed` and `paddleocr_handwritten`. The probe emits `route`
in engine‑name vocab (`pp.route`), and `pipeline.page_complete` emits `engines_used` in the same
engine‑name vocab. HARMFUL = a page whose labelled script is non‑Latin that ran a Paddle engine.

---

## Step 1 — Commit‑hygiene INVESTIGATION (report only; the plan's output is a report section, no new source file)

- [ ] 1a. Produce the runtime‑file inventory for the probe.
      Run `git status --porcelain` and `git ls-files src/ai/layer1_probe` and
      `git check-ignore -v` on the model files (read‑only). Classify EVERY runtime file the probe
      needs as committed / uncommitted / gitignored. From the investigation above the expected
      table is: `src/ai/layer1_probe/{aggregate,cv_features,imaging,inprocess,line_crops,probe,
      routing,script_classifier,script_verify,types,vlm_probe,__init__}.py` = tracked;
      `src/ai/layer1_probe/windows.py` = UNtracked (`??`); `models/lipika_script.labels.json` =
      tracked; `models/lipika_script.onnx` and `models/lipika_script.int8.onnx` = gitignored
      (by `*.onnx`). Explicitly state the `*.onnx`‑gitignored‑by‑design → ships via docker COPY
      point (F3/F4). Read `src/ai/layer1_probe/script_classifier.py` and record which model
      filename is the runtime default when `IDP_PROBE_ONNX_MODEL` is unset.
      Files: none created (this is the report's Step‑1a section).
      Verify: the report lists each file with a status; cross‑check by re‑running
      `git ls-files src/ai/layer1_probe` and confirming the committed set matches.

- [ ] 1b. The ONE allowed git action — clean‑worktree test.
      Run exactly: `git worktree add ../idp-clean HEAD`; then
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest ../idp-clean/tests/test_probe.py ../idp-clean/tests/test_probe_inprocess_images.py ../idp-clean/tests/test_pipeline_probe_document.py ../idp-clean/tests/test_pipeline_probe_integration.py -q`
      (run with cwd `../idp-clean` so repo‑relative imports resolve; set PYTHONPATH if the suite
      needs it — check how the repo's existing tests import `src`). Capture pass/fail per file and
      the first failing import/collection error. Then `git worktree remove ../idp-clean` (use
      `--force` if it refuses because of the pytest cache). Report WHICH files the clean HEAD
      checkout lacks that the tests need — expected: `windows.py` (untracked) and the two `.onnx`
      weights (gitignored) will be missing, so tests that construct the real ONNX observer will
      fail/skip in the clean tree while tests using stubs pass. Record the exact failure text.
      Files: none created (report section + transient worktree that is removed).
      Verify: `git worktree list` shows no `../idp-clean` after cleanup; `git status` in the main
      tree is unchanged from before 1b (no new tracked files, no commits).

- [ ] 1c. Read `scripts/deploy_idp_fargate.py` AND `scripts/deploy_idp_ecr_images.py` and report
      the layer3 app image build facts: build context dir, Dockerfile path, `--platform`, ECR repo,
      and whether `*.onnx` is included. The authoritative source is `deploy_idp_ecr_images.py`
      (F1): context = repo root, Dockerfile = `schema_chatbot_v2/Dockerfile`,
      `--platform linux/amd64 --provenance=false`, repo `idp-layer3-app`. Cross‑check
      `.dockerignore` (root) and confirm `schema_chatbot_v2/.dockerignore` does NOT exist; conclude
      `*.onnx` + `labels.json` ARE included (F4). State explicitly that `deploy_idp_fargate.py`
      contains no build logic (only task‑def/service) — this is the CONTRADICTION vs the brief.
      Files: none created (report section).
      Verify: quote the exact lines from `deploy_idp_ecr_images.py` (IMAGES dict + the
      `docker build …` line) and from root `.dockerignore` in the report.

- [ ] 1d. Propose a commit plan (report only — do NOT commit). Separate probe runtime work from
      unrelated bundled changes into named commits with explicit file lists. `.onnx` stays
      untracked in every commit. Proposed commits (adjust file lists to the actual `git status`
      captured in 1a):
      1. `feat(probe): in-process Layer 1.5 routing probe` — `src/ai/layer1_probe/windows.py`
         (new), modified `src/ai/layer1_probe/{inprocess,script_classifier,types}.py`,
         `models/lipika_script.labels.json` if changed; plus the probe tests
         `tests/test_probe*.py`, `tests/test_pipeline_probe_*.py`, `tests/test_probe_linewise.py`.
         (NOT the `.onnx` files.)
      2. `fix(layer3): save_document_graph owner fix + test` —
         `src/ai/layer3_extraction/extractor.py`, `tests/test_layer3_hardening.py`,
         `schema_chatbot_v2/tests/test_pipeline_jobs.py` (only the save_document_graph parts).
      3. `chore(config): S3 settings + sentence-transformers pin` —
         `src/config/settings.py`, `requirements.txt`, `requirements-dev.txt`.
      4. `test(schema): schema_chatbot pipeline test` — the schema_chatbot test file(s) if
         distinct from commit 2.
      5. `docs(agents): AGENTS.md / .agents rule edits` — `AGENTS.md`,
         `.agents/rules/aws_tagging.md`, `.agents/tasks/*` doc edits.
      6. `chore(deploy): probe deploy + measurement scripts` — the NEW
         `scripts/deploy_probe_image.py`, `scripts/deploy_probe_service.py`,
         `scripts/rollback_probe_service.py`, `scripts/summarize_probe_results.py`, their tests,
         and (decision point for the user) the currently‑untracked
         `scripts/deploy_idp_*.py` if they are to be version‑controlled.
      Files: none created (report section).
      Verify: every path in the commit plan appears in the 1a `git status` capture; no `.onnx`
      path appears in any commit's file list.

---

## Step 2 — `scripts/deploy_probe_image.py` (NEW; builds+pushes the layer3 app image; DO NOT run)

- [ ] 2. Create `scripts/deploy_probe_image.py`. It MUST mirror the build parameters of
      `scripts/deploy_idp_ecr_images.py` (F1), NOT `deploy_idp_fargate.py`. Structure it with no
      side effects at import (everything under functions + `if __name__ == "__main__": main()`), so
      the tests in item 2T can import it.
      Behavior:
      - Constants/derivation: build context = repo root (`Path(__file__).resolve().parents[1]`);
        Dockerfile = `schema_chatbot_v2/Dockerfile`; platform = `linux/amd64` with
        `--provenance=false`; ECR repo name = `idp-layer3-app`; region = `ap-south-1`. Prefer
        importing REGION/REPO/ECR/account from `deploy_idp_fargate` (import‑safe per F2); if import
        fails, mirror the literals with a comment citing `deploy_idp_fargate.py` lines 21,37 and
        `deploy_idp_ecr_images.py` lines 23,37,90.
      - Tag builder: `probe-<UTCyyyymmddHHMM>-<short_git_sha>` where sha =
        `git rev-parse --short HEAD`; append `-dirty` when `git status --porcelain` is non‑empty.
        A pure function `build_tag(now, sha, dirty) -> str` (no I/O) so it is unit‑testable.
      - REFUSE‑latest guard: a function `assert_not_latest(tag)` that raises if the resolved tag is
        exactly `latest` or ends with `:latest`; call it before any build command is emitted.
      - Docker LABELs: pass `--label` flags carrying `org.idp.git_sha=<sha>`,
        `org.idp.dirty=<true|false>`, `org.idp.modified_files=<comma list, truncated to e.g. 512
        chars>`, `org.idp.untracked_files=<...truncated>`. Derive the file lists from
        `git status --porcelain`. Truncate safely (no secrets expected, but still cap length).
      - Pre‑build model verification: a function that resolves
        `src/ai/layer1_probe/models/lipika_script.onnx` (and `.int8.onnx` if that is the runtime
        default per 1a) and `src/ai/layer1_probe/models/lipika_script.labels.json` against the build
        context, asserts they EXIST, and asserts they are NOT excluded by `.dockerignore` (parse the
        root `.dockerignore`; confirm no pattern matches `*.onnx`/the labels path). Print each file's
        size and `sha256` (streamed, not whole‑file in memory for the 111MB model).
      - AWS/ECR: mirror `deploy_idp_ecr_images.py` login flow (ECR `get_authorization_token` →
        `docker login` → build → push). Apply AGENTS.md mandatory tags on the ECR repo where the API
        supports it (`createdby=vinay.k@shellkode.com`, `purpose=internal`,
        `customer=idp-internal-35385`) — only if the script ensures the repo; if it assumes the repo
        exists, skip repo creation and note it.
      - `--dry-run`: print the exact `aws ecr get-login`/`docker login`, `docker build …`,
        `docker push …` commands and the final image URI, run NOTHING. Real run executes them and
        prints ONLY the final image URI on stdout (all diagnostics to stderr).
      - Account/region come from a boto3 session built INSIDE `main()` (never at import); mask any
        token in printed login command (replace the password with `****`).
      Files: `scripts/deploy_probe_image.py`.
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts/deploy_probe_image.py --dry-run`
      prints a `docker build --provenance=false --platform linux/amd64 -f schema_chatbot_v2/Dockerfile`
      line, a tag matching `^probe-\d{12}-[0-9a-f]{7,}(-dirty)?$`, the model size+sha256 lines, and a
      final `…/idp-layer3-app:probe-…` URI — and executes no AWS/docker call (confirm by it
      completing with no network/boto activity; dry‑run must not call `get_authorization_token`).

- [ ] 2T. Create `scripts/tests/test_deploy_probe_image.py` (or `tests/test_deploy_probe_image.py`
      following the repo's test layout — check where existing script tests live; default to
      `tests/`). Unit‑test the pure/import‑safe functions WITHOUT touching AWS or docker:
      `build_tag` format incl. `-dirty`; `assert_not_latest` raises for `latest`/`:latest`;
      model‑verification raises when a model path is missing (use a tmp fake context) and succeeds
      with fake `.onnx`/`.json` present; `.dockerignore` parser confirms `*.onnx` is not excluded;
      the label‑builder truncates long file lists. Mock `subprocess`/boto3 so nothing runs.
      Files: `tests/test_deploy_probe_image.py`.
      Verify: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_deploy_probe_image.py -q` passes.

---

## Step 3 — `scripts/deploy_probe_service.py` + `scripts/rollback_probe_service.py` (NEW; DO NOT run)

- [ ] 3a. Create `scripts/deploy_probe_service.py`. No side effects at import. It registers a NEW
      revision of the live task def with the probe image + flag, then updates the service.
      Behavior:
      - CLI args: `--image` (REQUIRED, full ECR URI incl. probe tag; reject a bare `:latest` via the
        same `assert_not_latest` used in Step 2 — import it from `deploy_probe_image` or duplicate
        with a cite), `--cluster` (default `idp-cluster`), `--service` (default
        `idp-layer3-service`), `--flag-off` (deploy with `IDP_PROBE_ENABLED=false` instead of
        `true`), `--dry-run`, `--region` (default from `deploy_idp_fargate.REGION` = `ap-south-1`).
      - Runtime task‑def lookup: `ecs.describe_services` → current `taskDefinition` ARN →
        `ecs.describe_task_definition` → the full `taskDefinition` dict. Do NOT hard‑code the family
        contents; clone what is live.
      - Clone+mutate: deep‑copy the live `taskDefinition`; set the single container's `image` to
        `--image`; upsert env var `IDP_PROBE_ENABLED` = `"true"` (or `"false"` with `--flag-off`)
        in that container's `environment` list (replace if present, else append); LEAVE
        `IDP_PROBE_N_SAMPLES` alone (do not add it — default applies). Identify the target container
        by name (`idp-web-console` per `deploy_idp_fargate.py` line ~200) or, if only one container,
        the sole entry.
      - STRIP register‑rejected fields before `register_task_definition`: remove
        `taskDefinitionArn, revision, status, requiresAttributes, compatibilities, registeredAt,
        registeredBy, deregisteredAt`. Keep `family, containerDefinitions, cpu, memory,
        networkMode, requiresCompatibilities, executionRoleArn, taskRoleArn, volumes,
        runtimePlatform, placementConstraints` (carry through whatever the live def had). A pure
        function `strip_for_register(taskdef) -> dict` so it is unit‑testable.
      - Rollback capture FIRST: before any register/update, write
        `rollback_<UTCyyyymmddHHMMSS>.json` containing the current task def ARN and the full current
        task‑definition JSON (so `rollback_probe_service.py` can re‑register if needed). Print the
        rollback file path. MASK secrets in what is written? No — the rollback file must be faithful
        for re‑registration, but it may contain env secrets (JWT_SECRET, ADMIN_PASSWORD, DB_URL).
        Write it with best‑effort restrictive perms and print a WARNING that it contains secrets and
        must not be committed (and ensure the filename pattern `rollback_*.json` is gitignored — add
        that pattern to the commit‑plan note, NOT by editing `.gitignore` here unless the user
        approves; flag it instead). In PRINTED output always mask secret env values.
      - Register + update: `ecs.register_task_definition(... tags=ECS_TAGS from deploy_idp_fargate
        with AGENTS.md mandatory tags)`; then `ecs.update_service(cluster, service,
        taskDefinition=<new arn>)`; wait via the `services_stable` waiter; print the NEW revision
        ARN and the deployment id.
      - `--dry-run`: print the exact `register_task_definition` request JSON (with ALL secret env
        values masked, e.g. `JWT_SECRET=****`, `ADMIN_PASSWORD=****`, DB password in `DATABASE_URL`
        masked) and the `update_service` call it WOULD make; write NO rollback file and call NO AWS
        mutation (describe calls are also skipped in dry‑run unless credentials are present —
        prefer: in dry‑run, if describe fails/no creds, fall back to a clearly‑labelled sample
        taskdef so the masking/strip logic is still demonstrable). Call nothing that mutates.
      - A reusable `mask_secrets(env_list) -> env_list` function (masks known secret keys +
        password in any `*_URL`/`DATABASE_URL`), unit‑testable.
      Files: `scripts/deploy_probe_service.py`.
      Verify: `… scripts/deploy_probe_service.py --image 530084039784.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-203001011200-abc1234 --dry-run`
      prints a register JSON whose container env contains `IDP_PROBE_ENABLED=true`, NO
      `IDP_PROBE_N_SAMPLES`, every secret masked, and makes no AWS mutation (no rollback file
      written). With `--flag-off` the printed env shows `IDP_PROBE_ENABLED=false`. Reject
      `--image …:latest` with a clear error.

- [ ] 3b. Create `scripts/rollback_probe_service.py`. No side effects at import.
      Behavior:
      - CLI args: one of `--rollback-file <rollback_*.json>` OR `--task-definition-arn <arn>`
        (exactly one required); `--cluster`/`--service` with the same defaults; `--region`;
        `--dry-run`.
      - If given a rollback file: read the stored ARN. If that task‑def revision still exists,
        `update_service` straight back to it. (Optional, document‑only: if it were deregistered,
        re‑register from the stored JSON via the same `strip_for_register`.) If given an ARN, use it
        directly. Wait `services_stable`; print the restored revision ARN + deployment id.
      - `--dry-run`: print the `update_service` call it would make (secrets masked if any appear);
        mutate nothing.
      - Reuse `mask_secrets`/`strip_for_register` by importing from `deploy_probe_service`.
      Files: `scripts/rollback_probe_service.py`.
      Verify: `… scripts/rollback_probe_service.py --task-definition-arn arn:aws:ecs:ap-south-1:530084039784:task-definition/idp-layer3-task:3 --dry-run`
      prints an `update_service` to revision `:3` and performs no AWS mutation.

- [ ] 3T. Create `tests/test_deploy_probe_service.py`. Unit‑test (all boto3 mocked, no network):
      `strip_for_register` removes exactly the register‑rejected keys and keeps the rest;
      env‑upsert sets `IDP_PROBE_ENABLED=true` (and `false` with flag‑off) and does NOT add
      `IDP_PROBE_N_SAMPLES`; `mask_secrets` masks `JWT_SECRET`, `ADMIN_PASSWORD`, and the password
      inside `DATABASE_URL` while leaving non‑secret envs intact; `assert_not_latest` rejects a
      `:latest` image; dry‑run writes no rollback file (assert file absent). Feed a synthetic
      live‑taskdef fixture (mirroring the shape of `deploy_idp_fargate.register_task`'s container
      def) so the clone/strip/upsert path runs without AWS.
      Files: `tests/test_deploy_probe_service.py`.
      Verify: `… -m pytest tests/test_deploy_probe_service.py -q` passes.

---

## Step 4 — Measuring

- [ ] 4a. Write the measurement‑data‑path findings into the final report (per F5). State plainly:
      per‑page probe decisions live ONLY in the structured JSON logs (not md footer, not job
      result, not DB); the authoritative fetch is a CloudWatch Logs export from log group
      `/ecs/idp-layer3-app` (stream prefix `web`); the relevant events and their fields are
      `probe.lipika_vs_vlm` (page_number, lipika_script, vlm_script, agree, lipika_conf, margin,
      n_windows), `probe.failed` (document_id), `pipeline.page_complete` (page_number,
      engines_used, confidence, capabilities). Give an exact CloudWatch Logs Insights query to pull
      them for a run and note the grouping caveat (`probe.lipika_vs_vlm` has no document_id; group
      by `correlation_id`). State which brief‑named sources (`PageMetadata.extra["probe"]`, DB rows)
      do NOT contain the probe decision, and why (cite md_writer/storage/pipeline_routes lines).
      Files: none created (report section).
      Verify: the report's 4a section names the exact log group, events, fields, and one runnable
      Logs Insights query; cross‑checked against pipeline.py/probe.py/logger.py line numbers above.

- [ ] 4b. Create `scripts/summarize_probe_results.py`. No side effects at import.
      Input model (consume what 4a proved exists): `--logs <file.json|file.ndjson>` = a CloudWatch
      export OR local `logs/pipeline.log` (one JSON object per line, as emitted by `JSONFormatter`).
      Optionally `--md <dir>` to also read `.md` PIPELINE_SUMMARY footers for the final‑engine
      cross‑check. Parse by the `event` field.
      Metrics to print (and, per the rules, say which cannot be computed instead of inventing):
      - Pages per route: count of `pipeline.page_complete` grouped by the engine in `engines_used`
        (map engine→route via the inverse of `_ROUTE_TO_ENGINE`). If the probe `route` is desired
        rather than the final engine, note that the probe route is only in `probe.*` logs and the
        in‑process path does not log a per‑page `probe.route` event — so "pages per route" is
        reported as pages per FINAL engine, with that caveat.
      - Escalation rate per route: `pipeline.page_complete` carries no `escalated` field; derive
        escalation from `engines_used` length > 1 OR presence of a `vlm_*` engine after a Paddle
        engine. If neither is reliably in the logs, report escalation from the `.md`
        PIPELINE_SUMMARY `escalated` flag when `--md` is given, else print "escalation rate: not
        available from logs alone — supply --md".
      - VLM probe calls per document: count `probe.lipika_vs_vlm` lines with non‑null `vlm_script`,
        grouped by `correlation_id` (document). State the grouping caveat.
      - lipika‑vs‑VLM agree/disagree: tally `agree==true` / `agree==false` over
        `probe.lipika_vs_vlm` lines where `vlm_script` is non‑null.
      - `--labels <csv>` (columns `doc,script`): per‑script accuracy = fraction of pages whose
        detected script (`lipika_script`, or `vlm_script` when present) equals the label; and
        HARMFUL COUNT = pages whose labelled script is NON‑Latin that ran a Paddle engine
        (`paddleocr_printed`/`paddleocr_handwritten`, read from `pipeline.page_complete`
        `engines_used`). If HARMFUL > 0, print the offending (doc, page, label, engine) rows and
        `sys.exit(1)`; else exit 0. Page↔label join is by (doc, page_number); because
        `probe.lipika_vs_vlm` lacks doc id, the label CSV join for script accuracy relies on
        `correlation_id`→doc mapping derivable from `probe.failed`/`pipeline.*` lines that DO carry
        document_id — if that mapping is absent for a run, print "script accuracy: cannot join
        labels (no doc id in probe logs)" and skip it, still computing the HARMFUL check from
        `pipeline.page_complete` which DOES let you map engine per page per doc via the md/job
        correlation. Never fabricate a metric.
      Output a compact text report; exit non‑zero ONLY for HARMFUL>0 (or bad input).
      Files: `scripts/summarize_probe_results.py`.
      Verify: see 4T.

- [ ] 4T. Create `tests/test_summarize_probe_results.py` with a small synthetic fixture: a handful
      of NDJSON log lines (`probe.lipika_vs_vlm` with mixed agree/disagree + some null `vlm_script`,
      `pipeline.page_complete` with `paddleocr_printed`/`docling`/`vlm_transcribe` engines) and a
      labels CSV. Assert: pages‑per‑route counts, agree/disagree tallies, VLM‑calls‑per‑doc, and
      per‑script accuracy match hand‑computed values; a fixture with a non‑Latin label routed to
      `paddleocr_printed` makes the script exit non‑zero and print the offending row; a clean
      fixture exits zero; a metric that cannot be computed prints its "not available" message
      rather than a number.
      Files: `tests/test_summarize_probe_results.py`, plus a tiny fixture under
      `tests/fixtures/probe/` (NDJSON + CSV — these are text, not PDFs, so not gitignored).
      Verify:
      `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_summarize_probe_results.py -q`
      passes (including the non‑zero‑exit case asserted via a subprocess or the function's return
      code).

---

## Step 5 — Hand‑off notes (final report section; no new file)

- [ ] 5. Compose the hand‑off section of the final report. Include:
      - Exact commands (using the venv python and the new scripts) for: build+push
        (`deploy_probe_image.py` real + `--dry-run`), deploy flag‑ON (`deploy_probe_service.py
        --image <uri>`), deploy flag‑OFF (`--flag-off`), rollback (`rollback_probe_service.py
        --rollback-file …` and `--task-definition-arn …`). Show `--dry-run` first for each.
      - CloudWatch tailing: log group `/ecs/idp-layer3-app` (cite `deploy_idp_fargate.py`
        logConfiguration lines ~205), stream prefix `web`, and a `aws logs tail` command plus a
        Logs Insights query filtering `event in ["probe.lipika_vs_vlm","probe.failed",
        "pipeline.page_complete"]`.
      - IAM permissions the user's SSO user needs to RUN these scripts (distinct from the task
        role): `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
        `ecr:InitiateLayerUpload/UploadLayerPart/CompleteLayerUpload`, `ecr:PutImage`,
        `ecr:DescribeRepositories` (+ `ecr:CreateRepository`/`ecr:TagResource` only if the image
        script ensures the repo); `ecs:DescribeServices`, `ecs:DescribeTaskDefinition`,
        `ecs:RegisterTaskDefinition`, `ecs:UpdateService`, `ecs:TagResource`; `iam:PassRole` for
        the execution + task roles referenced by the task def; `sts:GetCallerIdentity`;
        `logs:GetLogEvents`/`logs:StartQuery`/`logs:FilterLogEvents` for measuring.
      - CONFIRM — by QUOTING `scripts/deploy_idp_fargate.py::ensure_task_role` — that the TASK role
        `idp-ecs-task-role` already grants what the probe needs at runtime: quote the policy
        statements for `bedrock:InvokeModel`+`bedrock:InvokeModelWithResponseStream` (lines ~95–97),
        `lambda:InvokeFunction` on `idp-engine-*` (lines ~100–102), and S3 r/w (lines ~98–99). State
        that NO task‑role change is needed to turn the probe on — only the task‑def env var, which
        Step 3 adds. (This confirms it from the script, not assumption, per the brief.)
      Files: none created (report section).
      Verify: the final report's Step‑5 section contains runnable commands and the quoted role
      policy lines; the quotes match `deploy_idp_fargate.py` as read.

---

## Final report (the step's deliverable)

- [ ] 6. Assemble the full report with: per‑file summary of each new script and test; test counts
      (pytest results for items 2T/3T/4T, run with the repo venv python); Step‑1 findings a/b/c/d
      including the 1b worktree pass/fail and the exact missing files, and the explicit statement
      that `*.onnx` is NOT in git but IS in the docker build context; and a CONTRADICTIONS section
      listing: (i) build logic is in `deploy_idp_ecr_images.py`, not `deploy_idp_fargate.py` (F1);
      (ii) `metadata.extra["probe"]`/DB are NOT where per‑page probe results are retrievable —
      CloudWatch logs are (F5); (iii) `deploy_idp_fargate.py` and the other `deploy_idp_*.py` are
      themselves untracked (F2/F2b). Per the task rule, these contradictions are REPORTED, and none
      of them block authoring the scripts, so proceed — but surface them prominently.
      Files: none (this plan's consumer writes the report).
      Verify: report cross‑checked against every `[C]` line‑cited fact above.

---

## Assumptions / open items (reasonable defaults taken; not blocking)

- Test location: new tests go under `tests/` (matching `tests/test_probe*.py`); if the repo keeps
  script tests elsewhere, follow that — check during implementation.
- `rollback_*.json` contains real secrets; the plan FLAGS adding `rollback_*.json` to `.gitignore`
  but does not edit `.gitignore` (no production‑file edits without approval). The scripts mask
  secrets in all stdout.
- The runtime default ONNX filename (`lipika_script.onnx` vs `lipika_script.int8.onnx`) is
  resolved by reading `script_classifier.py` in item 1a; Step 2's verifier checks whichever is the
  default (and both if both ship).
- No live AWS call is made anywhere in authoring or verification; every script is exercised only
  via `--dry-run` and mocked unit tests.
