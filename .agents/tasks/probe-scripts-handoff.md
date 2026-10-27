# Probe Deployment & Measurement — Hand-off Notes & Consolidated Report

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a`  Branch: `feat/llm-observability`
Venv python: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`
Account `530084039784`, region `ap-south-1`. ECS cluster `idp-cluster`, service
`idp-layer3-service`, task family `idp-layer3-task`, ECR repo `idp-layer3-app`.

> Nothing in this hand-off was executed against AWS. No `docker build/push`, no `git push`,
> no `git commit`, no `.onnx` staged, no edits to existing prod source or to
> `scripts/deploy_idp_fargate.py`. All commands below are for the user to run after review.
> **Always run `--dry-run` first.**

---

## 1. Build + push the probe image — `scripts/deploy_probe_image.py`

The script auto-derives the tag `probe-<UTCyyyymmddHHMM>-<shortsha>[-dirty]` from git,
verifies the ONNX model + labels are in the build context (prints size + sha256), refuses
`:latest`, and mirrors the real build params from `deploy_idp_ecr_images.py`
(context = repo root, Dockerfile = `schema_chatbot_v2/Dockerfile`, `--provenance=false
--platform linux/amd64`, repo `idp-layer3-app`).

```powershell
# DRY RUN — prints login/build/push commands + the final image URI, runs nothing:
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_image.py --dry-run

# REAL build + push (resolves ECR login via boto3, builds, pushes):
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_image.py
```

The real run prints ONLY the final image URI on stdout (diagnostics go to stderr). Capture it:

```powershell
$IMAGE = c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_image.py
echo $IMAGE   # e.g. 530084039784.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-202610251144-acd90d1-dirty
```

---

## 2. Deploy with the probe switched ON — `scripts/deploy_probe_service.py`

Clones the CURRENT live task def, flips ONLY the container image and
`IDP_PROBE_ENABLED=true` (leaves `IDP_PROBE_N_SAMPLES` at its default), writes a
`rollback_<ts>.json` snapshot FIRST, registers a new revision, updates the service, waits
for `services-stable`. Pass the image URI from step 1.

```powershell
# DRY RUN — prints the register JSON (secrets masked) + the update-service call, no mutation, no rollback file:
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_service.py --image $IMAGE --dry-run

# REAL deploy, probe ON:
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_service.py --image $IMAGE
```

Defaults: `--cluster idp-cluster --service idp-layer3-service --region ap-south-1`. Override
with `--cluster/--service/--region` if needed; `--container` overrides the target container
(default `idp-web-console`).

---

## 3. Deploy the image with the probe OFF — `--flag-off`

Deploy the new image now with `IDP_PROBE_ENABLED=false`, then switch on later by re-running
step 2 without `--flag-off`.

```powershell
# DRY RUN:
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_service.py --image $IMAGE --flag-off --dry-run

# REAL deploy, probe OFF:
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\deploy_probe_service.py --image $IMAGE --flag-off
```

---

## 4. Rollback — `scripts/rollback_probe_service.py`

Two input modes (mutually exclusive, exactly one required): a rollback snapshot file written
by step 2, or a task-definition ARN directly. Updates the service back and waits for stable.
If the saved revision was deregistered, it re-registers from the stored JSON.

```powershell
# From the snapshot file (DRY RUN then real):
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\rollback_probe_service.py --rollback-file rollback_<ts>.json --dry-run
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\rollback_probe_service.py --rollback-file rollback_<ts>.json

# Or straight to a known-good revision ARN (the current pre-probe def is :3):
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\rollback_probe_service.py --task-definition-arn arn:aws:ecs:ap-south-1:530084039784:task-definition/idp-layer3-task:3 --dry-run
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\rollback_probe_service.py --task-definition-arn arn:aws:ecs:ap-south-1:530084039784:task-definition/idp-layer3-task:3
```

> The `rollback_<ts>.json` snapshot contains UNMASKED env secrets (needed to re-register).
> Do NOT commit it. Add `rollback_*.json` to `.gitignore` (flagged, not edited here).

---

## 5. Tail CloudWatch logs for the `probe.*` events

Log group is `/ecs/idp-layer3-app`, stream prefix `web`. This is fixed by the task def's
`logConfiguration` in `deploy_idp_fargate.py` (the `{REPO}` there is `idp-layer3-app`):

```python
# scripts/deploy_idp_fargate.py — register_task(), container def:
"logConfiguration": {"logDriver": "awslogs", "options": {"awslogs-group": f"/ecs/{REPO}", "awslogs-region": REGION, "awslogs-stream-prefix": "web"}},
```

Live tail (filtered client-side to probe events):

```powershell
aws logs tail /ecs/idp-layer3-app --region ap-south-1 --follow --format short --filter-pattern "probe"
```

Logs Insights query (pull the probe + routing events for a run):

```
fields @timestamp, correlation_id, event, page_number, lipika_script, vlm_script, agree, engines_used, escalated
| filter event in ["probe.lipika_vs_vlm", "probe.failed", "pipeline.page_complete"]
| sort @timestamp asc
| limit 1000
```

Run it and export for the summariser:

```powershell
aws logs start-query --region ap-south-1 --log-group-name /ecs/idp-layer3-app `
  --start-time <epoch> --end-time <epoch> `
  --query-string 'fields @timestamp, correlation_id, event, page_number, lipika_script, vlm_script, agree, engines_used, escalated | filter event in ["probe.lipika_vs_vlm","probe.failed","pipeline.page_complete"] | sort @timestamp asc | limit 1000'
aws logs get-query-results --region ap-south-1 --query-id <id>
```

Save the matched JSON log objects one-per-line to an `.ndjson` file, then run the summariser
(section 7 of the measurement findings below).

---

## 6. IAM permissions the SSO user needs to RUN these scripts

These are for the human/SSO caller, distinct from the task ROLE (which covers runtime — see
section 7). Minimum set:

**ECR (build + push via `deploy_probe_image.py`):**
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`
- `ecr:DescribeRepositories`, `ecr:TagResource` (the script best-effort tags the existing
  repo; it never creates one, so `ecr:CreateRepository` is not required)

**ECS (deploy / rollback via `deploy_probe_service.py` / `rollback_probe_service.py`):**
- `ecs:DescribeServices`
- `ecs:DescribeTaskDefinition`
- `ecs:RegisterTaskDefinition`
- `ecs:UpdateService`
- `ecs:DescribeTasks` (service-stable waiter)
- `ecs:TagResource`

**IAM:**
- `iam:PassRole` on both the task role (`idp-ecs-task-role`) and the execution role
  (`idp-ecs-task-execution-role`) carried by the task def (required to register a revision
  referencing those roles)

**STS:** `sts:GetCallerIdentity` (account assertion in the image script)

**CloudWatch Logs (measuring):**
- `logs:StartQuery`
- `logs:GetQueryResults`
- `logs:DescribeLogGroups`
- (plus `logs:FilterLogEvents` / `logs:GetLogEvents` for `aws logs tail`)

---

## 7. CONFIRMED from `deploy_idp_fargate.py` — the task ROLE already covers runtime needs

No task-role change is needed to turn the probe on. The probe runs inside the existing
`idp-ecs-task-role`, and `ensure_task_role()` in `scripts/deploy_idp_fargate.py` already
grants exactly what the probe path needs. Quoted verbatim from the file:

Bedrock InvokeModel (the VLM-probe calls go through Bedrock):
```python
{"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
 "Resource": [INFERENCE_PROFILE, f"arn:aws:bedrock:{REGION}:{ACCOUNT}:application-inference-profile/*"]},
```

S3 read/write:
```python
{"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
 "Resource": [f"arn:aws:s3:::{S3_BUCKET}", f"arn:aws:s3:::{S3_BUCKET}/*"]},
```

Lambda invoke (the Paddle engine Lambdas, `idp-engine-*`):
```python
{"Effect": "Allow", "Action": ["lambda:InvokeFunction"],
 "Resource": [f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-layer1-router",
              f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-*"]},
```

Conclusion: enabling the probe is purely a task-def ENV change (`IDP_PROBE_ENABLED=true`),
which step 2 performs by cloning the live def. The task role is already sufficient for
Bedrock InvokeModel, `lambda:InvokeFunction` on `idp-engine-*`, and S3.

---

## Step-1 findings (commit hygiene)

**1a — runtime file inventory for the probe:**
- Tracked (ship via git): all of `src/ai/layer1_probe/{aggregate,cv_features,imaging,inprocess,
  probe,routing,script_classifier,script_verify,types,vlm_probe,__init__}.py` and
  `models/lipika_script.labels.json`.
- UNtracked (`??`) and required at runtime: `src/ai/layer1_probe/windows.py` and
  `src/ai/layer2_conversion/vlm_block_rollup.py` (the latter is imported by `pipeline.py:37`).
- Gitignored by design (`*.onnx`, `.gitignore:50`): `models/lipika_script.onnx`
  (111,642,305 B) and `models/lipika_script.int8.onnx` (28,494,953 B). These ship ONLY via
  the docker build `COPY`, never via git.

**1b — clean-worktree test (the one allowed git action):** `git worktree add ../idp-clean HEAD`
then pytest, then `git worktree remove --force`.
- `tests/test_pipeline_probe_document.py` and `tests/test_pipeline_probe_integration.py`
  FAILED collection in clean HEAD: `ModuleNotFoundError: No module named
  'src.ai.layer2_conversion.vlm_block_rollup'` (pipeline.py:37). Clean HEAD also lacks
  `src/ai/layer1_probe/windows.py` and the two `.onnx` weights.
- `tests/test_probe.py` + `tests/test_probe_inprocess_images.py`: 44 passed (they use stubs,
  not the real pipeline import).
- Worktree removed; main tree `git status` unchanged. These untracked files must be committed
  (and the `.onnx` shipped via docker) for the probe path to import in a fresh checkout.

**1c — image build facts:** The build logic is NOT in `deploy_idp_fargate.py` (that file only
registers task-def/service). It lives in `scripts/deploy_idp_ecr_images.py`: context = repo
root, Dockerfile = `schema_chatbot_v2/Dockerfile`, `docker build --provenance=false
--platform linux/amd64`, repo `idp-layer3-app`. Root `.dockerignore` excludes `*.bin`,
`*.safetensors`, `.git/`, `tests/`, `.env*`, `.venv/`, `data/`, `logs/` — but NOT `*.onnx`
or `*.json`, and there is no `schema_chatbot_v2/.dockerignore`. So `*.onnx` + `labels.json`
ARE included in the image. `deploy_probe_image.py` verifies this at build time.

**1d — commit plan (proposed; nothing committed):** separate commits — `feat(probe)` (probe
runtime: `windows.py`, modified probe modules, labels.json, probe tests);
`fix(layer3)` (save_document_graph fix + its tests); `chore(config)` (S3 settings,
sentence-transformers pin, requirements); `test(schema)` (schema_chatbot test);
`docs(agents)` (AGENTS.md / .agents edits); `chore(deploy)` (the new probe scripts + tests,
and optionally the untracked `deploy_idp_*.py`). No `.onnx` path in any commit.

---

## Measurement availability (finding 4a)

Per-page probe decisions are **NOT persisted to disk or DB today.** `process_page` sets
`metadata.extra["probe"]`, but `extra` is never written out: `md_writer._document_footer`
emits only page/engines/confidence/escalated/etc. (no `extra`), the job result in
`pipeline_routes.py::_run_pipeline_job` persists DocumentIR blocks only, and
`layer3_extraction/storage.py` stores `engines_used/confidence/escalated` per page — never
the probe script/route decision.

The ONLY runtime-reachable probe signal is the **structured JSON log stream**
(`src/utils/logger.py` JSONFormatter -> stdout -> awslogs -> `/ecs/idp-layer3-app`, prefix
`web`). Relevant events:
- `probe.lipika_vs_vlm` — `page_number, lipika_script, lipika_conf, margin, n_windows,
  vlm_script, agree` (+ `correlation_id`). Source of lipika-vs-VLM agreement and VLM-probe
  call counts.
- `pipeline.page_complete` — `page_number, engines_used, confidence, capabilities`
  (+ `escalated`/`low_confidence` where present). Source of the FINAL engine per page
  (route proxy for the harmful check).
- `probe.failed` — `document_id, error`.

Grouping caveat: `probe.lipika_vs_vlm` carries no `document_id`, only `page_number`; pages
tie to a document via `correlation_id` (one per execution). The summariser groups by
`correlation_id` and falls back to a single synthetic doc when absent.

`scripts/summarize_probe_results.py` consumes an NDJSON log export (or local
`logs/pipeline.log`) and prints: pages per route (by final engine), escalation rate per route
(from the `escalated` flag, with an explicit "not available" message when absent), VLM probe
calls per document, lipika-vs-VLM agree/disagree, and with `--labels doc,script` per-script
accuracy + the HARMFUL count (labelled non-Latin pages routed to a Paddle engine; exits
non-zero if > 0). Metrics it cannot compute say so rather than invent a number.

```powershell
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\summarize_probe_results.py --logs probe_logs.ndjson
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts\summarize_probe_results.py --logs probe_logs.ndjson --labels labels.csv
```

---

## Per-file summary of the new scripts and tests

- `scripts/deploy_probe_image.py` — builds + pushes `idp-layer3-app` with a traceable
  `probe-<UTCts>-<sha>[-dirty]` tag. Pure helpers: `build_tag`, `assert_not_latest`,
  `parse_porcelain`, `build_labels`, `dockerignore_excludes`, `sha256_and_size`,
  `verify_models_in_context`, `image_uri`, `build_command`. `--dry-run` prints commands + URI
  (login password masked) and makes no AWS call. Real run resolves STS/ECR login, builds,
  pushes, prints only the URI. Refuses `:latest`.
- `scripts/deploy_probe_service.py` — clones the live task def, flips image +
  `IDP_PROBE_ENABLED` (true / false via `--flag-off`), leaves `IDP_PROBE_N_SAMPLES` alone,
  writes rollback snapshot first, registers + updates + waits. Pure helpers: `mask_secrets`,
  `mask_taskdef_for_print`, `strip_for_register`, `_select_container`, `upsert_env`,
  `apply_probe_changes`, `register_request_json`. `--dry-run` prints masked register JSON,
  writes no rollback file, no mutation. Rejects `:latest`.
- `scripts/rollback_probe_service.py` — rolls back via `--rollback-file` or
  `--task-definition-arn`; re-registers from snapshot if the revision is gone; waits for
  stable. `--dry-run` mutates nothing. Reuses helpers from `deploy_probe_service`.
- `scripts/summarize_probe_results.py` — log-driven metrics (see measurement section). Pure
  helpers: `iter_log_records`, `load_labels`, `summarize`, `harmful_rows`, `script_accuracy`,
  `render`. Exits non-zero only for HARMFUL > 0 or bad input.
- Tests: `tests/test_deploy_probe_image.py` (16), `tests/test_deploy_probe_service.py` (12),
  `tests/test_summarize_probe_results.py` (10) — all boto3/subprocess mocked, no network.
  Fixtures: `tests/fixtures/probe/{logs_clean.ndjson, labels_clean.csv, labels_harmful.csv}`.

### Test results
`.venv\Scripts\python.exe -m pytest tests/test_deploy_probe_image.py
tests/test_deploy_probe_service.py tests/test_summarize_probe_results.py -q` ->
**38 passed** (re-run during this hand-off; 0 failed). The clean-worktree probe suite in Step
1b: 44 passed (stub tests) with 2 files failing collection on the missing untracked module
(expected — see 1b).

---

## CONTRADICTIONS vs the task description (reported, non-blocking)

1. The brief says to mirror `deploy_idp_fargate.py`'s build helpers, but that file has **no
   build logic** — it only registers the ECS task def / service. The actual `docker build`
   + push for `idp-layer3-app` is in `scripts/deploy_idp_ecr_images.py`. `deploy_probe_image.py`
   mirrors the latter for context/Dockerfile/platform/repo, and imports constants
   (REGION/ACCOUNT/REPO/CLUSTER/SERVICE) from `deploy_idp_fargate.py` (import-safe).
2. The brief lists `PageMetadata.extra["probe"]` and "DB rows" as where per-page probe
   results can be fetched. Neither persists the probe decision today — the authoritative
   source is the CloudWatch JSON log stream (finding 4a above).
3. `scripts/deploy_idp_fargate.py` and the sibling `deploy_idp_*.py` scripts are themselves
   UNtracked in git. Importing `deploy_idp_fargate` is side-effect-free, so the probe scripts
   import it safely; each falls back to mirrored literals (with source-line cites) if the
   import fails in a stripped checkout.

---

## Final confirmation

- NO AWS command, `docker build/push`, `git push`, or `git commit` was performed.
- NO `.onnx` file was staged.
- NO edits to existing production source or to `scripts/deploy_idp_fargate.py`.
- The only AWS-touching paths in the scripts run exclusively via boto3 inside `main()`;
  every provided command above shows `--dry-run` first.
