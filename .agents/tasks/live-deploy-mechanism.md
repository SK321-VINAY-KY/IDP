# Live Deploy Mechanism & Probe Wiring — Investigation Findings

READ-ONLY investigation. No code or AWS resources were changed. All AWS calls were
`describe`/`list`/`get`/`get-caller-identity` only, run with the temporary SSO credentials
found in `.env2` (account 530084039784). Scratch scripts created under `.scratch_probe/` were
deleted afterward.

---

## Summary answer (read this first)

1. **Deploy mechanism = hand-rolled boto3 Python scripts, NOT CDK/SAM/Terraform/compose.** There is
   no `cdk.json`, no `app.py`, no `Stack(...)`, no `serverless.yml`, no `template.yaml`, no `*.tf`.
   The live system was provisioned by the staged scripts `scripts/deploy_idp_s3_rds.py` →
   `deploy_idp_ecr_images.py` → `deploy_idp_lambdas.py` → `deploy_idp_fargate.py`, each of which
   **hard-codes account `530084039784` and region `ap-south-1`** and asserts that account at runtime.
   `docker-compose.yml` is for local dev only (it builds two containers and runs a CLI demo; it has
   no AWS wiring). The web tier is ECS Fargate behind the ALB, created by `deploy_idp_fargate.py`.

2. **The "empty lambda list" discrepancy is an account mismatch.** Your AWS CLI default credentials
   are account **470561032098** (`user/RAG`) — an account with **nothing IDP in it**. The live app
   runs in account **530084039784** (SSO `AdministratorAccess`, `vinay.k@shellkode.com`), whose
   temporary creds are in `.env2`. A third account, **106611079163**, appears throughout `.env` and
   the older `deploy_layer2.py` / `deploy_layer3.py` / `test_step_functions.py` scripts — that is a
   **prior/abandoned deployment generation**. Querying 530084039784 with the `.env2` creds shows the
   ALB, ECS cluster/service, RDS, and the 4 Lambdas all present.

3. **The live pipeline does NOT use Step Functions at all, and the probe is NOT deployed.** In
   account 530084039784, `aws stepfunctions list-state-machines` returns **empty**, and
   `aws lambda get-function --function-name idp-layer1-probe` returns **ResourceNotFoundException**.
   The ECS task definition carries **no state-machine ARN** — only the two Paddle Lambda ARNs plus
   `PADDLE_PREFER_LAMBDA=true`. The live app runs the whole inspect→route→convert flow **in-process**
   inside the Fargate container (`src/ai/layer1_routing/pipeline.py::process_document`), and for OCR
   it invokes the Paddle engine Lambdas directly via `boto3 lambda.invoke` in
   `src/ai/layer2_conversion/scanned.py`. The two `step_functions/*.asl.json` files and
   `src/api/lambda_probe_handler.py` describe a **parallel orchestration that was never deployed to
   this account.**

4. **Therefore "update the probe into the live pipeline" is first-time integration, not a swap, and
   the natural integration point is the in-process ECS app — not Step Functions.** Wiring the probe
   by deploying `state_machine_with_probe.asl.json` would mean standing up an entire Step Functions
   orchestration the live app does not currently use (plus a probe Lambda, plus a nonexistent
   `${VLM_LAMBDA_ARN}`). The minimal, lower-risk path is to call the probe from inside the ECS app's
   `process_document` flow and redeploy the `idp-layer3-app` image via `deploy_idp_fargate.py`. This
   is a design decision the user should confirm — see Risks.

---

## Evidence

### A. Deploy mechanism is hand-rolled boto3 scripts (confirmed)

No IaC found. Searches for `aws_cdk`, `cdk.json`, `Stack(`, `serverless`, `template.yaml`, `*.tf`
returned nothing relevant in the repo. The deploy surface is entirely Python under `scripts/`.

**Two deployment generations exist in the repo:**

*Current / live generation — account 530084039784, creds from `.env2`:*

- `scripts/deploy_idp_s3_rds.py:14-18` — `REGION = "ap-south-1"`, `VPC_ID`, `DB_IDENTIFIER=idp-postgres-db`;
  `:166` asserts `account_id == "530084039784"`. Reuses the pre-created bucket `idp-internal-35385`
  (`ensure_s3`, `:40-46`, notes an SCP blocks `s3:CreateBucket` via CLI).
- `scripts/deploy_idp_ecr_images.py:12,70` — `REGION="ap-south-1"`, asserts `acct == "530084039784"`;
  builds+pushes 4 images (`idp-layer2-docling`, `idp-layer2-paddle-printed`,
  `idp-layer2-paddle-handwritten`, `idp-layer3-app`) using
  `docker build --provenance=false --platform linux/amd64 -f <dockerfile> .` then `docker push`
  (`:84-86`).
- `scripts/deploy_idp_lambdas.py:20-21,34-39` — asserts `530084039784`; creates L1 as a **ZIP**
  Lambda `idp-layer1-router` (`deploy_zip_lambda`, handler `src.api.lambda_layer1_handler.lambda_handler`,
  python3.11) and 3 **container** Lambdas (`idp-engine-docling`, `idp-engine-paddle-printed`,
  `idp-engine-paddle-handwritten`) from the ECR images.
- `scripts/deploy_idp_fargate.py:21-37` — the web tier. `ACCOUNT="530084039784"`, `CLUSTER="idp-cluster"`,
  `SERVICE="idp-layer3-service"`, `TASK_FAMILY="idp-layer3-task"`, `ALB_NAME="idp-web-alb"`,
  `L3_IMAGE=...dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:latest`. Creates the ALB
  (`ensure_alb`), target group on port 8000 (`/health` check), ECS task def (`register_task`) and
  service (`ensure_service`, which `update_service(..., forceNewDeployment=True)` when it already
  exists). The task env injects the Paddle Lambda ARNs and `PADDLE_PREFER_LAMBDA=true` (`:148-205`).
  `CORS_ORIGINS` includes `http://idp-web-alb-1724249754.ap-south-1.elb.amazonaws.com` (`:203`) —
  the exact live URL the user gave.

*Prior / abandoned generation — account 106611079163:*

- `scripts/deploy_layer2.py:131,ASL_FILE` — `--profile` default `106611079163_SK-ML-Sandbox-team-Ps`,
  `ASL_FILE = step_functions/state_machine.asl.json` (the **legacy, no-probe** machine). It only
  prepares ECR repos + a Step Functions IAM role; it does **not** call `create_state_machine`.
- `scripts/deploy_layer3.py:28-29` — profile `106611079163_SK-ML-Sandbox-team-Ps`, ECR repo
  `idp-schema-chatbot`, service `idp-schema-chatbot-service` (different names from the live
  530084039784 service `idp-layer3-service`).
- `scripts/test_step_functions.py:10-12` and `run_missouri_workflow.py:10-12` reference
  `arn:aws:states:ap-south-1:106611079163:stateMachine:idp-document-processing`.
- `.env` pins the paddle Lambda ARNs to **106611079163** (`PADDLE_PRINTED_LAMBDA_FUNCTION`,
  `PADDLE_HANDWRITTEN_LAMBDA_FUNCTION`) and Bedrock profiles to 106611079163. `.env` is the old
  generation's config; the **live ECS task env overrides all of these to 530084039784** (see §C).

**No script in the repo calls `create_state_machine` or `update_state_machine`.** A grep for those
APIs across `**/*.py` found only `start_execution` (test drivers) and a Step Functions IAM-role
helper (`deploy_layer2.py:65 ensure_stepfunctions_role`). So wherever the legacy
`idp-document-processing` machine lived (106611079163), it was registered manually/out-of-band, not
by repo tooling.

### B. Probe build/deploy path exists only on paper (confirmed)

- The probe **artifacts** exist: `Dockerfile.probe` (FROM `public.ecr.aws/lambda/python:3.12`, CMD
  `src.api.lambda_probe_handler.lambda_handler`), `requirements-probe.txt`, `src/api/lambda_probe_handler.py`,
  `src/ai/layer1_probe/*`, and `step_functions/state_machine_with_probe.asl.json`.
- The probe **deploy script does not exist.** A file search for "probe" under `scripts/` returns only
  `scripts/calibrate_probe_heuristic.py` (an offline calibration helper, not a deployer). No script
  references `Dockerfile.probe`, `idp-layer1-probe`, `requirements-probe`, or
  `state_machine_with_probe.asl.json`.
- `docs/IDE_PROMPTS.md` "Prompt 4 — deployment" (unexecuted TODO) spells out exactly what *would*
  need to happen: mirror `deploy_idp_ecr_images.py` + `deploy_idp_lambdas.py` to add a container
  Lambda **`idp-layer1-probe`** (mem 2048 MB, timeout 900 s, env `IDP_PROBE_PRINTED_SCRIPTS=latin`,
  `IDP_BEDROCK_VLM_MODEL_ID`), extend the exec role with `bedrock:InvokeModel`, and "Update the
  state-machine deploy step to use `state_machine_with_probe.asl.json` with `${PROBE_LAMBDA_ARN}`
  substituted. `${VLM_LAMBDA_ARN}` does not exist yet: STOP and tell me rather than inventing it."
- `state_machine_with_probe.asl.json:3-11` — `StartAt: ProbeDocument`, task
  `FunctionName: "${PROBE_LAMBDA_ARN}"` (an unresolved placeholder). It also references
  `${VLM_LAMBDA_ARN}` downstream, which does not exist.

Conclusion: the probe has **never been deployed** to the live account, and no automation exists to
deploy it. Prompt 4 was never run.

### C. Live AWS reconciliation (account 530084039784 / ap-south-1)

Credentials: `.env2` block `[530084039784_AdministratorAccess]`. Verified:
`aws sts get-caller-identity` →
`arn:aws:sts::530084039784:assumed-role/AWSReservedSSO_AdministratorAccess_.../vinay.k@shellkode.com`.

| Query | Result |
|---|---|
| `aws elbv2 describe-load-balancers` | `idp-web-alb` → `idp-web-alb-1724249754.ap-south-1.elb.amazonaws.com`, state **active** |
| `aws ecs list-clusters` | `arn:aws:ecs:ap-south-1:530084039784:cluster/idp-cluster` |
| `aws ecs list-services --cluster idp-cluster` | `.../idp-cluster/idp-layer3-service` |
| `aws ecs describe-task-definition --task-definition idp-layer3-task` | family `idp-layer3-task`, **rev 3**, image `530084039784.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:latest` |
| `aws stepfunctions list-state-machines` (ap-south-1) | **EMPTY** — no state machine exists |
| `aws lambda list-functions` (ap-south-1) | `idp-engine-docling`, `idp-engine-paddle-handwritten`, `idp-layer1-router`, `idp-engine-paddle-printed` (**no `idp-layer1-probe`**) |
| `aws lambda get-function --function-name idp-layer1-probe` | **ResourceNotFoundException: Function not found: arn:aws:lambda:ap-south-1:530084039784:function:idp-layer1-probe** |
| `aws ecr describe-repositories` | `idp-layer2-docling`, `idp-layer2-paddle-printed`, `idp-layer2-paddle-handwritten`, `idp-layer3-app`, `tvs-quick-demo` (**no probe repo**) |
| `aws lambda get-function-configuration --function-name idp-engine-paddle-printed` | PackageType **Image**, mem 10240 |
| `aws lambda get-function-configuration --function-name idp-layer1-router` | ZIP, python3.11, handler `src.api.lambda_layer1_handler.lambda_handler`, mem 1024, timeout 120 |

**Access-denied / out-of-scope notes (reported verbatim):**
- `aws stepfunctions list-state-machines --region ap-southeast-1` →
  `AccessDeniedException ... not authorized to perform: states:ListStateMachines ... with an explicit
  deny in a service control policy: arn:aws:organizations::377425386071:policy/.../p-azw3kr8z`.
  (The org applies SCPs; ap-south-1 is the sanctioned region.)
- `aws stepfunctions list-state-machines --region us-east-1` → empty.
- `aws lambda list-functions --region us-east-1` → only unrelated demos
  (`OrderSupportDemo-*`, `AgentCore-*`, `gateway_lambda`) — not IDP.

**Live ECS task env (rev 3, selected keys, confirmed via `describe-task-definition`):**
`PADDLE_PREFER_LAMBDA=true`, `PADDLE_PRINTED_LAMBDA_FUNCTION` / `PADDLE_HANDWRITTEN_LAMBDA_FUNCTION`
→ `arn:aws:lambda:ap-south-1:530084039784:function:idp-engine-paddle-{printed,handwritten}`,
`ROUTING_MODE=capability_based`, Bedrock `application-inference-profile/24n3l11zddro` (account
530084039784), `S3_BUCKET_NAME=idp-internal-35385`, RDS `idp-postgres-db...`. **There is no
`STATE_MACHINE_ARN`, no `PROBE_*`, and no Step Functions env of any kind.**

### D. How the live app actually orchestrates (confirmed from code)

- Web entrypoint: `schema_chatbot_v2/app/api/pipeline_routes.py:37`
  `from src.ai.layer1_routing.pipeline import process_document`. Jobs run on a background thread
  (`_run_pipeline_job`), i.e. **inside the Fargate container**.
- `src/ai/layer1_routing/pipeline.py` orchestrates `inspect → capabilities → plan → run → merge`
  entirely in-process and dispatches to `convert_digital_page` (Docling),
  `convert_scanned_page` / `convert_handwritten_via_paddle` (`src/ai/layer2_conversion/scanned.py`).
- `src/ai/layer2_conversion/scanned.py:243` — when `settings.paddle_prefer_lambda` is true it calls
  `_invoke_paddle_lambda(...)`, which does `boto3.client("lambda").invoke(FunctionName=<paddle ARN>, ...)`
  (`:109-113`). The invoke **payload is `{"image_base64": <jpeg b64>, "page_number": n}`** (`:104-107`)
  — note it sends an inline base64 image and does **not** send `ocr_lang`, `use_structure`,
  `image_s3_bucket`, or `image_s3_key`. (The probe→engine key contract documented in
  `.agents/tasks/probe-contract-findings.md` applies to the Step-Functions path, which is not live.)
- `src/ai/layer1_probe/*` and `src/api/lambda_probe_handler.py` are **not imported** by
  `pipeline.py` or anywhere under `schema_chatbot_v2/app/`. The probe is dead code at runtime today.

This matches the user's own statement: the app is already live; what they want is to get the probe
*into the pipeline*. The pipeline that is live is the in-process ECS one.

---

## Conclusions

- **(a) Deploy mechanism:** hand-rolled, idempotent boto3 scripts under `scripts/deploy_idp_*.py`,
  targeting account **530084039784 / ap-south-1**. No CDK/SAM/Terraform. The web tier is ECS Fargate
  (`idp-cluster` / `idp-layer3-service`, task `idp-layer3-task`) behind ALB `idp-web-alb`; Docling
  and the two Paddle engines are container Lambdas; Layer-1 router is a ZIP Lambda. `docker-compose.yml`
  is local-dev only.
- **(b) Where the live resources are:** account **530084039784**, region **ap-south-1**. The CLI's
  default creds (470561032098) and the `.env`/older-scripts account (106611079163) are **different
  accounts** — that is exactly why `aws lambda list-functions` looked empty. Topology:
  `ALB(idp-web-alb) → TG:8000 → ECS service idp-layer3-service (Fargate, idp-layer3-app:latest)`;
  the app calls `idp-engine-paddle-printed` / `-handwritten` (and `idp-engine-docling`,
  `idp-layer1-router`) **directly via lambda.invoke**. **No Step Functions state machine exists.**
- **(c) Is the probe in the live pipeline?** **No.** No `idp-layer1-probe` Lambda, no probe ECR repo,
  no state machine, and the probe modules are not imported by the running app. "Updating" the probe
  is therefore **first-time integration**, not an in-place update.
- **(d) Safe update path:** below.

---

## Safe, minimal path to put the probe into the LIVE pipeline

Two options. **Option 1 is recommended** because it matches how the app is actually orchestrated
today and avoids standing up an unused orchestration layer. Confirm the choice with the user before
implementing — the brief flags that first-time provisioning + an orchestration switch is a bigger
change than a simple "update."

### Option 1 (recommended): integrate the probe in-process in the ECS app

The live pipeline is in-process, so the lowest-risk integration is code, not new infra:

1. Call the probe (or its routing logic, `src/ai/layer1_probe/routing.py` / `probe.py`) from inside
   `src/ai/layer1_routing/pipeline.py::process_document`, at the point where a scanned/undetected
   page is about to be routed — matching the user's intended flow (PyMuPDF detects printed Latin →
   Docling; undetected/scanned → probe detects script/table; English scanned → Paddle; other script
   → VLM). The probe already runs CPU/Bedrock-only and needs no new AWS service.
2. If the probe should influence the Paddle call, thread `ocr_lang` into
   `src/ai/layer2_conversion/scanned.py::_invoke_paddle_lambda` (today its payload is only
   `{image_base64, page_number}` — `scanned.py:104-107`). The deployed engine handlers already read
   `ocr_lang`/`image_s3_*` from the event, so this is forward-compatible.
3. Rebuild and redeploy only the web image:
   `python scripts/deploy_idp_ecr_images.py --only idp-layer3-app` then
   `python scripts/deploy_idp_fargate.py` (which re-registers the task def and
   `forceNewDeployment=True`). Both already assert account 530084039784 and apply tags.
4. No new Lambda, no Step Functions, no `${VLM_LAMBDA_ARN}` dependency.

**Tagging:** `deploy_idp_ecr_images.py` and `deploy_idp_fargate.py` already set tags. ⚠️ They
currently use `createdby=vinay.k@shellkode.com` and `customer=idp-internal-35385`, but the workspace
rule mandates `createdby=vinay.k@shellkode` (no `.com`). The live resources were created with the
`.com` form. Reconcile per the user's latest instruction ("just update the customer tag" /
`customer=idp-internal-35385`) before any create/update, and keep all three mandatory tags:
`createdby=vinay.k@shellkode`, `purpose=internal`, `customer=idp-internal-35385`.

### Option 2: deploy the Step Functions probe pipeline (bigger change, only if the user wants SFN)

Only if the user explicitly wants the orchestration moved to Step Functions. This is **net-new infra**,
not an update:

1. Write a probe deploy script (Prompt 4 in `docs/IDE_PROMPTS.md`): build `Dockerfile.probe` → new ECR
   repo `idp-layer1-probe` → create container Lambda `idp-layer1-probe` (mem 2048, timeout 900, env
   `IDP_PROBE_PRINTED_SCRIPTS=latin`, `IDP_BEDROCK_VLM_MODEL_ID=<530084039784 inference profile>`),
   extend the Lambda exec role with `bedrock:InvokeModel` on the VLM profile ARN. Tag everything.
2. Create a Step Functions state machine from `state_machine_with_probe.asl.json`, substituting
   `${PROBE_LAMBDA_ARN}` and the four engine ARNs. **Blocker:** the ASL also references
   `${VLM_LAMBDA_ARN}`, which does not exist — per the repo's own instruction, STOP rather than invent
   it. Any page the probe routes to `vlm_transcribe` (all non-Latin by default) would fail until a VLM
   Lambda is built. Tag the state machine + its IAM role.
3. Switch the ECS app to *start executions* against the new machine instead of calling engines
   in-process — a non-trivial app change, since nothing in `schema_chatbot_v2/app` currently talks to
   Step Functions.

### Risks / ambiguities to surface to the user

- **"Update" is really "first deploy."** Neither the probe Lambda nor any state machine exists in the
  live account; the live app doesn't use Step Functions. Presenting this as a one-line probe update
  would be wrong — confirm scope.
- **Account confusion is the headline risk.** Three accounts are in play: live **530084039784**
  (`.env2`), legacy **106611079163** (`.env`, old scripts), and the CLI default **470561032098**
  (user/RAG, empty). Any deploy MUST use the `.env2` creds/role for 530084039784; the hardcoded
  account asserts in the scripts are a good guardrail. ⚠️ The `.env2` credentials are **temporary STS
  session tokens** and will expire — a fresh SSO login is needed before deploying.
- **`.env` is stale for prod.** Its ARNs point at 106611079163; the live task env overrides them. Do
  not rely on `.env` when reasoning about the live system.
- **VLM route is a live gap.** With `IDP_PROBE_PRINTED_SCRIPTS=latin`, the probe routes every non-Latin
  script (incl. Telugu/Hindi-Devanagari) to `vlm_transcribe`; that engine/Lambda is not provisioned.
  If Option 2 is chosen, VLM must be built first or non-Latin pages fail.
- **Secrets in the repo.** `.env2` (live STS creds) and `deploy_idp_fargate.py` (DB password, JWT
  secret, admin password in plain task env) contain live secrets. Flag for rotation; do not echo
  values. The task definition also stores DB/JWT/admin secrets as plaintext env rather than Secrets
  Manager — a separate hardening item.
- **SCP constraints.** The org enforces SCPs (observed explicit deny for Step Functions in
  ap-southeast-1; `s3:CreateBucket` blocked via CLI per `deploy_idp_s3_rds.py`). Stay in ap-south-1
  and reuse the existing tagged bucket.

---

*Citations verified by reading each repo file cited and by running the read-only AWS CLI calls listed
in §C with the `.env2` credentials for account 530084039784. "Confirmed" = observed directly in file
contents or CLI output; the recommended integration point (Option 1) is a design judgment grounded in
the confirmed in-process orchestration, not an observed deployment.*
