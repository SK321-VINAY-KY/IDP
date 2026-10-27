# process_document() Runtime Trace — Where does it actually run in the deployed system?

Read-only investigation. Nothing was edited, committed, invoked, or deployed. AWS calls were list/get/describe only.

Evidence convention: `[C]` = confirmed by reading the file/running the command; `[I]` = inferred.

---

## Summary answer (read this first)

- **`process_document()` is NOT invoked by any deployed Lambda or Step Functions state.** The only code that calls it in non-test code is (a) the **web app background thread** in the Fargate tier, and (b) CLI/demo scripts. `[C]`
- In **production that means the ECS/Fargate web container** (`idp-layer3-app`, service `idp-layer3-service`), where `process_document()` runs on a `threading.Thread` started by `/pipeline/run` and `/users/.../run`. `[C]`
- The deployed **`idp-layer1-router` Lambda does NOT call `process_document`** — it imports only `inspect_page` + `router` and returns a per-page routing manifest. It happens to *bundle* `pipeline.py` in its ZIP, but never calls it. `[C]`
- **Step Functions is not deployed.** There is no `create_state_machine`/`update_state_machine` call anywhere in the repo; only `.asl.json` templates and scripts that `start_execution` against a hardcoded ARN. Live `list-state-machines` returned empty in every account/region checked. `[C]`
- The `.asl.json` orchestrator references a **probe Lambda (`${PROBE_LAMBDA_ARN}`) that no `deploy_idp_*.py` script ever creates** (only `Dockerfile.probe` exists). `[C]`
- **Therefore `IDP_PROBE_ENABLED=1` belongs on the Fargate task** (`idp-layer3-task` in `scripts/deploy_idp_fargate.py`), because that is the only deployed component that runs `process_document()`. That image already contains onnxruntime + opencv + pymupdf + boto3 + the lipika `.onnx` model, and the task role already has `bedrock:InvokeModel` + `lambda:InvokeFunction`. `[C]`

> **Account caveat (important):** the AWS creds currently active in this environment resolve to account **470561032098** (`arn:aws:iam::470561032098:user/RAG`), which is **NOT** the deploy-target account the scripts hardcode (**530084039784**, ap-south-1). The live findings below therefore describe account 470561032098, where the only idp resource found is `idp-layer1-router` in **us-east-1**. The scripts' target account 530084039784 could not be inspected with the active creds. `.env2` holds separate SSO creds for 530084039784 (referenced by presence only — not used, not echoed).

---

## 1. Static caller table (`process_document` / `process_page`, tests excluded)

| Symbol | Caller file | Enclosing function | Entry-point kind | Evidence |
|---|---|---|---|---|
| `process_document` | `schema_chatbot_v2/app/api/pipeline_routes.py` | `_run_pipeline_job()` (run on `threading.Thread`, `daemon=True`) | **Web/API background worker inside Fargate web app** | import line 37; call ~line 679; thread spawned at ~line 568 (`target=_run_pipeline_job`); a second spawn path via `app/api/user_routes.py` (imports `_run_pipeline_job`, thread at ~line 275) `[C]` |
| `process_document` | `schema_chatbot_v2/app/api/pipeline_routes.py` | second call site ~line 1412 | Web/API route helper (same process) | call ~line 1412 `[C]` |
| `process_document` | `demo_end_to_end.py` | main demo flow | **CLI / demo script** | import line 65; call ~line 401 `[C]` |
| `process_document` | `demo_resume_pipeline.py` | main demo flow | **CLI / demo script** | import line 39; call ~line 206 `[C]` |
| `process_page` | `scripts/demo.py` | `run_demo()` | **CLI / demo script** | import line 35; call ~line 168 `[C]` |
| `process_page` | `scripts/run_escalation_test.py` | main | **CLI / dev test script** | import line 15; call ~line 44 `[C]` |
| `process_page` | `src/ai/layer1_routing/pipeline.py` | `process_document()` page loop | internal (serial + `_worker` concurrent) | calls ~line 745 and ~line 760 `[C]` |

No Lambda handler, no Step Functions task, and no queue consumer calls `process_document`/`process_page`. `[C]`

## 2. Handler / entry-point table (`src/api/` and `docker/`)

| Component | Handler | Core function it dispatches to | Calls `process_document`? | Evidence |
|---|---|---|---|---|
| `src/api/lambda_layer1_handler.py` | `lambda_handler` (alias `handler`) | `inspect_page` + `route_from_profile` / `capabilities_from_profile` + `build_engine_plan`; returns a per-page routing manifest | **No** — single-purpose router | full read of file `[C]` |
| `src/api/lambda_probe_handler.py` | `lambda_handler` (alias `handler`) | `src.ai.layer1_probe.probe.probe_document(...)` (standalone probe over `pdf_bytes` from S3) | No | full read of file `[C]` |
| `docker/docling/handler.py` | `lambda_handler` | `engine.process_pages(pdf, pages)` | No — single engine | lines 1–60 + call at line 82 `[C]` |
| `docker/paddle_printed/handler.py` | `lambda_handler` | `engine.process_pdf_pages` (alias `process_pages`) | No — single engine | lines 1–50; `engine.py:187` alias `[C]` |
| `docker/paddle_handwritten/handler.py` | `lambda_handler` | `engine.process_pdf_pages` (alias `process_pages`) | No — single engine | `engine.py:186` alias `[C]` |

Every engine Lambda runs ONE engine on specific page numbers. The orchestration that would chain them (`process_document`, or the Step Functions Map) is the missing piece in production (see §5b).

## 3. Deployment table (repo scripts)

| Component | Deploy mechanism | Deployed AS | Name(s) | Memory / Timeout | Key env vars | Evidence |
|---|---|---|---|---|---|---|
| Layer 1 router | `scripts/deploy_idp_lambdas.py::deploy_zip_lambda` | **Lambda (ZIP, python3.11)** | `idp-layer1-router` | 1024 MB / 120 s | `IDP_ROUTING_MODE=capability_based` | `[C]` |
| Layer 1 router (older/alt) | `scripts/deploy_layer1_lambda.py` | **Lambda (ZIP, python3.11)**, default region `us-east-1` | `idp-layer1-router` | **512 MB / 60 s** | `IDP_ROUTING_MODE=capability_based` | `[C]` — matches the LIVE function (§4) |
| Docling engine | `scripts/deploy_idp_lambdas.py::deploy_container_lambda` | **Lambda (container image)** | `idp-engine-docling` (`idp-layer2-docling:latest`) | 3008 MB / 300 s | — | `[C]` |
| Paddle printed engine | same | **Lambda (container image)** | `idp-engine-paddle-printed` (`idp-layer2-paddle-printed:latest`) | 3008 MB / 300 s | — | `[C]` |
| Paddle handwritten engine | same | **Lambda (container image)** | `idp-engine-paddle-handwritten` (`idp-layer2-paddle-handwritten:latest`) | 3008 MB / 300 s | — | `[C]` |
| Web/console + pipeline orchestrator | `scripts/deploy_idp_fargate.py` | **ECS Fargate** (cluster `idp-cluster`, service `idp-layer3-service`, task family `idp-layer3-task`, container `idp-web-console`, image `idp-layer3-app`, ALB `idp-web-alb`) | — | cpu 2048 / mem 4096 (task) | see env list below | `[C]` |
| ECR images | `scripts/deploy_idp_ecr_images.py` | ECR repos + image build/push | `idp-layer2-docling`, `idp-layer2-paddle-printed`, `idp-layer2-paddle-handwritten`, `idp-layer3-app` | — | — | `[C]` |
| S3 + RDS | `scripts/deploy_idp_s3_rds.py` | S3 bucket + RDS Postgres | bucket `idp-internal-35385`, `idp-postgres-db` | — | — | `[I]` (named in fargate script constants) |
| Probe Lambda | **`Dockerfile.probe` only — NO create_function anywhere** | would be a container Lambda | (intended `${PROBE_LAMBDA_ARN}`) | — | handler `src.api.lambda_probe_handler.lambda_handler` | `[C]` — not deployed by any script |
| Step Functions | **none** — `.asl.json` templates only, no `create_state_machine` | — | (ARN hardcoded in test scripts as `idp-document-processing`) | — | — | `[C]` |

**All `deploy_idp_*.py` hardcode account `530084039784` / `ap-south-1`** and `assert acct == "530084039784"`. `[C]`

**Fargate task env (from `deploy_idp_fargate.py::register_task`)** — actual list: `APP_ENV`, `PYTHONPATH=/app/schema_chatbot_v2:/app`, `IDP_APP_ENV`, `AWS_DEFAULT_REGION`, `BEDROCK_REGION`/`IDP_BEDROCK_REGION`, `LLM_PROVIDER`/`IDP_LLM_PROVIDER=bedrock`, `EXTRACTION_BACKEND`/`IDP_EXTRACTION_BACKEND=bedrock`, `BEDROCK_MODEL_ID`/`IDP_BEDROCK_MODEL_ID` (inference profile ARN), `BEDROCK_VLM_MODEL_ID`/`IDP_BEDROCK_VLM_MODEL_ID` (same profile), `VLM_MODEL_NAME`/`IDP_VLM_MODEL_NAME`, `BEDROCK_REASONING_EFFORT=low`, `LAYER3_STRATEGY`/`IDP_LAYER3_STRATEGY=graph_memory_concurrent`, `DATABASE_URL`/`IDP_DATABASE_URL`, `IDP_S3_BUCKET`/`S3_BUCKET_NAME=idp-internal-35385`, `PADDLE_PREFER_LAMBDA`/`IDP_PADDLE_PREFER_LAMBDA=true`, `PADDLE_LAMBDA_REGION`/`IDP_PADDLE_LAMBDA_REGION`, `PADDLE_PRINTED_LAMBDA_FUNCTION`/`IDP_PADDLE_PRINTED_LAMBDA_FUNCTION` (ARN), `PADDLE_HANDWRITTEN_LAMBDA_FUNCTION`/`IDP_PADDLE_HANDWRITTEN_LAMBDA_FUNCTION` (ARN), `ENABLE_VLM_DIRECT`/`IDP_ENABLE_VLM_DIRECT=false`, `ROUTING_MODE`/`IDP_ROUTING_MODE=capability_based`, `GRAPH_CONCURRENCY_LIMIT=16`, `PAGE_CONCURRENCY_LIMIT=25`, `ESCALATION_CONFIDENCE_THRESHOLD=0.70`, `JWT_SECRET`, `ADMIN_PASSWORD`, `USER_STORE_TYPE=postgres`, `ALLOW_FILE_USER_STORE_IN_PROD=true`, `LOG_LEVEL=INFO`, `CORS_ORIGINS`. `[C]`

- **`IDP_PROBE_ENABLED` is ABSENT from the Fargate task env.** `[C]`
- **`IDP_ENABLE_PROBE` (the old per-page hook flag) is ABSENT too.** `[C]`
- Neither the L1 router Lambda nor the engine Lambdas set either probe flag. `[C]`

**Task IAM role (`idp-ecs-task-role`, `ensure_task_role`)** grants: `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream` on the inference profile; S3 r/w on `idp-internal-35385`; `lambda:InvokeFunction` on `idp-layer1-router` and `idp-engine-*`; CloudWatch logs. `[C]`

## 4. Live AWS results (region-swept; read-only)

`aws sts get-caller-identity` **succeeded** → account **470561032098**, `arn:aws:iam::470561032098:user/RAG`. This is **not** the scripts' target account 530084039784, so these results do not reflect the script-defined deployment. `[C]`

| Check | ap-south-1 | us-east-1 | us-west-2 |
|---|---|---|---|
| `lambda list-functions` (idp-*) | none | **`idp-layer1-router`** | none |
| `stepfunctions list-state-machines` | empty | empty | — |
| `ecs list-clusters` | empty | empty | — |

Live `idp-layer1-router` (us-east-1) config via `get-function-configuration`: Runtime `python3.11`, PackageType `Zip`, **MemorySize 512, Timeout 60**, Handler `src.api.lambda_layer1_handler.lambda_handler`, LastModified `2026-09-23`, Env `{IDP_ROUTING_MODE: capability_based}`. `[C]` This matches `scripts/deploy_layer1_lambda.py` (512/60/us-east-1), NOT `deploy_idp_lambdas.py` (1024/120). `[I]` the live function was deployed by the older script.

`ec2 describe-instances` filtered to idp: not separately run after the account mismatch was identified; no EC2 is defined by any deploy script (compute is Lambda + Fargate only). `[C]` for the repo; `[I]` for live.

**Net live picture (account 470561032098):** only the Layer 1 router Lambda exists; **no engine Lambdas, no Step Functions, no ECS/Fargate web tier**. So in this account `process_document()` has nowhere to run at all. The full stack (if it exists) lives in 530084039784, which was not inspectable here.

## 5. The four plain answers

### a) What invokes `process_document` in production?

Nothing in any Lambda or Step Functions. In the intended production topology it runs **only inside the Fargate web container** `idp-layer3-app` (service `idp-layer3-service`), on a background `threading.Thread` started by the FastAPI routes:
- `schema_chatbot_v2/app/api/pipeline_routes.py:37` import; `:568` `threading.Thread(target=_run_pipeline_job, …, daemon=True)`; `_run_pipeline_job` calls `process_document(...)` at `~:679`. `[C]`
- `schema_chatbot_v2/app/api/user_routes.py` imports `_run_pipeline_job` and spawns the same thread (`~:275`). `[C]`

Everywhere else it is CLI/demo only (`demo_end_to_end.py`, `demo_resume_pipeline.py`). So: **the Fargate web app is the sole production invoker**; the engine Lambdas are leaf workers it calls via `paddle_prefer_lambda=true` (`src/ai/layer2_conversion/scanned.py:255` → `_invoke_paddle_lambda`). `[C]`

### b) Is the Step Functions state machine deployed and used?

**No — not deployed.** `grep` for `create_state_machine`/`update_state_machine` across the repo finds **zero** matches; only `start_execution`/`describe_execution` (in `scripts/test_step_functions.py`, `scripts/run_missouri_workflow.py`) and `tag_resource` against a hardcoded ARN `arn:aws:states:…:stateMachine:idp-document-processing` (`scripts/tag_existing_resources.py:102`). `[C]` Live `list-state-machines` was empty in both regions checked. `[C]` The orchestrator template `step_functions/state_machine_with_probe.asl.json` wires `ProbeDocument → ${PROBE_LAMBDA_ARN}` then a Map over engine Lambdas — but `${PROBE_LAMBDA_ARN}` points at a probe Lambda that **no deploy script creates** (only `Dockerfile.probe` exists, built context = repo root, handler `src.api.lambda_probe_handler.lambda_handler`). `[C]` So the SFN path is a **design artifact / test harness, not a live production path.**

### c) Where would `IDP_PROBE_ENABLED=1` have to be set, and what does that component need?

**Set it on the Fargate task** `idp-layer3-task` in `scripts/deploy_idp_fargate.py::register_task` (add `{"name": "IDP_PROBE_ENABLED", "value": "true"}` to the `env` list), because the Fargate web container is the only component that runs `process_document()` (answer a). The flag is read as `settings.probe_enabled` (`src/config/settings.py:345`, env `IDP_PROBE_ENABLED`, default `False`). `[C]`

That component already has what the per-document probe needs:
- **Memory:** task is cpu 2048 / mem 4096 MB. `[C]` onnxruntime + a small lipika model fit comfortably; VLM work is an API call (not local), so no extra RAM for that. `[I]`
- **Image contents:** `schema_chatbot_v2/Dockerfile` does `COPY . /app`, shipping `src/ai/layer1_probe/` and the model `src/ai/layer1_probe/models/lipika_script.onnx`. `[C]` `.dockerignore` excludes `*.bin`/`*.safetensors` but **not `*.onnx`**, so the model is included. `[C]` `schema_chatbot_v2/requirements.txt` installs `onnxruntime>=1.18.0`, `opencv-python-headless>=4.9.0`, `pymupdf>=1.28.0`, `boto3>=1.34`, `paddleocr`/`paddlepaddle` — all present, with a comment stating they sit idle until the probe flag is on. `[C]` A Bedrock-capable client exists (LLM_PROVIDER=bedrock; VLM model id env set). `[C]`
- **IAM:** task role already grants `bedrock:InvokeModel`/`…WithResponseStream` on the inference profile (for the VLM probe) and `lambda:InvokeFunction` on `idp-engine-*` (for the Paddle engines the probe-chosen routes resolve to). `[C]` No IAM change needed.

So turning the per-document probe on in production is **a one-line task-env addition + redeploy of the Fargate service** — no image or IAM change. `[I]` (based on the confirmed image/role contents). Caveat: this is all for account 530084039784; in the live-checked account 470561032098 there is no Fargate tier to set it on.

### d) Which are the Layer 1 and Layer 2 Lambdas?

- **Layer 1 (routing manifest):** `idp-layer1-router` → handler `src.api.lambda_layer1_handler.lambda_handler` (ZIP, python3.11). `[C]` (Live in us-east-1, acct 470561032098.)
- **Layer 2 (per-engine conversion):**
  - `idp-engine-docling` → `docker/docling/handler.py::lambda_handler` → `engine.process_pages` `[C]`
  - `idp-engine-paddle-printed` → `docker/paddle_printed/handler.py::lambda_handler` → `engine.process_pdf_pages` `[C]`
  - `idp-engine-paddle-handwritten` → `docker/paddle_handwritten/handler.py::lambda_handler` → `engine.process_pdf_pages` `[C]`
- There is also an intended **Layer 1.5 probe** Lambda (`src/api/lambda_probe_handler.py::lambda_handler`, built by `Dockerfile.probe`) but it is **not deployed by any script**. `[C]`

---

## Conclusion — where `IDP_PROBE_ENABLED` belongs

`process_document()` is a **Fargate-only** entry point. The per-document probe (the Option-B feature guarded by `settings.probe_enabled`) will only ever execute inside the `idp-layer3-app` ECS/Fargate web container. The single change required to enable it in production is adding `IDP_PROBE_ENABLED=true` to the task env in `scripts/deploy_idp_fargate.py::register_task` and redeploying the service; the image already bundles onnxruntime/opencv/pymupdf/boto3 + the lipika `.onnx` model, and the task role already carries the Bedrock and Lambda-invoke permissions the probe needs. The Lambda fleet (`idp-layer1-router` + `idp-engine-*`) and the Step Functions `.asl.json` do **not** run `process_document()` and are unaffected by the flag; the standalone probe Lambda (`lambda_probe_handler` / `probe_document`) is a separate, currently-undeployed path and is not where this flag takes effect.

### Recommendations (not implemented)
1. If the probe should run in prod, add `IDP_PROBE_ENABLED` to the Fargate task env (one line) and redeploy `idp-layer3-service`. Keep it `false` for a byte-identical baseline first.
2. The two `step_functions/*.asl.json` reference a probe Lambda with no deployment path — either add a `create_function` for `Dockerfile.probe` or treat the SFN orchestrator as decommissioned to avoid confusion.
3. Reconcile the two Layer-1 deploy scripts (`deploy_layer1_lambda.py` 512/60/us-east-1 vs `deploy_idp_lambdas.py` 1024/120/ap-south-1) — the live function matches the former, so the newer script has not been run against the live account.
4. Confirm which AWS account is authoritative: active creds = 470561032098 (only the L1 router exists there, in us-east-1); the deploy scripts target 530084039784/ap-south-1. The user should verify where the real Fargate stack lives before wiring the flag.
