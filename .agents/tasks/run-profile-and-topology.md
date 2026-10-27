# Run profile and deployment topology: job_4cf06e33305d

Read-only investigation. No code, infrastructure, or AWS resource was changed.
AWS access was limited to describe, list, get, and CloudWatch Logs reads. Credentials were loaded from `.env2` into the process environment and never printed. Secret-looking environment values in the task definition are shown as `***MASKED***`.
All times are IST (+05:30), as the app logged them. Durations come from millisecond log timestamps and from the `latency_ms` fields. Items marked **[inferred]** are conclusions drawn from timing patterns or code, not from a direct trace.

## 1. Summary

- **Run profiled:** "Chander Kochhar 01_compressed 2.pdf", 76 pages, job `job_4cf06e33305d`, on ECS task definition `idp-layer3-task:6` (2 vCPU, 8,192 MB). Job wall clock was 11:48:00.24 to 12:06:14.65, which is 18 min 14 s (1,094 s).
- **Dominant cost:** graph-memory build (Layer 3). It took 842 s, which is 77% of the job. Most of that time is LLM output generation: median 4,546 output tokens per page at about 126 tokens/s.
- **Graph concurrency is configured for 16 but runs on about 6 threads.** `asyncio.to_thread` uses the default thread pool, which is sized from CPU count. **[inferred]** from the call start times, which match the end times of earlier calls.
- **OCR concurrency is 25.** The pipeline runs 25 page workers, and each one calls the Paddle printed Lambda synchronously. The Lambda took a median of 48.5 s per page under that load. The logs contain no isolated single-page OCR time.
- **Failures:** 7 of 76 pages (12 through 18) failed in graph build with a Bedrock `ReadTimeoutError` after about 300 s each. The final graph has 957 nodes and 1,357 edges from the 69 pages that succeeded. The graph does not contain pages 12 to 18.
- **Deployment:** one ECS Fargate service runs the web console, Layer 1 inspection, the probe, Layer 2 orchestration, Layer 3 graph build, and the Q&A bot. The OCR engines are container-image Lambdas invoked synchronously with boto3. The LLM and VLM calls go to Bedrock. The Layer-1 router Lambda is not on the live path.
- **Cold start:** nothing is configured to mitigate it. The Lambdas have no provisioned or reserved concurrency and no warmers. The 25 OCR containers in the first wave each cold-start. The Fargate task is always on, so cold start there is a deploy cost, not a per-request cost.

## 2. Run summary

| Item | Value | Source |
|---|---|---|
| Document | `Chander Kochhar 01_compressed 2.pdf`, 76 pages, all pages scanned (PyMuPDF `char_count` 0) | `page.inspected` events |
| Job id | `job_4cf06e33305d` (owner `admin`) | `storage.document_graph_saved` |
| ECS cluster / service | `idp-cluster` / `idp-layer3-service` (desired 1, running 1, FARGATE, rolling deploy, ALB `idp-web-alb`, target group `idp-web-tg` port 8000, health path `/health`) | `aws ecs describe-services` |
| Task definition | `idp-layer3-task:6`, cpu 2048, memory 8192, awsvpc, task role `idp-ecs-task-role`, registered 2026-10-26 11:41:28 | `aws ecs describe-task-definition` |
| Running task | `69bf62d5026c4a9f8e3e21db25faaf06`, created 11:42:14, started 11:45:01 | `aws ecs describe-tasks` |
| Probe flag | `IDP_PROBE_ENABLED=true` | task definition env |
| Concurrency env | `PAGE_CONCURRENCY_LIMIT=25`, `GRAPH_CONCURRENCY_LIMIT=16` | task definition env |
| Job wall clock | 1,094.4 s (11:48:00.244 to 12:06:14.650) | `llm.factory.selected` to `storage.job_pdf_saved` |
| Outcome | 76 pages converted, 1 page escalated to VLM (page 64), graph 957 nodes / 1,357 edges, 7 graph pages failed | `pipeline.document_complete`, `layer3.graph_memory_concurrent.done` |

Other runs in the same log window:

- **10:47 to 10:48 (task `16b738...`):** every Bedrock call failed with `AccessDeniedException` for `bedrock:InvokeModel`, which was raised for `idp-ecs-task-role`. All 76 graph calls and all 76 Bedrock handwriting transcriptions failed. This run is not the profiled one. I did not check whether the IAM policy changed before the 11:48 run, which succeeded.
- **11:27 (task `d2cf9d...`):** one page-inspection pass, then no further events. The task was stopped at 11:45:30 during a rolling deploy. **[inferred]** the job was cut off by the deploy.

## 3. Part A: per-stage timings

### 3.1 Stage table

| # | Stage | Start | End | Duration | Basis |
|---|---|---|---|---|---|
| 1 | Job start, LLM client created | 11:48:00.24 | | | `llm.factory.selected` |
| 2 | Open PDF, render 76 pages at 150 DPI, DB lookups | 11:48:00.24 | 11:48:09.5 | about 9.3 s | derived. Rendering is not logged on its own. |
| 3 | Layer-1 inspection pass for the probe gate | 11:48:09.545 | 11:48:09.589 | 44 ms for 76 pages | 76 `page.inspected` events |
| 4 | Probe (`probe.complete_images`) | 11:48:09.59 | 11:48:34.92 | 25.33 s (logged `elapsed_s`) | probe log |
| 5 | Layer-1 routing and Layer-2 OCR, 25 workers | 11:48:34.93 | 11:51:54.31 | 199.4 s | `pipeline.concurrent_pages_start` to `pipeline.document_complete` |
| 5a | of which: page 64 VLM escalation | 11:51:16.57 | 11:51:51.87 | 35.3 s | `pipeline.vlm_transcribe_invoked`, `pipeline.escalation_attempt` |
| 6 | Graph build, page workers and LLM calls | 11:51:54.98 | 12:05:57.3 | 842 s (14.0 min) | `layer3.graph_memory_concurrent.start` to last graph call |
| 7 | Serial merge of page deltas | 12:05:57.3 | 12:06:06.6 | 9.3 s | `entity_resolution.*` events |
| 8 | Cross-page link (no model call) | 12:06:06.6 | 12:06:06.6 | under 0.1 s | `graph.crosspage_link.done` (3 links) |
| 9 | Schema resolution (1 LLM call) | 12:06:06.6 | 12:06:13.8 | 7.15 s | `llm.converse.call`, purpose `schema_resolution` |
| 10 | Persist graph, extraction run, PDF | 12:06:13.8 | 12:06:14.65 | about 0.9 s | `storage.*` events |
| | **Total** | 11:48:00.24 | 12:06:14.65 | **1,094 s** | |

The stages add up to 1,093 s, within 1 s of the job wall clock.

### 3.2 Layer 1: PyMuPDF inspection and probe

- **Inspection:** the run logs 152 `page.inspected` events, not 76. Two passes ran. The first pass, 76 events in 44 ms, is `inspect_page` called from `_run_document_probe` (`src/ai/layer1_routing/pipeline.py:673`). That pass gates which pages the probe checks. The second pass, 76 events spread over 129 s, runs inside each page worker in `process_page`. Per-page PyMuPDF cost is not available. The first-pass events are emitted in one burst.
- **Rendering:** `_build_pages_for_pdf` renders every page at 150 DPI into an RGB array and PNG bytes before `process_document` starts (`schema_chatbot_v2/app/api/pipeline_routes.py:114`, `:136`, `:142`). Its time is inside the 9.3 s in stage 2 and is not logged on its own.
- **Probe, document level:** 76 candidate pages, 0 blank, 0 timed out (`probe.complete_images`). `probe_document_images` (`src/ai/layer1_probe/probe.py:378`) loops over pages serially for ink, table, and JPEG work. It then samples 3 pages (`types.py:78`, `n_samples=3`).
- **Probe, samples:** each sample ran one lipika ONNX observation (`onnx_calls: 3`) and one Bedrock VLM call (`vlm_calls: 3`). The sample calls run in one thread wave (`probe.py:173-175`).
- **Lipika result:** lipika returned no script for all 3 samples (`lipika_resolved: 0`, `lipika_script: null`, `n_windows: 32`). The VLM decided every sample. Document script: `latin`, confidence 0.99, heterogeneous false.
- **Real lipika detections:** 0 pages. Pages that received a sampled answer: 3. The other 73 pages inherit the document prior **[inferred]** from `probe.py:197-211` (`script_source = "prior"`). The `probe.page_decision` event does not log `script_source`.
- **Probe decisions:** all 76 pages routed to `paddleocr_printed`. Script `latin` on all 76. `has_table` true on 29 pages.
- **Per-page probe cost:** not logged. The 25.3 s covers serial CV and JPEG encoding for 76 pages, 3 ONNX runs, and 3 VLM calls. The split between these is not available.

### 3.3 Layer 2: OCR and conversion

- **Routing:** `probe_route` and `final_engine` are `paddleocr_printed` on all 76 pages. `engines_used` is `["paddleocr_printed"]` on 75 pages.
- **Escalation:** page 64 had OCR confidence 0.538, below the 0.70 threshold (`router.escalation_decision`). It was sent to `vlm_transcribe` and returned confidence 0.92. That VLM call took 35.3 s and is not in the `llm.converse.call` metrics.
- **Handwriting:** every page plan listed `paddleocr_handwritten`, but no page ran it. No log group exists for `/aws/lambda/idp-engine-paddle-handwritten`, so this Lambda has no recorded invocations. I did not trace why the handwritten engine was skipped.
- **Per-page OCR time (Lambda handler):** `layer2.printed.lambda_converted.latency_ms`, n=76. Minimum 22.4 s, median 48.5 s, p90 73.1 s, maximum 91.4 s. The timer starts before the PaddleOCR engine is built (`docker/paddle_printed/engine.py:93-95`), so cold-container build time is included.
- **Per-page pipeline time:** `pipeline.page_complete.total_latency_ms`, n=76. Median 50.0 s, p90 75.0 s, maximum 94.2 s. The gap to the Lambda timer is about 2.4 to 2.8 s per page. That covers invoke and payload transfer, which the logs do not split out.
- **Single-page time in isolation:** not available. The first page finished 25 s after the first wave started (page 8, Lambda 22.4 s).

**Concurrency (answer to "are pages concurrent, and how many?"):**

- Layer-1 inspection and the probe CV loop are sequential.
- Layer 2 is concurrent: `process_document` uses `ThreadPoolExecutor(max_workers=min(76, 25))` (`src/ai/layer1_routing/pipeline.py:796-817`). The peak was 25 overlapping Lambda invocations. The peak was computed from timestamps minus `latency_ms`.
- Each worker calls `lambda.invoke` synchronously with `RequestResponse` (`src/ai/layer2_conversion/scanned.py:114-118`). The client uses `read_timeout=300` and `retries.max_attempts=0` (`scanned.py:47-52`).

**Merge (answer to "how are results combined?"):**

- `executor.map` returns results in input order, so the list is page 1 to 76 regardless of finish order (`pipeline.py:817-820`).
- `process_document` counts multi-engine, low-confidence, and escalated pages. It then writes one markdown file through `write_document` (`src/ai/output/md_writer.py:45`): 76 pages, 130,724 characters.
- `_run_pipeline_job` builds `pages_for_layer3 = [{"markdown", "page_number"}]` in page order (`pipeline_routes.py`, about line 770 to 790). That list feeds Layer 3. The IR store separately saves 6,112 blocks.

### 3.4 Graph build (Layer 3)

The detailed numbers and the code path are in section 4. The graph build starts at 11:51:54.98, when 16 page tasks are admitted, and ends at 12:06:06.6, after the merge.

## 4. Graph-build deep dive

### 4.1 LLM calls and latency

| Metric | Value | Source |
|---|---|---|
| `llm.converse.call` records in job | 78: 77 `graph_extraction` and 1 `schema_resolution` | `llm_metrics` |
| Calls per page | 1 for 75 pages. Page 18 had 2: the first hit the 8,000-token output limit, and the retry at 12,000 tokens timed out. | `stop_reason`, `max_tokens` |
| Model | Application inference profile `24n3l11zddro` (`IDP_prototype`, Claude Sonnet 4.5), region `ap-south-1` | `aws bedrock get-inference-profile` |
| Successful page calls (n=70) latency | min 7.4 s, p50 36.1 s, p90 51.3 s, max 182.8 s | `llm.converse.call` |
| Failed page calls (n=7) latency | 304.1 to 311.6 s each, median 308.5 s, sum 2,157 s | `stop_reason = ReadTimeoutError` |
| Successful input tokens | p50 2,608 (min 2,057, max 3,407). Total 185,241. | `input_tokens` |
| Successful output tokens | p50 4,546 (min 738, max 8,000). Total 327,424. | `output_tokens` |
| Output throughput | p50 126.5 tokens/s (p10 116, p90 131) **[inferred]** from output tokens divided by latency | derived |
| Schema resolution call | 7.15 s, 161,575 input tokens, 167 output tokens | `llm.converse.call` |

Three probe VLM calls and one escalation VLM call are not in `llm.converse.call`. The probe uses its own boto3 client (`src/ai/layer1_probe/vlm_probe.py:72-73`). Their latency is only available from log gaps.

### 4.2 Pages and failures

- **Quality:** 69 pages clean, 0 degraded, 0 partial (`llm.page.quality`). Degraded and partial handling never ran.
- **Failed pages:** 12, 13, 14, 15, 16, 17, 18. Each failed with botocore `ReadTimeoutError`. The worker catches the exception, logs `layer3.concurrent.worker_failed`, and writes a `FAILED` checkpoint (`src/ai/layer3_extraction/extractor.py:462-466`). There was no page-level retry.
- **Graph coverage:** `pages_covered` in the final graph lists 1 to 11 and 19 to 76. The graph is missing pages 12 to 18.

**Why each failed call takes about 300 s [inferred]:**

- botocore's default `read_timeout` is 60 s (`botocore/config.py`, `DEFAULT_TIMEOUT`). The Bedrock client sets no `Config` (`src/adapters/llm/bedrock_client.py:170-181`).
- In this botocore version, the default retry mode is `legacy`, with `max_attempts` 5 (`botocore/data/_retry.json`). `ReadTimeoutError` is a retryable connection error (`botocore/retryhandler.py`, `GENERAL_CONNECTION_ERROR`).
- 5 attempts times 60 s gives about 300 s, which matches the observed 304 to 312 s. I did not trace the retries directly.
- The app's own retry loop (`bedrock_client.py:228-260`) only retries names in `_THROTTLE_MARKERS`, which do not include timeouts. So the app does not add retries on top of botocore's.

### 4.3 Concurrency: configured 16, effective about 6

- **Config:** `GRAPH_CONCURRENCY_LIMIT=16` in the live task definition. The log line `layer3.graph_memory_concurrent.start` shows `concurrency_limit=16`.
- **Admission:** `asyncio.Semaphore(16)` (`extractor.py:426`). Pages 1 to 16 are admitted at 11:51:55.0. The semaphore is released when a page's call returns.
- **Execution:** `await asyncio.to_thread(self.llm.extract_graph_from_page, ...)` (`src/ai/layer3_extraction/graph_agent/agent.py:102`). `to_thread` runs on the loop's default `ThreadPoolExecutor`. The repo does not set a custom executor. Python 3.11 sizes the default pool to `min(32, os.cpu_count() + 4)`.
- **Observed:** no more than 6 Converse calls overlap at any moment. Calls 1 to 6 start at 11:51:55. Each later call starts when an earlier call on a thread finishes, as the start times show:
  - page 7 starts at 11:52:24.2, when page 2 finishes at 11:52:24.1
  - page 8 starts at 11:52:25.9, when page 5 finishes at 11:52:25.9
  - page 19 starts at 11:57:57.5, when page 13's timed-out call finishes at 11:57:57.4
- **Inference:** the 6 threads come from `os.cpu_count() + 4` with `cpu_count = 2`. This matches the 2 vCPU task size, but I did not check the CPU count inside the container.
- **Effect:** each timed-out call holds one thread for about 5 minutes, and the pages queued behind it wait. The 7 timeouts used 2,157 s of thread time. The sum of all 77 call latencies is 4,940 s. Divided by 6 threads, that is 823 s, against 842 s observed for the build stage. The lane model explains the build time.

### 4.4 How extracted text becomes nodes and relationships

1. **Entry point.** `_run_pipeline_job` (`schema_chatbot_v2/app/api/pipeline_routes.py:599`, started in a background thread at line 569) calls `process_document`, then calls `extract_document` with the page markdown list and the schema (about line 823). `extract_document` routes by strategy to `extract_with_graph_memory_concurrent` (`src/ai/layer3_extraction/extractor.py:766`, `:336`).
2. **Context per page.** `GraphMemoryManager.get_context_for_page` (`src/ai/layer3_extraction/graph_agent/memory_manager.py:865`) returns anchor nodes of types `Patient` and `Claim`, up to 5 anchors. These go into the prompt as existing graph context, capped at 35 nodes (`bedrock_client.py`).
3. **One LLM call per page.** The `graph_page_ingestion` prompt (`src/ai/layer3_extraction/prompts/templates/graph_page_ingestion.j2`; `prompt_config.yaml:21-25`, temperature 0) asks for one JSON object with three lists:
   - `entities`: `id`, `reused_node_id`, `type`, `label`, `value`, `evidence` (verbatim sentence), `confidence`, `is_schema_field`, `schema_field_name`
   - `relationships`: `source_node`, `target_node`, `relationship`, `evidence`, `status`
   - `reference_resolutions`: anaphora such as "the said applicant" resolved to an existing node id
4. **Parse.** `_parse_graph_json` (`bedrock_client.py:92`) parses the response directly, or falls back to the outer `{...}` substring, which marks the page partial. If the output hits `max_tokens`, the call is retried once at a higher ceiling (`bedrock_client.py:475-485`). If the JSON is invalid, a single re-ask runs (`graph_reask`). Neither the partial path nor re-ask ran in this job.
5. **Page delta.** `build_page_delta_async` (`graph_agent/agent.py:76`) builds the page's delta:
   - Each entity becomes a `GraphNode`. Category is `EXPLICIT` if it is a schema field, otherwise `CONTEXTUAL`. Evidence is linked to PDF blocks with `evidence_resolver`.
   - Each relationship becomes a `GraphEdge`, with the relationship type upper-cased.
   - Each reference resolution becomes a `REFERS_TO` edge (confidence 0.9, status `INFERRED`) when the target resolves, or an `UnresolvedRef` otherwise (`agent.py`, about lines 150 to 205).
   - The delta is not merged into the graph here. Workers only build deltas.
6. **Serial merge.** After all workers finish, the deltas are merged in page-number order (`extractor.py`, after the `asyncio.gather` near line 502). `ingest_page_delta` (`memory_manager.py:913`) runs `resolve_entity` (`memory_manager.py:657`):
   - candidate lookup
   - hard contradiction gate, for example `INCOMPATIBLE_TYPES`
   - score each candidate
   - margin test
   - Thresholds: match 0.85, ambiguity margin 0.15 (`memory_manager.py:165-166`).
   - Outcomes (`graph_agent/models.py:15-21`): `CONFIRMED_MATCH` fuses into the existing node (adds page, evidence, alias, observation), `NEW_ENTITY` creates a node, `UNCERTAIN_MATCH` is kept separate, and `REJECTED_CONTRADICTION` blocks the merge. Edges are rewritten to canonical ids. Every merge writes an audit record.
   - Counts for this run: `contradiction` 1,797 (candidate checks), `confirmed` 642, `new_entity` 591, `uncertain` 366.
7. **Checkpoints.** Each page writes `DELTA_PERSISTED` before the merge and `COMPLETED` after (`extractor.py`, merge loop).
8. **Cross-page linking.** `link_cross_page` (`memory_manager.py:505`) adds `SAME_ENTITY_AS` edges between same-type nodes on different pages that share a normalized identifier or name. It makes no model call. This run created 3 links and skipped 289 amount/date and 59 excluded-value nodes.
9. **Schema resolution.** `resolve_schema_from_graph` (`graph_agent/resolver.py:86`) sends 2,338 lines of graph evidence to one LLM call (`graph_schema_resolution.j2`) to fill 7 schema fields. The result: patient name "MRS CHANDER KOCHHAR", patient id 192673, admission 21/02/2025, discharge 24/02/2025, procedure "right bipolar hemiarthroplasty", approved claim amount 80698, and a primary diagnosis.
10. **Persist.** `save_document_graph` (`src/ai/layer3_extraction/storage.py:601`) stores the graph snapshot: 957 nodes, 1,357 edges.

**Graph contents:**

- Node types: Identifier 134, LineItem 113, Date 119, Amount 79, Status 74, Quantity 66, Document 60, Organization 51, Person 47, Event 44, Location 43, Entity 55, Attribute 27, Value 25, Finding 15, Diagnosis 5.
- Top relationships: HAS_IDENTIFIER 138, HAS_DATE 93, PART_OF 74, HAS_STATUS 56, LOCATED_IN 58, HAS_ATTRIBUTE 66, SIGNED_BY 33, CONTAINS 32, REFERRED_BY 28, HAS_RESULT 36.

Extraction accuracy was not evaluated in this investigation.

## 5. Part B: deployment topology

### 5.1 Topology

| Layer or section | Where it runs | Image or artifact | How it is invoked | Memory, timeout, CPU |
|---|---|---|---|---|
| Web console, Layer-1 inspection, probe, Layer-2 orchestration, Layer-3 graph build, query bot | ECS Fargate, `idp-cluster`, service `idp-layer3-service`, container `idp-web-console` | ECR `idp-layer3-app:probe-202610251725-acd90d1-dirty` (digest `sha256:d0e6...`). Built from `schema_chatbot_v2/Dockerfile` by `scripts/deploy_idp_ecr_images.py:37`. ECR reports 5.45 GB. | ALB `idp-web-alb` (internet-facing) to target group `idp-web-tg`, port 8000, path `/health`. Entrypoint `uvicorn app.main:app`. | 2 vCPU, 8,192 MB (live). Health check every 15 s, healthy threshold 2 (`deploy_idp_fargate.py:150-153`). |
| Layer-1 inspection | In process, `src/ai/layer1_routing/inspect.py` | Same image | Called by `_run_document_probe` and per-page workers | Same task |
| Probe (script, handwriting, table) | In process, `src/ai/layer1_probe/probe.py`. Lipika ONNX (`lipika_script.onnx`, 111.6 MB) is in the image because the Dockerfile runs `COPY . /app`. | Same image | Enabled by `IDP_PROBE_ENABLED=true`. VLM calls go to Bedrock. | Same task |
| OCR, printed | Lambda `idp-engine-paddle-printed` | ECR `idp-layer2-paddle-printed:latest`, about 906 MB (ECR reports 905,662,185 bytes). PackageType Image, x86_64. PaddleOCR PP-OCRv6 weights are baked in at build time. | `lambda.invoke` with `RequestResponse` from Fargate (`scanned.py:114-118`). Client `read_timeout=300`, retries 0. | 10,240 MB, timeout 900 s, ephemeral 4,096 MB. Reserved concurrency: none. Provisioned concurrency: none. |
| OCR, handwritten | Lambda `idp-engine-paddle-handwritten` | ECR `idp-layer2-paddle-handwritten:latest`, about 906 MB | Same invoke path, route `paddleocr_handwritten`. Not invoked in this run. | 3,008 MB, timeout 300 s. No reserved or provisioned concurrency. |
| Digital conversion (Docling) | Lambda `idp-engine-docling` | ECR `idp-layer2-docling:latest`, about 842 MB | Not used in this run (no digital pages) | 3,008 MB, timeout 300 s. No reserved or provisioned concurrency. |
| Layer-1 router | Lambda `idp-layer1-router`, ZIP, python3.11 | None | Not invoked by any app code. The Fargate task role is allowed to invoke it (`deploy_idp_fargate.py:96`). Step Functions definitions in `step_functions/` reference other Lambdas, not this one. | 1,024 MB, timeout 120 s |
| LLM (graph extraction, schema resolution) and VLM (escalation, probe) | Amazon Bedrock, application inference profile `24n3l11zddro` (Claude Sonnet 4.5), `ap-south-1` | Managed | boto3 `converse` from Fargate (`bedrock_client.py:182-260`; probe `vlm_probe.py:72-76`) | Managed |
| Storage | Postgres (`DATABASE_URL`), S3 bucket `idp-internal-35385`, markdown and IR files on container disk | | | |

All 4 Lambdas and the ECS task definition, service, and cluster carry the three AGENTS.md tags (`createdby`, `purpose`, `customer`). I checked this with `list-tags`.

### 5.2 Notes on the topology

- **Only the Fargate task runs the pipeline.** Layer 1 and the probe run in process. The OCR Lambdas and Bedrock are the only remote calls on the job path.
- **Drift between scripts and live:**
  - `deploy_idp_fargate.py:228` sets task memory to `"4096"`. The live revision has `8192`.
  - `deploy_idp_lambdas.py:169-170` creates the Layer-2 engines with 3,008 MB and 300 s. The live printed engine has 10,240 MB and 900 s. The repo's `deploy_paddle_lambdas.py` only updates the image. The live values were set outside the repo **[inferred]**.
- **Secrets in plain text.** The task definition env holds `JWT_SECRET`, `ADMIN_PASSWORD`, `DATABASE_URL`, and `IDP_DATABASE_URL` as plain environment values. They are masked in this report. Moving them to Secrets Manager or SSM would remove them from the task definition.
- **Step Functions.** `step_functions/state_machine_with_probe.asl.json` references a probe Lambda (`PROBE_LAMBDA_ARN`). No Lambda with `probe` in its name exists in `ap-south-1`. That state machine is not on the live path.

## 6. Cold start, tier by tier

### 6.1 OCR Lambdas (container images)

**Evidence from the run (`REPORT` lines in `/aws/lambda/idp-engine-paddle-printed`, 06:10 to 06:45 UTC):**

- 76 invocations. 25 of them carry `Init Duration`, which means a cold start. Init durations: min 0.66 s, median 0.96 s, max 1.40 s. The 25 cold starts equal the 25 concurrent workers, so the first wave created all 25 containers. The other 51 invocations reused warm containers **[inferred]**.
- Cold invocations: median billed duration 53.8 s. Warm invocations: median 47.3 s. The difference is about 6 s, but the spread is wide, so treat it as a rough figure.
- Max memory used: median 2,084 MB (range 1,360 to 2,420 MB).

**What the code and config do:**

- The Lambda Init phase is short (under 1.5 s), so module imports are not the main cold cost. PaddleOCR is built lazily on the first `process_image` call (`docker/paddle_printed/engine.py:37-56`). The engine is cached in the module-level `_paddle_engines` dict. The timer starts before that build (`engine.py:93-95`), so the build cost is inside `latency_ms`. Warm containers skip it.
- The Dockerfile bakes the PaddleOCR weights into the image at build time (`docker/paddle_printed/Dockerfile`, the `RUN python -c "from paddleocr import PaddleOCR ..."` step). That removes the network download at cold start.
- **Not configured:** provisioned concurrency (`list-provisioned-concurrency-configs` returns an empty list), reserved concurrency (`get-function-concurrency` returns no value), warmer or ping invocations (none found in `src/`, `scripts/`, or `docker/`).

**Effect:** the first wave of 25 pages pays the cold start. Later pages reuse warm containers.

### 6.2 Docling and handwritten OCR Lambdas

No invocations in this run, and no `/aws/lambda` log group exists for either function. Cold-start behavior is not measured. The configuration matches the printed engine: no provisioned or reserved concurrency and no warm-up.

### 6.3 Fargate task (the app)

- **No per-request cold start.** The service keeps 1 task running (desired 1, running 1).
- **Task launch cost:** the replacement task was created at 11:42:14.558 and started at 11:45:01.853, which is 2 min 47 s. That includes the pull of a 5.45 GB image and Fargate provisioning. The logs do not split out the pull time.
- **Rolling deploy:** the service started the new task at 11:42:14, registered it with the target group at 11:44:59, stopped the old task at 11:45:30, and reported the deployment complete at 11:46:33. Minimum healthy percent is 100 and maximum is 200. The circuit breaker is disabled.
- **Lazy model loads inside the app:**
  - The Q&A embedding model (`all-MiniLM-L6-v2`) loads on first use per process (`evidence_retriever.py:48-50`). The Dockerfile comment says the weights are not baked in and are downloaded from Hugging Face on the first query. See section 10 for the timing in your query log.
  - The lipika ONNX session is created lazily on first use (`script_classifier.py`, singleton `_SESSION`). Its load time is not logged.
- **Nothing pre-warms these.** Startup takes place only when a task starts.

### 6.4 Bedrock

- **No cold start in the service sense.** The boto3 client is created on first use (`bedrock_client.py:170-181`). The cost is negligible next to the calls.
- **The real problem is latency and timeouts.** A page call takes about 36 s at p50, and a dead call takes about 300 s to fail (section 4.2). No mitigation is configured beyond botocore defaults. There is no explicit `Config`, no streaming, and no page-level retry.

## 7. Bottlenecks observed

1. **Graph build: 842 s, 77% of the job.** Each call generates about 4,500 output tokens at about 126 tokens/s, which is about 36 s per call. Output length is the main driver.
2. **Effective graph concurrency is about 6, not 16.** The default thread pool caps the Bedrock calls. Config says 16, so the build runs about 2.7 times slower than the setting implies **[inferred]**.
3. **7 pages lost to Bedrock read timeouts.** Each timeout held a thread for about 300 s and blocked the pages queued behind it. The lost pages are missing from the graph. This is about 9% of the document.
4. **OCR at 25-way concurrency:** median 48.5 s per page in the Lambda, plus about 2.8 s of invoke overhead. 25 containers cold-started in the first wave.
5. **Probe:** 25.3 s. Lipika gave no script on any of the 3 samples, so each sample fell back to the VLM.
6. **Earlier failed run:** every Bedrock call failed on `AccessDeniedException`. The job still ran but produced no graph. This is a configuration failure, not a performance one.
7. **Single schema-resolution call with 161,575 input tokens:** 7.15 s. Fine at this size, but it grows with the graph.

## 8. Recommendations (not implemented)

1. **Make graph concurrency match the setting.** Pass a `ThreadPoolExecutor` sized to `graph_concurrency_limit` to `to_thread`, or call `loop.set_default_executor`. Check Bedrock throughput quotas for the inference profile before raising the limit. At 16 lanes, the lane model gives about 310 s instead of 823 s, before throttling.
2. **Bound failed calls and retry the page.** Set an explicit botocore `Config` on the Bedrock client. Keep `read_timeout` above the slowest good call, which was 183 s. Use 2 attempts instead of 5. The bigger gain is a page-level retry for `FAILED` pages, which the existing checkpoints already support. Trade-off: a shorter timeout risks cutting off legitimate long pages.
3. **Cut output tokens per page.** Each 1,000 fewer output tokens saves about 8 s per call at 126 tokens/s. Options: limit entity count, drop optional fields such as long evidence strings, or split dense pages.
4. **OCR cold starts.** Choose one: provisioned concurrency of 25 on the printed Lambda (always-on cost for 25 × 10 GB), a scheduled warm-up invoke before each job, or accept the first-wave cost. Do not move PaddleOCR construction to module import without measuring: the Lambda init phase has a time limit.
5. **Pin the thread count explicitly.** Use an env var for the executor size rather than relying on `os.cpu_count()`. This makes the behavior the same on every host.
6. **Sync the deploy scripts with live config.** Fix the memory value in `deploy_idp_fargate.py:228` (4096 vs 8192 live), and record the printed Lambda's 10,240 MB and 900 s in the repo.
7. **Move secrets to Secrets Manager or SSM.** Remove the plain-text values from the task definition.

## 9. Differences from the brief

- `_run_pipeline_job` is in `schema_chatbot_v2/app/api/pipeline_routes.py:599`, not in `src/ai/layer1_routing/pipeline.py`. The brief's path for `process_document` is correct.
- The run has 152 `page.inspected` events, not 76. Two passes run, as section 3.2 explains.
- 75 pages used only the printed engine. Page 64 also used the VLM. The handwritten engine never ran.
- The probe's lipika ONNX runs in process in Fargate, not in a Lambda. The image includes it.
- The Layer-1 router is not on the live path. Confirmed.
- The task memory is 8,192 MB live. The deploy script says 4,096.

## 10. Earlier questions: the query log and the retrieval method

**Your query log, step by step:**

1. `graph.query.started`, question "Name of the patient ?".
2. `evidence_retriever.model_loaded`, 16 s after the question. This is the first query in the process, so the sentence-transformer model loads now.
3. `evidence_retriever.index_built`, 9 s later. The index holds 1,589 evidence spans, each embedded as a 384-dimension vector.
4. `graph.query.answer_generated`, 3 s later. The answer is "KOCHHAR CHANDER (also referred to as MRS CHANDER KOCHHAR)", built from 8 sources.
5. `graph.query.grounding_checked`: 1 of 1 claims grounded, 0 unverified.

Total: about 28 s, of which about 25 s is first-query setup. Later queries on the same graph should be faster **[inferred]**, because the index is cached per graph (`get_index_for_graph`).

**Retrieval method:** it is a hybrid, not plain RAG and not plain BFS.

- **Seeds:** keyword seed nodes (`find_seed_nodes`, `query_service.py:125`) and vector seeds from the embedded evidence spans (`find_vector_seed_nodes`, line 349). The two sets are merged with keyword seeds first and capped at 10 (`_merge_seed_sets`, line 387; `DEFAULT_MAX_SEEDS=10`).
- **Traversal:** bounded, cycle-safe breadth-first search from the seeds, up to 3 hops, 25 nodes, and 30 edges (`query_service.py:24-27`). I read the constants and the module docstring, not the full traversal body.
- **Answer and check:** the LLM answers from the retrieved nodes and edges. `grounding_checker.check_grounding` then checks each claim against the retrieved evidence.

## 11. Method and files

- **Log pull:** `aws logs filter-log-events` on `/ecs/idp-layer3-app` from 1792982266051 ms, which is 4 hours before the newest stream event. That gave 11,918 events from 5 streams. Lambda REPORT lines came from `/aws/lambda/idp-engine-paddle-printed`, 06:10 to 06:45 UTC.
- **Raw exports:** `%TEMP%\l3_events.json` (UTF-16, contains document content), `%TEMP%\td6.json`, `%TEMP%\svc.json`, `%TEMP%\rep_idp-engine-paddle-printed.json`, plus the helper scripts `analyze_l3.py`, `jload.py`, and `awsenv.ps1`. The exports contain patient data from the document. Delete them when no longer needed.
- **Code read:** `src/ai/layer1_routing/pipeline.py`, `src/ai/layer1_probe/probe.py`, `src/ai/layer1_probe/types.py`, `src/ai/layer2_conversion/scanned.py`, `src/ai/layer3_extraction/extractor.py`, `src/ai/layer3_extraction/graph_agent/{agent,memory_manager,models,resolver,query_service}.py`, `src/adapters/llm/bedrock_client.py`, `src/ai/output/md_writer.py`, `schema_chatbot_v2/app/api/pipeline_routes.py`, `docker/paddle_printed/{handler,engine}.py`, `schema_chatbot_v2/Dockerfile`, `scripts/deploy_idp_fargate.py`, `scripts/deploy_idp_ecr_images.py`, `scripts/deploy_idp_lambdas.py`, `scripts/deploy_paddle_lambdas.py`, `.venv` botocore 1.43.80.
- **AWS calls (all read-only):** `sts get-caller-identity`, `logs describe-log-streams`, `logs filter-log-events`, `ecs describe-task-definition`, `ecs describe-services`, `ecs describe-tasks`, `ecs list-tags-for-resource`, `lambda list-functions`, `lambda get-function-configuration`, `lambda get-function`, `lambda get-function-concurrency`, `lambda list-provisioned-concurrency-configs`, `lambda list-tags`, `ecr describe-repositories`, `ecr describe-images`, `bedrock get-inference-profile`, `elbv2 describe-load-balancers`.
- **Not checked:** the IAM policy for `idp-ecs-task-role` (to see whether the 10:48 AccessDenied was fixed by a policy change), the container CPU count, Bedrock quotas, and whether the 300 s timeout comes from botocore's retries (it matches but was not traced).
