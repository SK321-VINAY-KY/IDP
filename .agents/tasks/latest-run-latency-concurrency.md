# Latest run: latency and concurrency per layer (job_988e2a763956, task definition :7)

Read-only investigation. No code, infrastructure, or AWS resource was changed. AWS access was limited to describe, list, get, and CloudWatch Logs reads. Credentials were loaded from `.env2` into the process environment and never printed. All times are IST (+05:30), as the app logged them.

## 1. Summary

- **Run on the post-fix build: yes.** The newest completed job is `job_988e2a763956` on `idp-layer3-task:7`, which carries both fixes. It is the only job on this log stream.
- **Wall clock fell from 1,094.4 s to 492.3 s (55% less).** The baseline is `job_4cf06e33305d` on `:6`. Both runs process the same document, "Chander Kochhar 01_compressed 2.pdf", 76 pages.
- **Fix 1 (graph thread pool) works.** Graph LLM calls peaked at 16 in flight, against 6 in the baseline. The page-call phase fell from 842.3 s to 234.7 s.
- **Fix 2 (explicit Bedrock timeouts and bounded retries) was not exercised.** The run had zero read timeouts. The longest Bedrock call was 95.9 s, well under the 240 s read timeout.
- **Five pages were lost to output truncation, not timeouts.** Pages 12, 14, 15, 16, and 17 hit the 8,000-token output cap. The 12,000-token retry truncated again. The app logged them as `degraded` and wrote no error. The graph now covers 71 pages (baseline 69) with 1,028 nodes and 1,406 edges (baseline 957 and 1,357).
- **Remaining cost is now spread across OCR and graph calls.** Layer-2 OCR took 202.2 s (41%). Graph page calls took 234.7 s (48%). The five truncated pages used 780.8 s of Bedrock call time, or 22% of all graph call time.

## 2. Run identity

| Item | Value | Source |
|---|---|---|
| Job id | `job_988e2a763956` | `storage.document_graph_saved`, `storage.job_pdf_saved` |
| Document | `Chander Kochhar 01_compressed 2.pdf`, 76 pages | `pipeline.concurrent_pages_start` (`page_count` 76) |
| Task | `654a5bc60bd94b269a02ebd255943005` (service `idp-layer3-service`, cluster `idp-cluster`) | `aws ecs describe-tasks` |
| Task definition | `idp-layer3-task:7`, registered 16:37:51, image `idp-layer3-app:probe-202610261024-acd90d1-dirty` | `aws ecs describe-task-definition --task-definition idp-layer3-task:7` |
| Task timing | created 16:38:12, started 16:41:05 | `aws ecs describe-tasks` |
| Log stream | `web/idp-web-console/654a5bc60bd94b269a02ebd255943005` in `/ecs/idp-layer3-app` | `aws logs` |
| Job start to end | 16:59:23.022 to 17:07:35.274, 492.25 s | `llm.factory.selected` to `storage.job_pdf_saved` |

The brief said the `:7` task started around 16:30 IST. ECS shows the task started at 16:41:05, so the job ran wholly on `:7`. The job started at 16:59:23, which is after the task started.

Comparison: the baseline `:6` run is `job_4cf06e33305d` (11:48:00 to 12:06:14 IST), from the baseline log stream `69bf62d5026c4a9f8e3e21db25faaf06`.

Like for like: same file, same page count, same escalated page (64), same probe outcome (`lipika_resolved` 0 on both). Each build has one run, so this comparison has no variance estimate.

## 3. Per-layer latency

| Stage | :7 time window (IST) | :7 duration | :7 share of 492.25 s | :6 duration | :6 share of 1,094.4 s |
|---|---|---|---|---|---|
| Open PDF, render pages, DB work | 16:59:23.02 to 16:59:32.27 | 9.25 s | 1.9% | 9.30 s | 0.9% |
| Layer-1 inspection, pass 1 | 16:59:32.27 to 16:59:32.31 | 0.04 s | 0.0% | 0.04 s | 0.0% |
| Probe (`probe.complete_images` `elapsed_s`) | 16:59:32.3 to 16:59:58.96 | 26.65 s | 5.4% | 25.33 s | 2.3% |
| Layer-2 OCR (`pipeline.concurrent_pages_start` to `pipeline.document_complete`) | 16:59:58.97 to 17:03:21.21 | 202.2 s | 41.1% | 199.4 s | 18.2% |
| Markdown and IR save | 17:03:21.21 to 17:03:21.85 | 0.64 s | 0.1% | 0.67 s | 0.1% |
| Layer-3 graph page calls (first call start to last call end) | 17:03:21.86 to 17:07:16.58 | 234.7 s | 47.7% | 842.3 s | 77.0% |
| Layer-3 serial merge (`entity_resolution.*`) | 17:07:16.58 to 17:07:27.62 | 11.0 s | 2.2% | 9.3 s | 0.8% |
| Cross-page link (no model call) | 17:07:27.63 | 0.01 s | 0.0% | 0.01 s | 0.0% |
| Schema resolution (1 LLM call) | 17:07:27.64 to 17:07:34.34 | 6.7 s | 1.4% | 7.2 s | 0.7% |
| Persist graph, extraction, and PDF | 17:07:34.34 to 17:07:35.27 | 0.94 s | 0.2% | 0.87 s | 0.1% |
| **End to end** | 16:59:23.02 to 17:07:35.27 | **492.25 s** | 100% | **1,094.4 s** | 100% |

Notes:

- The Graph build stage as a whole (`layer3.graph_memory_concurrent.start` to `.done`) took 252.5 s in this run and 858.8 s in the baseline.
- The stages add up to the wall clock within 0.2 s.

### 3.1 Layer 1: PyMuPDF inspection and probe

- `page.inspected` was logged 152 times, in two passes:
  - Pass 1: 76 events in 44 ms (16:59:32.267 to 16:59:32.311). This gates the probe. It is sequential.
  - Pass 2: 76 events from inside the OCR workers, spread over 121.6 s (16:59:58.97 to 17:02:00.54). It runs concurrently with OCR.
- Probe (`probe.complete_images`): 76 candidates, 3 VLM calls, 3 ONNX calls, `lipika_resolved` 0, document script `latin` at confidence 0.99, `timed_out` 0, `elapsed_s` 26.65.
- The probe runs before OCR starts, and nothing overlaps with it.
- Page rendering is not logged on its own. Its time is inside the 9.25 s setup window.

### 3.2 Layer 2: OCR

- Engines: `paddleocr_printed` on all 76 pages. One page (64) escalated to `vlm_transcribe` because OCR confidence was 0.54 against a 0.70 threshold. Confidence after escalation was 0.92.
- Escalation timing: `pipeline.vlm_transcribe_invoked` at 17:02:41.49, `pipeline.escalation_attempt` at 17:03:12.85, so 31.4 s. The baseline took 35.3 s. The escalation finished before the last OCR page, so it was not on the critical path.
- Per-page PaddleOCR Lambda time (`layer2.printed.lambda_converted` `latency_ms`, n=76):
  - Min 16.4 s, median 48.9 s, p90 67.7 s, max 83.7 s.
  - Baseline: median 48.5 s, p90 73.5 s, max 91.4 s.
- Per-page pipeline time (`pipeline.page_complete` `total_latency_ms`, n=76):
  - Min 19.1 s, median 49.3 s, p90 70.7 s, max 87.7 s.
- Cold versus warm (from Lambda `REPORT` lines in `/aws/lambda/idp-engine-paddle-printed`, 76 invocations):
  - 25 cold starts (`Init Duration` present), init 0.75 to 1.28 s each (median 1.11 s).
  - Cold median `Duration` 48.1 s, warm median 48.9 s. The cold start is not visible in per-page latency.

### 3.3 Layer 3: graph build

LLM calls (`llm.converse.call` in `src/adapters/llm/llm_metrics`):

- 83 calls in total: 82 `graph_extraction` and 1 `schema_resolution`.
- The 82 graph calls cover 76 pages. 70 pages took one call. Six pages took two calls:
  - Pages 12, 14, 15, 16, 17: first call hit the 8,000-token cap, then the retry at 12,000 tokens also hit the cap. Five pages degraded.
  - Page 18: first call hit the 8,000-token cap, then the 12,000-token retry completed with 8,061 output tokens. It recovered.
- Latency per graph call (all 82 calls):

| Metric | Value |
|---|---|
| min / p50 / p90 / max | 8.4 s / 38.5 s / 62.0 s / 95.9 s |
| mean, sum | 42.9 s, 3,518 s |
| Baseline (n=77, includes timeouts) | p50 37.1 s, p90 140.9 s, max 311.6 s |

- Tokens (graph calls): input p50 2,662 (sum 222,961). Output p50 4,758, max 12,000 (sum 445,280). Baseline output p50 4,547 (sum 327,424).
- Stop reasons: `end_turn` 71, `max_tokens` 11 (6 at the 8,000 cap, 5 at the 12,000 cap). Outcome `ok` on all 82 calls at the SDK level.
- The five degraded pages' calls:
  - First attempts (8,000 cap): 60.6, 62.0, 63.8, 64.8, 63.9 s. Sum 315.1 s.
  - Retries (12,000 cap): 89.6, 91.3, 95.0, 95.9, 93.9 s. Sum 465.7 s.
  - Total for the five pages: 780.8 s.
- Schema resolution: 6.70 s, 174,420 input tokens, 166 output tokens. Baseline 7.15 s with 161,575 input tokens.

Merge and graph build:

- Calls per page: the calls and page deltas are built in parallel, and the merge is serial in page order.
- Serial merge (`entity_resolution.*`) for this run: `new_entity` 617, `uncertain` 411, `contradiction` 2,052, `confirmed` 653. Merge span 11.0 s. Baseline: 591, 366, 1,797, 642, span 9.3 s.
- Cross-page link: `graph.crosspage_link.done` created 2 links (baseline 3).
- Output graph: `layer3.graph_memory_concurrent.done` shows 1,028 nodes, 1,406 edges, and 71 pages covered. `storage.document_graph_saved` confirms 1,028 nodes and 1,406 edges.

Pages not covered: 12, 14, 15, 16, 17. Each was marked `degraded` (`llm.page.quality`) and checkpointed as `DEGRADED` (`src/ai/layer3_extraction/extractor.py`, around lines 528 to 540). The code still ingests the degraded delta, but these pages are absent from `pages_covered`. **[inferred]** The truncated output was unparseable, so these pages contributed no nodes.

Persist: `storage.extraction_saved`, `storage.document_graph_saved`, and `storage.job_pdf_saved` completed within 0.94 s.

## 4. Concurrency per layer (configured versus observed)

Observed concurrency is derived from each event's timestamp minus its latency. The `llm.converse.call` and `layer2.printed.lambda_converted` events are logged when the call ends, so start = timestamp minus `latency_ms`. This was checked against the first graph call: page 8 started at 17:03:21.86 and ended at 17:03:30.29, which matches its 8.36 s latency.

| Layer | Configured | Observed peak | Observed mean over active span | Baseline (peak / mean) |
|---|---|---|---|---|
| Layer-1 inspection pass 1 | Sequential | 1 | 1 | 1 |
| Layer-1 probe | Serial CV loop, sample calls in one wave (code reading of the baseline) | Not logged per call | Not available | Not available |
| Layer-2 OCR (Paddle printed Lambda) | `PAGE_CONCURRENCY_LIMIT` = 25 | **25** (peak at 17:00:01.97) | 18.6 over 202.2 s | 25 / 19.3 |
| Layer-2 escalation VLM | 1 call | 1 | 1 | 1 |
| Layer-3 graph LLM calls | `GRAPH_CONCURRENCY_LIMIT` = 16 | **16** (peak at 17:03:22.01) | **15.0** over 234.7 s | 6 / 5.9 |
| Layer-3 merge | Serial by design | 1 | 1 | 1 |
| Schema resolution | 1 call | 1 | 1 | 1 |

Notes:

- The OCR mean of 18.6 is below the peak of 25 because the work runs unevenly. The sum of Lambda latencies is 3,761.6 s. Divided by 25 workers, that gives a lower bound of 150.5 s, against 202.2 s observed. The gap comes from workers that sit idle near the start and the tail.
- The graph mean of 15.0 is 94% of the lane bound. The sum of graph call latencies is 3,518 s. Divided by 16 lanes, that gives 219.9 s, against 234.7 s observed.
- The Layer-1 probe's sample VLM calls are not in `llm_metrics`, so their overlap cannot be measured from these logs.

## 5. Fix validation against the baseline

| Metric | Baseline `:6` (job_4cf06e33305d) | Post-fix `:7` (job_988e2a763956) | Change |
|---|---|---|---|
| Job wall clock | 1,094.4 s | 492.3 s | -602 s (-55%) |
| Graph build stage (start to done) | 858.8 s | 252.5 s | -606 s (-71%) |
| Graph page-call phase | 842.3 s | 234.7 s | -608 s (-72%) |
| Graph calls in flight, peak | 6 | 16 | matches the 16 configured |
| Graph calls in flight, mean over phase | 5.9 | 15.0 | |
| `ReadTimeoutError` (worker_failed events) | 7 | 0 | |
| Longest graph call | 311.6 s (timeout) | 95.9 s (completed) | |
| Thread time lost to dead calls | 2,157 s | 0 | |
| Pages failed or degraded | 7 failed (12 to 18) | 5 degraded (12, 14, 15, 16, 17) | |
| Pages covered in graph | 69 | 71 | +2 |
| Nodes / edges | 957 / 1,357 | 1,028 / 1,406 | +71 / +49 (+7.4% / +3.6%) |
| Layer-2 OCR stage | 199.4 s | 202.2 s | unchanged, as expected |

Fix 1: works. Effective graph concurrency rose from about 6 to 16. The page-call phase is about 3.6 times faster, and the lane math explains it.

Fix 2: not validated by this run. No read timeout occurred, so no dead call ran for the new 240 s limit. The new code sets connect 10 s, read 240 s, and 2 attempts in standard retry mode (`src/adapters/llm/bedrock_client.py`, `_bedrock_runtime_config`, around lines 129 to 172). The `:7` task sets no `IDP_BEDROCK_*` overrides, so these defaults apply. Botocore does not log its own retries, so the retry count is not available.

The comparison is not clean on one point. The baseline timeouts also held threads for about 300 s each, which the new run avoided. The concurrency fix and the absence of timeouts both changed in this run, so their separate effects are not measured.

## 6. Remaining bottlenecks, ranked by share of the 492 s

1. **Layer-2 OCR, 202 s (41%).** Work is 3,762 s of Lambda time spread over 25 workers. The lower bound is 150 s, and the observed time is 52 s above that bound.
2. **Graph page calls, 235 s (48%).** Lane-bound at 16. Output length is the driver: p50 4,758 output tokens per call. The baseline ran at about 126 output tokens per second. I did not re-measure throughput for this run.
3. **Five truncated pages, 781 s of Bedrock call time (22% of graph call time).** Each of the five pages holds one of the 16 lanes for about 150 s (first attempt plus retry) and then produces no graph output. This is both a latency and a correctness problem.
4. **Probe, 26.7 s (5%).** Serial, before OCR. Lipika resolved 0 of 3 samples, so the VLM decided every sample.
5. **Serial merge and schema call, 18 s (4%).** Merge 11.0 s, schema resolution 6.7 s.

Not on the critical path this run: the page 64 escalation (31.4 s, overlapped with the OCR tail), and the Layer-1 second inspection pass (inside the OCR window).

## 7. Recommendations (not implemented)

1. **Stop truncated pages from being dropped.** Options: allow a larger output cap on retry (confirm the model's maximum output tokens first), split dense pages before the call, or ask for a more compact JSON. This recovers the five pages and frees about 780 s of lane time.
2. **Test a higher `GRAPH_CONCURRENCY_LIMIT`.** The executor is sized `min(limit, 32)` (`extractor.py` around line 507), so the code ceiling is 32. A lane bound at 24 would be about 147 s, a saving of about 85 s, if Bedrock throughput quota allows. I did not check the quota.
3. **Test a higher `PAGE_CONCURRENCY_LIMIT`.** The lower bound drops from 150 s at 25 workers to 99 s at 38 workers. Observed gains will be smaller. The Lambda account limit is 400 unreserved concurrent executions (`aws lambda get-account-settings`), so it is not binding.
4. **Surface degraded pages in the job result.** Today they log at WARNING and checkpoint as `DEGRADED`, and the job still ends with `storage.job_pdf_saved`. Nothing marks the job as partial.
5. **Overlap the probe with OCR** (26.7 s) and **pipeline the merge** (11 s) only if the first three items leave them as the largest costs.
6. **Re-run the same document with a timeout-heavy condition** (for example a lower `IDP_BEDROCK_READ_TIMEOUT` in a test environment) to observe the bounded-timeout behavior. This run cannot validate fix 2.

## 8. Method and limits

Commands and sources:

- `aws ecs describe-tasks --cluster idp-cluster --tasks 654a5bc60bd94b269a02ebd255943005 69bf62d5026c4a9f8e3e21db25faaf06` for task definitions and timing.
- `aws ecs describe-task-definition --task-definition idp-layer3-task:7` for env and image.
- `aws logs describe-log-streams --log-group-name /ecs/idp-layer3-app --order-by LastEventTime --descending` to find streams.
- `aws logs get-log-events` (paged with `nextForwardToken`) for both streams, parsed offline. Structured events are JSON lines in `message`.
- `aws logs filter-log-events --log-group-name /aws/lambda/idp-engine-paddle-printed` for `REPORT` lines in 11:28 to 11:40 UTC.
- `aws lambda get-account-settings` for the concurrency limit.
- Code references: `src/ai/layer3_extraction/extractor.py` (executor and merge), `src/adapters/llm/bedrock_client.py` (timeouts and retries).

Limits:

- One run per build. Latency and concurrency numbers have no variance estimate.
- `llm_metrics` excludes the probe's sample VLM calls and the escalation VLM call. Their latency is only available from log gaps.
- Botocore retries are not logged. The number of retry attempts is not available.
- Bedrock quota for the inference profile was not checked. Any proposal to raise graph concurrency depends on it.
- The degraded pages' lost content is inferred from the absence of pages 12, 14 to 17 in `pages_covered`. The contents of the discarded output were not logged.
