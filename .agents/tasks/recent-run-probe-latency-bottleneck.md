# Recent run: probe behaviour, per-phase latency, and bottleneck

Read-only investigation. No source file, task definition, or AWS resource was changed. AWS access was limited to describe, list, and get calls on ECS and CloudWatch Logs. Credentials were loaded from `.env2` into the process environment and never printed. All times are IST (+05:30) as the app logged them, on 2026-10-26 unless marked otherwise.

## 1. Summary

1. **Probe:** It fired on all 76 pages. Lipika returned no script for all 3 sampled pages, because each top-1 vs top-2 margin (0.021 to 0.049) was below the 0.20 minimum. The Bedrock VLM decided the script for every sample, and all 76 pages routed to `paddleocr_printed`. The probe took 19.34 s, or 4.9% of the run.
2. **Latency:** The job took 397.45 s. OCR (Layer 2) took 194.28 s (48.9%). Graph page calls took 163.59 s (41.2%). Schema resolution, the serial merge, and the other stages together took about 16 s (4%).
3. **Bottleneck:** OCR is the largest single phase. Its 76 PaddleOCR Lambda calls ran through 25 lanes at a median of 47.4 s each, and the last regular page finished at 18:14:51.345. Graph calls are nearly as large (41%) and are almost fully lane-bound at 24 concurrent calls. Neither phase is bound by Fargate CPU or by the merge. Fargate memory and CPU were not measured.
4. **What changed from the :7 run:** Raising `GRAPH_CONCURRENCY_LIMIT` from 16 to 24 cut the graph call phase from 234.7 s to 163.6 s (-30%), and the effective concurrency now matches the setting. The wall clock fell from 492.3 s to 397.5 s. OCR was flat (202.2 s to 194.3 s). Truncation dropped from 5 pages to 4 pages (12, 14, 15, 17), and no Bedrock read timeout occurred.

## 2. Which run was analyzed

| Item | Value | Source |
|---|---|---|
| Job id | `job_40e44ea79ec2` | `storage.job_pdf_saved`, `storage.document_graph_saved` |
| Document | `Chander Kochhar 01_compressed 2.pdf`, 76 pages (same document as the :6 and :7 runs) | `pipeline.concurrent_pages_start` (`page_count` 76) |
| Log stream | `web/idp-web-console/c90d6a633ef543e5822583d80fc0e38b` in `/ecs/idp-layer3-app` | `aws logs describe-log-streams --order-by LastEventTime --descending` returns this stream first |
| Task | `c90d6a633ef543e5822583d80fc0e38b` on `idp-cluster`, status RUNNING | `aws ecs describe-tasks` |
| Task definition | `idp-layer3-task:8`, 2048 CPU, 8192 MB | `aws ecs describe-tasks` |
| Task timing | created 18:02:47, started 18:05:26 | `aws ecs describe-tasks` |
| Job start to end | 18:11:14.237 to 18:17:51.689, 397.45 s | `llm.factory.selected` to `storage.job_pdf_saved` |

The brief's example stream (`c90d6...`) is the newest one, and its job is `job_40e44ea79ec2`. The brief said its last event was 2026-10-27 05:01:29Z. When I fetched the full stream, events ran until 2026-10-27 11:06 IST. The later events are health checks and `/pipeline/status` access lines, not job events. The job's last event is `storage.job_pdf_saved` at 18:17:51.689.

Stream `4ac558a8e18e4c6e8aa6499ea24092fa` overlapped this run (about 17:34 to 18:11 IST). Its task is no longer described (`reason: MISSING`). I did not analyze it.

The log stream holds 19,813 events in total. This job accounts for 819 lines that name its job id.

## 3. Probe: how it worked in this run

Source: `src/ai/layer1_probe/` (`probe.py`, `routing.py`, `script_classifier.py`, `types.py`, `aggregate.py`, `vlm_probe.py`). Routing and capability code: `src/ai/layer1_routing/router.py`, `capability_router.py`, `capability_types.py`.

**Did it fire?** Yes. `probe.complete_images` logged `candidates` 76, `probed` 76, `timed_out` 0, `cache_hit` false, `onnx_calls` 3, `vlm_calls` 3, `lipika_resolved` 0, `lipika_vlm_agree` 0, `lipika_vlm_disagree` 0, `elapsed_s` 19.34. Document script: `latin`, confidence 0.987, `heterogeneous` false.

**Sampling.** Three pages were sampled. `probe.lipika_vs_vlm` logged these:

| Page | Lipika script | VLM script | Margin | `n_windows` |
|---|---|---|---|---|
| 1 | null | latin | 0.021 | 32 |
| 39 | null | latin | 0.049 | 32 |
| 76 | null | latin | 0.037 | 32 |

**Why lipika abstained.** The lipika observer returns no script when the top-1 vs top-2 margin is below `lipika_min_margin`, which defaults to 0.20 (`src/ai/layer1_probe/types.py:87-88`). All three margins were below 0.05. The logged `n_windows` of 32 equals `lipika_max_windows` (`types.py:88`, applied in `script_classifier.py:251`). The logs cannot show whether the uncapped window count was larger than 32.

**Routes and flags** (76 `probe.page_decision` events):

- `probe_route`: `paddleocr_printed` on 76 pages.
- `probe_script`: `latin` on 76 pages. For 73 pages this is most likely the document prior (`probe.py`, the prior branch around lines 197 to 211). The `page_decision` event does not log `script_source`, so this is inferred.
- `probe_has_table` true on 29 pages. `probe_is_blank` true on 0 pages.
- `probe_handwriting_prob`: median 0.01. No page reached 0.05 (the rounded maximum was 0.0). No page carried a handwriting flag from the probe.
- `final_engine`: `paddleocr_printed` on 75 pages and `vlm_transcribe` on page 64.

**Handwritten engine was planned but never ran.** `router.engine_plan_built` logged `plan: ["paddleocr_printed", "paddleocr_handwritten"]` on the scanned pages. `router.py:211-216` sets `has_handwriting` on every scanned page, whatever the image coverage. No `layer2.*handwritten*` event and no `paddleocr_handwritten` engine task appeared in the stream. The routing logic that drops the handwritten engine in favour of the printed engine was not traced fully. I did not confirm the exact mechanism. `capability_router.py:158-162` only makes HANDWRITING mandatory when the VLM reports handwriting above a ceiling, and this run had no per-page VLM classification.

**Escalation.** Page 64 had OCR confidence 0.54, below the 0.7 threshold (`router.escalation_decision`, 18:14:20.664). It was sent to `vlm_transcribe` and returned confidence 0.92 (`pipeline.escalation_attempt`, 18:14:53.803). That is 33.1 s. Its VLM call is not in `llm_metrics`.

**Probe duration.** 19.34 s, which is 4.9% of the wall clock. The log does not split this time among the CV loop, the three ONNX calls, and the three VLM calls.

## 4. Per-phase latency

Wall clock: 397.45 s (18:11:14.237 to 18:17:51.689).

| # | Phase | Window (IST) | Duration | % of wall | Basis |
|---|---|---|---|---|---|
| 1 | Setup: open PDF, render 76 pages at 150 DPI, page-inspection pass 1 | 18:11:14.237 to 18:11:20.184 | 5.95 s | 1.5% | Derived. Render is not logged alone. Probe start = probe end minus `elapsed_s`. |
| 2 | Probe (Layer 1) | 18:11:20.184 to 18:11:39.524 | 19.34 s | 4.9% | `probe.complete_images.elapsed_s` |
| 3 | OCR (Layer 2, 25 workers) | 18:11:39.527 to 18:14:53.806 | 194.28 s | 48.9% | `pipeline.concurrent_pages_start` to `pipeline.document_complete` |
| 4 | Markdown and IR save | 18:14:53.806 to 18:14:54.123 | 0.32 s | 0.1% | `ir_store.saved`, `layer3...start` |
| 5 | Graph page calls (24 lanes) | 18:14:54.129 to 18:17:37.712 | 163.59 s | 41.2% | First and last `llm.converse.call` (graph_extraction) window |
| 6 | Serial entity merge | 18:17:37.715 to 18:17:43.971 | 6.26 s | 1.6% | First and last `entity_resolution.*` event |
| 7 | Schema resolution (1 LLM call) | 18:17:43.982 to 18:17:51.234 | 7.25 s | 1.8% | `graph.schema_resolution.start` to `graph.extraction.completed` |
| 8 | Persist | 18:17:51.234 to 18:17:51.689 | 0.46 s | 0.1% | `storage.*` events |
| | Gaps between phases | | 0.02 s | 0.0% | |
| | **Total** | | **397.45 s** | **100%** | |

The eight rows add up to 397.46 s, within rounding.

### 4.1 OCR detail (Layer 2)

- `layer2.printed.lambda_converted.latency_ms`, n=76: min 14.8 s, median 47.4 s, p90 68.3 s, max 82.7 s, sum 3,694.7 s.
- `pipeline.page_complete.total_latency_ms`, n=76: median 48.2 s, max 92.9 s.
- Concurrency, computed as event timestamp minus `latency_ms`: peak 25 at 18:11:41.872 (equal to `workers: 25` in `pipeline.concurrent_pages_start`). Mean over the 190.3 s span: 19.4 (78% of 25).
- Lower bound from work: 3,694.7 s divided by 25 lanes is 147.8 s. Observed span is 190.3 s, so about 42 s is lost to uneven lane use.
- Tail: active calls fell from 16 to 6 to 1 across the last 40 s of the phase. Page 64 finished at 18:14:20.664 and its escalation ended at 18:14:53.803. Page 74 was the last regular page to finish, at 18:14:51.345. The escalation therefore set the end of the OCR phase, 2.5 s after page 74.
- Lambda cold starts were not measured in this investigation. The earlier `REPORT` analysis in `run-profile-and-topology.md` found no large cold-start effect.

### 4.2 Graph detail (Layer 3)

- 83 `llm.converse.call` (graph_extraction) records for 76 pages, plus 1 `schema_resolution`. Seven pages needed a second call (12, 13, 14, 15, 16, 17, 18). That accounts for 76 + 7 = 83.
- Outcome `ok` on all 83. Read timeouts: 0. Longest call: 94.7 s (page 14, retry).
- Latency: min 7.9 s, median 38.5 s, p90 62.0 s, max 94.7 s, sum 3,499.2 s, mean 42.2 s.
- Output tokens per call: median 4,846, p90 8,000, max 12,000, sum 450,389. Output throughput: median 128.5 tokens/s.
- Input tokens: sum 225,455 (median 2,667).
- Stop reasons: `end_turn` 72, `max_tokens` 11. Seven first attempts hit the 8,000-token cap. Four retries hit the 12,000-token cap (`bedrock.extract_graph.truncated_after_retry`, pages 12, 14, 15, 17).
- Pages that recovered on the 12,000 retry: 13, 16, 18.
- Pages still degraded (`llm.page.quality` = `degraded`): 12, 14, 15, 17. Their call time was 617.8 s in total (first attempts 247.5 s, retries 370.3 s). That is 17.7% of all graph call time for four pages out of 76.
- Concurrency, from the same timestamp-minus-latency method: peak 24 at 18:14:54.267 (equal to the `concurrency_limit: 24` in `layer3.graph_memory_concurrent.start`). Mean over the 163.6 s span: 21.4 (89% of 24). Lane lower bound: 3,499.2 s divided by 24 is 145.8 s, observed 163.6 s.
- Admission: pages 1 to 24 started at 18:14:54.1. Page 25 started at 18:15:02.1 once the first call returned. Page 76 started at 18:16:57.3.
- Merge: 3,759 `entity_resolution.*` events (`new_entity` 642, `contradiction` 2,037, `uncertain` 413, `confirmed` 667). The merge ran for 6.26 s.
- Final graph (`layer3.graph_memory_concurrent.done`): 1,055 nodes, 1,450 edges, 72 pages covered. Pages not covered: 12, 14, 15, 17. Cross-page links: 4.
- Schema resolution: 7.25 s, 180,110 input tokens, 166 output tokens. Resolved fields: patient name "MRS CHANDER KOCHHAR", patient id 192673, admission 21/02/2025, discharge 24/02/2025, procedure "CT ANGIOGRAPHY PULMONARY", approved claim amount 80698.

Code that sets graph concurrency: `src/ai/layer3_extraction/extractor.py:376` reads `graph_concurrency_limit`. Lines 505 to 509 create a dedicated `ThreadPoolExecutor(max_workers=min(concurrency_limit, 32), thread_name_prefix="graph-llm")`. This replaces the default pool that capped the earlier runs at about 6 threads. That is why the observed peak now reaches 24.

## 5. Bottleneck

**Biggest single contributor: the OCR phase (Layer 2), 194.3 s, 48.9% of the wall clock.**

- Why it dominates: 76 pages took a median 47.4 s each in the PaddleOCR Lambda. With 25 lanes, that is about three full rounds plus a tail. The OCR phase cannot finish faster than its work divided by its lanes (148 s), and the tail adds about 42 s on top.
- Type: **OCR-Lambda-bound.** Each lane waits on a synchronous `lambda.invoke`. Per-page Lambda time was not isolated, so the 47 s median cannot be split into cold start, PaddleOCR model time, and payload time from these logs.
- The page 64 VLM escalation (33 s) finished last and set the phase end, 2.5 s after the next-to-last page.

**Close second: graph page calls, 163.6 s, 41.2%.**

- Type: **Bedrock-bound and lane-bound.** The 24 lanes ran at 89% average utilization. Output length drives each call (median 4,846 output tokens at about 128 tokens/s). Four degraded pages each held a lane for about 150 s (a 60 s first attempt and a 90 s retry at the 12,000-token cap). Those four pages used 17.7% of graph call time.

**Not bottlenecks in this run:**

- Probe (19.3 s, 4.9%). Its three lipika ONNX runs were discarded, so their time bought no routing information.
- Serial merge (6.3 s) and schema resolution (7.3 s). Together they are 3.4%.
- Fargate CPU and memory. The logs do not measure either, so I cannot rank them against the other phases. The brief raised a concern that OCR concurrency holds page images in the 8 GB task. That memory use was not measured here.

## 6. What changed versus the prior runs

Compared with `latest-run-latency-concurrency.md` (`job_988e2a763956` on `:7`, 16 graph lanes):

| Metric | `:7` (prior) | `:8` (this run) | Change |
|---|---|---|---|
| Wall clock | 492.3 s | 397.5 s | -95 s (-19%) |
| Probe | 26.7 s | 19.3 s | -7 s (single runs, no variance estimate; no code change identified) |
| OCR phase | 202.2 s | 194.3 s | -8 s (within run-to-run noise) |
| Graph call phase | 234.7 s | 163.6 s | -71 s (-30%) |
| Graph peak / mean concurrency | 16 / 15.0 | 24 / 21.4 | Effective concurrency now tracks the setting |
| Graph lane lower bound | 219.9 s | 145.8 s | |
| Merge | 11.0 s | 6.3 s | |
| Schema resolution | 6.7 s | 7.3 s | |
| Degraded pages | 5 (12, 14, 15, 16, 17) | 4 (12, 14, 15, 17) | Page 16 recovered on retry, as did 13 and 18 |
| Read timeouts | 0 | 0 | Fix 2 from the prior report (bounded timeouts) still not exercised |
| Nodes / edges | 1,028 / 1,406 | 1,055 / 1,450 | Pages covered 71 to 72 |

Compared with `run-profile-and-topology.md` (`job_4cf06e33305d` on `:6`, 1,094 s), the graph call phase fell from 842 s to 164 s, and the 7 timeout pages (about 300 s each) are gone. Graph concurrency rose from about 6 to 24.

The raised graph concurrency did what it was meant to do: the graph phase now runs at the configured lane count. The total gain was smaller than the lane model would predict from the setting alone because of the four degraded pages.

## 7. Recommendations (not implemented)

1. **Raise OCR parallelism only after measuring container memory.** At 38 lanes, the OCR lane bound falls from 148 s to about 97 s. Each OCR worker thread keeps a page image in the Fargate task while its Lambda call is in flight. This is the risk the brief raised. Read the ECS `MemoryUtilization` metric for this task during the OCR phase first.
2. **Investigate the 47 s median per OCR page.** Lambda `Duration` for the PaddleOCR printed engine is the largest single cost. Logs cannot split its cold start from the model time. A test with one page at a time would show the baseline.
3. **Stop losing four pages to the output cap.** Each degraded page costs about 150 s of graph lane time and produces no nodes. Options: allow a higher first-attempt output cap (confirm the model's maximum output tokens first), split dense pages before the call, or ask for more compact JSON. Also surface degraded pages in the job result. Today they log a warning and the job still completes with `storage.job_pdf_saved`.
4. **Consider disabling or retuning lipika in the probe.** Its margins (0.02 to 0.05) never reach the 0.20 minimum, so all three samples fall to the VLM. This costs probe time with no routing gain. Any change to `lipika_min_margin` needs a labelled evaluation first.
5. **Check the handwritten route before trusting coverage.** The handwritten engine was planned on every scanned page but never invoked. If a real handwritten page arrives, it may go to the printed engine. This does not affect latency here, but it is a correctness question. Trace the matcher in `capability_router.py` with a known handwritten document.
6. **Leave the probe overlap and merge pipelining for last.** Each is 3% to 5% of wall clock. They matter only after items 1 to 3.

## 8. Method and limits

Sources:

- `aws ecs describe-tasks --cluster idp-cluster --tasks c90d6a633ef543e5822583d80fc0e38b 4ac558a8e18e4c6e8aa6499ea24092fa` for task definition, timing, and status.
- `aws logs describe-log-streams --log-group-name /ecs/idp-layer3-app --order-by LastEventTime --descending` for stream order.
- `aws logs get-log-events --start-from-head`, paged with `nextForwardToken`, for the full `c90d6...` stream (19,813 events, parsed offline).
- Source reads: `src/ai/layer1_probe/types.py`, `script_classifier.py`, `probe.py`, `aggregate.py`, `vlm_probe.py`; `src/ai/layer1_routing/router.py`, `capability_router.py`, `capability_types.py`, `pipeline.py`; `src/ai/layer3_extraction/extractor.py`.

Limits:

- One run. No variance estimate. The probe and OCR changes between the two runs may be noise.
- The probe's sample VLM and lipika calls are not in `llm_metrics`. Their time is included only in the 19.34 s probe total.
- Render time is not logged. It is derived as the gap from job start to the probe start.
- Fargate CPU, container memory, and Bedrock quota usage were not measured.
- Lambda cold starts were not measured for this run.
- The degraded pages' lost content is inferred from their absence from `pages_covered`. The discarded output was not logged.
- Concurrency figures are estimates from event timestamps minus `latency_ms`. Log lines are written when a call ends, so the start times are derived, not observed.
