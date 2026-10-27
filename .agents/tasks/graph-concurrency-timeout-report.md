# Report: graph LLM thread pool (FIX 1) and Bedrock timeouts (FIX 2)

Branch: `feat/llm-observability`. Working tree only. Nothing committed, nothing pushed.
Review iteration: first (no `graph-concurrency-timeout-review.json` existed).

## Scope confirmation

- No AWS calls. All LLM and Bedrock calls are mocked (`MagicMock` or a recording fake `boto3`).
- No deploy, no docker build, no IAM change, no `.onnx` edit.
- `scripts/deploy_idp_fargate.py`, `docker/`, `step_functions/`, and `src/ai/layer1_probe/vlm_probe.py` were not edited by this change.
- `BedrockLLMClient` (Layer 1 client, `bedrock_client.py` around lines 676-677) was not edited.
- No `retries` or `max_attempts` setting was added anywhere. The app-level throttle loop (`_call_converse`, `_THROTTLE_MARKERS`) was not edited.
- No git commit or push. `git log -1` is still `acd90d1`.

## Files changed

| File | Change |
|---|---|
| `src/ai/layer3_extraction/graph_agent/agent.py` | FIX 1. `build_page_delta_async` takes an optional `executor` keyword. When it is `None`, the old `asyncio.to_thread` call runs unchanged (Modes A and C use this). When it is set, the call runs through `loop.run_in_executor(executor, copy_context().run(...))`, which keeps the logger's `correlation_id` in the worker thread. |
| `src/ai/layer3_extraction/extractor.py` | FIX 1. In `extract_with_graph_memory_concurrent` only, the build-phase `gather` runs inside a dedicated `ThreadPoolExecutor(max_workers=min(concurrency_limit, 32), thread_name_prefix="graph-llm")`. `_build_page` passes `executor=llm_executor`. The semaphore, checkpoints, `worker_failed` handling, serial sorted merge, reconcile, and link steps are unchanged. The file also has uncommitted changes from before this task (3 lines). Those were kept as they were. |
| `src/adapters/llm/bedrock_client.py` | FIX 2. New helper `_bedrock_runtime_config()` returns a `botocore.config.Config` with connect and read timeouts only. Both `client` construction branches pass `config=`. No other lines changed (`git diff` shows only the helper and two `config=` arguments). |
| `tests/test_graph_concurrency_executor.py` | New. Five tests (details below). |
| `tests/test_bedrock_client_timeouts.py` | New. Seven test cases across four test functions (details below). |
| `.agents/tasks/graph-concurrency-timeout-report.md` | This file. |

## FIX 1: executor approach

- **Where the pool is created:** in `extract_with_graph_memory_concurrent`, around the build-phase `asyncio.gather(*tasks)`, after the semaphore is created with `concurrency_limit`.
- **Sizing:** `max_workers=min(concurrency_limit, 32)`. `concurrency_limit` is the same variable that feeds `asyncio.Semaphore(concurrency_limit)`. `os.cpu_count()` is not used. With the settings default of 8 that gives 8 threads. The live task definition uses 16, which gives 16 threads.
- **Shutdown:** `with ThreadPoolExecutor(...) as llm_executor:` calls `shutdown(wait=True)` when the build phase exits, so no `graph-llm` thread outlives the build. The pool is only created when `tasks` is non-empty.
- **Why a dedicated pool:** the default executor used by `asyncio.to_thread` has about `min(32, cpu_count + 4)` threads, which is about 6 on the 2-vCPU task. Each admitted page therefore waited in the pool queue even though the semaphore allowed 16.
- **Not changed:** Mode A (`extract_with_graph_memory_ordered`) and Mode C (`extract_with_graph_memory_batched`) still use the default pool. They are out of scope. The follow-up is to apply the same change there.

## FIX 2: timeout values

| Setting | Default | Env override |
|---|---|---|
| `connect_timeout` | 10 s | `IDP_BEDROCK_CONNECT_TIMEOUT` |
| `read_timeout` | 240 s | `IDP_BEDROCK_READ_TIMEOUT` |

- Values that are not numbers, or are `<= 0`, log `bedrock.config.invalid_timeout` and use the default.
- The `Config` applies to the `bedrock-runtime` client in both the `AWS_PROFILE` session branch and the default-credentials branch.
- **Retries: left untouched.** No `retries` or `max_attempts` key was added. Botocore's default (legacy mode, 5 attempts) still applies.
- **Caveat (not resolved, needs user confirmation):** with retries unchanged, a dead endpoint can now take up to 5 × 240 s = **1200 s, about 20 minutes**, per call. Today the same case takes about 300 s (5 × 60 s). A page that hangs holds one of the concurrency slots about four times longer. The timeout bounds each attempt, but the total is not bounded until retries are capped, which is the deferred retry task.
- Connect failures now fail after 10 s instead of botocore's 60 s. This is intended, but it is a behavior change.

## Tests

### New tests

`tests/test_graph_concurrency_executor.py` (5 tests, all pass):
- `test_a_concurrency_reaches_limit`: limit 16, 20 pages. A `threading.Barrier(16)` on the first 16 calls. The default asyncio pool is set to 6 threads to simulate the 2-vCPU task. Asserts the peak concurrency is 16, no failed pages, and 20 pages succeeded.
- `test_b_executor_shut_down_no_leaked_threads`: after the build, no thread whose name starts with `graph-llm` remains.
- `test_c_pool_size_follows_limit_capped_at_32`: limit 4 gives a pool of 4. Limit 64 gives 32. The `ThreadPoolExecutor` class is wrapped to record `max_workers` for `graph-llm` pools only.
- `test_d_merge_order_and_results_unchanged`: page 1 finishes last, page 6 finishes first. The graph snapshot and stats must equal a serial `limit=1` run, and nodes must appear in ascending page order.
- `test_e_correlation_id_propagates_to_worker_thread`: every worker call sees `correlation_id == "cid-test"`.

`tests/test_bedrock_client_timeouts.py` (5 test cases from four test functions, all pass):
- `test_a_default_branch_passes_timeout_config`: the `boto3.client` kwargs contain `config`, with `read_timeout == 240`, `connect_timeout == 10`, `retries is None`. No `retries` key is passed.
- `test_b_session_branch_passes_timeout_config`: same checks through `boto3.Session(profile_name=...).client(...)`.
- `test_c_env_overrides_are_honored`: `IDP_BEDROCK_READ_TIMEOUT=300` and `IDP_BEDROCK_CONNECT_TIMEOUT=5` are applied.
- `test_d_invalid_read_override_falls_back_to_default` (parametrized with `"abc"` and `"0"`): both fall back to 240 with no exception.

### Red check (new tests against the pre-change code)

I ran the new tests against a copy of the tree with the pre-change `agent.py`, `bedrock_client.py` (both from HEAD, which were clean before this task), and `extractor.py` (my two hunks reverted by hand).

| Test | Pre-change result | Why |
|---|---|---|
| executor `test_a` | FAIL | Pages failed because the 16-thread barrier broke at the 6-thread default pool. |
| executor `test_c` | FAIL | No `graph-llm` pool existed, so the recorded sizes were empty. |
| executor `test_b`, `test_d`, `test_e` | PASS | Guard behavior that must not change. |
| bedrock `test_a`, `test_b`, `test_c`, `test_d[abc]`, `test_d[0]` | FAIL | No `config` kwarg was passed. |

The red-check copy was deleted afterward.

### Commands and results

Before any edit (baseline), in the working tree:

```
.\.venv\Scripts\python.exe -m pytest tests/test_memory_manager.py tests/test_layer3_hardening.py tests/test_layer3_truncation_parse.py tests/test_llm_metrics.py tests/test_merge_determinism.py -q
→ 1 failed, 81 passed
   pre-existing failure: tests/test_layer3_hardening.py::test_schema_resolution_fuzzy_candidate_must_not_populate_empty_field
```

After the edits:

```
.\.venv\Scripts\python.exe -m pytest tests/test_graph_concurrency_executor.py tests/test_bedrock_client_timeouts.py tests/test_memory_manager.py tests/test_layer3_hardening.py tests/test_layer3_truncation_parse.py tests/test_llm_metrics.py tests/test_merge_determinism.py -q
→ 1 failed, 91 passed (81 existing + 10 new)
   same pre-existing failure as baseline
```

Other suites:

```
.\.venv\Scripts\python.exe -m pytest schema_chatbot_v2/tests/ -q -p no:cacheprovider
→ 181 passed, 1 warning
```

`tests/` and `schema_chatbot_v2/tests/` cannot be collected in one command because both define `tests.conftest`. They were run separately.

Full `tests/` run, excluding `tests/test_graph_memory_query_bot.py` and `test_paddle_printed_engine_local_path`. Run with `-v`, and it reached 165 PASSED lines before I stopped it:

```
.\.venv\Scripts\python.exe -m pytest tests/ -v -p no:cacheprovider --ignore=tests/test_graph_memory_query_bot.py --deselect=tests/test_layer2_engines_local.py::test_paddle_printed_engine_local_path
→ 1 failed in the portion that ran: tests/test_deploy_probe_service.py::test_ecs_tags_are_mandatory_agents_tags
```

The full `tests/` run did not finish in this environment. It stalls on tests that load OCR or embedding models (for example `test_paddle_printed_engine_local_path`). I stopped it. I did not get a full-suite before and after comparison. The changed areas were covered by the targeted set above.

### Failures outside this change (verified or noted)

- `tests/test_graph_memory_query_bot.py`: 3 failed, 13 passed (`test_12_graph_only_query_bot_fact`, `test_13_14_multi_hop_query`, `test_24_exact_end_to_end_demo`). The same 3 fail on the pre-change copy, so they are not caused by this change. Run alone it takes about 20 to 75 s.
- `tests/test_deploy_probe_service.py::test_ecs_tags_are_mandatory_agents_tags`: the test expects `ECS_TAGS["createdby"] == "vinay.k@shellkode"`. The AGENTS.md rule requires `vinay.k@shellkode.com`. **This is a real mismatch in `scripts/deploy_probe_service.py`, which I did not change.** Fixing it is a one-line change, but it touches the deploy script, so I left it for you to decide.

## Open items for the user

1. Confirm the Bedrock retry risk: the dead-call worst case goes from about 5 min to about 20 min with retries unchanged. The retry cap is the deferred task.
2. `BedrockLLMClient` (Layer 1 VLM and handwriting) still has no explicit timeout. Its calls can also hang for about 300 s. Say whether to apply the same Config there.
3. The probe client (`vlm_probe.py`) has `read_timeout=60`, which is shorter than the new 240 s for Layer 3. Probe calls can time out on their own.
4. Effective Bedrock concurrency for profile `24n3l11zddro` rises from about 6 to 16. Check the Bedrock throughput quota before deploying.
5. Modes A and C still use the default pool. Apply the same executor there if they run in production.
6. The `ECS_TAGS` mismatch above.
