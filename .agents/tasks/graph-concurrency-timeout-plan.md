# Implementation Plan: graph LLM thread pool and Bedrock timeouts

Branch: `feat/llm-observability`. Local code change only.

**Hard rules.** No AWS calls, no deploy, no docker, no git commit or push, no `.onnx` edits, no IAM changes. Do not touch `scripts/deploy_idp_fargate.py`, `docker/`, `step_functions/`, or the probe client in `src/ai/layer1_probe/vlm_probe.py`. Do not change any `retries` setting, and do not change the app-level throttle retry loop in `bedrock_client.py` (`_call_converse`, `_THROTTLE_MARKERS`).

**Restructuring decision.** No FEAT decomposition. The two fixes are small and dependent (the agent parameter has to exist before the orchestrator passes it), and they touch four files plus two new test files. The existing implement-and-review loop runs this plan as written.

## Pre-flight facts from exploration

- `deploy_idp_fargate.py` line 228 already reads `memory="8192"`. The brief says this sync is done, and the code agrees. No action.
- The working tree is already dirty. `src/ai/layer3_extraction/extractor.py` and `tests/test_layer3_hardening.py` have uncommitted changes. Edit on top of them. Do not run `git checkout`, `git stash`, or `git reset`.
- Production Layer 3 uses `BedrockExtractionClient`, via `src/adapters/llm/extraction_factory.py`. So the Config in the `client` property reaches the graph path.
- The thread-pool claim holds in code. `asyncio.to_thread` (`graph_agent/agent.py` line 102) uses the loop's default executor. The repo never sets a custom executor. Pool size is `min(32, cpu_count + 4)`, which is about 6 on 2 vCPU. The size is an inference from the report, so the tests force a small default pool instead of relying on the host CPU count.
- Botocore defaults relevant here: `read_timeout` 60s, `connect_timeout` 60s, `retries` None (legacy mode, 5 attempts). Verified with the repo's `.venv` (botocore 1.43.80, Python 3.11.9). Setting `connect_timeout` to 10 cuts the connect wait from 60s to 10s, which is a behavior change.
- `src/utils/logger.py` keeps `correlation_id` in a `ContextVar`. `asyncio.to_thread` copies the context into the worker thread. `loop.run_in_executor` does not. A plain swap would silently drop correlation IDs from logs written inside the LLM call. The plan copies the context explicitly.
- `src/adapters/llm/llm_metrics.py` guards its state with `threading.Lock`, so 16 concurrent worker threads are safe.
- Test command (from README): `.venv\Scripts\python.exe -m pytest <paths> -v`.
- Existing suites that cover this code: `tests/test_memory_manager.py` (concurrency test 9 and failure isolation), `tests/test_layer3_hardening.py` (concurrent path, agent direct calls), `tests/test_layer3_truncation_parse.py` and `tests/test_llm_metrics.py` (`BedrockExtractionClient`), `tests/test_merge_determinism.py` (concurrent merge order).

## Items

- [ ] 0. Record the baseline before any edit.
      Save SHA256 hashes of the four files this plan edits, so the final diff can be checked against the pre-change state. Run the existing suites and record the pass and fail counts.
      Files: none edited.
      Verify: `Get-FileHash` on `src/ai/layer3_extraction/graph_agent/agent.py`, `src/ai/layer3_extraction/extractor.py`, `src/adapters/llm/bedrock_client.py`, and `src/ai/layer3_extraction/graph_agent/memory_manager.py` is recorded. Then run `.venv\Scripts\python.exe -m pytest tests/test_memory_manager.py tests/test_layer3_hardening.py tests/test_layer3_truncation_parse.py tests/test_llm_metrics.py tests/test_merge_determinism.py -q`. Record the result. Any pre-existing failure is documented and is not caused by this change.

- [ ] 1. FIX 1, part A: add an optional executor to the per-page LLM call in `agent.py`.
      Add a keyword parameter `executor: Optional[concurrent.futures.Executor] = None` to `GraphExtractionAgent.build_page_delta_async`. Add the imports `concurrent.futures`, `contextvars`, and `functools`. Keep `Optional` (already imported).
      - When `executor is None`, keep the existing `await asyncio.to_thread(self.llm.extract_graph_from_page, page_md=clean_md, schema_fields=self.schema_fields, existing_nodes=context, page_number=page_number, total_pages=total_pages)` call byte-for-byte. Mode A (ordered) and Mode C (batched) keep this path.
      - When an executor is given:
        ```python
        call = functools.partial(
            contextvars.copy_context().run,
            self.llm.extract_graph_from_page,
            page_md=clean_md, schema_fields=self.schema_fields,
            existing_nodes=context, page_number=page_number, total_pages=total_pages,
        )
        llm_output = await asyncio.get_running_loop().run_in_executor(executor, call)
        ```
        The `copy_context().run` wrapper keeps the logger's `correlation_id` visible in the worker thread, which matches what `to_thread` already does.
      - Update the docstring. It currently says "Calls the synchronous LLM via asyncio.to_thread()". Change it to describe both paths.
      Files: `src/ai/layer3_extraction/graph_agent/agent.py`.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_memory_manager.py tests/test_layer3_hardening.py -q`. Results match the item 0 baseline. The default (no executor) path is unchanged.

- [ ] 2. FIX 1, part B: create the dedicated executor in the concurrent orchestrator.
      In `extract_with_graph_memory_concurrent` in `extractor.py`, `concurrency_limit` is already computed as `max(1, getattr(settings, "graph_concurrency_limit", 8))`. Use that same variable. Do not read `os.cpu_count()`.
      - Wrap only the build-phase `await asyncio.gather(*tasks)` in a dedicated pool:
        ```python
        if tasks:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(concurrency_limit, 32),
                thread_name_prefix="graph-llm",
            ) as llm_executor:
                await asyncio.gather(*tasks)
        ```
        The `with` block calls `shutdown(wait=True)` on exit, so no thread outlives the build. The `thread_name_prefix` lets a test find leaked threads.
      - In the existing `_build_page` closure, pass `executor=llm_executor` to `agent.build_page_delta_async(...)`. The closure reads `llm_executor` at call time, so defining `_build_page` before the `with` block is valid.
      - The semaphore (`asyncio.Semaphore(concurrency_limit)`) stays as is. It still controls admission. The pool size now matches it, so each admitted page gets a thread without queueing.
      - Do not change: the checkpoint writes, the `worker_failed` and `validate_page_delta` handling, the serial sorted merge loop, the reconcile and cross-page link steps, or `graph_out`. Leave Mode A (`extract_with_graph_memory_ordered`) and Mode C (`extract_with_graph_memory_batched`) untouched. They keep the default pool, and the report lists that as out of scope.
      Files: `src/ai/layer3_extraction/extractor.py`. Edit around the existing changes already in this file.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_memory_manager.py tests/test_layer3_hardening.py tests/test_merge_determinism.py -q`. Results match the item 0 baseline.

- [ ] 3. FIX 1, part C: tests for the executor. Create `tests/test_graph_concurrency_executor.py`. Use a fake LLM with no network. Use a plain `MagicMock` or a small class, and mirror the `resolve_schema_from_graph` mock from `tests/test_memory_manager.py`. Do not import boto3 or create a real Bedrock client.
      The fake `extract_graph_from_page` returns a minimal valid payload: one entity `{"id": f"p{n}_e1", "type": "Patient", "value": f"Patient {n}"}`, plus empty `relationships` and `reference_resolutions`, and `"quality": "clean"`. Set `settings.graph_concurrency_limit` with `monkeypatch.setattr`. Use `doc_id=None` so no database is touched.
      Tests:
      a. **Concurrency reaches the limit (the main regression test).** Use `limit=16` and 20 pages. Inside `asyncio.run(...)`, first call `asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=6))` to simulate the 2-vCPU default pool on any host. Shut that small pool down in a `finally`. The fake uses a `threading.Barrier(16, timeout=5)` for the first 16 invocations only (track the call index under a lock) and records the peak number of concurrent entries. Assert the peak equals 16, that no page failed (`graph_out["failed_pages"] == []`), and that `pages_succeeded == 20`.
         Red check: before items 1 and 2, this test must FAIL. The barrier breaks after 5s, and the observed peak is 6. That failure confirms the test catches the bug.
      b. **Executor is shut down (no leaked threads).** After the call returns, assert `[t for t in threading.enumerate() if t.name.startswith("graph-llm")] == []`.
      c. **Pool size is the limit, capped at 32.** Monkeypatch `extractor.concurrent.futures.ThreadPoolExecutor` with a subclass that records the `max_workers` kwarg and then calls the real class. Run with `limit=4` and assert the recorded size is 4. Run with `limit=64` and 3 pages, and assert the recorded size is 32.
      d. **Merge order and results unchanged.** Make the fake's sleep inversely proportional to page number (page 1 sleeps longest), so completion order is reversed. Assert that the merge order is ascending page order. Check it through `graph_out["merge_ledger"]` (confirm the record fields in `memory_manager.get_merge_ledger` before writing the assertion) or the node `source_pages` order. Also assert the final `graph_out["stats"]` equals the stats from a run with `limit=1`. Keep the sleeps short (under 0.05s each).
      e. **Logger context propagates into the worker thread.** Set `correlation_id` (from `src/utils/logger.py`) with `.set("cid-test")` before the call. The fake records `correlation_id.get()` inside its call. Assert every recorded value is `"cid-test"`.
      Files: `tests/test_graph_concurrency_executor.py` (new).
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_graph_concurrency_executor.py -v`. All five tests pass after items 1 and 2. Test (a) fails on the unmodified code, as noted above.

- [ ] 4. FIX 2, part A: add an explicit botocore Config to the Bedrock extraction client in `bedrock_client.py`.
      Add a module-level helper near the other helpers (before `class BedrockExtractionClient`):
      ```python
      def _bedrock_runtime_config():
          """Explicit botocore timeouts for bedrock-runtime. Retries are NOT set here (deferred task)."""
          from botocore.config import Config  # lazy import, same style as vlm_probe.py
          def _env_seconds(name: str, default: float) -> float:
              raw = os.getenv(name)
              if raw is None or not raw.strip():
                  return default
              try:
                  value = float(raw)
              except ValueError:
                  logger.warning("bedrock.config.invalid_timeout", env=name)
                  return default
              if value <= 0:
                  logger.warning("bedrock.config.invalid_timeout", env=name)
                  return default
              return value
          return Config(
              connect_timeout=_env_seconds("IDP_BEDROCK_CONNECT_TIMEOUT", 10),
              read_timeout=_env_seconds("IDP_BEDROCK_READ_TIMEOUT", 240),
          )
      ```
      Then in the `BedrockExtractionClient.client` property (lines 170-181), add `config=_bedrock_runtime_config()` to both construction calls:
      - session branch: `session.client("bedrock-runtime", region_name=self.region, config=_bedrock_runtime_config())`
      - default branch: `boto3.client("bedrock-runtime", region_name=self.region, config=_bedrock_runtime_config())`
      Do not pass `retries` or `max_attempts` anywhere.
      Do NOT touch `BedrockLLMClient.__init__` (lines around 676-677, the Layer 1 VLM client). See the flags section.
      Files: `src/adapters/llm/bedrock_client.py`.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_llm_metrics.py tests/test_layer3_truncation_parse.py -q`. Results match the item 0 baseline.

- [ ] 5. FIX 2, part B: tests for the timeout Config. Create `tests/test_bedrock_client_timeouts.py`. Replace the module's `boto3` with a fake using `monkeypatch.setattr(bedrock_client, "boto3", fake)`. The fake has `client(...)` and `Session(...)`, and each records its kwargs. Return a `MagicMock` from each. No AWS calls and no credentials are needed. Use `monkeypatch.delenv` for `AWS_PROFILE`, `IDP_AWS_PROFILE`, `IDP_BEDROCK_CONNECT_TIMEOUT`, and `IDP_BEDROCK_READ_TIMEOUT` in each test.
      Tests:
      a. **Default branch.** Construct `BedrockExtractionClient(model_id="m", region="ap-south-1")` and access `.client`. Assert the `boto3.client` kwargs contain `config`, that it is a `botocore.config.Config`, that `read_timeout == 240`, that `connect_timeout == 10`, and that `config.retries is None`. Assert `"retries"` is not a key in the kwargs.
      b. **Session branch.** `setenv("AWS_PROFILE", "test-profile")`. Assert the same checks on the `session.client` kwargs, and that `Session` received `profile_name="test-profile"`.
      c. **Env overrides.** `IDP_BEDROCK_READ_TIMEOUT=300` and `IDP_BEDROCK_CONNECT_TIMEOUT=5` produce `read_timeout == 300` and `connect_timeout == 5`.
      d. **Invalid overrides fall back.** `IDP_BEDROCK_READ_TIMEOUT="abc"` and `"0"` each fall back to 240. Assert no exception is raised.
      Red check: before item 4, test (a) fails with `KeyError: 'config'`.
      Files: `tests/test_bedrock_client_timeouts.py` (new).
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_bedrock_client_timeouts.py -v`. All four pass.

- [ ] 6. Full regression.
      Run the targeted set first, then the whole test suite. Compare against the item 0 baseline. Any new failure is a blocker. Any failure that was already in the baseline is reported as pre-existing.
      Files: none edited.
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_graph_concurrency_executor.py tests/test_bedrock_client_timeouts.py tests/test_memory_manager.py tests/test_layer3_hardening.py tests/test_layer3_truncation_parse.py tests/test_llm_metrics.py tests/test_merge_determinism.py -v`, then `.venv\Scripts\python.exe -m pytest tests/ schema_chatbot_v2/tests/ -q`. The count of failures is equal to or less than the item 0 baseline, and the new tests all pass.

- [ ] 7. Scope check on the final diff.
      Compare the four edited files against the item 0 hashes. Confirm the changes are limited to the intended hunks: the executor and the agent parameter (items 1 and 2), and the helper plus the two `client` calls (item 4). Confirm `deploy_idp_fargate.py`, `docker/`, `step_functions/`, `.onnx` files, `vlm_probe.py`, `BedrockLLMClient`, and the `_call_converse` retry loop are unchanged. Confirm no `retries` string was added to `bedrock_client.py`.
      Files: none edited.
      Verify: `git diff --stat` lists only the four source files and the two new test files, plus anything that was already dirty before item 0. Run `git diff -- src/adapters/llm/bedrock_client.py` and confirm the only additions are the helper and the two `config=` arguments.

## Decisions and rationale

- **Dedicated `ThreadPoolExecutor` over `loop.set_default_executor`.** The default executor is process-wide and also serves other `to_thread` callers. A dedicated pool scoped to one build keeps the thread count tied to this build's limit and makes shutdown explicit.
- **Pool size = `min(concurrency_limit, 32)`.** Each semaphore slot gets one thread, so no admitted page waits in the pool queue. The cap of 32 follows the brief.
- **`copy_context().run` instead of a bare `run_in_executor`.** It keeps the same context inheritance that `to_thread` provides.
- **`executor=None` keeps the old path.** Modes A and C do not change behavior.
- **Timeouts from env vars with defaults 10s and 240s.** These match the brief. Invalid values fall back to the defaults instead of crashing startup.

## Flags for the implementer (stop and ask, do not change)

1. **Worst-case wall time per dead call goes up, not down.** Today a dead call takes about 300s (botocore's 60s read timeout times 5 legacy attempts, matching the observed 304 to 312s). With `read_timeout=240` and retries unchanged, the same dead call can take up to 5 × 240s, about 20 minutes. A dead page now holds one of the 16 lanes about four times longer. The timeout is not meaningless, because each attempt is bounded, but the total is not bounded without a retry cap. Implement the values as specified. Do not add a retry cap. Report the ~20-minute worst case in the final summary and ask the user to confirm before treating the change as final.
2. **`BedrockLLMClient` (Layer 1 VLM and handwriting, `bedrock_client.py` around lines 676-677) builds the same unconfigured `bedrock-runtime` client.** It is not in scope and was not changed. Its VLM calls can also hang about 300s. Ask whether to apply the same Config there.
3. **Probe client (`src/ai/layer1_probe/vlm_probe.py`)** already has its own Config (`read_timeout=60`, adaptive retries). Left alone, as the brief requires. Its 60s read timeout is shorter than the new 240s for Layer 3, so probe calls can time out on their own.
4. **Connect timeout changes from botocore's 60s to 10s.** Connect failures now fail in 10s. This is intended by the brief, but it is a behavior change.
5. **Throughput risk.** Effective Bedrock concurrency rises from about 6 to 16 for inference profile `24n3l11zddro`. Throttling then becomes more likely, and the app-level throttle loop (4 retries with backoff) absorbs some of it. Throughput quotas for this profile were not checked. Check them before deploying.
6. **Mode C (`extract_with_graph_memory_batched`) and Mode A keep the default pool.** Out of scope, so no change. Note it for the follow-up.
7. **Retry task (deferred).** The retry cap and page-level retry for `FAILED` pages belong to the later task. This plan adds no retry.

## Assumptions

- Pre-existing dirty changes in `extractor.py` and `tests/test_layer3_hardening.py` are kept as they are. The edits in this plan are layered on top.
- The line numbers in the brief are approximate. Locate code by symbol name: `build_page_delta_async`, `extract_with_graph_memory_concurrent`, `_build_page`, `BedrockExtractionClient.client`.
- `graph_concurrency_limit` defaults to 8 in settings and is 16 in the live task definition. The fix reads the same variable the semaphore uses, so both values work the same way.
