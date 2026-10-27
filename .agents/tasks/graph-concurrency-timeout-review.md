# Graph LLM thread pool and Bedrock read/connect timeouts

This change gives the concurrent graph extractor its own thread pool, sized from the same `graph_concurrency_limit` that feeds the page semaphore, so every admitted page gets a worker thread instead of waiting in the default asyncio pool (about 6 threads on the 2-vCPU task). It also puts explicit connect and read timeouts on the Layer 3 `bedrock-runtime` client. Retries are untouched, as the user asked. The executor path sits behind an `executor=None` default, so Modes A and C keep their current behavior. The tests cover concurrency reaching the limit, pool shutdown, pool sizing, unchanged merge order, correlation ID propagation, and the Config values and env overrides.

**Watch for:**
- **Confirmed:** with retries unchanged, a dead Bedrock endpoint can hold a page for up to 5 × 240 s, about 20 minutes, where it took about 5 minutes before. The user deferred retries, so the user needs to accept this explicitly.
- **Likely:** if the build-phase `gather` raises, the `with` block calls `shutdown(wait=True)` on the event loop thread, which blocks the loop until in-flight LLM calls finish.

**Verdict**: APPROVED

## High-level view

The executor change is contained to the build phase of `extract_with_graph_memory_concurrent`. The semaphore still controls admission, and the pool size follows the same limit, so the two stay in step. The one new edge is the failure path: an exception that escapes the page wrapper now blocks the event loop while the pool drains.

The timeout change is one helper applied to the Layer 3 Bedrock client. The 240 s read timeout suits normal graph calls, but botocore's default of 5 attempts still applies, so the wait for a dead endpoint grows about fourfold. The missing piece is a retry cap, which is the deferred task.

<details>
<summary>Issues (5)</summary>

1. **Retry worst case for dead Bedrock calls** — with `read_timeout=240` and botocore's default 5 attempts, a dead endpoint can block a page for about 20 minutes. Get explicit confirmation from the user, or cap retries in the deferred retry task before relying on this in production.
2. **Event loop blocked on build-phase failure** — when `gather` raises, the `ThreadPoolExecutor` context exit runs `shutdown(wait=True)` on the loop thread. Move the shutdown off the loop, for example with `await loop.run_in_executor(None, llm_executor.shutdown, True)` in a `try/finally`, so the loop keeps running while the pool drains.
3. **Non-finite timeout env values accepted** — `float("inf")` and `float("nan")` pass the `<= 0` check in `_env_seconds`, so `IDP_BEDROCK_READ_TIMEOUT=inf` disables the read timeout. Add a `math.isfinite` check that falls back to the default.
4. **Read timeout may be tight for large graph responses** — the graph call budget is 8000 tokens, with a retry ceiling of 12000 tokens. A non-streaming Converse response on a slow model can take longer than 240 s to produce. Measure output throughput before relying on the default.
5. **No test for the build-phase exception path** — the executor tests cover success and shutdown only. Add a test where a page raises outside the `build_page_delta_async` try block, and assert the call returns or raises promptly with no leaked `graph-llm` threads.

</details>

<details>
<summary>Details</summary>

### Pool sizing and where the pool lives

The pool is created inside `extract_with_graph_memory_concurrent`, right before the build-phase `asyncio.gather`. Its size is `min(concurrency_limit, 32)`, where `concurrency_limit` is the value the `asyncio.Semaphore` uses. The default asyncio executor is `min(32, cpu_count + 4)`, which is about 6 on the 2-vCPU task, so 16 admitted pages would queue behind 6 threads without this change. The 6-thread figure comes from that formula and the vCPU count, not from a measurement on the running task.

```
extract_with_graph_memory_concurrent
  semaphore(limit) ── admits pages
  with ThreadPoolExecutor(min(limit,32), "graph-llm") as llm_executor:
      gather(_build_page(p) for p in pages)
          _build_page: semaphore -> build_page_delta_async(executor=llm_executor)
                         -> run_in_executor(llm_executor, copy_context().run(llm.extract))
  <pool joined>
  serial merge in sorted page order -> checkpoints -> reconcile -> link
```

The `copy_context().run` wrapper keeps the logger's `correlation_id` visible in the worker thread. `run_in_executor` does not copy context on its own, so without the wrapper the LLM call's log lines would lose their correlation ID.

### Exception path and the event loop

Inside `_build_page`, the LLM call and the validation step are wrapped, but the context fetch and the `PROCESSING` checkpoint write are not. A database error in one of those propagates out of `gather` while sibling tasks are still awaiting threads. The `with` block's exit then calls `shutdown(wait=True)` on the loop thread. The loop cannot run any other coroutine until the in-flight LLM calls return, and those can take minutes. At HEAD, the same exception left the sibling tasks pending, and asyncio's teardown cancelled them without blocking the loop. The merged graph is not affected, but the failure now takes up to one in-flight LLM call to surface.

### Bedrock timeout Config

`_bedrock_runtime_config()` builds a botocore `Config` with `connect_timeout` (default 10 s, env `IDP_BEDROCK_CONNECT_TIMEOUT`) and `read_timeout` (default 240 s, env `IDP_BEDROCK_READ_TIMEOUT`). Both `client` construction branches pass it. The helper sets no `retries` or `max_attempts`, and the only occurrence of "retries" in the file is its docstring.

Before this change, botocore used its defaults: 60 s connect, 60 s read, and 5 legacy-mode attempts. The report matches that to the observed 304 to 312 s dead-call duration. With the new values and retries unchanged, the dead-call worst case becomes 5 × 240 s, about 20 minutes. Connect failures now fail in 10 s instead of 60 s. Both changes are what the brief asked for. The 20-minute worst case is the main thing to accept or reject before deployment.

The env parser handles blank, non-numeric, zero, and negative values by logging `bedrock.config.invalid_timeout` and using the default. It does not reject `inf` or `nan`, which parse as floats and pass the `<= 0` check.

The 240 s read timeout is set against a graph-call budget of 8000 tokens, with a ceiling retry at 12000 tokens (`settings.py` lines 76-80). Converse is non-streaming, so the socket stays quiet until the whole response is generated. Whether 12000 tokens finishes inside 240 s depends on the model's output speed, which the repo does not record.

### Test coverage

The executor tests (`tests/test_graph_concurrency_executor.py`) show the first 16 calls are in flight together against a 6-thread default pool, confirm no `graph-llm` thread survives a normal run, check pool sizing at limits 4 and 64 (capped at 32), compare a reversed-completion run to a serial run, and check correlation ID propagation.

Not tested: the build-phase exception path (issue 5), and a page cancelled mid-call. Modes A and C are unchanged, and their existing suites cover them.

The Bedrock tests (`tests/test_bedrock_client_timeouts.py`) replace `boto3` with a recorder and check the Config on both branches, the env overrides, the fallback for `abc` and `0`, and the absence of a `retries` kwarg. They do not test `inf` or `nan` (issue 3), and they do not exercise real botocore retry behavior against a network.

The implementer's evidence reports the targeted suites passing except `test_schema_resolution_fuzzy_candidate_must_not_populate_empty_field`, which also failed before the change. I did not re-run the suites. The full `tests/` run did not finish in their environment, so there is no full-suite before and after comparison.

### Out of scope, observed in the working tree

Several files were already dirty before this task and are not part of these hunks. These are three `owner=` lines in `extractor.py`, `.dockerignore`, `tests/test_layer3_hardening.py`, and the probe files. The `owner=` lines are valid because `save_document_graph` accepts `owner`.

The report's note about `test_ecs_tags_are_mandatory_agents_tags` is real. `scripts/deploy_probe_service.py` sets `createdby` to `vinay.k@shellkode`, while AGENTS.md requires `vinay.k@shellkode.com`. That script is untracked and outside this diff, so it is not a finding here, but the user should fix it.

No deploy script, Docker, Step Functions, ONNX, IAM, or `vlm_probe.py` file changed in this task, and nothing was committed. HEAD is still `acd90d1`.

</details>

## File map

<details>
<summary>Files changed in this task</summary>

- `src/ai/layer3_extraction/graph_agent/agent.py`: optional `executor` parameter on `build_page_delta_async`, with `copy_context().run` wrapper and `run_in_executor` path.
- `src/ai/layer3_extraction/extractor.py`: dedicated `graph-llm` pool around the concurrent build-phase `gather`, passed into `_build_page`. The three `owner=` lines in the diff were already dirty.
- `src/adapters/llm/bedrock_client.py`: `_bedrock_runtime_config()` helper and `config=` on both `bedrock-runtime` client constructions.
- `tests/test_graph_concurrency_executor.py` (new): five executor tests.
- `tests/test_bedrock_client_timeouts.py` (new): timeout Config tests.

Full diff: `git diff` on `feat/llm-observability` against HEAD `acd90d1`, plus the two untracked test files.

</details>
