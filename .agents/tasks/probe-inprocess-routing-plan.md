# Implementation Plan — Per-document probe drives in-process routing (Option B)

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a` · Branch: `feat/llm-observability` · NO worktree.
Python for ALL commands: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`.

Ground truth: `.agents/tasks/probe-inprocess-routing-findings.md` (decision = Option B). This plan is
grounded in the actual files read: `probe.py`, `inprocess.py`, `pipeline.py`, `scanned.py`,
`vlm_probe.py`, `routing.py`, `types.py`, `aggregate.py`, `settings.py`, `inspect.py`, and the existing
tests `tests/test_probe.py`, `tests/test_pipeline_probe_integration.py`.

## Absolute constraints (must hold for every item)
- Edits ONLY in: `src/ai/layer1_routing/pipeline.py`, `src/ai/layer2_conversion/scanned.py`,
  `src/config/settings.py` (one flag), `src/ai/layer1_probe/*`, and NEW test files.
- NO git commit/stage; never stage/commit/track/delete any `.onnx`. No edits under `docker/` or `step_functions/`.
- Keep the dormant per-page hook: `inprocess.probe_page_route`, `pipeline._maybe_probe_route`
  (seam pipeline.py ~444-453), `_probe_enabled` (`IDP_ENABLE_PROBE`), and
  `tests/test_pipeline_probe_integration.py` all stay. If a page's `page_context` has a `"probe"` dict,
  SKIP `_maybe_probe_route` for that page; otherwise old behavior is unchanged.
- Do NOT change the escalation ladder, confidence thresholds, or `routing_mode` logic.
- Do NOT touch `src/api/lambda_probe_handler.py`; it must keep working via the refactored shared core.
- Byte-identical when `settings.probe_enabled` is False.
- `RouteChoice.route` is ALREADY an engine-name string — use it DIRECTLY as the `EngineTask.engine`.
  Do NOT remap through `_ROUTE_TO_ENGINE`.

## Baseline (before)
Collected: `tests/test_probe.py` + `test_pipeline_probe_integration.py` + `test_probe_linewise.py` +
`test_probe_script_onnx.py` = **57 tests** (one `@pytest.mark.slow` real-model test skips without the model).
Record the full before count in the final report via the Verify command in item 9.

---

- [ ] 1. Add the `probe_enabled` settings flag.
      In `src/config/settings.py`, add `probe_enabled: bool = False` into the Core-routing-thresholds
      block right after `enable_vlm_direct` (around line 341), matching the existing pydantic-settings
      style. With `env_prefix="IDP_"` this reads `IDP_PROBE_ENABLED` automatically. No `_fallback_unprefixed_env`
      entry is required (only prefixed env needed).
      Files: `src/config/settings.py`
      Verify: `.venv\Scripts\python.exe -c "from src.config.settings import settings; print(settings.probe_enabled)"` prints `False`.

- [ ] 2. Refactor `probe.py` into a shared Phase-B/C core without changing `probe_document` behavior.
      Extract the per-document script-prior + per-page decision logic that currently lives inline in
      `probe_document` into internal helpers that operate on in-memory images and signals, so BOTH the
      Lambda path and the new in-process path share it (lipika-vs-vlm telemetry and the VLM-overrides-lipika
      rule live ONLY in the shared core). Concrete shape:
        - Keep `_observe(...)` logic but move it behind a factory that closes over `(cfg, vlm, onnx, stats)`
          so it is reusable. Keep the exact telemetry (`logger.info("probe.lipika_vs_vlm", ...)`) and the
          confident-latin extra-VLM / VLM-overrides-latin seam byte-for-byte.
        - Add `_sample_and_aggregate(nonblank, jpegs, cfg, vlm, onnx, stats) -> DocPrior` wrapping the
          `pick_sample_pages` → `samples_disagree`/`additional_sample_pages` → `aggregate_doc` block
          (currently inline lines ~ "if nonblank and not cache_hit").
        - Add `_decide_pages(candidates, nonblank, timed_out, feats, prior, cfg) -> dict[int, PageProbe]`
          wrapping Phase C (nearest-sample inheritance via `_page_values` + `decide_route` + blank + timeout).
        - `probe_document` keeps its full signature and still: opens fitz, renders, uploads to S3,
          builds `feats`/`jpegs`, then calls the shared helpers. Its returned dict/stats stay identical.
      Files: `src/ai/layer1_probe/probe.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe.py -q` — all pass unchanged (Lambda path intact).

- [ ] 3. Add `probe_document_images(...)` in `probe.py` (in-memory entry point, no fitz / no S3).
      Signature: `def probe_document_images(images, page_numbers, *, profiles=None, cfg, vlm=None, onnx=None, time_left=None) -> dict[int, PageProbe]`.
      `images`: mapping `page_number -> RGB numpy array` OR encoded bytes (PNG/JPEG). Decode bytes via
      `cv2.imdecode` + `cv2.cvtColor(..., COLOR_BGR2RGB)` (reuse `_gray_mask_from_jpeg` style) when not an ndarray.
      Build Phase-A signals per image IN MEMORY (`imaging.analysis_gray` + `imaging.ink_mask` +
      `cv_features.ink_ratio` for blank + `cv_features.detect_table`), and produce a JPEG per page via
      `imaging.encode_jpeg` ONLY to feed the VLM/observer `_observe` path (no S3 put, no upload pool,
      `image_s3_*` left None). Then call the SAME `_sample_and_aggregate` and `_decide_pages` helpers from
      item 2. `candidates` = the `page_numbers` passed in (the caller has already filtered with `needs_probe`);
      `profiles` is optional and only used to set `PageProbe.script`/source on not-needed passthroughs if any.
      Return `{page_number -> PageProbe}`.
      Files: `src/ai/layer1_probe/probe.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe.py -q` still green; plus a new unit test
      (item 10) drives `probe_document_images` directly with numpy + PNG/JPEG bytes and a StubVLM/onnx=None.

- [ ] 4. Magic-byte image-format detection at the VLM boundary (`vlm_probe.py`).
      In `BedrockScriptProbe._call`, detect format from the leading bytes: `b"\x89PNG"` → `"png"`,
      `b"\xff\xd8\xff"` → `"jpeg"`, anything else/ambiguous → `"jpeg"` (preserves the Lambda path which feeds
      real JPEG). Pass the detected string into the Converse `{"image": {"format": <fmt>, ...}}` block.
      Keep raw bytes (no decode/re-encode). Rename the param mentally (still `bytes`); keep `observe(page_number, jpeg)`
      signature unchanged so callers/tests are untouched.
      CHOSEN APPROACH: magic-byte detection (lighter than re-encode) — report this in the final summary.
      Files: `src/ai/layer1_probe/vlm_probe.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe.py::test_bedrock_probe_normalises_and_survives_garbage -q`
      passes; plus new unit test (item 10) asserts format string per magic bytes.

- [ ] 5. Per-lang caching in `_get_paddle_engine` (`scanned.py`).
      Change signature to `_get_paddle_engine(mode: str = "printed", lang: str = "en")`. Key `_paddle_engines`
      by `(mode, lang)` instead of `mode`; pass `lang=lang` into both `PaddleOCR(...)` constructors. In
      `_ocr_convert`, pass the resolved lang into the local-fallback call: `engine = _get_paddle_engine(mode=mode, lang=(ocr_lang or "en"))`.
      Leave `low_confidence_word_ratio`'s `_get_paddle_engine(mode="printed")` call as-is (defaults to `"en"`).
      Do NOT touch any other ocr_lang plumbing (already done per findings 1.4).
      Files: `src/ai/layer2_conversion/scanned.py`
      Verify: `.venv\Scripts\python.exe -c "import src.ai.layer2_conversion.scanned as s, inspect; print(inspect.signature(s._get_paddle_engine))"`
      shows `(mode: str = 'printed', lang: str = 'en')`; imports clean (no paddleocr import triggered).

- [ ] 6. `process_document`: run the per-document probe ONCE over in-memory images (flag-guarded, try/except).
      In `src/ai/layer1_routing/pipeline.py::process_document`, BEFORE the page loop, add a guarded block:
      `if settings.probe_enabled:` → for each `page_data`, compute `profile = inspect_page(page_data["page"], page_data["page_number"])`,
      derive the Layer-1 route `route_from_profile(profile)` (None allowed), and call
      `probe.needs_probe(profile.model_dump(), route, cfg)` (cfg = `ProbeConfig.from_env()`). Collect the
      page_numbers that need probing and their images from `page_data["context"]["image_array"]`
      (fallback: decode `page_data["image_bytes"]`). If ANY page needs probing, wrap in try/except:
        - Build the VLM client + onnx observer the mockable way: `vlm = BedrockScriptProbe(model_id=cfg.vlm_model_id, region=cfg.vlm_region)` when `cfg.vlm_enabled` and a model id exists else None; `onnx = inprocess._get_observer()` (singleton, swappable in tests). Keep these behind a small module-level indirection (e.g. `_build_probe_deps(cfg)`) so tests can monkeypatch.
        - Call `decisions = probe.probe_document_images(images, page_numbers, profiles=..., cfg=cfg, vlm=vlm, onnx=onnx)`.
        - On success, for each probed page whose `PageProbe.probed`/status is routable (not `not_needed`),
          set `page_data["context"]["probe"] = {route, ocr_lang, script, script_confidence, handwriting_prob, has_table, is_blank, route_reason}`.
        - On ANY exception: `logger.error("probe.failed", document_id=document_id, error=str(exc))` (or
          `logger.warning` to match existing style — use `logger.error` with event name `probe.failed`) and
          attach NO `"probe"` keys (every page falls back to today's routing). Never abort.
      Pages the probe marks `not_needed`/digital get NO `"probe"` key. Do NOT re-render or touch S3.
      Files: `src/ai/layer1_routing/pipeline.py`
      Verify: covered by items 7 + 11 integration tests (flag on/off, probe-raises).

- [ ] 7. `process_page`: consume `page_context["probe"]` at the existing seam.
      In `src/ai/layer1_routing/pipeline.py::process_page`, at the probe seam (currently lines ~444-453,
      right after `primary_task = plan[0] ...`), change to:
      `if "probe" in page_context:` → build `primary_task = EngineTask(engine=page_context["probe"]["route"], priority=1, reason=f"probe:{page_context['probe'].get('route_reason','')}")`;
      `plan = [primary_task]`; if `page_context["probe"].get("ocr_lang")` set `page_context["ocr_lang"]`;
      record the full dict in `metadata.extra["probe"] = dict(page_context["probe"])`; and SKIP
      `_maybe_probe_route` for this page. `else:` keep the EXISTING block
      (`if primary_task is not None: _choice = _maybe_probe_route(...)`) UNCHANGED. Use `probe.route` directly
      (no `_ROUTE_TO_ENGINE`). A `"skip"` route flows through the existing skip engine handling. The escalation
      ladder below runs unchanged.
      Files: `src/ai/layer1_routing/pipeline.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_probe_integration.py -q` — existing
      dormant-hook tests still pass (flag off path + `IDP_ENABLE_PROBE` path unchanged).

- [ ] 8. Regression check of the whole existing probe surface.
      Run the existing probe + pipeline suites to confirm the refactor (items 2-7) is behavior-preserving
      with the flag OFF and `IDP_ENABLE_PROBE` semantics intact.
      Files: (none)
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_pipeline_probe_integration.py tests/test_probe_linewise.py tests/test_probe_script_onnx.py -q`
      — same pass count as baseline (57 collected; slow test skips).

- [ ] 9. NEW test file for the shared-core / vlm_probe / scanned unit behavior:
      `tests/test_probe_inprocess_images.py`. Mock the Bedrock client (StubVLM), inject onnx (fake or None);
      NO Paddle, NO AWS, NO torch. Cover:
        - `probe_document_images` with numpy RGB inputs → per-page PageProbe (latin→paddleocr_printed+en,
          non-latin→vlm_transcribe, blank→skip).
        - `probe_document_images` with PNG bytes and with JPEG bytes inputs decode correctly and route the same.
        - Magic-byte format detection in `BedrockScriptProbe._call`: feed a fake client that records the
          `format` field; assert `"png"` for PNG bytes, `"jpeg"` for JPEG bytes, `"jpeg"` for ambiguous.
        - VLM probe CALL COUNT in a multi-page latin doc == `n_samples` (3); and when samples disagree
          (StubVLM `per_page` map returns latin vs devanagari), more calls happen (widening) and non-sampled
          pages inherit the nearest sample.
        - `_get_paddle_engine(mode, lang)` caches a distinct engine per `(mode, lang)` — assert via a
          monkeypatched fake `PaddleOCR` constructor recording `lang`, without importing real paddleocr.
      Files: `tests/test_probe_inprocess_images.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_probe_inprocess_images.py -q` — all new tests pass.

- [ ] 10. NEW integration test file for `process_document`/`process_page` wiring:
      `tests/test_pipeline_probe_document.py`. Reuse the stub/fixture patterns from
      `tests/test_pipeline_probe_integration.py` (StubLLM, FakeObserver via `inprocess._OBSERVER`,
      monkeypatch `pipeline.inspect_page`, monkeypatch `scanned._invoke_paddle_lambda` to capture payload,
      monkeypatch the probe deps builder so no real Bedrock/boto). Build `pages` lists of page_data dicts
      with `context={image_array, image_bytes}`. Cover:
        - flag OFF: no `"probe"` key attached, `_maybe_probe_route` path unchanged, results identical to the
          existing integration tests (assert `metadata.extra` has no `"probe"`).
        - flag ON, scanned Latin printed page: plan is ONE `paddleocr_printed` task; mocked Lambda payload
          carries `ocr_lang == "en"`.
        - flag ON, VLM returns latin + handwriting_prob 0.9 (confident-latin extra-VLM path): routes to
          `paddleocr_handwritten`.
        - flag ON, non-Latin (telugu/devanagari): routes to `vlm_transcribe`; assert escalation ladder NOT
          re-entered for that page (escalation_attempts == 0 / no second engine).
        - flag ON, blank page: `skip` engine, page skipped.
        - MIXED DOCUMENT: digital(Docling) + printed scan + blank + table scan → per-page engines
          `docling, paddleocr_printed, skip, paddleocr_printed`; `use_structure` only RECORDED in
          `extra["probe"]`/PageProbe, not acted on. Assert engines differ per page.
        - HETEROGENEOUS DOCUMENT: sampled pages disagree → assert widening happened (VLM call count >
          n_samples) and non-sampled pages inherit nearest sample route.
        - probe RAISES: monkeypatch `probe.probe_document_images` to raise → every page falls back to today's
          routing, `probe.failed` logged (capture via caplog/logger), document completes with all pages.
        - digital pages never probed (no `"probe"` key; needs_probe False) and output unchanged.
        - BOTH flags on (`IDP_ENABLE_PROBE` + `probe_enabled`): new probe WINS; assert `_maybe_probe_route`
          NOT called for probed pages (spy/monkeypatch `pipeline._maybe_probe_route` and assert 0 calls on
          probed pages).
        - Assert VLM probe call count per document ~= n_samples (3) in the sampling tests.
      Files: `tests/test_pipeline_probe_document.py`
      Verify: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_probe_document.py -q` — all pass.

- [ ] 11. Full fast-suite run + report (NO slow real-engine tests).
      Run the whole fast suite EXCLUDING `tests/test_layer2_engines_local.py::*_engine_local_path` (real
      Docling/Paddle) and the `@pytest.mark.slow` real-model test. Report before/after counts, per-file
      change summary, VLM calls added per document (~n_samples ≈ 3; more on disagreement), and anything awkward.
      Files: (none)
      Verify: `.venv\Scripts\python.exe -m pytest tests -q --deselect tests/test_layer2_engines_local.py::test_docling_engine_local_path --deselect tests/test_layer2_engines_local.py::test_paddle_printed_engine_local_path --deselect tests/test_layer2_engines_local.py::test_paddle_handwritten_engine_local_path -m "not slow"`
      — green; after-count = before-count + new tests from items 9 and 10.

## Notes / awkward spots discovered
- `process_document` does not currently inspect pages; item 6 adds a per-page `inspect_page` + `route_from_profile`
  pass ONLY to compute `needs_probe`. This duplicates the inspect that `process_page` does again (inspect is
  ~10ms, no S3, no behavior change) — acceptable and keeps `process_page` untouched for non-probed pages.
- `needs_probe` takes a profile **dict**, so pass `profile.model_dump()`.
- `_maybe_probe_route` must be skipped only for pages that actually received a `"probe"` dict; pages the
  per-document probe marks `not_needed` get no key and keep the old path (incl. `IDP_ENABLE_PROBE`).
- VLM format bug is only reachable under this design (PNG reaches the probe); magic-byte detection chosen.
