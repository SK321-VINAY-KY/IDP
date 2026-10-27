# Paddle engines consume probe image/lang output, backward-compatible

Both PaddleOCR engines (printed + handwritten) and their Lambda handlers now understand the Layer 1.5 probe contract: four optional event keys (`image_s3_bucket`, `image_s3_key`, `ocr_lang`, `use_structure`), resolved from the event top level or from a nested `event["page"]` object. When a probe image key is present the handler OCRs the pre-rendered JPEG from S3 instead of downloading the PDF; any failure logs and falls through to the existing PDF path. The single PaddleOCR singleton was replaced with a module-level dict cache keyed by normalized language, and each result dict gained `ocr_lang` and `image_source`. The change is confined to the five intended files.

Watch for: nothing blocking. The sentinel-based fallback preserves the old code path exactly when no new keys are present (confirmed). `use_structure` is resolved but intentionally inert (confirmed). After-run suite evidence was not recorded by the coder (confirmed) — a single spot-check of the new mocked test file was run to close the gap and all 10 tests pass.

**Verdict**: APPROVED

## High-level view

The handler change hinges on a `results = None` sentinel inserted at the top of the existing `try` block. The probe image branch runs first and only sets `results` on success; the pre-existing base64/S3-PDF/local chain is now guarded by `if results is not None: pass elif ...`, so an event carrying none of the new keys leaves `results` as `None` and lands in the identical `elif "image_base64"` → `elif s3 ...` ladder it used before. The success body stays a dict and the error body stays a `json.dumps` string. Backward compatibility holds.

Key resolution is a small helper `_resolve_probe_key` that returns `event[key]` when present, else `event["page"][key]` when `page` is a dict, else `None`. Top-level precedence is correct, and the helper is byte-identical across both handlers.

The engine change swaps `_paddle_engine: Any = None` for `_paddle_engines: Dict[str, Any] = {}` and makes `get_paddle_engine(lang=None)` normalize the lang (None/""→"en") and cache one instance per lang. Constructor kwargs are untouched except `lang`; the handwritten engine keeps its distinguishing `text_det_thresh=0.2`, and `ocr_version="PP-OCRv6"` is unchanged on both. There is no PPStructureV3 code in the printed engine — there never was — so the "PPStructureV3 for en only" instruction is vacuously satisfied by building a plain `PaddleOCR` per lang, matching the reconciled note in the plan. No `use_table_recognition` setting exists to change.

Results gained exactly two keys (`ocr_lang`, `image_source`) in the success dict, the error dict, and the out-of-range dict, so the schema is uniform and no existing key was renamed or dropped.

Tests are a new fully-mocked file covering all five required cases, parametrized over both engine/handler pairs. They never load a real model and finish in under a second.

<details>
<summary>Issues (2)</summary>

1. **After-run suite evidence not recorded** (non-blocking) — the coder saved a before-baseline (`baseline_test_results.txt`, 8 passed) but no after-run of `test_layer2_engines_local.py` and no recorded run of the new mocked file. A one-time spot-check of `tests/test_engines_probe_inputs.py` was run during review (10 passed, 0.89s) to close the doubt; the task's "report before/after" deliverable is still thinner than requested but the code is verified green.
2. **`use_structure` / `image_s3_bucket` assigned-but-mostly-unused in handwritten handler** (informational) — `use_structure` is deliberately inert per spec (comment says "recorded only, not acted on"); it is resolved but neither logged nor stored. Harmless, matches the requirement that it only be recorded.

</details>

<details>
<summary>Details</summary>

### Sentinel fallback keeps the legacy path byte-for-byte

The one behavioral risk in this change is whether a legacy event (no probe keys) still takes the exact old path. It does. The new code inserts `results = None` as the first statement in the `try`, runs the probe image branch only when `image_s3_key` is truthy, and then gates the entire original ladder behind `if results is not None: pass elif ...`. With no new keys, `image_s3_key` is `None`, the probe branch is skipped, `results` stays `None`, and control reaches `elif "image_base64" in event ...` — the same first rung as before. The 400 guard for missing pages, the success payload shape (`body` is the dict `response_payload`), and the 500 handler (`body` is `json.dumps(...)`) are all unchanged. The dict-vs-string body asymmetry the task calls out is preserved (confirmed in both handlers).

When the probe image path *fails*, the `except Exception as img_exc` block logs via `print(...)` and resets `results = None`, so the fallback re-enters the same ladder. This covers a missing key (branch never entered), an S3 fetch error, and a PIL decode error uniformly — all three fall back to the PDF path as required (confirmed; test `test_image_fetch_error_falls_back_to_pdf` exercises the S3 error case for both engines).

### Image path wiring

```
image_s3_key present
   └─► get_object(Bucket = image_s3_bucket or s3_bucket, Key = unquote_plus(image_s3_key))
          └─► PIL.open(BytesIO).convert("RGB") ─► np.array
                 └─► engine.process_image(img, pages[0] if pages else 1,
                                          lang=ocr_lang, image_source="s3_image")
```

Bucket fallback (`image_s3_bucket or s3_bucket`), key unquoting, RGB decode, and the `process_image` call with `image_source="s3_image"` all match the spec. `target_page` is derived from the event's `pages`/`page_number` (`pages[0] if pages else 1`), so page numbering still comes from the event as today. Test `test_image_key_path_used_pdf_not_downloaded` asserts `get_object` is called exactly once with the image key and bucket, that `process_pdf_pages` is never called, and that `process_image` received `image_source="s3_image"` and `lang="fr"` — a clean proof the PDF is not downloaded on the image path.

### Lang-keyed engine cache

`get_paddle_engine(lang=None)` normalizes via `_normalize_lang` (None or whitespace-only → "en") and memoizes one `PaddleOCR` per normalized lang in `_paddle_engines`. The constructor kwargs are identical to the prior singleton except `lang=norm_lang`; the handwritten engine retains `text_det_thresh=0.2` and both keep `ocr_version="PP-OCRv6"`, `device="cpu"`, `enable_mkldnn=False`, and the orientation/unwarp flags. `process_image` normalizes once and passes the normalized lang down; `process_pdf_pages` normalizes once and threads `lang=norm_lang, image_source="pdf"` into every per-page `process_image` call. Test `test_lang_cache_constructs_once_per_lang` injects a fake `paddleocr` module and asserts two `lang="en"` calls construct once while `lang="fr"` triggers a second construction — the cache-per-lang contract, verified without the real package.

### Result schema uniformity

`ocr_lang` (the normalized lang actually used) and `image_source` were added to the success dict, the exception dict, and the out-of-range-page dict in `process_pdf_pages`, so every result dict carries the two new keys regardless of path. No existing key (`page_number, engine, markdown, confidence, lines_extracted, blocks, latency_ms, status`, plus `error` on the error path) was renamed or removed. Test `test_missing_keys_uses_pdf_path_with_new_keys` asserts the original key set is a subset of the result and that `image_source == "pdf"` on the PDF path.

### Test coverage

The new `tests/test_engines_probe_inputs.py` is parametrized over both engine/handler pairs and covers all five required cases: image-key path with PDF-not-downloaded assertion, keys nested under `event["page"]`, missing-keys→PDF with the two new keys present, S3 image error→PDF fallback, and engine-level lang caching. All mocks follow the existing `patch.object(handler, "get_s3_client", ...)` style; the lang-cache test injects a fake `paddleocr` into `sys.modules` so no real model loads. Spot-check run during review: 10 passed in 0.89s.

Not tested: the PIL decode-failure branch specifically (only the S3 get_object failure is exercised); both share the same `except` so the fallback is covered in effect, but a corrupt-image-bytes case is not asserted directly. Non-blocking.

### Scope

The task commit touches only `docker/paddle_printed/{engine,handler}.py`, `docker/paddle_handwritten/{engine,handler}.py`, and the new `tests/test_engines_probe_inputs.py`. No Dockerfile, requirements file, or step_functions JSON appears in the commit. (The working tree contains many other unrelated modified/untracked files from parallel work, but none are part of this task's commit.)

</details>

<details>
<summary>File map</summary>

- `docker/paddle_printed/engine.py` — singleton → lang-keyed dict cache; `_normalize_lang` helper; `lang`/`image_source` params on `process_image`, `lang` on `process_pdf_pages`; two new result keys on all three result dicts.
- `docker/paddle_printed/handler.py` — `_resolve_probe_key` helper; probe-key resolution; image-S3-first branch with sentinel fallback to the unchanged PDF chain.
- `docker/paddle_handwritten/engine.py` — identical to printed engine changes; retains `text_det_thresh=0.2`.
- `docker/paddle_handwritten/handler.py` — identical to printed handler changes; engine name `paddleocr_handwritten`.
- `tests/test_engines_probe_inputs.py` — new, 10 parametrized fully-mocked tests covering the five required cases.

Full diff: `git show de95e3d`.

</details>
