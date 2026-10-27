# Probe → Paddle Engine Contract Findings

## Bottom line (summary answer)

**The probe → engine contract is fully mapped and working end-to-end.** The Layer‑1.5 probe
producer emits all four optional keys with the **exact names** the handlers read
(`image_s3_bucket`, `image_s3_key`, `ocr_lang`, `use_structure`), with compatible types. The
Step Functions state machine that includes the probe
(`step_functions/state_machine_with_probe.asl.json`) nests the whole Map item under a `page` key,
which is exactly the **fallback lookup location** (`event["page"][key]`) that both handlers
implement. No top-level copies of the probe keys are passed, so at runtime the four keys are
resolved through the `event["page"]` branch of `_resolve_probe_key`. That branch works correctly,
so the flow is wired.

Two non-breaking observations worth noting (details in Gaps): the probe computes a rich
per-page object but the state machine deliberately forwards the *entire* page object under
`page`, so every probe field is available even though the handler only reads four; and the
legacy state machine (`state_machine.asl.json`, no probe) passes none of these keys, which is
the backward-compatible path the handlers were explicitly designed to tolerate.

---

## Evidence

### 1. Probe producer — what it emits

The per-page output object is the pydantic `PageProbe` model, constructed in `probe.py` Phase C
and serialized via `.model_dump()`.

`PageProbe` field declarations — `src/ai/layer1_probe/types.py:175-195`:

```python
class PageProbe(BaseModel):
    page_number: int
    ...
    ocr_lang: Optional[str] = None            # types.py:180
    use_structure: bool = False               # types.py:181
    ...
    image_s3_bucket: Optional[str] = None      # types.py:192
    image_s3_key: Optional[str] = None         # types.py:193
    image_width: Optional[int] = None
    image_height: Optional[int] = None
```

Construction with real values for probed pages — `src/ai/layer1_probe/probe.py` Phase C
(`results[pn] = PageProbe(...)`):

```python
results[pn] = PageProbe(
    page_number=pn, probed=True, status="ok", route=choice.route, ocr_lang=choice.ocr_lang,
    use_structure=choice.use_structure, route_reason=choice.reason,
    ...
    image_s3_bucket=img_bucket if image_keys.get(pn) else None,
    image_s3_key=image_keys.get(pn), image_width=f["w"], image_height=f["h"],
)
```

- `ocr_lang` / `use_structure` come from `RouteChoice` returned by `decide_route`
  (`src/ai/layer1_probe/routing.py`): `ocr_lang` is a PaddleOCR lang code string from
  `SCRIPT_TO_PADDLE_LANG` (e.g. `"en"`), `use_structure` is `has_table` (bool).
- `image_s3_bucket` is set to the resolved image bucket (`img_bucket = cfg.image_bucket or
  s3_bucket`, `probe.py` Phase A) only when an image key exists, else `None`.
- `image_s3_key` is the uploaded JPEG key `pages/<doc>/<n>.jpg` (built in `_upload`,
  `probe.py` Phase A), or `None` when the page is blank / upload failed / timed out.

The whole per-page list is returned under `pages`:

```python
# probe.py, return value of probe_document()
return {
    "document_id": document_id,
    "s3_bucket": s3_bucket,
    "s3_key": s3_key,
    "doc_prior": prior.model_dump(exclude={"samples"}),
    "pages": pages_out,       # list[PageProbe.model_dump()]
    "stats": stats,
}
```

Documented output contract agrees — `docs/PROBE_INTEGRATION.md` "Output contract (per page…)"
lists `page_number, route, ocr_lang, use_structure, status, script, handwriting_prob, has_table,
image_s3_bucket, image_s3_key`. (Confirmed by reading both the code and the doc.)

> Note on SLIM_OUTPUT: `PROBE_INTEGRATION.md` mentions a slim form and a `SLIM_OUTPUT (1)` env
> var, but `probe.py` as read does **not** implement slimming — it returns the full
> `PageProbe.model_dump()` for every page. This does not affect the contract (the four keys are
> present in both the full and the documented slim form); it is a doc/code drift, noted under Gaps.

### 2. Consumer — what the handlers read

Both handlers are identical in this logic. `docker/paddle_printed/handler.py` and
`docker/paddle_handwritten/handler.py`:

`_resolve_probe_key` (printed `handler.py:31-38`, handwritten `handler.py:31-38`):

```python
def _resolve_probe_key(event, key):
    if key in event:            # top level wins
        return event[key]
    page = event.get("page")
    if isinstance(page, dict):
        return page.get(key)    # fallback: event["page"][key]
    return None
```

Resolution of the four keys (printed `handler.py:56-60`):

```python
image_s3_bucket = _resolve_probe_key(event, "image_s3_bucket")
image_s3_key   = _resolve_probe_key(event, "image_s3_key")
ocr_lang       = _resolve_probe_key(event, "ocr_lang")
use_structure  = _resolve_probe_key(event, "use_structure")  # recorded only, not acted on
```

Engine call on the image path (printed `handler.py:88-92`): passes `lang=ocr_lang` to
`engine.process_image`. Engine signatures confirm the `lang` param exists
(`docker/paddle_printed/engine.py:83`, `docker/paddle_handwritten/engine.py:82`) and so does
`process_pdf_pages` (`:133` / `:132`).

### 3. Nesting / state machine

Two ASL definitions exist in the repo (found via file search, not assumed):

- `step_functions/state_machine_with_probe.asl.json` — the probe-enabled machine.
- `step_functions/state_machine.asl.json` — the legacy, probe-less machine.

In the probe machine, the Map `ItemSelector` sets the per-item payload
(`state_machine_with_probe.asl.json`, `ProcessDocumentPages` → `ItemSelector`):

```json
"ItemSelector": {
  "document_id.$": "$.document_id",
  "s3_bucket.$": "$.s3_bucket",
  "s3_key.$": "$.s3_key",
  "page_number.$": "$$.Map.Item.Value.page_number",
  "route.$": "$$.Map.Item.Value.route",
  "page.$": "$$.Map.Item.Value"          <-- whole probe page object nested under "page"
}
```

Then each Paddle task forwards that `page` object into the Lambda payload
(e.g. `InvokePaddlePrinted` → `Parameters.Payload`):

```json
"Payload": {
  "document_id.$": "$.document_id",
  "s3_bucket.$": "$.s3_bucket",
  "s3_key.$": "$.s3_key",
  "page_number.$": "$.page_number",
  "engine": "paddleocr_printed",
  "page.$": "$.page"                      <-- probe keys arrive here, NOT at top level
}
```

Same `"page.$": "$.page"` forward is present in `InvokePaddleHandwritten`, `InvokeDocling`, and
`InvokeVlmTranscribe`. The four probe keys (`image_s3_bucket`, `image_s3_key`, `ocr_lang`,
`use_structure`) are **not** promoted to the top level of the Lambda payload — only
`page_number` and `route` are lifted out. Therefore at runtime the handler's top-level lookup
(`key in event`) misses and the `event["page"][key]` fallback is what actually returns the
values. **Nesting verdict: keys arrive under `event["page"]`; the handler's fallback branch is
the live path.** (Confirmed by reading the ASL.)

Legacy machine `state_machine.asl.json`: its `ItemSelector` lifts only `document_id, s3_bucket,
s3_key, page_number, route` and its task payloads contain **no** `page` key and none of the four
probe keys. An event from this machine has none of the new keys → handlers take the legacy PDF
path unchanged. This is the intended backward-compatible behavior.

Doc vs state machine agreement: `PROBE_INTEGRATION.md` describes `Map(page)` and routes
`skip | docling | paddleocr_printed | paddleocr_handwritten | vlm_transcribe`, which matches the
`RouteEngineChoice` states in `state_machine_with_probe.asl.json` exactly. No disagreement on the
routing/nesting contract. (The doc's `VLM_LAMBDA_ARN`-does-not-exist caveat matches the ASL,
which references `${VLM_LAMBDA_ARN}` — a known, documented gap unrelated to the Paddle contract.)

---

## Per-key mapping table

| Consumer key (handler) | Producer source | Exact name match | Type match | Verdict |
|---|---|---|---|---|
| `image_s3_bucket` | `PageProbe.image_s3_bucket` (`types.py:192`, set in `probe.py` Phase C) | ✅ exact | `Optional[str]` → handler treats as str, falls back to `s3_bucket` when falsy | ✅ mapped |
| `image_s3_key` | `PageProbe.image_s3_key` (`types.py:193`, set in `probe.py` Phase C) | ✅ exact | `Optional[str]`; `None` on blank/failed/timeout → handler falls back to PDF | ✅ mapped |
| `ocr_lang` | `PageProbe.ocr_lang` ← `RouteChoice.ocr_lang` (`routing.py`) | ✅ exact | `Optional[str]` (e.g. `"en"`); handler normalizes None/"" → "en" in engine | ✅ mapped |
| `use_structure` | `PageProbe.use_structure` ← `RouteChoice.use_structure` (`routing.py`) | ✅ exact | `bool`; handler records only, never acts on it | ✅ mapped |

All four are nested under `event["page"]` by the state machine and resolved via the handler's
`event["page"][key]` fallback branch.

---

## Gaps / mismatches

No contract-breaking mismatches were found. The following are non-breaking observations:

1. **Doc/code drift on SLIM_OUTPUT (cosmetic, non-breaking).** `PROBE_INTEGRATION.md` documents a
   slim per-page output and a `SLIM_OUTPUT` env var, but `probe.py` as read returns the full
   `PageProbe.model_dump()` and has no slimming logic. The four contract keys are present either
   way, so this does not affect the engine contract — but Step Functions state is capped at 256 KB
   and the full object per page is larger than the documented slim form. Worth reconciling, but
   out of scope for this task.

2. **Probe fields the handler ignores (by design, not a defect).** The state machine forwards the
   *entire* page object (`page.$": "$$.Map.Item.Value"`), so fields like `script`,
   `handwriting_prob`, `has_table`, `table_confidence`, `image_width`, `image_height`,
   `script_confidence` are all present in `event["page"]`. The handlers intentionally read only
   the four keys. `use_structure` in particular is resolved but, per the task's hard constraint,
   recorded only and never used to change PP-Structure behavior — consistent with the probe
   emitting it as a hint.

3. **Legacy machine passes nothing (intended).** `state_machine.asl.json` emits none of the four
   keys. Combined with the handlers treating all four as optional, this is the backward-compatible
   path and is working as designed. No action needed.

4. **`image_s3_bucket` can be `None` while `image_s3_key` is set? No.** In `probe.py` Phase C,
   `image_s3_bucket` is set to `img_bucket` *only when* `image_keys.get(pn)` is truthy, and
   `image_s3_key = image_keys.get(pn)`. So whenever `image_s3_key` is non-null, `image_s3_bucket`
   is also non-null — and even if it weren't, the handler falls back to `s3_bucket`
   (`img_bucket = image_s3_bucket or s3_bucket`). No runtime gap.

---

## Conclusion

The producer (probe) and consumer (both Paddle handlers) agree on key names, types, and nesting.
The probe-enabled Step Functions definition nests the page object under `page`, which the handlers
resolve through their documented `event["page"]` fallback. The legacy definition omits the keys,
which the handlers tolerate via the optional/fallback design. The probe → engine contract is
**fully mapped and will work end-to-end at runtime** for the probe-enabled machine, and remains
backward compatible for the legacy machine.

No code changes are recommended as part of this contract check. If desired as separate follow-ups
(out of scope here): reconcile the SLIM_OUTPUT doc/code drift to keep Map state under 256 KB, and
provision the `${VLM_LAMBDA_ARN}` the ASL references (a pre-existing, documented gap unrelated to
the Paddle contract).

*Citations verified by reading each file listed. "Inferred" items (runtime resolution path) are
inferred from the ASL ItemSelector/Payload structure combined with the handler's
`_resolve_probe_key` logic; everything else is confirmed directly from file contents.*
