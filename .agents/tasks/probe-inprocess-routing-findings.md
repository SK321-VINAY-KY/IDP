# Probe → in-process routing: READ-ONLY integration findings

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a`  ·  Branch: `feat/llm-observability` (HEAD `705fe37`)
Mode: strictly read-only. No edits, no builds, no tests, no AWS were run.
Legend: **[C]** = confirmed by reading the file  ·  **[I]** = inferred.

---

## BOTTOM LINE (read this first)

**The plan does NOT match the code as written — and the mismatch is not a small gap. A different in-process probe integration already exists on this branch, committed and tested, and it uses a fundamentally different model than the one the user's request describes.** This is the STOP-AND-ASK.

- The user's request describes wiring the **per-document VLM-sampling probe** (`probe_document` / `needs_probe`) into `process_document`, running it **once per document**, attaching `page_context["probe"] = {...}`, reading that in `process_page`, and gating it with a **new `settings.probe_enabled` (env `IDP_PROBE_ENABLED`)** flag.
- What is actually on the branch (commit `4d3af94` "script-only lipika ONNX probe + in-process pipeline wiring", plus `de95e3d`, `705fe37`) is a **per-page ONNX-observer hook**: `process_page` calls `_maybe_probe_route(...)` → `src/ai/layer1_probe/inprocess.py::probe_page_route(...)`, which runs the **ONNX line-wise observer only** (no VLM, no per-document sampling, no `probe_document`), gated by **`IDP_ENABLE_PROBE`** (an `os.getenv`, **not** a `settings` field).

So before any editing, the user must decide **which of two integrations is the target**:
- **(A)** Keep/extend the existing per-page ONNX hook (then most of the request's items are already partly done, but the VLM-per-document behaviour, `page_context["probe"]`, `extra["probe"]`, and the `settings.probe_enabled` name are all absent); or
- **(B)** Replace it with the per-document VLM-sampling design the request describes (then the existing `inprocess.py`/`_maybe_probe_route` wiring and its tests must be reconciled or removed — the request says "do not delete the standalone probe Lambda code" but says nothing about the already-wired in-process hook).

Everything below is the evidence. The decision points are collected at the end.

---

## 1. Per-file findings (signatures + line refs)

### 1.1 `src/ai/layer1_probe/inprocess.py` — the existing in-process entry **[C]**

Public function (lines 31–52):

```python
def probe_page_route(page_context: dict, page_number: int, profile: Any, cfg: ProbeConfig,
                     *, render_dpi: Optional[int] = None) -> Optional[RouteChoice]:
    render_dpi = render_dpi or cfg.page_image_dpi
    rgb = page_context.get("image_array")        # line 35
    if rgb is None:
        return None
    gray = imaging.analysis_gray(rgb, render_dpi, cfg.analysis_dpi)
    mask = imaging.ink_mask(gray)
    blank = cv_features.ink_ratio(mask) < cfg.blank_ink_ratio
    has_table, _ = cv_features.detect_table(...)
    obs = _get_observer().observe(page_number, gray, mask, rgb=rgb)
    return decide_route(script=..., script_conf=..., mixed=..., is_blank=blank,
                        has_table=has_table, cfg=cfg)
```

- Input: a **per-page `page_context` dict** and reads `page_context["image_array"]` — a decoded **RGB numpy array** (confirmed by the caller, §1.3 and §Contradiction (i)). It does **not** take `image_bytes`, and never re-renders / never opens `fitz`.
- Builds the observer via a lazy module singleton `_get_observer()` (lines 23–29) using `LipikaOnnxScriptObserver(cfg=ProbeConfig.from_env())`. Tests swap `inprocess._OBSERVER` directly.
- **It does NOT call `probe_document`.** It also does **not** call the VLM at all — only the ONNX observer. `handwriting_prob` is never computed here (it passes no `handwriting_prob=` to `decide_route`), so `decide_route`'s latin-handwritten branch can never fire from this path.
- Module docstring is explicit (lines 1–13): imports ONLY `imaging`, `cv_features`, `routing`, and the observer — "never fitz / probe.py — so PyMuPDF stays off the hot path."

### 1.2 `src/ai/layer1_probe/probe.py` — the per-document probe **[C]**

`needs_probe` (lines 40–54):

```python
def needs_probe(profile: Optional[dict], given_route: Optional[str], cfg: ProbeConfig) -> bool:
    if cfg.force_probe_all:
        return given_route != ROUTE_SKIP
    if given_route in (ROUTE_SKIP, ROUTE_DOCLING):
        return False
    if given_route is not None:           # printed/handwritten/vlm from L1 -> verify script
        return True
    p = profile or {}
    return bool(
        p.get("is_scanned")
        or not p.get("has_text", False)
        or p.get("char_count", 0) <= cfg.digital_char_threshold
        or p.get("primary_script", "unknown") == "unknown"
    )
```

Note: it takes a **profile `dict`** (`p.get(...)`), not a `PageProfile` model.

`probe_document` (lines 72–81, signature):

```python
def probe_document(
    pdf_bytes: bytes, *, document_id: str, s3_bucket: Optional[str], s3_key: Optional[str],
    pages_in: Optional[List[dict]], cfg: ProbeConfig, s3: Any = None, vlm: Any = None,
    onnx: Any = None, etag: Optional[str] = None,
    time_left: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
```

Crux question — can `probe_document` run over already-decoded page images, `s3=None`, no re-render?

- **No, not as written.** Line 73 opens the PDF (`fitz.open(stream=pdf_bytes, ...)`) and `total = len(doc)`; the Phase-A loop (lines 116–146) renders every candidate with `imaging.render_rgb(doc[pn-1], ...)` (line 124). `pages_in` is only a list of `{page_number, profile?, route?}` dicts used for **candidate selection** (lines 85–114) — there is **no path to feed pre-decoded RGB arrays or page images in**. It always needs `pdf_bytes` and always re-renders.
- `s3=None` is tolerated: uploads are guarded by `do_upload = bool(cfg.write_page_images and s3 is not None and img_bucket)` (line 118) and the cache read/write is guarded by `s3 is not None` (lines 157, 196). So with `s3=None` it skips S3 cleanly — **but it still re-renders from `pdf_bytes` with fitz**, which the request forbids ("Do not render pages again").
- Therefore **"run once over existing page images, no S3, no re-render" is NOT supported by `probe_document` today.** It would require **either** a new helper **or** reusing the already-shipped `inprocess.probe_page_route` (which is per-page, not once-per-document, and has no VLM).

Minimal new-helper shape if the request's design (B) is chosen — something like:

```python
# NEW in probe.py or inprocess.py — no fitz, no S3, images already decoded
def probe_document_images(
    images: dict[int, np.ndarray],          # page_number -> decoded RGB (page_context["image_array"])
    *, pages_in: list[dict], cfg: ProbeConfig,
    vlm: Any = None, onnx: Any = None,
) -> dict[int, PageProbe]:
    # Phase A: CV signals from each in-memory RGB (imaging.analysis_gray + ink_mask + detect_table)
    # Phase B: pick_sample_pages over non-blank candidates; observe via onnx then vlm fallback
    #          (reuse the _observe logic + aggregate_doc); honour vlm_for_handwriting
    # Phase C: decide_route per page from own/nearest sample; return PageProbe per page
```

This would reuse `aggregate.pick_sample_pages / samples_disagree / aggregate_doc / nearest_sample`, `routing.decide_route`, and `vlm_probe.BedrockScriptProbe` — i.e. almost all of `probe_document`'s Phase B/C logic minus the fitz render and S3. It is a real, non-trivial new function, not a one-liner.

### 1.3 `src/ai/layer1_routing/pipeline.py` **[C]**

(a) **`process_document` page loop** (lines 607–705). Each entry in `pages` must supply (docstring lines 630–635): `page` (PyMuPDF page), `page_number` (int), `context` (dict with `pdf_path, image_array, image_bytes`), `image_bytes` (PNG bytes). It loops (serial if ≤1 page or `page_concurrency_limit==1`, else `ThreadPoolExecutor`) calling `process_page(page=..., page_context=page_data["context"], page_image_bytes=page_data["image_bytes"], ...)`. There is **no document-level probe call** here today — probing happens per page inside `process_page`.

Confirmed shape of each page dict by the real caller `demo_resume_pipeline.py::_build_page_list` (lines 103–127) and `demo_end_to_end.py::_build_pages_for_pdf` (lines 358–373): `context` carries `image_array` = `np.array(Image.open(png).convert("RGB"))` (a **decoded RGB numpy array**) and `image_bytes` = `pix.tobytes("png")` (**PNG bytes**). So **page images ARE already in memory per page** — supporting a "no re-render, no S3" probe in principle, but as per-page arrays, not as `pdf_bytes`.

(b) **`process_page` routing decision point** (lines 330–end). `page_context` is rebuilt line 366: `page_context = {**page_context, "image_bytes": page_image_bytes, "page_number": page_number}`. Then:
- `routing_mode == "capability_based"` (lines 391–404): `caps = capabilities_from_profile(profile)`, `plan = build_engine_plan(caps)`.
- else `single_engine` (lines 406–433): `single_route = route_from_profile(profile)` with a scanned/complex fallback, then `plan = [EngineTask(engine=_ROUTE_TO_ENGINE.get(single_route, single_route), priority=1, ...)]`.
- `primary_task = plan[0] if plan else None` (line 442).
- **Existing probe seam** (lines 444–453):

```python
if primary_task is not None:
    _choice = _maybe_probe_route(profile, page_context, primary_task.engine, page_number)
    if _choice is not None:
        primary_task = EngineTask(engine=_choice.route, priority=1, reason=f"probe:{_choice.reason}")
        plan = [primary_task]
        if _choice.ocr_lang:
            page_context["ocr_lang"] = _choice.ocr_lang
```

This is **exactly the single seam** the request wants — but it already exists and already injects a single-task plan **before** the escalation ladder, for **both** routing modes (the seam is after the mode dispatch). The ladder (lines ~497–590) runs afterwards unchanged. Note `_choice.route` is written **directly** as the engine name (e.g. `"vlm_transcribe"`, `"paddleocr_printed"`) because the probe's `RouteChoice.route` already uses engine-name strings (§1.5/§types), so no `_ROUTE_TO_ENGINE` remap is applied to the probe route here.

(c) **`_ROUTE_TO_ENGINE`** (lines 50–57):

```python
_ROUTE_TO_ENGINE = {
    "digital":        "docling",
    "scanned":        "paddleocr_printed",
    "handwritten":    "paddleocr_handwritten",
    "vlm_transcribe": "vlm_transcribe",
    "skip":           "skip",
}
```

The request says the map should contain keys `skip, paddleocr_printed, paddleocr_handwritten, vlm_transcribe`. **It does NOT** — the keys are the Layer-1 route vocabulary (`digital/scanned/handwritten/vlm_transcribe/skip`) and the **values** are the engine names (`docling/paddleocr_printed/paddleocr_handwritten/vlm_transcribe/skip`). So "map probe.route with `_ROUTE_TO_ENGINE`" as literally described is **wrong against this dict**: probe routes are *already* engine names and would not be found as keys (except `vlm_transcribe`/`skip`). This is a real contradiction (see table row iii).

(d) **`PageMetadata.extra`** — yes. `src/ai/schemas/page_metadata.py` line ~210: `extra: Dict[str, Any] = field(default_factory=dict)`. So `metadata.extra["probe"] = {...}` is supported. **[C]** Nothing writes to it today.

(e) **Escalation ladder** (lines ~497–590): `while (confidence < settings.escalation_confidence_threshold or not markdown.strip()) and escalation_attempts < settings.max_escalation_attempts:` → `next_escalation_route(current_route, ...)` → `_ROUTE_TO_ENGINE.get(next_route, ...)`. It runs **after** `_run_plan(plan, ...)` and is independent of how `plan` was chosen. Injecting a probe-driven single-task plan before `_run_plan` (as the existing seam does) does not touch thresholds/ladder/routing_mode. **Confirmed the seam is clean.**

### 1.4 `src/ai/layer2_conversion/scanned.py` **[C]**

Signatures:

```python
def _invoke_paddle_lambda(image_array, page_number, mode,
                          image_bytes=None, ocr_lang=None) -> Optional[tuple[str,float,list[Block]]]:   # line 60
def _get_paddle_engine(mode: str = "printed") -> Any:                                                   # line 173
def _ocr_convert(image_array, page_number, mode, image_bytes=None, ocr_lang=None) -> tuple[...]:        # line 235
def convert_scanned_page(image_array, page_number, image_bytes=None, ocr_lang=None) -> tuple[...]:      # line 329
def convert_handwritten_via_paddle(image_array, page_number, image_bytes=None, ocr_lang=None) -> ...:   # line 347
```

- **`ocr_lang` is ALREADY threaded** through all of `_invoke_paddle_lambda`, `_ocr_convert`, `convert_scanned_page`, `convert_handwritten_via_paddle` (commit `de95e3d`). The request item 4 ("add an optional ocr_lang parameter to convert_scanned_page, convert_handwritten_via_paddle, _ocr_convert and _invoke_paddle_lambda") is **already done** for these four.
- **Lambda payload** (lines 117–127): `payload = {"image_base64": b64_str, "page_number": page_number}` and **`if ocr_lang: payload["ocr_lang"] = ocr_lang`**. So the "add ocr_lang to the payload ONLY when set" behaviour is **already implemented** — the payload is byte-identical to the old one when `ocr_lang` is falsy. The request's premise that the payload "today sends no ocr_lang" is only true for the flag-off / no-lang case; the **code already supports** the keyed add.
- **`_get_paddle_engine` does NOT cache by lang** — it caches by `mode` only (`_paddle_engines: dict[str, Any]` keyed by `"printed"`/`"handwritten"`, lines 32, 184) and hardcodes `lang="en"` for both engines (lines 201, 220). So the request's "Local fallback `_get_paddle_engine(mode, lang)` caches one engine per lang (default en)" is **NOT implemented** — this is the one genuine remaining change in item 4. **[C]**
- **Three call sites** of `_invoke_paddle_lambda**, all inside `_ocr_convert`: line 251 (prefer-lambda path), line 292 (ImportError fallback), line 311 (generic-exception fallback). All already forward `ocr_lang=ocr_lang`. **[C]**

### 1.5 `src/ai/layer1_probe/vlm_probe.py` — `BedrockScriptProbe` **[C]**

```python
def _call(self, jpeg: bytes) -> str:
    resp = self.client.converse(
        modelId=self.model_id,
        messages=[{"role": "user", "content": [
            {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},   # line ~96 — HARDCODED jpeg
            {"text": _PROMPT},
        ]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.0},
    )
def observe(self, page_number: int, jpeg: bytes) -> Optional[SampleObs]:   # takes raw image bytes
```

- `observe(page_number, jpeg)` takes **raw image bytes** and passes them straight through; it does **not** decode, inspect magic bytes, or re-encode. `format="jpeg"` is hardcoded.
- In the per-document `probe.py` path the bytes are always real JPEG (produced by `imaging.encode_jpeg`, line 136), so there the hardcoded `"jpeg"` is correct.
- The mismatch only arises if design (B) feeds `process_document`'s **PNG** `image_bytes` to `BedrockScriptProbe`. **Note: the CURRENT in-process path (`inprocess.py`) never calls the VLM at all**, so this PNG-vs-JPEG bug is **latent / not reachable today** — it only becomes live under design (B).
- **Recommendation (simpler fix, if B is built):** detect the real format from magic bytes at the probe boundary and pass the correct `format` string (`b"\x89PNG" → "png"`, `b"\xff\xd8" → "jpeg"`). Bedrock Converse accepts `png` and `jpeg` directly, so a 2–3 line format-sniff in `_call`/`observe` avoids a full decode+re-encode. Re-encoding to JPEG is also valid but costs a Pillow/cv2 decode+encode per sampled page for no routing benefit. **Magic-byte detection is the lighter change.** (Alternatively, since `process_document` already renders, the caller could hand the probe the JPEG it would make — but there is no JPEG in `page_context` today, only PNG `image_bytes` + RGB `image_array`.)

### 1.6 `src/config/settings.py` — flag style **[C]**

Pydantic-settings `BaseSettings` with `env_prefix="IDP_"` (lines 19–24). Existing boolean-flag templates:

```python
enable_vlm_direct: bool = False     # line ~ (Core routing thresholds block)
paddle_prefer_lambda: bool = True
graph_merge_label_gate: bool = True
```

So a new flag in the existing style is:

```python
probe_enabled: bool = False         # env IDP_PROBE_ENABLED (prefix IDP_ + field name)
```

With `env_prefix="IDP_"`, field `probe_enabled` is read from `IDP_PROBE_ENABLED` automatically — **this matches the request's requested env name exactly.** Note some flags also get an un-prefixed fallback inside `_fallback_unprefixed_env` (the `@model_validator(mode="before")`); a new flag only needs that if an un-prefixed `PROBE_ENABLED` must also work.

**Flag-name contradiction:** the existing code gates the probe with **`IDP_ENABLE_PROBE`** read via a bare `os.getenv` in `pipeline.py::_probe_enabled()` (line 62–63) — **not** a `settings` field, and a **different env var** from the requested `IDP_PROBE_ENABLED`. Neither `IDP_PROBE_ENABLED` nor a `settings.probe_enabled` exists anywhere (grep found zero hits outside design docs). The `.env` / `.env2` files contain **no** PROBE variables at all, so the probe is OFF today by default via the missing env var. **[C]**

---

## 2. Contradiction-check table (TRUE / FALSE / PARTIAL + evidence)

| # | Plan assumption | Verdict | Evidence |
|---|---|---|---|
| i | Each per-page dict/context in `process_document` already holds the page image bytes, so the probe need not re-render or hit S3 | **TRUE** | `context` carries `image_array` (decoded RGB numpy) and `image_bytes` (PNG). `demo_resume_pipeline.py:103–127`, `demo_end_to_end.py:358–373`; consumed at `pipeline.py:120/128/138`. Images are per-page, in memory. |
| ii | The probe can be run once per document over those images without `probe_document`'s fitz re-render (or a small new helper is needed — describe it) | **FALSE (new helper needed)** | `probe_document` always `fitz.open(stream=pdf_bytes)` and `render_rgb` per page (`probe.py:73, 124`). No entry point accepts pre-decoded arrays. The existing `inprocess.probe_page_route` runs per-page over `image_array` but is **not** once-per-document and has **no VLM**. A new `probe_document_images(...)` helper (shape in §1.2) is required for design (B). |
| iii | `_ROUTE_TO_ENGINE` maps exactly `skip/paddleocr_printed/paddleocr_handwritten/vlm_transcribe` → engine names | **FALSE** | `pipeline.py:50–57` keys are `digital/scanned/handwritten/vlm_transcribe/skip`; the **values** are `docling/paddleocr_printed/paddleocr_handwritten/vlm_transcribe/skip`. The probe's `RouteChoice.route` is already an engine-name string (`ROUTE_PRINTED="paddleocr_printed"`, etc., `types.py:56–60`), so it is used **directly** (pipeline.py:447), not remapped through this dict. |
| iv | `PageMetadata` has an `extra` dict | **TRUE** | `page_metadata.py`: `extra: Dict[str, Any] = field(default_factory=dict)`. Nothing writes `extra["probe"]` yet. |
| v | `_invoke_paddle_lambda` today sends no `ocr_lang` | **PARTIAL** | The function **already accepts** `ocr_lang` and **already adds** `payload["ocr_lang"]` only when truthy (`scanned.py:117–127`). When the probe is off / lang unset the payload is byte-identical to the old `{image_base64, page_number}`. So "sends no ocr_lang" is true only for the flag-off case; the keyed-add is already implemented (commit `de95e3d`). |
| vi | `process_page` has a clean single seam to inject a probe-chosen single-task plan BEFORE the ladder without altering thresholds/ladder/routing_mode | **TRUE — and already present** | The seam exists at `pipeline.py:444–453` (`_maybe_probe_route` → replace `plan=[primary_task]`, set `page_context["ocr_lang"]`), after mode dispatch and before `_run_plan`/ladder. It already works for both routing modes and leaves the ladder untouched. |
| vii | The VLM probe hardcodes jpeg while `process_document` supplies PNG | **TRUE (but latent)** | `vlm_probe.py:_call` hardcodes `format="jpeg"`; `process_document` `image_bytes` is PNG. **However the current in-process path never calls the VLM**, so the clash is only reachable under design (B). In the existing `probe.py` path the bytes are genuine JPEG, so no clash there. |

Rows that are **FALSE or PARTIAL** (ii, iii, v, vii) are contradictions the user must hear before editing. Rows i/iv/vi are consistent with the plan.

---

## 3. Integration-seam notes for the 6 planned CHANGES

1. **Flag `settings.probe_enabled` (env `IDP_PROBE_ENABLED`, default False).**
   Slots into `settings.py` as `probe_enabled: bool = False` (prefix `IDP_` → `IDP_PROBE_ENABLED`, matches §1.6 templates).
   **Conflict:** the live gate today is `os.getenv("IDP_ENABLE_PROBE")` in `pipeline.py:_probe_enabled()` — a *different* env var and not a settings field. The plan's flag must either replace that gate (changing the env name turns off anyone relying on `IDP_ENABLE_PROBE`) or the two must be reconciled. **Decision needed.**

2. **`process_document`: run the probe ONCE per doc over in-memory images, `try/except`, attach `page_context["probe"]`.**
   No place to slot in today — `process_document` currently does no probing; probing is per-page in `process_page`. Requires: (a) the **new `probe_document_images` helper** (§1.2, row ii) because `probe_document` needs `pdf_bytes`+fitz; (b) a new block in `process_document` that gathers `{pn: page_data["context"]["image_array"]}` + per-page profiles/routes, calls `needs_probe` to see if ANY page qualifies, runs the helper once in `try/except` (on failure log `probe.failed`, attach nothing → per-page fallback), and writes `page_data["context"]["probe"] = {...}` per page. This is a **net-new integration**, and it overlaps/duplicates the existing per-page hook — they cannot both drive routing.

3. **`process_page`: if `page_context` has `"probe"`, build single-task plan from `probe.route`; record `extra["probe"]`; ladder runs after.**
   The seam at `pipeline.py:444–453` is the right spot, but today it calls `_maybe_probe_route` (the ONNX per-page hook), **not** a `page_context["probe"]` lookup. Under design (B) this becomes: `if "probe" in page_context: primary_task = EngineTask(engine=page_context["probe"]["route"], ...)`. Writing `metadata.extra["probe"] = {...}` is supported (row iv). **Note:** `probe.route` is already an engine name, so it is assigned directly (do **not** remap through `_ROUTE_TO_ENGINE`, row iii). Pages with no `"probe"` key fall through to today's code — matches the request.

4. **Thread `ocr_lang` through the four scanned.py functions; payload add only when set; `_get_paddle_engine(mode, lang)` cache per lang.**
   Three of four are **already done** (rows v, §1.4): `ocr_lang` param + conditional `payload["ocr_lang"]` + forwarding at all three call sites. The **only** remaining change is `_get_paddle_engine` — add a `lang: str = "en"` param and key `_paddle_engines` by `(mode, lang)` instead of `mode`, passing `lang` into the `PaddleOCR(lang=...)` ctor. Low-risk, isolated.

5. **Image format: PNG vs hardcoded jpeg at the VLM boundary; add a PNG-input test.**
   Only relevant under design (B) (row vii — latent today). Simpler fix: magic-byte sniff in `vlm_probe.BedrockScriptProbe._call`/`observe` to set `format` to `"png"`/`"jpeg"` from the bytes (§1.5). Re-encoding is heavier. **Recommend magic-byte detection.**

6. **Keep lipika telemetry + VLM-overrides-lipika rule unchanged.**
   Both live **only** in `probe.py::_observe` (the per-document path): the `probe.lipika_vs_vlm` log (lines ~245–261) and the "VLM contradicts latin → VLM wins; else adopt only handwriting_prob" seam (lines ~226–244). The existing `inprocess.py` path has **neither** (no VLM, no lipika-vs-vlm telemetry). If design (B) adds `probe_document_images`, that logic must be carried into the new helper to honour this item; it is not reusable from `inprocess.py` as-is.

---

## 4. Decisions the user must confirm BEFORE editing

1. **Which integration is the target?** A pre-existing, committed, tested **per-page ONNX-observer** in-process hook (`inprocess.probe_page_route` + `pipeline._maybe_probe_route`, seam at `pipeline.py:444–453`) already drives routing when `IDP_ENABLE_PROBE` is on. The request describes a **different** design: a **per-document VLM-sampling** probe run once in `process_document`, `page_context["probe"]`, `extra["probe"]`, `settings.probe_enabled`. **These are two different probes.** Extend the existing one, or build the requested one (and reconcile/remove the existing wiring + its tests `tests/test_pipeline_probe_integration.py`)? This is the core STOP-AND-ASK.
2. **Flag name/location:** request wants `settings.probe_enabled` / `IDP_PROBE_ENABLED`; code has `IDP_ENABLE_PROBE` (bare `os.getenv`, no settings field). Rename, or keep both? Renaming silently disables any current `IDP_ENABLE_PROBE` users.
3. **`probe_document` cannot run no-render/no-S3 over pre-decoded images** (row ii). Confirm building a new `probe_document_images(images, pages_in, cfg, vlm, onnx)` helper (§1.2) rather than reusing `probe_document`.
4. **`_ROUTE_TO_ENGINE` does not have the keys the request names** (row iii). Confirm that probe `RouteChoice.route` values (already engine names) are used directly, not remapped.
5. **`ocr_lang` is already wired through three of the four functions and the payload** (row v). Confirm the only scanned.py change is `_get_paddle_engine` per-lang caching.
6. **PNG-vs-JPEG is latent, not live** (row vii), because the current in-process path has no VLM. Confirm this fix is only needed if design (B) feeds PNG to `BedrockScriptProbe`; recommend magic-byte detection.
7. **Telemetry + VLM-overrides-lipika rule** live only in `probe.py::_observe`, not in `inprocess.py`. Under design (B) they must be reproduced in the new helper; confirm.

**Net:** the request's description does **not** match the branch as written. The largest divergence (per-page ONNX hook already shipped vs per-document VLM probe requested) must be resolved by the user before any editing begins.

---
*Content from source files was read directly; line numbers are from the files as of HEAD `705fe37` on `feat/llm-observability`. No code was executed, built, or tested; no AWS calls were made.*
