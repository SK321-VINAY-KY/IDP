# Layer-1 (PyMuPDF) routing vs Layer-1.5 Probe routing — what happens to the original?

Read-only investigation. Every claim below is cited `file:line` and marked **confirmed by reading** or **inferred**. Nothing was edited or executed.

---

## Bottom line

The original PyMuPDF Layer-1 routing is **NOT replaced**. It stays the first pass and still runs first — in the probe-enabled pipeline it is **option (b): Layer-1 is the first pass, and the probe only takes over scanned/ambiguous/non-digital pages, refining or overriding their route**. Concretely:

- Layer-1 (`lambda_layer1_handler.py`) runs first and emits one `route` per page plus the full `PageProfile`. This is unchanged in both state machines.
- The probe Lambda runs as a **new stage inserted between Layer-1 and the page Map** (only in `state_machine_with_probe.asl.json`). It consumes Layer-1's `pages[]` (each carrying `route` + `profile`).
- The probe **passes through unchanged** any page Layer-1 routed to `skip` or `docling`, and any page whose profile looks like clean digital text — those keep the Layer-1 route (`needs_probe()` returns `False`). For those pages the probe literally copies `given_route` into its output (`route=g.get("route") or ROUTE_DOCLING`).
- The probe **takes over and re-routes** scanned pages, pages with no/low embedded text, unknown-script pages, and any page Layer-1 already pointed at an OCR/VLM engine (printed/handwritten/vlm) — for the latter it re-verifies the script before confirming.
- If the probe Lambda fails, Step Functions catches it and continues with the Layer-1 routes untouched (`ProbeFallbackToLayer1Routes`).
- The **legacy** `state_machine.asl.json` has no probe at all — Layer-1 routes go straight to the Map. So whether the probe touches routing at all is a deploy-time choice of which ASL you install.

One important nuance (**confirmed**): in the current probe-enabled wiring, Layer-1's per-page `route` is **always present** in the Map item, because `lambda_layer1_handler.py` always sets a `route` for every page. That means inside `needs_probe`, the `given_route is not None` branch is the one that fires for most pages — see the decision table caveat below.

---

## 1. What PyMuPDF Layer-1 produces

**Page profile object — `PageProfile`**, built by `inspect_page()` in `src/ai/layer1_routing/inspect.py:95-137` (**confirmed**). Fields set (`inspect.py:126-137`): `page_number`, `has_text` (= `char_count > 0`), `char_count`, `image_coverage`, `has_tables`, `is_scanned`, `has_vector_drawings`, `primary_script`, `complexity_score`, `dpi_estimate`.

- `is_scanned` = `char_count < scanned_char_count_threshold AND image_coverage > scanned_image_coverage_threshold` (`inspect.py:117-119`, **confirmed**).
- `primary_script` from cheap Unicode-range scan of embedded words, default `"unknown"` when no words, `"latin"` when words but no Indic hits (`inspect.py:33-60`, **confirmed**).
- `complexity_score` 0–5 heuristic (`inspect.py:140-154`, **confirmed**).

**Does Layer-1 assign a route/engine per page?** Yes. Two code paths both emit a per-page route:

- The Lambda handler `lambda_layer1_handler.py:132-180` (**confirmed**): after `inspect_page()` it computes a `route`. In default `single_engine` mode it calls `route_from_profile(profile)` and, if that returns `None`, forces `"scanned"` for scanned/complex/image-heavy pages else `"digital"` (`lambda_layer1_handler.py:147-160`). It always appends `"route": primary_route` to each page report (`lambda_layer1_handler.py:182-189`). In `capability_based` mode `primary_route = plan[0].engine` (`lambda_layer1_handler.py:137-145`).
- The in-process orchestrator `pipeline.py:process_page` (`pipeline.py:233-456`, **confirmed**) does the same routing internally but runs the engines itself; it is used by demo/CLI scripts, not by the Step Functions probe path.

**Routes Layer-1 can emit** (**confirmed**): `route_from_profile()` (`router.py:78-110`) returns `"skip"`, `"scanned"`, `"digital"`, or `None`; the handler maps the `None`/escalation space into `"scanned"`/`"digital"`. The full route vocabulary used downstream is `digital | scanned | handwritten | vlm_transcribe | skip` (`router.py:22`, `pipeline.py:50-57`). Note the engine-name vs route-name mapping: `digital→docling`, `scanned→paddleocr_printed`, `handwritten→paddleocr_handwritten` (`pipeline.py:50-57`, **confirmed**). In `capability_based` mode the Lambda emits the *engine name* directly as the route (e.g. `docling`, `paddleocr_printed`) (`lambda_layer1_handler.py:137-145`, **confirmed**).

---

## 2. How the probe relates to the Layer-1 profile — `needs_probe()`

`needs_probe(profile, given_route, cfg)` in `src/ai/layer1_probe/probe.py:50-66` (**confirmed**). Decision order, exactly as coded:

| # | Condition (in order) | Result |
|---|---|---|
| 1 | `cfg.force_probe_all` is True | **probe** every page except `given_route == "skip"` (`probe.py:51-52`) |
| 2 | `given_route in {"skip", "docling"}` | **passthrough** — `return False` (`probe.py:53-54`) |
| 3 | `given_route is not None` (i.e. printed / handwritten / vlm from L1) | **probe** — `return True`, to re-verify script (`probe.py:55-56`) |
| 4 | else (no given route) — `profile.is_scanned` | **probe** (`probe.py:58-60`) |
| 5 | else — `not profile.has_text` | **probe** (`probe.py:61`) |
| 6 | else — `profile.char_count <= cfg.digital_char_threshold` (default 100) | **probe** (`probe.py:62`) |
| 7 | else — `profile.primary_script == "unknown"` | **probe** (`probe.py:63`) |
| 8 | else (digital text, known script, enough chars) | **passthrough** — `return False` |

Pages the probe **SKIPs** (passthrough, keep Layer-1 route): `skip` and `docling` routes (step 2), and — only when `given_route is None` — clean digital pages with text, >100 chars, and a known script (step 8).

Pages the probe **TAKES OVER**: scanned pages, text-less pages, near-empty pages, unknown-script pages, and (step 3) any page Layer-1 already routed to an OCR/VLM engine — those are re-examined and may be re-routed by the probe's own routing table.

**Caveat (confirmed + inferred):** Because `lambda_layer1_handler.py` always attaches a `route` to every page (§1), `given_route` is almost never `None` in the real Step Functions payload. So in practice steps 4–8 (the profile-based branch) only fire when a caller supplies pages **without** a route (e.g. the probe inspecting the PDF itself — `probe.py:95-99`, where `profile = inspect_page(...)` is computed only when both `profile is None` and `route is None`). With real Layer-1 output, routing reduces to: `skip`/`docling` → passthrough; everything else → probe re-routes. This matches the doc's note that `docling`/`skip` pages are not probed (`docs/PROBE_INTEGRATION.md`, "Known limits").

---

## 3. What route do NOT-probed pages carry — Layer-1's route or a probe default?

They carry **Layer-1's route** (`given_route`), with `docling` as the only fallback. In `probe_document`, for `needs_probe == False` pages (`probe.py:100-106`, **confirmed**):

```python
results[pn] = PageProbe(page_number=pn, probed=False, status="not_needed",
                        route=g.get("route") or ROUTE_DOCLING,
                        route_reason="passthrough_digital_or_blank",
                        script=(profile or {}).get("primary_script", "unknown"),
                        script_source="embedded")
```

So `route = given_route`, falling back to `ROUTE_DOCLING` (`"docling"`) only if Layer-1 gave no route (`probe.py:102`, **confirmed**). `probed=False`, `status="not_needed"`. The probe does **not** impose a default engine on these — it preserves Layer-1's decision.

(Other non-probed statuses: `status="error"` keeps `given_route or ROUTE_PRINTED` on a render failure — `probe.py:143-145`; `status="skipped_timeout"` recomputes a prior-only route when the time budget runs out — `probe.py:214-222`. Both **confirmed**.)

---

## 4. Who calls what at runtime — the order

**Two Lambda entry points** (**confirmed**):
- Layer-1: `src/api/lambda_layer1_handler.py:71` `lambda_handler` → `inspect_page` + `route_from_profile`/`build_engine_plan`. Emits `{pages:[{page_number, profile, route, ...}]}`.
- Probe: `src/api/lambda_probe_handler.py:62` `lambda_handler` → `probe_document(...)` (`lambda_probe_handler.py:84-88`). Reads `event["pages"]` (Layer-1 output) and the PDF from S3.

**Order is set by the Step Functions ASL, not by Python imports.** Two machines exist:

- **`state_machine.asl.json` (legacy, no probe)** — `StartAt: ProcessDocumentPages` (a Map) directly over `$.pages` (**confirmed**, `state_machine.asl.json:3-10`). Layer-1 routes go straight to the engines. `RouteEngineChoice` only knows `docling`, `paddleocr_handwritten`, default `paddleocr_printed` — no `skip`, no `vlm_transcribe` (`state_machine.asl.json:27-41`).
- **`state_machine_with_probe.asl.json` (probe-enabled)** — `StartAt: ProbeDocument` (**confirmed**, `state_machine_with_probe.asl.json:3-5`). Flow: `ProbeDocument` (probe Lambda) → `UseProbePages` (replaces `$.pages` with `$.probe.pages`) → `ProcessDocumentPages` (Map) → `FormatWorkflowOutput` (`...with_probe.asl.json:44-65`). On probe failure, `Catch` → `ProbeFallbackToLayer1Routes` → continues the Map with the original `$.pages` (Layer-1 routes) (`...with_probe.asl.json:24-33, 55-59`). The Map item now also forwards the whole probe page object as `page` (`...with_probe.asl.json:72`), and `RouteEngineChoice` adds `skip` and `vlm_transcribe` branches (`...with_probe.asl.json:80-106`).

So the runtime order in the probe-enabled machine is:

```
S3 PDF → L1 Lambda (inspect_page → PageProfile + route)   [runs first, unchanged]
        → SFN: ProbeDocument (probe_document: render/CV/sample/VLM/ONNX → per-page route + ocr_lang + image S3 key)
        → UseProbePages (swap $.pages = probe pages)
        → Map(page) → skip | docling | paddleocr_printed | paddleocr_handwritten | vlm_transcribe
```

This is confirmed verbatim by `docs/PROBE_INTEGRATION.md` ("Where it sits" diagram and "If the probe Lambda fails, `ProbeFallbackToLayer1Routes` continues with the Layer 1 routes unchanged").

**Note (confirmed):** nothing in Step Functions calls `pipeline.process_document` — the probe path never runs the in-process pipeline. `process_document`/`process_page` are invoked only by demo/CLI scripts (`demo_end_to_end.py:401`, `demo_resume_pipeline.py:206`, `scripts/demo.py`, `scripts/run_escalation_test.py`). The probe does, however, import and call Layer-1's `inspect_page` directly (`probe.py:95`) for the self-inspection case.

---

## 5. Which pages are routed by Layer-1 alone vs re-routed by the probe

**Routed by Layer-1 alone (probe passes the route through unchanged):**
- Every page in the **legacy** `state_machine.asl.json` (no probe stage at all).
- In the probe machine: pages Layer-1 routed to **`skip`** or **`docling`** (`needs_probe` step 2 → `probed=False, status="not_needed"`, keeps `given_route`).
- When pages arrive without a route (not the normal L1 output): clean digital pages (has text, >100 chars, known script) — `needs_probe` step 8 passthrough.
- Pages reached only on the probe failure fallback (`ProbeFallbackToLayer1Routes`).

**Re-routed (or route-confirmed) by the probe:**
- Any page Layer-1 routed to `paddleocr_printed`, `paddleocr_handwritten`, or `vlm_transcribe` — the probe re-verifies script/handwriting/table and emits its own `route` + `ocr_lang` + `use_structure` via `decide_route()` (`probe.py:55-56`, `probe.py:170-189`, `routing.py:14-40`). The probe can change the engine (e.g. printed→vlm for an unsupported script, or printed→handwritten for Latin handwriting).
- When no route is supplied: scanned / text-less / near-empty (≤100 chars) / unknown-script pages (`needs_probe` steps 4–7).
- Blank pages detected by the probe's CV become `route="skip"` regardless of Layer-1 (`probe.py:205-209`, **confirmed**).

The probe's routing table (`routing.py:14-40`, **confirmed**): blank→skip; unknown/other script or conf<0.60 or mixed→vlm_transcribe; Latin + hw≥0.30→paddleocr_handwritten; non-Latin handwritten→vlm_transcribe; printed script in `PRINTED_SCRIPTS` (default just `latin`)→paddleocr_printed (+`use_structure` if ruled table); printed script not in set→vlm_transcribe.

---

## Conclusions and recommendations

1. **The original PyMuPDF Layer-1 is intact and still authoritative for digital/skip pages.** The probe is additive and refinement-only; it was deliberately designed to pass `docling`/`skip` through untouched (`probe.py:53-54`, doc "Known limits"). **Confirmed.**

2. **Known gap worth flagging (from the doc, confirmed against code):** because `docling` pages are never probed, a typed form **with handwritten annotations** that Layer-1 labelled `digital`/`docling` will skip the probe entirely and never be seen as handwriting. If that case matters, either (a) have Layer-1 route such mixed pages to an OCR route so step-3 re-verification kicks in, or (b) set `IDP_PROBE_FORCE_PROBE_ALL=1` (probes everything except `skip`, `probe.py:51-52`). **Recommendation only — not implemented.**

3. **Operational coupling:** the probe-enabled machine references `${VLM_LAMBDA_ARN}` (`...with_probe.asl.json` `InvokeVlmTranscribe`) which the doc says does not exist yet. Any script the probe routes to `vlm_transcribe` (all non-Latin by default, since `PRINTED_SCRIPTS=latin`) will fail in the Map until that Lambda is deployed. **Confirmed via doc "Known limits" + ASL.**

4. **The `needs_probe` profile branch (steps 4–8) is mostly dormant in production** because Layer-1 always attaches a route (§2 caveat). It only matters when the probe inspects a PDF with no Layer-1 manifest. This is **inferred** from the always-set `route` in `lambda_layer1_handler.py:182-189` combined with `probe.py:95-99`; worth a comment in the code to avoid confusion.
