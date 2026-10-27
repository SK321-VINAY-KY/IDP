# Design Review — Probe/lipika In-Process Integration (`probe-lipika-design.md`)

Reviewer: design-review subagent (fresh read, no prior context).
Scope reviewed: LOCAL build only, NO AWS actions. Verdict is mechanical: any HIGH or MEDIUM finding → CHANGES_REQUESTED.

## Verdict

**CHANGES_REQUESTED** — 1 HIGH, 2 MEDIUM, 3 NIT.

The design is unusually thorough and nearly all of its "verified" source claims check out against the real code. The blocking problems are concentrated in one place: the **test-teardown inventory in §5/§8/§10 is still incomplete**, despite §12.2 Finding 1 claiming the "complete, verified eight-test set." At least one behavioral test asserts `paddleocr_handwritten` routing and is not in the list, and two more tests read `SampleObs.handwriting_prob`, which the design removes. Under the promised rewrite these tests fail, directly contradicting the design's own "leave the suite green" guarantee.

---

## Findings

### 1. HIGH — `test_e2e_heterogeneous_doc_uses_nearest_sample_and_widens_sampling` asserts `paddleocr_handwritten` and is missing from the teardown inventory

`tests/test_probe.py` (verified, line ~300) ends with:

```python
pg = {p["page_number"]: p for p in out["pages"]}
assert pg[1]["route"] == "paddleocr_printed" and pg[10]["route"] == "paddleocr_handwritten"
```

The test drives pages 6–10 through a `StubVLM` with `hw=0.9` (via `per = {pn: ("latin", .95, .9) ...}`), so page 10 currently routes to `paddleocr_handwritten`. Under the new script-only router (§5), a Latin page at any confidence routes to `paddleocr_printed` and **no input can ever yield `paddleocr_handwritten`** (the design even mandates an assertion to that effect in §8b). So after the rewrite this assertion fails.

This test is **not** in the design's "complete, verified eight-test set" (§5 "Repo-wide cleanup grep" / §8 / §10). That set names: `test_routing_table`, `test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm`, `test_table_sets_structure_hint_only_for_printed`, `test_blank_routes_to_skip`, `test_e2e_devanagari_handwritten_goes_to_vlm_and_latin_hw_to_handwritten_engine`, `test_onnx_classifier_contract`, and the two `handwriting_heuristic` tests. The heterogeneous-doc test is absent. §12.2 Finding 1 explicitly claims the inventory is complete and mandates a "pre-implementation re-grep" — but that grep was run only for `hw_prob`/`printed_scripts`, and this test reaches handwriting routing through the `StubVLM`/`per_page` path, not through a `hw_prob=`/`printed_scripts=` token, so the stated grep terms do not catch it.

Because the design promises the suite stays green and the gate relies on that, this is blocking.

**CONCRETE FIX:** Add `test_e2e_heterogeneous_doc_uses_nearest_sample_and_widens_sampling` to the §5/§8/§10 inventory with an explicit rewrite: the page-10 expectation becomes `paddleocr_printed` (Latin → printed under the new router), and the "second half handwritten" comment/`per_page` hw values become meaningless and should be dropped or repurposed to script-based disagreement (e.g. pages 6–10 return a different *script* so the nearest-sample/widening behavior the test actually exercises is preserved). Broaden the mandated pre-implementation grep to include `paddleocr_handwritten` and `hw=` over `tests/`, not just `hw_prob`/`printed_scripts`.

### 2. MEDIUM — Two tests read `SampleObs.handwriting_prob` after the field is removed; neither is listed for update

The design removes `handwriting_prob` from `SampleObs` (§5 `types.py` changes). Two tests then break and are not in the teardown inventory:

- `test_vlm_parse_clips_and_normalizes` region, `tests/test_probe.py` line ~228 (verified):
  ```python
  assert (o.script, o.script_confidence, o.handwriting_prob, o.has_table) == ("devanagari", 1.0, 0.2, True)
  ```
  After the field is dropped from `SampleObs`, `o.handwriting_prob` raises `AttributeError` at runtime (pydantic v2 default is `extra='ignore'`, so the `BedrockScriptProbe` still *constructs* fine, but the attribute no longer exists to read). The design's §5 `vlm_probe.py` change ("drop `handwriting_fraction` from the prompt and the parsed `SampleObs`") makes this worse, not better — the test still asserts the attribute.
- `test_aggregate_and_disagreement_and_nearest`, line ~206 (verified): builds `SampleObs(..., handwriting_prob=.05)` / `handwriting_prob=.9`. The kwargs are silently ignored post-removal (so no crash), but the test is relying on a field the design deletes and it is not in the inventory; its `samples_disagree([a,b])` assertion happens to survive because `a`/`b` differ by script, but that is luck, not design intent.

**CONCRETE FIX:** Add both tests to the §8/§10 inventory. For the VLM-parse test, change the tuple assertion to `(o.script, o.script_confidence, o.has_table) == ("devanagari", 1.0, True)` (drop `handwriting_prob`). For the aggregate test, drop the `handwriting_prob=` kwargs from both `SampleObs(...)` constructions and keep the script-based disagreement assertion.

### 3. MEDIUM — `StubVLM` helper constructs `SampleObs(handwriting_prob=...)` and is never mentioned

`tests/test_probe.py` line ~114–122 (verified):

```python
class StubVLM:
    def __init__(self, script="latin", conf=0.95, hw=0.02, table=False, per_page=None):
        ...
    def observe(self, pn, jpeg):
        s, c, h = self.per_page.get(pn, (self.script, self.conf, self.hw))
        return SampleObs(page_number=pn, script=s, script_confidence=c, handwriting_prob=h,
                         has_table=self.table, source="vlm")
```

Every e2e test uses `StubVLM`. After `handwriting_prob` is removed, the `handwriting_prob=h` kwarg is silently dropped (pydantic `extra='ignore'`), so this does not crash — but the entire `hw`/`per_page` handwriting plumbing becomes dead, and tests that *relied* on it to drive handwriting routing (Finding 1) now behave differently with no compile-time signal. The design's modified-files list (§10) touches `tests/test_probe.py` but never mentions `StubVLM`, so an implementer following the enumerated edits will leave a misleading stub in place.

**CONCRETE FIX:** In the §10 `tests/test_probe.py` edit set, explicitly state that `StubVLM.__init__`'s `hw=`/`per_page` handwriting element and the `handwriting_prob=h` kwarg in `observe` are removed (the `per_page` tuple shrinks from `(script, conf, hw)` to `(script, conf)`), so the stub matches the new `SampleObs` and no test depends on a now-ignored field.

### 4. NIT — §4 maps lipika "Perso-Arabic" → `arabic`, but the VLM fallback's `_norm_script` has no such alias

§4 requires the export script to remap lipika's "Perso-Arabic" label to probe `arabic`. That is fine for the ONNX observer (the mapping is baked into `labels.json`). But the VLM fallback path builds `SampleObs.script` via `vlm_probe._norm_script`, which (verified) has aliases for hindi/english/punjabi/etc. and `"arabic"` is already in `SCRIPTS`, so a VLM returning `"arabic"` maps correctly — however a VLM returning `"perso-arabic"` or `"persian"` would fall through to `"other"` → VLM. This is harmless (both route to VLM now) but the design claims name-mismatch handling is fully specified; it is only specified for the ONNX path.

**CONCRETE FIX:** Add one sentence to §4 noting the mismatch remap applies to the ONNX `labels.json` only; the VLM path already routes any unrecognized script to `other`→VLM, which is route-equivalent, so no `_norm_script` change is needed. (Documentation-only; no behavior change.)

### 5. NIT — ONNX contract test rewrite guidance is vague on the `handwriting_prob` assertion it also contains

§8/§10 say to rewrite `test_onnx_classifier_contract` "for the whole-image lipika contract or replace with the §8a fake-session test." The existing test (verified, line ~379) ends with `assert obs.script == "devanagari" and obs.handwriting_prob > 0.9 and obs.source == "onnx"`. The rewrite guidance names the input-shape change (`[N,1,48,320]` → whole image) but does not call out that the `obs.handwriting_prob` assertion must be deleted (it will `AttributeError` otherwise). An implementer replacing only the model-building half could leave the assertion.

**CONCRETE FIX:** In §8a/§10, state that the rewritten observer test asserts `obs.script` and `obs.source == "onnx"` only, with **no** `handwriting_prob` assertion, matching Finding (c)'s "handwriting fields gone" checks.

### 6. NIT — `requirements-probe.txt` comment reword is slightly inaccurate vs the stated goal

§7 says reword the `onnxruntime` comment from "only used if … is set" to "required: lipika script model is baked in." The current line (verified) is `onnxruntime>=1.18.0  # only used if IDP_PROBE_ONNX_MODEL is set; ~15MB, keep for the upgrade path`. Under the §9 STOP-AND-ASK no-model fallback the model may be **absent**, in which case onnxruntime is installed but idle — so "required: … baked in" is not strictly true in that branch. Minor, but the design elsewhere is careful about the no-model state.

**CONCRETE FIX:** Reword to something branch-accurate, e.g. `# lipika script observer; model baked at src/ai/layer1_probe/models/ when present, VLM-only fallback when absent`.

---

## Verified Assumptions (checked against source)

1. `routing.py` imports `ROUTE_HANDWRITTEN, ROUTE_PRINTED, ROUTE_SKIP, ROUTE_VLM, SCRIPT_TO_PADDLE_LANG, RouteChoice, ProbeConfig`, and `decide_route` has the `hw_prob` param + `printed_scripts`/`hw_threshold` gating — confirmed. The §5 rewrite (drop `ROUTE_HANDWRITTEN` import, drop `hw_prob`, drop `printed_scripts`) is consistent with the current body.
2. `script_classifier.py::OnnxScriptClassifier.__init__(self, model_path, labels_path, session=None)` accepts an injected `session` — confirmed. The old line-crop contract (`IN_H, IN_W = 48, 320`, `_parse` of `<script>_<mode>`, `mixed_scripts = second > 0.25`) exists as described.
3. `probe.py::_observe` calls `onnx.observe(pn, gray, mask)`; two `decide_route(...)` call sites pass `hw_prob=hw` (Phase C `nonblank` loop and `timed_out` loop); blank pages build `PageProbe` directly with no `decide_route`; `_page_values` returns a 6-tuple including `handwriting_prob`; Phase A computes `hw_h = ... handwriting_heuristic(mask)`; `obs is None` fallback is `SampleObs(page_number=pn, handwriting_prob=feats[pn]["hw_h"], source="none")` — all confirmed.
4. `types.py`: `SampleObs`/`DocPrior` have `handwriting_prob`; `PageProbe` has `handwriting_prob` and `hw_source`; `ProbeConfig` has `hw_threshold`, `printed_scripts` (+ `from_env` parsing of `HW_THRESHOLD`/`PRINTED_SCRIPTS`); `ROUTE_HANDWRITTEN` defined; `SCRIPT_TO_PADDLE_LANG` keys include `latin→en`, `devanagari→devanagari`, `tamil→ta`, `telugu→te`, `kannada→ka`, `arabic→ar` — confirmed. The §4 table's lang-key column matches.
5. `aggregate.py`: `samples_disagree` has the `hw_gap` branch; `aggregate_doc` averages `handwriting_prob` and sets `DocPrior(handwriting_prob=...)`; `nearest_sample.usable` has `or s.handwriting_prob is not None` — confirmed.
6. `cv_features.py`: `handwriting_heuristic`, `line_periodicity`, `HW_PERIODICITY_TYPED`, `HW_PERIODICITY_RANGE` exist; `detect_table`, `ink_ratio` stay — confirmed.
7. `lambda_probe_handler.py`: guard `if cfg.onnx_model_path and cfg.onnx_labels_path and os.path.exists(cfg.onnx_model_path):` — confirmed (short-circuits on `None` defaults, so the §12.1 Finding-1 correction is valid). `_SLIM_FIELDS` includes `"handwriting_prob"` — confirmed.
8. `vlm_probe.py`: prompt asks for `handwriting_fraction`; `observe` sets `handwriting_prob=_clip01(data.get("handwriting_fraction"))`; `_norm_script` maps hindi→devanagari/english→latin and falls back to `other`/`unknown` — confirmed.
9. `pipeline.py`: `_ROUTE_TO_ENGINE` keys are route strings (`digital/scanned/handwritten/vlm_transcribe/skip`), values are engine names; `process_page` converges on `primary_task = plan[0] if plan else None`, then `metadata.set_routing(...)`, then `raw_results = _run_plan(plan, ...)`; `single_engine` fallback sets `single_route="scanned"` on `is_scanned or complexity_score>=4 or image_coverage>…`; `_run_engine_task` `paddleocr_printed` branch calls `convert_scanned_page(...)`; the `src.ai.schemas.page` import line does NOT include `PageProfile`; direct attribute access to `profile.is_scanned/primary_script/...` — all confirmed.
10. `imaging.py::analysis_gray(rgb, render_dpi, analysis_dpi)` computes `f = analysis_dpi/render_dpi` and downsizes when `analysis_dpi < render_dpi`, trusting arg 2 as true render DPI — confirmed.
11. `pipeline_routes.py`: `RENDER_DPI = 150` module constant; `_render_page` returns `np.array(Image.open(...).convert("RGB"))`; `_build_pages_for_pdf` builds `"context"` with `pdf_path`/`image_array`/`image_bytes`; the file does `from src.ai.layer1_routing.pipeline import process_document` (so the rejected core→web back-import would indeed be a cycle) — all confirmed. The §6 Finding-3 "stash `render_dpi` in context" fix is sound.
12. `inspect.py`: `has_text = char_count > 0`, `primary_script`, `is_scanned` are always set; `PageProfile` (`src/ai/schemas/page.py`) declares all three as required fields — confirmed. §6 Finding-2 (direct access, no `getattr`) is valid.
13. `router.py`: `_PRI_DIGITAL=1 < _PRI_PRINTED=2 < _PRI_HANDWRITTEN=3`; `build_engine_plan` sorts by priority so `plan[0]` is lowest priority; a true scan → `has_printed_scan`+`has_handwriting` → `plan[0]=paddleocr_printed`; an Indic page early-returns `has_indic_script`+`has_printed_scan` → `plan[0]=paddleocr_printed`; a digital-with-text page → `has_digital_text` → `plan[0]=docling` — all confirmed. The §6 `capability_based` table is accurate, including the honest "digital Indic page not re-probed" edge.
14. `scanned.py`: `_invoke_paddle_lambda(image_array, page_number, mode, image_bytes=None)`; payload is exactly `{"image_base64", "page_number"}`; `_ocr_convert` invokes `_invoke_paddle_lambda` at three sites (prefer-lambda primary, `ImportError` fallback, broad `except` fallback); `convert_scanned_page`/`convert_handwritten_via_paddle` exist with `image_bytes=None` — confirmed. §6 Finding-6 (thread `ocr_lang` to all three) is accurate.
15. `requirements-probe.txt` already has `onnxruntime>=1.18.0`, `numpy`, `opencv-python-headless`, `pymupdf`, `boto3`, `pydantic`; no Pillow; no torch — confirmed (§7 Finding-8 valid, Pillow correctly not added).
16. `Dockerfile.probe`: `ENV` sets `IDP_PROBE_PRINTED_SCRIPTS=latin` and `IDP_PROBE_PAGE_IMAGE_DPI=144`; `RUN mkdir -p /opt/probe_models` + commented `# COPY models/probe/`; `COPY src/ /var/task/src/`; `PYTHONPATH=/var/task` — confirmed.
17. `.dockerignore` excludes `*.bin`, `*.safetensors`, `docs/`, `tests/`, `.env*` but NOT `*.onnx`/`*.json` — confirmed (§7 model-baking and §2 docs-excluded claims valid).
18. The four validation images exist: `docs/Screenshot 2026-10-23 140614.png`, `140632.png`, `142509.png`, `142632.png` — confirmed.
19. `schema_chatbot_v2/requirements.txt` has `pillow>=11.0.0`, `numpy>=1.24.0`, no `onnxruntime`, no explicit `opencv-python-headless`; `schema_chatbot_v2/Dockerfile` is `python:3.11-slim`, apt-installs `libgl1 libglib2.0-0 libgomp1`, `PYTHONPATH=/app/schema_chatbot_v2:/app`, `COPY . /app` — all confirmed (§7 layer-3 additions valid).

## Unverified / Wrong Assumptions

1. **WRONG (Finding 1):** §5/§8/§10's "complete, verified eight-test set" is incomplete. `test_e2e_heterogeneous_doc_uses_nearest_sample_and_widens_sampling` (line ~300) asserts `paddleocr_handwritten` and is not listed; it will fail after the router rewrite.
2. **WRONG (Finding 2):** The design's claim that only the enumerated tests reference handwriting is contradicted by `test_..._vlm_parse` (`o.handwriting_prob` assertion, line ~228) and `test_aggregate_and_disagreement_and_nearest` (`handwriting_prob=` kwargs, line ~206), neither of which is in the inventory.
3. **INCOMPLETE (Finding 3):** The `StubVLM` test helper constructs `SampleObs(handwriting_prob=...)` and is used by every e2e test, but is not mentioned anywhere in the modified-files edit set.
4. **UNVERIFIED (acknowledged by the design, routed through §9):** The real lipika → ONNX export — `fontrecog`/`transformers` loadability, isolating the ConvNeXtV2-Tiny backbone + SCRIPT head, `torch.onnx.export` opset-17 success, INT8 parity on the 4 images, and lipika's exact preprocessing constants/input size. These cannot be verified without network + the export run and are correctly deferred to the STOP-AND-ASK contingency; no fabricated model is shipped. Not counted as a blocking finding because the design's §9 handles it and the test suite is designed to be model-independent via injected fake sessions (that mechanism, §3/§8, is verified against the preserved `session=` ctor).
