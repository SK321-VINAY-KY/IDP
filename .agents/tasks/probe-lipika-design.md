# Technical Design — Integrate the Layer‑1.5 Probe into the Live In‑Process Pipeline with a lipika ONNX Script Detector

Status: ready to implement. Scope: LOCAL build only. **No AWS actions of any kind** (no deploy,
no ECR push, no Lambda/ECS create/update, no boto3 calls to live services). All work is on the
current branch in `c:\Users\Dell\Desktop\IDP\engineer_a`. Python/pytest run via the repo venv at
`.venv\Scripts\python.exe`.

Prior investigation is authoritative and is NOT re‑derived here:
`.agents/tasks/live-deploy-mechanism.md` (live app runs the whole pipeline in‑process in the ECS
Fargate container via `pipeline.py::process_document`; no Step Functions; probe is dead code),
`.agents/tasks/layer1-vs-probe-routing.md` (Layer‑1 PyMuPDF routes first; the probe only touches
scanned/undetected pages), and `.agents/tasks/probe-contract-findings.md` (the four probe output
keys `image_s3_bucket`/`image_s3_key`/`ocr_lang`/`use_structure` are the engine contract and must
stay).

---

## 1. Overview

Today the probe (`src/ai/layer1_probe/*`) is dead code: nothing in the live app imports it, and the
"ONNX script classifier" it was built around (`OnnxScriptClassifier` in `script_classifier.py`)
expects a line‑crop model that was never shipped — the probe runs VLM‑only in practice. This design
does three cohesive things:

1. **Replaces the script detector** with a lipika (`loopdesk-ai/lipika`, Apache‑2.0) ONNX model that
   classifies the *whole page image* into one of 13 Indic scripts + a Latin OOD class, exported
   offline into the repo and baked into the image.
2. **Simplifies routing to a script‑only model** (plus blank): Latin → `paddleocr_printed` (ocr_lang
   `en`); any other detected script with enough confidence → `vlm_transcribe` (for now); blank →
   `skip`; unknown/low‑confidence → `vlm_transcribe`. The handwriting signal is dropped entirely and
   the `paddleocr_handwritten` route becomes unreachable (engine code stays).
3. **Wires the probe into the live in‑process pipeline** (`pipeline.py::process_document` →
   `process_page`) behind a default‑OFF feature flag, threading `ocr_lang` into the Paddle Lambda
   payload only when present.

The guiding constraint is **inertness in prod**: with the flag off, `process_document` behaves
byte‑for‑byte as today. The probe is additive and refinement‑only, exactly as the existing
Step‑Functions design intended, but relocated to the in‑process seam that is actually live.

**Technology stack (locked once approved):**

- Offline export (STEP 1, repo venv only, never in runtime): `torch`, `transformers`, `fontrecog`
  (lipika's package), `onnx`, `onnxruntime`, `optimum`/`onnxruntime.quantization` for
  `quantize_dynamic`. These are *dev/offline* dependencies, installed ad hoc into `.venv`, **not**
  added to any runtime requirements file.
- Runtime (probe observer + live app): `onnxruntime`, `numpy`, `Pillow`, `opencv-python-headless`.
  **No torch, no transformers, no fontrecog at runtime.**
- Model format: ONNX (FP32 shipped default) + an INT8 dynamic‑quantized variant produced regardless;
  INT8 only becomes the default if it preserves the 4 validation predictions.
- Pipeline language/framework: unchanged — Python 3.11 (web app image) / 3.12 (probe Lambda image),
  pydantic v2, the existing `ProbeConfig`/`SampleObs`/`PageProbe` contracts.

---

## 2. STEP 1 — Offline ONNX export (`scripts/export_lipika_script_onnx.py`)

### Purpose and placement

A one‑time, developer‑run script under `scripts/`. It installs/imports lipika + torch **into the
repo venv** (torch is ~GB and offline‑only — it must never appear in a runtime requirements file),
loads `loopdesk-ai/lipika`, exports only the **ConvNeXtV2‑Tiny backbone + the 13‑way SCRIPT head**
to ONNX (dropping the family/weight/italic heads and the retrieval index), validates, quantizes, and
re‑validates. Outputs land in `src/ai/layer1_probe/models/`.

### lipika loading and head selection

Per the model card, lipika loads via:

```python
from fontrecog.infer.predictor import Predictor
predictor = Predictor.from_pretrained("loopdesk-ai/lipika")   # model.safetensors + config.json + label_map.json
result = predictor.predict(pil_rgb_image)                      # result["script"] == "telugu" / "devanagari" / "latin" ...
```

The script must introspect the loaded `torch.nn.Module` to find (a) the ConvNeXtV2‑Tiny backbone and
(b) the script classification head, then build a thin `nn.Module` wrapper whose `forward(x)` returns
**only the script logits** (`backbone(x) -> pooled features -> script_head -> logits[13 or 14]`). The
family/weight/italic heads and any retrieval/index tensors are excluded from the traced graph.
`config.json` / `label_map.json` give the script index → name mapping; the script writes that mapping
to the sidecar metadata JSON (see below) — it is **not** hard‑coded.

Export via `torch.onnx.export` with:
- input name `pixel_values`, dynamic batch axis (axis 0), fixed spatial size taken from lipika's
  preprocessing config (ConvNeXtV2‑Tiny is typically 224×224 RGB — the script reads the real value
  from lipika's processor/config rather than assuming);
- opset 17 (matches a modern `onnxruntime>=1.18`);
- output name `script_logits`.

### Preprocessing fidelity (document the choice)

lipika's native `predict` does **patch‑voting over a whole RGB image** with RGB normalization. Two
options for what the ONNX graph + runtime observer reproduce:

- **Option A — single forward on a center/resized whole image** with lipika's exact RGB mean/std
  normalization and input size. Simpler graph, one `session.run`, deterministic, trivially testable.
- **Option B — patch‑voting** replicated in the observer (tile the page, run N forwards, average
  softmax). Closer to native accuracy on busy pages but N× slower and more code.

**Decision: ship Option A (single forward).** The probe already *downscales a page region* before
classification and only needs a document‑level script prior over 2–3 sampled pages, not per‑glyph
accuracy. Single‑forward keeps the runtime observer a thin `onnxruntime` call, keeps the ONNX graph
free of a voting loop, and is fully mockable in tests. The export script records the exact
normalization constants and input size into the metadata JSON so the runtime observer uses identical
preprocessing. **If Option A fails validation** (drops any of the 4 known predictions vs torch),
fall back to replicating patch‑voting in the observer (preprocessing in Python, the ONNX graph still
single‑forward per patch) and note it in the metadata JSON (`"preprocessing": "patch_vote"`). The
export script prints which mode it settled on.

### Outputs

- `src/ai/layer1_probe/models/lipika_script.onnx` — FP32 backbone+script‑head.
- `src/ai/layer1_probe/models/lipika_script.int8.onnx` — INT8 dynamic‑quantized (saved regardless).
- `src/ai/layer1_probe/models/lipika_script.labels.json` — metadata:
  `{"labels": ["latin","devanagari",...], "input_size": [224,224], "mean": [...], "std": [...],
  "preprocessing": "single_forward|patch_vote", "shipped_default": "fp32|int8",
  "source_model": "loopdesk-ai/lipika", "exported_at": "..."}`. The `labels` array is the
  **script‑index → probe‑vocabulary name** mapping already remapped per §4 (so the runtime observer
  never needs lipika's raw names).

### Validation + quantization reporting

Validation runs in the **repo venv only** (STEP 1 is a developer‑run offline script, never executed in
Docker); `docs/` is excluded by `.dockerignore` and is **never needed in any image** — the §7
`docker build` smoke test does not and must not reproduce this validation (Finding 5).

Validate on the four `docs/` images (confirmed present):
- `docs/Screenshot 2026-10-23 140614.png` → expect `latin`
- `docs/Screenshot 2026-10-23 140632.png` → expect `latin`
- `docs/Screenshot 2026-10-23 142509.png` → expect `telugu`
- `docs/Screenshot 2026-10-23 142632.png` → expect `devanagari`

For each image, print `predicted_script, confidence` for **(1) torch `predictor.predict`,
(2) FP32 ONNX, (3) INT8 ONNX**, side by side. Then:
- If all three agree on the 4 labels and INT8 confidences are within a small tolerance → set
  `shipped_default = "int8"` in metadata.
- **If INT8 flips any of the 4 predictions → keep FP32 as the shipped default**, say so in stdout,
  and still save `lipika_script.int8.onnx` for the record.

The export script is **not** run by CI and is **not** imported at runtime; it is purely a developer
tool. It must guard its heavy imports so that merely importing the module (e.g. accidental test
collection) does not import torch — do the `import torch`/`import fontrecog` inside `main()`.

### Error handling (STEP 1)

- lipika/fontrecog install or import fails, or the head cannot be isolated/traced → the script exits
  non‑zero with a clear message and **writes nothing**. This triggers the §9 stop‑and‑ask fallback;
  the implementer must NOT fabricate a model or substitute another.
- `torch.onnx.export` raises (unsupported op) → report the op, exit non‑zero. Same fallback.
- A validation image is missing → warn and skip that image but continue; exit non‑zero only if
  zero images could be validated.

---

## 3. STEP 2 — lipika‑ONNX observer (`script_classifier.py`)

### Interface (unchanged contract the probe calls)

`probe.py::_observe` calls `onnx.observe(page_number, gray, mask) -> Optional[SampleObs]` (verified:
`src/ai/layer1_probe/probe.py`, inside `_observe`, `obs = onnx.observe(pn, gray, mask)`). The new
observer keeps that positional signature so `probe.py` needs **no change to its call site**, and adds
one optional keyword‑only `rgb=None` that only the §6 in‑process wrapper uses (see the input contract
below). It is a drop‑in replacement for `OnnxScriptClassifier`.

**Decision: replace the class in‑place but rename it `LipikaOnnxScriptObserver`, and keep
`OnnxScriptClassifier = LipikaOnnxScriptObserver` as a module‑level alias** so
`lambda_probe_handler.py::_load_models` (which imports `OnnxScriptClassifier`) and existing tests
that import the old name keep working. The old line‑crop logic (`extract_line_crops`, `_prep`, the
`48×320` grayscale contract, `_parse` of `"<script>_<mode>"` labels) is **removed** — lipika is a
whole‑image RGB classifier, not a line‑crop print/hw model.

**Preserve the `session=` constructor injection (Finding 4).** The current
`OnnxScriptClassifier.__init__(self, model_path, labels_path, session=None)` accepts a pre‑built
`session` (verified `script_classifier.py`). `LipikaOnnxScriptObserver.__init__` **keeps this exact
`session=None` keyword**: when a `session` is passed, it is used verbatim and no `onnxruntime` import
happens; when `None`, `_get_session()` builds the module singleton from the resolved model path. This
injection point is the single mechanism every observer‑dependent test uses to supply a **fake session**
whose `.run(...)` returns canned logits (§8), so the routing/integration tests never load a real model
and stay green under the §9 no‑model fallback. The module‑singleton path (`_get_session()`) is used
only in production; a test that needs the singleton behavior builds it through a `session=` fake and a
`_reset_for_tests()` hook (§11) to assert "constructed once."

### New implementation

```python
# Module scope — singleton session, built ONCE, reused across warm invocations.
_SESSION = None          # onnxruntime.InferenceSession | None
_META = None             # parsed labels.json
_INPUT_NAME = None
```

- `observe(page_number, gray, mask, *, rgb=None)`:
  1. **Input contract (two call sites, two input kinds — verified against real code):** the
     Step‑Functions probe path (`probe.py::_observe` → `onnx.observe(pn, gray, mask)`) only has the
     **grayscale** analysis image, while the in‑process pipeline path (§6, built from
     `pipeline_routes.py::_render_page` which returns `np.array(Image.open(...).convert("RGB"))` —
     confirmed a true RGB array) has the **real RGB** page. To serve both correctly and keep the
     existing positional signature intact, add an optional keyword `rgb: Optional[np.ndarray] = None`.
     When `rgb` is provided (in‑process), classify from it; when it is `None` (Step‑Functions), the
     observer reconstructs a 3‑channel image via `cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)`. Color is
     not required for script ID (lipika is robust), so grayscale→RGB replication is an acceptable
     fallback; using the real RGB when available is strictly better and free. **`probe.py`'s call
     site is unchanged** (`rgb` defaults to `None`); only the §6 in‑process wrapper passes `rgb=`.
  2. **Downscale, not full‑res:** resize the chosen 3‑channel image to the metadata `input_size`
     (e.g. 224×224) with lipika's normalization (per‑channel mean/std, 0–1 scaling) read from
     `labels.json`. This matches the task's "feed a downscaled page region, not full‑res" requirement
     — the model input is a small fixed square regardless of page size.
  3. `logits = _SESSION.run(None, {_INPUT_NAME: x})[0]` → softmax → top‑1 index + prob.
  4. Map index → script name via `_META["labels"]` (already in probe vocabulary from §4).
  5. Return `SampleObs(page_number=pn, script=<mapped>, script_confidence=<top prob>, source="onnx")`.
     **No `handwriting_prob`** (field removed per §5). `mixed_scripts` is set from a simple
     margin test (`second_prob > _MIXED_MARGIN`) so the existing disagreement/VLM‑fallback logic still
     functions. **The threshold is provisional (Finding 7):** the old `0.25` was tuned for per‑crop
     *averaged* votes, and a single whole‑image 14‑class softmax has a different second‑place
     distribution, so `0.25` is not re‑validated. **Decision: expose it as a named module‑level
     constant `_MIXED_MARGIN = 0.25` in `script_classifier.py`** (not a magic literal), with a comment
     that it is a provisional carry‑over to be re‑tuned once real lipika logits are available; it only
     pushes borderline pages to the VLM (the safe direction). Promoting it to a `ProbeConfig` field is
     deferred — a module constant keeps the observer self‑contained and is still trivially
     monkeypatchable in tests.

### Singleton construction

The `InferenceSession` is created **once at module scope** via a `_get_session()` that lazily builds
and caches `_SESSION` on first `observe`/first explicit `init`. Tests assert the session is built
exactly once across multiple `observe` calls (see §8a). Model path resolution:

- Reuse the existing env var **`IDP_PROBE_ONNX_MODEL`** for the model path (keeps
  `ProbeConfig.onnx_model_path` wiring intact). Default when unset: the baked‑in
  `src/ai/layer1_probe/models/lipika_script.onnx` (or `.int8.onnx` if metadata
  `shipped_default == "int8"`), resolved relative to the module file.
- Labels/metadata path: reuse **`IDP_PROBE_ONNX_LABELS`**, default
  `src/ai/layer1_probe/models/lipika_script.labels.json` next to the model.
- Expose these defaults as **module‑level functions** on `script_classifier.py` so callers that do
  not thread env vars can resolve them:
  ```python
  _HERE = os.path.dirname(os.path.abspath(__file__))
  def default_model_path() -> str:
      # prefer int8 iff the sidecar metadata says so, else fp32; both resolved module‑relative
      ...  # returns src/ai/layer1_probe/models/lipika_script[.int8].onnx
  def default_labels_path() -> str:
      return os.path.join(_HERE, "models", "lipika_script.labels.json")
  ```
  `_get_session()` uses `IDP_PROBE_ONNX_MODEL` if set, else `default_model_path()`; likewise labels.

**Step‑Functions handler reconciliation (Finding 1 — the previous draft's claim was wrong).** The
existing guard in `lambda_probe_handler.py::_load_models` is
`if cfg.onnx_model_path and cfg.onnx_labels_path and os.path.exists(cfg.onnx_model_path):` — verified
in `src/api/lambda_probe_handler.py`. `cfg.onnx_model_path`/`cfg.onnx_labels_path` come from
`ProbeConfig.from_env()` and default to `None` when `IDP_PROBE_ONNX_MODEL`/`IDP_PROBE_ONNX_LABELS`
are unset, and `Dockerfile.probe` does **not** set those env vars. So the guard short‑circuits on
`None and …` and the observer is **never** constructed on the Step‑Functions path, no matter whether
the baked file exists. The "module‑relative default" lives only inside the observer and the handler
never reaches it.

**Decision: make `_load_models` fall back to the observer's module‑relative defaults when the env
vars are unset**, so the baked model loads on *both* the Step‑Functions path and the in‑process path
from one source of truth. Change the handler to:

```python
from src.ai.layer1_probe import script_classifier
model_path = cfg.onnx_model_path or script_classifier.default_model_path()
labels_path = cfg.onnx_labels_path or script_classifier.default_labels_path()
if os.path.exists(model_path):
    from src.ai.layer1_probe.script_classifier import OnnxScriptClassifier  # alias → LipikaOnnxScriptObserver
    _onnx = OnnxScriptClassifier(model_path, labels_path)
    logger.info("probe.onnx_loaded", path=model_path)
```

This keeps `Dockerfile.probe`'s `IDP_PROBE_ONNX_MODEL`/`IDP_PROBE_ONNX_LABELS` as **optional
overrides only** (consistent with §7 — not set by default), and under the §9 no‑model fallback
`os.path.exists(model_path)` is `False`, so `_onnx` stays `None` and the handler runs VLM‑only exactly
as today. `lambda_probe_handler.py` is therefore in the §5/§10 modified‑files set (two‑line change:
the default‑path fallback + reusing the one import). For the in‑process pipeline (§6) the observer is
constructed through the same singleton via `inprocess.py`, which also relies on these module‑relative
defaults. This makes the handler and the in‑process hook load the identical baked model with zero
env‑var configuration.

### VLM fallback intact

`probe.py::_observe` keeps its structure: run ONNX first; if `obs is None`, `script_confidence <
cfg.onnx_min_conf`, or `mixed_scripts`, fall through to the VLM observer when enabled. The VLM
`SampleObs` must also drop `handwriting_prob` (§5) — `vlm_probe.py` stops populating it. The
`obs is None` final fallback `SampleObs(page_number=pn, source="none")` no longer passes
`handwriting_prob=feats[pn]["hw_h"]`.

### Runtime deps

`onnxruntime` + `numpy` + `Pillow`/OpenCV only. **No torch.** `onnxruntime` and `opencv-python-headless`
are already in `requirements-probe.txt`; `numpy` too. See §7 for the Layer‑3 app image.

### Error handling (STEP 2)

- Model file missing at the resolved path → `_get_session()` returns `None`; `observe` returns `None`
  (probe falls back to VLM). Logged once at WARNING (`probe.onnx_model_missing`, model path by name).
  This is recoverable — the pipeline still works via VLM.
- `onnxruntime.InferenceSession` construction raises (corrupt model, unsupported opset) → caught,
  logged once at ERROR, `_SESSION` set to a sentinel so we do not retry per call; `observe` returns
  `None`. Recoverable (VLM fallback).
- `session.run` raises on a page → caught, logged at WARNING with `page_number`, return `None` for
  that page. Recoverable.
- Label index out of range vs `_META["labels"]` → treat as `script="unknown", confidence=0.0`
  (routes to VLM). Logged at WARNING.

---

## 4. Script‑name mapping (lipika → probe vocabulary)

lipika's 13‑way set maps onto the probe `SCRIPTS` vocabulary (`types.py`) and the
`SCRIPT_TO_PADDLE_LANG` keys. The export script writes the already‑mapped names into
`labels.json`, so the runtime observer uses probe names directly. Mapping table (every mismatch
documented):

**The `In SCRIPT_TO_PADDLE_LANG?` column is reference‑only (Finding 6).** Routing gates **strictly on
`script == "latin"`** (§5 `decide_route` step 4); the `ka`/`ta`/`te`/… lang keys are informational and
must **not** be wired into a per‑script Paddle path in this build. Every non‑Latin script → VLM,
regardless of whether a Paddle lang key exists.

| lipika label        | probe `SCRIPTS` name | In `SCRIPT_TO_PADDLE_LANG`? (ref only) | Note |
|---------------------|----------------------|-----------------------------|------|
| Latin (OOD class)   | `latin`              | yes → `en`                  | Exact. The only script routed to Paddle. |
| Devanagari          | `devanagari`         | yes → `devanagari`          | Exact. (Routed to VLM now — see §5.) |
| Bengali             | `bengali`            | **no**                      | In `SCRIPTS`; no Paddle lang key → VLM. |
| Gujarati            | `gujarati`           | **no**                      | In `SCRIPTS`; → VLM. |
| Gurmukhi            | `gurmukhi`           | **no**                      | In `SCRIPTS`; → VLM. |
| Kannada             | `kannada`            | yes → `ka`                  | Lang key exists but routed to VLM now. |
| Malayalam           | `malayalam`          | **no**                      | In `SCRIPTS`; → VLM. |
| Meetei Mayek        | `other`             | **no**                      | **Not in `SCRIPTS`** → map to `other` → VLM. |
| Odia                | `odia`               | **no**                      | In `SCRIPTS`; → VLM. |
| Ol Chiki            | `other`             | **no**                      | **Not in `SCRIPTS`** → map to `other` → VLM. |
| Perso‑Arabic        | `arabic`             | yes → `ar`                  | **Name mismatch:** lipika "Perso‑Arabic" → probe `arabic`. Routed to VLM now. |
| Tamil               | `tamil`              | yes → `ta`                  | Exact. Routed to VLM now. |
| Telugu              | `telugu`             | yes → `te`                  | Exact. Routed to VLM now. |

Mismatches to encode in the export mapping: **Meetei Mayek** and **Ol Chiki** have no probe‑vocabulary
entry → `other`; **Perso‑Arabic** → `arabic`. Everything else is a direct lowercase match. No change
to `SCRIPTS` or `SCRIPT_TO_PADDLE_LANG` is required — but see §5 on why only `latin` matters for
routing now.

---

## 5. STEP 3 — Script‑only routing + schema changes

### `routing.py::decide_route` rewrite

New signature (drop `hw_prob`):

```python
def decide_route(*, script: str, script_conf: float, mixed: bool,
                 is_blank: bool, has_table: bool, cfg: ProbeConfig) -> RouteChoice:
```

Decision order:

1. `is_blank` → `RouteChoice(ROUTE_SKIP, reason="blank_page")`.
2. `script in ("unknown","other") or script_conf < cfg.min_script_conf` → `ROUTE_VLM`,
   `reason=f"script_unresolved({script},{conf:.2f})"`.
3. `mixed` → `ROUTE_VLM`, `reason="mixed_scripts"`.
4. `script == "latin"` → `ROUTE_PRINTED`, `ocr_lang="en"` (from `SCRIPT_TO_PADDLE_LANG["latin"]`),
   `use_structure=has_table`, `reason="latin_print"`.
5. **else (any other detected script with sufficient confidence)** → `ROUTE_VLM`,
   `reason=f"{script}_to_vlm_no_dedicated_ocr"`. **Add a code comment:** dedicated per‑script OCR
   engines (Telugu/Devanagari/Tamil/…) are a future goal; for now every non‑Latin script goes to the
   VLM.

There is **no** handwriting branch and **no** `paddleocr_handwritten` return anywhere. `use_structure`
stays recorded for the Latin/printed path exactly as today (it is the `has_table` CV signal; the
engine records it but does not act on it per the contract findings).

**Final `routing.py` top‑of‑file imports (Finding 5 — must land atomically with the body).** The
current file imports
`ProbeConfig, RouteChoice, SCRIPT_TO_PADDLE_LANG, ROUTE_HANDWRITTEN, ROUTE_PRINTED, ROUTE_SKIP, ROUTE_VLM`
(verified). The rewritten body no longer references `ROUTE_HANDWRITTEN` (no handwriting branch) or
`cfg.printed_scripts`/`cfg.hw_threshold`. The new import block is exactly:

```python
from src.ai.layer1_probe.types import (
    ProbeConfig, RouteChoice, SCRIPT_TO_PADDLE_LANG,
    ROUTE_PRINTED, ROUTE_SKIP, ROUTE_VLM,
)
```

i.e. **drop `ROUTE_HANDWRITTEN` from `routing.py`'s import** (the constant still *exists* in `types.py`
per §5 — it is just no longer imported here). `SCRIPT_TO_PADDLE_LANG` is kept because step 4 still does
`SCRIPT_TO_PADDLE_LANG["latin"]`. The full rewritten body (for implementer precision):

```python
def decide_route(*, script: str, script_conf: float, mixed: bool,
                 is_blank: bool, has_table: bool, cfg: ProbeConfig) -> RouteChoice:
    if is_blank:
        return RouteChoice(route=ROUTE_SKIP, reason="blank_page")
    if script in ("unknown", "other") or script_conf < cfg.min_script_conf:
        return RouteChoice(route=ROUTE_VLM, reason=f"script_unresolved({script},{script_conf:.2f})")
    if mixed:
        return RouteChoice(route=ROUTE_VLM, reason="mixed_scripts")
    if script == "latin":
        return RouteChoice(route=ROUTE_PRINTED, ocr_lang=SCRIPT_TO_PADDLE_LANG["latin"],
                           use_structure=has_table, reason="latin_print")
    # Future: dedicated per-script OCR (Telugu/Devanagari/Tamil/…). For now every
    # non-Latin detected script escalates to the VLM.
    return RouteChoice(route=ROUTE_VLM, reason=f"{script}_to_vlm_no_dedicated_ocr")
```

This body references no removed `ProbeConfig` field (`printed_scripts`, `hw_threshold`), so once
`types.py` drops those fields the module still imports and routes without `AttributeError`.

### `PRINTED_SCRIPTS` / `cfg.printed_scripts` fate (judgment call, explained)

Today routing gates on `script in cfg.printed_scripts` (default `{"latin"}`) **and** a Paddle lang
key. The new rule is "Latin → Paddle, everything else → VLM," which is strictly `script == "latin"`.
`printed_scripts` is therefore **no longer a routing gate**.

**Decision: remove `printed_scripts` from the routing logic entirely, and remove the field +
`IDP_PROBE_PRINTED_SCRIPTS` parsing from `ProbeConfig`.** Rationale: keeping a dead config field that
silently does nothing is a worse long‑term trap than removing it — the task says "prefer removing/
ignoring it cleanly." The `Dockerfile.probe` currently sets `IDP_PROBE_PRINTED_SCRIPTS=latin`; that
ENV line is removed in §7. If a future per‑script‑OCR effort needs an allowlist it can reintroduce a
purpose‑named field. This is low‑risk because the only consumers are `decide_route` (being rewritten),
`from_env` (removed), and the Dockerfile ENV (removed) — confirmed by repo‑wide grep (§5 grep list).

### `types.py` changes

- `SampleObs`: **remove `handwriting_prob`**. Keep `script`, `script_confidence`, `mixed_scripts`,
  `has_table`, `source`.
- `DocPrior`: **remove `handwriting_prob`**.
- `PageProbe`: **remove `handwriting_prob` and `hw_source`**. Keep `use_structure`, `ocr_lang`,
  `has_table`, `image_s3_*`, etc. — the engine contract keys are untouched.
- `ProbeConfig`: remove `hw_threshold`, `printed_scripts`, and their `from_env` parsing
  (`HW_THRESHOLD`, `PRINTED_SCRIPTS`). Keep `min_script_conf`, `onnx_min_conf`, `onnx_model_path`,
  `onnx_labels_path`, VLM knobs, rendering/blank/table knobs.
- `RouteChoice`: unchanged.
- `ROUTE_HANDWRITTEN` constant **stays defined** (engine code and `_ROUTE_TO_ENGINE` in pipeline.py
  still reference the handwritten engine); it is simply never returned by `decide_route`.

### `probe.py` changes (Phase A/C)

- Phase A: stop computing `hw_h` (`cv_features.handwriting_heuristic(mask)` call removed). The
  `feats` dict drops `hw_h`.
- `_observe`: the `obs is None` fallback becomes `SampleObs(page_number=pn, source="none")` (no
  `handwriting_prob`).
- Phase C `_page_values`: return tuple drops the handwriting element; callers stop unpacking it. The
  current tuple is `(script, script_confidence, handwriting_prob, mixed_scripts, source, samp)`
  (verified) and the handwriting element plus both `prior.handwriting_prob`/`samp.handwriting_prob`
  reads are removed, so the new tuple is `(script, script_confidence, mixed_scripts, source, samp)`.
- **Both `decide_route(...)` call sites in `probe.py` pass `hw_prob=hw` today and MUST be updated in
  the same edit (Finding 5 — verified exact lines):**
  1. **Phase C page build** — `choice = decide_route(script=script, script_conf=sconf, hw_prob=hw,
     mixed=mixed, is_blank=False, has_table=has_table, cfg=cfg)` (in the `for pn in nonblank:` loop).
  2. **Timeout/prior‑only build** — `choice = decide_route(script=script, script_conf=sconf,
     hw_prob=hw, mixed=mixed, is_blank=False, has_table=False, cfg=cfg)` (in the `for pn in
     timed_out:` loop).
  Both drop the `hw_prob=hw` argument to match the new keyword‑only signature. (There is no third
  call — the blank path builds `PageProbe` directly without `decide_route`, verified.) The
  surrounding `PageProbe(...)` constructions in both loops also drop `handwriting_prob=`/`hw_source=`,
  and the Phase A `hw_h`/`hw_source` plumbing feeding them is removed.

### `aggregate.py` changes

- `samples_disagree`: drop the `hw_gap` handwriting‑spread branch; disagreement is now script‑only
  (more than one distinct confident script).
- `aggregate_doc`: drop `handwriting_prob` averaging and the `DocPrior(handwriting_prob=...)` kwarg.
- `nearest_sample`: `usable` filter drops the `or s.handwriting_prob is not None` clause (now just
  `s.script != "unknown"`).

### `cv_features.py` changes

`handwriting_heuristic` and `line_periodicity` become unused in **runtime** once Phase A stops calling
them (Phase A's only caller — confirmed in `probe.py`:
`hw_h = None if blank else cv_features.handwriting_heuristic(mask)`). **Decision: remove both** (and the
`HW_PERIODICITY_TYPED`/`HW_PERIODICITY_RANGE` constants), since no *runtime or test‑kept* code needs
them after the teardown. `detect_table`, `ink_ratio`, and the blank logic **stay**.

**Non‑runtime references that MUST be handled in the same change** (verified by repo‑wide grep — the
earlier "only `calibrate_probe_heuristic.py`" claim was incomplete):
- `scripts/calibrate_probe_heuristic.py` → calls `cvf.line_periodicity(...)`. It is an offline helper,
  never imported by runtime or tests. **Decision: delete the file** (cleaner than leaving a function
  that imports a now‑removed symbol; it has no runtime role and keeping a dangling deprecated helper is
  the trap §5's `printed_scripts` decision argues against).
- `tests/test_probe.py` → `test_handwriting_heuristic_orders_wobbly_above_typed` and
  `test_heuristic_returns_none_when_too_little_text` call `cv_features.handwriting_heuristic`. These two
  tests exercise a feature being deleted. **Decision: remove both tests** as part of §8's
  `test_probe.py` update (they would otherwise fail at import/collection once the function is gone).
- `.scratch_probe/run_probe_signals.py` and `.scratch_probe/run_probe_signals2.py` → both call
  `cv_features.handwriting_heuristic` and `line_periodicity`. These are throwaway scratch scripts (a
  dot‑prefixed scratch dir), not runtime, not tests, not packaged. **Decision: leave them untouched**
  (they are not collected by pytest, not imported anywhere, and not shipped) — but note them here so
  the implementer is not surprised by the grep hits. If desired they can be deleted; the design does
  not require touching `.scratch_probe/`.

### `lambda_probe_handler.py` changes

`_SLIM_FIELDS` currently lists `"handwriting_prob"`. **Remove it** from the tuple. The slim output and
S3 report keep the four engine‑contract keys. This handler is Step‑Functions‑only (not live) but is
kept consistent so the probe module set compiles and its test passes.

### `vlm_probe.py` changes

`BedrockScriptProbe.observe` stops setting `handwriting_prob`. The prompt may keep asking for
`handwriting_fraction` (harmless) or drop it; **Decision: drop `handwriting_fraction` from the prompt
and the parsed `SampleObs`** to keep the no‑handwriting contract clean. `_norm_script` already maps
aliases (hindi→devanagari, english→latin, etc.) and is reused.

### Backward‑compat of the probe OUTPUT dict (contract preserved)

Per `probe-contract-findings.md`, the engine‑facing keys are `image_s3_bucket`, `image_s3_key`,
`ocr_lang`, `use_structure`. All four remain on `PageProbe`. Only the handwriting fields are removed.
`ocr_lang` is still a PaddleOCR lang code string (`"en"` for Latin, `None` otherwise); `use_structure`
is still the recorded `has_table` bool on the printed path.

### Repo‑wide cleanup grep

Before finishing STEP 3, grep and clean the probe path for:
`handwriting_prob`, `hw_prob`, `hw_source`, `handwriting_heuristic`, `line_periodicity`,
`hw_threshold`, `HW_THRESHOLD`, `printed_scripts`, `PRINTED_SCRIPTS`. **Known hits (full list, verified
by grep):**
- Runtime/packaged, must be edited: `types.py`, `routing.py`, `probe.py`, `aggregate.py`,
  `script_classifier.py`, `vlm_probe.py`, `cv_features.py`, `lambda_probe_handler.py`,
  `Dockerfile.probe`.
- Tests, must be edited (see §8): `tests/test_probe.py`. **Full verified list of every test that
  references `hw_prob`, `printed_scripts`, `handwriting_prob`, or `paddleocr_handwritten`** (re‑grepped
  over `tests/` — the earlier list was incomplete and missed three routing tests, per review Finding 1):
  - `test_routing_table` (line ~167) — parametrized rows pass `hw_prob=…` and one expects
    `paddleocr_handwritten`. Disposition in §8b: drop the `hw_prob` kwarg from every row and remove the
    handwritten row.
  - `test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm` (line ~180) — builds
    `_cfg(printed_scripts={"latin","devanagari"})` and calls `decide_route(..., hw_prob=.0, …)`
    expecting Devanagari → `paddleocr_printed`. **Disposition: rewrite to the new rule** — Devanagari →
    `vlm_transcribe` (drop both `printed_scripts=` and `hw_prob=`); the mixed‑Latin half already expects
    VLM and stays once `hw_prob=`/`printed_scripts=` are removed. (See §8b.)
  - `test_table_sets_structure_hint_only_for_printed` (line ~188) — two `decide_route(..., hw_prob=.0/.9, …)`
    calls; the second asserts `hw_prob=.9 → use_structure False`, which has no meaning under a
    handwriting‑free router. **Disposition: drop the `hw_prob=` args and the `hw_prob=.9` assertion;
    replace with a Latin‑only `use_structure` check** (`script="latin", has_table=True → use_structure
    True`; `has_table=False → use_structure False`). (See §8b.)
  - `test_blank_routes_to_skip` (line ~195) — `decide_route(..., hw_prob=0, …)`. **Disposition: drop the
    `hw_prob=0` arg; keep the blank → `skip` assertion.** (See §8b.)
  - `test_e2e_devanagari_handwritten_goes_to_vlm_and_latin_hw_to_handwritten_engine` — asserts the
    now‑removed handwriting routing. Disposition in §8: rewrite to the script‑only expectation
    (Devanagari → VLM, Latin → printed).
  - `test_onnx_classifier_contract` — builds the old `[N,1,48,320]` line‑crop ONNX. Disposition in §8:
    rewrite for the whole‑image lipika contract or replace with the §8a fake‑session test.
  - `test_handwriting_heuristic_orders_wobbly_above_typed` and
    `test_heuristic_returns_none_when_too_little_text` — exercise the deleted `handwriting_heuristic`.
    Disposition: **remove both** (per the `cv_features.py` section).
  **Before implementation, re‑run the grep** (`hw_prob`, `printed_scripts`) over `tests/` to confirm no
  further case appears; the eight entries above are the complete current set.
- Non‑runtime: `scripts/calibrate_probe_heuristic.py` (**delete**, per the `cv_features.py` section);
  `.scratch_probe/run_probe_signals.py` and `.scratch_probe/run_probe_signals2.py` (scratch, not
  collected/imported/shipped — **leave as‑is**, documented so the grep hits are expected).

---

## 6. STEP 4 — Wire the probe into the live in‑process pipeline

### The exact seam

The live path is `schema_chatbot_v2/app/api/pipeline_routes.py` → `process_document(pages, ...)` →
`process_page(...)` in `src/ai/layer1_routing/pipeline.py`. Each `pages[i]` already carries a rendered
`context` dict with `pdf_path`, `image_array` (numpy RGB) and `image_bytes` (PNG) — confirmed in
`_build_pages_for_pdf`. So the probe does **not** need S3 or PyMuPDF rendering here; it classifies the
already‑rendered page image in memory.

**Decision: hook at the single converged point in `process_page` where the plan exists but has not
yet run — immediately after `primary_task = plan[0] if plan else None` is assigned and BEFORE
`_run_plan(plan, page_context, llm_client)` is called** (verified line order in
`src/ai/layer1_routing/pipeline.py::process_page`: the `capability_based` branch builds
`plan = build_engine_plan(caps)`; the `else`/`single_engine` branch builds
`plan = [EngineTask(engine=_ROUTE_TO_ENGINE.get(single_route, single_route), ...)]`; both then fall
through to `primary_task = plan[0] if plan else None`, the `metadata.set_routing(...)` call, and then
`raw_results = _run_plan(plan, ...)`). Hooking at `primary_task`/`plan[0]` is mode‑agnostic by
construction, so it covers **both** `single_engine` and `capability_based` without per‑branch code and
without depending on which `ROUTING_MODE` the live ECS task runs. (The deploy findings note the live
task may run `capability_based`; pinning the hook to the converged plan rather than to `single_route`
removes any dependence on that.)

**Route‑string vs engine‑name — a correctness detail the implementer must not get wrong.** The
pipeline's `plan[0].engine` holds an **engine name** (`docling`, `paddleocr_printed`,
`paddleocr_handwritten`, `vlm_transcribe`, `skip`) — these are the *values* of `_ROUTE_TO_ENGINE`, not
its route‑string *keys* (`digital`, `scanned`, `handwritten`, …). The probe's `decide_route` already
returns `RouteChoice.route` using the **same engine‑name constants** (`ROUTE_PRINTED =
"paddleocr_printed"`, `ROUTE_VLM = "vlm_transcribe"`, `ROUTE_SKIP = "skip"` in `types.py` — verified).
Therefore the override assigns `choice.route` **directly** onto `plan[0].engine`; it must **NOT** pass
`choice.route` through `_ROUTE_TO_ENGINE` (that table is keyed by route strings and would miss on
`"paddleocr_printed"`). The earlier draft's phrasing "map via `_ROUTE_TO_ENGINE`" was wrong and is
corrected here: `choice.route` is already an engine name.

The hook decides per page:

- Layer‑1 produced `profile`, and the plan's primary engine was chosen from it. If the page is
  digital/Docling (`plan[0].engine == "docling"`, i.e. printed text with an embedded layer), the probe
  is **not** invoked (passthrough, unchanged).
- If the page is scanned/undetected (see the gating predicate below), the probe runs on the page's
  in‑memory RGB image to detect script + table + blank, then `decide_route` picks the engine.

Implement a small helper
`_maybe_probe_route(profile, page_context, primary_engine: str) -> Optional[RouteChoice]` and call it
once, right after `primary_task` is assigned. When it returns a `RouteChoice`, rebuild the primary
task so the override is visible to the rest of `process_page` (metadata + `_run_plan`):

```python
# after: primary_task = plan[0] if plan else None
if primary_task is not None:
    _choice = _maybe_probe_route(profile, page_context, primary_task.engine, page_number)
    if _choice is not None:
        primary_task = EngineTask(engine=_choice.route, priority=1,
                                  reason=f"probe:{_choice.reason}")
        plan = [primary_task]                      # _run_plan runs plan[0]
        if _choice.ocr_lang:
            page_context["ocr_lang"] = _choice.ocr_lang
```

`plan` is reassigned (not mutated in place) so both `metadata.set_routing(...)` and `_run_plan(plan,…)`
below observe the override. `page_context` already flows into `_run_engine_task` (§ "threading" below),
so stashing `ocr_lang` there is the integration point for the Paddle payload.

### The feature flag (inert by default)

New boolean env var **`IDP_ENABLE_PROBE`**, default **false/OFF**.

**Where the flag is read (concrete, not "ProbeConfig/settings" hand‑waving).** `ProbeConfig` today has
no probe‑enable field and `settings.py` parses its own env vars in `_prepare_settings`. To keep the
flag a single source of truth and avoid import cost when off, read it **directly inside
`_maybe_probe_route` at call time** with a module‑level helper in `pipeline.py`:

```python
import os
def _probe_enabled() -> bool:
    return (os.getenv("IDP_ENABLE_PROBE") or "").strip().lower() in ("1", "true", "yes", "on")
```

`_maybe_probe_route(profile, page_context, primary_engine, page_number)` calls `_probe_enabled()`
first and returns `None` before importing anything from `src.ai.layer1_probe` when the flag is falsy.
When enabled, it then evaluates `_should_probe(profile, primary_engine)` (the §gating‑predicate helper)
and returns `None` if that is `False`; only then does it lazily `from src.ai.layer1_probe.inprocess
import probe_page_route` and call
`probe_page_route(page_context, page_number, profile, ProbeConfig.from_env(), render_dpi=page_context.get("render_dpi", cfg.page_image_dpi))`
where the render DPI comes from `page_context["render_dpi"]` (stashed by the web layer's
`_build_pages_for_pdf`, **not** imported from `pipeline_routes` — see the render‑DPI subsection below
for why the cross‑layer import is rejected, Finding 3). Rationale
for reading the env var here rather than adding
a `settings` field: `settings` is a process‑wide singleton validated at import; a raw `os.getenv`
inside the hook keeps the probe fully self‑contained, is trivially monkeypatchable per‑test
(`monkeypatch.setenv`), and guarantees **zero** `layer1_probe` import when off. `ProbeConfig.from_env()`
is still used *inside* the hook (once enabled) for the probe's own knobs; it gains **no**
enable field.

- **OFF** → `_maybe_probe_route` returns `None` immediately (the `_probe_enabled()` guard runs before
  any `layer1_probe` import), so `process_page`/`process_document` behave EXACTLY as today — no import
  cost, no behavior change. This is the prod default.
- **ON** → the probe runs for scanned/undetected pages only. Digital/Docling and skip pages are never
  probed.

**Gating predicate (Finding 3 — single positive rule).** The one intended rule is: *probe a page iff
Layer‑1 could not confidently read it as digital text* — i.e. the page is scanned, has no embedded text
layer, or has an unknown script. Engine membership is **not** part of the rule; the prior draft's
`primary_engine in (paddle…, vlm)` clause made the scanned/no‑text/unknown clauses dead for any
OCR‑routed page and conflated two different intents. The predicate is evaluated inside
`_maybe_probe_route` using only the values in scope at the converged hook — the Layer‑1 `profile`
object and the already‑chosen `primary_engine` string. (`single_route` is a local of the `else` branch
and is NOT in scope at the converged `primary_task = plan[0]` point, so the predicate must not
reference it.) Encode it **positively** as a named helper:

```python
def _should_probe(profile: PageProfile, primary_engine: str) -> bool:
    if primary_engine in ("docling", "skip"):
        return False                 # digital/blank passthrough — never probe
    return (profile.is_scanned
            or not profile.has_text
            or profile.primary_script == "unknown")
```

`_maybe_probe_route` returns `None` when `_should_probe(...)` is `False`.

**Direct attribute access, not `getattr`‑with‑default (Finding 2).** `PageProfile` always defines
`is_scanned`, `has_text` (`has_text = char_count > 0`), and `primary_script` — verified in
`inspect.py` — so the fields are guaranteed present. The earlier draft used
`getattr(profile, "has_text", False)` etc.; those defaults can never fire in normal operation but would
**silently** mask a future field rename by treating every page as "has text / not scanned" and quietly
disabling probing with no error. Because `_should_probe` is the **sole owner** of the
"only scanned/undetected pages are probed" invariant (§"Invariant ownership"), a silent wrong default
there would change production routing with no trace. **Decision: access `profile.is_scanned`,
`profile.has_text`, `profile.primary_script` directly** so a contract drift raises `AttributeError`
loudly instead of degrading silently. This matches how `pipeline.py` already treats the profile —
`process_page` reads `profile.is_scanned`, `profile.primary_script`, `profile.complexity_score`,
`profile.image_coverage` by direct attribute access (verified), so `_should_probe` stays consistent
with the surrounding code. `_maybe_probe_route` is only reached after `inspect_page` has built
`profile`, so the object is always a real `PageProfile`; no `getattr` default and no separate
`isinstance` guard is needed. The `profile: PageProfile` annotation in the helper is documentation
only — if the implementer adds it as a real annotation, import `PageProfile` from
`src.ai.schemas.page` (the same module `pipeline.py` already imports `EngineTask`,
`PageCapabilities`, `PageOutput`, `VLMAnalysis` from — verified); `PageProfile` is **not** currently
in that import line, so the annotation can also be left as a comment to avoid touching the import.
`profile.has_text` is defined as `has_text = char_count > 0` in `inspect.py` (verified), and the
`PageProfile` schema lives in `src/ai/schemas/page.py`.

**Explicit decision on the Layer‑1 `scanned`‑routed digital page.** `pipeline.py`'s `single_engine`
fallback sets `single_route = "scanned"` (→ `paddleocr_printed`) when
`profile.is_scanned or profile.complexity_score >= 4 or image_coverage > …` — verified in
`pipeline.py` lines ~362‑380 — so a page that *has text* but is complex/image‑heavy can reach the hook
with `primary_engine == "paddleocr_printed"` while `is_scanned == False` and `has_text == True`.
**Under the rule above, such a page is NOT probed** (none of `is_scanned`/`not has_text`/unknown‑script
holds), so it keeps its Layer‑1 `paddleocr_printed` route unchanged. This is the deliberate choice: the
probe only *refines* pages Layer‑1 was unsure about (true scans, no text layer, or unknown script); a
page Layer‑1 confidently classified as digital‑with‑text — even one it sent to OCR for complexity
reasons — is left to Layer‑1's decision. This keeps the probe strictly additive and matches the
"scanned/undetected only" invariant without the ambiguity of engine‑membership gating.

**The same analysis for the `capability_based` branch — the one the live task may actually run
(Finding 4).** The `single_engine` case above is only half the picture; the deploy notes say the live
ECS task may run `ROUTING_MODE=capability_based`, where `process_page` builds
`plan = build_engine_plan(capabilities_from_profile(profile))`. Because the hook is pinned to the
**converged** `primary_task = plan[0]`, it is mode‑agnostic, but the gate still depends on what
`plan[0].engine` is for that branch. Verified by reading `router.py::capabilities_from_profile` and
`build_engine_plan` (plus the priority constants `_PRI_DIGITAL=1 < _PRI_PRINTED=2 <
_PRI_HANDWRITTEN=3`, and `build_engine_plan` sorts by priority so `plan[0]` is the lowest‑priority
active engine):

| Page kind (capability_based) | capabilities set | `plan[0].engine` | profile signal | `_should_probe` |
|---|---|---|---|---|
| **(a) True scan** (`is_scanned=True`, near‑zero chars) | `has_printed_scan=True` (+`has_handwriting=True`); **no** `has_digital_text` | `paddleocr_printed` (pri 2; handwritten is pri 3) | `is_scanned=True` | **probe** (is_scanned) ✓ |
| **(b) Digital‑with‑text, complex / image‑heavy** (`has_text`, `char_count>digital_threshold`) | `has_digital_text=True` **and** possibly `has_printed_scan`/`has_figures` | `docling` (pri 1 wins) | `has_text=True`, `is_scanned=False`, script known | **not probed** — `docling` is excluded up front ✓ |
| **(c) Unknown / Indic script** (`primary_script` in `_INDIC_SCRIPTS`, or unknown) | early‑return `has_indic_script=True` + `has_printed_scan=True` (digital‑text branch never reached) | `paddleocr_printed` (indic placeholder, pri 2) | `primary_script != known‑latin` (unknown/indic) | **probe** (`primary_script=="unknown"` for unknown; Indic pages have a known non‑latin script so they are probed only if `is_scanned`/`not has_text` — see note) ✓ |

Key difference from `single_engine`: in `capability_based`, a digital‑with‑text page **always** gets
`docling` as `plan[0]` (priority 1 beats the `paddleocr_printed`/`figures` tasks), so it is excluded by
the `docling` guard, never by the `is_scanned`/`has_text` clauses — the gate reaches the same
"not probed" outcome as `single_engine`, by a different path. A **true scan** gets
`plan[0]=paddleocr_printed` and is probed via `is_scanned`. An **unknown‑script** page is probed via
`primary_script=="unknown"`.

**One honest edge the gate does NOT cover, stated explicitly:** an **Indic‑script** page
(`primary_script` ∈ {devanagari, telugu, …}) that is *digital‑with‑text and not scanned* takes the
`capabilities_from_profile` early return (`has_indic_script`→`paddleocr_printed`) in
`capability_based`, so `plan[0]=paddleocr_printed` — but under `_should_probe` it is **not** probed,
because `is_scanned=False`, `has_text=True`, and `primary_script` is a *known* Indic name (not
`"unknown"`). That means a digital Indic page keeps Layer‑1's `paddleocr_printed` route rather than
being escalated to the VLM by the probe. **This is the deliberate, consistent choice:** the probe
refines only pages Layer‑1 was *unsure* about (scans, no text layer, or script it could not identify);
a page Layer‑1 confidently read as digital‑with‑known‑script — Latin or Indic — is left to Layer‑1.
Widening the gate to re‑probe confidently‑classified digital Indic pages would contradict the
"scanned/undetected only" invariant and is explicitly **out of scope** for this build. If a future
effort wants the probe to override Layer‑1's Indic‑placeholder routing, it must revisit this predicate
deliberately; it is not a silent gap. (The `single_engine` branch does not hit this edge because its
Indic handling flows through `route_from_profile`, but the invariant — confident digital page is not
re‑probed — is identical in both modes.)

### What the probe call looks like in‑process

Rather than call `probe_document` (which is S3/PDF‑oriented and renders a whole doc), the in‑process
hook uses the **observer + CV + routing** directly on the one page's in‑memory image. The page's
`context["image_array"]` is a **true RGB** `np.ndarray` — verified in
`pipeline_routes.py::_render_page` (`arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))`). The
thin wrapper reads it from `page_context.get("image_array")`:

1. `rgb = page_context.get("image_array")`; if it is `None` (no in‑memory image for this call) →
   return `None` (passthrough, pipeline unchanged). Derive the analysis image with the **existing**
   helpers, which take RGB: `gray = imaging.analysis_gray(rgb, render_dpi, cfg.analysis_dpi)` and
   `mask = imaging.ink_mask(gray)`. (`analysis_gray` does `cv2.cvtColor(rgb, COLOR_RGB2GRAY)` —
   confirmed — so passing the real RGB is correct and no color channels are wasted.)

   **Render‑DPI source (Finding 2).** `analysis_gray(rgb, render_dpi, analysis_dpi)` computes the
   downscale factor as `analysis_dpi / render_dpi` and **trusts the second argument to be the true
   render DPI of `rgb`** (verified in `src/ai/layer1_probe/imaging.py`:
   `if analysis_dpi < render_dpi: f = analysis_dpi/render_dpi; gray = cv2.resize(gray, None, fx=f, fy=f, …)`).
   The live page image is rendered at **`RENDER_DPI = 150`** — a module constant in
   `schema_chatbot_v2/app/api/pipeline_routes.py` (`RENDER_DPI = 150; MAT = pymupdf.Matrix(RENDER_DPI/72, RENDER_DPI/72)`,
   verified) — **not** at `cfg.page_image_dpi` (which defaults to 144 in `types.py` and is the
   Step‑Functions PyMuPDF render DPI, irrelevant to the already‑rendered in‑process image). Passing
   144 for a 150‑dpi image would make the analysis image ~4% too large and nudge the
   `blank_ink_ratio`/`table_min_*` CV thresholds that run on `mask`.

   **Decision: carry the render DPI in `page_context`, not via a cross‑layer import (Finding 3).** The
   earlier draft had `pipeline.py::_maybe_probe_route` `import RENDER_DPI from
   schema_chatbot_v2.app.api.pipeline_routes`. That is a **layering/import‑cycle hazard** and is
   rejected: `pipeline_routes.py` (web layer) already does
   `from src.ai.layer1_routing.pipeline import process_document` (verified), so having the core
   `src/` module import back from the `schema_chatbot_v2` web package creates a circular dependency
   and breaks every non‑web caller of `process_document` that does not have `schema_chatbot_v2` on
   `sys.path` — the probe Lambda image, direct unit tests of `pipeline.py`, and any non‑web
   entrypoint. The constant must not cross from the web layer down into core.

   **Chosen fix — option (b): the web layer stashes its own `RENDER_DPI` in the page context it
   already builds.** `_build_pages_for_pdf` in `pipeline_routes.py` (verified: it constructs each
   page's `"context"` dict with `pdf_path`, `image_array`, `image_bytes`) adds one key:
   `"render_dpi": RENDER_DPI` (the existing module constant, still the single source of truth, read in
   the same file where it is defined — no second copy, no import). `pipeline.py::_maybe_probe_route`
   then reads it from the context it already has in scope and forwards it:

   ```python
   render_dpi = page_context.get("render_dpi", cfg.page_image_dpi)
   ... probe_page_route(page_context, page_number, profile, cfg, render_dpi=render_dpi)
   ```

   This keeps `pipeline.py` free of any `schema_chatbot_v2` import (no cycle, safe for the probe
   Lambda image and direct tests), preserves the single source of truth (the web layer owns
   `RENDER_DPI` and injects it), and defaults cleanly to `cfg.page_image_dpi` for any caller whose
   context dict lacks the key (e.g. a non‑web entrypoint or a test that builds a bare context) —
   preserving prior behavior there. `probe_page_route` forwards it as
   `imaging.analysis_gray(rgb, render_dpi, cfg.analysis_dpi)`. (`probe_page_route`'s signature is
   `probe_page_route(page_context, page_number, profile, cfg, *, render_dpi: Optional[int] = None)`,
   with `render_dpi = render_dpi or cfg.page_image_dpi` inside, so a `None` passed from a context
   without the key still falls back to the config DPI.)

   **Added to the modified‑files set:** `schema_chatbot_v2/app/api/pipeline_routes.py` gains the
   one‑line `"render_dpi": RENDER_DPI` entry in `_build_pages_for_pdf`'s context dict (§10).
2. `blank = cv_features.ink_ratio(mask) < cfg.blank_ink_ratio`; if blank → `RouteChoice(ROUTE_SKIP,
   reason="blank_page")`.
3. `has_table, _ = cv_features.detect_table(mask, cfg.table_min_h_lines, cfg.table_min_v_lines,
   cfg.table_min_intersections)`.
4. `obs = observer.observe(page_number, gray, mask, rgb=rgb)` — passing the **real RGB** through the
   new keyword so lipika classifies the color page, not a gray→RGB reconstruction (the Step‑Functions
   path, which lacks RGB, omits the kwarg per §3). If `obs is None`, `obs.script_confidence <
   cfg.onnx_min_conf`, or `obs.mixed_scripts`, and the VLM is configured, fall back to
   `BedrockScriptProbe.observe(page_number, <jpeg>)` — note the VLM observer needs **JPEG bytes**, so
   reuse `page_context["image_bytes"]` (PNG) re‑encoded or, simpler, skip the in‑process VLM fallback
   and let `obs=None`→`script="unknown"` flow to the VLM route via `decide_route` step 2 below.
   **Decision: skip the in‑process VLM observer** — when ONNX yields nothing confident, return a
   `RouteChoice(ROUTE_VLM, ...)` directly (via `decide_route` with `script="unknown"`), which already
   sends the page to `vlm_transcribe`. This avoids a second Bedrock call inside the live request path
   and keeps the hook to one ONNX inference. The document‑level VLM sampling stays a Step‑Functions‑only
   concern.
5. `choice = decide_route(script=(obs.script if obs else "unknown"),
   script_conf=(obs.script_confidence if obs else 0.0),
   mixed=(obs.mixed_scripts if obs else False), is_blank=blank, has_table=has_table, cfg=cfg)`.
   (Note the §5 `decide_route` signature drops `hw_prob` — the call passes no handwriting argument.)
6. Return `choice` (a `RouteChoice`). The caller (`_maybe_probe_route` in `pipeline.py`) assigns
   `choice.route` **directly** to `plan[0].engine` — `choice.route` is already an engine name
   (`paddleocr_printed`/`vlm_transcribe`/`skip`), so it must **not** be routed through
   `_ROUTE_TO_ENGINE` — and stashes `choice.ocr_lang` into `page_context["ocr_lang"]` when truthy.

This keeps the live hook lightweight (no S3, no second render, at most one ONNX inference, zero Bedrock
calls) and reuses the exact routing table and observer that the tests cover. A thin wrapper function
**`probe_page_route(page_context, page_number, profile, cfg, *, render_dpi=None) -> Optional[RouteChoice]`**
in a new `src/ai/layer1_probe/inprocess.py` is the single integration surface the pipeline imports, so
`pipeline.py`'s import of probe code is one symbol, imported lazily **inside** `_maybe_probe_route`
only after the `_probe_enabled()` guard passes. **Decision: new `inprocess.py`** rather than exporting
from `probe.py`, because `probe.py` imports `fitz` (PyMuPDF) at module top and the in‑process hook must
not drag that in on the hot path; `inprocess.py` imports only `imaging`, `cv_features`, `routing`, and
the observer.

### Threading `ocr_lang` into the Paddle Lambda payload

`src/ai/layer2_conversion/scanned.py::_invoke_paddle_lambda` builds `payload = {"image_base64": …,
"page_number": …}` (verified — exactly those two keys). Its current signature is
`_invoke_paddle_lambda(image_array, page_number, mode, image_bytes=None)` (verified). Change:

- `_invoke_paddle_lambda` gains a trailing keyword param `ocr_lang: Optional[str] = None`.
- `convert_scanned_page` / `convert_handwritten_via_paddle` gain an optional `ocr_lang: Optional[str]
  = None` parameter, threaded through `_ocr_convert` → `_invoke_paddle_lambda`. `_ocr_convert` itself
  gains the same optional `ocr_lang` param.
- **`_ocr_convert` forwards `ocr_lang` to EVERY `_invoke_paddle_lambda(...)` call — all THREE call
  sites (Finding 6).** Verified in `src/ai/layer2_conversion/scanned.py`: `_ocr_convert` invokes
  `_invoke_paddle_lambda` three times — (1) the prefer‑lambda primary path (~line 244), (2) the
  `except (ImportError, ModuleNotFoundError)` fallback retry (~line 286), and (3) the broad
  `except Exception` fallback retry (~line 305). The new `ocr_lang` must be passed on **all three**
  (`_invoke_paddle_lambda(image_array, page_number, mode, image_bytes, ocr_lang=ocr_lang)`), otherwise
  the two exception‑fallback paths would silently drop the lang when the primary lambda call errored.
- `pipeline.py::_run_engine_task` passes `ocr_lang=page_context.get("ocr_lang")` when calling
  `convert_scanned_page` — **only on the `paddleocr_printed` branch** (the other engines do not take
  `ocr_lang`).
- In `_invoke_paddle_lambda`, add `ocr_lang` to the payload **only when it is truthy**:
  ```python
  payload = {"image_base64": b64_str, "page_number": page_number}
  if ocr_lang:
      payload["ocr_lang"] = ocr_lang
  ```
  So when the probe is off or yields no `ocr_lang`, the payload is **byte‑identical** to today
  (`{"image_base64", "page_number"}`), satisfying the "byte‑identical when absent" requirement. The
  deployed printed handler already reads `ocr_lang` via `_resolve_probe_key` (contract findings), so
  this is forward‑compatible and inert until the flag is on.

### Error handling (STEP 4)

- Probe hook raises for any reason (observer error, CV error, lazy‑import error) → caught inside
  `_maybe_probe_route` with a broad `except Exception`, logged at WARNING
  (`pipeline.probe_hook_failed`, page_number), and the function returns `None`. Because the caller only
  overrides `plan`/`primary_task` when a non‑`None` `RouteChoice` comes back, a `None` leaves
  `plan[0]` exactly as Layer‑1 built it — the page keeps its **Layer‑1 route** (fatal‑to‑probe,
  recoverable‑to‑pipeline — the in‑process analogue of the Step‑Functions
  `ProbeFallbackToLayer1Routes` catch). The pipeline never fails because of the probe.
- Flag parsing: any value not in `{"1","true","yes","on"}` (case‑insensitive) → treated as OFF (safe
  default). No log needed; OFF is the normal state.
- `image_array` absent from `page_context` → hook returns `None` (passthrough). This can happen for
  callers that build `page_context` without a rendered image; the probe simply does not engage.
- `ocr_lang` present but the final engine is Docling/VLM → ignored (only `_run_engine_task`'s
  `paddleocr_printed` branch forwards it; see the threading section).

### Invariant ownership

- **"Probe is inert when flag is OFF"** is owned by `_maybe_probe_route`'s early return (pipeline
  layer), because that is the only place the flag gates behavior and the only place probe code is
  imported.
- **"Only scanned/undetected pages are probed"** is owned by the `_should_probe(profile,
  primary_engine)` predicate in `_maybe_probe_route` (pipeline layer, not the probe module), because
  the pipeline already holds the Layer‑1 `profile` and the already‑chosen `primary_engine` at the hook
  point. The rule is positive and single (`is_scanned or not has_text or primary_script=="unknown"`,
  with `docling`/`skip` excluded up front); a Layer‑1 `scanned`‑routed *digital‑with‑text* page is
  deliberately NOT probed. (It does not reference `single_route`, which is out of scope at the
  converged hook.)
- **"Engine payload stays byte‑identical without ocr_lang"** is owned by `_invoke_paddle_lambda`
  (layer‑2), because that is where the payload dict is constructed.

---

## 7. STEP 5 — Packaging (edit only; NO AWS build/push)

### `requirements-probe.txt`

Already has `onnxruntime`, `numpy`, `opencv-python-headless`, `pymupdf`, `boto3`, `pydantic`. **Do NOT
add Pillow (Finding 8):** the probe runtime path (`imaging.py`, `cv_features.py`,
`script_classifier.py`, `inprocess.py`) imports only `cv2` and `numpy` — verified, no module on the
probe‑Lambda runtime path imports `PIL`. Adding Pillow would be a dead dependency. (Pillow *is* added
to the layer‑3 app image below, but only because that image's own `pipeline_routes.py` already uses it;
it is not introduced by the probe.) So `requirements-probe.txt` needs **no new package** — only the
`onnxruntime` comment is reworded from "only used if … is set" to "required: lipika script model is
baked in." Ensure **no torch, no transformers, no fontrecog** appear.

### `Dockerfile.probe`

Verified current contents: the `ENV` block sets `IDP_PROBE_PRINTED_SCRIPTS=latin` and
`IDP_PROBE_PAGE_IMAGE_DPI=144`; there is a `RUN mkdir -p /opt/probe_models` followed by a commented
`# COPY models/probe/ /opt/probe_models/`; the last build step is `COPY src/ /var/task/src/`; and
`PYTHONPATH=/var/task`.

- Remove the `IDP_PROBE_PRINTED_SCRIPTS=latin` line from the `ENV` block (field retired, §5). Leave
  `IDP_PROBE_PAGE_IMAGE_DPI=144` and `PYTHONPATH=/var/task` as is.
- **Model baking — relies on the existing `COPY src/ /var/task/src/`.** Because the observer's
  zero‑config default resolves the model **relative to its own module file**
  (`src/ai/layer1_probe/models/lipika_script.onnx`), and `COPY src/ /var/task/src/` already copies the
  whole `src/` tree into `/var/task/src/` (with `PYTHONPATH=/var/task`), the baked model ships
  automatically **provided `src/ai/layer1_probe/models/` is not excluded by `.dockerignore`**.
  **Verified against `.dockerignore`:** the repo‑root `.dockerignore` excludes `*.bin` and
  `*.safetensors` but **NOT `*.onnx` or `*.json`**, so `src/ai/layer1_probe/models/lipika_script.onnx`,
  `.int8.onnx`, and `.labels.json` are copied by `COPY src/` as‑is. (The excluded `*.bin`/`*.safetensors`
  are only lipika's *source* weights, which never enter the image — they live in the HF cache during the
  offline §2 export.) **Decision: do not add a second `COPY` for the models dir** — a redundant
  `COPY src/ai/layer1_probe/models/ ...` would be dead weight. Replace the stale
  `RUN mkdir -p /opt/probe_models` + commented `# COPY models/probe/ /opt/probe_models/` lines with a
  short comment stating the lipika ONNX ships via the `src/` COPY at the module‑relative path, and that
  `.dockerignore` lets `*.onnx` through. This keeps the module‑relative path the single source of truth
  and retires the `/opt/probe_models` divergence.
- Add `IDP_PROBE_ONNX_MODEL` / `IDP_PROBE_ONNX_LABELS` as *optional* ENV overrides in a comment, not
  set by default (the baked module‑relative path is the default).

### Layer‑3 app image (`schema_chatbot_v2/requirements.txt` + `schema_chatbot_v2/Dockerfile`)

Because the live integration is **in‑process in `idp-layer3-app`**, that image must be able to run the
observer. Verified from `schema_chatbot_v2/Dockerfile`: base `python:3.11-slim`, apt‑installs
`libgl1 libglib2.0-0 libgomp1` (so OpenCV's shared libs are present), `PYTHONPATH=/app/schema_chatbot_v2:/app`,
and `COPY . /app` copies the whole repo (including `src/ai/layer1_probe/models/`) to `/app`.

- `schema_chatbot_v2/requirements.txt`: already has `pillow>=11.0.0` and `numpy>=1.24.0` (confirmed).
  **Add `onnxruntime>=1.18.0`** (currently absent — confirmed by grep). `opencv-python-headless` is not
  listed there; the probe hook uses `imaging.analysis_gray`/`ink_mask` and `cv_features`, both of which
  `import cv2` (confirmed). **Add `opencv-python-headless>=4.9.0`.** The image already apt‑installs
  `libgl1`/`libglib2.0-0`/`libgomp1`, so the headless wheel's runtime libs are satisfied. **No torch,
  no transformers, no fontrecog.** (Note: `paddleocr`/`paddlepaddle` already pull a `cv2` transitively,
  but pinning `opencv-python-headless` explicitly makes the probe's dependency first‑class and avoids
  relying on a transitive one.)
- **Model path resolution in this image:** the observer resolves `lipika_script.onnx` relative to its
  module file → `/app/src/ai/layer1_probe/models/lipika_script.onnx`, importable because `/app` is on
  `PYTHONPATH`. `COPY . /app` ships it (and `.dockerignore` passes `*.onnx`/`*.json`, as verified
  above), so **no extra `COPY` is needed**. Add only a short comment in the Dockerfile noting the probe
  model path and that `IDP_ENABLE_PROBE` is OFF by default (so the added onnxruntime/opencv deps sit
  idle until the flag is turned on).

### Local smoke test (optional, no AWS)

A local `docker build -f Dockerfile.probe .` and `docker build -f schema_chatbot_v2/Dockerfile .` are
allowed **only if docker is available locally**. If docker is absent, **skip and note it** — do not
build or push to AWS. The design does not require the image to be built to be correct.

---

## 8. STEP 6 — Tests (fully mocked; no real model, no AWS, no torch)

New/updated tests under `tests/` (run with `.venv\Scripts\python.exe -m pytest`). All use a **mocked
onnxruntime session** — no real model load, no torch import.

**Single test mechanism for every observer‑dependent test (Finding 4).** Every test that needs the
observer to return a specific script — §8a (mapping/singleton), §8d (flag‑on Latin→printed and
non‑Latin→VLM) — constructs `LipikaOnnxScriptObserver(..., session=<fake>)` (or injects the fake into
the in‑process path via the §11 reset hook + `session=` ctor), where `<fake>.run(...)` returns canned
logits peaking on the desired script index. **No observer test loads a real baked model.** This is what
makes the §9 STOP‑AND‑ASK fallback and §8 mutually consistent: under §9 the real runtime observer
returns `None` (no model file), but §8d's "observer returns `latin`" is produced by the injected fake
session, not by a real model, so the full routing/integration suite is model‑independent and passes
whether or not the lipika export succeeded. The fake session also returns deterministic logits, so the
tests assert exact `script`/`ocr_lang`/route without tolerance.

**(a) Observer maps ONNX outputs → correct `SampleObs.script`, singleton once.**
Patch the observer's `InferenceSession` (or inject a fake session whose `.run` returns canned logits):
- logits peaking on the `latin` index → `obs.script == "latin"`;
- peaking on `telugu` → `"telugu"`; on `devanagari` → `"devanagari"`.
- Call `observe` multiple times and assert the session constructor ran **exactly once** (module
  singleton reused across warm invocations) — e.g. patch with a counter or assert `mock.call_count ==
  1`.
- `source == "onnx"`, and `SampleObs` has **no** `handwriting_prob` attribute.

**(b) `decide_route` table (script‑only).** This is the rewrite of the existing `test_routing_table`
plus the three other routing tests Finding 1 flagged (`test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm`,
`test_table_sets_structure_hint_only_for_printed`, `test_blank_routes_to_skip`). After the rewrite
**no routing test passes `hw_prob=` or builds `_cfg(printed_scripts=…)`** — both are gone from the file.
Parametrized `test_routing_table` rows (no `hw_prob` kwarg):
- `latin`, conf 0.9 → `paddleocr_printed`, `ocr_lang == "en"`;
- `telugu`, conf 0.9 → `vlm_transcribe`, `ocr_lang is None`;
- `devanagari`, conf 0.9 → `vlm_transcribe` (this is where the old
  `test_devanagari_print_goes_to_paddle_…` assertion is replaced — Devanagari now escalates);
- `latin`, conf 0.3 → `vlm_transcribe` (low‑conf);
- `unknown`, conf 0.0 → `vlm_transcribe`;
- `is_blank=True` → `skip` (folds in `test_blank_routes_to_skip` — drop its `hw_prob=0` arg);
- **assert no input ever yields `paddleocr_handwritten`** (iterate the whole param set and assert
  `route != "paddleocr_handwritten"`; the old `latin`+`hw_prob=.8 → paddleocr_handwritten` row is deleted).

Separate (non‑parametrized) rewrites in the same file:
- `test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm` → **rewrite**: drop
  `_cfg(printed_scripts=…)` and `hw_prob=`; assert `decide_route(script="devanagari", script_conf=.9,
  mixed=False, …)` → `vlm_transcribe`, and keep the mixed‑Latin half (`mixed=True` → `vlm_transcribe`).
- `test_table_sets_structure_hint_only_for_printed` → **rewrite**: drop both `hw_prob=` args and the
  `hw_prob=.9 → use_structure False` assertion (meaningless without handwriting); assert
  `script="latin", has_table=True → use_structure is True` and `has_table=False → use_structure is False`.
- `test_blank_routes_to_skip` → **edit**: drop `hw_prob=0`; keep `is_blank=True → skip`.

The old handwriting params/signature and every `printed_scripts=`/`hw_prob=` reference are removed from
`test_probe.py`.

**(c) Handwriting fields gone.**
`assert "handwriting_prob" not in SampleObs.model_fields` and `not in PageProbe.model_fields`;
`"hw_source" not in PageProbe.model_fields`. Also assert `decide_route` has no `hw_prob` parameter
(`inspect.signature`).

**(d) Pipeline integration behind the flag.**
- `IDP_ENABLE_PROBE` unset/off → `process_document` on a scanned page produces the same route as
  today (probe hook returns None; assert the engine chosen equals the Layer‑1 route). Use a mocked
  `llm_client` and monkeypatch the layer‑2 converters so no real OCR/AWS runs.
- Flag on, observer + engines mocked:
  - a **scanned non‑Latin** page (observer returns `telugu`) → routes to `vlm_transcribe`;
  - a **scanned Latin** page (observer returns `latin`) → routes to `paddleocr_printed` **and**
    `ocr_lang="en"` is threaded into a **mocked `_invoke_paddle_lambda`** payload (assert the mock
    received `payload["ocr_lang"] == "en"`).
- A digital/Latin page is never probed with the flag on (observer not called).

**(e) `_invoke_paddle_lambda` payload byte‑identical when no `ocr_lang`.**
Call the converter path without `ocr_lang` (mock the boto3 client) and assert the serialized payload
equals `{"image_base64": <...>, "page_number": n}` exactly — no `ocr_lang` key.

**Running tests:** run the existing fast suite (`tests/test_probe.py` and the new tests) via the repo
venv. **Do NOT run** the slow real‑PaddleOCR tests
`tests/test_layer2_engines_local.py::*_engine_local_path` (~8 min each) — select around them
(`-k "not engine_local_path"` or `--deselect`). Update `test_probe.py`'s existing routing/e2e tests
that reference `hw_prob`, `handwriting_prob`, `paddleocr_handwritten`, and the old ONNX line‑crop
contract — specifically (names verified in `tests/test_probe.py`): `test_onnx_classifier_contract`
(builds a `[N,1,48,320]`→`[N,K]` line‑crop ONNX — rewrite for the whole‑image lipika contract or
replace with the §8a fake‑session test), `test_routing_table` (its parametrized rows pass `hw_prob=…`
and expect `paddleocr_handwritten` — drop the `hw_prob` kwarg and the handwritten rows, matching §8b),
and `test_e2e_devanagari_handwritten_goes_to_vlm_and_latin_hw_to_handwritten_engine` (asserts the now‑
removed handwriting routing — rewrite to the script‑only expectation: Devanagari→VLM, Latin→printed).
Also remove the two `handwriting_heuristic` tests per §5.

---

## 9. STOP‑AND‑ASK contingency (must not block non‑model work)

If, during STEP 1, **lipika cannot be pip‑installed**, `fontrecog` will not import, or exporting just
the script head proves infeasible (model won't trace, head can't be isolated), the implementer must
**NOT** fake a model or substitute another model. Instead:

1. Send a `send_message` **warning** describing precisely what failed (install error, import error,
   or trace/export error) and that the model artifact cannot be produced.
2. **Still deliver STEPS 3, 4, and 6** with a **mocked script detector**: the observer module keeps
   its interface and singleton structure, but `_get_session()` returns `None` when no model file is
   present (already the designed behavior), so the *runtime* `observe` returns `None`. In the
   in‑process hook (§6) that yields `obs=None` → `decide_route(script="unknown", …)` → `ROUTE_VLM`
   (`vlm_transcribe`), matching the §6 decision to **not** run an in‑process VLM *observer* — the page
   simply routes to the VLM *engine*, which is the existing escalation target. The routing rewrite, the
   handwriting removal, the pipeline flag + `ocr_lang` threading, and **all tests** are fully
   implementable without the real ONNX file **because every observer‑dependent test injects a fake
   `InferenceSession` via the preserved `session=` ctor arg (§3, §8), never the runtime
   `_get_session()` path**. Concretely, §8d's "observer returns `latin` → `paddleocr_printed` +
   `ocr_lang=en`" is driven by a fake session returning latin‑peaking logits, so it passes under this
   no‑model fallback exactly as it does with a real model. The only thing the missing model changes is
   *production* behavior (pages route to VLM), never the test outcomes.
3. The baked model ships only via `COPY src/` (§7), so with an empty `src/ai/layer1_probe/models/` the
   image still builds and the observer finds no model → `observe` returns `None` → pages route to the
   VLM engine. No Dockerfile COPY is conditional on the model existing, so nothing breaks. Document the
   no‑model state in the metadata JSON (`"shipped_default": "none"`) or by the directory's absence.

This keeps the script‑only routing, the handwriting teardown, the live wiring, and the test suite
unblocked even if the lipika export is impossible, and it never ships a fabricated model.

---

## 10. Files touched (summary)

New:
- `scripts/export_lipika_script_onnx.py` (offline, dev‑only)
- `src/ai/layer1_probe/models/lipika_script.onnx` + `.int8.onnx` + `.labels.json` (build artifacts;
  may be absent under the §9 fallback)
- `src/ai/layer1_probe/inprocess.py` — thin
  `probe_page_route(page_context, page_number, profile, cfg, *, render_dpi=None) -> Optional[RouteChoice]`, the single
  symbol `pipeline.py` lazily imports inside `_maybe_probe_route`. **Not** exported from `probe.py`
  (which imports `fitz` at module top and must stay off the hot path — §6).
- new tests in `tests/` (observer mapping/singleton, script‑only routing, flag integration, payload)

Modified:
- `src/ai/layer1_probe/script_classifier.py` (lipika observer, singleton, alias)
- `src/ai/layer1_probe/routing.py` (script‑only `decide_route`)
- `src/ai/layer1_probe/types.py` (drop handwriting fields + `hw_threshold`/`printed_scripts`)
- `src/ai/layer1_probe/probe.py` (drop handwriting; Phase A/C)
- `src/ai/layer1_probe/aggregate.py` (script‑only disagreement/prior)
- `src/ai/layer1_probe/cv_features.py` (remove `handwriting_heuristic`/`line_periodicity`)
- `src/ai/layer1_probe/vlm_probe.py` (drop handwriting)
- `src/api/lambda_probe_handler.py` (drop `handwriting_prob` from `_SLIM_FIELDS`; **and** make
  `_load_models` fall back to `script_classifier.default_model_path()`/`default_labels_path()` when
  `cfg.onnx_model_path`/`onnx_labels_path` are `None`, so the baked model loads without env vars —
  Finding 1)
- `src/ai/layer1_routing/pipeline.py` (`_maybe_probe_route` hook + `ocr_lang` into `_run_engine_task`)
- `src/ai/layer2_conversion/scanned.py` (`ocr_lang` param + conditional payload key)
- `requirements-probe.txt` (comment reword only; no new package — Pillow NOT added per Finding 8;
  onnxruntime/opencv already present)
- `Dockerfile.probe` (bake model, drop retired ENV)
- `schema_chatbot_v2/requirements.txt` (add onnxruntime + opencv-python-headless)
- `schema_chatbot_v2/Dockerfile` (comment only; `COPY . /app` already ships models)
- `schema_chatbot_v2/app/api/pipeline_routes.py` (one‑line: add `"render_dpi": RENDER_DPI` to the
  `context` dict built in `_build_pages_for_pdf`, so `pipeline.py` reads the true render DPI from the
  context instead of importing the web‑layer constant — Finding 3)
- `tests/test_probe.py` — complete edit set (per §5 grep list + §8b):
  rewrite `test_routing_table` (drop `hw_prob=`, drop handwritten row),
  `test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm` (drop `printed_scripts=`/`hw_prob=`,
  Devanagari→VLM), `test_table_sets_structure_hint_only_for_printed` (drop `hw_prob=`,
  Latin‑only `use_structure`), `test_blank_routes_to_skip` (drop `hw_prob=`),
  `test_e2e_devanagari_handwritten_goes_to_vlm_and_latin_hw_to_handwritten_engine` (script‑only),
  `test_onnx_classifier_contract` (whole‑image lipika contract / fake‑session);
  **remove** `test_handwriting_heuristic_orders_wobbly_above_typed` and
  `test_heuristic_returns_none_when_too_little_text`

Deleted:
- `scripts/calibrate_probe_heuristic.py` (offline calibrator for the removed heuristic — non‑runtime,
  not imported anywhere; deleted rather than left dangling)

Untouched (documented to explain grep hits):
- `.scratch_probe/run_probe_signals.py`, `.scratch_probe/run_probe_signals2.py` (scratch, not
  collected/imported/shipped)

No AWS resource creation/configuration is in scope, so the workspace mandatory‑tag rule does not apply
to any action in this design. If a later, separate step provisions AWS resources, the three mandatory
tags (`createdby=vinay.k@shellkode`, `purpose=internal`, `customer=idp-internal-35385`) must be
attached then.

---

## 11. Testability summary

- **Unit‑testable (no AWS, no torch, no real model):** `decide_route` table; observer logit→script
  mapping and singleton behavior via a fake `InferenceSession`; schema‑field removal assertions;
  `_invoke_paddle_lambda` payload shape via a mocked boto3 client; `_maybe_probe_route` flag gating.
- **Integration‑testable in‑process (mocked engines):** `process_document` with the flag on/off,
  asserting route selection and `ocr_lang` threading.
- **Not unit‑tested here (offline/manual):** the real lipika→ONNX export and the 4‑image accuracy +
  INT8 report — these are produced and reported by `scripts/export_lipika_script_onnx.py` at build
  time, not in CI.

A design note on testability risk: the observer's module‑scope singleton must be resettable in tests
(a `_reset_for_tests()` or module‑reload fixture) so the "constructed once" assertion and the
per‑test fake session do not leak across tests. Build the singleton behind a `_get_session()` indirection
precisely so it is patchable.

---

## 12. Responses to design reviews

### 12.1 First review round (1 HIGH, 4 MEDIUM, 3 NIT)

Verdict under review: CHANGES_REQUESTED (1 HIGH, 4 MEDIUM, 3 NIT). Every finding was verified against
the real source before acting; all are **addressed** (none backlogged or ignored).

- **Finding 1 (HIGH) — Step‑Functions handler never loads the baked model. ADDRESSED.** Confirmed the
  guard `if cfg.onnx_model_path and cfg.onnx_labels_path and os.path.exists(...)` short‑circuits on the
  `None` defaults (`src/api/lambda_probe_handler.py`, `types.py`), and `Dockerfile.probe` sets no such
  env var. Removed the false §3 claim and chose fix (a): `script_classifier.py` now exposes
  `default_model_path()`/`default_labels_path()` (module‑relative), and `_load_models` falls back to
  them when the env vars are unset, guarding on `os.path.exists`. This loads the baked model on both
  the Step‑Functions and in‑process paths from one source of truth, keeps `IDP_PROBE_ONNX_*` as
  optional overrides (consistent with §7), and stays VLM‑only under the §9 no‑model fallback. Added to
  §10's `lambda_probe_handler.py` entry.
- **Finding 2 (MEDIUM) — in‑process analysis DPI mismatch. ADDRESSED.** Confirmed `RENDER_DPI = 150`
  is a module constant in `pipeline_routes.py` and `analysis_gray` trusts its 2nd arg as the true
  render DPI (`imaging.py`). §6 step 1 now threads the real `RENDER_DPI` (imported from
  `pipeline_routes`, single source of truth) through `_maybe_probe_route` →
  `probe_page_route(..., render_dpi=RENDER_DPI)` → `analysis_gray(rgb, render_dpi, cfg.analysis_dpi)`,
  with a `cfg.page_image_dpi` default for non‑pipeline callers. The render‑DPI source is stated
  explicitly.
- **Finding 3 (MEDIUM) — over‑broad / double‑negative gating predicate. ADDRESSED.** Rewrote the
  predicate as a positive single‑rule helper `_should_probe(profile, primary_engine)`:
  `docling`/`skip` → never probe; else probe iff `is_scanned or not has_text or
  primary_script=="unknown"`. Dropped the engine‑membership clause. Explicitly decided that a Layer‑1
  `scanned`‑routed *digital‑with‑text* page (complexity/image‑heavy fallback, verified in
  `pipeline.py`) is **NOT** probed — the probe only refines pages Layer‑1 was unsure about. Updated the
  invariant‑ownership section to match.
- **Finding 4 (MEDIUM) — §9 fallback vs §8d test coupling. ADDRESSED.** §3 now explicitly preserves
  the `session=None` ctor injection on `LipikaOnnxScriptObserver`, and §8/§9 state that **every**
  observer‑dependent test (including §8d's Latin→printed case) drives the result through an injected
  fake `InferenceSession`, never the runtime `_get_session()`. This makes the routing/integration suite
  model‑independent so the §9 no‑model fallback leaves §8 green.
- **Finding 5 (MEDIUM) — atomic config‑field teardown. ADDRESSED.** §5 now gives the exact final
  `routing.py` import block (drops `ROUTE_HANDWRITTEN`, keeps `SCRIPT_TO_PADDLE_LANG`,
  `ROUTE_PRINTED/SKIP/VLM`, `RouteChoice`, `ProbeConfig`) and the full rewritten `decide_route` body
  that references no removed `ProbeConfig` field. The two `hw_prob=`‑passing `decide_route` call sites
  in `probe.py` (Phase C `nonblank` loop and the `timed_out` loop) are enumerated verbatim, confirming
  there is no third call. So the field removal + signature change land atomically with no dangling
  `cfg.printed_scripts`/`cfg.hw_threshold` reference.
- **Finding 6 (NIT) — §4 lang‑key column invites per‑script wiring. ADDRESSED.** Added a header stating
  the `SCRIPT_TO_PADDLE_LANG` column is reference‑only and routing gates strictly on `script == "latin"`;
  relabeled the column `(ref only)`.
- **Finding 7 (NIT) — unvalidated `0.25` mixed‑scripts margin. ADDRESSED.** Promoted it to a named
  module constant `_MIXED_MARGIN = 0.25` in `script_classifier.py`, marked provisional/re‑tunable, with
  a note it only pushes borderline pages to the VLM (safe direction).
- **Finding 8 (NIT) — unjustified Pillow add. ADDRESSED.** Confirmed `requirements-probe.txt` already
  has `onnxruntime`/`numpy`/`opencv-python-headless` and the probe runtime path imports only
  `cv2`/`numpy`. Dropped the Pillow add entirely; §7 now changes only the onnxruntime comment. Updated
  §10 accordingly.

Unverified external dependency (unchanged, routed through §9): the real lipika→ONNX export (loadability
via `fontrecog`, head isolation, INT8 parity, opset/runtime match) remains the one risk that cannot be
verified without network/export at build time, and it is handled by the §9 STOP‑AND‑ASK contingency — no
fabricated model is ever shipped.

### 12.2 Second review round (`probe-lipika-design-review.json` / `.md` — 1 HIGH, 3 MEDIUM, 2 NIT)

Verdict under review: CHANGES_REQUESTED. Every finding was re‑verified against the live source before
acting; all six are **addressed** (none backlogged or ignored). The two NITs are resolved too.

- **Finding 1 (HIGH) — test‑teardown inventory missed three routing tests. ADDRESSED.** Re‑grepped
  `tests/` for `hw_prob` and `printed_scripts` and confirmed the three unlisted tests
  (`test_devanagari_print_goes_to_paddle_when_enabled_and_mixed_goes_to_vlm` ~L180,
  `test_table_sets_structure_hint_only_for_printed` ~L188, `test_blank_routes_to_skip` ~L195) plus
  `test_routing_table` ~L167 all pass `hw_prob=`/`printed_scripts=`. §5's "Repo‑wide cleanup grep" now
  lists the **complete, verified eight‑test set** with an explicit disposition for each (Devanagari→VLM
  rewrite; table test → Latin‑only `use_structure` and drop the `hw_prob=.9` assertion; blank test →
  drop `hw_prob`; routing‑table → drop `hw_prob` kwarg and the handwritten row), §8b spells out the
  rewritten assertions, and §10's `tests/test_probe.py` entry enumerates every edit. A pre‑implementation
  re‑grep of `tests/` for `hw_prob`/`printed_scripts` is mandated so the suite the design promises to
  leave green actually does.
- **Finding 2 (MEDIUM) — `_should_probe` `getattr`‑with‑default masking. ADDRESSED.** The predicate now
  accesses `profile.is_scanned`, `profile.has_text`, `profile.primary_script` **directly** (no `getattr`
  default), matching how `process_page` already reads the profile, so a future field rename raises
  `AttributeError` loudly instead of silently disabling probing. Verified `PageProfile` guarantees all
  three fields (`inspect.py`: `has_text = char_count > 0`) and lives in `src/ai/schemas/page.py`; the
  `PageProfile` annotation is documented as optional (import from `src.ai.schemas.page` if added).
- **Finding 3 (MEDIUM) — cross‑layer `RENDER_DPI` import cycle. ADDRESSED.** Rejected the
  `pipeline.py` → `schema_chatbot_v2.app.api.pipeline_routes` import (verified the web layer already
  imports `process_document` from core, so the back‑import is a cycle that breaks the probe Lambda image
  and direct tests). Adopted review option (b): `_build_pages_for_pdf` stashes `"render_dpi": RENDER_DPI`
  in the `context` dict it already builds (verified that dict and the `RENDER_DPI = 150` constant are in
  the same file), and `_maybe_probe_route` reads `page_context.get("render_dpi", cfg.page_image_dpi)` —
  no `schema_chatbot_v2` import in core, single source of truth preserved, clean default for non‑web
  callers. §6 render‑DPI subsection, the feature‑flag call snippet, and §10 are updated; the stale
  "imported from `pipeline_routes`" wording in §12.1's Finding‑2 note is superseded by this.
- **Finding 4 (MEDIUM) — `capability_based` branch unanalyzed. ADDRESSED.** Read
  `router.capabilities_from_profile` / `build_engine_plan` and the priority constants
  (`_PRI_DIGITAL=1 < _PRI_PRINTED=2 < _PRI_HANDWRITTEN=3`, sorted so `plan[0]` is the lowest priority).
  §6 now gives the full scan / digital‑complex / unknown‑script → `plan[0].engine` → `_should_probe`
  table for `capability_based` alongside the existing `single_engine` one: a true scan →
  `paddleocr_printed` → probed (is_scanned); a digital‑with‑text complex page → `docling` (priority 1
  wins) → not probed; an unknown‑script page → `paddleocr_printed` → probed (`primary_script=="unknown"`).
  The analysis also states the one honest edge the gate deliberately does not cover — a *digital,
  non‑scanned* known‑Indic page keeps its Layer‑1 `paddleocr_printed` route and is not re‑probed —
  consistent with the "scanned/undetected only" invariant, declared explicitly out of scope rather than
  left silent. No predicate change was needed.
- **Finding 5 (NIT) — `docs/` excluded by `.dockerignore`. ADDRESSED.** Added a sentence at the top of
  §2's validation subsection: validation runs in the repo venv only; `docs/` is excluded by
  `.dockerignore` and is never needed in any image, and the §7 `docker build` smoke test must not try to
  reproduce it.
- **Finding 6 (NIT) — `ocr_lang` threading omitted the two except‑branch Lambda calls. ADDRESSED.**
  §6's threading section now states `_ocr_convert` forwards `ocr_lang` to **all three**
  `_invoke_paddle_lambda(...)` call sites (verified ~L244 primary, ~L286 `ImportError` fallback, ~L305
  broad‑`except` fallback), names the new `ocr_lang=None` param on `_invoke_paddle_lambda`, and reaffirms
  `_run_engine_task` passes `ocr_lang` only on the `paddleocr_printed` branch — so a lambda retry after a
  primary failure cannot silently drop the lang, while the no‑`ocr_lang` payload stays byte‑identical.

Unverified external dependency (unchanged from §12.1, routed through §9): the real lipika→ONNX export —
handled by the STOP‑AND‑ASK contingency; no fabricated model is ever shipped. The full six‑step scope
and the NO‑AWS constraint remain intact.
