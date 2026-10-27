# Layer 1.5 Probe — what it is, what's in the bundle, how to wire it

## Where it sits

```
S3 PDF ─► L1 Lambda (PyMuPDF profile, unchanged)
            └─► Step Functions
                 ├─ ProbeDocument  (NEW Lambda, once per document)
                 │    A. per scanned page: render ONCE @144dpi → blank / ruled-table / handwriting-heuristic
                 │       → JPEG to s3://<bucket>/pages/<doc>/<n>.jpg
                 │    B. per DOCUMENT: sample 3 pages (more only if they disagree)
                 │       → ONNX crop classifier (optional) → VLM fallback → doc script prior (cached by ETag)
                 │    C. per page: script/handwriting from nearest sample, table/blank from own CV
                 │       → routing table → route + ocr_lang + image key
                 └─ Map(page) ─► skip | docling | paddleocr_printed | paddleocr_handwritten | vlm_transcribe
```
If the probe Lambda fails, `ProbeFallbackToLayer1Routes` continues with the Layer 1 routes unchanged.

## Routing table (src/ai/layer1_probe/routing.py — edit here)

| Condition | Route |
|---|---|
| blank page | skip |
| script unknown/other, or confidence < 0.60, or mixed scripts | vlm_transcribe |
| Latin + handwriting_prob ≥ 0.30 | paddleocr_handwritten (lang en) |
| non-Latin + handwritten | vlm_transcribe |
| printed + script in `IDP_PROBE_PRINTED_SCRIPTS` | paddleocr_printed + `ocr_lang` |
| printed + script NOT in that set | vlm_transcribe |
| ruled table found + printed route | `use_structure: true` hint |

Default `IDP_PROBE_PRINTED_SCRIPTS=latin`, so every non-Latin page goes to the VLM until you
verify a PaddleOCR recognizer for that script and add it (e.g. `latin,devanagari`).

## Output contract (per page, slim form returned to Step Functions)
`page_number, route, ocr_lang, use_structure, status, script, handwriting_prob, has_table,
image_s3_bucket, image_s3_key`. Full report (confidences, sources, ink ratio…) is written to
`s3://<bucket>/probe-reports/<doc>.json` because Step Functions caps state at 256 KB.
`status`: ok | blank | not_needed | skipped_timeout | error.
`image_s3_key` may be null (blank page, upload failed, timeout) → engines must fall back to rendering from the PDF.

## Env vars (all optional; prefix IDP_PROBE_)
PAGE_IMAGE_DPI (144), WRITE_PAGE_IMAGES (1), IMAGE_BUCKET, N_SAMPLES (3), MAX_SAMPLES (6),
VLM_ENABLED (1), VLM_MODEL_ID (falls back to IDP_BEDROCK_VLM_MODEL_ID), MIN_SCRIPT_CONF (0.6),
HW_THRESHOLD (0.30), PRINTED_SCRIPTS (latin), ONNX_MODEL / ONNX_LABELS, CACHE (1), RESERVE_S (45),
SLIM_OUTPUT (1), FORCE_PROBE_ALL (0).

## Known limits (read before trusting it)
* Not run on AWS or on real scans. Tested with synthetic scanned PDFs + stub S3/VLM (32 tests).
* VLM call is Bedrock **Converse** with an image block against your Qwen3-VL inference profile. Untested against Bedrock.
* Handwriting heuristic was validated only on synthetic typed-vs-wobbly pages. It is only used when neither ONNX nor VLM answered.
  Run `scripts/calibrate_probe_heuristic.py` on labelled pages.
* No trained ONNX classifier is provided; `script_classifier.py` documents the model contract.
* Table detector finds bordered/grid tables only. Borderless scanned tables are not detected.
* Pages Layer 1 already routed to `docling`/`skip` are not probed (so typed-form + handwriting-annotation pages are not seen).
* PaddleOCR `lang` codes in `SCRIPT_TO_PADDLE_LANG` must be verified against your installed version.
* The Step Functions machine has no VLM engine today. The new `InvokeVlmTranscribe` state points at `${VLM_LAMBDA_ARN}`, which does not exist yet.
