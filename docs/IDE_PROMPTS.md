# Prompts for your IDE agent (run in order; each assumes the previous one is done)

## Prompt 1 — drop in the probe
I have added these new files from an architecture bundle. Do NOT modify existing files in this step.
- src/ai/layer1_probe/{__init__,types,imaging,cv_features,script_classifier,vlm_probe,aggregate,routing,script_verify,probe}.py
- src/api/lambda_probe_handler.py
- Dockerfile.probe, requirements-probe.txt
- step_functions/state_machine_with_probe.asl.json
- scripts/calibrate_probe_heuristic.py
- tests/test_probe.py, docs/PROBE_INTEGRATION.md
Install `onnx` and `pytest` in the dev env, run `pytest tests/test_probe.py -q`, and report failures. Check that
`python -c "import src.ai.layer1_probe.probe"` works from the repo root and that nothing in `src/ai/layer1_routing/*`
changed. Read docs/PROBE_INTEGRATION.md for the design.

## Prompt 2 — make both Paddle engines consume the probe output
Files: docker/paddle_printed/{handler,engine,Dockerfile}.py and docker/paddle_handwritten/{handler,engine,Dockerfile}.
The probe adds these optional keys to each engine event, either top-level or nested under event["page"] (the new
state machine passes the whole Map item as "page"): `image_s3_bucket`, `image_s3_key`, `ocr_lang`, `use_structure`.
1. In each handler, read those keys from the event or event["page"]. If `image_s3_key` is present, fetch the JPEG from S3,
   decode with PIL to an RGB numpy array and call engine.process_image(img, page_number, lang=ocr_lang) instead of
   downloading the whole PDF. If the key is missing or the fetch fails, fall back to the existing PDF path unchanged.
2. In each engine, replace the single PaddleOCR singleton with a module-level dict cache keyed by lang
   (default "en"); thread `lang` through process_image/process_pdf_pages. For printed: keep PPStructureV3 only for
   lang=="en"; for any other lang use plain PaddleOCR(lang=...). Do not change the returned result schema; add
   `ocr_lang` to each result.
3. In the Dockerfiles, bake recognition models for the langs in IDP_PROBE_PRINTED_SCRIPTS (start with en only; leave a
   clearly marked ARG for adding more). Before adding any other lang, check that the installed PaddleOCR version actually
   ships a recognizer for it and report which scripts are unsupported.
4. Add tests next to tests/test_layer2_engines_local.py covering: image-key path, missing-key fallback, lang cache reuse.

## Prompt 3 — post-OCR script verification (record only, no re-routing yet)
src/ai/layer1_probe/script_verify.py is stdlib-only. In both Paddle engines call verify_ocr_output(text, expected_script,
confidence) on each page's markdown and add `script_check: {ok, observed_script, reason}` to the result. Map ocr_lang
back to a script with the inverse of SCRIPT_TO_PADDLE_LANG. Copy script_verify.py into the engine images (follow the
existing COPY pattern for document_ir.py with an import fallback). Do not change routing behaviour.

## Prompt 4 — deployment
Mirror the existing pattern in scripts/deploy_idp_ecr_images.py and scripts/deploy_idp_lambdas.py to add a container
Lambda `idp-layer1-probe` built from Dockerfile.probe (repo root context): memory 2048 MB, timeout 900 s,
env IDP_PROBE_PRINTED_SCRIPTS=latin and IDP_BEDROCK_VLM_MODEL_ID from settings. Extend the exec role policy with
bedrock:InvokeModel (and bedrock-runtime Converse equivalents) on the VLM inference profile ARN used in
src/config/settings.py. S3 Get/Put on the bucket is already granted. Update the state-machine deploy step to use
step_functions/state_machine_with_probe.asl.json with ${PROBE_LAMBDA_ARN} substituted. ${VLM_LAMBDA_ARN} does not exist yet:
STOP and tell me rather than inventing it.

## Prompt 5 — smoke test
Extend scripts/test_step_functions.py with a payload containing one typed Latin scanned page, one ruled-table scan, one blank
page, and (if available) one Devanagari scan. After the run, print each page's route/ocr_lang from the probe report in
s3://<bucket>/probe-reports/<doc>.json and confirm page images exist under pages/<doc>/.
