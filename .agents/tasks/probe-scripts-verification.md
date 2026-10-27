# Probe Deployment & Measurement Scripts — Verification Record

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a`  Branch: `feat/llm-observability`  HEAD: `acd90d1`
Python: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`

No AWS command, docker build/push, git commit, or git push was run. The ONLY git
mutation was the single permitted Step‑1b worktree add/remove.

## New files

Scripts:
- `scripts/deploy_probe_image.py`
- `scripts/deploy_probe_service.py`
- `scripts/rollback_probe_service.py`
- `scripts/summarize_probe_results.py`

Tests + fixtures:
- `tests/test_deploy_probe_image.py`
- `tests/test_deploy_probe_service.py`
- `tests/test_summarize_probe_results.py`
- `tests/fixtures/probe/logs_clean.ndjson`
- `tests/fixtures/probe/labels_clean.csv`
- `tests/fixtures/probe/labels_harmful.csv`

## Commands run and results

### Step 1b — clean worktree test (the one permitted git action)

```
git -C <repo> worktree add ../idp-clean HEAD          # created detached HEAD acd90d1
git -C <repo> worktree list                            # confirmed ../idp-clean present
# in ../idp-clean:
.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_inprocess_images.py \
    tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py -q
.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_inprocess_images.py -q
git -C <repo> worktree remove ../idp-clean --force     # removed
git -C <repo> worktree list                            # only main tree remains
```

Result:
- `tests/test_pipeline_probe_document.py` and `tests/test_pipeline_probe_integration.py`
  **failed collection** in the clean HEAD with:
  `ModuleNotFoundError: No module named 'src.ai.layer2_conversion.vlm_block_rollup'`
  (pipeline.py:37 imports it). That module is **untracked** on HEAD, as is
  `src/ai/layer1_probe/windows.py`. So the clean checkout lacks:
  `src/ai/layer2_conversion/vlm_block_rollup.py` (untracked) and
  `src/ai/layer1_probe/windows.py` (untracked), plus the two `*.onnx` weights
  (gitignored). These must be committed / shipped for the probe path to import.
- `tests/test_probe.py` + `tests/test_probe_inprocess_images.py`: **44 passed** in the
  clean tree (they use stubs, not the real pipeline import).
- After removal, `git worktree list` shows only the main tree; `git status` unchanged.

### New unit tests (repo venv, single run, no network/docker/aws)

```
.venv\Scripts\python.exe -m pytest tests/test_deploy_probe_image.py \
    tests/test_deploy_probe_service.py tests/test_summarize_probe_results.py -q
```

Result: **38 passed in 0.77s** (after one fixup — see note below).

Per-file pytest counts:
- `tests/test_deploy_probe_image.py` — 16 tests
- `tests/test_deploy_probe_service.py` — 12 tests
- `tests/test_summarize_probe_results.py` — 10 tests

Note: one summarizer test initially asserted `total==3` for per‑script accuracy;
corrected to `4` because page 4 of the fixture carries a `lipika_script` (latin)
with `vlm_script=null`, so it IS counted in the join. Re‑run: all green.

### Dry‑run smoke checks (no AWS, no docker executed)

`deploy_probe_image.py --dry-run`:
- stdout = ONLY the image URI:
  `530084039784.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-202610251144-acd90d1-dirty`
- stderr printed model size+sha256 for `lipika_script.onnx` (111,642,305 B) and
  `lipika_script.labels.json` (541 B); the `docker build --provenance=false
  --platform linux/amd64 -f schema_chatbot_v2/Dockerfile …` command; and the
  `docker login -u AWS -p **** …` line (password masked). Exit 0. No boto3/ECR
  call in dry‑run (account taken from the mirrored constant, clearly labelled).

`rollback_probe_service.py --task-definition-arn …:3 --dry-run`:
- printed the `update_service` call to revision `:3`; "no AWS mutation performed";
  exit 0.

`deploy_probe_service.py` dry‑run paths are covered by the unit tests
(`test_dry_run_masks_and_writes_no_rollback`, `test_dry_run_flag_off`,
`test_rejects_latest_image_arg`): register JSON printed with
`IDP_PROBE_ENABLED=true/false`, `JWT_SECRET`/`ADMIN_PASSWORD` masked, DB password
masked, NO `IDP_PROBE_N_SAMPLES` added, NO rollback file written, NO AWS mutation.

### Cleanup
- Transient worktree `../idp-clean` removed.
- Temp stderr capture file deleted.
- No `rollback_*.json` left in the repo (confirmed by glob).
