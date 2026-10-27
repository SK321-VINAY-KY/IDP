# Probe Preflight + Additive-Change — Verification Note

Repo: `c:\Users\Dell\Desktop\IDP\engineer_a` · branch `feat/llm-observability`
Iteration: FIRST (no `probe-preflight-review.json` present — implemented from `probe-preflight-plan.md`).
Venv python for ALL pytest: `c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`

## Compliance confirmation
- NO `git commit`, NO `git push` performed.
- NO AWS command / boto3-to-live-service run. `deploy_probe_image.py --dry-run` is local-only (see 3c).
- NO `docker build` / `docker push` run.
- NO `.onnx` file staged, tracked, or committed (`git ls-files "*.onnx"` empty; nothing staged).
- Flag-off behaviour byte-identical: all additions stay inside `if "probe" in page_context:`; the
  new `final_engine` local only feeds that event and never touches routing/output/metadata.
- Only allowed files edited: `src/ai/layer1_routing/pipeline.py` (additive),
  `scripts/summarize_probe_results.py`, `tests/test_pipeline_probe_document.py`,
  `tests/test_summarize_probe_results.py`, and new fixtures
  `tests/fixtures/probe/logs_final_engine.ndjson`, `tests/fixtures/probe/labels_final_engine.csv`.
  No edits to `deploy_idp_fargate.py`, `settings.py`, deploy scripts, the Dockerfile, or `.dockerignore`.

---

## Item 1 — extend `probe.page_decision` (pipeline.py, additive)

Per-file change summary (`src/ai/layer1_routing/pipeline.py`, inside `process_page`):
1. After the initial `markdown, confidence, engines_used = _merge_results(mergeable_results)` and
   BEFORE the escalation `while` loop, added a new local:
   `final_engine = engines_used[0] if engines_used else None`.
2. Updated `final_engine` at the adoption points INSIDE the ladder (the only places output is
   actually adopted):
   - VLM-wins branch (markdown/confidence/page_blocks replaced by the VLM result):
     `final_engine = escalation_engine`.
   - OCR-to-OCR branch (after the re-merge): `final_engine = engines_used[0] if engines_used else final_engine`.
   - VLM-loses branch: left unchanged (prior OCR engine stays adopted even though `vlm_transcribe`
     is appended to `engines_used`).
3. Extended the existing `logger.info("probe.page_decision", ...)` call with TWO additive kwargs:
   `engines_used=output.engines_used` (full list) and `final_engine=final_engine` (adopted winner).
   `engine_used` is UNCHANGED (`output.engines_used[0] if output.engines_used else None`). No existing
   field removed/renamed. No image bytes, no markdown/text.

### `final_engine` / `engines_used` source convention actually used
- `engines_used` field = `output.engines_used` — the full post-ladder list on the returned PageOutput
  (primary first; escalation engine appended by the ladder).
- `final_engine` field = a NEW local tracked through the ladder = the engine whose OUTPUT was adopted.
  This is NOT a single existing field: `engines_used[0]` is the primary (wrong for the VLM-wins case),
  `engines_used[-1]` is the last-appended (wrong for the VLM-loses case, which appends `vlm_transcribe`
  without adopting it), and `PageMetadata.final_engine` == `engines_used[0]` (no extra info). The adopted
  engine IS deterministic at the ladder's win/lose branch points, so tracking it in a local is additive
  and behaviour-preserving — NOT a STOP-and-report situation (matches the plan's CRITICAL FINDING).
- Observable truth asserted in tests: for the printed→VLM escalation (VLM wins), the real pipeline
  produces `engines_used == ["paddleocr_printed", "vlm_transcribe"]`, `engine_used == "paddleocr_printed"`
  (unchanged primary = engines_used[0]), and `final_engine == "vlm_transcribe"` (adopted winner).

Tests (`tests/test_pipeline_probe_document.py`):
- Updated `test_probe_page_decision_emitted_on_probed_page`: asserts `engines_used == ["paddleocr_printed"]`
  and `final_engine == "paddleocr_printed"` (no-escalation case), all prior assertions kept.
- Updated `test_probe_page_decision_shows_post_ladder_values`: asserts `engines_used == out.engines_used`,
  `"vlm_transcribe" in engines_used`, `final_engine == "vlm_transcribe"`.
- Added `test_probe_page_decision_printed_escalates_to_vlm_final_engine`: primary `paddleocr_printed`
  returns empty text → ladder escalates and adopts VLM; asserts
  `engines_used == ["paddleocr_printed", "vlm_transcribe"]`, `final_engine == "vlm_transcribe"`,
  `engine_used == engines_used[0] == "paddleocr_printed"`, `escalated is True`.
- `test_probe_page_decision_not_emitted_flag_off` kept unchanged (proves flag-off emits nothing).

---

## Item 2 — summarizer uses `final_engine` (summarize_probe_results.py)

Per-file change summary (`scripts/summarize_probe_results.py`, inside `summarize()`):
- When reading `probe.page_decision`, the FINAL adopted engine stored in `pd_final_engine[key]` now uses
  a 3-level fallback: prefer `rec["final_engine"]`; else `rec["engines_used"][-1]` (non-empty list);
  else the legacy `rec["engine_used"]`. A comment notes that under the current pipeline `engine_used`
  is the PRIMARY and is used here only as the last-resort back-compat fallback.
- No other logic changed: `harmful_rows`, `near_miss_rows`, per-route / per-final-route stats, and the
  non-zero exit on HARMFUL>0 already consume `pd_final_engine` via `summary["page_engine"]`
  (confirmed — no direct `engine_used` read needed redirecting). The `probe_route` distribution
  (pages-per-probe-route) and the near-miss metric are preserved verbatim. Cannot-compute messaging
  unchanged.

Tests + fixtures:
- New `tests/fixtures/probe/logs_final_engine.ndjson` (doc `docF`): p1 probe_route=paddleocr_printed,
  escalated, `final_engine=paddleocr_printed` (HARMFUL); p2 probe_route=paddleocr_printed, escalated away,
  `final_engine=vlm_transcribe` (near-miss). On BOTH pages `engine_used=paddleocr_printed` (primary),
  proving the summariser keys off `final_engine`, not `engine_used`.
- New `tests/fixtures/probe/labels_final_engine.csv`: `docF,devanagari`.
- Added `test_final_engine_harmful_and_near_miss`: HARMFUL==1 (p1), near-miss==1 (p2), p2 not harmful.
- Added `test_final_engine_fallback_to_engines_used_last`: no `final_engine`, uses `engines_used[-1]`.
- Added `test_final_engine_fallback_to_engine_used`: neither field, falls back to `engine_used`
  (back-compat with existing `logs_page_decision.ndjson`).
- Added `test_cli_final_engine_harmful_exits_non_zero`: CLI exits non-zero, HARMFUL=1, near-miss=1.
- Existing `logs_page_decision.ndjson` / `logs_clean.ndjson` tests still pass via the `engine_used`
  fallback (those rows predate `final_engine`/`engines_used`).

---

## Item 3 — LOCAL PREFLIGHT (read-only)

### 3a — test counts (repo venv python, single run)
Command (full set):
```
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe -m pytest tests/test_probe.py tests/test_probe_inprocess_images.py tests/test_pipeline_probe_document.py tests/test_pipeline_probe_integration.py tests/test_summarize_probe_results.py -q
```
Result: **91 passed** (collection succeeded, 0 failed, 0 errors). Per-file (run individually):

| File | Passed |
|------|--------|
| tests/test_probe.py | 31 |
| tests/test_probe_inprocess_images.py | 13 |
| tests/test_pipeline_probe_document.py | 15 |
| tests/test_pipeline_probe_integration.py | 7 |
| tests/test_summarize_probe_results.py | 25 |
| **Total** | **91** |

`tests/test_layer2_engines_local.py::*_engine_local_path` was NOT run (per instruction).

### 3b — lipika model + labels in build context, sizes + sha256
Build context root = repo ROOT (`deploy_probe_image.py` `ROOT = parents[1]`). All three present:

| File | Size (bytes) | sha256 |
|------|--------------|--------|
| src/ai/layer1_probe/models/lipika_script.onnx | 111642305 | 21c0af2eb471f322b3f7b5bcc2debb2b9ccf6c8143eae4f2949eb92237402fc6 |
| src/ai/layer1_probe/models/lipika_script.int8.onnx | 28494953 | b1d55551e9bb41c4723d47b0bed097222bd366f1fa6d50c82a372b8f3d9f225f |
| src/ai/layer1_probe/models/lipika_script.labels.json | 541 | 42343a16c15bc1d86fd49302985894436c61ff0d5a4558d7cfa2097969c8f79a |

Only ONE `.dockerignore` exists (repo root); no `schema_chatbot_v2/.dockerignore`. Root `.dockerignore`
excludes `.git/`, `.venv/`, `tests/`, `docs/`, `node_modules/`, `.env*`, `*.db`, `*.sqlite3`, `*.bin`,
`*.safetensors`, `.cache/`, `.kiro/`, `logs/`, `data/`, etc. It has NO `*.onnx` and NO `*.json` pattern,
so the three model files are NOT excluded. (They are gitignored, so they ship via the Dockerfile
`COPY . /app`, not via git — must NOT be staged.)

### 3c — `deploy_probe_image.py --dry-run` AWS determination
RAN (no AWS creds needed). Inspected `main()`: the `--dry-run` branch uses the mirrored
`EXPECTED_ACCOUNT` constant, builds the URI, and prints a masked `docker login`, `docker build`, and
`docker push` to stderr plus the URI to stdout. All AWS calls (`import boto3`, `sts.get_caller_identity()`,
`ecr.*`) live ONLY in the real-run branch AFTER the `if args.dry_run: ... return 0`. Before that branch
it calls `git_short_sha`, `git_porcelain` (local git) and `verify_models_in_context` (local sha256 read)
— all local. Ran:
```
c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe scripts/deploy_probe_image.py --dry-run
```
Process exit code = 0. stdout (single line) = the image URI
`530084039784.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-<stamp>-<sha>-dirty`. stderr printed
the model `[model]` lines and the masked login/build/push commands. Built/pushed nothing, touched no AWS.
(Note: running under PowerShell `2>&1` shows a spurious `Exit Code: 1` because PS treats native-command
stderr as error output; the true process exit code is 0, confirmed by redirecting both streams.)

### 3d — Dockerfile ships the probe runtime
`schema_chatbot_v2/Dockerfile` uses a single `COPY . /app` from the build-context root (quoted):
```
COPY . /app
```
This copies the entire WORKING TREE regardless of git-tracked status, so it ships
`src/ai/layer1_probe/` (including `windows.py` and the `models/` dir with the onnx + labels) and
`src/ai/layer2_conversion/vlm_block_rollup.py` (via the full `src/` copy). No per-file COPY is needed;
NO required runtime file is missed. The Dockerfile comment confirms the probe model resolves at
`/app/src/ai/layer1_probe/models/lipika_script.onnx` and that `.dockerignore` does not exclude
`*.onnx` / `*.json`.

---

## Item 4 — exact command list (from REAL argparse; placeholders)

`PY = c:\Users\Dell\Desktop\IDP\engineer_a\.venv\Scripts\python.exe`
Placeholders: `<CLUSTER>` (default `idp-cluster`), `<SERVICE>` (default `idp-layer3-service`),
`<REGION>` (default `ap-south-1`), `<ACCOUNT>`, `<IMAGE_URI>` (stdout of the build step),
`<LOG_GROUP>` (`/ecs/idp-layer3-app`), `<START_EPOCH>`, `<END_EPOCH>`, `<LOGS_NDJSON>`, `<LABELS_CSV>`.

Real argparse (re-verified against the scripts):
- `deploy_probe_image.py`: `--dry-run`, `--region` (default `ap-south-1`). NO `--image` — it PRODUCES
  the image URI on stdout.
- `deploy_probe_service.py`: `--image` (REQUIRED), `--cluster`, `--service`, `--container`,
  `--flag-off` (store_true → `IDP_PROBE_ENABLED=false`; absent → `true`), `--region`, `--dry-run`.
- `rollback_probe_service.py`: REQUIRED mutually-exclusive `--rollback-file` XOR `--task-definition-arn`;
  plus `--cluster`, `--service`, `--region`, `--dry-run`.
- `summarize_probe_results.py`: `--logs` (REQUIRED), `--labels` (optional). NO `--dry-run`.

1. Login / credential check (documented precheck; also the account STS assert the real build does):
```
aws sts get-caller-identity --region <REGION>
```
2. Image build DRY-RUN (no AWS creds; prints URI + masked cmds):
```
PY scripts/deploy_probe_image.py --dry-run --region <REGION>
```
3. Real build + push (REQUIRES AWS creds + docker — DO NOT run in this workflow; capture stdout URI):
```
PY scripts/deploy_probe_image.py --region <REGION>
```
4. Deploy with probe FLAG OFF — dry-run FIRST, then real:
```
PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --flag-off --dry-run
PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --flag-off
```
5. Deploy with probe FLAG ON — dry-run first, then real (omit `--flag-off` → `IDP_PROBE_ENABLED=true`):
```
PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION> --dry-run
PY scripts/deploy_probe_service.py --image <IMAGE_URI> --cluster <CLUSTER> --service <SERVICE> --region <REGION>
```
6. Rollback to `idp-layer3-task:3` (ARN form; `--task-definition-arn` XOR `--rollback-file`):
```
PY scripts/rollback_probe_service.py --task-definition-arn arn:aws:ecs:<REGION>:<ACCOUNT>:task-definition/idp-layer3-task:3 --cluster <CLUSTER> --service <SERVICE> --region <REGION> --dry-run
PY scripts/rollback_probe_service.py --task-definition-arn arn:aws:ecs:<REGION>:<ACCOUNT>:task-definition/idp-layer3-task:3 --cluster <CLUSTER> --service <SERVICE> --region <REGION>
```
(Alternatively `--rollback-file <ROLLBACK_FILE>` to restore from a snapshot.)
7. CloudWatch log export (AWS CLI the USER runs) + summarizer:
```
aws logs filter-log-events --log-group-name <LOG_GROUP> --start-time <START_EPOCH> --end-time <END_EPOCH> --region <REGION>
PY scripts/summarize_probe_results.py --logs <LOGS_NDJSON> --labels <LABELS_CSV>
```
(The summarizer exits non-zero if HARMFUL > 0.)

---

## Item 5 — secret-file warning (NAMES ONLY)

Build context = repo root recursively (honoring `.dockerignore`; `.git/`, `.venv/`, `node_modules/`,
`__pycache__/` excluded from the scan). Files whose NAME looks like a secret / should not be baked in:

| Filename (path) | Looks like | Excluded by `.dockerignore`? |
|-----------------|-----------|------------------------------|
| `.env` | env secrets | YES (`.env*`, root) |
| `.env2` | env secrets (holds live STS creds) | YES (`.env*`, root) |
| `.env.example` | env template (not a live secret) | YES (`.env*`, root) |
| `schema_chatbot_v2/.env.example` | env template | **NO** — `.env*` is a root-level pattern; it does not match this nested path (would need `**/.env*`). It is a `.example` template, not a live secret, but note it is NOT excluded. |
| `.secrets.baseline` | detect-secrets baseline | **NO** — no matching pattern (name starts with `.secrets`, not `.env*`). It is a detect-secrets config, not raw credentials, but it is NOT excluded. |

- `.env*` confirmation: the rule covers BOTH `.env` and `.env2` at the repo root (the latter holds live
  STS creds) — both excluded YES.
- `rollback_*.json`: NONE present on disk currently. Pattern status — gitignored YES (`.gitignore` line
  `rollback_*.json`); dockerignored **NO** (no `rollback_*` pattern in `.dockerignore`). If such a file is
  ever generated by `deploy_probe_service.py`, it WOULD enter the build context — must be deleted or
  dockerignored before any image build, and never committed (it contains UNMASKED task-def env secrets).
- No `*.pem`, `*.key`, `id_rsa`, `*.pfx`, `*.p12`, or `*credentials*` files found in the context.
- `*.onnx` model files are NOT secrets but are gitignored and MUST NOT be staged/committed (ship via
  the Dockerfile `COPY . /app`).

Contents of none of the above were read or printed.

---

## Contradictions vs the brief
1. Item 4 of the brief grouped `--image` / flag-off under the image-build script; in reality
   `deploy_probe_image.py` takes only `--dry-run` / `--region` and OUTPUTS the URI, while `--image`
   and `--flag-off` belong to `deploy_probe_service.py`. The command list uses the REAL flags.
2. `rollback_probe_service.py` requires EXACTLY ONE of `--rollback-file` / `--task-definition-arn`
   (mutually exclusive, required) — the brief lists both as if independently optional.
3. `final_engine` is NOT already stored in any single post-ladder field (`engines_used[0]`,
   `engines_used[-1]`, and `PageMetadata.final_engine` each misreport at least one escalation case).
   It is recovered additively by tracking the adopted engine in a local var as the ladder runs — this is
   deterministic and behaviour-preserving, so NOT a STOP-and-report situation.
4. `.env*` in `.dockerignore` excludes root `.env`/`.env2`/`.env.example` but, being a root-level pattern,
   does NOT exclude the nested `schema_chatbot_v2/.env.example`; and `.secrets.baseline` /
   `rollback_*.json` are not dockerignored at all (flagged in Item 5).
