# IDP System — Full Pipeline Documentation
## Layer 1 + Layer 2 (Routing + Conversion) · Layer 3 (Extraction) · API · PostgreSQL · Frontend

> **Responsibility split**
> - **Layer 1 + Layer 2** — Accept any PDF, detect what's on each page, pick the right extraction engine(s), output structured Markdown with provenance metadata. Schema-agnostic.
> - **Layer 3** — Consume the `.md` output + a target schema, scan each page for matching fields, accumulate results in a scratchpad, return structured JSON. Schema-aware.
> - **API** — FastAPI service wiring all layers together. Accepts PDF + schema from frontend, returns JSON.
> - **PostgreSQL** — Persists every extraction run (engines used, confidence, result JSON) for audit and replay.
> - **Frontend** — React/Vite SPA on port 5173. Uploads PDF + schema, displays extraction results.

---

## System Architecture

```mermaid
flowchart TB
P["IDP PRODUCT"]
A["Layer 1 + Layer 2\nRouting + Conversion\nSchema-agnostic\nProduces .md + provenance"]
B["Layer 3 Extraction\nSchema-aware\nConsumes .md + target schema\nProduces structured JSON"]
API["FastAPI :8000\n/api/extract"]
DB["PostgreSQL\nprocessing_results table"]
FE["Frontend :5173\nReact/Vite SPA"]

FE -->|"PDF + target_schema (JSON)"| API
API --> A
A -->|"list[PageOutput]"| B
B -->|"result JSON"| API
API -->|"structured JSON response"| FE
API --> DB
```

### Layer 3 Cache / Schema Discovery Phase
Before batch processing, a target schema is either:
1. **Supplied by the user** via the API (`target_schema` JSON field list) — the API builds a dynamic Pydantic model at request time via `dynamic_schema.py`.
2. **Defined in a JSON file** (`tests/schemas/*.json`) for dev/test runs via `run_extraction.py`.

---

## Layer 1 + Layer 2 Data Flow

```mermaid
flowchart TB
PDF["PDF"]
A["Step A — inspect_page()\nPyMuPDF heuristics\nchar_count · image_coverage · is_scanned\nprimary_script · complexity_score · has_tables\nhas_vector_drawings · dpi_estimate"]
B["Step B — Route Resolution\nsettings.routing_mode"]
S["single_engine\nroute_from_profile() → route or None"]
C["capability_based\ncapabilities_from_profile() → PageCapabilities"]
V["VLM analysis for ambiguous pages\nanalyze_page() → enrich capabilities"]
PLAN["Engine Plan Execution + Merge\n1. docling — digital text\n2. paddleocr_printed — scanned printed\n3. paddleocr_handwritten — handwriting\n4. vlm_transcribe — fallback"]
E["Escalation Ladder\ndigital → printed OCR → handwritten OCR → VLM → GIVE UP"]
OUT["Step 5 — Output\nPageOutput list (in-memory)\n+ optional .md file write"]

PDF --> A --> B
B --> S
B --> C
S -->|definitive| PLAN
S -->|ambiguous| V
C -->|clear| PLAN
C -->|ambiguous| V
V --> PLAN
PLAN --> E --> OUT
```

---

## Layer 3 Data Flow

```mermaid
flowchart TB
PO["list[PageOutput]\nfrom Layer 1+2 pipeline"]
PL["page_loader.py\nload_pages_with_confidence()\nExtracts markdown + page_number per page\nReturns list[dict]"]
SC["Target Schema\nDynamic Pydantic model\nbuilt from user-supplied field list\n{name, description} per field"]
EXT["extractor.py\nextract_by_page_scan()\n\n1. Cache schema fields once\n2. For each page:\n   - skip if no missing fields\n   - call check_page_for_fields()\n   - update scratchpad\n3. Hydrate final Pydantic model"]
SP["scratchpad.py\nScratchpad\n\nIn-memory accumulator\nfirst-seen value wins per field\nstores {value, page} per hit"]
LLM["SarvamExtractionClient\ncheck_page_for_fields(\n  page_md,\n  fields_to_check,\n  page_number,\n  total_pages\n)\n\nReturns [{field, value}]"]
RES["Final BaseModel instance\nAll found fields filled\nMissing fields = empty string"]

PO --> PL
PL --> EXT
SC --> EXT
EXT --> LLM
LLM -->|"matches list"| SP
SP --> EXT
EXT --> RES
```

---

## Technology Stack

| Role | Library | Notes |
|------|---------|-------|
| **PDF inspection** (Layer 1 Step A) | **PyMuPDF** | `get_text("words")`, `get_image_info()`, `get_drawings()`, script detection, DPI estimate. ~10ms/page, zero model calls. |
| **Digital text extraction** (Layer 2) | **Docling 2.x** | Native Markdown output with headings + tables. Falls back to `pymupdf.get_text("text")` if Docling unavailable. |
| **Scanned OCR** (Layer 2) | **PaddleOCR PP-OCRv6** | Singleton per mode (printed/handwritten). CPU-only. Confidence = weighted avg of per-word `rec_scores`. |
| **Handwritten OCR** (Layer 2) | **PaddleOCR handwriting-tuned** | Separate singleton, lower `det_db_thresh` (0.20 vs 0.30) for thinner strokes. |
| **VLM classification** (Layer 1 Step B) | **Ollama** or **Google Gemini** | Only invoked for genuinely ambiguous pages. Digital pages > 100 chars skip VLM entirely. |
| **Field extraction LLM** (Layer 3) | **Sarvam AI (`sarvam-105b`)** | OpenAI-compatible REST endpoint. `check_page_for_fields()` uses raw `OpenAI` client (not instructor) to avoid `response_model=None` conflicts. `extract()` uses instructor for structured output. |
| **LLM fallback** (Layer 3) | **Ollama `qwen2.5:7b`** | `OllamaExtractionClient` mirrors Sarvam interface. Switched via `IDP_EXTRACTION_BACKEND=ollama`. |
| **JSON fence stripping** | `_strip_fences()` in `sarvam_client.py` | Sarvam wraps responses in ` ```json ``` ` — stripped before `json.loads()`. Was root cause of 52% → 100% accuracy jump. |
| **API framework** | **FastAPI 0.115** + **Uvicorn** | Single endpoint `/api/extract`. CORS enabled for frontend on port 5173. |
| **Database** | **PostgreSQL** via **SQLAlchemy 2.x** | `processing_results` table. One row per extraction run. DB failure is non-fatal — API still returns result. |
| **Schema validation** | **Pydantic v2** | `DocumentExtraction` for dev; `build_dynamic_schema()` for API. All fields `str` with `""` default — no JSON nulls. |
| **Configuration** | **pydantic-settings** + `.env` | `IDP_` prefix for all settings. `IDP_EXTRACTION_BACKEND`, `IDP_SARVAM_API_KEY`, `IDP_DATABASE_URL`. |
| **Logging** | Custom JSON logger | Every page scan decision logged with `page_number`, `fields_found`, `scratchpad` snapshot. |
| **Frontend** | **React + Vite** | Port 5173. Sends `multipart/form-data` with `file` + `target_schema` JSON string. |

---

## Repository Layout

```
backend/
│
├── run_extraction.py          ← Generic dev/test runner. Accepts --doc and --schema CLI args.
│                                 Reads fixture .md + schema .json, runs Layer 3, prints + saves result.
│                                 Usage: python run_extraction.py --doc sdg_goals_output --schema sdg_goals_schema
│
├── src/
│   ├── config/
│   │   └── settings.py        All tunables. IDP_ prefix env vars:
│   │                            extraction_backend ("sarvam"|"ollama")
│   │                            sarvam_api_key, sarvam_model_name ("sarvam-105b")
│   │                            sarvam_base_url
│   │                            extraction_model_name ("qwen2.5:7b") — Ollama fallback
│   │                            extraction_chunk_size (=3) — kept for reference, not used
│   │                            database_url (PostgreSQL)
│   │                            max_extraction_retries (=2)
│   │
│   ├── utils/
│   │   └── logger.py          JSON structured logger, correlation_id context var.
│   │
│   ├── adapters/
│   │   └── llm/
│   │       ├── base.py          LLMClient ABC — Layer 1/2 vision interface only.
│   │       ├── extraction_base.py  ExtractionLLMClient Protocol — Layer 3 text interface.
│   │       │                         Methods: extract(), check_page_for_fields()
│   │       ├── extraction_factory.py  get_extraction_client() → Sarvam or Ollama.
│   │       ├── sarvam_client.py  ⭐ Layer 3 LLM implementation.
│   │       │                       _strip_fences() — strips ```json``` wrappers before parsing.
│   │       │                       extract() — instructor + response_model for structured output.
│   │       │                       check_page_for_fields() — raw OpenAI client, parses matches JSON.
│   │       ├── extraction_client.py  Ollama mirror of sarvam_client.py.
│   │       └── ollama_client.py  Layer 1/2 VLM (classify_page, transcribe_handwriting).
│   │
│   ├── ai/
│   │   ├── schemas/
│   │   │   ├── page.py          ⭐ CROSS-TEAM CONTRACT.
│   │   │   │                      PageProfile — Layer 1 Step A output.
│   │   │   │                      PageClassification — Layer 1 Step B light VLM output.
│   │   │   │                      PageOutput — THE contract handed to Layer 3:
│   │   │   │                        page_number, markdown, engine_used, confidence,
│   │   │   │                        escalated, escalation_attempts, low_confidence,
│   │   │   │                        primary_script, complexity_score, has_images.
│   │   │   └── extraction_schema.py  DocumentExtraction — dev/test schema (19 SDG goals).
│   │   │                              Not used by API — API uses dynamic_schema.py.
│   │   │
│   │   ├── layer1_routing/      Layer 1: inspect → route → plan. (Engineer A's code.)
│   │   │   ├── inspect.py
│   │   │   ├── router.py
│   │   │   └── pipeline.py      process_document() → list[PageOutput]
│   │   │
│   │   ├── layer2_conversion/   Layer 2: actual OCR/extraction engines. (Engineer A's code.)
│   │   │   ├── digital.py       Docling + PyMuPDF fallback.
│   │   │   ├── scanned.py       PaddleOCR printed + handwritten modes.
│   │   │   └── handwritten.py   Thin shim → scanned.py.
│   │   │
│   │   └── layer3_extraction/   ⭐ Layer 3: field extraction from .md. (Engineer B's code.)
│   │       ├── extractor.py     extract_by_page_scan() — main entry point.
│   │       │                      1. Cache schema fields [{name, description}] once.
│   │       │                      2. For each page: send only missing fields to LLM.
│   │       │                      3. scratchpad.update() with matches.
│   │       │                      4. Stop when all fields found OR all pages scanned.
│   │       │                      5. schema.model_validate({defaults, ...scratchpad}).
│   │       │
│   │       ├── scratchpad.py    Scratchpad — in-memory per-field accumulator.
│   │       │                      _hits: {field_name: {value, page}}
│   │       │                      update() — first-seen wins; unknown fields ignored.
│   │       │                      missing_fields — fields not yet found.
│   │       │                      snapshot() — [{field, value, page}] logged after each page.
│   │       │                      provenance() — {field: page_number} audit trail.
│   │       │
│   │       ├── page_loader.py   Bridges Layer 2 output → Layer 3 input.
│   │       │                      load_pages_with_confidence(page_outputs) → list[dict]
│   │       │                        Extracts {markdown, page_number} per page.
│   │       │                        Name is a legacy artifact — confidence was removed.
│   │       │                        No confidence data is passed to Layer 3.
│   │       │                      load_pages_from_fixture(doc_id) → list[dict]
│   │       │                        Reads tests/fixtures/<doc_id>.md, splits on
│   │       │                        <!-- Page N | ... --> markers.
│   │       │
│   │       ├── schema_validation.py  extract_with_retry() — wraps extract_by_page_scan()
│   │       │                          in a retry loop (max_extraction_retries=2).
│   │       │                          Catches ValidationError and retries.
│   │       │
│   │       ├── storage.py       PostgreSQL persistence via SQLAlchemy.
│   │       │                      ProcessingResult table — one row per extraction run.
│   │       │                      Columns: doc_id, page_count, engines_used, avg_confidence,
│   │       │                        schema_name, result_json (JSONB), strategy_used,
│   │       │                        processing_time_seconds, created_at.
│   │       │                      DB failure is non-fatal — API still returns result.
│   │       │
│   │       └── prompts/
│   │           ├── configs/
│   │           │   └── prompt_config.yaml   extraction: (temp=0, max_tokens=2000)
│   │           │                             page_field_check: (temp=0, max_tokens=1000)
│   │           ├── loader.py    render_prompt() + prompt_params() via Jinja2 + lru_cache.
│   │           └── templates/
│   │               ├── extraction_system.j2   Used by extract() — full doc extraction.
│   │               └── page_field_check.j2   ⭐ Used by check_page_for_fields().
│   │                                            Tells LLM: page N of M, field list,
│   │                                            page content, "check ALL fields",
│   │                                            returns {"matches": [{field, value}]}.
│   │
│   └── api/
│       ├── main.py              ⭐ FastAPI app. Single extraction endpoint.
│       │                          POST /api/extract — multipart: file + target_schema.
│       │                          GET  /api/health
│       │                          Flow: parse schema → Layer1+2 → page_loader → Layer3 → save_db → return.
│       │                          CORS: localhost:5173 allowed.
│       │
│       ├── dynamic_schema.py    build_dynamic_schema(fields) → type[BaseModel].
│       │                          Slugifies field names, deduplicates, creates Pydantic model
│       │                          at request time. All fields str with "" default.
│       │
│       └── document_processor.py  build_pages_for_document(pdf_path) → (doc, pages).
│                                    Opens PDF with PyMuPDF, renders each page at 200 DPI,
│                                    returns page objects + image arrays for Layer 1+2.
│
└── tests/
    ├── fixtures/
    │   └── sdg_goals_output.md   Layer 2 output fixture (18 pages, SDG Goals document).
    │                               Format: <!-- Page N | route=... | chars=... --> markers.
    │                               Used by run_extraction.py for offline Layer 3 testing.
    └── schemas/
        └── sdg_goals_schema.json  Target schema JSON for sdg_goals_output fixture.
                                    Format: [{"name": "...", "description": "..."}, ...]
                                    Can be swapped without any code changes.
```

---

## Layer 3 → Layer 3 Contract Detail

### `page_field_check.j2` prompt (what Sarvam sees per page)

```
You are extracting structured data from page {page_number} of {total_pages}.

Target schema fields:
- document_title: The main title of this document...
- source_url: A URL mentioned near the top...
- ...only the fields STILL MISSING are sent here...

Page content:
{page_md}

Instructions:
- Go through EVERY schema field one by one.
- Only extract if the value is on THIS page.
- Extract exact values as written.

Respond with ONLY:
{"matches": [{"field": "<name>", "value": "<value>"}]}
```

### `_strip_fences()` — the critical parse fix

Sarvam wraps JSON responses in markdown code fences:
```
```json
{"matches": [...]}
```
```

Without stripping, `json.loads()` throws `JSONDecodeError` and the entire page's correct data is discarded. This single bug caused 52% accuracy before the fix. Fix:
```python
raw = re.sub(r"^```(?:json)?\s*", "", raw)
raw = re.sub(r"\s*```$", "", raw).strip()
```

### Scratchpad field resolution logic

```
For every page (1 to N):
  1. fields_to_check = schema fields NOT yet in scratchpad._hits
  2. If fields_to_check is empty → stop scanning (all fields found)
  3. Send page_md + fields_to_check to LLM → get matches list
  4. If matches is empty → page had no relevant fields → scratchpad unchanged, move to next page
  5. If matches has hits → scratchpad.update() stores {value, page} for each matched field
  6. That field is removed from fields_to_check on the next iteration (won't be asked again)
```

**Every page is sent to the LLM** (unless all fields already found). Pages that don't contain any target fields simply return `{"matches": []}` and contribute nothing to the scratchpad — they are checked but produce no hits. The page is not skipped before the LLM call; it is always checked.

First-seen wins. A field once stored is never overwritten — it is dropped from `fields_to_check` so subsequent pages are never even asked about it. Unknown field names returned by the LLM are silently dropped (hallucination guard: `if field not in self._all_fields`).

---

## PostgreSQL Schema

```sql
CREATE TABLE processing_results (
    id                        SERIAL PRIMARY KEY,
    doc_id                    VARCHAR NOT NULL,
    page_count                INTEGER,
    routes_used               JSONB,          -- Layer 1+2 routes per doc
    engines_used              JSONB,          -- Layer 1+2 engines per doc
    primary_scripts           JSONB,          -- scripts detected (latin, devanagari...)
    avg_confidence            FLOAT,          -- avg OCR confidence across pages
    min_confidence            FLOAT,          -- worst page OCR confidence
    low_confidence_page_count INTEGER,
    escalated_page_count      INTEGER,
    total_escalation_attempts INTEGER,
    max_complexity_score      INTEGER,
    has_images                BOOLEAN,
    schema_name               VARCHAR NOT NULL,
    result_json               JSONB NOT NULL, -- the extracted field values
    llm_provider              VARCHAR,        -- "sarvam" or "ollama"
    model_name                VARCHAR,        -- "sarvam-105b" or "qwen2.5:7b"
    strategy_used             VARCHAR,        -- "page_scan"
    processing_time_seconds   FLOAT,
    created_at                TIMESTAMP DEFAULT now()
);
```

---

## API Endpoints

### `POST /api/extract`
**Request:** `multipart/form-data`
- `file` — PDF file
- `target_schema` — JSON string: `[{"name": "field_name", "description": "..."}, ...]`

**Response:**
```json
{
  "success": true,
  "data": {
    "document_title": "5th SDG Youth Summer Camp...",
    "source_url": "https://...",
    "first_goal": "End poverty..."
  },
  "meta": {
    "strategy_used": "page_scan",
    "processing_time_seconds": 18.4,
    "llm_provider": "sarvam",
    "saved_to_db": true,
    "db_result_id": 54
  }
}
```

### `GET /api/health`
Returns `{"status": "ok"}`.

---

## Dev/Test Runner

```bash
# Default: sdg_goals_output.md + sdg_goals_schema.json
python run_extraction.py

# Custom document and schema
python run_extraction.py --doc my_invoice --schema invoice_schema
```

`tests/schemas/invoice_schema.json`:
```json
[
  {"name": "invoice_number", "description": "The invoice number"},
  {"name": "vendor_name",    "description": "Name of the vendor"},
  {"name": "total_amount",   "description": "Total amount due"},
  {"name": "due_date",       "description": "Payment due date"}
]
```

No code changes needed. Drop `.md` in `tests/fixtures/`, drop `.json` in `tests/schemas/`, run.

---

## Configuration (`.env`)

```env
# Layer 3 extraction LLM
IDP_EXTRACTION_BACKEND=sarvam
IDP_SARVAM_API_KEY=sk_...
IDP_SARVAM_MODEL_NAME=sarvam-105b
IDP_SARVAM_BASE_URL=https://api.sarvam.ai/v1

# Ollama fallback (set IDP_EXTRACTION_BACKEND=ollama to use)
IDP_OLLAMA_BASE_URL=http://localhost:11434/v1
IDP_EXTRACTION_MODEL_NAME=qwen2.5:7b

# Layer 1+2 VLM
IDP_VLM_MODEL_NAME=qwen2-vl:2b

# Database
IDP_DATABASE_URL=postgresql://postgres:password@localhost:5432/idp
```

---

## How to Run

**Backend (Layer 1+2+3 API):**
```bash
cd backend
.venv\Scripts\activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Layer 3 dev test (no PDF, fixture only):**
```bash
cd backend
.venv\Scripts\activate
python run_extraction.py --doc sdg_goals_output --schema sdg_goals_schema
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Architectural Principles

1. **Layer 1+2 stays schema-agnostic.** No field extraction logic in Layer 1+2. All schema-aware code lives in Layer 3.

2. **Layer 3 sees only `.md` text.** It never touches the original PDF, never calls PyMuPDF, never knows which OCR engine ran.

3. **Dynamic schema at request time.** The API builds a Pydantic model from user-supplied field definitions at runtime. No schema changes require redeployment.

4. **Provenance is data, not logging.** Every extracted field knows which page it came from (`scratchpad.provenance()`). Logged in structured JSON after every page.

5. **DB failure is non-fatal.** `_save_to_db()` catches all exceptions. The API returns the extraction result regardless of DB state.

6. **Sarvam JSON fence stripping is mandatory.** `_strip_fences()` must run before every `json.loads()` on Sarvam responses. Omitting it causes silent data loss, not a crash.

7. **First-seen wins for field conflicts.** Once a field is in the scratchpad, it is removed from `fields_to_check` and never re-queried. No overwrite possible in normal flow.
