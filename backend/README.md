# IDP Backend

Intelligent Document Processing — backend pipeline.

**Layer 1 + Layer 2** (Engineer A): Accept a PDF, detect what's on each page, pick the right OCR engine, output Markdown text with provenance metadata.

**Layer 3** (Engineer B): Take the Markdown text, scan each page against a user-defined target schema, extract field values using Sarvam AI, return structured JSON.

**API + Storage** (Engineer B): FastAPI HTTP layer, dynamic schema building, PostgreSQL persistence.

---

## Quick Start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Start the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Dev test (no PDF needed)
python run_extraction.py 
```

---

## Project Structure

```
backend/
├── src/
│   ├── adapters/llm/          LLM clients (Ollama VLM + Sarvam extraction)
│   ├── ai/
│   │   ├── layer1_routing/    Page inspection + route decision
│   │   ├── layer2_conversion/ OCR engines (Docling, PaddleOCR)
│   │   ├── layer3_extraction/ Field extraction from Markdown
│   │   └── schemas/           Shared Pydantic contracts
│   ├── api/                   FastAPI app + document processor
│   └── config/settings.py     All settings (env var overridable)
├── tests/
│   ├── fixtures/              Saved Layer 2 .md outputs for offline testing
│   └── schemas/               Target schema JSON files
├── run_extraction.py          Dev runner (no PDF needed)
└── generate_fixture.py        Generate .md fixture from a real PDF
```

---

## Layer 3 — Field Extraction

### What It Does

Layer 3 takes the Markdown text from Layer 1+2 and extracts specific fields from it. The user defines which fields they want (e.g. `invoice_number`, `vendor_name`, `total_amount`). Layer 3 finds those values in the text and returns them as JSON.

---

### Evolution of the Approach

#### Phase 1 — Strategy A and B (Deleted)

**Strategy A (short docs, < 10 pages):** Concatenate all pages → one LLM call → extract everything at once.
*Problem:* Inconsistent. LLM sees too much and gets confused on documents with repetitive structure.

**Strategy B (long docs, ≥ 10 pages):** Summarise each page → ask LLM which pages have which fields → extract from relevant pages only.
*Problem:* Navigation step made mistakes. Still sent multiple pages together. Two extra LLM calls before extraction even started.

Both were deleted. Files removed: `router.py`, `strategy_short.py`, `strategy_long_pageindex.py`.


---

#### Phase 2 — Current: Page-by-Page Scan with Scratchpad

**Core idea:** Check each page individually. Accumulate results in a scratchpad. Stop when all fields are found.

**How it works:**

1. **Cache the schema once.** Build `[{name, description}]` from the target schema before the page loop starts.

2. **Loop through every page:**
   - Find fields still missing from scratchpad
   - If all found → stop early
   - Send page text + missing fields to Sarvam AI (with page number and total pages in the prompt)
   - Sarvam returns what it found on this page
   - Store in scratchpad — first-seen wins, already-found fields are never re-asked
3. **Build final result.** Overlay scratchpad values on schema defaults. Validate. Return JSON.

---

### Layer 3 Files

| File | What it does |
|---|---|
| `layer3_extraction/extractor.py` | Main function `extract_by_page_scan()`. Page loop, early exit, final model. |
| `layer3_extraction/scratchpad.py` | In-memory accumulator. First-seen wins. Hallucination guard. |
| `layer3_extraction/page_loader.py` | Converts `PageOutput` list to `list[{markdown, page_number}]`. Reads fixture files. |
| `layer3_extraction/schema_validation.py` | Retry wrapper — retries up to 2 times on validation failure. |
| `layer3_extraction/storage.py` | Saves extraction run to PostgreSQL. |
| `prompts/templates/page_field_check.j2` | Per-page scan prompt. Page N of M, field list, page content. |
| `prompts/templates/extraction_system.j2` | System prompt for whole-document extraction calls. |
| `prompts/configs/prompt_config.yaml` | Maps prompt names to templates and LLM parameters. |
| `adapters/llm/extraction_base.py` | Protocol interface for extraction LLM clients. |
| `adapters/llm/sarvam_client.py` | Sarvam AI client. Contains `_strip_fences()`. |
| `adapters/llm/extraction_client.py` | Ollama fallback client. |
| `adapters/llm/extraction_factory.py` | Returns right client based on `IDP_EXTRACTION_BACKEND`. |

---

## API

### `POST /api/extract`

**Request:** `multipart/form-data`
- `file` — PDF file
- `target_schema` — JSON string: `[{"name": "field_name", "description": "what to look for"}]`

**Response:**
```json
{
  "success": true,
  "data": {
    "document_title": "5th SDG Youth Summer Camp...",
    "goal_1_title": "End poverty in all its forms everywhere"
  },
  "meta": {
    "strategy_used": "page_scan",
    "processing_time_seconds": 18.4,
    "llm_provider": "sarvam",
    "saved_to_db": true,
    "db_result_id": 63
  }
}
```

### `GET /api/health`
Returns `{"status": "ok"}`.

---

## API Files

| File | What it does |
|---|---|
| `src/api/main.py` | FastAPI app. `/api/extract` orchestrates the full pipeline. |
| `src/api/dynamic_schema.py` | Builds a Pydantic model at runtime from user-supplied fields. No hardcoded field names. |
| `src/api/document_processor.py` | Opens PDF, renders pages at 200 DPI, packages page dicts for Layer 1+2. |

---

## PostgreSQL

### Table: `extraction_runs`

One row per extraction request.

| Column | What it stores |
|---|---|
| `id` | Auto-generated ID |
| `doc_id` | PDF filename |
| `page_count` | Total pages |
| `schema_name` | Field keys used (comma-joined) |
| `result_json` | Extracted field values (JSONB) |
| `page_details` | Per-page engine, confidence, escalated, capabilities (JSONB) |
| `llm_provider` | `"sarvam"` or `"ollama"` |
| `model_name` | `"sarvam-105b"` or `"qwen2.5:7b"` |
| `processing_time_seconds` | Wall-clock time |
| `created_at` | Timestamp |

**`page_details` shape:**
```json
{
  "1":  {"engines_used": ["paddleocr_printed"], "confidence": 0.99, "escalated": false, "low_confidence": false, "capabilities": ["has_digital_text"]},
  "17": {"engines_used": ["paddleocr_printed"], "confidence": 0.99, "escalated": false, "low_confidence": false, "capabilities": ["has_digital_text"]}
}
```

Null for fixture runs (no real `PageOutput` objects).

---

## Dev & Test Tools

### `run_extraction.py` — Dev runner (no PDF needed)

```bash
python run_extraction.py                                        
```

Reads `tests/fixtures/<doc>.md` + `tests/schemas/<schema>.json`. Runs Layer 3 only. Saves result to DB.

### `generate_fixture.py` — Generate fixture from real PDF

```bash
python generate_fixture.py --pdf tests/fixtures/my_doc.pdf
# → saves tests/fixtures/my_doc_output.md
```

Runs the PDF through the full Layer 1+2 pipeline and saves the Markdown output as a fixture file. Run once, then use `run_extraction.py` for all subsequent testing.

### `tests/fixtures/`

Saved Layer 2 outputs in `.md` format. Each page has a comment header:
```
<!-- Page 1 | engines=paddleocr_printed | confidence=0.9911 | escalated=false | ... -->
```

### `tests/schemas/`

JSON schema files — list of `{name, description}` objects. Swap files to test different extraction tasks with no code changes.

---

## Configuration

All settings use `IDP_` prefix in `.env`:

```env
# Layer 3 extraction LLM
IDP_EXTRACTION_BACKEND=sarvam       # "sarvam" or "ollama"
IDP_SARVAM_API_KEY=sk_...
IDP_SARVAM_MODEL_NAME=sarvam-105b
IDP_SARVAM_BASE_URL=https://api.sarvam.ai/v1
IDP_EXTRACTION_MODEL_NAME=qwen2.5:7b   # Ollama fallback

# Layer 1+2 VLM
IDP_LLM_PROVIDER=ollama
IDP_VLM_MODEL_NAME=qwen2-vl:2b

# Database
IDP_DATABASE_URL=postgresql://postgres:password@localhost:5432/idp
```

---

