# Engineer A — Pipeline (Layer 1 + Layer 2)

Implements page inspection, routing, the escalation ladder, and the three
Layer 2 conversion engines (digital/scanned/handwritten), per
`IDP_Pilot_Build_Guide.md` and `IDP_Pilot_Split_Plan.md`.

## What's here

```
src/
  config/settings.py              # Pydantic settings, all thresholds tunable via .env
  utils/logger.py                 # Structured JSON logger + correlation ID (framework pattern)
  ai/schemas/page.py               # THE CONTRACT — PageProfile, PageClassification, PageOutput
  ai/layer1_routing/
    inspect.py                     # Step A: PyMuPDF programmatic inspection
    router.py                      # Routing table, dead-zone fix, mixed-content detection, escalation ladder
    pipeline.py                    # Top-level orchestration: inspect -> route -> convert -> escalate -> PageOutput
  ai/layer2_conversion/
    digital.py                     # Docling
    scanned.py                     # PaddleOCR v6 (CPU)
    handwritten.py                 # TrOCR (CPU) — top tier of the escalation ladder
  adapters/llm/
    base.py                        # Abstract LLMClient — swap target for Bedrock later
    ollama_client.py                # Local Ollama (Qwen2-VL-2B) implementation
tests/
  test_router_smoke.py            # Passing smoke tests for routing/escalation logic (no model calls)
```

## What's implemented vs. what's a known pilot-scope limitation

**Fully implemented and tested:**
- Routing table (digital/scanned/skip via Step A, VLM fallback via Step B)
- The handwriting_pct dead-zone fix (0.1–0.3 gap now resolves conservatively to `handwritten`)
- Mixed-content page detection (forces VLM classification instead of trusting a pure-digital route)
- Escalation ladder with hard cap (`MAX_ESCALATION_ATTEMPTS`), verified to terminate rather than loop
- Structured logging with correlation ID at every routing/escalation decision

**Implemented but with a noted limitation (see inline comments in code):**
- `handwritten.py` — TrOCR is a line-level recognizer; the pilot passes the whole
  page as one region unless a caller supplies pre-split line regions. Multi-line
  handwritten pages will read as one blob until a layout-splitting step is added.
- `_detect_tables()` in `inspect.py` — the horizontal/vertical line geometry check
  is a placeholder; real bbox-angle geometry logic still needs to be filled in.
- Indic script pages currently route to `scanned` as a placeholder (IndicPhotoOCR
  is out of pilot scope per the build guide) — logged as a warning so it's visible
  in the audit trail rather than silent, not left undefined.

## How this hands off to Engineer B

Call `process_document(pages, llm_client)` — it returns `list[PageOutput]`.
Engineer B's Layer 3 depends only on that return type, never on anything inside
this package. Until this pipeline is wired to real PDF input, B should build
against the fixture set described in `IDP_Pilot_Split_Plan.md` §3.

## Setup and Running

### Dependencies

Install all dependencies:
```bash
pip install pydantic==1.10.13 tenacity --break-system-packages
python3 tests/test_router_smoke.py
```

These test the routing/escalation *logic* without requiring PaddleOCR/TrOCR/
Docling model downloads. Full integration tests (real OCR calls) need the
Docker Compose stack from the build guide running, with models pulled via Ollama.



## Engineer B — Contextual Search & Extraction (Layer 3)
 
Consumes finished per-page Markdown (produced by Layer 1 + Layer 2) and
extracts structured fields as validated JSON. Supports both:
- **Ollama** (local, open-source Qwen2.5 — no API keys, fully offline)
- **Sarvam** (API-based backend, configured via `IDP_EXTRACTION_BACKEND` and `IDP_SARVAM_API_KEY`)

This is a **pilot** — the core extraction logic is built and tested
against real Layer 2 output. Currently **working and tested**.
 
## What it does
 
Per the original system design (Section 5), once all pages are uniform
Markdown, extraction doesn't care which OCR engine produced them. Two
strategies, chosen automatically by page count:
 
- **Strategy A — Direct extraction (< 10 pages):** concatenate all
  pages into one block, extract the schema in a single request.

- **Strategy B — PageIndex navigation (≥ 10 pages):** 3-pass approach —
  (1) summarize each page cheaply, (2) ask the model which pages are
  relevant to which schema fields, (3) extract only from that narrowed
  page set. Avoids dumping very long documents into a single prompt and
  improves accuracy by keeping the final extraction call focused.
  (The original design also names a second long-document option, RLM —
  recursive hierarchical summarization. Not implemented in this pilot;
  PageIndex was built and validated first.)

## Structure
 
```
backend/src/
├── adapters/llm/
│   ├── extraction_base.py       # ExtractionLLMClient interface (Layer 3's own —
│   │                             #   distinct from adapters/llm/base.py, which is
│   │                             #   Layer 1/2's vision-classification interface)
│   ├── extraction_client.py     # Ollama implementation (Qwen2.5)
│   └── extraction_factory.py    # returns the configured client
├── ai/
│   ├── layer3_extraction/
│   │   ├── page_loader.py       # see "Integration with Layer 1/2" below
│   │   ├── router.py            # picks Strategy A vs B by page count
│   │   ├── strategy_short.py
│   │   ├── strategy_long_pageindex.py
│   │   ├── schema_validation.py # retry/repair loop on validation failure
│   │   └── prompts/
│   │       ├── templates/       # extraction_system.j2, page_summary.j2, navigation.j2
│   │       ├── versions/v1.0/   # frozen snapshot; versions/latest.txt points to it
│   │       ├── configs/prompt_config.yaml  # temperature/max_tokens per prompt
│   │       └── loader.py        # renders templates via Jinja2
│   └── schemas/
│       └── extraction_schema.py # what fields to extract — edit per document type
└── config/settings.py            # extraction_model_name, short_doc_page_limit, etc.
```
 
## Design notes
 
- **Extraction logic never talks to a specific model directly.**
  `router.py` and both strategies depend only on `ExtractionLLMClient`
  (an interface, `extraction_base.py`). Swapping the underlying model
  or provider is contained entirely to `extraction_client.py` +
  `extraction_factory.py` — nothing else changes.
- **Two models, split by task**, to reduce runtime on CPU-only
  hardware: `extraction_model_name` (7B, e.g. `qwen2.5:latest`) handles
  `extract()` and `navigate()`, where accuracy matters most.
  `summary_model_name` (3B, e.g. `qwen2.5:3b-instruct`) handles
  `summarize_page()`, which runs once per page and doesn't need the
  larger model's reasoning power.
- **Per-call context window (`num_ctx`)**, passed via `extra_body` in
  `extraction_client.py`, sized to what each call actually needs:
  smaller for single-page summaries, larger for the final extraction
  call (which sees the full concatenated relevant-page content).
  Undersized context windows were the confirmed/suspected cause of an
  earlier null-output bug — Ollama's small default silently truncates
  long inputs rather than erroring.
- **Prompts are centralized** in `.j2` template files, not inline
  strings, and versioned. `navigation.j2` was tightened to explicitly
  require integer page numbers (fixes a bug where the model returned
  `"1"` instead of `1`, breaking downstream page-list arithmetic).
  `extraction_system.j2` was tightened to distinguish literal vs.
  reasonably-inferred field values and to require JSON-only output.
- 

## Integration with Layer 1/2
 
`page_loader.py` :

- **`load_pages_from_fixture(doc_id: str)`** — Reads a
  saved `.md` fixture (e.g. `tests/fixtures/sdg_goals_output.md`) with
  `<!-- Page N | ... -->` markers, for manual testing without running
  the full Layer 1/2 pipeline.

 
```
 
## Config
 
Add to `.env` (never committed — see `.env.example` for the template):
```
# LLM Models
IDP_EXTRACTION_MODEL_NAME=qwen2.5:7b
IDP_SUMMARY_MODEL_NAME=qwen2.5:3b-instruct
IDP_SHORT_DOC_PAGE_LIMIT=10

# Backend selection
IDP_EXTRACTION_BACKEND=sarvam  # or "ollama"
IDP_SARVAM_API_KEY=your_key_here

# Database
IDP_DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/idp

# Ollama (if using local backend)
IDP_OLLAMA_BASE_URL=http://localhost:11434/v1
```

Reuses `IDP_OLLAMA_BASE_URL`, already defined for Layer 1/2's vision
client — same local Ollama instance, different models pulled per
request depending on the call.
 
## Running it (fixture-based, tested 2026-08-25)
 
```bash
# Pull Ollama models if using local backend
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b-instruct

# Install dependencies
pip install -r requirements.txt --break-system-packages
 
cd backend
python run_extraction.py
```

**Recent fixes (2026-08-25):**
- ✅ Removed Groq configuration (unused, was causing Pydantic validation errors)
- ✅ Added `extraction_backend` field to Settings (supports "ollama" or "sarvam")
- ✅ Updated all dependencies in requirements.txt (Pydantic v2, SQLAlchemy, psycopg2, etc.)
- ✅ **Tested with Sarvam backend** — extracts document fields and saves to PostgreSQL
- ✅ Database storage working (DocumentExtraction schema saves successfully)

`run_extraction.py` currently points at a hardcoded fixture via
`load_pages_from_fixture("sdg_goals_output")` — swap the argument to
target `camscanner_output` or any other saved fixture, or extend the
script into a real CLI once live Layer 2 wiring is in place.
 
## What to change for a new document type
 
Only `backend/src/ai/schemas/extraction_schema.py` needs editing —
define the fields you want as a `pydantic.BaseModel` (nested models and
lists are supported), then pass it into `route_and_extract(...)`
instead of the current schema. Nothing else in `layer3_extraction/`
needs to change; the router and both strategies are generic over
`schema: type[BaseModel]` throughout.
