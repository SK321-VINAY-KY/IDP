# IDP System — Engineer A Pipeline (Layer 1 Routing + Layer 2 Conversion)

> **Engineer A responsibility**: Accept any PDF, detect what's on each page, pick the right extraction engine(s), and output structured Markdown with full provenance metadata. This is the **document-type-agnostic** conversion half of the IDP product. Field-level extraction (schema matching) lives in Engineer B's Layer 3 and operates exclusively on the `.md` output produced here.

---

## 🏗️ System Architecture

```
                              ┌───────────────────────────────────────────────────────┐
                              │                    IDP PRODUCT                        │
                              └───────────────────────────────────────────────────────┘
                                                         │
 ┌───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┐
 │                                                       │                                                       │
 │  ENGINEER A (THIS REPO)                               │                  ENGINEER B                           │
 │  Layer 1 Routing  +  Layer 2 Conversion               │                  Layer 3 Extraction                     │
 │  Produces:  structured .md  +  provenance metadata    │                  Consumes:  .md  +  cached schema       │
 │  Schema-agnostic  (works on ANY document type)        │                  Schema-aware  (resumes / invoices /   │
 │                                                       │                  receipts / forms / etc.)              │
 │                                                       │                                                       │
 │  PDF → inspect → route → engines → merge → .md        │                  .md → field extraction → JSON        │
 │        +  .schema_ref.json sidecar                    │                                                         │
 └───────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

### Engineer B Cache-Building Phase (Schema Discovery)

Before Engineer A batch-processes a corpus, Engineer B's schema cache is populated. This subsystem lives inside `schema_chatbot_v2/` and is served on port 8000. **It does not use Engineer A's pipeline** — it uploads 2–5 sample PDFs to Sarvam Document AI for OCR, then sends the merged text to Sarvam LLM to derive a shared schema. The result is persisted to `schema_registry/schema_<id>.json` as Engineer B's cache.

### Engineer A Data Flow (Per-Document Pipeline)

For every PDF processed:

```
 PDF
  │
  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step A  —  inspect_page()              [PyMuPDF heuristics only, ~10 ms, no GPU / no model] │
│   char_count, image_coverage, is_scanned, primary_script, complexity_score (0..5),          │
│   has_tables, has_vector_drawings, dpi_estimate                                             │
└─────────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                  │
                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step B  —  Route Resolution  (two modes controlled by settings.routing_mode)                │
│                                                                                              │
│   ◦  Mode:  "single_engine"  (DEFAULT, backwards-compatible)                                │
│      route_from_profile(profile) → one route string OR None (ambiguous → VLM Step)          │
│         ├─ definitive:  wrap into 1-task EnginePlan                                         │
│         └─ ambiguous:   llm_client.analyze_page(image_bytes) → VLMAnalysis                  │
│                         If VLM direct-extraction >= threshold, terminal result.              │
│                         Else capabilities_from_vlm_analysis() → build_engine_plan()          │
│                                                                                              │
│   ◦  Mode:  "capability_based"  (opt-in, Stage 1)                                           │
│      capabilities_from_profile(profile) → PageCapabilities SET (multiple booleans)          │
│      If ambiguous → llm_client.analyze_page() → enrich with VLM-derived caps                │
│      build_engine_plan(capabilities, hints) → list[EngineTask] (multi-engine)               │
└─────────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                  │
                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 3  —  Engine Plan Execution + Merge                                                    │
│   Engine priority (lower # runs first, wins tie):                                           │
│     1. docling                 digital embedded text (conf ~0.97)                           │
│     2. paddleocr_printed       scanned printed sheets  (PP-OCRv6 via PaddleX)               │
│     3. paddleocr_handwritten  handwritten fill-ins     (lowered det_db_thresh)              │
│     4. vlm_transcribe          escalation fallback (Ollama / Gemini)                        │
│                                                                                              │
│   Merge: line-level normalized dedup + char-count-weighted confidence avg                   │
└─────────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                  │
                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 4  —  Escalation Ladder  (if merged confidence < escalation_confidence_threshold)     │
│   digital  →  paddleocr_printed  →  paddleocr_handwritten  →  vlm_transcribe  →  GIVE UP   │
│   Hard cap max_escalation_attempts=1 prevents loops; terminal = low_confidence=True flag    │
└─────────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                  │
                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 5  —  Output Write                                                                     │
│   md_writer.write_document() → dataset_output/<stem>.md                                     │
│       • per-page HTML comments: engine, confidence, capabilities, latency_ms               │
│       • PIPELINE_SUMMARY JSON footer (parseable audit block)                               │
│   + sidecar <stem>.schema_ref.json linking document → schema_id from Engineer B cache      │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧰 Technology Stack, By Role

| Role / Feature | Library | Scope | Notes |
|----------------|---------|-------|-------|
| **PDF inspection** (Step A) | **PyMuDF (`pymupdf`)** | [inspect.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer1_routing/inspect.py) | `get_text("words")`, `get_image_info()`, `get_drawings()`, Unicode block script detection, page-area arithmetic, DPI estimation, complexity heuristics. ~10 ms / page, zero model calls. |
| **Page rendering** | **PyMuPDF + Pillow + NumPy** | [pipeline.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer1_routing/pipeline.py#L101-L143) | Render pages at 150 DPI → `np.ndarray` (RGB, H×W×3) for PaddleOCR; same pixmap encoded to PNG bytes for VLM calls. |
| **Digital text extraction** | **Docling 2.x** | [digital.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer2_conversion/digital.py#L15-L72) | Uses Docling's native page-range support. Outputs structured Markdown with headings, tables. Cold-start model load ~15–20 s first call; subsequent pages sub-second. Gracefully falls back to plain `pymupdf.get_text("text")` if Docling package missing on host. |
| **Scanned printed OCR** | **PaddleOCR (PP-OCRv6) via PaddleX** | [scanned.py `convert_scanned_page()`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer2_conversion/scanned.py#L101-L132) | `PP-OCRv6_medium_det` + `PP-OCRv6_medium_rec` detector + recognizer. Lazy-loaded singleton per mode (printed vs handwritten). Runs on CPU only. Confidence rolled up as weighted average of per-word `rec_scores`. |
| **Handwritten OCR** | **PaddleOCR (handwritten-tuned instance)** | [scanned.py `convert_handwritten_via_paddle()`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer2_conversion/scanned.py#L139-L176) | Same PaddleOCR class as printed, **separate singleton instance**. Tuned via `settings.paddle_handwriting_det_db_thresh` (default 0.20 vs printed's 0.30) to catch thinner strokes; `use_textline_orientation=True` for less predictable handwriting angles. PaddleOCR DBNet performs line-level segmentation internally — no external TrOCR-style line pre-splitting needed. |
| **VLM classification + transcription** (Step B ambiguous) | **Ollama (REST)** OR **Google Gemini (REST)** OR **MockLLMClient** | [adapters/llm/](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/adapters/llm/) | Abstract `LLMClient` ABC with 3 methods: `classify_page` (light), `analyze_page` (rich VLMAnalysis), `transcribe_handwriting` (escalation). Typed outputs via Pydantic. Controlled by `settings.llm_provider` (default `"ollama"`) + `settings.vlm_model_name`. Only invoked when Step A heuristics are genuinely ambiguous — digital pages with `char_count > 100` skip VLM entirely. |
| **VLM: Ollama adapter** | **httpx** + **`http://localhost:11434/v1/chat/completions`** | [ollama_client.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/adapters/llm/ollama_client.py) | OpenAI-compatible REST endpoint, base-64 inline image payloads. Model: default `qwen2.5vl:7b` or override via env. |
| **VLM: Gemini adapter** | **`google.genai` SDK** | [gemini_client.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/adapters/llm/gemini_client.py) | Uses `IDP_GEMINI_API_KEY` from root `.env`. Native Gemini structured-output + binary image inline upload support. |
| **Schema Discovery: Phase 1 NL chat** | **httpx** → Sarvam `/v1/chat/completions` | [schema_chatbot_v2/app/llm/sarvam_adapter.py `_chat()`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/llm/sarvam_adapter.py#L265-L296) | **No SDK used** — plain REST POST with `api-subscription-key` header. Constrained via `response_format: {type: json_schema}`. Two call sites: `extract()` (user NL → field ops), `infer_schema_from_pdfs()` step 2 (OCR texts → schema JSON). |
| **Schema Discovery: Phase 1 Doc AI OCR** | **`sarvamai` SDK (lazy import)** | [sarvam_adapter.py `_digitise()`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/llm/sarvam_adapter.py#L225-L249) + [doc_ai_client lazy property](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/llm/sarvam_adapter.py#L217-L223) | **Only place `sarvamai` SDK is used** — wraps the async job lifecycle: `doc_ai.digitise(file=[…])` → poll `get_status(job_id)` → `get_download_url()` → httpx GET of ZIP → parse out primary `.md` page content via `zipfile`. Reason for SDK: async poll loop + multipart upload naming + ZIP output convention would be ~30 lines of bespoke code to match what the SDK provides. SDK is lazily imported inside the property getter (not module top-level) so `sarvamai` stays optional for non-doc-intake usages. |
| **Schema Discovery: HTTP server** | **FastAPI 0.111 + Uvicorn** | [schema_chatbot_v2/app/](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/) | Three endpoints: `/health`, `/chat` (interactive REVIEW state machine), `/schema/infer` (multipart upload → async Sarvam Doc AI). Served on port 8000. |
| **All configuration** | **Pydantic Settings** (`pydantic-settings`) + **`python-dotenv`** | [config/settings.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/config/settings.py) + [schema_chatbot_v2/app/config.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/config.py) | Every threshold and feature flag tunable via env var (`IDP_` prefix for Engineer A, no prefix for chatbot). No recompilation needed between environments. |
| **Structured logging** | Custom JSON formatter + **`correlation_id`** context var | [utils/logger.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/utils/logger.py) | Every decision (inspect, route, engine start/complete, escalation attempt, merge, output write) emits one JSON line with `correlation_id`, `page_number`, structured fields. Written to `logs/pipeline.log` when a logger sink is configured. |
| **Engineer A ↔ Engineer B Contract types** | **Pydantic v1 (v2 in chatbot)** | [ai/schemas/page.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/schemas/page.py) | Typed data classes: `PageProfile`, `PageClassification`, `VLMAnalysis`, `PageCapabilities`, `EngineTask`, `PageOutput`. All I/O between modules uses these — any structural change needs bilateral sign-off. |
| **Multi-engine merge** | Pure Python (sets, normalized strings, weighted avg) | [pipeline.py `_merge_results_with_capabilities()`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer1_routing/pipeline.py#L207-L269) | Stage 1 line-level dedup: primary engine lines go in unconditionally, subsequent engine lines appended only if their normalized whitespace-collapsed lowercase form is unseen. Final confidence = weighted average by character contribution. Stage 2 (future): bbox-level spatial dedup when OCR engines expose geometry. |
| **Capability-based matcher** (Stage 1 routing mode) | Pure Python set operations | [capability_router.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer1_routing/capability_router.py) | `PROCESSOR_CAPABILITIES` map = {engine_name: set(requirements)}. Match engines whose requirements are a SUBSET of the detected `PageCapabilities`. Produces `(matched, unmatched, missing_reasons)` tuple used by `build_engine_plan()` to rank tasks. |
| **Markdown output writer** | Standard-library only (f-strings, JSON) | [output/md_writer.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/output/md_writer.py) | Pure stdlib — no markdown library. Concatenates provenance HTML comments, page Markdown bodies, then a `<!-- PIPELINE_SUMMARY [{…}, …] -->` JSON footer (parseable with regex by Engineer B without rendering Markdown). |
| **Structured LLM outputs (chatbot)** | **Instructor** (OpenAI wrapper) + **Pydantic** | [schema_chatbot_v2/app/core/validator.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/core/validator.py) | JSON-schema validation rules for field add/update/remove ops, type-checking `string/number/date/currency/enum/list[...]` types in schema. |
| **Conversation state (chatbot)** | in-memory dict (single-process dev) or Redis (pluggable) | [schema_chatbot_v2/app/core/conversation_manager.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/core/conversation_manager.py) + [schema_state.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/schema_chatbot_v2/app/core/schema_state.py) | REVIEW state machine with transitions `{COLLECT → INFER → REVIEW → CONFIRMED}`. Side effects only when user issues `/json` / `/confirm`. |
| **Testing (routing logic, no models)** | **pytest 8.x** | [tests/test_router_smoke.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/tests/test_router_smoke.py), [test_capability_router.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/tests/test_capability_router.py) | Pure-logic parameterized tests: every routing table branch, escalation ladder cap, Indic-script warning, blank page skip, mixed-content VLM trigger, capability matcher edge cases. Zero model downloads — runs in < 1 s. |

---

## 📂 Repository Layout: What Every File Does

```
engineer_a/
│
├── .env                          ← Engineer A env vars, IDP_ prefix. Currently only IDP_GEMINI_API_KEY
│                                   (activates Gemini adapter when settings.llm_provider="gemini").
│
├── requirements.txt              ← Engineer A pip deps: pydantic-settings, pymupdf, docling,
│                                   paddleocr, pillow, numpy, httpx, pydantic-v1, google-genai,
│                                   pytest 8.x. Install into .venv/.
│
├── .venv/                        ← Python 3.11.9 virtualenv. PRE-INSTALLED deps:
│     └── Scripts/python.exe         paddleocr 3.7.0, docling 2.x, fastapi, rapidocr, torch,
│                                      huggingface, google-genai, httpx, pymupdf, pydantic,
│                                      numpy, pillow, pytest.
│                                      ⚠️  Use this interpreter (NOT system Python 3.14).
│                                      System Python lacks paddleocr.
│
├── demo_resume_pipeline.py       ← Engineer A pipeline-only harness.
│                                   Wires: PDF → PyMuPDF render → process_document() → md_writer.
│                                   Uses MockLLMClient intentionally to surface any VLM dependency.
│                                   Intended for resume PDFs in tests/fixtures/; works on any corpus
│                                   that passes through Step A without VLM-classifiable pages.
│
├── demo_end_to_end.py            ← Unified harness. Two phases:
│                                   • PHASE 1: hit chatbot :8000 /health → POST /schema/infer with
│                                     N selected sample PDFs → REVIEW loop /json → persist to
│                                     schema_registry/.
│                                   • PHASE 2: iterate dataset/ → Engineer A process_document()
│                                     (MockLLMClient) → md_writer → dataset_output/ + sidecar
│                                     schema_ref.json linking each doc → schema_id.
│
├── dataset/                      ← Input PDFs. Engineer A batch phase consumes every *.pdf here.
│                                   Can mix digital PDFs (Docling handles) + raster-scanned image
│                                   PDFs (PaddleOCR handles). Mixing document TYPES here reduces
│                                   schema quality in Phase 1 inference — keep same-type in one run.
│
├── dataset_output/               ← Engineer A output sink.
│     ├── <stem>.md                  Per-PDF Markdown w/ per-page provenance HTML comments +
│     │                               PIPELINE_SUMMARY JSON audit footer.
│     └── <stem>.schema_ref.json     Sidecar: source_pdf, schema_id, schema_registry_file path,
│                                     document_type, per-page engine usage summary from Phase 2.
│                                     Parsed by Engineer B as the schema-id lookup handle.
│
├── schema_registry/              ← Engineer B cache of CONFIRMED schemas written by Phase 1.
│     └── schema_<hash>.json         JSON shape: schema_id, document_type, fields[] (Pydantic
│                                     field spec incl. name, type, required, item_type, pattern,
│                                     currency, description, seen_in_samples), provenance
│                                     (inference_source, created_at), sample_documents[], metadata.
│                                     Write-once confirmed. Engineer B reads this at startup.
│
├── schema_chatbot_v2/            ← Phase 1 Schema Discovery HTTP service (FastAPI, port 8000).
│     ├── .env                       LLM_PROVIDER=sarvam; SARVAM_API_KEY; SARVAM_MODEL= sarvam-105b;
│     │                               SARVAM_BASE_URL; SARVAM_DOC_AI_LANGUAGE / POLL / TIMEOUT.
│     ├── requirements.txt           fastapi, uvicorn[standard], pydantic-v2, httpx, python-dotenv,
│     │                               python-multipart (for /schema/infer upload), boto3 (bedrock),
│     │                               sarvamai>=0.1 (Doc AI digitise only).
│     ├── README.md                  Chatbot deployment docs.
│     └── app/
│          ├── main.py               FastAPI app factory: lifespan (init LLM adapter + state stores),
│          │                            routes include, /health probe.
│          ├── config.py             Chatbot-side pydantic-settings singleton w/ sarvam + session cfg.
│          ├── api/routes.py         HTTP endpoints:
│          │                            GET  /health
│          │                            POST /chat  (conversation_id, message, mode → turns + state)
│          │                            POST /schema/infer  (files: list[UploadFile] → schema JSON)
│          ├── models/api_models.py  Pydantic v2 schemas for request/response (ChatRequest,
│          │                            ChatTurn, SchemaInferenceResponse, etc.)
│          ├── llm/
│          │    ├── base.py          LLMAdapter ABC: extract(), phrase_question(),
│          │    │                      infer_schema_from_pdfs().
│          │    ├── factory.py       build_adapter(settings.llm_provider) factory dispatcher.
│          │    ├── prompts.py       System/user prompt templates for extraction, document
│          │    │                      inference, question phrasing fallback string.
│          │    ├── sarvam_adapter.py  ⭐ core Sarvam integration:
│          │    │                      • _chat() → httpx POST /chat/completions (NO SDK).
│          │    │                      • extract() → _chat() w/ ExtractionResult json_schema.
│          │    │                      • phrase_question() → _chat() to rephrase gap templates.
│          │    │                      • infer_schema_from_pdfs(): _digitise() each PDF then
│          │    │                        aggregate over _chat() w/ SchemaProposal json_schema.
│          │    │                      • doc_ai_client property: LAZY from sarvamai import
│          │    │                        SarvamAI — SDK used *only* for Doc AI digitise lifecycle.
│          │    │                      • _digitise(): doc_ai.digitise() → poll loop with
│          │    │                        settings.sarvam_doc_ai_poll_interval_s until timeout →
│          │    │                        get_download_url → httpx → zip bytes.
│          │    │                      • _extract_markdown(): zipfile → find first non-metadata
│          │    │                        .md → utf-8 decode.
│          │    ├── ollama_adapter.py   Local Ollama adapter. Mirrors sarvam_adapter endpoints.
│          │    └── bedrock_adapter.py  AWS Bedrock adapter placeholder for prod deploy.
│          └── core/
│               ├── conversation_manager.py  REVIEW state machine.
│               │      handle_message() dispatches by state:
│               │        COLLECT → /infer → INFER
│               │        INFER  → propose_schema() → REVIEW
│               │        REVIEW → extract() → {add, update, remove} ops; /json → persist, /confirm → CONFIRMED
│               │      Emits gaps on first review pass if any fields lack type/required.
│               ├── schema_state.py    In-memory CRUD for schema dict:
│               │      propose(), add_field(), update_field(), remove_field(), list_fields().
│               │      Field name uniqueness enforcement + type-system checks.
│               └── validator.py       JSON-schema style rules for field type validation,
│                                      list-of-X nested type expansions, enum/currency keys.
│
├── src/
│   ├── config/
│   │   └── settings.py             ⭐ All tunables. Pydantic BaseSettings with env_prefix="IDP_":
│   │                                    routing_mode ("single_engine" | "capability_based"),
│   │                                    digital_char_count_threshold (=100 shortcut),
│   │                                    scanned_char_count_threshold (=30),
│   │                                    escalation_confidence_threshold (=0.70),
│   │                                    max_escalation_attempts (=1),
│   │                                    vlm_direct_extraction_confidence_threshold (=0.85),
│   │                                    capability_low_confidence_floor (=0.50),
│   │                                    llm_provider ("ollama"),
│   │                                    ollama_base_url, vlm_model_name,
│   │                                    render_dpi (=150),
│   │                                    paddle_printed_det_db_thresh (=0.30),
│   │                                    paddle_handwriting_det_db_thresh (=0.20),
│   │                                    vlm_classify_thresholds (scan_handwriting_midband…),
│   │                                    dead_zone (handwriting_pct 0.10-0.30 gap fix),
│   │                                    output_dir default.
│   │
│   ├── utils/
│   │   └── logger.py               setup_logger(service), json formatter, ContextVar
│   │                                   correlation_id used by every JSON log line.
│   │
│   ├── adapters/
│   │   └── llm/
│   │        ├── base.py            LLMClient ABC:
│   │        │                         classify_page(image_bytes, hint) → PageClassification
│   │        │                         analyze_page(image_bytes, hint) → VLMAnalysis
│   │        │                         transcribe_handwriting(image_bytes) → str md
│   │        ├── factory.py         get_llm_client(settings.llm_provider, **overrides) dispatcher
│   │        ├── ollama_client.py   Ollama REST via httpx localhost:11434. Base64 PNG inline.
│   │        ├── gemini_client.py   google.genai SDK, genai.configure(api_key). PIL + bytes ingest.
│   │        └── schema_models.py   Pydantic v1 models for typed VLM returns: PageClassification,
│   │                                   VLMAnalysis fields incl. has_printed_text_pct,
│   │                                   has_handwriting_pct, has_tables, has_diagrams, script,
│   │                                   layout_quality_score, digital_text_hint,
│   │                                   direct_markdown_extract + confidence.
│   │
│   └── ai/
│        ├── schemas/
│        │   ├── page.py            ⭐ CROSS-TEAM CONTRACT — all types shared between modules:
│        │   │                           PageProfile: Step A output (page_number, has_text,
│        │   │                             char_count, word_count, image_count, image_coverage,
│        │   │                             is_scanned, primary_script, has_tables,
│        │   │                             has_vector_drawings, complexity_score, dpi_estimate,
│        │   │                             raw_flags).
│        │   │                           PageClassification: VLM Step B light return.
│        │   │                           VLMAnalysis: VLM Step B full return.
│        │   │                           PageCapabilities: Stage 1 Set-of-booleans capacity spec.
│        │   │                           EngineTask: (engine, priority, mode, image_bytes?,
│        │   │                             pdf_path?, page_number?, context?).
│        │   │                           PageOutput: FINAL contract w/ Engineer B:
│        │   │                             page_number, markdown, engines_used, confidence,
│        │   │                             capabilities (List[str]), escalated, low_confidence.
│        │   └── page_metadata.py   Internal: EngineResult, RoutingDecision, EscalationRecord,
│        │                                PageMetadata. Full audit trail; not part of B contract.
│        │
│        ├── layer1_routing/        ← ⭐ Step A + Step B + top-level orchestration.
│        │   ├── inspect.py
│        │   │       inspect_page(page, page_number) -> PageProfile   (Step A — all pages)
│        │   │         _count_chars_words(): pymupdf get_text("words")
│        │   │         _analyze_images(): get_image_info() → bbox area frac → image_coverage
│        │   │         _detect_tables(): get_drawings() horizontal/vertical line geometry stub
│        │   │         _detect_scanned(): char_count < 30 AND image_coverage > 0.25
│        │   │         _detect_script(): unicodedata.category + block ranges per char
│        │   │         _compute_complexity(): tables(+2) + vectors(+1) + sparse(+1) + partial_img(+1)
│        │   │         _estimate_dpi(): widest pixel width / page inches
│        │   │
│        │   ├── router.py
│        │   │       route_from_profile()      → "digital" | "scanned" | "skip" | None
│        │   │         decision tree: blank → skip; Indic → scanned (warn); has_text+chars>100 →
│        │   │           (mixed?→VLM); complexity>=4→VLM; else→VLM
│        │   │       route_from_classification() → "digital" | "scanned" | "handwritten" | "vlm"
│        │   │       capabilities_from_profile() → PageCapabilities Set
│        │   │         {has_digital_text, has_printed_scan, has_handwriting, has_tables,
│        │   │          has_figures, has_indic_script, is_blank, has_mixed_content}
│        │   │         + dead_zone fix for handwriting_pct 0.10–0.30 (biases conservative)
│        │   │       capabilities_from_classification()  → PageCapabilities from VLMAnalysis
│        │   │       build_engine_plan(capabilities, hints, route) → list[EngineTask]
│        │   │         single_engine: 1 task
│        │   │         multi_engine:  matched engines by priority, extras=skip for is_blank
│        │   │       route_and_build_plan(profile, vlm_client, image_bytes) → (tasks, routing_decision)
│        │   │         wraps the Step B two-path above: definitive or VLM-enriched
│        │   │       pick_escalation_engine(failed, current_caps) → Optional[str]
│        │   │         maps current capability → next rung on escalation ladder
│        │   │
│        │   ├── capability_router.py
│        │   │       PROCESSOR_CAPABILITIES dict literal
│        │   │       match_engines(caps) → (matched list[(engine, prio)], unmatched, missing_reasons)
│        │   │         pure set subset check per engine's requirements vs PageCapabilities
│        │   │
│        │   └── pipeline.py         ⭐ TOP-LEVEL ORCHESTRATION.
│        │       build_page_contexts(pdf_path) → list[render ctx dicts] (pymupdf render, np arr, PNG bytes)
│        │       _run_engine_task(task, llm_client, ctx) → EngineResult
│        │         dispatch table: docling→convert_digital_page(), paddleocr_printed→…, etc.
│        │       _merge_results_with_capabilities(results, tasks) → (merged_md, conf, engines_used)
│        │         Stage 1 line-dedup + weighted confidence (documented in merge section)
│        │       _apply_escalation_ladder(page_number, initial_res, initial_conf, tasks, ctx, caps)
│        │         while conf<threshold AND rungs left AND attempts<cap → _run next rung → merge
│        │         returns (final_res, final_conf, final_engines, escalated_flag, attempts)
│        │       process_page(page, page_number, ctx, llm_client, corr_id) → (PageOutput, PageMetadata)
│        │         THE CORE FUNCTION:  inspect → route_and_build_plan → run tasks → merge →
│        │         escalate → build types + return
│        │       process_document(pages, llm_client, doc_name, doc_id, write_output,
│        │                           output_dir, overwrite) -> list[(PageOutput, PageMetadata)]
│        │         iterates pages, calls process_page, logs document_complete,
│        │         optionally md_writer.write_document().
│        │
│        ├── layer2_conversion/     ← Step 5 — actual extraction engines.
│        │   ├── digital.py
│        │   │       _DoclingStore._get_instance()   singleton lazy DoclingConverter()
│        │   │       convert_digital_page(pdf_path, page_number) → (md:str, conf:float)
│        │   │         Docling pipeline.convert(path, pages=[page_n-1]) → dljson → _to_markdown()
│        │   │         except ImportError: pymupdf fallback get_text("text")
│        │   │       _to_markdown(doc, page_num) → iterate doc.pages[0].content → H/H2/P/TABLE MD
│        │   │
│        │   └── scanned.py
│        │           _paddle_engines: dict[str, PaddleOCR]   singleton cache per mode
│        │           _get_paddle_engine(mode="printed"|"handwritten") → PaddleOCR instance
│        │             settings thresholds applied per-instance (det_db_thresh, textline_orientation)
│        │           _paddle_result_to_markdown(lines, page_number) → md
│        │             lines sorted by y-then-x → ## heading heuristic → bullet heuristic → "## Page N"
│        │           convert_scanned_page(image_array) → (md, conf)
│        │             printed instance.ocr() → full result rollup
│        │           convert_handwritten_via_paddle(image_array) → (md, conf)
│        │             handwritten instance.ocr() → full result rollup
│        │
│        └── output/
│            └── md_writer.py
│                  write_document(page_outputs_metadata, pdf_path, out_dir, overwrite)
│                    _build_page_header(page_n, engines, caps, conf, latency) → HTML comment
│                    _build_summary_json(page_metadatas) → <!-- PIPELINE_SUMMARY [JSON] -->
│                    write_file → dataset_output/<stem>.md
│                  write_schema_sidecar(pdf_path, md_path, schema_doc, out_dir)
│                    writes <stem>.schema_ref.json with schema_id + registry path + doc_type + page summaries
│
├── tests/
│   ├── fixtures/                    PDFs for routing+extraction tests:
│   │     ├── MY_resume.pdf          (2294 chars, digital → Docling)
│   │     ├── Vinay_Resume.pdf       (2312 chars, digital → Docling)
│   │     ├── camscanner_handwritten.pdf (0 chars, scanned image + handwritten annotations)
│   │     └── sdg_goals.pdf          (digital with tables; complexity>1 may trigger VLM paths)
│   ├── test_router_smoke.py         Core smoke tests (parameterized): blank/skip, pure digital,
│   │                                 pure scanned, mixed content (must return None VLM trigger),
│   │                                 Indic script (→ scanned with warning), low chars not blank,
│   │                                 escalation rung progression, hard cap at MAX_ATTEMPTS.
│   ├── test_router_edgecases.py     Additional boundary: empty profile, all-zero img, all-1 img,
│   │                                 complexity=4+ (VLM trigger), high chars still flagged mixed,
│   │                                 route_from_classification branches for printed vs handwritten.
│   ├── test_capability_router.py    PROCESSOR_CAPABILITIES unit tests. pure set operations.
│   ├── test_vlm_routing.py          Hooks for real LLMClient.analyze_page() wired for
│   │                                 integration runs only (skipped in default smoke env).
│   ├── test_pdf_extraction.py       Real Docling/PaddleOCR integration harness (fixture-based).
│   └── test_handwritten_extraction.py  PaddleOCR handwritten mode route only (escalation path from
│                                     scanned ladder).
│
└── logs/  (created at runtime)   pipeline.log sink when logger configured with file handler.
```

---

## 🔗 Engineer A → Engineer B Contract

Everything Engineer B consumes from this repo is one of two artifacts:

### 1. `PageOutput` (in-memory type — via Python call)
Defined in [ai/schemas/page.py#L112-L129](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/schemas/page.py#L112-L129):
```python
class PageOutput(BaseModel):
    page_number: int
    markdown: str                      # Cleaned markdown body ready for B's regex/LLM extraction
    engines_used: List[str]            # ["docling"]  or ["paddleocr_printed", "paddleocr_handwritten"]
    confidence: float                  # 0.0–1.0  merged weight
    capabilities: List[str]            # Subset of {"has_digital_text","has_printed_scan", …}
    escalated: bool                    # True if any rung on escalation ladder was climbed
    low_confidence: bool               # True if final confidence < settings.capability_low_confidence_floor
```

### 2. `.md` file + sidecar (disk artifacts — Engineer B preferred)

`.md` shape in [output/md_writer.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/output/md_writer.py):
```
# <file>.pdf
<!-- IDP Pipeline Output  [metadata block: Generated, Pages, Engines, Avg conf, Escalated, Low-conf] -->
---
<!-- PAGE N | engine=… | conf=… | caps=[…] | latency=…ms -->
<markdown content of page N>
<!-- /PAGE N -->
---
<!-- PAGE N+1 | engine=… | conf=… | caps=[…] | latency=…ms -->
...
---
<!-- PIPELINE_SUMMARY [JSON per-page array] -->
```

Engineer B **must parse only between PAGE open/close HTML comments** for page-level provenance reuse; `PIPELINE_SUMMARY` block is a fast pre-scan audit (JSON array) without a Markdown parser.

Sidecar `<stem>.schema_ref.json` — document-level schema link:
```json
{
  "source_pdf": "MY_resume.pdf",
  "output_md": "MY_resume.md",
  "schema_id": "schema_d3437ac77d3c",
  "schema_registry_file": "schema_d3437ac77d3c.json",
  "document_type": "resume",
  "pages": [ {"page_number": 1, "engines_used": ["docling"],
              "confidence": 0.970, "capabilities": ["has_digital_text"],
              "escalated": false, "low_confidence": false, "chars": 2750, "latency_ms": 17897.3} ]
}
```

---

## 🔌 VLM Provider Wiring (Engineer A Step B)

Switch via `settings.llm_provider` → dispatched in [adapters/llm/factory.py#get_llm_client()](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/adapters/llm/factory.py):

| Provider | `llm_provider` value | Package | How images sent |
|----------|----------------------|---------|-----------------|
| Ollama (local dev default) | `"ollama"` | httpx only, no SDK | base64 PNG within `/chat/completions` content array |
| Google Gemini | `"gemini"` | `google-genai` SDK (pip) | PIL Image or bytes directly to `GenerativeModel.generate_content()` |
| Mock (for deterministic tests / demo harnesses that don't want network) | — | none (MockLLMClient subclass throws RuntimeError on every method) — intentional: makes VLM dependency LOUD if any page reaches it without a real provider configured |

Sarvam is **not** used as Engineer A's VLM — only for Phase 1 schema discovery chatbot, which is a separate service.

---

## 🧠 Core Routing Algorithms

### Step B Dead-Zone Fix
For pages where the handwritten/printed split is genuinely uncertain (VLMAnalysis handwriting_pct between 0.10 and 0.30), the original routing table has an ambiguous gap: neither engine wins confidently enough. `capabilities_from_classification()` in [router.py](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer1_routing/router.py) resolves this conservatively: **has_handwriting=True AND has_printed_scan=True** → engine plan runs BOTH, escalates if merged output is still poor. This is the dead-zone fix referenced in legacy tests.

### Multi-Engine Merge Priority Tiebreak
In `capability_based` mode, multiple engines can run on one page. The order is set by the `priority` int on each `EngineTask`:
- Docling → priority 1 (highest, wins all baseline lines)
- paddleocr_printed → priority 2
- paddleocr_handwritten → priority 3
- vlm_transcribe → priority 4

The merge function then **only appends new lines** from lower-priority engines that weren't already in higher-priority output, which prevents the classic "printed+handwritten overlap duplicates text twice" bug.

### Mixed Content VLM Trigger (single_engine mode)
`char_count > 100` is the "shortcut pure digital" fast path — no VLM cost. But if ALSO `0.02 < image_coverage < 0.85`, that means embedded figures exist alongside text (not a full-page scan), which can confuse Docling's layout model. In that specific case, `route_from_profile()` returns `None` → Step B MUST invoke VLM.analyze_page(). This prevents false "all digital" routing on brochures and spec sheets where Docling silently drops figure-callout content.

---

## ⚖️ Architectural Principles (Non-Negotiable)

1. **Engineer A stays schema-agnostic**. If code in this repo ever applies `field.extract()` using a schema from Engineer B, the responsibility boundary has been broken. All field-level logic goes in Engineer B's repo against the `.md` outputs only.
2. **Provenance is replayable data, not logging**. Page HTML comments MUST contain exact `engine=` + `conf=` + `caps=` + `latency=` values sufficient for Engineer B's downstream debug re-runs WITHOUT re-invoking PyMuPDF/OCR/Docling.
3. **Capability Set > one label**. A "printed form with handwritten fill-ins" is not `scanned` OR `handwritten`. It is both. Use `capability_based` routing mode + multi-engine merge + dedup instead of forcing a one-route pick.
4. **Thou shalt not make VLM the default path**. Every page > 100 chars embedded text MUST route Docling directly per the shortcut. VLM is expensive and slow. Use it only for genuinely ambiguous cases where Step A heuristics actually produce `None` / ambiguous capability flags.
5. **Escalation is a ladder, not a restart**. Each rung is a superset engine that handles the failure mode of the last. Re-running the same engine with a different seed is never an escalation step.
6. **Singletons for heavy engines**. Both Docling (in [digital.py `_DoclingStore`](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer2_conversion/digital.py#L28-L37)) and PaddleOCR (in [scanned.py `_paddle_engines` dict](file:///C:/Users/Dell/Desktop/IDP/engineer_a/src/ai/layer2_conversion/scanned.py#L20-L23)) cache at the module top level per mode. Instantiating a fresh PaddleX model for every page = 30 s/page cold start. With singleton reuse, subsequent pages < 3 s scanned.
