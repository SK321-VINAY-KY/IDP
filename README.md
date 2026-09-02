# Intelligent Document Processing (IDP) System — End-to-End Platform

> An enterprise-grade Intelligent Document Processing (IDP) platform featuring multi-engine document conversion, capability-based routing, interactive AI schema discovery, and relational extraction storage.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Architectural Details & Key Mechanisms](#-architectural-details--key-mechanisms)
3. [Cloning & Environment Setup](#-cloning--environment-setup)
4. [API Keys & Configuration Guide](#-api-keys--configuration-guide)
5. [Execution & Run Instructions](#-execution--run-instructions)
6. [Database Setup & Storage Engine](#-database-setup--storage-engine)
7. [Automated Test Suite](#-automated-test-suite)
8. [Docker Deployment](#-docker-deployment)
9. [Repository File Guide (4–6 Lines Per File)](#-repository-file-guide)

---

## 🏗️ System Architecture

The IDP platform operates across three tightly integrated layers:
- **Layer 1: Routing & Heuristics** — Inspects raw PDF pages with zero-model heuristics, evaluates page layout complexity, and dispatches single or multi-engine extraction plans with VLM escalation.
- **Layer 2: Conversion & Engine Execution** — Executes the optimal extraction engine (Docling, PaddleOCR printed, or tuned handwritten DBNet), performs line-level normalized deduplication, and generates standardized Markdown with provenance metadata.
- **Layer 3: Schema Discovery, Extraction & Admin Application** — FastAPI web platform (`schema_chatbot_v2`) providing multi-tenant JWT authentication, interactive schema derivation via Sarvam Document AI and LLM, ReportLab PDF job reporting, PostgreSQL persistence, and interactive JSON Q&A.

```
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │                  INTELLIGENT DOCUMENT PROCESSING            │
                                  └─────────────────────────────────────────────────────────────┘
                                                                 │
      ┌──────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┐
      │                                                                                                                     │
      ▼                                                                                                                     ▼
┌────────────────────────────────────────────────────────┐                            ┌────────────────────────────────────────────────────────┐
│               LAYER 1: ROUTING & INSPECTION            │                            │             LAYER 3: SCHEMA DISCOVERY & APP            │
│  • PyMuPDF heuristic inspection (~10ms / page)         │                            │  • FastAPI 0.111 web server on port 8000               │
│  • Primary script & complexity score (0..5)            │                            │  • Multi-tenant Auth: RBAC (Admin / User) with JWT     │
│  • Single-engine or capability-based matching          │                            │  • User Store: JSONFileUserStore (data/users.json)     │
│  • VLM fallback: Ollama (Qwen2.5-VL) / Gemini          │                            │  • Tab Isolation: Scoped sessionStorage in browser     │
└──────────────────────────┬─────────────────────────────┘                            │  • Discovery State Machine: COLLECT → INFER → REVIEW   │
                           │                                                          │  • Sarvam Document AI OCR + Sarvam-105b LLM           │
                           ▼                                                          │  • ReportLab PDF job report generation                 │
┌────────────────────────────────────────────────────────┐                            │  • Query Bot: Interactive JSON QA endpoint             │
│              LAYER 2: CONVERSION ENGINES               │                            │  • Storage: PostgreSQL 18 (with SQLite fallback)       │
│  • Digital PDF: Docling 2.x (structured tables/text)   │                            └──────────────────────────┬─────────────────────────────┘
│  • Scanned Printed: PP-OCRv6 via PaddleOCR/PaddleX     │                                                       │
│  • Handwritten: PaddleOCR with det_db_thresh=0.20      │                                                       │
│  • Escalation Ladder: Digital → Printed → Hand → VLM   │                                                       │
│  • Line-level normalized deduplication & confidence    │                                                       │
│  • Output: <doc>.md with PIPELINE_SUMMARY JSON audit   │                                                       │
└──────────────────────────┬─────────────────────────────┘                                                       │
                           │                                                                                     │
                           └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │   POSTGRESQL 18 / RELATIONAL    │
                                              │  • documents (PDF blobs)        │
                                              │  • schemas (confirmed JSON)     │
                                              │  • document_markdowns (.md)     │
                                              │  • extraction_runs (JSON data)  │
                                              │  • job_pdfs (ReportLab reports) │
                                              └─────────────────────────────────┘
```

---

## 🔍 Architectural Details & Key Mechanisms

### 1. Step A: Fast Heuristic Inspection (Zero-Model)
- Evaluates raw page metrics using **PyMuPDF (`fitz`)** in ~10 milliseconds without loading GPU weights or invoking external APIs.
- Computes character count, word density, vector drawings, table line intersections, and image coverage ratio (`image_area / page_area`).
- Estimates raster DPI and inspects character Unicode blocks to detect Latin vs. Indic scripts.
- Assigns a heuristic `complexity_score` (0 to 5) based on dense layouts, vector sketches, and table structures.

### 2. Step B: Dual Routing Modes & Dead-Zone Resolution
- **`single_engine` mode**: Fast single-engine assignment. High-density digital pages (`char_count > 100` and `image_coverage < 0.02`) shortcut directly to Docling without VLM intervention.
- **`capability_based` mode**: Evaluates a mathematical set of booleans (`PageCapabilities`) and matches against engine capability subsets.
- **Dead-Zone Resolution**: When handwriting coverage falls in the ambiguous 10%–30% band, the router engages both printed and handwritten engines simultaneously rather than guessing a single winner.
- **VLM Disambiguation**: When heuristics are inconclusive (`route is None`), the page is rendered as PNG bytes and passed to the configured VLM adapter (`Ollama` or `Gemini`) for layout classification.

### 3. Escalation Ladder & Multi-Engine Merging
- If extraction confidence falls below `settings.escalation_confidence_threshold` (0.70), the page climbs the escalation ladder: `digital → paddleocr_printed → paddleocr_handwritten → vlm_transcribe`.
- Capped at `max_escalation_attempts=1` to prevent latency spirals; low-confidence pages are explicitly flagged in metadata.
- Multi-engine results are merged using line-level normalized deduplication: primary engine lines take precedence, and lower-priority lines are only appended if their normalized lowercase representation is unique.

### 4. Storage Architecture & Graceful Database Fallback
- `src/ai/layer3_extraction/storage.py` maintains SQLAlchemy ORM models across 5 distinct tables:
  1. `documents`: Uploaded PDF binaries with SHA-256 integrity hashes and file metadata.
  2. `schemas`: Target JSON schema definitions, field lists, and sample associations.
  3. `document_markdowns`: Converted Markdown texts (.md) from Layer 2.
  4. `extraction_runs`: Extracted JSON structured entities with performance timings and model telemetry.
  5. `job_pdfs`: Binary PDF report artifacts generated by ReportLab.
- On startup, the engine attempts connection to PostgreSQL 18 via pre-ping. If connection or authentication fails, it logs a warning and transparently switches to local SQLite (`sqlite:///./idp_storage.db`), ensuring zero downtime.

### 5. Multi-Tab Session Scoping & Persistent User Store
- **Per-Tab Isolation**: The web application stores JWT bearer tokens and roles in browser `sessionStorage` rather than `localStorage`. Tab A login never automatically logs into Tab B, and refreshing retains session state.
- **`JSONFileUserStore`**: Users are persisted to disk at `schema_chatbot_v2/data/users.json` using atomic file writes (`.tmp` replacement).
- **Cryptographic Security**: Passwords are saved strictly as salted PBKDF2-HMAC-SHA256 hashes (100,000 iterations); plaintext passwords are never stored. Server restarts preserve all existing users without overwriting.

---

## 🚀 Cloning & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SK321-VINAY-KY/IDP.git
cd IDP/engineer_a
```

### 2. Create and Activate a Python 3.11 Virtual Environment
> ⚠️ **Important**: Python 3.11 is strongly recommended. PaddleOCR and Docling require Python <= 3.11.

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core pipeline dependencies (PyMuPDF, Docling, PaddleOCR, PyTorch)
pip install -r requirements.txt

# Install Layer 3 Web Application dependencies (FastAPI, Uvicorn, ReportLab, Sarvam SDK)
pip install -r schema_chatbot_v2/requirements.txt
```

---

## 🔑 API Keys & Configuration Guide

Create and configure `.env` in the repository root and inside `schema_chatbot_v2/`.

### 1. Root `.env` Configuration (`c:\Users\Dell\Desktop\IDP\engineer_a\.env`)
```env
# --- LLM & VLM Providers ---
LLM_PROVIDER=sarvam
IDP_LLM_PROVIDER=sarvam
SARVAM_API_KEY=sk_your_sarvam_api_key_here
IDP_SARVAM_API_KEY=sk_your_sarvam_api_key_here
SARVAM_MODEL=sarvam-105b
IDP_SARVAM_MODEL_NAME=sarvam-105b
SARVAM_BASE_URL=https://api.sarvam.ai/v1
IDP_SARVAM_BASE_URL=https://api.sarvam.ai/v1

# --- Google Gemini VLM (Optional for Layer 1 Ambiguity) ---
IDP_GEMINI_API_KEY=your_google_gemini_api_key_here

# --- Sarvam Document AI OCR Settings ---
SARVAM_TIMEOUT_S=180
SARVAM_DOC_AI_LANGUAGE=en-IN
SARVAM_DOC_AI_POLL_INTERVAL_S=6
SARVAM_DOC_AI_TIMEOUT_S=120

# --- PostgreSQL Database ---
DATABASE_URL=postgresql://postgres:12345@localhost:5432/idp
IDP_DATABASE_URL=postgresql://postgres:12345@localhost:5432/idp

# --- Storage & Logging ---
SESSION_STORE=memory
LOG_LEVEL=INFO
```

### 2. Schema Chatbot `.env` (`schema_chatbot_v2/.env`)
```env
LLM_PROVIDER=sarvam
SARVAM_API_KEY=sk_your_sarvam_api_key_here
SARVAM_MODEL=sarvam-105b
SARVAM_BASE_URL=https://api.sarvam.ai/v1
SARVAM_TIMEOUT_S=180
SARVAM_DOC_AI_LANGUAGE=en-IN
SARVAM_DOC_AI_POLL_INTERVAL_S=6
SARVAM_DOC_AI_TIMEOUT_S=120
SESSION_STORE=memory
LOG_LEVEL=INFO
DATABASE_URL=postgresql://postgres:12345@localhost:5432/idp
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme
JWT_SECRET=idp-schema-pipeline-dev-secret-key-change-me
```

---

## ⚡ Execution & Run Instructions

### 1. Run the FastAPI Web Application & Dashboard
Start the Uvicorn application server:
```bash
# From schema_chatbot_v2 directory:
cd schema_chatbot_v2
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Or from workspace root:
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir schema_chatbot_v2 --host 127.0.0.1 --port 8000
```
- **Web UI & Admin Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Run Layer 1 & 2 Document Pipeline (Batch Conversion)
To convert PDFs in `dataset/` to structured Markdown in `dataset_output/`:
```bash
# Run resume pipeline conversion demo
.venv\Scripts\python.exe demo_resume_pipeline.py

# Run end-to-end multi-phase conversion harness
.venv\Scripts\python.exe demo_end_to_end.py
```

---

## 🗄️ Database Setup & Storage Engine

The application natively connects to **PostgreSQL 18** on `localhost:5432`.

### Verify Database Connection
Run a quick Python command to confirm tables are created and connected:
```bash
.venv\Scripts\python.exe -c "from src.ai.layer3_extraction.storage import get_engine, init_db; init_db(); print('Connected Dialect:', get_engine().dialect.name)"
```

### Table Structure in PostgreSQL (`idp` Database):
- `documents`: Stores PDF filename, SHA-256 checksum, content type, and binary blob (`BYTEA`).
- `schemas`: Stores schema ID, document type, field count, and complete JSON schema definition (`JSONB`).
- `document_markdowns`: Stores document ID, Markdown filename, raw Markdown text, and page references.
- `extraction_runs`: Stores structured JSON extraction outputs, LLM model used, and latency metrics.
- `job_pdfs`: Stores generated ReportLab PDF job report documents for asynchronous download.

---

## 🧪 Automated Test Suite

The repository contains 161 automated test cases covering routing tables, escalation logic, capability matching, Sarvam token caps, user stores, and session persistence:

```bash
# Run the complete test suite
.venv\Scripts\python.exe -m pytest tests/ schema_chatbot_v2/tests/ -v

# Run specific test suites
.venv\Scripts\python.exe -m pytest schema_chatbot_v2/tests/test_user_store.py -v
.venv\Scripts\python.exe -m pytest schema_chatbot_v2/tests/test_sarvam_adapter.py -v
.venv\Scripts\python.exe -m pytest schema_chatbot_v2/tests/test_storage_and_reports.py -v
```

---

## 🐳 Docker Deployment

Run both the conversion pipeline and the web service inside isolated containers:

```bash
# 1. Build container images
docker compose build

# 2. Run the Schema Chatbot web app in the background
docker compose up -d schema-chatbot

# 3. Execute batch document conversion
docker compose run --rm idp-layer12 python demo_resume_pipeline.py

# 4. Run tests inside Docker
docker compose run --rm schema-chatbot pytest /app/schema_chatbot_v2/tests -v
```

---

## 📂 Repository File Guide

*Detailed breakdown of every file across the codebase with its primary responsibility and underlying technology (4 to 6 lines per file).*

### Core Pipeline & Routing (`src/ai/layer1_routing/`)

#### `src/ai/layer1_routing/inspect.py`
Inspects incoming PDF pages using PyMuPDF to extract text density, bounding boxes, vector lines, and image coverage ratios.
Calculates heuristic metrics including character count, DPI estimates, script detection, and layout complexity score (0–5).
Runs in approximately 10ms per page as a zero-model heuristic step before any neural network or VLM invocation.
Built using PyMuPDF (`fitz`), Python standard library `unicodedata`, and native math geometry arithmetic.

#### `src/ai/layer1_routing/router.py`
Determines optimal extraction routes and builds multi-engine execution tasks based on page inspection profiles.
Implements the dead-zone resolution heuristic for pages with 10%–30% handwriting to prevent misclassification.
Evaluates single-engine shortcut pathways (e.g. pure digital pages) and routes ambiguous pages to VLM analysis.
Built with Python standard library typing, Pydantic contract models, and custom routing decision trees.

#### `src/ai/layer1_routing/capability_router.py`
Defines the `PROCESSOR_CAPABILITIES` dictionary mapping OCR engines to required document feature sets.
Matches detected page capabilities against engine prerequisites using strict mathematical subset operations.
Ranks viable candidate engines by priority and generates explainability diagnostics for unmatched engines.
Implemented using pure Python sets, dataclasses, and immutable tuple mappings without external library dependencies.

#### `src/ai/layer1_routing/pipeline.py`
Coordinates the end-to-end processing lifecycle for individual pages and multi-page PDF documents.
Renders pages at 150 DPI into NumPy arrays for OCR engines and PNG bytes for VLM analysis.
Applies the escalation ladder when initial confidence is low and deduplicates multi-engine line outputs.
Built with PyMuPDF, NumPy, Pillow (PIL), and standard library concurrent orchestration logic.

---

### Conversion Engines (`src/ai/layer2_conversion/`)

#### `src/ai/layer2_conversion/digital.py`
Extracts structured Markdown from born-digital PDF pages using Docling's layout-aware parsing pipeline.
Preserves complex reading orders, nested lists, and multi-column tables with high semantic fidelity.
Caches a singleton `DoclingConverter` instance to avoid costly model re-initialization on subsequent pages.
Employs Docling 2.x, PyMuPDF fallback extraction, and Pydantic validation models.

#### `src/ai/layer2_conversion/scanned.py`
Performs optical character recognition on scanned printed pages and handwritten fill-in forms.
Maintains two tuned singleton instances of PaddleOCR: standard printed PP-OCRv6 and a low-threshold handwritten DBNet.
Orders recognized text bounding boxes geometrically and computes character-weighted recognition confidence scores.
Built using PaddleOCR / PaddleX, PyTorch, NumPy, and standard-library regex layout heuristics.

---

### Extraction Storage & Markdown Output (`src/ai/`)

#### `src/ai/layer3_extraction/storage.py`
Provides centralized relational persistence across 5 ORM tables for documents, schemas, markdowns, runs, and job PDFs.
Initializes PostgreSQL 18 connections with connection pooling and implements an automated fallback to local SQLite.
Offers helper functions to save raw PDFs, store confirmed schemas, record extraction runs, and retrieve ReportLab PDFs.
Constructed with SQLAlchemy ORM, psycopg2-binary, SQLite3, hashlib SHA-256, and Pydantic.

#### `src/ai/output/md_writer.py`
Formats extracted page bodies into clean Markdown documents featuring per-page provenance HTML comments.
Generates an audit-ready `<!-- PIPELINE_SUMMARY [...] -->` JSON footer encapsulating engine choices, confidence, and timings.
Writes document-level `<stem>.schema_ref.json` sidecar files linking documents to confirmed schema definitions.
Built purely with Python standard library string formatting, Pathlib, and JSON serialization.

---

### Schemas & Contract Types (`src/ai/schemas/`)

#### `src/ai/schemas/page.py`
Defines the cross-module contract dataclasses: `PageProfile`, `PageClassification`, `VLMAnalysis`, and `PageOutput`.
Guarantees strict type safety between Layer 1 routing, Layer 2 conversion, and Layer 3 schema extraction.
Encapsulates merged confidence ratings, capability flag lists, escalation status, and engine attribution strings.
Implemented using Pydantic v1 BaseModel specifications with rigid field constraints and default values.

#### `src/ai/schemas/page_metadata.py`
Maintains internal tracking structures including `EngineResult`, `RoutingDecision`, `EscalationRecord`, and `PageMetadata`.
Captures granular latency measurements, intermediate engine failures, and execution traces for audit logging.
Supplements public contract types with execution telemetry without exposing internal mechanics downstream.
Built using Pydantic BaseModel and Python standard library typing constructs.

---

### VLM & LLM Adapters (`src/adapters/llm/`)

#### `src/adapters/llm/base.py`
Establishes the abstract base class `LLMClient` declaring methods for page classification, analysis, and transcription.
Defines synchronous and asynchronous signatures ensuring pluggable compatibility across different vision-language providers.
Enforces typed returns using Pydantic models for structured visual page decomposition.
Built using Python's `abc` module and typing annotations.

#### `src/adapters/llm/factory.py`
Implements a factory pattern function `get_llm_client()` that instantiates the configured VLM adapter.
Dispatches between local Ollama instances, Google Gemini cloud clients, or lightweight deterministic mock clients.
Reads configuration settings dynamically while allowing runtime keyword argument overrides.
Constructed using standard Python factory patterns and module imports.

#### `src/adapters/llm/ollama_client.py`
Integrates local Vision-Language Models (e.g., Qwen2.5-VL) via Ollama's OpenAI-compatible REST endpoint.
Encodes page pixmaps into base64 PNG data URLs and transmits them via HTTP POST to `localhost:11434`.
Parses returned JSON payloads into strongly typed `VLMAnalysis` structures with layout quality scores.
Built with `httpx`, base64 encoding, and Pydantic schema validation.

#### `src/adapters/llm/gemini_client.py`
Connects to Google Gemini multimodal models using the official Google GenAI SDK for visual document inspection.
Transmits rendered page image bytes directly to generate structured layout and handwritten transcription analyses.
Utilizes `IDP_GEMINI_API_KEY` loaded securely from root `.env` configuration.
Built with `google-genai` SDK, Pillow (PIL), and Pydantic models.

#### `src/adapters/llm/schema_models.py`
Specifies Pydantic v1 response models for VLM inference including printed vs. handwritten percentage splits.
Contains typed schema representations for table detection, diagram flags, and direct markdown transcription output.
Ensures JSON responses from external VLMs adhere strictly to the internal Layer 1 routing expectations.
Constructed using Pydantic v1 schema definitions.

---

### Configuration & Utilities (`src/config/`, `src/utils/`)

#### `src/config/settings.py`
Centralized application configuration managing pipeline thresholds, engine modes, and service connection strings.
Loads environment variables from `.env` using Pydantic Settings with automatic `IDP_` prefix fallback resolution.
Defines tunable parameters for DPI rendering, confidence floors, DBNet detection thresholds, and database URLs.
Built with `pydantic-settings`, `python-dotenv`, and Python `pathlib`.

#### `src/utils/logger.py`
Configures structured JSON logging across the application with contextual metadata for distributed tracing.
Tracks individual document processing lifecycles using a Python `ContextVar` to inject consistent `correlation_id` values.
Formats log records with ISO timestamps, log levels, service names, and custom structured dictionary arguments.
Implemented using Python standard library `logging` and `contextvars`.

---

### Web Application & API Routes (`schema_chatbot_v2/app/`)

#### `schema_chatbot_v2/app/main.py`
Initializes the FastAPI application instance, configures CORS middleware, mounts static UI assets, and registers routers.
Manages application lifespan events by initializing the active LLM adapter and verifying database connectivity.
Exposes root health probe endpoints and redirects browser root paths to the static admin portal.
Built using FastAPI, Starlette, Uvicorn, and Python standard library logging.

#### `schema_chatbot_v2/app/config.py`
Defines the Pydantic Settings configuration dataclass for the Layer 3 schema discovery web service.
Manages settings for Sarvam Document AI, LLM model names, JWT secrets, session timeouts, and database URLs.
Resolves configuration values from `schema_chatbot_v2/.env` and system environment variables.
Constructed using `pydantic-settings` and `python-dotenv`.

#### `schema_chatbot_v2/app/api/auth_routes.py`
Implements user authentication endpoints including `/auth/login`, `/auth/register`, and `/auth/me`.
Validates OAuth2 password request forms, compares PBKDF2 password hashes, and issues signed JWT bearer tokens.
Provides current user profile and role introspection to front-end clients.
Built with FastAPI, OAuth2PasswordRequestForm, and `app.core.auth` helper utilities.

#### `schema_chatbot_v2/app/api/routes.py`
Exposes the core interactive chatbot endpoints: `POST /chat` and `POST /schema/infer`.
Handles conversational turns against the schema discovery state machine to add, update, and confirm schema fields.
Accepts multipart PDF document uploads and triggers Sarvam Document AI OCR extraction and schema derivation.
Constructed using FastAPI, Pydantic v2 schemas, and ConversationManager orchestration.

#### `schema_chatbot_v2/app/api/pipeline_routes.py`
Provides pipeline execution endpoints for document intake, batch extraction jobs, status polling, and report downloads.
Generates comprehensive ReportLab PDF job report documents detailing page metrics, field counts, and extraction accuracy.
Hosts the Query Bot endpoint (`POST /api/query-bot/ask`) for interactive natural language queries over extracted JSON.
Built with FastAPI, ReportLab Platypus, SQLAlchemy ORM storage, and Sarvam LLM adapters.

#### `schema_chatbot_v2/app/api/user_routes.py`
Delivers self-service document management endpoints for standard non-admin users.
Allows users to upload personal PDFs, list uploaded documents, and initiate extraction jobs against confirmed schemas.
Restricts document access to the authenticated document owner to maintain tenant data boundaries.
Built using FastAPI, HTTP bearer authentication dependencies, and SQLAlchemy storage.

#### `schema_chatbot_v2/app/api/admin_routes.py`
Supplies administrative control endpoints for listing all registered users, updating user roles, and managing schemas.
Provides administrative oversight over pipeline jobs, execution histories, and system health status.
Secured with role-based access control (RBAC) requiring verified administrator privileges.
Constructed with FastAPI, JWT security dependencies, and UserStore abstractions.

---

### Core State & Authentication (`schema_chatbot_v2/app/core/`)

#### `schema_chatbot_v2/app/core/auth.py`
Implements password hashing, salt verification, and JWT creation/validation logic.
Utilizes standard library PBKDF2-HMAC-SHA256 with 100,000 iterations to eliminate external binary hashing dependencies.
Decodes bearer tokens, verifies cryptographic expiration, and injects current user identity into FastAPI dependencies.
Built using Python standard libraries `hashlib`, `hmac`, `secrets`, and `pyjwt`.

#### `schema_chatbot_v2/app/core/conversation_manager.py`
Drives the interactive schema discovery state machine through `COLLECT`, `INFER`, `REVIEW`, and `CONFIRMED` states.
Parses natural language user feedback to modify schema fields, alter data types, and update validation constraints.
Generates proactive clarification questions when inferred schema fields contain ambiguous types or missing descriptions.
Built using Python dataclasses, SchemaState CRUD, and LLMAdapter abstractions.

#### `schema_chatbot_v2/app/core/schema_state.py`
Maintains an in-memory structured representation of field definitions throughout an active schema review session.
Enforces field name uniqueness, validates data type changes (string, number, date, list, object), and tracks required flags.
Exports finalized schemas to JSON definitions matching the schema registry storage contract.
Constructed with Python standard library dictionaries, copy utilities, and Pydantic validation.

#### `schema_chatbot_v2/app/core/state_machine.py`
Enumerates the lifecycle states of schema discovery: `START`, `COLLECT`, `INFER`, `REVIEW`, and `CONFIRMED`.
Validates valid transitions between conversational phases and guards against illegal state modifications.
Ensures schemas cannot be committed to disk without user confirmation.
Implemented with Python `enum.Enum` and standard library state validation logic.

#### `schema_chatbot_v2/app/core/validator.py`
Validates schema JSON definitions against strict JSON-schema standards and structural integrity constraints.
Ensures nested list items, currency codes, date formatting patterns, and enumeration choices adhere to specifications.
Provides detailed error messages indicating problematic field names and corrective actions.
Constructed using standard library regex patterns and JSON schema inspection logic.

#### `schema_chatbot_v2/app/core/activity_log.py`
Records user interactions, pipeline triggers, and administrative operations into an append-only event log.
Structures activity events with timestamps, user identities, action categories, and execution outcomes.
Enables administrative audit tracking and user history inspection.
Implemented using standard library collections, time utilities, and JSON formatting.

#### `schema_chatbot_v2/app/core/log_buffer.py`
Maintains a thread-safe circular memory buffer capturing recent application log entries.
Exposes recent server log lines via API endpoints to power the real-time admin monitoring dashboard.
Prevents memory exhaustion by evicting older log lines past a fixed capacity limit.
Built with Python `collections.deque` and threading synchronization locks.

---

### LLM Adapters & Prompts (`schema_chatbot_v2/app/llm/`)

#### `schema_chatbot_v2/app/llm/base.py`
Defines the `LLMAdapter` abstract interface declaring methods for document extraction, question phrasing, and schema inference.
Standardizes how LLM providers parse structured JSON from text and digitize PDF documents.
Enforces consistent parameter signatures across local and cloud LLM implementations.
Built using Python's `abc` module and asynchronous type annotations.

#### `schema_chatbot_v2/app/llm/factory.py`
Factory module responsible for instantiating the appropriate `LLMAdapter` based on environment configuration.
Supports seamless switching between Sarvam AI, Ollama, AWS Bedrock, and Mock adapters.
Validates required API keys and connection parameters before returning active adapter instances.
Implemented using Python factory design patterns and conditional module loading.

#### `schema_chatbot_v2/app/llm/sarvam_adapter.py`
Core adapter integrating Sarvam AI's Document AI OCR and `sarvam-105b` large language model.
Calls Sarvam's REST API with defensive token cap truncation detection and token budget management (`max_tokens=2560`).
Manages the asynchronous Document AI digitization lifecycle: multipart upload, polling, ZIP download, and Markdown extraction.
Built with `httpx`, lazy-imported `sarvamai` SDK, `zipfile`, and custom `TruncatedCompletionError` handling.

#### `schema_chatbot_v2/app/llm/ollama_adapter.py`
Implements the `LLMAdapter` interface for local model execution via Ollama's REST API.
Enables local development and offline schema discovery without external commercial API dependencies.
Converts OpenAI-compatible chat completion JSON into internal schema proposal structures.
Built using `httpx` asynchronous client and Pydantic response parsing.

#### `schema_chatbot_v2/app/llm/bedrock_adapter.py`
Cloud production adapter designed for deploying schema discovery on Amazon Web Services (AWS) Bedrock.
Invokes Anthropic Claude or Amazon Titan models using the AWS Boto3 SDK with AWS IAM credential management.
Structures requests and extracts structured JSON outputs following Bedrock Converse API conventions.
Constructed using `boto3` and AWS SDK runtime clients.

#### `schema_chatbot_v2/app/llm/mock_adapter.py`
Deterministic mock adapter for automated integration tests and offline unit validation.
Simulates Document AI digitization and returns predefined schema definitions without network overhead.
Enables comprehensive continuous integration (CI) testing without consuming API credits.
Implemented purely in Python using static dictionaries and simulated async delays.

#### `schema_chatbot_v2/app/llm/prompts.py`
Houses system prompt templates, few-shot examples, and JSON schema formatting instructions for LLM calls.
Guides models to extract consistent field structures, identify primary document types, and formulate polite clarification questions.
Contains specialized prompt blocks for invoices, resumes, medical bills, and complex government forms.
Maintained as pure Python multi-line string templates.

---

### Storage, Models & Client (`schema_chatbot_v2/app/storage/`, `models/`, `cli/`)

#### `schema_chatbot_v2/app/storage/user_store.py`
Provides persistent storage for application user accounts using the `JSONFileUserStore` implementation.
Loads users from `data/users.json` on startup, writes updates atomically via `.tmp` replacement, and preserves PBKDF2 password hashes.
Seeds a default administrator account from environment variables only if the store is empty, avoiding overwrite on restart.
Built with Python standard library `json`, `pathlib`, `uuid`, and `InMemoryUserStore` for tests.

#### `schema_chatbot_v2/app/storage/session_store.py`
Manages conversation and review sessions through `JSONFileSessionStore` and `InMemorySessionStore`.
Persists active conversation states, turn counts, schema drafts, and ownership tokens across server restarts.
Offers configurable backends selectable via the `SESSION_STORE` environment variable (`memory` or `file`).
Built with Pydantic BaseModel, `pathlib`, `uuid`, and Python standard library `json`.

#### `schema_chatbot_v2/app/models/api_models.py`
Defines request and response Pydantic models for all Layer 3 FastAPI routes.
Encapsulates structures for chat requests, schema field operations, token responses, and job submissions.
Enforces type validation, field constraints, and automatic OpenAPI schema generation.
Built with Pydantic v2 BaseModel and Field declarations.

#### `schema_chatbot_v2/app/output/schema_renderer.py`
Renders confirmed schema definitions into human-readable Markdown documentation and graphical summaries.
Generates sample JSON templates from schema definitions illustrating expected extraction output shapes.
Assists users in reviewing schema structures before finalizing extraction configurations.
Constructed using Python standard library string templates and JSON formatting.

#### `schema_chatbot_v2/cli/client.py`
Command-line interface (CLI) client for interacting with the schema chatbot service directly from the terminal.
Supports session initiation, interactive message dispatching, document uploading, and schema confirmation from CLI.
Formats JSON server responses into terminal tables and color-coded status banners.
Built with Python standard library `urllib`, `argparse`, and terminal ANSI escape sequences.

---

### Frontend UI & Demo Harnesses (`static/`, root scripts)

#### `schema_chatbot_v2/static/index.html`
Semantic single-page web interface providing document upload, schema chatbot interaction, and admin management tabs.
Includes responsive panels for pipeline status monitoring, JSON data viewing, ReportLab PDF downloads, and Query Bot Q&A.
Features light/dark theme toggling and cache-busted asset inclusion (`app.js?v=18`).
Structured using semantic HTML5 elements, CSS custom properties, and accessible form controls.

#### `schema_chatbot_v2/static/app.js`
Client-side JavaScript application driving the dynamic UI without third-party frontend frameworks.
Uses `sessionStorage` for `TOKEN_KEY` and `ROLE_KEY` to guarantee strict per-tab authentication isolation.
Manages asynchronous API communication, polling loops for pipeline jobs, and query bot chat rendering.
Built with vanilla modern ECMAScript (ES6+), Fetch API, DOM manipulation, and sessionStorage.

#### `schema_chatbot_v2/static/style.css`
Custom CSS stylesheet implementing a modern design system with curated dark and light mode themes.
Styles glassmorphic cards, admin data tables, query bot message bubbles, and high-contrast JSON viewers.
Provides smooth CSS transitions, responsive grid layouts, and mobile-friendly media queries.
Authored in pure Vanilla CSS using CSS variables, flexbox, and grid layouts.

#### `demo_resume_pipeline.py`
Demonstration script executing the Layer 1 & 2 extraction pipeline on sample resume documents.
Renders PDF pages, evaluates routing heuristics, executes Docling or PaddleOCR, and writes output Markdown.
Uses `MockLLMClient` to verify that pure digital documents process without external network model dependencies.
Built with PyMuPDF, `pipeline.process_document()`, and `md_writer`.

#### `demo_end_to_end.py`
Comprehensive two-phase demonstration harness linking Phase 1 schema discovery with Phase 2 document conversion.
Calls `/schema/infer` on the running chatbot to derive a shared schema, then processes corpus PDFs through Engineer A.
Outputs Markdown files and schema sidecars linking each processed document to its registry schema identifier.
Built using `httpx`, PyMuPDF, Pathlib, and the Layer 1 routing pipeline.

---

## 📜 License
Internal IDP Platform — Proprietary & Confidential.
