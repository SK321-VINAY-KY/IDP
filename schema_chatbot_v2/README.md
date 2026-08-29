# Schema Discovery Chatbot

Phase 1 MVP: a chatbot that interviews a non-technical user in plain English
and progressively builds a validated JSON target schema for an IDP
(Intelligent Document Processing) pipeline.

## Architecture

```
USER
  │
  ▼
CLI client / (future React frontend)
  │  HTTP
  ▼
FastAPI  (app/api/routes.py)
  │
  ▼
ConversationManager        <- the only orchestrator (app/core/conversation_manager.py)
  │           │
  ▼           ▼
StateMachine  LLMAdapter (Ollama now / Bedrock later / Mock for tests)
  │           (app/llm/*)
  ▼
SchemaState   <- progressively built, deterministic mutations only
  │           (app/core/schema_state.py)
  ▼
Validator     (app/core/validator.py)
  │
  ▼
SessionStore  <- in-memory now, swappable for DynamoDB/Redis
  (app/storage/session_store.py)
```

**The core boundary:** the LLM interprets user messages and phrases
questions. It never decides what happens next or mutates state directly —
it returns a structured `ExtractionResult` proposal, and
`ConversationManager` is the only code that applies it to `SchemaState` or
advances `ConversationState`. This is what keeps the bot from silently
skipping required fields or "confirming" something the user never agreed to.

## Why it's built this way (matches the design doc + our earlier discussion)

- **Deterministic state machine, not an LLM-driven one** (`app/core/state_machine.py`)
  — an explicit transition table, not an if/elif chain, so it doesn't
  calcify into spaghetti once corrections/interrupts are added.
- **Progressive schema building** (`app/core/schema_state.py`) — the schema
  is a live object mutated turn by turn, never generated in one shot at the
  end.
- **Gap-driven questioning** — each field tracks its own missing attributes
  (`FieldSpec.missing_attributes()`); the bot always asks about the single
  next gap, in the order fields were added.
- **Corrections work at any state** — "remove X" is parsed and applied
  before the state-specific handler runs, so a correction mid-interview
  doesn't get lost (this was flagged as a gap in the original design doc
  and is now handled explicitly — see `ConversationManager._apply_removals`).
- **LLM failures degrade gracefully** — every adapter method catches
  provider errors and malformed JSON, and question phrasing has a
  deterministic template fallback (`app/llm/prompts.py::fallback_question`)
  so the bot never crashes or hangs if the LLM is unreachable or returns junk.
- **Swappable everything** — `LLMAdapter` (Ollama/Bedrock/Mock) and
  `SessionStore` (in-memory now) are both interfaces. Switching from Ollama
  to Bedrock is `LLM_PROVIDER=bedrock` in `.env`, not a code change.

## Document intake (upload 2-5 samples instead of the text interview)

Alongside the plain-text interview, `POST /schema/infer` lets the user upload
2-5 sample PDFs of the same document type. One inference pass proposes a
`document_type` + field list; the *same* deterministic gap-detection the text
interview uses (`SchemaState.next_gap()`) then decides what, if anything,
still needs asking - a field the samples disagreed about (different
formats, or present in some but not others) simply shows up as a normal gap
question, so no new gap-handling logic was needed for this.

```
POST /schema/infer   (multipart/form-data, field name "files", 2-5 PDFs)
  -> same response envelope as /chat: {session_id, message, state, schema, ...}
```

```bash
python -m cli.client --from-documents invoice1.pdf invoice2.pdf invoice3.pdf
```

**Provider support** (`LLMAdapter.infer_schema_from_pdfs`):

| Provider | How it reads the PDFs |
| --- | --- |
| `sarvam` | Two steps: each sample is OCR'd via Sarvam **Document AI** (`Digitise`, powered by Sarvam Vision) into Markdown, then the plain text-only `sarvam-105b` model compares the N markdown texts and proposes the schema - vision and reasoning stay decoupled. |
| `bedrock` | One step: Claude reads the raw PDF bytes natively via a Converse `document` content block, no OCR round-trip. Same "not exercised against live AWS" caveat as the rest of `BedrockAdapter`. |
| `mock` | No real PDF support (zero dependencies, by design) - expects each "sample" as UTF-8 text in a tiny `name: type: yes|no` format; see the docstring on `MockLLMAdapter.infer_schema_from_pdfs`. Used by `tests/test_mock_document_inference.py` and the document-intake tests in `test_conversation_manager.py`. |
| `ollama` | Not supported (no vision model configured) - returns a graceful `extraction_failed` result rather than raising, so the caller falls back to the plain-text interview. |

New env vars (only used by the `sarvam` provider's document intake):
`SARVAM_DOC_AI_LANGUAGE`, `SARVAM_DOC_AI_POLL_INTERVAL_S`, `SARVAM_DOC_AI_TIMEOUT_S`.

## What's intentionally NOT in this MVP

- **Cardinality / relationships** ("can a document contain multiple X?")
  — the design doc's RELATIONSHIPS state. Skipped for Phase 1 to avoid
  scope creep; the field checklist (`REQUIRED_ATTRIBUTES` in
  `schema_state.py`) is the single place to add it later.
- **Persistence beyond memory** — `SessionStore` is an interface for a
  reason; a `DynamoDBSessionStore` just needs to implement `create/get/save/delete`.
- **Real Bedrock testing** — `BedrockAdapter` is structurally complete and
  mirrors `OllamaAdapter`'s interface exactly, but hasn't been run against
  a live AWS account. Verify the `converse()` tool-use response parsing
  against your actual Bedrock access before relying on it.
- **The "which ambiguous question wins" priority rule** flagged earlier —
  currently strict FIFO by field-add order. Fine for Phase 1, worth
  revisiting if multi-field ambiguity turns out to be common.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

### Run against Ollama (your current setup)

```bash
# make sure ollama is running and you've pulled a model, e.g.:
#   ollama pull llama3.1
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1
uvicorn app.main:app --reload
```

### Run against Sarvam AI

```bash
export LLM_PROVIDER=sarvam
export SARVAM_API_KEY=sk_your_key_here      # from https://dashboard.sarvam.ai
export SARVAM_MODEL=sarvam-105b             # or sarvam-105b-conversations
uvicorn app.main:app --reload
```

### Run against the mock adapter (zero dependencies, for demos/CI)

```bash
LLM_PROVIDER=mock uvicorn app.main:app --reload
```

### Later, on AWS

```bash
export LLM_PROVIDER=bedrock
export BEDROCK_REGION=ap-south-1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
uvicorn app.main:app
```

## Try it

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
python -m cli.client --show-schema
```

Example session:

```
Bot: What type of documents are you processing?
You: insurance claims
Bot: What information would you like to extract?
You: customer name, policy number, claim amount and claim date
Bot: Is 'customer_name' always present in the document, or can it be missing sometimes?
You: always
...
Bot: Here's what I'll extract: ...  Is this correct?
You: yes
Bot: Schema created successfully.
```

## API

```
POST /chat
  {}                                        -> starts a new session
  {"session_id": "...", "message": "..."}   -> continues it

GET /session/{session_id}
POST /session/{session_id}/reset

GET /health
```

Response shape:
```json
{
  "session_id": "...",
  "message": "...",
  "state": "FIELD_DETAILS",
  "schema": { "document_type": "...", "fields": [...] },
  "completed": false,
  "schema_id": null,
  "errors": null
}
```

## Tests

```bash
pytest -v
```

23 tests, all offline (state machine transitions, schema mutation/gap
detection, validation rules, and 5 full end-to-end conversations through
`ConversationManager` using `MockLLMAdapter` — no network required).

## Repo layout

```
app/
  config.py                 env-driven settings
  main.py                   FastAPI app
  api/routes.py             HTTP layer
  core/
    state_machine.py        deterministic transition table
    schema_state.py         the live, progressively-built schema
    validator.py            pre-completion validation rules
    conversation_manager.py the orchestrator
  llm/
    base.py                 LLMAdapter interface + ExtractionResult contract
    prompts.py               shared prompt text + fallback question templates
    ollama_adapter.py        local dev (JSON-mode)
    sarvam_adapter.py        Sarvam AI (OpenAI-compatible, structured outputs)
    bedrock_adapter.py       AWS target (tool-use / function calling)
    mock_adapter.py          deterministic, no network - used by tests
    factory.py                LLM_PROVIDER=... -> adapter instance
  storage/session_store.py  SessionStore interface + in-memory impl
  models/api_models.py      request/response schemas
cli/client.py                terminal test client
tests/                       23 tests, run with `pytest -v`
```

## Suggested next phases (from the original plan, unchanged)

1. ✅ Phase 1 — this repo
2. Phase 2 — DynamoDB-backed `SessionStore`
3. Phase 3 — verify `BedrockAdapter` against live AWS, tune the tool schema
4. Phase 4 — schema registry (persist + version confirmed schemas)
5. Phase 5 — wire `schema_id` into the actual IDP pipeline
6. Phase 6 — React/Next.js frontend against the existing REST API (no
   backend changes needed — that's the point of the decoupling)
