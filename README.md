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

## Running the smoke tests

```bash
pip install pydantic==1.10.13 tenacity --break-system-packages
python3 tests/test_router_smoke.py
```

These test the routing/escalation *logic* without requiring PaddleOCR/TrOCR/
Docling model downloads. Full integration tests (real OCR calls) need the
Docker Compose stack from the build guide running, with models pulled via Ollama.

## Setup for real runs

```bash
pip install -r requirements.txt --break-system-packages
# Ollama must be running locally with the VLM model pulled:
#   ollama pull qwen2-vl:2b
```
# IDP
Intelligent Document Processing
