"""
File: demo_end_to_end.py
Purpose: Unified end-to-end demo that runs the full IDP product flow:

    PHASE 1 — Schema Discovery (via running chatbot on :8000)
      1. Scan dataset/ folder and list PDFs of the same document type
      2. Ask user to pick 2–5 sample PDFs of the target type
      3. POST samples to POST /schema/infer → Sarvam OCR → shared schema proposal
      4. Interactive REVIEW mode: user inspects the proposed schema and edits
         it (add/remove/rename fields, change types, make required/optional)
         using natural-language commands or direct JSON edit
      5. On confirmation → the schema is persisted to schema_registry/ as a
         durable JSON file (schema_<id>.json) — the Engineer B cache.

    PHASE 2 — Engineer A Pipeline (batch, no VLM needed for digital pages)
      1. Iterate over EVERY PDF in dataset/
      2. Run process_document(write_output=True) → document .md output
      3. Sidecar file <stem>.schema_ref.json is written alongside each .md
         containing the schema_id so Engineer B knows which schema to apply.
      4. Final report per document + overall summary.

The script is document-type-agnostic. If you replace dataset/ with invoices,
receipts, tax forms, lab reports, etc., everything works identically:
the chatbot infers whatever schema is appropriate for the new samples,
the pipeline runs the same routing regardless of domain.

Requirements:
  - Chatbot server running on http://localhost:8000 (schema_chatbot_v2)
    with a valid LLM provider configured (sarvam recommended — see .env)
  - Engineer A deps installed (pip install -r requirements.txt)
  - dataset/ folder populated with your target PDFs (one document type at a
    time recommended — mixing invoices + resumes in the same batch will
    produce a union schema that's poorer for both)

Outputs:
  - schema_registry/schema_<id>.json        — confirmed schema cache
  - dataset_output/<pdf_stem>.md            — pipeline markdown output
  - dataset_output/<pdf_stem>.schema_ref.json — sidecar linking doc→schema
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is in schema_chatbot_v2 reqs
    print("[setup] pip install httpx (required for chatbot HTTP client)")
    sys.exit(2)

sys.path.insert(0, os.path.dirname(__file__))

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, VLMAnalysis
from src.ai.layer1_routing.pipeline import process_document
from src.config.settings import settings
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "dataset_output"
SCHEMA_REGISTRY = ROOT / "schema_registry"
CHATBOT_URL_DEFAULT = "http://localhost:8000"
RENDER_DPI = 150
MAT = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
USER_BULLET = "\n  You> " if os.name != "nt" else "\n  You> "
BOT_BULLET = "  Bot> "
SEP = "=" * 72
SUBSEP = "-" * 72


# ---------------------------------------------------------------------------
# Phase 1 — Schema Discovery helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (DATASET_DIR, OUTPUT_DIR, SCHEMA_REGISTRY):
        d.mkdir(parents=True, exist_ok=True)


def _chat(
    client: httpx.Client, session_id: str | None, message: str | None
) -> dict:
    """Call POST /chat. Pass session_id=None, message=None for first turn."""
    resp = client.post("/chat", json={"session_id": session_id, "message": message})
    resp.raise_for_status()
    return resp.json()


def _infer_schema(client: httpx.Client, sample_paths: list[Path]) -> dict:
    """POST /schema/infer with 2-5 sample PDFs. Returns parsed JSON."""
    files = []
    for sp in sample_paths:
        fh = sp.open("rb")
        files.append(("files", (sp.name, fh, "application/pdf")))
    try:
        resp = client.post("/schema/infer", files=files, timeout=300)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    if resp.status_code != 200:
        raise RuntimeError(f"schema/infer failed [{resp.status_code}]: {resp.text}")
    return resp.json()


def _print_schema(schema: dict | None, indent: int = 2) -> None:
    if not schema:
        print("    (no schema yet)")
        return
    text = json.dumps(schema, indent=2, sort_keys=False)
    for line in text.splitlines():
        print(" " * indent + line)


def _pick_samples_interactive(all_pdfs: list[Path]) -> list[Path]:
    """Ask the user to choose 2-5 sample PDFs from dataset/ for inference."""
    print(f"\n  Found {len(all_pdfs)} PDFs in dataset/:")
    for i, p in enumerate(all_pdfs, 1):
        size_kb = p.stat().st_size // 1024
        print(f"    [{i:2d}] {p.name:45s}  {size_kb:>6d} KB")

    print(
        "\n  Pick 2-5 samples of the SAME document type for schema inference.\n"
        "  Enter numbers separated by spaces, e.g. 1 3 5.\n"
        "  (Enter = first 2 samples if >=2 available.)"
    )
    while True:
        raw = input(USER_BULLET).strip()
        if raw == "":
            if len(all_pdfs) >= 2:
                return all_pdfs[:2]
            print("  Need at least 2 PDFs. Try again.")
            continue
        try:
            idxs = sorted({int(x) for x in raw.split()})
        except ValueError:
            print("  Enter numbers only, e.g. 1 2 5")
            continue
        if not all(1 <= i <= len(all_pdfs) for i in idxs):
            print(f"  Out of range. Choose 1..{len(all_pdfs)}")
            continue
        if not (2 <= len(idxs) <= 5):
            print("  Must pick 2-5 samples (chatbot MIN/MAX constraint).")
            continue
        return [all_pdfs[i - 1] for i in idxs]


def _review_interactive(client: httpx.Client, data: dict) -> dict:
    """Drive the REVIEW chat loop until the session becomes COMPLETED."""
    session_id = data["session_id"]

    print(f"\n  Session ID : {session_id}")
    print(f"  Initial state: {data.get('state')}")
    if data.get("message"):
        print(f"\n{BOT_BULLET}{data['message']}")

    print(f"\n  Proposed schema:")
    _print_schema(data.get("schema"), indent=4)

    if data.get("errors"):
        print("\n  Validation errors to resolve first:")
        for e in data["errors"]:
            print(f"    - {e}")

    if data.get("completed"):
        print("\n  Schema already confirmed on first turn (unusual but valid).")
        return data

    print(
        "\n  Commands available during review:\n"
        "    Type any natural-language edit, e.g.:\n"
        "      - Rename 'name' to 'full_name' and make it required\n"
        "      - Add a field 'total_experience_years' of type number\n"
        "      - Make github_url optional instead of required\n"
        "      - Confirm\n"
        "    /schema   — re-print the current schema JSON\n"
        "    /json     — open $EDITOR to paste/type raw JSON schema to replace\n"
        "    /bot      — print the last bot message again\n"
        "    /abandon  — quit without confirming schema (stops the demo)\n"
    )

    last_data = data
    while True:
        try:
            raw = input(USER_BULLET).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)

        if raw == "":
            continue

        if raw == "/abandon":
            print("\n  Abandoned per user request. No schema persisted.")
            sys.exit(0)

        if raw == "/schema":
            print("  Current schema:")
            _print_schema(last_data.get("schema"), indent=4)
            continue

        if raw == "/bot":
            print(f"\n{BOT_BULLET}{last_data.get('message', '(no message)')}")
            continue

        if raw == "/json":
            last_data = _manual_json_edit(client, last_data)
        else:
            resp = client.post(
                "/chat",
                json={"session_id": session_id, "message": raw},
                timeout=120,
            )
            resp.raise_for_status()
            last_data = resp.json()
            print(f"\n{BOT_BULLET}{last_data.get('message', '')}")
            if last_data.get("schema"):
                print("\n  Schema now:")
                _print_schema(last_data.get("schema"), indent=4)
            if last_data.get("errors"):
                print("\n  Validation errors:")
                for e in last_data["errors"]:
                    print(f"    ! {e}")

        if last_data.get("completed"):
            schema_id = last_data.get("schema_id") or "(missing)"
            print(f"\n  [CONFIRMED] schema_id = {schema_id}")
            break

    return last_data


def _manual_json_edit(client: httpx.Client, last_data: dict) -> dict:
    """Advanced escape hatch: let the user paste raw JSON schema instead of
    using the natural-language chat. We still round-trip through the chat
    endpoint so validation + completion state stay correct."""
    print(
        "\n  Paste/type the complete JSON schema below.\n"
        "  When finished, enter a single line containing exactly:  ENDJSON"
    )
    lines: list[str] = []
    while True:
        try:
            line = input("    JSON> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return last_data
        if line.strip() == "ENDJSON":
            break
        lines.append(line)
    try:
        obj = json.loads("\n".join(lines))
    except json.JSONDecodeError as e:
        print(f"  ! Invalid JSON: {e}. Discarding edit.")
        return last_data

    # Round-trip via /chat with a structured instruction so schema_state
    # gets updated atomically. This lets the validator run and the
    # confirmation check fire for a valid one.
    payload_json = json.dumps(obj)
    msg = (
        "Please apply this exact JSON schema and then confirm it if valid:\n"
        f"```json\n{payload_json}\n```"
    )
    resp = client.post(
        "/chat",
        json={"session_id": last_data["session_id"], "message": msg},
        timeout=120,
    )
    resp.raise_for_status()
    new_data = resp.json()
    print(f"\n{BOT_BULLET}{new_data.get('message', '')}")
    if new_data.get("schema"):
        print("\n  Schema now:")
        _print_schema(new_data.get("schema"), indent=4)
    return new_data


def _persist_schema(data: dict, sample_paths: list[Path]) -> Path:
    """Write the confirmed schema + provenance to schema_registry/ as JSON."""
    schema_id = data.get("schema_id")
    if not schema_id:
        schema_id = f"schema_{int(time.time())}"
        data["schema_id"] = schema_id

    record = {
        "schema_id": schema_id,
        "confirmed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": data.get("session_id"),
        "state": data.get("state"),
        "completed": data.get("completed", True),
        "sample_documents": [sp.name for sp in sample_paths],
        "document_type": (data.get("schema") or {}).get("document_type"),
        "schema": data.get("schema"),
        "provenance": {
            "source": "schema_chatbot_v2 /schema/infer + interactive review",
            "chatbot_url": CHATBOT_URL_DEFAULT,
        },
    }
    out = SCHEMA_REGISTRY / f"{schema_id}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Phase 2 — Engineer A Pipeline helpers
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Stand-in LLM. Digital pages never need VLM calls because Step A of
    the router decides conclusively from char_count/digital_text. If any
    page DOES reach VLM (scanned, mixed-content, handwriting) this client
    loudly fails — use a real LLMClient adapter for those batches."""

    def classify_page(
        self, image_bytes: bytes, page_profile_hint: dict
    ) -> PageClassification:
        raise RuntimeError(
            "MockLLMClient.classify_page called. Your batch contains pages "
            "that need a real VLM (scanned / mixed / handwriting). Run with "
            "--routing-mode single_engine + a real LLM, or use capability_"
            "based mode with a VLM adapter wired."
        )

    def analyze_page(self, image_bytes, page_profile_hint) -> VLMAnalysis:
        raise RuntimeError(
            "MockLLMClient.analyze_page called — see classify_page()."
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        raise RuntimeError(
            "MockLLMClient.transcribe_handwriting called — see classify_page()."
        )


def _render_page(page) -> tuple[np.ndarray, bytes]:
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    png = pix.tobytes("png")
    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return arr, png


def _build_pages_for_pdf(pdf_path: Path) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        arr, png = _render_page(page)
        pages.append({
            "page": page,
            "page_number": i + 1,
            "context": {
                "pdf_path": str(pdf_path),
                "image_array": arr,
                "image_bytes": png,
            },
            "image_bytes": png,
        })
    return pages  # doc kept alive via page references


def _run_pipeline_for_all(
    pdfs: list[Path],
    schema_record_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Run the pipeline on every PDF. Returns (successes, failures).

    For digital batches this is fully automatic. For pages that actually
    need VLM (scanned/mixed/handwriting) the mock will raise — the error
    handler catches it, flags that doc as FAILED, and the summary at the
    end tells the user to wire up a real LLMClient for those.
    """
    llm_client = MockLLMClient()
    schema_record = json.loads(schema_record_path.read_text(encoding="utf-8"))
    schema_id = schema_record["schema_id"]

    successes: list[dict] = []
    failures: list[dict] = []

    for pdf in pdfs:
        print(f"\n  >> {pdf.name}")
        t0 = time.monotonic()
        try:
            pages = _build_pages_for_pdf(pdf)
            print(f"     - {len(pages)} pages rendered")

            results = process_document(
                pages=pages,
                llm_client=llm_client,
                document_name=pdf.name,
                document_id=f"batch-{pdf.stem}",
                write_output=True,
                output_dir=str(OUTPUT_DIR),
                overwrite=True,
            )

            md_path = OUTPUT_DIR / f"{pdf.stem}.md"
            if not md_path.exists():
                raise FileNotFoundError(
                    f"md_writer didn't produce {md_path} (check write_output wiring)"
                )

            ref_path = OUTPUT_DIR / f"{pdf.stem}.schema_ref.json"
            ref_path.write_text(
                json.dumps({
                    "source_pdf": pdf.name,
                    "output_md": md_path.name,
                    "schema_id": schema_id,
                    "schema_registry_file": schema_record_path.name,
                    "document_type": schema_record.get("document_type"),
                    "pages": [
                        {
                            "page_number": output.page_number,
                            "engines_used": output.engines_used,
                            "capabilities": output.capabilities,
                            "confidence": output.confidence,
                            "escalated": output.escalated,
                            "low_confidence": output.low_confidence,
                            "chars": len(output.markdown.strip()),
                        }
                        for output, _meta in results
                    ],
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            elapsed = time.monotonic() - t0
            outputs = [r[0] for r in results]
            avg_conf = (
                sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0
            )
            total_chars = sum(len(o.markdown.strip()) for o in outputs)
            print(
                f"     - OK in {elapsed:.1f}s  avg_conf={avg_conf:.3f}  "
                f"chars={total_chars}  escalated={sum(1 for o in outputs if o.escalated)}"
            )
            successes.append({
                "pdf": pdf.name,
                "md": str(md_path),
                "schema_ref": str(ref_path),
                "avg_conf": avg_conf,
                "chars": total_chars,
                "pages": len(outputs),
                "elapsed_s": round(elapsed, 2),
            })

        except Exception as exc:
            logger.error("pipeline.batch_doc_failed", pdf=pdf.name,
                         error=str(exc), error_type=type(exc).__name__)
            elapsed = time.monotonic() - t0
            failures.append({
                "pdf": pdf.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(elapsed, 2),
            })
            print(f"     ! FAILED ({type(exc).__name__}): {exc}")

    return successes, failures


# ---------------------------------------------------------------------------
# Banner & entry
# ---------------------------------------------------------------------------

def _banner() -> None:
    print("\n" + SEP)
    print("  IDP End-to-End Demo")
    print(SEP)
    print(f"  dataset/         : {DATASET_DIR}")
    print(f"  dataset_output/  : {OUTPUT_DIR}")
    print(f"  schema_registry/ : {SCHEMA_REGISTRY}")
    print(f"  chatbot URL      : {CHATBOT_URL_DEFAULT}")
    print(f"  routing_mode     : {settings.routing_mode}")
    print(SEP)


def _health_check(client: httpx.Client) -> None:
    try:
        r = client.get("/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"\n  [chatbot] /health = {info.get('status')}  "
              f"provider={info.get('llm_provider', 'unknown')}")
    except Exception as exc:
        print(
            f"\n  [FATAL] Cannot reach chatbot at {CHATBOT_URL_DEFAULT}/health\n"
            f"         {type(exc).__name__}: {exc}\n"
            f"\n  Start the server first:\n"
            f"       cd schema_chatbot_v2\n"
            f"       python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n"
            f"\n  Then re-run this demo."
        )
        sys.exit(1)


def main() -> int:
    global DATASET_DIR, OUTPUT_DIR, SCHEMA_REGISTRY, CHATBOT_URL_DEFAULT
    parser = argparse.ArgumentParser(
        description="End-to-end IDP demo: schema discovery chatbot + Engineer A batch pipeline"
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR),
                        help=f"Folder with target PDFs (default: {DATASET_DIR})")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help=f"Pipeline .md outputs (default: {OUTPUT_DIR})")
    parser.add_argument("--schema-registry", default=str(SCHEMA_REGISTRY),
                        help=f"Confirmed schema store (default: {SCHEMA_REGISTRY})")
    parser.add_argument("--chatbot-url", default=CHATBOT_URL_DEFAULT,
                        help="Base URL of running chatbot (default: http://localhost:8000)")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Re-use latest schema in schema_registry/ instead of running chatbot")
    parser.add_argument("--samples", nargs="+", metavar="PDF",
                        help="Pre-select 2-5 sample PDFs (paths or filenames inside dataset/)")
    args = parser.parse_args()
    DATASET_DIR = Path(args.dataset).resolve()
    OUTPUT_DIR = Path(args.output_dir).resolve()
    SCHEMA_REGISTRY = Path(args.schema_registry).resolve()
    CHATBOT_URL_DEFAULT = args.chatbot_url.rstrip("/")

    _ensure_dirs()
    _banner()
    set_correlation_id(f"demo-e2e-{int(time.time())}")

    all_pdfs = sorted(p for p in DATASET_DIR.glob("*.pdf") if p.is_file())
    if not all_pdfs:
        print(f"\n  [FATAL] No PDFs found in {DATASET_DIR}/. Put your documents there first.")
        return 1
    print(f"  Found {len(all_pdfs)} PDFs in dataset/")

    client = httpx.Client(base_url=CHATBOT_URL_DEFAULT, timeout=120)
    _health_check(client)

    # =========================== PHASE 1 =====================================
    print("\n" + SEP)
    print("  PHASE 1 — Schema Discovery (chatbot)")
    print(SEP)

    if args.skip_phase1:
        existing = sorted(SCHEMA_REGISTRY.glob("schema_*.json"), key=lambda p: p.stat().st_mtime)
        if not existing:
            print(f"\n  [FATAL] --skip-phase1 but no schema_*.json in {SCHEMA_REGISTRY}/")
            return 1
        schema_record_path = existing[-1]
        rec = json.loads(schema_record_path.read_text(encoding="utf-8"))
        print(f"\n  Reusing latest schema: {schema_record_path.name}")
        print(f"  schema_id  : {rec.get('schema_id')}")
        print(f"  doc type   : {rec.get('document_type')}")
        print(f"  confirmed  : {rec.get('confirmed_at')}")
        sample_names = rec.get("sample_documents") or []
        sample_paths = [DATASET_DIR / n for n in sample_names if (DATASET_DIR / n).exists()]
    else:
        if args.samples:
            resolved: list[Path] = []
            for s in args.samples:
                p = Path(s)
                if not p.is_absolute():
                    p2 = DATASET_DIR / s
                    if p2.exists():
                        p = p2.resolve()
                    else:
                        p = p.resolve()
                resolved.append(p)
            if not (2 <= len(resolved) <= 5):
                print(f"\n  [FATAL] --samples needs 2-5 paths, got {len(resolved)}")
                return 1
            for p in resolved:
                if not p.exists():
                    print(f"\n  [FATAL] sample not found: {p}")
                    return 1
            sample_paths = resolved
        else:
            sample_paths = _pick_samples_interactive(all_pdfs)

        print(f"\n  Samples chosen for inference:")
        for i, sp in enumerate(sample_paths, 1):
            sz_kb = sp.stat().st_size // 1024
            print(f"    {i}. {sp.name}  ({sz_kb} KB)")

        print("\n  Calling POST /schema/infer … (Sarvam OCR + LLM proposal)")
        print("  Expect 60–180 s for 2 digital PDFs (Sarvam async OCR polls every 6 s)")
        t0 = time.monotonic()
        data = _infer_schema(client, sample_paths)
        print(f"  Inference returned in {time.monotonic()-t0:.1f}s.")

        print("\n" + SUBSEP)
        print("  REVIEW & EDIT — shape the schema, then confirm")
        print(SUBSEP)
        confirmed = _review_interactive(client, data)
        schema_record_path = _persist_schema(confirmed, sample_paths)
        print(f"\n  Schema persisted to: {schema_record_path}")

    # =========================== PHASE 2 =====================================
    print("\n" + SEP)
    print("  PHASE 2 — Engineer A Batch Pipeline (all PDFs in dataset/)")
    print(SEP)
    print(f"  Targets : {len(all_pdfs)} PDFs")
    print(f"  Schema  : {schema_record_path.name}  "
          f"(id={json.loads(schema_record_path.read_text(encoding='utf-8'))['schema_id']})")
    print(
        f"  Routing : {settings.routing_mode}  "
        f"(MockLLM — digital-only pages supported in this demo; scanned pages will "
        f"loudly fail so you know to wire a real VLM)"
    )

    t_start = time.monotonic()
    successes, failures = _run_pipeline_for_all(all_pdfs, schema_record_path)
    wall = time.monotonic() - t_start

    # =========================== SUMMARY =====================================
    print("\n" + SEP)
    print("  DONE — Summary")
    print(SEP)
    print(f"  Total PDFs   : {len(all_pdfs)}")
    print(f"  Successful   : {len(successes)}")
    print(f"  Failed       : {len(failures)}")
    print(f"  Wall time    : {wall:.1f} s")
    print(f"  Schema file  : {schema_record_path}")
    print(f"  Output dir   : {OUTPUT_DIR}")

    if successes:
        print(f"\n  Pipeline outputs:")
        for s in successes:
            print(f"    [OK] {s['pdf']:45s}  pages={s['pages']:2d}  "
                  f"conf={s['avg_conf']:.3f}  chars={s['chars']:6d}  "
                  f"t={s['elapsed_s']:6.1f}s")
            print(f"         md   : {s['md']}")
            print(f"         ref  : {s['schema_ref']}")

    if failures:
        print(f"\n  Failures — action needed:")
        for f in failures:
            print(f"    [FAIL] {f['pdf']}  after {f['elapsed_s']}s")
            print(f"           {f['error_type']}: {f['error']}")
        print(
            "\n  If failures mention MockLLMClient.* called — the batch contained "
            "non-digital pages (scanned / handwriting / mixed content). Wire a "
            "real LLMClient implementation (or set routing_mode=single_engine + "
            "provide Ollama/sarvamai credentials) then re-run with "
            "--skip-phase1 to reuse the confirmed schema."
        )

    print("\n" + SEP)
    print(f"  Result: {'ALL OK' if not failures else f'{len(failures)} DOCUMENT(S) FAILED'}")
    print(SEP + "\n")
    client.close()
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
