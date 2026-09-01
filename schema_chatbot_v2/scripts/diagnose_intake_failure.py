"""
scripts/diagnose_intake_failure.py

Diagnostic for schema_chatbot_v2 intake failures where
proposal.extraction_failed is False but proposal.document_type is null.

Calls the REAL adapter code without modification — imports SarvamAdapter,
_digitise, build_document_inference_prompt, and _chat directly and adds
instrumentation around each stage.

Usage (from schema_chatbot_v2/ directory):
    python scripts/diagnose_intake_failure.py

Repro files  : place 170.pdf and 168.pdf in schema_chatbot_v2/scripts/
Control files: uses existing tests/fixtures/ PDFs (MY_resume.pdf and/or
               Vinay_Resume.pdf) as known-good baselines.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

# ── path setup so app.* imports resolve regardless of launch directory ──────
_HERE    = Path(__file__).resolve().parent          # schema_chatbot_v2/scripts/
_APP_ROOT = _HERE.parent                            # schema_chatbot_v2/
_FIXTURES = _APP_ROOT.parent / "tests" / "fixtures" # engineer_a/tests/fixtures/
sys.path.insert(0, str(_APP_ROOT))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(_APP_ROOT / ".env")

import httpx
from app.config import settings
from app.llm.sarvam_adapter import SarvamAdapter
from app.llm.prompts import (
    DOCUMENT_INFERENCE_SYSTEM_PROMPT,
    build_document_inference_prompt,
)

# ── tokenizer (no Sarvam-native tokenizer published yet) ────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> tuple[int, str]:
        return len(_enc.encode(text)), "tiktoken/cl100k_base (approx)"
except ImportError:
    def count_tokens(text: str) -> tuple[int, str]:  # type: ignore[misc]
        return len(text) // 4, "chars÷4 (approx, tiktoken not installed)"

# ── file inventory ───────────────────────────────────────────────────────────
_CANDIDATES = {
    "168.pdf":          _HERE / "168.pdf",
    "170.pdf":          _HERE / "170.pdf",
    "MY_resume.pdf":    _FIXTURES / "MY_resume.pdf",
    "Vinay_Resume.pdf": _FIXTURES / "Vinay_Resume.pdf",
}
FILES = {name: path for name, path in _CANDIDATES.items() if path.exists()}

SEP   = "─" * 72
DSEP  = "═" * 72

# ── helpers ──────────────────────────────────────────────────────────────────

def _read(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _preview(text: str, chars: int = 300) -> str:
    """First + last N chars with a gap marker in between."""
    if len(text) <= chars * 2:
        return text
    return text[:chars] + f"\n... [{len(text) - chars * 2} chars omitted] ...\n" + text[-chars:]


def _digitise_instrumented(adapter: SarvamAdapter, index: int, pdf_bytes: bytes) -> dict:
    """
    Calls the real _digitise() internals with extra instrumentation.
    Does NOT modify _digitise() — reimplements its body with added prints
    so every intermediate state is visible.
    """
    result: dict[str, Any] = {
        "job_id": None,
        "job_status_final": None,
        "poll_count": 0,
        "ocr_text": "",
        "ocr_text_len": 0,
        "zip_contents": [],
        "error": None,
    }

    try:
        job = adapter.doc_ai_client.doc_ai.digitise(
            file=[(f"sample_{index}.pdf", io.BytesIO(pdf_bytes), "application/pdf")],
            language=settings.sarvam_doc_ai_language,
            output_format="md",
        )
        result["job_id"]     = job.job_id
        result["job_status_initial"] = job.status

        terminal = {"completed", "partially_completed", "failed", "rejected"}
        deadline = time.monotonic() + settings.sarvam_doc_ai_timeout_s
        status   = job.status
        job_id   = job.job_id
        polls    = 0

        while status.lower() not in terminal:
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} timed out")
            time.sleep(settings.sarvam_doc_ai_poll_interval_s)
            status = adapter.doc_ai_client.doc_ai.get_status(job_id=job_id).status
            polls += 1

        result["job_status_final"] = status
        result["poll_count"]       = polls

        if status.lower() in ("failed", "rejected"):
            result["error"] = f"job ended with status={status!r}"
            return result

        download = adapter.doc_ai_client.doc_ai.get_download_url(job_id=job_id)
        resp = httpx.get(download.url, timeout=adapter.timeout_s)
        resp.raise_for_status()
        zip_bytes = resp.content

        # Inspect ZIP contents before extracting
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            result["zip_contents"] = zf.namelist()
            candidates = [n for n in zf.namelist() if n.endswith(".md") and "metadata/" not in n]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.endswith(".md")]
            if not candidates:
                result["error"] = "no .md file in ZIP"
                return result
            md_text = zf.read(candidates[0]).decode("utf-8", errors="replace")

        result["ocr_text"]     = md_text
        result["ocr_text_len"] = len(md_text)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _call_chat_instrumented(adapter: SarvamAdapter, sys_prompt: str, user_prompt: str) -> dict:
    """
    Calls the Sarvam /v1/chat/completions endpoint the way _chat() does,
    but captures the full raw response JSON including the usage block.
    Does NOT call adapter._chat() — reimplements the HTTP call so we can
    inspect the raw response before any content routing logic.
    """
    payload: dict[str, Any] = {
        "model": adapter.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature":     0.1,
        "max_tokens":      8192,
        "reasoning_effort": "low",
    }

    result: dict[str, Any] = {
        "http_status": None,
        "raw_response_json": None,
        "usage": None,
        "prompt_tokens_api": None,
        "completion_tokens_api": None,
        "total_tokens_api": None,
        "raw_content": None,
        "raw_reasoning_content": None,
        "content_used": None,
        "parsed_document_type": None,
        "parse_error": None,
        "error": None,
    }

    try:
        resp = httpx.post(
            f"{adapter.base_url}/chat/completions",
            json=payload,
            headers={
                "api-subscription-key": adapter.api_key or "",
                "Content-Type": "application/json",
            },
            timeout=max(adapter.timeout_s, 180.0),
        )
        result["http_status"] = resp.status_code

        try:
            data = resp.json()
            result["raw_response_json"] = data
        except Exception:
            result["raw_response_json"] = resp.text
            result["error"] = "response was not valid JSON"
            return result

        # Usage block
        usage = data.get("usage", {})
        result["usage"]                 = usage
        result["prompt_tokens_api"]     = usage.get("prompt_tokens")
        result["completion_tokens_api"] = usage.get("completion_tokens")
        result["total_tokens_api"]      = usage.get("total_tokens")

        # Content routing — mirrors _chat() logic exactly
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content          = msg.get("content") or ""
            reasoning        = msg.get("reasoning_content") or ""
            result["raw_content"]           = content
            result["raw_reasoning_content"] = reasoning

            # Mirror _chat() routing: prefer content; fall back to reasoning
            if not content:
                used = reasoning
            elif "{" in reasoning and "{" not in content:
                used = reasoning
            else:
                used = content
            result["content_used"] = used
        else:
            result["error"] = "no choices in response"
            return result

        # Parse document_type
        try:
            parsed = adapter._parse_json(result["content_used"])
            result["parsed_document_type"] = parsed.get("document_type")
        except Exception as exc:
            result["parse_error"] = str(exc)

        if not resp.is_success:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── per-file diagnostic ───────────────────────────────────────────────────────

def diagnose_file(name: str, path: Path, adapter: SarvamAdapter) -> dict:
    print(f"\n{DSEP}")
    print(f"  FILE: {name}  ({path})")
    print(f"  Size: {path.stat().st_size:,} bytes")
    print(DSEP)

    pdf_bytes = _read(path)

    # ── Stage 1: OCR ────────────────────────────────────────────────────
    print(f"\n  [Stage 1] Sarvam Document AI — Digitise")
    print(SEP)
    ocr = _digitise_instrumented(adapter, index=1, pdf_bytes=pdf_bytes)

    print(f"  Job ID          : {ocr.get('job_id')}")
    print(f"  Status (initial): {ocr.get('job_status_initial')}")
    print(f"  Status (final)  : {ocr.get('job_status_final')}")
    print(f"  Poll count      : {ocr.get('poll_count')}")
    print(f"  ZIP contents    : {ocr.get('zip_contents')}")
    print(f"  OCR text length : {ocr.get('ocr_text_len')} chars")

    if ocr.get("error"):
        print(f"  ERROR           : {ocr['error']}")
    else:
        print(f"\n  OCR text preview (first + last 300 chars):")
        print("  " + _preview(ocr["ocr_text"]).replace("\n", "\n  "))

    if ocr.get("error"):
        return {"name": name, "ocr": ocr, "prompt": {}, "chat": {}, "error": ocr["error"]}

    # ── Stage 2: Prompt construction ────────────────────────────────────
    print(f"\n  [Stage 2] Prompt construction")
    print(SEP)

    # Exactly as infer_schema_from_pdfs() does:
    capped_text  = ocr["ocr_text"][:3500] if ocr["ocr_text_len"] > 3500 else ocr["ocr_text"]
    capped_len   = len(capped_text)
    was_capped   = ocr["ocr_text_len"] > 3500

    user_prompt  = build_document_inference_prompt([capped_text])
    sys_prompt   = (
        DOCUMENT_INFERENCE_SYSTEM_PROMPT
        + "\n\nCRITICAL: Keep reasoning brief. Output ONLY the valid JSON object "
          "with keys 'document_type' and 'fields'. Do not wrap in commentary."
    )
    full_prompt  = sys_prompt + "\n" + user_prompt

    sys_tokens,  sys_label  = count_tokens(sys_prompt)
    user_tokens, user_label = count_tokens(user_prompt)
    full_tokens, _          = count_tokens(full_prompt)

    prompt_info = {
        "ocr_text_len_raw": ocr["ocr_text_len"],
        "capped_at_3500": was_capped,
        "capped_text_len": capped_len,
        "user_prompt_len_chars": len(user_prompt),
        "sys_prompt_len_chars": len(sys_prompt),
        "sys_tokens": sys_tokens,
        "user_tokens": user_tokens,
        "full_prompt_tokens": full_tokens,
        "token_method": sys_label,
    }

    print(f"  OCR text raw length : {ocr['ocr_text_len']} chars")
    print(f"  Capped at 3,500?    : {was_capped}  (post-cap: {capped_len} chars)")
    print(f"  System prompt       : {len(sys_prompt)} chars  ≈ {sys_tokens} tokens  [{sys_label}]")
    print(f"  User prompt         : {len(user_prompt)} chars  ≈ {user_tokens} tokens")
    print(f"  TOTAL prompt        : {full_tokens} tokens  (sarvam-105b context window: 16,384)")
    if full_tokens > 12_000:
        print(f"  ⚠️  WARNING: prompt is within 4k of the 16k context ceiling")

    # ── Stage 3: Chat completion ─────────────────────────────────────────
    print(f"\n  [Stage 3] Sarvam /v1/chat/completions — raw response")
    print(SEP)
    chat = _call_chat_instrumented(adapter, sys_prompt, user_prompt)

    print(f"  HTTP status         : {chat.get('http_status')}")
    print(f"  prompt_tokens  (API): {chat.get('prompt_tokens_api')}")
    print(f"  completion_tokens   : {chat.get('completion_tokens_api')}")
    print(f"  total_tokens   (API): {chat.get('total_tokens_api')}")
    print(f"  raw content length  : {len(chat.get('raw_content') or '')} chars")
    print(f"  reasoning length    : {len(chat.get('raw_reasoning_content') or '')} chars")
    print(f"  content branch used : {'reasoning_content' if chat.get('content_used') == chat.get('raw_reasoning_content') and chat.get('raw_reasoning_content') else 'content'}")

    if chat.get("error"):
        print(f"  ERROR               : {chat['error']}")

    print(f"\n  Raw content (full):")
    raw_c = chat.get("raw_content") or ""
    print("  " + (raw_c[:2000] if raw_c else "(empty)").replace("\n", "\n  "))
    if len(raw_c) > 2000:
        print(f"  ... [{len(raw_c) - 2000} chars truncated in display only]")

    print(f"\n  Parsed document_type: {chat.get('parsed_document_type')!r}")
    if chat.get("parse_error"):
        print(f"  Parse error         : {chat['parse_error']}")

    return {"name": name, "ocr": ocr, "prompt": prompt_info, "chat": chat}


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print(f"\n{DSEP}")
    print("  SUMMARY TABLE")
    print(DSEP)
    hdr = (
        f"  {'File':<22}  {'OCR status':<20}  {'OCR len':>7}  "
        f"{'Cap?':>5}  {'Prompt tok (approx)':>20}  "
        f"{'Prompt tok (API)':>16}  {'Comp tok':>8}  "
        f"{'document_type result'}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in results:
        name  = r["name"][:22]
        ocr   = r.get("ocr", {})
        pr    = r.get("prompt", {})
        ch    = r.get("chat", {})

        ocr_status  = ocr.get("job_status_final") or ocr.get("error", "ERROR")[:20]
        ocr_len     = ocr.get("ocr_text_len", "—")
        capped      = "YES" if pr.get("capped_at_3500") else "no"
        ptok_approx = pr.get("full_prompt_tokens", "—")
        ptok_api    = ch.get("prompt_tokens_api", "—")
        ctok        = ch.get("completion_tokens_api", "—")
        doc_type    = repr(ch.get("parsed_document_type"))

        print(
            f"  {name:<22}  {str(ocr_status):<20}  {str(ocr_len):>7}  "
            f"{capped:>5}  {str(ptok_approx):>20}  "
            f"{str(ptok_api):>16}  {str(ctok):>8}  "
            f"{doc_type}"
        )
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(DSEP)
    print("  Sarvam Intake Failure Diagnostic")
    print(f"  Model    : {settings.sarvam_model}")
    print(f"  Base URL : {settings.sarvam_base_url}")
    print(f"  API key  : {'SET' if settings.sarvam_api_key else 'MISSING — will fail'}")
    print(DSEP)

    if not FILES:
        print("\nERROR: No test files found.")
        print("Expected locations:")
        for name, path in _CANDIDATES.items():
            status = "✓ exists" if path.exists() else "✗ missing"
            print(f"  {status}  {path}")
        sys.exit(1)

    print(f"\nFiles to process ({len(FILES)}):")
    for name, path in FILES.items():
        print(f"  {'REPRO  ' if name in ('168.pdf','170.pdf') else 'CONTROL'}  {name}  ({path.stat().st_size:,} bytes)")

    adapter = SarvamAdapter()
    results = []

    for name, path in FILES.items():
        r = diagnose_file(name, path, adapter)
        results.append(r)

    print_summary(results)

    # Also dump full raw response JSONs for offline inspection
    dump_path = _HERE / "diagnose_output.json"
    # Sanitize for JSON serialization (remove bytes objects etc.)
    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        if isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"
        return obj

    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump([_safe(r) for r in results], f, indent=2, default=str)
    print(f"  Full raw responses written → {dump_path}\n")


if __name__ == "__main__":
    main()
