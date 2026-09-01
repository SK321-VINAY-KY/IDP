"""
Diagnostic script to investigate completion truncation with reasoning_effort=None and max_tokens=1536.

Tests:
  - Repro files: dataset/168.pdf, dataset/170.pdf
  - Control files: dataset/MY_resume.pdf, dataset/Vinay_Resume.pdf
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx

# Ensure paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "schema_chatbot_v2"))

from app.llm.sarvam_adapter import SarvamAdapter, TruncatedCompletionError
from app.llm.prompts import (
    DOCUMENT_INFERENCE_SYSTEM_PROMPT,
    build_document_inference_prompt,
)


def safe_preview(text: str, n: int = 200) -> str:
    """Safely format preview string."""
    if not text:
        return "<empty>"
    clean = text.encode("utf-8", errors="replace").decode("utf-8").replace("\n", " ")
    return clean[:n] + ("..." if len(clean) > n else "")


def main():
    print("=" * 100)
    print("DIAGNOSTIC RUN: Investigating Truncation with reasoning_effort=None, max_tokens=1536")
    print("=" * 100)

    adapter = SarvamAdapter()
    print(f"Sarvam Model: {adapter.model}")
    print(f"Base URL:     {adapter.base_url}")
    print(f"API Key set:  {bool(adapter.api_key)}")
    print("-" * 100)

    recorded_statuses: Dict[str, str] = {}
    last_job_id: List[Optional[str]] = [None]

    doc_ai = adapter.doc_ai_client.doc_ai
    orig_digitise = doc_ai.digitise
    orig_get_status = doc_ai.get_status

    def hooked_digitise(*args, **kwargs):
        job = orig_digitise(*args, **kwargs)
        last_job_id[0] = job.job_id
        recorded_statuses[job.job_id] = job.status
        return job

    def hooked_get_status(*args, **kwargs):
        res = orig_get_status(*args, **kwargs)
        job_id = kwargs.get("job_id") or (args[0] if args else None)
        if job_id:
            last_job_id[0] = job_id
            recorded_statuses[job_id] = res.status
        return res

    doc_ai.digitise = hooked_digitise
    doc_ai.get_status = hooked_get_status

    files_to_test = [
        ("Control 1", ROOT / "dataset" / "MY_resume.pdf"),
        ("Control 2", ROOT / "dataset" / "Vinay_Resume.pdf"),
        ("Repro 1", ROOT / "dataset" / "168.pdf"),
        ("Repro 2", ROOT / "dataset" / "170.pdf"),
    ]

    summary_rows: List[Dict[str, Any]] = []

    for label, pdf_path in files_to_test:
        print(f"\n{'#' * 100}")
        print(f"TESTING [{label}]: {pdf_path.name}")
        print(f"{'#' * 100}")

        if not pdf_path.exists():
            print(f"ERROR: File does not exist: {pdf_path}")
            summary_rows.append({
                "filename": pdf_path.name,
                "finish_reason": "FILE_NOT_FOUND",
                "completion_tokens": "N/A",
                "reasoning_content_len": 0,
                "content_len": 0,
                "ocr_len": 0,
                "doc_type_resolved": "No (file missing)",
            })
            continue

        pdf_bytes = pdf_path.read_bytes()
        print(f"File size: {len(pdf_bytes)} bytes")

        # STEP 1: Digitise (OCR)
        print("\n--- STEP 1: OCR via Sarvam Digitise ---")
        ocr_text = ""
        ocr_status = "unknown"
        t0 = time.monotonic()
        try:
            ocr_text = adapter._digitise(1, pdf_bytes)
            job_id = last_job_id[0]
            ocr_status = recorded_statuses.get(job_id, "completed")
            print(f"OCR Job ID: {job_id}")
            print(f"OCR Status: {ocr_status}")
            print(f"OCR Text Length: {len(ocr_text)} chars")
            print(f"OCR Time: {round(time.monotonic() - t0, 2)}s")
            print(f"OCR Preview (first 200 chars): {safe_preview(ocr_text, 200)}")
        except Exception as exc:
            print(f"OCR Failed: {exc}")
            summary_rows.append({
                "filename": pdf_path.name,
                "finish_reason": "OCR_FAILED",
                "completion_tokens": "N/A",
                "reasoning_content_len": 0,
                "content_len": 0,
                "ocr_len": 0,
                "doc_type_resolved": f"No (OCR error: {exc})",
            })
            continue

        # STEP 2: Build prompt
        capped_text = ocr_text[:3500] if len(ocr_text) > 3500 else ocr_text
        user_prompt = build_document_inference_prompt([capped_text])
        sys_prompt = (
            DOCUMENT_INFERENCE_SYSTEM_PROMPT
            + "\n\nCRITICAL: Keep reasoning brief. Output ONLY the valid JSON object with keys 'document_type' and 'fields'. Do not wrap in commentary."
        )

        # STEP 3: Call Sarvam Chat completions with current production parameters (reasoning_effort=None, max_tokens=2560)
        print("\n--- STEP 2: Calling Sarvam /chat/completions (reasoning_effort=None, max_tokens=2560) ---")
        payload = {
            "model": adapter.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2560,
        }
        # Note: reasoning_effort is omitted/None when reasoning_effort is None

        finish_reason = "unknown"
        completion_tokens = "N/A"
        prompt_tokens = "N/A"
        content_str = ""
        reasoning_str = ""
        doc_type_resolved = "No"

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
            print(f"HTTP Status: {resp.status_code}")
            if not resp.is_success:
                print(f"Error Body: {resp.text}")
                resp.raise_for_status()

            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", "N/A")
            completion_tokens = usage.get("completion_tokens", "N/A")
            print(f"Usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={usage.get('total_tokens')}")

            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason", "unknown")
                msg = choice.get("message", {})
                content_str = msg.get("content") or ""
                reasoning_str = msg.get("reasoning_content") or ""

            print(f"finish_reason: {finish_reason}")
            print(f"reasoning_content length: {len(reasoning_str)} chars")
            print(f"reasoning_content preview: {safe_preview(reasoning_str, 200)}")
            print(f"content length: {len(content_str)} chars")
            print(f"content preview: {safe_preview(content_str, 200)}")

            # Check if document_type resolves via adapter parsing
            raw_text = content_str if content_str else reasoning_str
            if "{" in reasoning_str and "{" not in content_str:
                raw_text = reasoning_str

            try:
                parsed = adapter._parse_json(raw_text)
                if isinstance(parsed, dict) and parsed.get("document_type"):
                    doc_type_resolved = f"Yes ({parsed.get('document_type')})"
                elif isinstance(parsed, dict):
                    doc_type_resolved = "No (document_type is null)"
                else:
                    doc_type_resolved = "No (non-dict JSON)"
            except Exception as e:
                doc_type_resolved = f"No (Parse error: {e})"

            print(f"document_type resolution: {doc_type_resolved}")

        except Exception as exc:
            print(f"Chat API Call Failed: {exc}")
            finish_reason = f"CALL_ERROR: {exc}"
            doc_type_resolved = f"No (Error: {exc})"

        summary_rows.append({
            "filename": pdf_path.name,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "reasoning_content_len": len(reasoning_str),
            "content_len": len(content_str),
            "ocr_len": len(ocr_text),
            "doc_type_resolved": doc_type_resolved,
            "reasoning_preview": safe_preview(reasoning_str, 120),
            "content_preview": safe_preview(content_str, 120),
        })

    # COMPARISON TABLE
    print("\n" + "=" * 125)
    print("COMPARISON TABLE (reasoning_effort=None, max_tokens=2560)")
    print("=" * 125)
    header = f"{'filename':<18} | {'finish_reason':<14} | {'compl_tokens':<12} | {'reasoning_len':<13} | {'content_len':<11} | {'ocr_len':<9} | {'document_type resolved?'}"
    print(header)
    print("-" * 125)
    for r in summary_rows:
        row_str = (
            f"{r['filename']:<18} | "
            f"{str(r['finish_reason']):<14} | "
            f"{str(r['completion_tokens']):<12} | "
            f"{str(r['reasoning_content_len']):<13} | "
            f"{str(r['content_len']):<11} | "
            f"{str(r['ocr_len']):<9} | "
            f"{r['doc_type_resolved']}"
        )
        print(row_str)
    print("=" * 125)

    # Detailed Previews
    print("\nDETAILED PREVIEWS:")
    for r in summary_rows:
        print(f"\n[{r['filename']}]:")
        print(f"  OCR length:            {r['ocr_len']}")
        print(f"  Finish reason:         {r['finish_reason']}")
        print(f"  Completion tokens:     {r['completion_tokens']}")
        print(f"  Reasoning preview:     {r.get('reasoning_preview')}")
        print(f"  Content preview:       {r.get('content_preview')}")


if __name__ == "__main__":
    main()
