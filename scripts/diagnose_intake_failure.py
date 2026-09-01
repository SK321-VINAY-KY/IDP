"""
Diagnostic script to root-cause document intake failures in schema_chatbot_v2.

Tests:
  - Repro files: dataset/170.pdf, dataset/168.pdf
  - Control files: dataset/Vinay_Resume.pdf, dataset/MY_resume.pdf

Calls real SarvamAdapter._digitise() and Sarvam chat completion endpoints
with instrumentation to track:
  1. OCR job status, returned text length, preview
  2. build_document_inference_prompt() character length
  3. Real / approximated token counts (tiktoken cl100k_base)
  4. Full raw chat completions API response JSON (usage block, finish reason, raw content)
  5. Parsed document_type (null check)
  6. Final side-by-side summary table
"""
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

from app.llm.sarvam_adapter import SarvamAdapter
from app.llm.prompts import (
    DOCUMENT_INFERENCE_SYSTEM_PROMPT,
    build_document_inference_prompt,
)

# Optional tiktoken for token count approximation
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    enc = None


def get_token_count_approx(text: str) -> str:
    if enc is not None:
        count = len(enc.encode(text))
        return f"{count} (tiktoken cl100k_base approximation)"
    return f"{len(text) // 4} (estimated chars/4)"


def safe_preview(text: str, n: int = 300) -> str:
    """Safely format string preview without failing on non-printable or surrogate characters."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def main():
    print("=" * 80)
    print("DIAGNOSTIC RUN: Schema Chatbot Document Intake Failure Investigation")
    print("=" * 80)

    adapter = SarvamAdapter()
    print(f"Sarvam Model: {adapter.model}")
    print(f"Base URL:     {adapter.base_url}")
    print(f"Timeout:      {adapter.timeout_s}s")
    print(f"API Key set:  {bool(adapter.api_key)}")
    print("-" * 80)

    # Instrumentation around Sarvam Document AI client to capture OCR job status
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
        ("Repro 1", ROOT / "dataset" / "170.pdf"),
        ("Repro 2", ROOT / "dataset" / "168.pdf"),
        ("Control 1", ROOT / "dataset" / "Vinay_Resume.pdf"),
        ("Control 2", ROOT / "dataset" / "MY_resume.pdf"),
    ]

    summary_rows: List[Dict[str, Any]] = []

    for label, pdf_path in files_to_test:
        print(f"\n{'#' * 80}")
        print(f"TESTING [{label}]: {pdf_path.name}")
        print(f"Path: {pdf_path}")
        print(f"{'#' * 80}")

        if not pdf_path.exists():
            print(f"ERROR: File does not exist: {pdf_path}")
            summary_rows.append({
                "filename": pdf_path.name,
                "ocr_status": "FILE_NOT_FOUND",
                "ocr_len": 0,
                "prompt_tokens": "N/A",
                "completion_tokens": "N/A",
                "document_type": "N/A",
            })
            continue

        pdf_bytes = pdf_path.read_bytes()
        print(f"File size: {len(pdf_bytes)} bytes")

        # -------------------------------------------------------------
        # STEP 1: SarvamAdapter._digitise()
        # -------------------------------------------------------------
        print("\n--- STEP 1: Running SarvamAdapter._digitise() ---")
        ocr_status = "unknown"
        ocr_text = ""
        last_job_id[0] = None

        t0 = time.monotonic()
        try:
            ocr_text = adapter._digitise(1, pdf_bytes)
            job_id = last_job_id[0]
            ocr_status = recorded_statuses.get(job_id, "completed")
            print(f"OCR Job ID: {job_id}")
            print(f"OCR Job Status: {ocr_status}")
            print(f"Raw returned text length: {len(ocr_text)} characters")
            print(f"Time taken: {round(time.monotonic() - t0, 2)}s")

            # Text preview (first 300 chars + last 300 chars)
            first_300 = safe_preview(ocr_text[:300])
            last_300 = safe_preview(ocr_text[-300:]) if len(ocr_text) > 300 else first_300
            print("\n[Preview First 300 chars]:")
            print(first_300)
            print("\n[Preview Last 300 chars]:")
            print(last_300)

        except Exception as exc:
            job_id = last_job_id[0]
            ocr_status = recorded_statuses.get(job_id, "failed")
            print(f"ERROR during _digitise: {exc}")
            print(f"OCR Job ID: {job_id}")
            print(f"OCR Job Status: {ocr_status}")
            summary_rows.append({
                "filename": pdf_path.name,
                "ocr_status": ocr_status,
                "ocr_len": 0,
                "prompt_tokens": "N/A",
                "completion_tokens": "N/A",
                "document_type": f"ERROR: {exc}",
            })
            continue

        # -------------------------------------------------------------
        # STEP 2: Build actual user_prompt via build_document_inference_prompt()
        # -------------------------------------------------------------
        print("\n--- STEP 2: Building user_prompt ---")
        # Cap at 3,500 characters exactly as infer_schema_from_pdfs does
        capped_text = ocr_text[:3500] if len(ocr_text) > 3500 else ocr_text
        user_prompt = build_document_inference_prompt([capped_text])
        sys_prompt = (
            DOCUMENT_INFERENCE_SYSTEM_PROMPT
            + "\n\nCRITICAL: Keep reasoning brief. Output ONLY the valid JSON object with keys 'document_type' and 'fields'. Do not wrap in commentary."
        )

        print(f"Original text length: {len(ocr_text)}")
        print(f"Capped text length:   {len(capped_text)} (capped at 3500)")
        print(f"User prompt char len: {len(user_prompt)}")
        print(f"Sys prompt char len:  {len(sys_prompt)}")

        # -------------------------------------------------------------
        # STEP 3: Token counts
        # -------------------------------------------------------------
        print("\n--- STEP 3: Token Counts ---")
        combined_text = sys_prompt + "\n\n" + user_prompt
        token_count_str = get_token_count_approx(combined_text)
        print(f"Calculated token count (sys_prompt + user_prompt): {token_count_str}")

        # -------------------------------------------------------------
        # STEP 4: Call Sarvam chat completions endpoint
        # -------------------------------------------------------------
        print("\n--- STEP 4: Calling Sarvam chat completions endpoint ---")
        payload = {
            "model": adapter.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
            "reasoning_effort": "low",
        }

        resp_json: Dict[str, Any] = {}
        raw_content = ""
        prompt_tokens_api: Any = "N/A"
        completion_tokens_api: Any = "N/A"
        total_tokens_api: Any = "N/A"

        t_chat0 = time.monotonic()
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
            chat_elapsed = round(time.monotonic() - t_chat0, 2)
            print(f"HTTP Status: {resp.status_code} ({chat_elapsed}s)")

            if not resp.is_success:
                print(f"Error Response Body: {resp.text}")
                resp.raise_for_status()

            resp_json = resp.json()
            print("\n[FULL RAW RESPONSE JSON]:")
            print(json.dumps(resp_json, indent=2, ensure_ascii=False))

            # Usage block
            usage = resp_json.get("usage", {})
            prompt_tokens_api = usage.get("prompt_tokens", "N/A")
            completion_tokens_api = usage.get("completion_tokens", "N/A")
            total_tokens_api = usage.get("total_tokens", "N/A")
            print(f"\n[USAGE BLOCK]: prompt_tokens={prompt_tokens_api}, completion_tokens={completion_tokens_api}, total_tokens={total_tokens_api}")

            # Extract content exactly as SarvamAdapter._chat does
            choices = resp_json.get("choices", [])
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                print(f"Finish reason: {finish_reason}")
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                print(f"message.content length: {len(content)}")
                print(f"message.reasoning_content length: {len(reasoning)}")

                if not content:
                    raw_content = reasoning
                elif "{" in reasoning and "{" not in content:
                    raw_content = reasoning
                else:
                    raw_content = content

            print("\n[RAW CONTENT STRING (passed to _parse_json)]:")
            print(raw_content)

        except Exception as exc:
            print(f"ERROR during chat completion: {exc}")
            summary_rows.append({
                "filename": pdf_path.name,
                "ocr_status": ocr_status,
                "ocr_len": len(ocr_text),
                "prompt_tokens": prompt_tokens_api,
                "completion_tokens": completion_tokens_api,
                "document_type": f"CHAT_ERROR: {exc}",
            })
            continue

        # -------------------------------------------------------------
        # STEP 5: Parse raw content and check document_type
        # -------------------------------------------------------------
        print("\n--- STEP 5: Parsing raw content ---")
        doc_type_val = None
        is_null = True
        try:
            parsed = adapter._parse_json(raw_content)
            print(f"Parsed JSON keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
            if isinstance(parsed, dict):
                doc_type_val = parsed.get("document_type")
                is_null = doc_type_val is None
                fields_count = len(parsed.get("fields", []))
                print(f"document_type: {repr(doc_type_val)}")
                print(f"document_type is null: {is_null}")
                print(f"fields count: {fields_count}")
            else:
                print(f"Parsed output is not a dict: {parsed}")
        except Exception as exc:
            print(f"ERROR parsing raw content via _parse_json: {exc}")
            doc_type_val = f"PARSE_ERROR: {exc}"

        summary_rows.append({
            "filename": pdf_path.name,
            "ocr_status": ocr_status,
            "ocr_len": len(ocr_text),
            "prompt_tokens": prompt_tokens_api,
            "completion_tokens": completion_tokens_api,
            "document_type": repr(doc_type_val) if not isinstance(doc_type_val, str) or not doc_type_val.startswith("PARSE_ERROR") else doc_type_val,
        })

    # -------------------------------------------------------------
    # STEP 6: Side-by-side Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print("FINAL SIDE-BY-SIDE SUMMARY TABLE")
    print("=" * 110)
    header = f"{'filename':<22} | {'OCR status':<20} | {'OCR text length':<15} | {'prompt tokens (actual)':<22} | {'completion tokens':<17} | {'document_type result'}"
    print(header)
    print("-" * 110)
    for row in summary_rows:
        line = (
            f"{row['filename']:<22} | "
            f"{str(row['ocr_status']):<20} | "
            f"{str(row['ocr_len']):<15} | "
            f"{str(row['prompt_tokens']):<22} | "
            f"{str(row['completion_tokens']):<17} | "
            f"{row['document_type']}"
        )
        print(line)
    print("=" * 110)


if __name__ == "__main__":
    main()
