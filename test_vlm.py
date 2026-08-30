"""
File: test_vlm.py
Purpose: Standalone VLM test — render a PDF page, send to Qwen2.5-VL via
         Ollama with STREAMING so tokens print as generated (no wall timeout).
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: requests, pymupdf
"""
import base64
import json
import sys
import os
import requests
import pymupdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL      = "qwen2.5vl:7b"
DPI        = 100   # lower = smaller image = fewer vision tokens = faster on CPU

TESTS = [
    {
        "label": "CamScanner handwritten (OR/LPP notes)",
        "pdf":   r"C:\Users\Dell\Desktop\IDP\engineer_a\tests\fixtures\camscanner_handwritten.pdf",
        "page":  0,
    },
    {
        "label": "SDG Goals (clean digital PDF)",
        "pdf":   r"C:\Users\Dell\Desktop\IDP\engineer_a\tests\fixtures\sdg_goals.pdf",
        "page":  0,
    },
]

PROMPT = """Analyze this document page image.

Return ONLY valid JSON — no markdown, no code fences:

{
  "page_type": "digital | scanned | handwritten | mixed",
  "handwriting": true,
  "tables": false,
  "complex_layout": false,
  "noise_level": 0.0,
  "reason": "one short sentence"
}

Rules:
- page_type: digital (selectable text), scanned (printed but rasterised),
             handwritten (manuscript), mixed (combination)
- handwriting: true if ANY handwritten strokes visible
- tables: true if a grid/tabular structure is present
- complex_layout: true if multiple columns, mixed regions, non-linear order
- noise_level: 0.0 (clean) to 1.0 (very degraded)
- reason: PAGE CHARACTERISTICS ONLY — never extract names, numbers, amounts
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_page(pdf_path: str, page_index: int, dpi: int = DPI) -> str:
    doc  = pymupdf.open(pdf_path)
    page = doc[page_index]
    mat  = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    out  = f"_vlm_test_p{page_index}.png"
    pix.save(out)
    doc.close()
    size_kb = os.path.getsize(out) // 1024
    print(f"  Rendered → {out}  ({size_kb} KB  @ {dpi} DPI)")
    return out


def call_vlm_streaming(image_path: str) -> str:
    """
    Call Ollama with stream=True and print tokens as they arrive.
    Returns the full response text when done.
    No wall-clock timeout — we see progress token by token.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model":  MODEL,
        "prompt": PROMPT,
        "images": [b64],
        "stream": True,
    }

    print(f"  Calling {MODEL} (streaming)...")
    print("  " + "-" * 55)
    print("  ", end="", flush=True)

    full_text = ""
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=None) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_text += token
            if chunk.get("done"):
                break

    print()   # newline after streaming output
    return full_text.strip()


def parse_json_response(raw: str) -> dict:
    # Strip markdown fences if the model adds them despite instructions
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": "model did not return valid JSON"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # Verify Ollama is up
    try:
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        print(f"\nOllama running  — models: {models}")
    except Exception as e:
        print(f"\nERROR: Ollama not reachable: {e}")
        sys.exit(1)

    if MODEL not in models:
        print(f"ERROR: {MODEL} not found. Pull it with: ollama pull {MODEL}")
        sys.exit(1)

    print(f"Model  : {MODEL}")
    print(f"DPI    : {DPI}  (lower = faster on CPU)\n")

    for test in TESTS:
        label  = test["label"]
        pdf    = test["pdf"]
        page_i = test["page"]

        if not os.path.exists(pdf):
            print(f"\n[SKIP] PDF not found: {pdf}\n")
            continue

        print(f"{'='*65}")
        print(f"  {label}")
        print(f"  PDF  : {os.path.basename(pdf)}  (page {page_i + 1})")
        print(f"{'='*65}")

        png = render_page(pdf, page_i)
        raw = call_vlm_streaming(png)
        os.remove(png)

        print()
        result = parse_json_response(raw)

        if "parse_error" in result:
            print(f"  [WARN] Could not parse JSON. Raw:\n  {result.get('raw_response','')[:300]}")
        else:
            print(f"  {'─'*45}")
            print(f"  page_type      : {result.get('page_type','?')}")
            print(f"  handwriting    : {result.get('handwriting','?')}")
            print(f"  tables         : {result.get('tables','?')}")
            print(f"  complex_layout : {result.get('complex_layout','?')}")
            print(f"  noise_level    : {result.get('noise_level','?')}")
            print(f"  reason         : {result.get('reason','?')}")
        print()


if __name__ == "__main__":
    run()
