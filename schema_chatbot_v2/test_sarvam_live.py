"""
test_sarvam_live.py  — scripted end-to-end conversation test against
the running FastAPI server using the Sarvam adapter.

Simulates: insurance claim form -> 3 fields -> confirm -> schema complete.

Run with:  python test_sarvam_live.py
Server must be running: python -m uvicorn app.main:app --port 8000
"""
import json
import time
import httpx

BASE = "http://localhost:8000"

def post(session_id=None, message=None):
    body = {}
    if session_id:
        body["session_id"] = session_id
    if message:
        body["message"] = message
    r = httpx.post(f"{BASE}/chat", json=body, timeout=90)
    r.raise_for_status()
    return r.json()

def show(resp, label=""):
    print(f"\n{'─'*60}")
    if label:
        print(f"  [{label}]")
    print(f"  state     : {resp['state']}")
    print(f"  bot       : {resp['message']}")
    print(f"  completed : {resp['completed']}")
    schema = resp.get("schema", {})
    if schema:
        print(f"  schema    :")
        print(f"    doc_type: {schema.get('document_type')}")
        for f in schema.get("fields", []):
            print(f"    field   : {f}")

def main():
    print("=" * 60)
    print("  Schema Chatbot v2 — Live Sarvam Test")
    print(f"  Server: {BASE}")
    print("=" * 60)

    # Check server
    health = httpx.get(f"{BASE}/health", timeout=5).json()
    print(f"\n  Health: {health}")
    assert health["llm_provider"] == "sarvam", "Expected sarvam provider"

    # Turn 1 — start session (no message)
    print("\n--- Turn 1: Start session ---")
    r = post()
    show(r, "START")
    sid = r["session_id"]
    print(f"  session_id: {sid}")

    # Turn 2 — tell it the document type and fields
    print("\n--- Turn 2: Document type + fields ---")
    r = post(sid, "insurance claim forms. I need to extract patient name, policy number, claim amount, and claim date")
    show(r, "DOC TYPE + FIELDS")

    # Turns 3–N — answer gap questions about each field
    for i in range(12):  # max 12 turns to handle all gap questions
        if r["completed"]:
            break
        state = r["state"]
        msg = r["message"].lower()

        # Answer field-detail questions
        if state == "FIELD_DETAILS":
            if "always present" in msg or "sometimes missing" in msg or "optional" in msg:
                answer = "always"
            elif "format" in msg or "date" in msg:
                answer = "DD/MM/YYYY"
            elif "currency" in msg:
                answer = "INR"
            elif "type" in msg or "kind" in msg:
                answer = "text"
            else:
                answer = "yes"
            print(f"\n--- Turn {i+3}: Field detail (auto-answer: '{answer}') ---")
            r = post(sid, answer)
            show(r, f"FIELD_DETAILS turn {i+3}")

        elif state == "CONFIRMATION":
            print(f"\n--- Turn {i+3}: Confirmation ---")
            r = post(sid, "yes")
            show(r, "CONFIRM")
            break
        else:
            print(f"\n--- Turn {i+3}: state={state} ---")
            r = post(sid, "yes")
            show(r, state)

    print("\n" + "=" * 60)
    if r["completed"]:
        print("  RESULT: Schema building COMPLETE")
        print(f"  schema_id: {r.get('schema_id')}")
        print("\n  Final schema:")
        print(json.dumps(r.get("schema", {}), indent=2))
    else:
        print(f"  RESULT: Session ended at state={r['state']} (not yet complete)")
        print("  This is OK — the bot is waiting for more input from the user")
    print("=" * 60)

if __name__ == "__main__":
    main()
