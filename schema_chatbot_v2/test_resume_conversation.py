"""
test_resume_conversation.py  — scripted resume schema conversation.

Simulates a real non-technical user describing a resume:
  - Gives document type + multiple fields in one message
  - Answers gap questions as they come
  - Makes a mid-session correction (removes a field)
  - Confirms the final schema

Run with:  python test_resume_conversation.py
Server must be running on port 8000.
"""
import json
import httpx

BASE = "http://localhost:8000"
SEP  = "─" * 65


def chat(session_id=None, message=None):
    body = {}
    if session_id:
        body["session_id"] = session_id
    if message:
        body["message"] = message
    r = httpx.post(f"{BASE}/chat", json=body, timeout=90)
    r.raise_for_status()
    return r.json()


def show(resp, user_said=None):
    print(SEP)
    if user_said:
        print(f"  YOU : {user_said}")
    print(f"  BOT : {resp['message']}")
    print(f"  state={resp['state']}  completed={resp['completed']}")
    schema = resp.get("schema", {})
    fields = schema.get("fields", [])
    if fields:
        print(f"  --- schema ({len(fields)} fields) ---")
        for f in fields:
            parts = [f["name"], f.get("type","?")]
            if f.get("required") is not None:
                parts.append("required" if f["required"] else "optional")
            if f.get("item_type"):
                parts.append(f"items={f['item_type']}")
            if f.get("currency"):
                parts.append(f"currency={f['currency']}")
            if f.get("pattern"):
                parts.append(f"pattern={f['pattern']}")
            print(f"    • {' | '.join(parts)}")
    print()


def main():
    print("=" * 65)
    print("  Schema Chatbot v2 — RESUME conversation test (Sarvam live)")
    print("=" * 65)

    # ── Turn 1: start session ────────────────────────────────────────
    r = chat()
    sid = r["session_id"]
    show(r, user_said=None)
    print(f"  session_id: {sid}\n")

    # ── Turn 2: document type + all fields upfront ───────────────────
    # Realistic: user names fields casually, mixed naming styles
    msg = (
        "these are resumes / CVs. "
        "I need: candidate name, email, phone number, "
        "current job title, years of experience, education (degree and institution), "
        "skills list, and linkedin url"
    )
    r = chat(sid, msg)
    show(r, user_said=msg)

    # ── Turns 3–N: answer gap questions automatically ─────────────────
    turn = 3
    while not r["completed"] and turn <= 20:
        state = r["state"]
        bot_msg = r["message"].lower()

        # Pick a sensible answer based on what the bot is asking
        if state == "FIELD_DETAILS":
            if "always present" in bot_msg or "optional" in bot_msg or "sometimes missing" in bot_msg or "required" in bot_msg:
                # Phone and LinkedIn are optional on resumes
                if any(w in bot_msg for w in ["phone", "linkedin", "url"]):
                    answer = "it can be missing — not everyone includes it"
                else:
                    answer = "always present"

            elif "type" in bot_msg or "kind of value" in bot_msg:
                if "experience" in bot_msg:
                    answer = "a number, like 5"
                elif "skill" in bot_msg:
                    answer = "a list of text items"
                else:
                    answer = "text"

            elif "format" in bot_msg or "pattern" in bot_msg:
                if "phone" in bot_msg:
                    answer = "+91-XXXXXXXXXX"
                elif "email" in bot_msg:
                    answer = "standard email format"
                elif "date" in bot_msg or "year" in bot_msg:
                    answer = "YYYY"
                else:
                    answer = "no specific format"

            elif "currency" in bot_msg:
                answer = "no currency, it's just a count"

            elif "items" in bot_msg or "list of what" in bot_msg or "each item" in bot_msg:
                answer = "text, each skill is a short string like Python or Excel"

            elif "degree" in bot_msg or "institution" in bot_msg or "nested" in bot_msg:
                answer = "yes it has sub-fields: degree name and institution name"

            else:
                answer = "yes"

        elif state == "CONFIRMATION":
            # First see the schema, then ask for a correction before confirming
            if turn == 3:
                # Haven't seen confirmation yet — this shouldn't happen turn 3
                answer = "yes"
            else:
                # Make a correction: remove linkedin_url, it's not needed
                answer = "actually remove the linkedin_url field, we don't need it"
                r = chat(sid, answer)
                show(r, user_said=answer)
                turn += 1
                # Now confirm for real
                answer = "yes that looks correct, confirm"

        else:
            answer = "yes"

        r = chat(sid, answer)
        show(r, user_said=answer)
        turn += 1

    # ── Final result ──────────────────────────────────────────────────
    print("=" * 65)
    if r["completed"]:
        print("  RESULT: Schema COMPLETED")
        print(f"  schema_id: {r.get('schema_id')}")
        print()
        print("  FULL FINAL SCHEMA:")
        print(json.dumps(r.get("schema", {}), indent=2))
    else:
        print(f"  RESULT: Reached turn {turn}, state={r['state']} — conversation ongoing")
        print("  (bot is waiting for the next user reply)")
        print()
        print("  SCHEMA SO FAR:")
        print(json.dumps(r.get("schema", {}), indent=2))
    print("=" * 65)


if __name__ == "__main__":
    main()
