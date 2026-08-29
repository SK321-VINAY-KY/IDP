"""
test_corrections.py  — three targeted correction scenarios against the
live Sarvam-backed chatbot.

Test 1 — TYPO CORRECTION
  User types a field name wrong ("naem"), bot adds it,
  user says "its not naem its name", bot renames it.

Test 2 — REQUIRED → OPTIONAL
  User says a field is required, then says "make that optional",
  bot flips the flag.

Test 3 — DROP A FIELD MID-SESSION
  Bot extracts a field the LLM inferred (e.g. it guessed "address"),
  user says "that field is not necessary, drop it",
  bot removes it cleanly.

Each test is an independent session — failures don't affect each other.
Run with:  python test_corrections.py
Server must be running on port 8000.
"""
import json
import httpx

BASE  = "http://localhost:8000"
SEP   = "─" * 62
PASS  = "✓ PASS"
FAIL  = "✗ FAIL"


# ── helpers ──────────────────────────────────────────────────────────

def chat(session_id=None, message=None):
    body = {}
    if session_id:
        body["session_id"] = session_id
    if message:
        body["message"] = message
    r = httpx.post(f"{BASE}/chat", json=body, timeout=90)
    r.raise_for_status()
    return r.json()


def field_names(resp) -> list:
    return [f["name"] for f in resp.get("schema", {}).get("fields", [])]


def field_attr(resp, field_name, attr):
    for f in resp.get("schema", {}).get("fields", []):
        if f["name"] == field_name:
            return f.get(attr)
    return None


def show_turn(label, user_said, resp):
    print(f"\n  [{label}]")
    print(f"  YOU : {user_said}")
    print(f"  BOT : {resp['message']}")
    fields = resp.get("schema", {}).get("fields", [])
    print(f"  fields ({len(fields)}): " + ", ".join(
        f"{f['name']}({'req' if f.get('required') else 'opt'})"
        for f in fields
    ))


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── Test 1 — Typo correction ─────────────────────────────────────────

def test_typo_correction():
    section("TEST 1 — Typo correction  (naem → name → nmae)")

    r = chat()
    sid = r["session_id"]

    # Step 1: give document + field with typo "naem"
    msg = "these are employee records. fields: naem, email, department"
    r = chat(sid, msg)
    show_turn("after initial fields", msg, r)

    names_after = field_names(r)
    has_typo = "naem" in names_after
    print(f"\n  Check: 'naem' added to schema  → {PASS if has_typo else FAIL}  (fields: {names_after})")

    # Step 2: correct the typo — natural language, first correction style
    msg2 = "its not naem its name"
    r = chat(sid, msg2)
    show_turn("after typo fix 1", msg2, r)

    names_after2 = field_names(r)
    fixed1 = "name" in names_after2 and "naem" not in names_after2
    print(f"  Check: 'naem' → 'name' corrected → {PASS if fixed1 else FAIL}  (fields: {names_after2})")

    # Step 3: introduce a second typo "nmae" to test a different phrasing
    msg3 = "also add nmae as a backup display field"
    r = chat(sid, msg3)
    show_turn("after adding 'nmae'", msg3, r)

    # Step 4: correct it with the exact phrasing from the task spec
    msg4 = "its not nmae its display_name"
    r = chat(sid, msg4)
    show_turn("after typo fix 2", msg4, r)

    names_after4 = field_names(r)
    fixed2 = "display_name" in names_after4 and "nmae" not in names_after4
    print(f"  Check: 'nmae' → 'display_name' corrected → {PASS if fixed2 else FAIL}  (fields: {names_after4})")

    return fixed1 and fixed2


# ── Test 2 — Required → Optional flip ────────────────────────────────

def test_required_to_optional():
    section("TEST 2 — required → optional  ('make that field optional')")

    r = chat()
    sid = r["session_id"]

    # Step 1: define fields, explicitly say middle_name is required
    msg = "processing staff profiles. I need full_name (required), middle_name (required), employee_id (required)"
    r = chat(sid, msg)
    show_turn("after adding fields", msg, r)

    # Verify middle_name landed as required
    req_before = field_attr(r, "middle_name", "required")
    print(f"\n  Check: middle_name.required = True initially → {PASS if req_before is True else FAIL}  (got {req_before})")

    # Step 2: flip it — exact phrasing from the task spec
    msg2 = "make middle_name optional, not everyone has one"
    r = chat(sid, msg2)
    show_turn("after making optional", msg2, r)

    req_after = field_attr(r, "middle_name", "required")
    flipped = req_after is False
    print(f"  Check: middle_name.required flipped to False → {PASS if flipped else FAIL}  (got {req_after})")

    # Step 3: also test the shorter phrasing "that field is optional"
    msg3 = "actually employee_id should also be optional for contractors"
    r = chat(sid, msg3)
    show_turn("after second flip", msg3, r)

    req3 = field_attr(r, "employee_id", "required")
    flipped2 = req3 is False
    print(f"  Check: employee_id.required flipped to False → {PASS if flipped2 else FAIL}  (got {req3})")

    return flipped and flipped2


# ── Test 3 — Drop an inferred field ──────────────────────────────────

def test_drop_inferred_field():
    section("TEST 3 — Drop a field  ('that field is not necessary, drop it')")

    r = chat()
    sid = r["session_id"]

    # Step 1: give a document where the LLM is likely to infer extra fields.
    # "customer invoices" often makes it propose address, tax_id, etc.
    msg = "these are customer invoices. extract invoice_number, customer_name, amount, due_date"
    r = chat(sid, msg)
    show_turn("after initial fields", msg, r)

    fields_after_init = field_names(r)
    print(f"\n  Fields extracted: {fields_after_init}")

    # Step 2: check if any extra field was inferred — if so, drop it;
    # if not, we explicitly add one then drop it to still exercise the path
    expected = {"invoice_number", "customer_name", "amount", "due_date"}
    extras   = [f for f in fields_after_init if f not in expected]

    if extras:
        target = extras[0]
        print(f"  LLM inferred extra field: '{target}' — will ask bot to drop it")
    else:
        # Manually add a field first, then drop it
        add_msg = "also add notes field"
        r = chat(sid, add_msg)
        show_turn("adding 'notes' to drop", add_msg, r)
        target = "notes"
        print(f"  No extra field inferred — added '{target}' manually to test drop")

    # Step 3: drop it using the exact phrasing from the task spec
    msg3 = f"that {target} field is not necessary, drop it"
    r = chat(sid, msg3)
    show_turn(f"after dropping '{target}'", msg3, r)

    fields_after_drop = field_names(r)
    dropped = target not in fields_after_drop
    print(f"  Check: '{target}' removed from schema → {PASS if dropped else FAIL}  (fields: {fields_after_drop})")

    # Step 4: also test the shorter "remove X" phrasing
    msg4 = "remove due_date as well"
    r = chat(sid, msg4)
    show_turn("after removing 'due_date'", msg4, r)

    fields_after_remove = field_names(r)
    removed2 = "due_date" not in fields_after_remove
    print(f"  Check: 'due_date' removed via 'remove X' phrasing → {PASS if removed2 else FAIL}  (fields: {fields_after_remove})")

    return dropped and removed2


# ── main ──────────────────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═"*60 + "╗")
    print("║  Schema Chatbot v2 — Correction & Mutation Tests (Sarvam)" + " "*2 + "║")
    print("╚" + "═"*60 + "╝")

    # Verify server
    h = httpx.get(f"{BASE}/health", timeout=5).json()
    print(f"\n  Server  : {h}")
    assert h["llm_provider"] == "sarvam"

    results = {}
    for name, fn in [
        ("Typo correction",       test_typo_correction),
        ("Required → Optional",   test_required_to_optional),
        ("Drop inferred field",    test_drop_inferred_field),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  ERROR in '{name}': {exc}")
            results[name] = False

    print(f"\n{'='*62}")
    print("  SUMMARY")
    print(f"{'='*62}")
    for name, passed in results.items():
        print(f"  {PASS if passed else FAIL}  {name}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED — see details above'}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
