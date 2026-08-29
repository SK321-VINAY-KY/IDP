"""
Prompt construction shared across providers, and deterministic fallback
text used if a provider call fails or declines to compose a reply itself
(neither per-field question phrasing nor the overall reply should ever be
a hard dependency on the LLM being up).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

EXTRACTION_SYSTEM_PROMPT = """You are the natural-language-understanding layer of a document schema \
builder having an ONGOING, open-ended conversation with a user about the \
schema they want. You do NOT control what actually happens to the schema \
and you must NOT invent information the user did not say - but unlike a \
rigid interview, you are shown the ENTIRE current schema, every field still \
missing information, and any validation problems on every single turn, and \
you should use as much of the user's message as applies, in one pass. A \
message might name the document type, add three fields, fix another \
field's type, remove a fourth, and answer two open questions - extract all \
of it, not just the first thing you notice.

Respond with STRICT JSON matching this shape and nothing else - no \
markdown fences, no commentary:

{
  "document_type": string or null,
  "operations": [
    {"op": "add" | "update" | "remove", "field_name": string,
     "type": string or null, "required": bool or null,
     "currency": string or null, "item_type": string or null,
     "pattern": string or null, "description": string or null}
  ],
  "confirmation": true, false, or null,
  "reply": string or null,
  "needs_clarification": bool,
  "clarification_reason": string or null
}

Rules:
- "type" must be one of: string, number, integer, boolean, date, object,
  array - or your best guess in the user's own words if none of those fit;
  unrecognized values are normalized or safely dropped downstream, so
  don't withhold an operation just because you're unsure of the exact
  type keyword.
- Use "op": "add" for a field that doesn't exist in "schema_so_far" yet,
  "update" to change an attribute of a field that already exists (e.g.
  answering an open gap, or a correction like "make total a number"), and
  "remove" for "remove X" / "drop X" / "delete X". Only set the fields on
  an operation that this message actually gives you - leave the rest null.
- "confirmation": true/false ONLY when the user is clearly approving or
  rejecting the schema as a whole (e.g. after you've shown it back to
  them) - not for answering an individual yes/no gap like "is this always
  present?".
- "reply" is what gets shown to the user verbatim, so write it as the
  actual next thing you'd say in this conversation - acknowledge what you
  just changed if anything, then either ask about the single most useful
  remaining gap/error, or (if the schema has no open gaps and no
  validation errors) show a brief summary and ask the user to confirm.
  Keep it natural and conversational, not a template. Leave "reply" null
  only if you genuinely can't produce one (use "needs_clarification"
  instead in that case).
- If the user's message is genuinely ambiguous and you cannot confidently
  extract structured data, set "needs_clarification": true and explain why
  in "clarification_reason" - still fill in "reply" with a clarifying
  question if you can.
- Never fabricate fields, attributes, or a confirmation the user did not
  actually state.
"""


def build_extraction_user_prompt(state: str, user_message: str, context: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "conversation_state": state,
            "schema_so_far": context.get("schema"),
            "document_type_missing": context.get("document_type_missing", False),
            # ALL open gaps, not just one - this is what lets a single reply
            # address several of them at once.
            "open_gaps": context.get("gaps", []),
            "validation_errors": context.get("validation_errors", []),
            "user_message": user_message,
        },
        indent=2,
    )


def fallback_question(field_name: str, attribute: str) -> str:
    """Deterministic template used if the LLM phrasing/reply call fails."""
    templates = {
        "type": f"What kind of value is '{field_name}' — text, a number, a date, or something else?",
        "required": f"Is '{field_name}' always present in the document, or can it be missing sometimes?",
        "item_type": f"'{field_name}' can have multiple values — what type is each individual value?",
    }
    return templates.get(attribute, f"Can you tell me more about '{field_name}' ({attribute})?")


def greeting_message() -> str:
    """Opens the one REVIEW loop - replaces the old fixed
    ASK_DOCUMENT_TYPE-only opener so the very first turn can already accept
    a document type, a field list, or both at once."""
    return (
        "Let's build your schema. You can upload a few sample documents, or "
        "just tell me about them - what type of documents are these, and "
        "what would you like to extract?"
    )


def document_type_question() -> str:
    return "What type of documents are you processing? (e.g. invoices, insurance claims, receipts)"


def fields_question() -> str:
    return (
        "What information would you like to extract? For example: customer name, "
        "account number, date, amount."
    )


def confirmation_prompt(summary: str) -> str:
    return f"Here's what I'll extract:\n\n{summary}\n\nIs this correct?"


# ---- document-intake (upload N samples -> propose a schema in one shot) ----

DOCUMENT_INFERENCE_SYSTEM_PROMPT = """You are the natural-language-understanding layer of a document schema \
builder. You are being shown 2-5 sample documents that are all the SAME \
document type (e.g. all invoices, all insurance claims). You do NOT control \
the conversation flow - you only propose a first-pass schema. Depending on \
the provider, the samples are given to you either as attached documents or \
as pre-OCR'd text, one per sample.

Compare the samples and respond with STRICT JSON matching this shape and \
nothing else - no markdown fences, no commentary:

{
  "document_type": string or null,
  "fields": [
    {
      "name": string,
      "type": string or null,
      "required": bool or null,
      "item_type": string or null,
      "pattern": string or null,
      "currency": string or null,
      "seen_in_samples": integer,
      "notes": string or null
    }
  ]
}

Rules:
- "type" must be one of: string, number, integer, boolean, date, object, array,
  or null if you are not confident.
- "seen_in_samples" is the number of the given samples this field actually
  appeared in - count honestly, don't assume it's in all of them.
- If the samples DISAGREE about a field's type or formatting (e.g. one has
  a plain number and another has a currency symbol, or the field is only
  present in some samples), set "type" and/or "required" to null for that
  field rather than guessing, and briefly explain the disagreement in
  "notes". This schema then lands in the same open REVIEW conversation as
  the text interview, so anything left null just becomes a normal thing to
  ask the user about there.
- Only set "required": true if the field appears in EVERY sample; only set
  it false if it's consistently present in some but genuinely absent (not
  just illegible) in others; leave it null if you're unsure.
- Never invent a field that doesn't appear in at least one sample.
- List each field once, using a normalized lowercase_with_underscores name.
"""


def build_document_inference_prompt(document_texts: List[str]) -> str:
    """Used by text-only providers (e.g. Sarvam, after OCR'ing each sample)."""
    return json.dumps(
        {
            "num_samples": len(document_texts),
            "samples": [{"sample_index": i + 1, "text": t} for i, t in enumerate(document_texts)],
        },
        indent=2,
    )


def document_inference_user_text(num_samples: int) -> str:
    """Used alongside attached-document content blocks (e.g. Bedrock)."""
    return (
        f"Here are {num_samples} sample documents, all of the same document type. "
        "Compare them and propose a schema as instructed in the system prompt."
    )


def document_intake_intro() -> str:
    return "I read through your sample documents. "


def document_intake_failed_message(failure_reason: Optional[str] = None) -> str:
    reason = f" ({failure_reason})" if failure_reason else ""
    return f"I couldn't confidently read those documents{reason}. Let's do this by chat instead - " + document_type_question()
