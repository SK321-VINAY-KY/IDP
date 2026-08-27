"""
Prompt construction shared across providers, and deterministic fallback
question templates used if a provider call fails (per-field question
phrasing should never be a hard dependency on the LLM being up).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

EXTRACTION_SYSTEM_PROMPT = """You are the natural-language-understanding layer of a document schema \
builder. You do NOT control the conversation flow and you must NOT invent \
information the user did not say.

Given the conversation state, the schema built so far, and the user's latest \
message, extract ONLY what the user actually stated. Respond with STRICT JSON \
matching this shape and nothing else - no markdown fences, no commentary:

{
  "document_type": string or null,
  "new_fields": [ {"name": string, "type": string or null, "required": bool or null,
                    "currency": string or null, "item_type": string or null} ],
  "field_answers": [ {"field_name": string, "attribute": string, "value": any} ],
  "removals": [string, ...],
  "confirmation": true, false, or null,
  "needs_clarification": bool,
  "clarification_reason": string or null
}

Rules:
- "type" must be one of: string, number, integer, boolean, date, object, array.
- Only set "confirmation" when the user is responding to a yes/no confirmation prompt.
- Only include "field_answers" entries when the user is directly answering a
  question about a SPECIFIC field attribute that was just asked about.
- If the user says things like "remove X" or "drop X", put X's normalized
  field name in "removals".
- If the user's message is genuinely ambiguous and you cannot confidently
  extract structured data, set "needs_clarification": true and explain why.
- Never fabricate fields or attributes the user did not mention.
"""


def build_extraction_user_prompt(state: str, user_message: str, context: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "conversation_state": state,
            "schema_so_far": context.get("schema"),
            "current_gap": context.get("current_gap"),
            "user_message": user_message,
        },
        indent=2,
    )


def fallback_question(field_name: str, attribute: str) -> str:
    """Deterministic template used if the LLM phrasing call fails."""
    templates = {
        "type": f"What kind of value is '{field_name}' — text, a number, a date, or something else?",
        "required": f"Is '{field_name}' always present in the document, or can it be missing sometimes?",
        "item_type": f"'{field_name}' can have multiple values — what type is each individual value?",
    }
    return templates.get(attribute, f"Can you tell me more about '{field_name}' ({attribute})?")


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
  "notes". A later step will ask the user directly about anything left null.
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
    return "I read through your sample documents. A few things need confirming:\n\n"


def document_intake_failed_message(failure_reason: Optional[str] = None) -> str:
    reason = f" ({failure_reason})" if failure_reason else ""
    return f"I couldn't confidently read those documents{reason}. Let's do this by chat instead - " + document_type_question()
