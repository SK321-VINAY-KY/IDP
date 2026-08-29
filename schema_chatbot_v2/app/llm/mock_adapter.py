"""
A deterministic, rule-based 'LLM' used for tests and for running the whole
system with zero external dependencies. It parses simple patterns rather
than truly understanding language - good enough to exercise the entire
REVIEW-loop / schema-building / validation pipeline end-to-end.

This is what test_conversation_manager.py runs against, so the pipeline is
fully testable without Ollama, Bedrock, or Sarvam being reachable. It
deliberately handles a few "several things in one message" cases (a field
list plus a removal, or "make total a number") so the tests can actually
exercise the collapsed-state design's main point, not just prove the
plumbing works one gap at a time.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.llm.base import ExtractionResult, FieldObservation, FieldOp, LLMAdapter, SchemaProposal
from app.llm.prompts import fallback_question

_TYPE_WORDS = {
    "text": "string", "string": "string", "name": "string",
    "number": "number", "amount": "number", "total": "number", "float": "number",
    "date": "date",
    "boolean": "boolean", "yes/no": "boolean", "bool": "boolean",
    "integer": "integer", "count": "integer", "int": "integer",
}

_REQUIRED_TRUE_WORDS = ("always", "mandatory", "required")
_REQUIRED_FALSE_WORDS = ("sometimes", "optional", "not always", "not required")

_REMOVE_RE = re.compile(r"(?:remove|drop|delete)\s+([a-zA-Z0-9_ ]+?)(?=(?:,| and |$))", re.I)
_ADD_RE = re.compile(r"(?:also\s+)?(?:add|include)\s+(?:a|an)?\s*([a-zA-Z0-9_ ]+?)(?=(?:,| and |$))", re.I)
_RETYPE_RE = re.compile(
    r"(?:make|set)\s+([a-zA-Z0-9_ ]+?)\s+(?:a|an|to)?\s*"
    r"(string|number|integer|boolean|date|object|array|text|float|int|bool)\b", re.I,
)
_REQUIRED_RE = re.compile(
    r"([a-zA-Z0-9_ ]+?)\s+is\s+(always|mandatory|required|sometimes|optional|not always|not required)", re.I,
)


class MockLLMAdapter(LLMAdapter):
    def extract(self, state: str, user_message: str, context: Dict[str, Any]) -> ExtractionResult:
        text = user_message.strip()
        lower = text.lower()
        operations: List[FieldOp] = []
        remainder = lower

        # 1. Removals - can co-occur with anything else in the message.
        for name in _REMOVE_RE.findall(lower):
            operations.append(FieldOp(op="remove", field_name=name.strip()))
            remainder = _REMOVE_RE.sub("", remainder)

        # 2. Additions - explicit "also add a due date", "add invoice_id"
        for name in _ADD_RE.findall(remainder):
            clean_name = name.strip()
            operations.append(FieldOp(op="add", field_name=clean_name, **self._guess_attrs(clean_name)))
            remainder = _ADD_RE.sub("", remainder)

        # 3. Explicit re-type corrections ("make total a number").
        for field_name, raw_type in _RETYPE_RE.findall(remainder):
            operations.append(FieldOp(op="update", field_name=field_name.strip(), type=raw_type))
            remainder = _RETYPE_RE.sub("", remainder)

        # 4. Explicit required corrections ("invoice_number is always present").
        for field_name, word in _REQUIRED_RE.findall(remainder):
            operations.append(FieldOp(op="update", field_name=field_name.strip(), required=word in _REQUIRED_TRUE_WORDS))
            remainder = _REQUIRED_RE.sub("", remainder)

        # 5. Confirmation - only meaningful once nothing else is open.
        stripped = remainder.strip(" .!")
        if stripped in ("yes", "y", "correct", "yep", "confirmed"):
            return ExtractionResult(confirmation=True, operations=operations)
        if stripped in ("no", "n", "nope", "incorrect"):
            return ExtractionResult(confirmation=False, operations=operations)

        # 6. Document type - only when it's still missing and this message
        #    doesn't look like a field list (comma/"and"-separated).
        document_type = None
        if context.get("document_type_missing") and stripped and not self._looks_like_field_list(stripped):
            document_type = text.strip(" .")
            remainder = ""

        # 7. Try to answer open gaps directly (e.g. "text, always", "number", "always").
        gaps = context.get("gaps") or []
        if remainder.strip(" .") and gaps and not document_type:
            answered = self._answer_gaps(remainder, gaps)
            if answered:
                operations.extend(answered)
                remainder = ""

        # 8. Field list ("invoice number, vendor name and total amount") or single field.
        if remainder.strip(" .") and not document_type:
            if self._looks_like_field_list(remainder):
                for name in self._split_field_list(remainder):
                    operations.append(FieldOp(op="add", field_name=name, **self._guess_attrs(name)))
                remainder = ""
            elif not context.get("document_type_missing"):
                field_name = remainder.strip(" .")
                operations.append(FieldOp(op="add", field_name=field_name, **self._guess_attrs(field_name)))
                remainder = ""
            elif not operations:
                return ExtractionResult(needs_clarification=True, clarification_reason="could not parse that")

        return ExtractionResult(document_type=document_type, operations=operations)

    def phrase_question(self, gap_field: str, gap_attribute: str, context: Dict[str, Any]) -> str:
        return fallback_question(gap_field, gap_attribute)

    def infer_schema_from_pdfs(self, samples: List[bytes]) -> SchemaProposal:
        """
        No real PDF/OCR support here (this adapter has zero external
        dependencies, by design). Instead, each "sample" is expected to be
        plain UTF-8 text encoded as bytes, in this tiny deterministic format,
        so tests can exercise the whole document-intake flow offline:

            document_type: invoice
            policy_number: string: yes
            amount: number: yes
            notes: string: no

        (one `name: type: yes|no` line per field; `type`/the yes-no token
        may be blank to simulate a field the mock "isn't sure" about)
        """
        total = len(samples)
        doc_types: List[str] = []
        field_data: Dict[str, Dict[str, Any]] = {}

        for raw in samples:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return SchemaProposal(
                    extraction_failed=True,
                    failure_reason="mock adapter expects plain-text samples, not real PDF bytes",
                )

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("document_type:"):
                    doc_types.append(line.split(":", 1)[1].strip())
                    continue

                parts = [p.strip() for p in line.split(":")]
                if not parts or not parts[0]:
                    continue
                name = parts[0].lower().replace(" ", "_").replace("-", "_")
                ftype = parts[1] if len(parts) > 1 and parts[1] else None
                required_token = parts[2].lower() if len(parts) > 2 and parts[2] else None
                required = None
                if required_token in ("yes", "true", "required"):
                    required = True
                elif required_token in ("no", "false", "optional"):
                    required = False

                entry = field_data.setdefault(name, {"types": set(), "requireds": set(), "count": 0})
                entry["count"] += 1
                if ftype:
                    entry["types"].add(ftype)
                if required is not None:
                    entry["requireds"].add(required)

        fields = []
        for name, data in field_data.items():
            types, requireds = data["types"], data["requireds"]
            inconsistent = len(types) > 1 or len(requireds) > 1
            fields.append(
                FieldObservation(
                    name=name,
                    type=next(iter(types)) if len(types) == 1 else None,
                    required=next(iter(requireds)) if len(requireds) == 1 else None,
                    seen_in_samples=data["count"],
                    total_samples=total,
                    notes="inconsistent across samples" if inconsistent else None,
                )
            )

        document_type = doc_types[0] if doc_types else None
        return SchemaProposal(document_type=document_type, fields=fields)

    # ---- helpers ----

    @staticmethod
    def _looks_like_field_list(text: str) -> bool:
        return "," in text or re.search(r"\band\b", text, flags=re.I) is not None

    @staticmethod
    def _split_field_list(text: str) -> List[str]:
        text = re.sub(r"\band\b", ",", text, flags=re.I)
        parts = [p.strip(" .") for p in text.split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _guess_attrs(name: str) -> Dict[str, Any]:
        return {}

    @classmethod
    def _answer_gaps(cls, lower: str, gaps: List[Dict[str, str]]) -> List[FieldOp]:
        """
        Answers as many of the given open gaps as this one message plausibly
        speaks to - not just the first one - so a reply like "text, always"
        can answer both a type gap and a required gap in a single turn
        instead of needing two round trips. Matches at most one type-shaped
        gap and one required gap per message (the mock can't disambiguate
        which of two same-attribute gaps a bare word like "text" refers to,
        so it doesn't guess).
        """
        type_gap = next((g for g in gaps if g["attribute"] in ("type", "item_type")), None)
        required_gap = next((g for g in gaps if g["attribute"] == "required"), None)

        type_val = None
        for word, t in _TYPE_WORDS.items():
            if word in lower:
                type_val = t
                break

        required_val = None
        if any(w in lower for w in _REQUIRED_FALSE_WORDS):
            required_val = False
        elif any(w in lower for w in _REQUIRED_TRUE_WORDS):
            required_val = True

        ops: List[FieldOp] = []
        if type_val is not None and type_gap is not None:
            attr = type_gap["attribute"]
            ops.append(FieldOp(op="update", field_name=type_gap["field_name"], **{attr: type_val}))
        if required_val is not None and required_gap is not None:
            ops.append(FieldOp(op="update", field_name=required_gap["field_name"], required=required_val))
        return ops
