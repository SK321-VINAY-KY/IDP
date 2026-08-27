"""
A deterministic, rule-based 'LLM' used for tests and for running the whole
system with zero external dependencies. It parses simple patterns rather
than truly understanding language - good enough to exercise the entire
state machine / schema-building / validation pipeline end-to-end.

This is what test_conversation_manager.py runs against, so the pipeline
is fully testable without Ollama or Bedrock being reachable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.llm.base import ExtractionResult, FieldAnswer, FieldObservation, LLMAdapter, NewFieldProposal, SchemaProposal
from app.llm.prompts import fallback_question

_TYPE_WORDS = {
    "text": "string", "string": "string", "name": "string",
    "number": "number", "amount": "number", "total": "number",
    "date": "date",
    "boolean": "boolean", "yes/no": "boolean",
    "integer": "integer", "count": "integer",
}


class MockLLMAdapter(LLMAdapter):
    def extract(self, state: str, user_message: str, context: Dict[str, Any]) -> ExtractionResult:
        text = user_message.strip()
        lower = text.lower()

        # Corrections: "remove X" / "drop X"
        removals = []
        m = re.findall(r"(?:remove|drop|delete)\s+([a-zA-Z0-9_ ]+)", lower)
        for name in m:
            removals.append(name.strip())

        if state == "ASK_DOCUMENT_TYPE":
            doc_type = text.rstrip(".")
            return ExtractionResult(document_type=doc_type)

        if state == "ASK_FIELDS":
            names = self._split_field_list(text)
            new_fields = [NewFieldProposal(name=n, **self._guess_attrs(n)) for n in names]
            return ExtractionResult(new_fields=new_fields, removals=removals)

        if state == "FIELD_DETAILS":
            gap = context.get("current_gap") or {}
            field_name = gap.get("field_name")
            attribute = gap.get("attribute")
            field_answers = []
            if field_name and attribute:
                value = self._parse_answer(attribute, lower)
                if value is not None:
                    field_answers.append(FieldAnswer(field_name=field_name, attribute=attribute, value=value))
            # A correction (e.g. "remove X") can arrive even when it doesn't
            # answer the pending gap question - don't drop it just because
            # the gap itself couldn't be parsed from this message.
            if not field_answers and not removals:
                return ExtractionResult(needs_clarification=True, clarification_reason="could not parse answer")
            return ExtractionResult(field_answers=field_answers, removals=removals)

        if state == "CONFIRMATION":
            if lower in ("yes", "y", "correct", "yep", "confirmed"):
                return ExtractionResult(confirmation=True, removals=removals)
            if lower in ("no", "n", "nope", "incorrect"):
                return ExtractionResult(confirmation=False, removals=removals)
            if removals:
                return ExtractionResult(removals=removals)
            return ExtractionResult(needs_clarification=True, clarification_reason="expected yes/no")

        return ExtractionResult(needs_clarification=True, clarification_reason=f"unhandled state {state}")

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
    def _split_field_list(text: str) -> List[str]:
        text = re.sub(r"\band\b", ",", text, flags=re.I)
        parts = [p.strip(" .") for p in text.split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _guess_attrs(name: str) -> Dict[str, Any]:
        lower = name.lower()
        for word, t in _TYPE_WORDS.items():
            if word in lower:
                return {"type": t}
        return {}

    @staticmethod
    def _parse_answer(attribute: str, lower: str):
        if attribute == "required":
            if any(w in lower for w in ("always", "yes", "mandatory", "required")):
                return True
            if any(w in lower for w in ("sometimes", "no", "optional", "not always")):
                return False
            return None
        if attribute == "type":
            for word, t in _TYPE_WORDS.items():
                if word in lower:
                    return t
            return None
        if attribute == "item_type":
            for word, t in _TYPE_WORDS.items():
                if word in lower:
                    return t
            return "string"
        return None
