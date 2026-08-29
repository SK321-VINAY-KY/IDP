"""
The live, progressively-built target schema.

This is intentionally NOT an LLM output. It is a plain data structure that
deterministic code mutates in small, auditable steps. The LLM proposes
changes (via app.llm.base.ExtractionResult); this module is the only place
that actually applies them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SUPPORTED_TYPES = {"string", "number", "integer", "boolean", "date", "object", "array"}

# Common synonyms a model (or a user typing "make it a float") might use.
# normalize_type() maps these onto SUPPORTED_TYPES rather than letting them
# through raw - an unrecognized type used to be able to slip into FieldSpec
# and silently fail validation later; now it either normalizes or turns
# back into a gap immediately, in the same turn.
_TYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string", "name": "string",
    "float": "number", "double": "number", "decimal": "number", "num": "number", "amount": "number",
    "int": "integer", "count": "integer",
    "bool": "boolean", "yes/no": "boolean", "flag": "boolean",
    "datetime": "date", "timestamp": "date", "day": "date",
    "list": "array", "list_of": "array",
    "dict": "object", "nested": "object",
}

# The attributes we consider "required to know" before a field is complete.
# Kept small and explicit on purpose (see design note in README about
# avoiding a 30-question interrogation).
REQUIRED_ATTRIBUTES = ["type", "required"]


def normalize_type(raw: Optional[str]) -> "tuple[Optional[str], Optional[str]]":
    """
    Guards every type value before it reaches a FieldSpec.

    Returns (normalized_type_or_None, note_or_None):
      - raw is None                -> (None, None): nothing to normalize.
      - raw already a valid type   -> (raw, None): passed through as-is.
      - raw is a known alias       -> (normalized, "normalized ...' note):
        the value is fixed up AND the caller is told, so it can be surfaced
        to the user instead of silently rewritten.
      - raw is unrecognized        -> (None, "unrecognized type ...' note):
        deliberately NOT written into the field. This keeps the field
        incomplete (next_gap()/all_gaps() will pick it back up) rather than
        letting a bad value corrupt the schema and fail validate_schema()
        with no path back into the conversation.
    """
    if raw is None:
        return None, None
    key = raw.strip().lower()
    if key in SUPPORTED_TYPES:
        return key, None
    if key in _TYPE_ALIASES:
        normalized = _TYPE_ALIASES[key]
        return normalized, f"normalized type '{raw}' to '{normalized}'"
    return None, (
        f"'{raw}' isn't a type I recognize (string, number, integer, boolean, "
        "date, object, array) - left it unset so it doesn't silently break"
    )


class FieldSpec(BaseModel):
    name: str
    type: Optional[str] = None
    required: Optional[bool] = None
    description: Optional[str] = None
    pattern: Optional[str] = None
    currency: Optional[str] = None
    # For type == "array"
    item_type: Optional[str] = None
    # For type == "object" (e.g. gst -> {cgst, sgst})
    fields: Dict[str, "FieldSpec"] = Field(default_factory=dict)

    def missing_attributes(self) -> List[str]:
        missing = []
        for attr in REQUIRED_ATTRIBUTES:
            if getattr(self, attr) is None:
                missing.append(attr)
        if self.type == "array" and self.item_type is None:
            missing.append("item_type")
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_attributes()) == 0

    def to_json_schema(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type, "required": bool(self.required)}
        if self.description:
            out["description"] = self.description
        if self.pattern:
            out["pattern"] = self.pattern
        if self.currency:
            out["currency"] = self.currency
        if self.type == "array":
            out["item_type"] = self.item_type
        if self.type == "object" and self.fields:
            out["fields"] = {k: v.to_json_schema() for k, v in self.fields.items()}
        return out


FieldSpec.model_rebuild()


class Gap(BaseModel):
    """A single missing piece of information the chatbot still needs."""
    field_name: str
    attribute: str  # e.g. "type", "required", "item_type"


class SchemaState(BaseModel):
    document_type: Optional[str] = None
    fields: Dict[str, FieldSpec] = Field(default_factory=dict)
    # Order fields were added in, so we ask about them in a stable order.
    field_order: List[str] = Field(default_factory=list)

    # ---- mutations (all deterministic, all auditable) ----

    def set_document_type(self, doc_type: str) -> None:
        self.document_type = doc_type.strip().lower().replace(" ", "_")

    def add_field(self, name: str, **attrs: Any) -> FieldSpec:
        key = _normalize_field_name(name)
        if key in self.fields:
            # Merge in any new attributes rather than clobbering what we know.
            existing = self.fields[key]
            for k, v in attrs.items():
                if v is not None:
                    setattr(existing, k, v)
            return existing
        spec = FieldSpec(name=key, **{k: v for k, v in attrs.items() if v is not None})
        self.fields[key] = spec
        self.field_order.append(key)
        return spec

    def remove_field(self, name: str) -> bool:
        key = _normalize_field_name(name)
        if key in self.fields:
            del self.fields[key]
            self.field_order.remove(key)
            return True
        return False

    def update_field_attribute(self, name: str, attribute: str, value: Any) -> bool:
        key = _normalize_field_name(name)
        if key not in self.fields:
            return False
        if attribute not in ("type", "required", "description", "pattern", "currency", "item_type"):
            return False
        setattr(self.fields[key], attribute, value)
        return True

    def apply_operations(self, operations: List[Any]) -> List[str]:
        """
        Applies a batch of add/update/remove operations (app.llm.base.FieldOp)
        in one pass - the mechanism that makes "add a field, fix another
        field's type, and remove a third, all from one message" possible.
        Still fully deterministic: this is the only method that turns LLM
        output into schema mutations, same design boundary as add_field/
        remove_field/update_field_attribute above.

        Returns human-readable notes about anything normalized or rejected
        along the way (e.g. a type alias that got fixed up, or an
        unrecognized type that was deliberately left unset), so the caller
        can surface it instead of silently swallowing it.
        """
        notes: List[str] = []
        for op in operations:
            if op.op == "remove":
                self.remove_field(op.field_name)
                continue

            attrs: Dict[str, Any] = {}
            if op.type is not None:
                normalized, note = normalize_type(op.type)
                if note:
                    notes.append(note)
                if normalized is not None:
                    attrs["type"] = normalized
            for attr in ("required", "currency", "item_type", "pattern", "description"):
                value = getattr(op, attr, None)
                if value is not None:
                    attrs[attr] = value

            key = _normalize_field_name(op.field_name)
            if op.op == "add" or key not in self.fields:
                self.add_field(op.field_name, **attrs)
            else:  # op.op == "update" on a field that already exists
                for attr, value in attrs.items():
                    self.update_field_attribute(op.field_name, attr, value)
        return notes

    # ---- queries ----

    def next_gap(self) -> Optional[Gap]:
        """Returns the next missing piece of info, in field-add order."""
        gaps = self.all_gaps()
        return gaps[0] if gaps else None

    def all_gaps(self) -> List[Gap]:
        """
        Every missing piece of info across every field, in field-add order -
        not just the first one. This is what lets the LLM see (and a user
        answer) several open gaps in a single turn instead of one at a time.
        """
        gaps: List[Gap] = []
        for key in self.field_order:
            for attr in self.fields[key].missing_attributes():
                gaps.append(Gap(field_name=key, attribute=attr))
        return gaps

    def has_fields(self) -> bool:
        return len(self.fields) > 0

    def is_complete(self) -> bool:
        return self.document_type is not None and self.has_fields() and self.next_gap() is None

    def to_json_schema(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "fields": [self.fields[k].to_json_schema() for k in self.field_order],
        }

    def human_summary(self) -> str:
        lines = [f"Document type: {self.document_type}", "", "Fields to extract:"]
        for key in self.field_order:
            f = self.fields[key]
            bits = [f.type or "?"]
            if f.required:
                bits.append("required")
            if f.item_type:
                bits.append(f"of {f.item_type}")
            lines.append(f"  - {f.name} ({', '.join(bits)})")
        return "\n".join(lines)


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")
