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

# The attributes we consider "required to know" before a field is complete.
# Kept small and explicit on purpose (see design note in README about
# avoiding a 30-question interrogation).
REQUIRED_ATTRIBUTES = ["type", "required"]


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

    # ---- queries ----

    def next_gap(self) -> Optional[Gap]:
        """Returns the next missing piece of info, in field-add order."""
        for key in self.field_order:
            spec = self.fields[key]
            missing = spec.missing_attributes()
            if missing:
                return Gap(field_name=key, attribute=missing[0])
        return None

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
