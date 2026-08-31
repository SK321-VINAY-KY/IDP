"""Validation for a completed schema, before it's allowed to be stored."""
from __future__ import annotations

from typing import List

from app.core.schema_state import SUPPORTED_TYPES, SchemaState


def validate_schema(schema: SchemaState) -> List[str]:
    """Returns a list of human-readable error strings. Empty list == valid."""
    errors: List[str] = []

    if not schema.document_type:
        errors.append("document_type is not set")

    if not schema.fields:
        errors.append("schema has no fields")

    seen_names = set()
    for key in schema.field_order:
        field = schema.fields[key]

        if field.name in seen_names:
            errors.append(f"duplicate field name: {field.name}")
        seen_names.add(field.name)

        if field.type is None:
            errors.append(f"field '{field.name}' has no type")
        elif field.type not in SUPPORTED_TYPES:
            errors.append(f"field '{field.name}' has unsupported type '{field.type}'")

        if field.type == "array" and not field.item_type:
            errors.append(f"field '{field.name}' is an array but has no item_type")

        if field.required is None:
            errors.append(f"field '{field.name}' does not specify whether it is required")

        if field.type == "object" and not field.fields:
            errors.append(f"field '{field.name}' is an object but defines no nested fields")

    return errors
