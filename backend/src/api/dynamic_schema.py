"""
File: dynamic_schema.py
Purpose: Turns a user-submitted list of {name, description} fields into a
         Pydantic model, so Layer 3's extract_by_page_scan() — which only
         requires `type[BaseModel]` — can run against an arbitrary,
         request-time target schema.
Owner: api@idp-pilot
Created: 2026-08-26
"""
import re
from typing import List

from pydantic import BaseModel, Field, create_model, field_validator


class SchemaFieldIn(BaseModel):
    name: str
    description: str = ""


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug


def build_dynamic_schema(fields: List[SchemaFieldIn]) -> type[BaseModel]:
    """
    Every field is extracted as a string, normalizing missing values to ""
    so the model doesn't emit JSON nulls for fields it couldn't find.
    """
    field_defs: dict[str, tuple] = {}
    seen: set[str] = set()

    for f in fields:
        key = _slugify(f.name)
        if not key:
            continue
        # de-duplicate slugs (e.g. "Total $" and "Total!" both -> "total")
        base_key, suffix = key, 2
        while key in seen:
            key = f"{base_key}_{suffix}"
            suffix += 1
        seen.add(key)
        field_defs[key] = (str, Field("", description=f.description or f.name))

    if not field_defs:
        raise ValueError("No valid schema fields were provided.")

    def normalize_missing_values(cls, value):
        return "" if value is None else value

    validators = {
        "_normalize_missing_values": field_validator(*field_defs.keys(), mode="before")(
            normalize_missing_values
        )
    }

    return create_model("UserExtractionSchema", __validators__=validators, **field_defs)