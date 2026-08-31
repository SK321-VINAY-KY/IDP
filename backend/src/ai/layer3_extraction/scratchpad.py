"""
File: scratchpad.py
Purpose: In-memory accumulator for per-field extraction hits across pages.
         Simple overwrite — every new non-empty value replaces the previous one.
         Last page to find a field wins.

Owner: engineer-b@idp-pilot
Created: 2026-08-27
"""
from typing import Any, Dict, List


class Scratchpad:
    def __init__(self, schema_field_names: List[str]) -> None:
        self._all_fields: List[str] = list(schema_field_names)
        self._hits: Dict[str, Dict[str, Any]] = {}

    def update(self, page_number: int, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Record matches from one page.
        Any non-empty value overwrites the previous value for that field.
        Unknown fields and empty values are ignored.
        Returns list of field names that were added or updated.
        """
        updated: List[str] = []
        for m in matches:
            field = m.get("field", "")
            value = m.get("value", "")
            if field not in self._all_fields or value in ("", None):
                continue
            self._hits[field] = {"value": value, "page": page_number}
            updated.append(field)
        return updated

    @property
    def missing_fields(self) -> List[str]:
        return [f for f in self._all_fields if f not in self._hits]

    @property
    def is_complete(self) -> bool:
        return len(self._hits) == len(self._all_fields)

    def to_values_dict(self) -> Dict[str, Any]:
        """Plain {field_name: value} — used to hydrate the Pydantic schema."""
        return {name: hit["value"] for name, hit in self._hits.items()}

    def provenance(self) -> Dict[str, int]:
        """Returns {field_name: page_number} for logging."""
        return {name: hit["page"] for name, hit in self._hits.items()}

    def snapshot(self) -> List[Dict[str, Any]]:
        """Flat list of current state — logged after each page."""
        return [
            {"field": name, "value": hit["value"], "page": hit["page"]}
            for name, hit in self._hits.items()
        ]

    def __repr__(self) -> str:
        return f"Scratchpad({len(self._hits)}/{len(self._all_fields)} found, missing={self.missing_fields})"
