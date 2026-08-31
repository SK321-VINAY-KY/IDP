from typing import Any, Dict, List, Optional


class Scratchpad:
    """
    Multi-page extraction scratchpad.
    Tracks all field candidates found page-by-page across the entire document.
    """
    def __init__(self, schema_field_names: List[str]) -> None:
        self._all_fields: List[str] = list(schema_field_names)
        # field -> list of {"value": val, "page": page_num}
        self._candidates: Dict[str, List[Dict[str, Any]]] = {f: [] for f in self._all_fields}
        # field -> first found hit {"value": val, "page": page_num}
        self._hits: Dict[str, Dict[str, Any]] = {}

    def update(self, page_number: int, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Record matches found on a page.
        Stores every hit in _candidates, and sets the first-seen in _hits.
        """
        updated = []
        negative_patterns = {"not found", "not mentioned", "n/a", "none", "not available", "null", "nil"}
        for m in matches:
            field = m.get("field", "")
            value = m.get("value", "")
            if field not in self._all_fields or value in ("", None):
                continue

            # Skip negative answers like "Not found on this page"
            val_clean = str(value).strip().lower()
            if any(neg in val_clean for neg in negative_patterns) and len(val_clean) < 40:
                continue

            candidate = {"value": value, "page": page_number}
            self._candidates[field].append(candidate)

            if field not in self._hits:
                self._hits[field] = candidate
                updated.append(field)
            else:
                updated.append(field)

        return updated

    @property
    def all_fields(self) -> List[str]:
        return list(self._all_fields)

    @property
    def missing_fields(self) -> List[str]:
        return [f for f in self._all_fields if not self._candidates[f]]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0

    def get_candidates(self, field: str) -> List[Dict[str, Any]]:
        return self._candidates.get(field, [])

    def to_values_dict(self) -> Dict[str, Any]:
        """First-seen candidate value per field."""
        return {name: hit["value"] for name, hit in self._hits.items()}

    def provenance(self) -> Dict[str, int]:
        """Returns {field_name: page_number} for first-seen hits."""
        return {name: hit["page"] for name, hit in self._hits.items()}

    def snapshot(self) -> List[Dict[str, Any]]:
        """Flat list of found fields with page and value."""
        return [
            {"field": name, "value": hit["value"], "page": hit["page"]}
            for name, hit in self._hits.items()
        ]

    def format_evidence_for_llm(self) -> str:
        """
        Format all collected page-by-page candidates into a readable context
        to feed to the LLM for final extraction.
        """
        lines = ["Here are the candidate values found across document pages:"]
        for field in self._all_fields:
            cands = self._candidates.get(field, [])
            if not cands:
                lines.append(f"- {field}: Not found on any scanned page")
            else:
                formatted_cands = [f'"{c["value"]}" (Page {c["page"]})' for c in cands]
                lines.append(f"- {field}: {', '.join(formatted_cands)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        found_count = len(self._all_fields) - len(self.missing_fields)
        return f"Scratchpad({found_count}/{len(self._all_fields)} found, missing={self.missing_fields})"

