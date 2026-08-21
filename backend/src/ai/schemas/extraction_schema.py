"""
File: extraction_schema.py
Purpose: Flattened extraction schema — no nested objects, no lists.
         Normalize missing extraction values to empty strings so the local model
         does not emit JSON nulls for optional fields.
Owner: engineer-b@idp-pilot
Created: 2026-08-20
"""
import re

from pydantic import BaseModel, Field, field_validator


class DocumentExtraction(BaseModel):
    document_title: str = Field("", description="Title of the document")
    total_goals: str = Field("", description="Total number of goals mentioned, e.g. '17'")
    total_targets: str = Field("", description="Total number of targets mentioned, e.g. '169'")
    first_goal_title: str = Field("", description="Title of Goal 1")
    last_goal_title: str = Field("", description="Title of the final goal (Goal 17)")

    @field_validator("document_title", "total_goals", "total_targets", "first_goal_title", "last_goal_title", mode="before")
    @classmethod
    def normalize_missing_values(cls, value):
        if value is None:
            return ""
        return value

    @classmethod
    def complete_from_text(cls, result: "DocumentExtraction", content: str) -> "DocumentExtraction":
        values = result.model_dump()

        title_match = re.search(
            r"(?m)^(?!\s*(?:<!--|\[Page|#|Goal\b|-|Target\b))\s*(\S.+?)\s*$",
            content,
        )
        goals_match = re.search(r"\b(\d+)\s+Sustainable\s+Development\s+Goals\b", content, re.I)
        targets_match = re.search(r"\b(\d+)\s+Targets\b", content, re.I)
        goal_matches = re.finditer(
            r"(?m)^\s*(?:#{1,6}\s*)?Goal\s+(\d+)\.\s*(.+?)\s*$", content
        )
        goals = {int(match.group(1)): match.group(2).strip() for match in goal_matches}

        if title_match:
            values["document_title"] = title_match.group(1).strip()
        if goals_match:
            values["total_goals"] = goals_match.group(1)
        if targets_match:
            values["total_targets"] = targets_match.group(1)
        if 1 in goals:
            values["first_goal_title"] = goals[1]
        if goals:
            values["last_goal_title"] = goals[max(goals)]

        return cls.model_validate(values)