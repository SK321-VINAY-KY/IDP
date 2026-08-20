"""
File: extraction_schema.py
Purpose: Flattened extraction schema — no nested objects, no lists.
         Normalize missing extraction values to empty strings so the local model
         does not emit JSON nulls for optional fields.
Owner: genai-platform@shellkode
Created: 2026-08-20
"""
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