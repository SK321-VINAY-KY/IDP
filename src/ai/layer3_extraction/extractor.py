"""
File: extractor.py
Purpose: Layer 3 extraction — page-by-page field scanning.

  Flow:
    1. Cache target schema fields (name + description) once for the run.
    2. For each page, ask the LLM which schema fields appear on it.
       Only fields still missing are sent — already-found fields are skipped.
    3. Accumulate hits in the scratchpad (first-seen wins).
    4. After all pages, hydrate and return the Pydantic model.

Owner: engineer-b@idp-pilot
Created: 2026-08-27
"""
from typing import Any, Dict, List, Union

from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.scratchpad import Scratchpad
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_by_page_scan(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
) -> BaseModel:
    """
    Scan every page against the cached target schema fields.
    First-seen value wins per field.
    Return a hydrated Pydantic model.

    Args:
        pages_md:  Either list[dict] with "markdown"/"page_number" keys,
                   or plain list[str] (page_number inferred from index).
        schema:    Target Pydantic BaseModel class to extract into.
        llm:       ExtractionLLMClient (Sarvam or Ollama).
    """
    # Normalise input — support both list[dict] and list[str]
    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)

    # --- 1. Cache schema fields once ---
    schema_fields: List[Dict[str, str]] = [
        {"name": k, "description": v.description or k}
        for k, v in schema.model_fields.items()
    ]
    field_names = [f["name"] for f in schema_fields]
    scratchpad = Scratchpad(schema_field_names=field_names)

    logger.info(
        "layer3.page_scan.start",
        total_pages=total_pages,
        schema_fields=field_names,
    )

    # --- 2. Scan every page ---
    for page_data in pages:
        page_md     = page_data["markdown"]
        page_number = page_data["page_number"]

        if not page_md.strip():
            logger.debug("layer3.page_scan.skip_empty", page_number=page_number)
            continue

        # Only ask about fields still missing
        fields_to_check = [f for f in schema_fields if f["name"] in scratchpad.missing_fields]
        if not fields_to_check:
            logger.info("layer3.page_scan.all_found", stopped_at_page=page_number)
            break

        matches = llm.check_page_for_fields(
            page_md,
            fields_to_check,
            page_number=page_number,
            total_pages=total_pages,
        )
        updated = scratchpad.update(page_number, matches)

        logger.info(
            "layer3.page_scan.page_done",
            page_number=page_number,
            total_pages=total_pages,
            fields_found=len(matches),
            fields_updated=updated,
            fields_remaining=scratchpad.missing_fields,
            scratchpad=scratchpad.snapshot(),
        )

    # --- 3. Final coverage report ---
    missing = scratchpad.missing_fields
    if missing:
        logger.warning("layer3.page_scan.fields_not_found", missing_fields=missing)
    else:
        logger.info("layer3.page_scan.complete", provenance=scratchpad.provenance())

    # --- 4. Hydrate Pydantic schema ---
    try:
        default_instance = schema()
    except Exception:
        default_instance = None

    default_values: Dict[str, Any] = (
        default_instance.model_dump() if default_instance is not None else {}
    )
    result = schema.model_validate({**default_values, **scratchpad.to_values_dict()})

    logger.info(
        "layer3.page_scan.result",
        extracted_fields={k: v for k, v in result.model_dump().items() if v not in ("", None)},
        provenance=scratchpad.provenance(),
    )
    return result
