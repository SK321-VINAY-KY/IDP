"""
File: extractor.py
Purpose: Layer 3 extraction — multi-page field scanning with LLM scratchpad resolution.

  Flow:
    1. Cache target schema fields (name + description) once for the entire run.
    2. Scan EVERY document page against ALL target schema fields (no early stop).
    3. Accumulate all page hits and candidate values into the Scratchpad.
    4. Feed the full Scratchpad evidence to the LLM to extract and resolve
       final structured values according to the target schema.

Owner: engineer-b@idp-pilot
Created: 2026-08-27 | Updated: 2026-08-31
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
    Collects findings for ALL fields page-by-page into the Scratchpad,
    then provides the Scratchpad evidence to the LLM to produce the final
    hydrated Pydantic model.

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

    # --- 1. Cache target schema fields once ---
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

    # --- 2. Scan every page against ALL fields ---
    for page_data in pages:
        page_md     = page_data["markdown"]
        page_number = page_data["page_number"]

        if not page_md.strip():
            logger.debug("layer3.page_scan.skip_empty", page_number=page_number)
            continue

        # Check ALL schema fields on every page
        matches = llm.check_page_for_fields(
            page_md,
            schema_fields,
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
            scratchpad=scratchpad.snapshot(),
        )

    # --- 3. Coverage summary ---
    missing = scratchpad.missing_fields
    if missing:
        logger.warning("layer3.page_scan.fields_not_found", missing_fields=missing)
    else:
        logger.info("layer3.page_scan.all_found_in_scan", provenance=scratchpad.provenance())

    # --- 4. Give Scratchpad evidence to LLM for final extraction ---
    scratchpad_values = scratchpad.to_values_dict()
    if not scratchpad_values:
        logger.warning("layer3.page_scan.no_candidates_found_fallback_to_markdown")
        # Fall back to full document markdown if page scanning missed fields
        evidence_text = "\n\n".join([p["markdown"] for p in pages if p.get("markdown")])
    else:
        evidence_text = scratchpad.format_evidence_for_llm()

    logger.info("layer3.llm_final_extraction.start", evidence_lines=len(evidence_text.splitlines()))

    try:
        result = llm.extract(evidence_text, schema)
    except Exception as exc:
        logger.warning(
            "layer3.llm_final_extraction.fallback_to_scratchpad",
            error=str(exc),
        )
        # Fallback to direct scratchpad values if LLM call fails
        try:
            default_instance = schema()
        except Exception:
            default_instance = None

        default_values: Dict[str, Any] = (
            default_instance.model_dump() if default_instance is not None else {}
        )
        result = schema.model_validate({**default_values, **scratchpad_values})

    # Merge scratchpad candidates for any missing / blank fields in result
    res_dict = result.model_dump()
    merged = dict(res_dict)
    for k, v in scratchpad_values.items():
        if k in merged and (merged[k] in ("", None)) and v not in ("", None):
            merged[k] = v
    if merged != res_dict:
        result = schema.model_validate(merged)

    logger.info(
        "layer3.page_scan.result",
        extracted_fields={k: v for k, v in result.model_dump().items() if v not in ("", None)},
        provenance=scratchpad.provenance(),
    )
    return result

