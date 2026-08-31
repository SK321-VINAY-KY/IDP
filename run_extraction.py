"""
run_extraction.py — generic dev/test runner for Layer 3.

Usage:
    python run_extraction.py
    python run_extraction.py --doc sdg_goals_output --schema sdg_goals_schema
    python run_extraction.py --doc my_invoice --schema invoice_schema

Arguments:
    --doc     Name of the fixture file (without .md) in tests/fixtures/
              Default: sdg_goals_output
    --schema  Name of the schema file (without .json) in tests/schemas/
              Default: sdg_goals_schema

The schema JSON must be a list of {"name": "...", "description": "..."} objects.
"""
import argparse
import json
import time
from pathlib import Path

from src.ai.layer3_extraction.page_loader import load_pages_from_fixture
from src.ai.layer3_extraction.extractor import extract_by_page_scan
from src.ai.layer3_extraction.schema_validation import extract_with_retry
from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.storage import init_db, save_extraction_run
from src.config.settings import settings

SCHEMAS_DIR = Path("tests/schemas")
REGISTRY_DIR = Path("schema_registry")

def load_schema(schema_name: str):
    schema_path = SCHEMAS_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        schema_path = REGISTRY_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_name}.json in {SCHEMAS_DIR} or {REGISTRY_DIR}")
    with open(schema_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "schema" in raw:
        raw_fields = raw["schema"].get("fields", [])
    elif isinstance(raw, list):
        raw_fields = raw
    else:
        raise ValueError(f"Unsupported schema structure in {schema_path}")
    fields = []
    for item in raw_fields:
        if isinstance(item, dict):
            fields.append(SchemaFieldIn(
                name=item["name"],
                description=item.get("description", "")
            ))
        else:
            fields.append(SchemaFieldIn(name=str(item)))
    return build_dynamic_schema(fields)


def main():
    parser = argparse.ArgumentParser(description="Run Layer 3 extraction from fixture + schema files")
    parser.add_argument("--doc",    default="sdg_goals_output", help="Fixture doc name (no .md)")
    parser.add_argument("--schema", default="schema", help="Schema file name (no .json)")
    args = parser.parse_args()

    print(f"Doc   : tests/fixtures/{args.doc}.md")
    print(f"Schema: tests/schemas/{args.schema}.json")

    init_db()

    # Load pages from fixture
    pages = load_pages_from_fixture(args.doc)
    print(f"Loaded {len(pages)} pages")

    # Load schema from JSON file
    schema = load_schema(args.schema)
    print(f"Schema fields: {list(schema.model_fields.keys())}")

    llm = get_extraction_client()

    start = time.time()
    result = extract_with_retry(
        lambda: extract_by_page_scan(pages, schema, llm),
        schema,
    )
    elapsed = round(time.time() - start, 2)

    result_id = save_extraction_run(
        doc_id=args.doc,
        page_count=len(pages),
        schema_name=args.schema,
        result_json=result.model_dump(),
        llm_provider=settings.extraction_backend,
        model_name=settings.extraction_model_name if settings.extraction_backend == "ollama" else settings.sarvam_model_name,
        processing_time_seconds=elapsed,
        page_outputs=None,   # fixture run — no real PageOutput objects
    )
    print(f"\nSaved to database, row id={result_id}, took {elapsed}s")
    print(result.model_dump_json(indent=2))

    # Also persist to dataset_output/<doc>.extracted.json
    output_dir = Path("dataset_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / f"{args.doc}.extracted.json"
    json_out.write_text(
        json.dumps({
            "fixture_doc": args.doc,
            "schema_used": args.schema,
            "processing_time_seconds": elapsed,
            "extracted_data": result.model_dump(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved JSON output to: {json_out}")


if __name__ == "__main__":
    main()