"""
run_extraction.py — CLI entry point for Layer 3 Navigation-First Extraction Pipeline.

Usage:
    python run_extraction.py
    python run_extraction.py --doc "Chander Kochhar 01_compressed 2.md"
    python run_extraction.py --doc dataset_output/sdg_goals_output.md --focus "extract all goals" "identify milestones"

Arguments:
    --doc      Markdown file path or doc name in dataset_output/ or tests/fixtures/
    --focus    Optional steering hints / extraction focus list
    --output   Optional path to write the output JSON (default: <doc_stem>.layer3_result.json)
"""
import argparse
import sys
from pathlib import Path

from src.ai.layer3_extraction.pipeline import run_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Layer 3 Navigation-First Pipeline on a document markdown file"
    )
    parser.add_argument(
        "--doc",
        default="Chander Kochhar 01_compressed 2.md",
        help="Path or name of the markdown file to process (default: Chander Kochhar 01_compressed 2.md)",
    )
    parser.add_argument(
        "--focus",
        nargs="*",
        default=None,
        help="Optional extraction focus hints (e.g. --focus 'check discounts' 'ignore footers')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON output",
    )
    parser.add_argument(
        "--target-pages",
        nargs="*",
        type=int,
        default=None,
        help="Explicit list of page numbers to extract (bypasses Phase A segmentation and Phase B navigation)",
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        for cand in [
            Path("dataset_output") / doc_path.name,
            Path("tests/fixtures") / doc_path.name,
            Path("tests/fixtures") / f"{doc_path.name}.md",
        ]:
            if cand.exists():
                doc_path = cand
                break

    if not doc_path.exists():
        print(f"Error: Document file '{args.doc}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running Layer 3 Navigation Pipeline on: {doc_path}")
    if args.focus:
        print(f"Extraction Focus: {args.focus}")
    if args.target_pages:
        print(f"Target Pages (Bypass Phase A/B): {args.target_pages}")

    result = run_pipeline(
        doc_path=doc_path,
        output_json_path=args.output,
        extraction_focus=args.focus,
        target_pages=args.target_pages,
    )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Segments detected   : {len(result.segments)}")
    print(f"Navigated pages     : {result.navigation_map.all_navigated_pages}")
    print(f"Candidate nodes     : {len(result.candidate_nodes)}")
    print(f"Candidate edges     : {len(result.candidate_edges)}")
    print(f"Resolved entities   : {len(result.resolved_entities)}")
    print(f"Actual LLM calls    : {result.cost_summary.actual_llm_calls_total}")
    print(f"Cost reduction      : {result.cost_summary.reduction_percentage}% vs exhaustive scan")
    print(f"Elapsed time        : {result.elapsed_seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()