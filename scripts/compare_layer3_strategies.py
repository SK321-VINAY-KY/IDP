"""
File: compare_layer3_strategies.py
Purpose: Benchmark comparing 3 Layer 3 strategies:
1. page_scan (Classic Scratchpad)
2. graph_memory (Sequential Graph Extraction)
3. graph_memory_concurrent (Concurrent Blackboard / Memory Manager)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ai.layer3_extraction.extractor import (
    extract_by_page_scan,
    extract_with_graph_memory,
    extract_with_graph_memory_concurrent,
    extract_document,
)
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
from src.adapters.llm.extraction_factory import get_extraction_client
from src.config.settings import settings


class MedicalDischargeSchema(BaseModel):
    patient_name: str = Field(default="", description="Full name of the patient")
    admission_date: str = Field(default="", description="Date of hospital admission")
    procedure_performed: str = Field(default="", description="Surgical or medical procedure performed on the patient")
    approved_claim_amount: str = Field(default="", description="Total approved claim amount for the admission")


# Deterministic Multi-Page Medical Document with Cross-Page Anaphora
MULTI_PAGE_DOC: List[Dict[str, Any]] = [
    {
        "page_number": 1,
        "markdown": (
            "# City Hospital Discharge Summary\n\n"
            "**Patient Name**: Rahul Sharma\n"
            "**Patient ID**: P12345\n"
            "**Admitted On**: 12/08/2026\n"
            "**Attending Physician**: Dr. Arun Sharma\n"
            "**Primary Diagnosis**: Acute appendicitis\n"
        ),
    },
    {
        "page_number": 2,
        "markdown": (
            "## Clinical Course & Surgical Intervention\n\n"
            "The above patient underwent laparoscopic appendectomy on 13/08/2026.\n"
            "The procedure was uncomplicated. Post-operative recovery was satisfactory.\n"
            "Cafeteria is open 24 hours for visitors on the first floor.\n"
        ),
    },
    {
        "page_number": 3,
        "markdown": (
            "## Insurance & Billing Settlement\n\n"
            "Claim Reference: CLM-99881\n"
            "The approved claim amount for the above admission was INR 87,500 under Policy POL-776.\n"
            "Visitor parking pass validated for Level 2.\n"
        ),
    },
]

GROUND_TRUTH = {
    "patient_name": "Rahul Sharma",
    "admission_date": "12/08/2026",
    "procedure_performed": "laparoscopic appendectomy",
    "approved_claim_amount": "INR 87,500",
}


def run_comparison(concurrency_limit: int = 8, use_mock: bool = False):
    print("=" * 110)
    print("LAYER 3 STRATEGY 3-WAY BENCHMARK: PAGE-SCAN vs GRAPH-MEMORY (SEQ) vs GRAPH-MEMORY (CONCURRENT)")
    print("=" * 110)
    print(f"Backend configured: {settings.extraction_backend}")
    print(f"Concurrency limit for concurrent strategy: {concurrency_limit}")
    print(f"Target Schema fields: {list(MedicalDischargeSchema.model_fields.keys())}")
    print(f"Pages in test document: {len(MULTI_PAGE_DOC)}")
    print("-" * 110)

    # Configure concurrency setting
    settings.graph_concurrency_limit = concurrency_limit

    if use_mock:
        from unittest.mock import MagicMock
        llm = MagicMock()
        def mock_extract_page(page_md, schema_fields, existing_nodes, page_number, total_pages):
            time.sleep(0.3)
            if page_number == 1:
                return {
                    "entities": [
                        {"id": "p1_e1", "type": "Patient", "label": "patient_name", "value": "Rahul Sharma", "is_schema_field": True, "schema_field_name": "patient_name"},
                        {"id": "p1_e2", "type": "Date", "label": "admission_date", "value": "12/08/2026", "is_schema_field": True, "schema_field_name": "admission_date"},
                    ],
                    "relationships": [],
                    "reference_resolutions": [],
                }
            elif page_number == 2:
                return {
                    "entities": [
                        {"id": "p2_e1", "type": "Procedure", "label": "procedure_performed", "value": "laparoscopic appendectomy", "is_schema_field": True, "schema_field_name": "procedure_performed"},
                    ],
                    "relationships": [],
                    "reference_resolutions": [{"phrase": "the above patient", "target_type": "Patient", "resolved_to_node_id": "p1_e1"}],
                }
            else:
                return {
                    "entities": [
                        {"id": "p3_e1", "type": "Amount", "label": "approved_claim_amount", "value": "INR 87,500", "is_schema_field": True, "schema_field_name": "approved_claim_amount"},
                    ],
                    "relationships": [],
                    "reference_resolutions": [],
                }
        llm.extract_graph_from_page.side_effect = mock_extract_page
        llm.resolve_schema_from_graph.return_value = MedicalDischargeSchema(
            patient_name="Rahul Sharma",
            admission_date="12/08/2026",
            procedure_performed="laparoscopic appendectomy",
            approved_claim_amount="INR 87,500",
        )
        llm.extract.return_value = MedicalDischargeSchema(
            patient_name="Rahul Sharma",
            admission_date="12/08/2026",
            procedure_performed="laparoscopic appendectomy",
            approved_claim_amount="INR 87,500",
        )
        llm.check_page_for_fields.return_value = [
            {"field_name": "patient_name", "field_value": "Rahul Sharma", "confidence": 1.0},
            {"field_name": "admission_date", "field_value": "12/08/2026", "confidence": 1.0},
            {"field_name": "procedure_performed", "field_value": "laparoscopic appendectomy", "confidence": 1.0},
            {"field_name": "approved_claim_amount", "field_value": "INR 87,500", "confidence": 1.0},
        ]
    else:
        llm = get_extraction_client()

    # --------------------------------------------------------------------------
    # Strategy 1: Classic Page Scan (Scratchpad)
    # --------------------------------------------------------------------------
    print("\n[1/3] Running Classic Strategy: page_scan (Scratchpad)...")
    t0 = time.time()
    graph_out_scan: Dict[str, Any] = {}
    res_page_scan = extract_by_page_scan(MULTI_PAGE_DOC, MedicalDischargeSchema, llm, graph_out=graph_out_scan)
    time_page_scan = round(time.time() - t0, 2)
    dict_page_scan = res_page_scan.model_dump()
    print(f"Page Scan completed in {time_page_scan}s")

    # --------------------------------------------------------------------------
    # Strategy 2: Graph Memory Strategy (Sequential)
    # --------------------------------------------------------------------------
    print("\n[2/3] Running Sequential Graph-Memory Strategy: graph_memory...")
    t1 = time.time()
    graph_out_seq: Dict[str, Any] = {}
    res_graph_seq = extract_with_graph_memory(MULTI_PAGE_DOC, MedicalDischargeSchema, llm, graph_out=graph_out_seq)
    time_graph_seq = round(time.time() - t1, 2)
    dict_graph_seq = res_graph_seq.model_dump()
    graph_stats_seq = graph_out_seq.get("stats", {})
    print(f"Graph Memory (Sequential) completed in {time_graph_seq}s")

    # --------------------------------------------------------------------------
    # Strategy 3: Graph Memory Strategy (Concurrent)
    # --------------------------------------------------------------------------
    print(f"\n[3/3] Running Concurrent Graph-Memory Strategy: graph_memory_concurrent (limit={concurrency_limit})...")
    t2 = time.time()
    graph_out_conc: Dict[str, Any] = {}
    res_graph_conc = extract_document(
        MULTI_PAGE_DOC,
        MedicalDischargeSchema,
        llm,
        strategy="graph_memory_concurrent",
        graph_out=graph_out_conc,
    )
    time_graph_conc = round(time.time() - t2, 2)
    dict_graph_conc = res_graph_conc.model_dump()
    graph_stats_conc = graph_out_conc.get("stats", {})
    print(f"Graph Memory (Concurrent) completed in {time_graph_conc}s")

    # --------------------------------------------------------------------------
    # Evaluation & Comparison Metrics
    # --------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("EVALUATION & COMPARISON RESULTS")
    print("=" * 110)

    col_w = 20
    print(f"{'Field':<{col_w}} | {'Ground Truth':<{col_w}} | {'Page Scan':<{col_w}} | {'Graph (Seq)':<{col_w}} | {'Graph (Conc)':<{col_w}}")
    print("-" * 110)

    matches_scan = 0
    matches_seq = 0
    matches_conc = 0

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass

    comparison_rows = []
    for field, expected in GROUND_TRUTH.items():
        val_scan = dict_page_scan.get(field, "")
        val_seq = dict_graph_seq.get(field, "")
        val_conc = dict_graph_conc.get(field, "")

        scan_correct = expected.lower() in val_scan.lower() if val_scan else False
        seq_correct = expected.lower() in val_seq.lower() if val_seq else False
        conc_correct = expected.lower() in val_conc.lower() if val_conc else False

        if scan_correct:
            matches_scan += 1
        if seq_correct:
            matches_seq += 1
        if conc_correct:
            matches_conc += 1

        scan_disp = f"{val_scan[:14]} {'[OK]' if scan_correct else '[X]'}"
        seq_disp = f"{val_seq[:14]} {'[OK]' if seq_correct else '[X]'}"
        conc_disp = f"{val_conc[:14]} {'[OK]' if conc_correct else '[X]'}"

        print(f"{field:<{col_w}} | {expected[:18]:<{col_w}} | {scan_disp:<{col_w}} | {seq_disp:<{col_w}} | {conc_disp:<{col_w}}")

        comparison_rows.append({
            "field": field,
            "ground_truth": expected,
            "page_scan_value": val_scan,
            "page_scan_correct": scan_correct,
            "graph_seq_value": val_seq,
            "graph_seq_correct": seq_correct,
            "graph_conc_value": val_conc,
            "graph_conc_correct": conc_correct,
        })

    accuracy_scan = round(matches_scan / len(GROUND_TRUTH) * 100, 1)
    accuracy_seq = round(matches_seq / len(GROUND_TRUTH) * 100, 1)
    accuracy_conc = round(matches_conc / len(GROUND_TRUTH) * 100, 1)

    speedup = round(time_graph_seq / time_graph_conc, 2) if time_graph_conc > 0 else 1.0

    print("-" * 110)
    print(f"Accuracy Score:            | Scan: {accuracy_scan}% ({matches_scan}/{len(GROUND_TRUTH)}) | Seq: {accuracy_seq}% ({matches_seq}/{len(GROUND_TRUTH)}) | Conc: {accuracy_conc}% ({matches_conc}/{len(GROUND_TRUTH)})")
    print(f"Processing Time:           | Scan: {time_page_scan}s | Seq: {time_graph_seq}s | Conc: {time_graph_conc}s (Speedup: {speedup}x)")
    print(f"Discovered Graph Nodes:    | Scan: 0 | Seq: {graph_stats_seq.get('total_nodes', 0)} nodes | Conc: {graph_stats_conc.get('total_nodes', 0)} nodes")
    print(f"Discovered Graph Edges:    | Scan: 0 | Seq: {graph_stats_seq.get('total_edges', 0)} edges | Conc: {graph_stats_conc.get('total_edges', 0)} edges")
    print(f"Unresolved References:     | Scan: N/A | Seq: 0 | Conc: {graph_out_conc.get('unresolved_reference_count', 0)}")
    print(f"Page Worker Failures:      | Scan: 0 | Seq: 0 | Conc: {graph_out_conc.get('pages_failed', 0)}")

    # Save artifact
    output_dir = ROOT_DIR / "dataset_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "strategy_comparison_report.json"
    report_file.write_text(
        json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "concurrency_limit": concurrency_limit,
            "accuracy": {
                "page_scan": accuracy_scan,
                "graph_memory_sequential": accuracy_seq,
                "graph_memory_concurrent": accuracy_conc,
            },
            "processing_time_seconds": {
                "page_scan": time_page_scan,
                "graph_memory_sequential": time_graph_seq,
                "graph_memory_concurrent": time_graph_conc,
                "speedup_factor": speedup,
            },
            "graph_statistics": {
                "sequential": graph_stats_seq,
                "concurrent": graph_stats_conc,
            },
            "concurrency_metadata": {
                "pages_total": graph_out_conc.get("pages_total", len(MULTI_PAGE_DOC)),
                "pages_succeeded": graph_out_conc.get("pages_succeeded", len(MULTI_PAGE_DOC)),
                "pages_failed": graph_out_conc.get("pages_failed", 0),
                "unresolved_reference_count": graph_out_conc.get("unresolved_reference_count", 0),
            },
            "fields_comparison": comparison_rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved detailed comparison report to: {report_file}")
    print("=" * 110)

    return {
        "time_page_scan": time_page_scan,
        "time_graph_seq": time_graph_seq,
        "time_graph_conc": time_graph_conc,
        "accuracy_scan": accuracy_scan,
        "accuracy_seq": accuracy_seq,
        "accuracy_conc": accuracy_conc,
        "nodes_seq": graph_stats_seq.get("total_nodes", 0),
        "nodes_conc": graph_stats_conc.get("total_nodes", 0),
        "edges_seq": graph_stats_seq.get("total_edges", 0),
        "edges_conc": graph_stats_conc.get("total_edges", 0),
        "speedup": speedup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B/C Compare Layer 3 Strategies")
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="Concurrency limit for graph_memory_concurrent")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for local testing")
    args = parser.parse_args()
    run_comparison(concurrency_limit=args.concurrency, use_mock=args.mock)
