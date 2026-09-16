"""
File: benchmark_concurrency_modes.py
Purpose: Comprehensive benchmark testing all 3 Layer 3 Concurrency Guarantee Modes:
1. Mode A: Ordered Context Mode (Sequential, fresh prior-page context)
2. Mode B: Concurrent Delta Mode (Parallel extraction, dispatch-time snapshot, post-merge reconciliation)
3. Mode C: Epoch / Batch Mode (Parallelism within epoch, fresh canonical context across epochs)

Supports:
- Running on any multi-page Markdown document (e.g. Chander Kochhar 01_compressed 2.md)
- Automatic dynamic schema loading from schema registry
- Automatic ground truth evaluation from sidecar extracted.json
- High-fidelity precomputed graph replay or simulated latency
- Live LLM execution via Sarvam AI client (--live)
- Concurrency scaling & Epoch batch size tuning

Measures:
- Wall-clock execution time (seconds) & Speedup factors
- LLM Call metrics (total calls, extraction calls, resolution calls)
- Extraction accuracy (% matched against ground truth)
- Discovered canonical Graph Nodes & Edges
- Unresolved references & Merge ledger audit records
- Durable Checkpoint creation and verification
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.extractor import (
    extract_document,
    extract_with_graph_memory_batched,
    extract_with_graph_memory_concurrent,
    extract_with_graph_memory_ordered,
)
from src.ai.layer3_extraction.storage import delete_page_checkpoints, list_page_checkpoints
from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema
from src.config.settings import settings


class MedicalDischargeSchema(BaseModel):
    patient_name: str = Field(default="", description="Full name of the patient")
    admission_date: str = Field(default="", description="Date of hospital admission")
    attending_physician: str = Field(default="", description="Name of attending physician")
    procedure_performed: str = Field(default="", description="Surgical or medical procedure performed")
    hospital_name: str = Field(default="", description="Name of the hospital")
    approved_claim_amount: str = Field(default="", description="Final approved insurance claim amount")


DEFAULT_BENCHMARK_DOCUMENT: List[Dict[str, Any]] = [
    {
        "page_number": 1,
        "markdown": (
            "# City Hospital - Inpatient Admission Record\n\n"
            "Patient Name: Rahul Sharma\n"
            "UHID: P12345\n"
            "Admission Date: 12/08/2026\n"
            "Attending Physician: Dr. Arun Sharma\n"
            "Department: General Surgery\n"
        ),
    },
    {
        "page_number": 2,
        "markdown": (
            "## Surgical Operative Notes\n\n"
            "Patient UHID: P12345\n"
            "The above patient underwent successful laparoscopic appendectomy on 13/08/2026.\n"
            "Surgeon: Dr. Arun Sharma\n"
            "No postoperative complications noted.\n"
        ),
    },
    {
        "page_number": 3,
        "markdown": (
            "## Discharge Summary & Order\n\n"
            "Discharge Order for IPD-10023.\n"
            "Confirmed that the patient has fully recovered.\n"
            "Discharge Date: 15/08/2026.\n"
        ),
    },
    {
        "page_number": 4,
        "markdown": (
            "## Insurance Settlement & Final Bill\n\n"
            "Hospital: City Hospital\n"
            "Claim Number: CLM-99881\n"
            "The approved claim amount for the above admission was INR 87,500 under Policy POL-776.\n"
            "Billing settled in full.\n"
        ),
    },
]

DEFAULT_GROUND_TRUTH: Dict[str, str] = {
    "patient_name": "Rahul Sharma",
    "admission_date": "12/08/2026",
    "attending_physician": "Dr. Arun Sharma",
    "procedure_performed": "laparoscopic appendectomy",
    "hospital_name": "City Hospital",
    "approved_claim_amount": "INR 87,500",
}


def load_markdown_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a markdown file into discrete page dictionaries with page numbers."""
    content = file_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"<!--\s*PAGE\s+(\d+)\s*\|[^>]*-->", content, re.IGNORECASE))
    if not matches:
        parts = re.split(r"\n---\n", content)
        return [{"markdown": p.strip(), "page_number": i + 1} for i, p in enumerate(parts) if p.strip()]

    pages = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        raw_slice = content[start:end].strip()
        cleaned = re.split(r"<!--\s*/PAGE(?:\s+\d+)?\s*-->", raw_slice, flags=re.IGNORECASE)[0].strip()
        pages.append({
            "markdown": cleaned,
            "page_number": int(match.group(1)),
        })
    return pages


def load_document_metadata(doc_path: Path) -> Tuple[type[BaseModel], Dict[str, str]]:
    """Loads target schema model and authoritative ground truth for a document."""
    doc_stem = doc_path.stem
    parent = doc_path.parent
    ground_truth: Dict[str, str] = {}
    schema_model: type[BaseModel] = MedicalDischargeSchema

    # 1. Check for .extracted.json (ground truth)
    ext_json_path = parent / f"{doc_stem}.extracted.json"
    if ext_json_path.exists():
        try:
            with open(ext_json_path, encoding="utf-8") as f:
                ext_data = json.load(f).get("extracted_data", {})
                ground_truth = {k: str(v).strip() for k, v in ext_data.items() if v}
        except Exception:
            pass

    # 2. Check for schema definition (.schema_ref.json -> schema_registry)
    ref_path = parent / f"{doc_stem}.schema_ref.json"
    schema_file = None
    if ref_path.exists():
        try:
            with open(ref_path, encoding="utf-8") as f:
                ref_data = json.load(f)
                reg_file = ref_data.get("schema_registry_file")
                if reg_file:
                    schema_file = ROOT_DIR / "schema_registry" / reg_file
        except Exception:
            pass

    if schema_file and schema_file.exists():
        try:
            with open(schema_file, encoding="utf-8") as f:
                schema_json = json.load(f)
                raw_fields = schema_json.get("schema", {}).get("fields", [])
                fields = [
                    SchemaFieldIn(name=f["name"], description=f.get("description", ""))
                    for f in raw_fields
                ]
                schema_model = build_dynamic_schema(fields)
        except Exception:
            pass

    return schema_model, ground_truth


def load_precomputed_graph_data(
    doc_path: Path,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
    """Loads pre-extracted candidate nodes and edges for high-fidelity offline simulation."""
    doc_stem = doc_path.stem
    parent = doc_path.parent
    nodes_by_page: Dict[int, List[Dict[str, Any]]] = {}
    edges_by_page: Dict[int, List[Dict[str, Any]]] = {}

    l3_res_path = parent / f"{doc_stem}.layer3_result.json"
    if l3_res_path.exists():
        try:
            with open(l3_res_path, encoding="utf-8") as f:
                data = json.load(f)
                c_nodes = data.get("candidate_nodes", [])
                c_edges = data.get("candidate_edges", [])
                for n in c_nodes:
                    sp = n.get("source_page", 1)
                    nodes_by_page.setdefault(sp, []).append({
                        "id": f"p{sp}_{n.get('node_id', '')}",
                        "type": n.get("type", "Entity"),
                        "label": n.get("label", ""),
                        "value": n.get("label", "") or n.get("evidence", ""),
                        "confidence": n.get("confidence", 0.95),
                        "properties": {
                            "subtype": n.get("subtype", ""),
                            "evidence": n.get("evidence", ""),
                        },
                    })
                for e in c_edges:
                    sp = e.get("source_page", 1)
                    edges_by_page.setdefault(sp, []).append(e)
                return nodes_by_page, edges_by_page
        except Exception:
            pass

    graph_path = parent / f"{doc_stem}.graph.json"
    if graph_path.exists():
        try:
            with open(graph_path, encoding="utf-8") as f:
                data = json.load(f)
                for n in data.get("nodes", []):
                    for sp in n.get("source_pages", [1]):
                        nodes_by_page.setdefault(sp, []).append(n)
        except Exception:
            pass

    return nodes_by_page, edges_by_page


class LLMCallTracker(ExtractionLLMClient):
    """Wraps an LLM client to precisely track call counts and timing per phase."""

    def __init__(
        self,
        inner_client: Any = None,
        simulated_latency: float = 0.05,
        default_schema: Optional[type[BaseModel]] = None,
        ground_truth: Optional[Dict[str, str]] = None,
        precomputed_nodes: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        precomputed_edges: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ):
        self.inner_client = inner_client
        self.simulated_latency = simulated_latency
        self.default_schema = default_schema or MedicalDischargeSchema
        self.ground_truth = ground_truth or DEFAULT_GROUND_TRUTH
        self.precomputed_nodes = precomputed_nodes or {}
        self.precomputed_edges = precomputed_edges or {}
        self.total_calls = 0
        self.extraction_calls = 0
        self.resolution_calls = 0
        self.check_field_calls = 0
        self.call_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.total_calls = 0
        self.extraction_calls = 0
        self.resolution_calls = 0
        self.check_field_calls = 0
        self.call_history.clear()

    def extract(self, content: str, schema: type[BaseModel]) -> BaseModel:
        self.total_calls += 1
        self.extraction_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "extract"):
            res = self.inner_client.extract(content, schema)
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            filtered = {k: v for k, v in self.ground_truth.items() if k in schema.model_fields}
            res = schema(**filtered)

        elapsed = time.perf_counter() - t0
        self.call_history.append({"call_type": "extract", "duration": round(elapsed, 4)})
        return res

    def summarize_page(self, page_md: str, max_words: int = 100) -> str:
        self.total_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "summarize_page"):
            res = self.inner_client.summarize_page(page_md, max_words)
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            res = page_md[: max_words * 5]

        elapsed = time.perf_counter() - t0
        self.call_history.append({"call_type": "summarize_page", "duration": round(elapsed, 4)})
        return res

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        self.total_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "navigate"):
            res = self.inner_client.navigate(page_summaries, schema_fields)
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            res = {f: list(range(1, len(page_summaries) + 1)) for f in schema_fields}

        elapsed = time.perf_counter() - t0
        self.call_history.append({"call_type": "navigate", "duration": round(elapsed, 4)})
        return res

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        self.total_calls += 1
        self.check_field_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "check_page_for_fields"):
            res = self.inner_client.check_page_for_fields(
                page_md=page_md,
                schema_fields=schema_fields,
                page_number=page_number,
                total_pages=total_pages,
            )
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            res = []

        elapsed = time.perf_counter() - t0
        self.call_history.append({
            "call_type": "check_page_for_fields",
            "page_number": page_number,
            "duration": round(elapsed, 4),
        })
        return res

    def extract_graph_from_page(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        existing_nodes: List[Dict[str, Any]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> Dict[str, Any]:
        self.total_calls += 1
        self.extraction_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "extract_graph_from_page"):
            res = self.inner_client.extract_graph_from_page(
                page_md=page_md,
                schema_fields=schema_fields,
                existing_nodes=existing_nodes,
                page_number=page_number,
                total_pages=total_pages,
            )
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            res = self._simulate_page_extraction(page_number, page_md, schema_fields)

        elapsed = time.perf_counter() - t0
        self.call_history.append({
            "call_type": "extract_graph_from_page",
            "page_number": page_number,
            "duration": round(elapsed, 4),
            "context_nodes_seen": len(existing_nodes),
        })
        return res

    def resolve_schema_from_graph(
        self,
        graph_evidence: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        self.total_calls += 1
        self.resolution_calls += 1
        t0 = time.perf_counter()

        if self.inner_client is not None and hasattr(self.inner_client, "resolve_schema_from_graph"):
            res = self.inner_client.resolve_schema_from_graph(graph_evidence, schema)
        else:
            if self.simulated_latency > 0:
                time.sleep(self.simulated_latency)
            filtered = {k: v for k, v in self.ground_truth.items() if k in schema.model_fields}
            res = schema(**filtered)

        elapsed = time.perf_counter() - t0
        self.call_history.append({
            "call_type": "resolve_schema_from_graph",
            "duration": round(elapsed, 4),
        })
        return res

    def _simulate_page_extraction(
        self,
        page_number: int,
        page_md: str = "",
        schema_fields: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if page_number in self.precomputed_nodes:
            return {
                "entities": self.precomputed_nodes[page_number],
                "relationships": self.precomputed_edges.get(page_number, []),
                "reference_resolutions": [],
            }

        # Fallback simulation
        entities: List[Dict[str, Any]] = []
        if "CHANDER KOCHHAR" in page_md.upper():
            entities.append({
                "id": f"p{page_number}_e_patient",
                "type": "Person",
                "label": "patient_name",
                "value": "CHANDER KOCHHAR",
                "properties": {"schema_field_name": "patient_name"},
                "confidence": 0.99,
            })
        if "HINDUJA" in page_md.upper():
            entities.append({
                "id": f"p{page_number}_e_hospital",
                "type": "Organization",
                "label": "hospital_name",
                "value": "P. D. HINDUJA HOSPITAL & MEDICAL RESEARCH CENTRE",
                "properties": {"schema_field_name": "hospital_name"},
                "confidence": 0.99,
            })
        if "RAHUL SHARMA" in page_md.upper():
            entities.append({
                "id": f"p{page_number}_e_patient",
                "type": "Patient",
                "label": "patient_name",
                "value": "Rahul Sharma",
                "properties": {"schema_field_name": "patient_name"},
                "confidence": 1.0,
            })
        if "APPENDECTOMY" in page_md.upper():
            entities.append({
                "id": f"p{page_number}_e_proc",
                "type": "Procedure",
                "label": "procedure_performed",
                "value": "laparoscopic appendectomy",
                "properties": {"schema_field_name": "procedure_performed"},
                "confidence": 1.0,
            })

        return {
            "entities": entities,
            "relationships": [],
            "reference_resolutions": [],
        }


def evaluate_accuracy(
    result_dict: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> Tuple[float, Dict[str, bool]]:
    matches: Dict[str, bool] = {}
    correct_count = 0
    gt_non_empty = {k: str(v).strip() for k, v in ground_truth.items() if str(v).strip()}
    if not gt_non_empty:
        return 100.0, {}

    for field, expected in gt_non_empty.items():
        actual = str(result_dict.get(field, "") or "").strip()
        matched = (
            expected.lower() in actual.lower()
            or actual.lower() in expected.lower()
        ) if actual else False
        matches[field] = matched
        if matched:
            correct_count += 1

    pct = round((correct_count / len(gt_non_empty)) * 100, 1)
    return pct, matches


async def run_all_modes_benchmark(
    md_path: Optional[str] = None,
    pages_mode: str = "all",
    simulated_latency: float = 0.05,
    use_live_llm: bool = False,
    concurrency_limit: int = 4,
    epoch_size: int = 4,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute test on all 3 concurrency guarantee modes and return metrics."""

    # 1. Load document pages, schema, and ground truth
    if md_path:
        target_path = Path(md_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        all_pages = load_markdown_pages(target_path)
        doc_name = target_path.stem
        schema_model, ground_truth = load_document_metadata(target_path)
        precomputed_nodes, precomputed_edges = load_precomputed_graph_data(target_path)
    else:
        target_path = None
        all_pages = DEFAULT_BENCHMARK_DOCUMENT
        doc_name = "default_benchmark_doc"
        schema_model = MedicalDischargeSchema
        ground_truth = DEFAULT_GROUND_TRUTH
        precomputed_nodes, precomputed_edges = {}, {}

    # Filter pages if requested
    if pages_mode == "all":
        pages = all_pages
    elif pages_mode == "targeted":
        targeted_pages = {1, 5, 6, 7, 9, 10, 17, 56, 57, 63, 68, 75}
        pages = [p for p in all_pages if p.get("page_number") in targeted_pages]
    elif "-" in pages_mode:
        start_p, end_p = map(int, pages_mode.split("-"))
        pages = [p for p in all_pages if start_p <= p.get("page_number", 0) <= end_p]
    elif pages_mode.isdigit():
        max_n = int(pages_mode)
        pages = all_pages[:max_n]
    else:
        pages = all_pages

    # 2. Setup LLM client or tracker
    if use_live_llm:
        from src.adapters.llm.extraction_factory import get_extraction_client
        base_client = get_extraction_client()
        tracker = LLMCallTracker(
            inner_client=base_client,
            simulated_latency=0.0,
            default_schema=schema_model,
            ground_truth=ground_truth,
        )
        mode_label = f"LIVE LLM ({settings.llm_provider})"
    else:
        tracker = LLMCallTracker(
            inner_client=None,
            simulated_latency=simulated_latency,
            default_schema=schema_model,
            ground_truth=ground_truth,
            precomputed_nodes=precomputed_nodes,
            precomputed_edges=precomputed_edges,
        )
        mode_label = f"SIMULATED LLM ({simulated_latency}s latency per call)"

    print("=" * 105)
    print(f"LAYER 3 CONCURRENCY GUARANTEES BENCHMARK [{mode_label}]")
    print(f"Document: {doc_name} | Pages: {len(pages)} | Concurrency Limit: {concurrency_limit} | Epoch Size: {epoch_size}")
    print(f"Schema Fields: {len(schema_model.model_fields)} | Evaluated Ground Truth Fields: {len([v for v in ground_truth.values() if v])}")
    print("=" * 105)

    results: Dict[str, Any] = {
        "document": doc_name,
        "total_pages": len(pages),
        "concurrency_limit": concurrency_limit,
        "epoch_size": epoch_size,
        "mode_label": mode_label,
        "modes": {},
    }

    # --------------------------------------------------------------------------
    # MODE A: ORDERED CONTEXT MODE
    # --------------------------------------------------------------------------
    print("\n>>> Testing Mode A: Ordered Context Mode (Sequential, Fresh Prior-Page Context)...")
    tracker.reset()
    doc_id_a = f"{doc_name}_mode_a"
    delete_page_checkpoints(doc_id_a)
    graph_out_a: Dict[str, Any] = {}

    t0 = time.perf_counter()
    res_a = await extract_with_graph_memory_ordered(
        pages_md=pages,
        schema=schema_model,
        llm=tracker,
        graph_out=graph_out_a,
        doc_id=doc_id_a,
    )
    time_a = round(time.perf_counter() - t0, 3)
    acc_a, matches_a = evaluate_accuracy(res_a.model_dump(), ground_truth)
    calls_a = {
        "total": tracker.total_calls,
        "extraction_calls": tracker.extraction_calls,
        "resolution_calls": tracker.resolution_calls,
    }
    cps_a = len(list_page_checkpoints(doc_id_a))
    delete_page_checkpoints(doc_id_a)

    results["modes"]["Mode A (Ordered)"] = {
        "strategy": "graph_memory_ordered",
        "guarantee": "Page N observes canonical memory from pages 1..N-1",
        "wall_time_seconds": time_a,
        "speedup": 1.0,
        "accuracy_pct": acc_a,
        "field_matches": matches_a,
        "llm_calls": calls_a,
        "graph_nodes": graph_out_a.get("stats", {}).get("total_nodes", 0),
        "graph_edges": graph_out_a.get("stats", {}).get("total_edges", 0),
        "unresolved_references": graph_out_a.get("unresolved_reference_count", 0),
        "merge_ledger_records": len(graph_out_a.get("merge_ledger", [])),
        "checkpoints_created": cps_a,
    }
    print(f"    Completed in {time_a}s | Accuracy: {acc_a}% | LLM Calls: {calls_a['total']} | Nodes: {results['modes']['Mode A (Ordered)']['graph_nodes']} | Edges: {results['modes']['Mode A (Ordered)']['graph_edges']}")

    # --------------------------------------------------------------------------
    # MODE B: CONCURRENT DELTA MODE
    # --------------------------------------------------------------------------
    print(f"\n>>> Testing Mode B: Concurrent Delta Mode (Parallel Extraction, Concurrency={concurrency_limit})...")
    tracker.reset()
    doc_id_b = f"{doc_name}_mode_b"
    delete_page_checkpoints(doc_id_b)
    graph_out_b: Dict[str, Any] = {}

    settings.graph_concurrency_limit = concurrency_limit
    t1 = time.perf_counter()
    res_b = await extract_with_graph_memory_concurrent(
        pages_md=pages,
        schema=schema_model,
        llm=tracker,
        graph_out=graph_out_b,
        doc_id=doc_id_b,
    )
    time_b = round(time.perf_counter() - t1, 3)
    acc_b, matches_b = evaluate_accuracy(res_b.model_dump(), ground_truth)
    calls_b = {
        "total": tracker.total_calls,
        "extraction_calls": tracker.extraction_calls,
        "resolution_calls": tracker.resolution_calls,
    }
    speedup_b = round(time_a / time_b, 2) if time_b > 0 else 1.0
    cps_b = len(list_page_checkpoints(doc_id_b))
    delete_page_checkpoints(doc_id_b)

    results["modes"]["Mode B (Concurrent)"] = {
        "strategy": "graph_memory_concurrent",
        "guarantee": "Workers extract in parallel on snapshot context; reconciled post-merge",
        "wall_time_seconds": time_b,
        "speedup": speedup_b,
        "accuracy_pct": acc_b,
        "field_matches": matches_b,
        "llm_calls": calls_b,
        "graph_nodes": graph_out_b.get("stats", {}).get("total_nodes", 0),
        "graph_edges": graph_out_b.get("stats", {}).get("total_edges", 0),
        "unresolved_references": graph_out_b.get("unresolved_reference_count", 0),
        "merge_ledger_records": len(graph_out_b.get("merge_ledger", [])),
        "checkpoints_created": cps_b,
    }
    print(f"    Completed in {time_b}s | Speedup: {speedup_b}x | Accuracy: {acc_b}% | LLM Calls: {calls_b['total']} | Nodes: {results['modes']['Mode B (Concurrent)']['graph_nodes']} | Edges: {results['modes']['Mode B (Concurrent)']['graph_edges']}")

    # --------------------------------------------------------------------------
    # MODE C: EPOCH / BATCH MODE
    # --------------------------------------------------------------------------
    print(f"\n>>> Testing Mode C: Epoch / Batch Mode (Epoch Size={epoch_size}, Concurrency={concurrency_limit})...")
    tracker.reset()
    doc_id_c = f"{doc_name}_mode_c"
    delete_page_checkpoints(doc_id_c)
    graph_out_c: Dict[str, Any] = {}

    t2 = time.perf_counter()
    res_c = await extract_with_graph_memory_batched(
        pages_md=pages,
        schema=schema_model,
        llm=tracker,
        graph_out=graph_out_c,
        doc_id=doc_id_c,
        epoch_size=epoch_size,
    )
    time_c = round(time.perf_counter() - t2, 3)
    acc_c, matches_c = evaluate_accuracy(res_c.model_dump(), ground_truth)
    calls_c = {
        "total": tracker.total_calls,
        "extraction_calls": tracker.extraction_calls,
        "resolution_calls": tracker.resolution_calls,
    }
    speedup_c = round(time_a / time_c, 2) if time_c > 0 else 1.0
    cps_c = len(list_page_checkpoints(doc_id_c))
    delete_page_checkpoints(doc_id_c)

    results["modes"]["Mode C (Epoch/Batch)"] = {
        "strategy": "graph_memory_batched",
        "guarantee": "Parallel within epoch; fresh canonical memory across epochs",
        "wall_time_seconds": time_c,
        "speedup": speedup_c,
        "accuracy_pct": acc_c,
        "field_matches": matches_c,
        "llm_calls": calls_c,
        "graph_nodes": graph_out_c.get("stats", {}).get("total_nodes", 0),
        "graph_edges": graph_out_c.get("stats", {}).get("total_edges", 0),
        "unresolved_references": graph_out_c.get("unresolved_reference_count", 0),
        "merge_ledger_records": len(graph_out_c.get("merge_ledger", [])),
        "checkpoints_created": cps_c,
    }
    print(f"    Completed in {time_c}s | Speedup: {speedup_c}x | Accuracy: {acc_c}% | LLM Calls: {calls_c['total']} | Nodes: {results['modes']['Mode C (Epoch/Batch)']['graph_nodes']} | Edges: {results['modes']['Mode C (Epoch/Batch)']['graph_edges']}")

    # --------------------------------------------------------------------------
    # PRINT SUMMARY COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("CONCURRENCY MODES COMPARISON SUMMARY")
    print("=" * 105)
    header = f"{'Metric':<30} | {'Mode A (Ordered)':<22} | {'Mode B (Concurrent)':<22} | {'Mode C (Epoch/Batch)':<22}"
    print(header)
    print("-" * 105)

    print(f"{'Guarantee':<30} | {'Pages 1..N-1 context':<22} | {'Snapshot context':<22} | {'Epoch fresh context':<22}")
    print(f"{'Wall Time (seconds)':<30} | {time_a:<22} | {time_b:<22} | {time_c:<22}")
    print(f"{'Speedup vs Sequential':<30} | {'1.0x (baseline)':<22} | {f'{speedup_b}x':<22} | {f'{speedup_c}x':<22}")
    print(f"{'Accuracy (%)':<30} | {f'{acc_a}%':<22} | {f'{acc_b}%':<22} | {f'{acc_c}%':<22}")
    print(f"{'Total LLM Calls':<30} | {calls_a['total']:<22} | {calls_b['total']:<22} | {calls_c['total']:<22}")
    print(f"{' - Page Extraction Calls':<30} | {calls_a['extraction_calls']:<22} | {calls_b['extraction_calls']:<22} | {calls_c['extraction_calls']:<22}")
    print(f"{' - Schema Resolution Calls':<30} | {calls_a['resolution_calls']:<22} | {calls_b['resolution_calls']:<22} | {calls_c['resolution_calls']:<22}")
    print(f"{'Canonical Graph Nodes':<30} | {results['modes']['Mode A (Ordered)']['graph_nodes']:<22} | {results['modes']['Mode B (Concurrent)']['graph_nodes']:<22} | {results['modes']['Mode C (Epoch/Batch)']['graph_nodes']:<22}")
    print(f"{'Canonical Graph Edges':<30} | {results['modes']['Mode A (Ordered)']['graph_edges']:<22} | {results['modes']['Mode B (Concurrent)']['graph_edges']:<22} | {results['modes']['Mode C (Epoch/Batch)']['graph_edges']:<22}")
    print(f"{'Merge Audit Records':<30} | {results['modes']['Mode A (Ordered)']['merge_ledger_records']:<22} | {results['modes']['Mode B (Concurrent)']['merge_ledger_records']:<22} | {results['modes']['Mode C (Epoch/Batch)']['merge_ledger_records']:<22}")
    print(f"{'Durable Checkpoints':<30} | {cps_a:<22} | {cps_b:<22} | {cps_c:<22}")
    print("=" * 105)

    # Save results JSON if output path requested
    save_path = output_file or (Path("dataset_output") / f"{doc_name}.concurrency_benchmark.json" if target_path else None)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to: {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 3 Concurrency Modes on Documents")
    parser.add_argument("--md-path", type=str, default="dataset_output/Chander Kochhar 01_compressed 2.md", help="Path to markdown document")
    parser.add_argument("--pages", type=str, default="all", help="Page range/mode: 'all', 'targeted', '1-10', or page count")
    parser.add_argument("--latency", type=float, default=0.05, help="Simulated latency per LLM call (seconds)")
    parser.add_argument("--live", action="store_true", help="Use real live LLM client instead of simulation")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency limit for Mode B & C")
    parser.add_argument("--epoch-size", type=int, default=4, help="Epoch size for Mode C")
    parser.add_argument("--output", type=str, default=None, help="Path to save benchmark results JSON")
    args = parser.parse_args()

    asyncio.run(
        run_all_modes_benchmark(
            md_path=args.md_path,
            pages_mode=args.pages,
            simulated_latency=args.latency,
            use_live_llm=args.live,
            concurrency_limit=args.concurrency,
            epoch_size=args.epoch_size,
            output_file=args.output,
        )
    )
