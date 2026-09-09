"""
tests/test_layer3_pipeline.py — Dedicated Layer 3 Extraction & Chatbot Test Suite & Runner.

Features:
1. ONLY Layer 3 Execution:
   - Ingests any Markdown file (.md) and any JSON schema.
   - Parses multi-page markers (<!-- PAGE X ... -->) or raw text.
   - Compiles dynamic Pydantic schema using build_dynamic_schema.
   - Executes Layer 3 extraction using either 'graph_memory' or 'page_scan' strategy.
   - Persists extracted JSON and document graph memory to DB (PostgreSQL / SQLite)
     and to checkpoint files in dataset_output/.

2. Interactive Chatbot & Query Bot:
   - Interactive terminal chatbot: ask questions about the document in natural language.
   - One-shot question answering via `--query "<question>"`.
   - Accurate grounded answers with verbatim evidence and source page citations.
   - End-to-end integration test with FastAPI's `/api/query-bot/ask` endpoint.

3. Two Ways to Run:
   a) As an automated test suite with pytest:
      pytest tests/test_layer3_pipeline.py -v

   b) As an interactive standalone CLI script:
      python tests/test_layer3_pipeline.py --md tests/fixtures/MY_resume.md --schema tests/schemas/schema.json
      python tests/test_layer3_pipeline.py --md sdg_goals_output --schema schema --query "What is the first goal?"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root and schema_chatbot_v2 are in Python path
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCHEMA_APP_DIR = ROOT_DIR / "schema_chatbot_v2"
if str(SCHEMA_APP_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_APP_DIR))

from pydantic import BaseModel, Field
import pytest

# Core Layer 3 imports
from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import extract_document
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import NodeEvidence
from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryResult, GraphQueryService
from src.ai.layer3_extraction.page_loader import PAGE_CLOSING_MARKER, PAGE_MARKER
from src.ai.layer3_extraction.storage import (
    init_db,
    save_document_graph,
    save_extraction_run,
)
from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema
from src.config.settings import settings


# ==============================================================================
# 1. Flexible Input Loaders (Markdown Pages & JSON Schemas)
# ==============================================================================

def load_markdown_pages(
    md_input: Union[str, Path],
    max_pages: Optional[int] = None,
    selected_pages: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Loads and normalizes an input markdown document into page dictionaries:
    [{"page_number": int, "markdown": str}, ...]
    Supports page slicing via max_pages or explicit selected_pages.
    """
    path_cand = Path(md_input)
    doc_path: Optional[Path] = None
    candidates: List[Path] = [path_cand]

    if path_cand.is_file():
        doc_path = path_cand
    else:
        candidates = [
            ROOT_DIR / "tests" / "fixtures" / f"{md_input}.md",
            ROOT_DIR / "tests" / "fixtures" / str(md_input),
            ROOT_DIR / "dataset_output" / f"{md_input}.md",
            ROOT_DIR / "dataset_output" / str(md_input),
            ROOT_DIR / f"{md_input}.md",
            ROOT_DIR / str(md_input),
        ]
        for c in candidates:
            if c.is_file():
                doc_path = c
                break

    if not doc_path or not doc_path.exists():
        raise FileNotFoundError(
            f"Could not find markdown file for: {md_input}. "
            f"Checked: {[str(c) for c in candidates]}"
        )

    doc_name = doc_path.stem
    content = doc_path.read_text(encoding="utf-8")

    # Check for PAGE_MARKER comments (<!-- PAGE 1 | ... -->)
    matches = list(PAGE_MARKER.finditer(content))
    if not matches:
        return [{"page_number": 1, "markdown": content.strip()}], doc_name

    pages: List[Dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        raw_slice = content[start:end].strip()
        cleaned_markdown = PAGE_CLOSING_MARKER.split(raw_slice)[0].strip()
        page_num = int(match.group(1))

        if selected_pages and page_num not in selected_pages:
            continue

        pages.append({
            "page_number": page_num,
            "markdown": cleaned_markdown,
        })

        if max_pages and len(pages) >= max_pages:
            break

    return pages, doc_name


def load_target_schema(schema_input: Union[str, Path, dict, list]) -> Tuple[type[BaseModel], List[Dict[str, str]], str]:
    """
    Loads an arbitrary schema into a dynamic Pydantic model.
    """
    raw_data: Any = None
    schema_name = "custom_schema"

    if isinstance(schema_input, (dict, list)):
        raw_data = schema_input
    else:
        path_cand = Path(schema_input)
        schema_path: Optional[Path] = None
        candidates: List[Path] = [path_cand]

        if path_cand.is_file():
            schema_path = path_cand
        else:
            candidates = [
                ROOT_DIR / "tests" / "schemas" / f"{schema_input}.json",
                ROOT_DIR / "tests" / "schemas" / str(schema_input),
                ROOT_DIR / "schema_registry" / f"{schema_input}.json",
                ROOT_DIR / "schema_registry" / str(schema_input),
                ROOT_DIR / f"{schema_input}.json",
                ROOT_DIR / str(schema_input),
            ]
            for c in candidates:
                if c.is_file():
                    schema_path = c
                    break

        if not schema_path or not schema_path.exists():
            raise FileNotFoundError(
                f"Could not find schema file for: {schema_input}. "
                f"Checked: {[str(c) for c in candidates]}"
            )

        schema_name = schema_path.stem
        with open(schema_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    # Normalize various JSON structures:
    raw_fields: List[Any] = []
    if isinstance(raw_data, list):
        raw_fields = raw_data
    elif isinstance(raw_data, dict):
        if "schema" in raw_data and isinstance(raw_data["schema"], dict) and "fields" in raw_data["schema"]:
            raw_fields = raw_data["schema"]["fields"]
        elif "fields" in raw_data and isinstance(raw_data["fields"], list):
            raw_fields = raw_data["fields"]
        elif "properties" in raw_data and isinstance(raw_data["properties"], dict):
            raw_fields = [
                {"name": k, "description": v.get("description", k)}
                for k, v in raw_data["properties"].items()
            ]
        else:
            raise ValueError(f"Unrecognized schema structure in {schema_name}")

    fields_in: List[SchemaFieldIn] = []
    field_descriptors: List[Dict[str, str]] = []

    for item in raw_fields:
        if isinstance(item, dict):
            name = item.get("name") or item.get("field") or "field"
            desc = item.get("description") or name
            fields_in.append(SchemaFieldIn(name=name, description=desc))
            field_descriptors.append({"name": name, "description": desc})
        elif isinstance(item, str):
            fields_in.append(SchemaFieldIn(name=item, description=item))
            field_descriptors.append({"name": item, "description": item})

    schema_model = build_dynamic_schema(fields_in)
    return schema_model, field_descriptors, schema_name


# ==============================================================================
# 2. Layer 3 Execution Runner
# ==============================================================================

def run_layer3_only(
    md_file: Union[str, Path],
    schema_file: Union[str, Path, dict, list],
    strategy: str = "graph_memory",
    llm: Optional[ExtractionLLMClient] = None,
    save_to_db: bool = True,
    owner: str = "admin",
    max_pages: Optional[int] = None,
    selected_pages: Optional[List[int]] = None,
    use_cached: bool = False,
) -> Dict[str, Any]:
    """
    Executes ONLY Layer 3 extraction on the given markdown document and target schema.
    Supports max_pages/selected_pages filtering and use_cached fast reload.
    """
    pages, doc_id = load_markdown_pages(md_file, max_pages=max_pages, selected_pages=selected_pages)
    schema_model, field_descriptors, schema_name = load_target_schema(schema_file)

    out_dir = ROOT_DIR / "dataset_output"
    cached_extracted = out_dir / f"{doc_id}.extracted.json"
    cached_graph = out_dir / f"{doc_id}.graph.json"

    # Fast path: load precomputed extraction & graph memory if requested
    if use_cached and cached_extracted.is_file():
        print(f"[*] Loading precomputed Layer 3 extraction from {cached_extracted.name}...")
        cdata = json.loads(cached_extracted.read_text(encoding="utf-8"))
        extracted_dict = cdata.get("extracted_data") or cdata
        graph_obj: Optional[GraphMemory] = None
        if cached_graph.is_file():
            gdata = json.loads(cached_graph.read_text(encoding="utf-8"))
            graph_obj = GraphMemory.from_dict(gdata)
            print(f"[*] Loaded GraphMemory: {len(graph_obj.nodes)} entities, {len(graph_obj.edges)} relationships.")
        return {
            "doc_id": doc_id,
            "schema_name": schema_name,
            "page_count": len(pages),
            "extracted_data": extracted_dict,
            "graph": graph_obj,
            "elapsed_seconds": cdata.get("processing_time_seconds", 0.0),
            "db_run_id": None,
        }

    if llm is None:
        llm = get_extraction_client()

    init_db()

    graph_out: Dict[str, Any] = {}
    start_time = time.time()

    print(f"[*] Running Layer 3 Extraction...")
    print(f"   - Document       : {doc_id} ({len(pages)} pages)")
    print(f"   - Target Schema  : {schema_name} ({len(field_descriptors)} fields)")
    print(f"   - Strategy       : {strategy}")

    extracted_model = extract_document(
        pages_md=pages,
        schema=schema_model,
        llm=llm,
        strategy=strategy,
        graph_out=graph_out,
    )
    elapsed = round(time.time() - start_time, 2)
    extracted_dict = extracted_model.model_dump()

    # Retrieve graph object if available
    graph_obj: Optional[GraphMemory] = None
    if strategy in ("graph_memory", "graph_memory_concurrent") and "snapshot" in graph_out:
        graph_obj = GraphMemory.from_dict(graph_out["snapshot"])

    db_run_id: Optional[int] = None
    if save_to_db:
        db_run_id = save_extraction_run(
            doc_id=doc_id,
            page_count=len(pages),
            schema_name=schema_name,
            result_json=extracted_dict,
            llm_provider=getattr(settings, "extraction_backend", "sarvam"),
            model_name=getattr(settings, "sarvam_model_name", "sarvam-105b"),
            processing_time_seconds=elapsed,
            page_outputs=None,
        )

        if graph_obj:
            save_document_graph(
                doc_id=f"{doc_id}.pdf",
                graph_dict=graph_obj.to_dict(),
                job_id=f"job_{doc_id}",
                owner=owner,
                schema_id=schema_name,
                strategy=strategy,
            )

        out_dir = ROOT_DIR / "dataset_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id}.extracted.json").write_text(
            json.dumps({
                "doc_id": doc_id,
                "schema_name": schema_name,
                "strategy": strategy,
                "processing_time_seconds": elapsed,
                "extracted_data": extracted_dict,
            }, indent=2),
            encoding="utf-8",
        )
        if graph_obj:
            (out_dir / f"{doc_id}.graph.json").write_text(
                json.dumps(graph_obj.to_dict(), indent=2),
                encoding="utf-8",
            )

    return {
        "doc_id": doc_id,
        "schema_name": schema_name,
        "page_count": len(pages),
        "extracted_data": extracted_dict,
        "graph": graph_obj,
        "elapsed_seconds": elapsed,
        "db_run_id": db_run_id,
        "graph_metadata": graph_out,
    }


# ==============================================================================
# 3. Chatbot & Query Bot Service
# ==============================================================================

def ask_query_bot(
    question: str,
    graph: Optional[GraphMemory] = None,
    extracted_data: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Answers natural language questions about the document.
    Prioritizes GraphQueryService (multi-hop traversal + page evidence) when graph
    is available, and cleanly falls back to extracted JSON data otherwise.
    """
    if not question.strip():
        return {"answer": "Please ask a valid question.", "sources": [], "mode": "none"}

    # Mode 1: Document Knowledge Graph reasoning
    if graph and len(graph.nodes) > 0:
        query_service = GraphQueryService()
        result: GraphQueryResult = query_service.query(graph=graph, question=question)
        return {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
            "mode": "graph",
            "hops_traversed": result.hops_traversed,
        }

    # Mode 2: Extracted JSON Fallback assistant
    import httpx
    api_key = os.getenv("IDP_SARVAM_API_KEY") or os.getenv("SARVAM_API_KEY") or ""
    base_url = (os.getenv("IDP_SARVAM_BASE_URL") or os.getenv("SARVAM_BASE_URL") or "https://api.sarvam.ai/v1").rstrip("/")
    model_name = os.getenv("IDP_SARVAM_MODEL_NAME") or os.getenv("SARVAM_MODEL") or "sarvam-105b"

    data_json = json.dumps(extracted_data or {}, indent=2)
    prompt = (
        "You are a factual document QA assistant. Answer the user's question directly based on the extracted data below.\n\n"
        f"EXTRACTED DATA:\n{data_json}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a concise, direct answer in 1 or 2 sentences."
    )

    if api_key:
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 400,
                },
                headers={
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            if resp.is_success:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                content = re.sub(r"^(?:Assistant|Answer):\s*", "", content)
                return {"question": question, "answer": content, "sources": [], "mode": "json_fallback"}
        except Exception:
            pass

    # Deterministic fallback
    norm_q = question.lower()
    for k, v in (extracted_data or {}).items():
        if k.replace("_", " ") in norm_q or norm_q in str(v).lower():
            return {
                "question": question,
                "answer": f"{k.replace('_', ' ').title()}: {v}",
                "sources": [],
                "mode": "deterministic_fallback",
            }

    return {
        "question": question,
        "answer": "The requested information could not be found in the extracted document data.",
        "sources": [],
        "mode": "not_found",
    }


def run_interactive_chatbot(
    doc_id: str,
    graph: Optional[GraphMemory] = None,
    extracted_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Runs an interactive terminal chatbot loop with the user."""
    print("\n" + "=" * 70)
    print(f"[CHATBOT] LAYER 3 CHATBOT & QUERY BOT -- Document: {doc_id}")
    print("=" * 70)
    print("Ask any question about the document in plain English.")
    print("Commands: 'exit' or 'quit' to stop | 'data' for JSON | 'graph' for stats")
    print("-" * 70)

    while True:
        try:
            q = input("\nQuery Bot > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("\nSession ended. Goodbye!\n")
                break
            if q.lower() == "data":
                print("\n=== EXTRACTED DATA ===")
                print(json.dumps(extracted_data or {}, indent=2))
                continue
            if q.lower() == "graph":
                print("\n=== GRAPH MEMORY STATS ===")
                if graph:
                    stats = graph.stats()
                    print(f"Entities (Nodes) : {stats.get('node_count')}")
                    print(f"Relationships    : {stats.get('edge_count')}")
                    print(f"Entity Types     : {list(stats.get('type_distribution', {}).keys())}")
                else:
                    print("No GraphMemory active for this run.")
                continue

            res = ask_query_bot(q, graph=graph, extracted_data=extracted_data, doc_id=doc_id)
            print(f"\n[Answer]: {res['answer']}")
            if res.get("sources"):
                print("[Sources]:")
                for s in res["sources"]:
                    node_label = f" ({s['node']})" if s.get("node") else ""
                    print(f"   * Page {s.get('page', '?')}{node_label}: \"{s.get('evidence', '')}\"")
        except (KeyboardInterrupt, EOFError):
            print("\nSession closed.")
            break


# ==============================================================================
# 4. Pytest Automated Tests
# ==============================================================================

class MockTestExtractionLLM(ExtractionLLMClient):
    """Compliant mock LLM for unit tests without external API calls."""

    def check_page_for_fields(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> List[Dict[str, Any]]:
        results = []
        for f in schema_fields:
            name = f["name"]
            if "name" in name or "title" in name:
                results.append({"field": name, "value": "Sample Document Title", "confidence": 0.95})
            elif "goal" in name:
                results.append({"field": name, "value": "End poverty everywhere", "confidence": 0.92})
            elif "target" in name or "total" in name:
                results.append({"field": name, "value": "17", "confidence": 0.90})
        return results

    def extract(self, content: str, schema: type[BaseModel], **kwargs) -> BaseModel:
        init_args = {}
        for k in schema.model_fields.keys():
            if "title" in k or "name" in k:
                init_args[k] = "Test Title"
            elif "goal" in k:
                init_args[k] = "Goal 1: Test"
            else:
                init_args[k] = "100"
        return schema(**init_args)

    def summarize_page(self, page_md: str, max_words: int = 100) -> str:
        return "Page summary"

    def navigate(self, page_summaries: List[str], schema_fields: List[str]) -> Dict[str, List[int]]:
        return {f: [1] for f in schema_fields}

    def extract_graph_from_page(
        self,
        page_md: str,
        schema_fields: List[Dict[str, str]],
        existing_nodes: List[Dict[str, Any]],
        page_number: int = 0,
        total_pages: int = 0,
    ) -> Dict[str, Any]:
        return {
            "entities": [
                {
                    "node_id": "test_candidate",
                    "type": "Person",
                    "label": "candidate_name",
                    "value": "Alex Morgan",
                    "source_page": page_number,
                    "evidence": "Alex Morgan is a Lead AI Architect",
                }
            ],
            "relationships": [],
        }

    def extract_page_delta(self, page_md: str, page_number: int, total_pages: int, context_entities: List[Dict]) -> Dict:
        return self.extract_graph_from_page(page_md, [], context_entities, page_number, total_pages)

    def resolve_schema_from_graph(self, graph_evidence: str, schema: type[BaseModel]) -> BaseModel:
        init_args = {}
        for k in schema.model_fields.keys():
            init_args[k] = "Resolved Value"
        return schema(**init_args)


def test_layer3_loaders():
    """Verify markdown pages and JSON schemas are loaded and parsed properly."""
    pages, doc_name = load_markdown_pages("tests/fixtures/MY_resume.md")
    assert len(pages) >= 1
    assert pages[0]["page_number"] == 1
    assert "PROFILE" in pages[0]["markdown"]

    model, fields, schema_name = load_target_schema("tests/schemas/schema.json")
    assert len(fields) >= 3
    assert "document_title" in model.model_fields


def test_layer3_execution_mock():
    """Verify Layer 3 runs end-to-end with Mock LLM using page_scan strategy."""
    mock_llm = MockTestExtractionLLM()
    result = run_layer3_only(
        md_file="tests/fixtures/MY_resume.md",
        schema_file="tests/schemas/schema.json",
        strategy="page_scan",
        llm=mock_llm,
        save_to_db=False,
    )
    assert result["doc_id"] == "MY_resume"
    assert isinstance(result["extracted_data"], dict)
    assert len(result["extracted_data"]) > 0


def test_layer3_chatbot_graph_reasoning():
    """Verify GraphQueryService performs multi-hop retrieval and gives page citations."""
    graph = GraphMemory()
    graph.create_node(
        node_id="p1",
        node_type="Person",
        label="Candidate Name",
        value="Vinay KY",
        source_page=1,
        evidence=NodeEvidence(page_number=1, text="VINAY KY - CSE AIML Student"),
    )
    graph.create_node(
        node_id="u1",
        node_type="University",
        label="University",
        value="Dayananda Sagar University",
        source_page=1,
        evidence=NodeEvidence(page_number=1, text="Dayananda Sagar University - B.Tech in CSE"),
    )
    graph.create_edge("e1", "p1", "u1", "ATTENDED", source_page=1)

    res = ask_query_bot("Who is the candidate?", graph=graph)
    assert res["mode"] == "graph"
    assert "Vinay KY" in res["answer"]
    assert any(s["page"] == 1 for s in res["sources"])


def test_layer3_fastapi_query_bot_api():
    """Verify the web application's POST /api/query-bot/ask route responds."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.storage.user_store import Role, get_user_store
    from app.core.auth import create_access_token

    store = get_user_store()
    user = store.get_by_username("admin")
    if not user:
        user = store.create("admin", "changeme", Role.ADMIN)
    token = create_access_token(user)

    client = TestClient(app)
    resp = client.post(
        "/api/query-bot/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "question": "What is the document title?",
            "extracted_data": {"document_title": "SDG Youth Summer Camp"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "answer" in data


# ==============================================================================
# 5. Standalone Interactive CLI Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Layer 3 extraction on a markdown file & schema with an interactive Chatbot."
    )
    parser.add_argument(
        "--md",
        default="tests/fixtures/MY_resume.md",
        help="Path or name of the markdown file (default: tests/fixtures/MY_resume.md)",
    )
    parser.add_argument(
        "--schema",
        default="tests/schemas/schema.json",
        help="Path or name of the JSON schema file (default: tests/schemas/schema.json)",
    )
    parser.add_argument(
        "--strategy",
        choices=["graph_memory", "graph_memory_concurrent", "page_scan"],
        default="graph_memory",
        help="Layer 3 strategy: graph_memory, graph_memory_concurrent, or page_scan",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Optional one-shot question to ask the Query Bot immediately after extraction.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Launch interactive conversation chatbot session (default: True).",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Skip interactive chatbot prompt after extraction.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process from the markdown file.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated list of page numbers to process (e.g. 1,2,5,9).",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="Load precomputed extraction & graph from dataset_output/ if available.",
    )
    parser.add_argument(
        "--no-db",
        dest="save_to_db",
        action="store_false",
        default=True,
        help="Skip saving results to PostgreSQL/SQLite database.",
    )
    args = parser.parse_args()

    # Parse selected pages
    selected_pages = None
    if args.pages:
        selected_pages = [int(p.strip()) for p in args.pages.split(",") if p.strip().isdigit()]

    try:
        reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure_stdout) and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            reconfigure_stdout(encoding="utf-8", errors="replace")
        reconfigure_stderr = getattr(sys.stderr, "reconfigure", None)
        if callable(reconfigure_stderr) and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
            reconfigure_stderr(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 75)
    print("       INTELLIGENT DOCUMENT PROCESSING -- LAYER 3 TEST HARNESS")
    print("=" * 75)

    # 1. Run Layer 3
    result = run_layer3_only(
        md_file=args.md,
        schema_file=args.schema,
        strategy=args.strategy,
        save_to_db=args.save_to_db,
        max_pages=args.max_pages,
        selected_pages=selected_pages,
        use_cached=args.cached,
    )

    print("\n" + "-" * 75)
    print(f"[OK] Extraction completed in {result['elapsed_seconds']}s (DB Run ID: {result['db_run_id']})")
    print("-" * 75)
    print(json.dumps(result["extracted_data"], indent=2))
    print("-" * 75)

    # 2. Handle One-Shot Query if supplied
    if args.query:
        print(f"\n[?] Asking Query Bot: \"{args.query}\"")
        ans = ask_query_bot(
            question=args.query,
            graph=result["graph"],
            extracted_data=result["extracted_data"],
            doc_id=result["doc_id"],
        )
        print(f"[Answer]: {ans['answer']}")
        if ans.get("sources"):
            print("[Sources]:")
            for s in ans["sources"]:
                print(f"   * Page {s.get('page')}: \"{s.get('evidence')}\"")

    # 3. Handle Interactive Chatbot Session
    if args.interactive and not args.query:
        run_interactive_chatbot(
            doc_id=result["doc_id"],
            graph=result["graph"],
            extracted_data=result["extracted_data"],
        )


if __name__ == "__main__":
    main()
