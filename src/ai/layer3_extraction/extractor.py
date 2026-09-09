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
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.scratchpad import Scratchpad
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_by_page_scan(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
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
    if graph_out is not None:
        graph_out["strategy"] = "page_scan"
        graph_out["stats"] = {"total_nodes": 0, "total_edges": 0, "strategy": "page_scan"}
        graph_out["scratchpad"] = scratchpad.snapshot()
        graph_out["provenance"] = scratchpad.provenance()
    return result


def extract_with_graph_memory(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Layer 3 Graph-Memory Extraction Strategy.
    Processes pages sequentially, maintaining an evolving contextual knowledge graph
    with cross-page reference resolution, entity reuse, and relationship provenance,
    then resolves the final target schema strictly from graph memory.
    """
    from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph

    # Normalise input — support both list[dict] and list[str]
    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    graph = GraphMemory()
    agent = GraphExtractionAgent(schema=schema, graph=graph, llm=llm)

    logger.info("layer3.graph_memory.start", total_pages=total_pages)

    # Ingest every non-empty page sequentially into Graph Memory
    for page_data in pages:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")
        agent.process_page(page_number=page_num, markdown=page_md, total_pages=total_pages)

    # Synthesize final target schema strictly from the complete Graph Memory
    result = resolve_schema_from_graph(graph=graph, schema=schema, llm=llm)
    logger.info("layer3.graph_memory.done", graph_stats=graph.stats())

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory"
        graph_out["stats"] = graph.stats()
        graph_out["snapshot"] = graph.snapshot()

    return result


async def extract_with_graph_memory_concurrent(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Concurrent Blackboard / Memory Manager Graph Extraction Strategy.
    Processes pages concurrently with isolated PageDelta generation,
    ingests deltas through a centralized GraphMemoryManager under an async lock,
    reconciles cross-page references post-merge, and resolves the target schema
    strictly from the final canonical GraphMemory.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.config.settings import settings

    # Normalise input — support both list[dict] and list[str]
    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)
    concurrency_limit = max(1, getattr(settings, "graph_concurrency_limit", 8))

    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
    )
    # Agent receives manager.graph for constructor compatibility, but build_page_delta_async does NOT mutate it
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    logger.info(
        "layer3.graph_memory_concurrent.start",
        total_pages=total_pages,
        concurrency_limit=concurrency_limit,
        anchor_types=manager.anchor_types,
    )

    semaphore = asyncio.Semaphore(concurrency_limit)
    pages_succeeded: List[int] = []
    pages_failed: List[int] = []
    failed_errors: Dict[int, str] = {}

    async def _process_page(page_data: Dict[str, Any]) -> None:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")

        if not str(page_md or "").strip():
            logger.debug("layer3.concurrent.skip_empty", page_number=page_num)
            pages_succeeded.append(page_num)
            return

        async with semaphore:
            # 1. Fetch targeted initial anchor context (outside LLM lock)
            context = await manager.get_context_for_page(page_num)

            # 2. Build isolated PageDelta (LLM call runs concurrently outside manager lock)
            try:
                delta = await agent.build_page_delta_async(
                    page_number=page_num,
                    markdown=page_md,
                    total_pages=total_pages,
                    context=context,
                )
            except Exception as exc:
                logger.error(
                    "layer3.concurrent.worker_failed",
                    page_number=page_num,
                    error=str(exc),
                )
                pages_failed.append(page_num)
                failed_errors[page_num] = str(exc)
                return

        # 3. Ingest delta into central GraphMemory (canonical merge under lock)
        await manager.ingest_page_delta(delta)
        pages_succeeded.append(page_num)

    # Launch concurrent worker tasks
    tasks = [_process_page(p) for p in pages]
    if tasks:
        await asyncio.gather(*tasks)

    pages_succeeded.sort()
    pages_failed.sort()

    if pages_failed:
        logger.warning(
            "layer3.graph_memory_concurrent.partial_failures",
            failed_pages=pages_failed,
            succeeded_pages=pages_succeeded,
            errors=failed_errors,
        )

    # 4. Reconcile cross-page anaphoric references post-merge
    await manager.reconcile_cross_references()

    # 5. Synthesize final target schema strictly from the complete Graph Memory
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    logger.info(
        "layer3.graph_memory_concurrent.done",
        graph_stats=manager.graph.stats(),
        pages_total=total_pages,
        pages_succeeded=len(pages_succeeded),
        pages_failed=len(pages_failed),
    )

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_concurrent"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        graph_out["concurrency_limit"] = concurrency_limit
        graph_out["unresolved_reference_count"] = manager.unresolved_count

    return result


def _run_coroutine_sync(coro):
    """
    Safely execute an asyncio coroutine from synchronous code,
    preventing RuntimeError when an event loop is already running in the current thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def extract_document(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    strategy: Optional[str] = None,
    graph_out: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Unified entry point for Layer 3 extraction with strategy switch.
    strategy: "page_scan", "graph_memory", or "graph_memory_concurrent" (defaults to settings.layer3_strategy).
    """
    from src.config.settings import settings

    selected_strategy = strategy or getattr(settings, "layer3_strategy", "graph_memory")
    if selected_strategy == "graph_memory_concurrent":
        return _run_coroutine_sync(
            extract_with_graph_memory_concurrent(
                pages_md=pages_md,
                schema=schema,
                llm=llm,
                graph_out=graph_out,
            )
        )
    elif selected_strategy == "graph_memory":
        return extract_with_graph_memory(pages_md, schema, llm, graph_out=graph_out)
    return extract_by_page_scan(pages_md, schema, llm, graph_out=graph_out)



