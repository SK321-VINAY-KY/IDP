"""
File: extractor.py
Purpose: Layer 3 extraction — multi-page field scanning and hardened Graph Memory extraction.

Supports three explicit Graph Memory concurrency modes:
- Mode A (Ordered Context Mode): Pages processed sequentially; Page N observes canonical
  memory produced by pages 1..N-1.
- Mode B (Concurrent Delta Mode): Pages extracted concurrently up to concurrency_limit;
  workers receive dispatch-time context snapshots; cross-page consistency recovered
  through serialized canonical merge and post-merge reconciliation.
- Mode C (Epoch / Batch Mode): Pages partitioned into epochs; concurrent extraction within
  an epoch, followed by canonical merge so the next epoch sees fresh canonical context.

Hardened with:
- PageDelta validation before persistence and canonical mutation (malformed deltas never reach graph).
- Durable page checkpoints (PostgreSQL / SQLite) with deterministic idempotency.
- Crash-safe recovery: skip completed pages and replay deltas idempotently on worker restart.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
    **kwargs: Any,
) -> BaseModel:
    """
    Scan every page against the cached target schema fields.
    Collects findings for ALL fields page-by-page into the Scratchpad,
    then provides the Scratchpad evidence to the LLM to produce the final
    hydrated Pydantic model.
    """
    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    schema_fields: List[Dict[str, str]] = [
        {"name": k, "description": v.description or k}
        for k, v in schema.model_fields.items()
    ]
    field_names = [f["name"] for f in schema_fields]
    scratchpad = Scratchpad(schema_field_names=field_names)

    logger.info("layer3.page_scan.start", total_pages=total_pages, schema_fields=field_names)

    for page_data in pages:
        page_md = page_data["markdown"]
        page_number = page_data["page_number"]

        if not page_md.strip():
            logger.debug("layer3.page_scan.skip_empty", page_number=page_number)
            continue

        matches = llm.check_page_for_fields(
            page_md,
            schema_fields,
            page_number=page_number,
            total_pages=total_pages,
        )
        scratchpad.update(page_number, matches)

    scratchpad_values = scratchpad.to_values_dict()
    if not scratchpad_values:
        evidence_text = "\n\n".join([p["markdown"] for p in pages if p.get("markdown")])
    else:
        evidence_text = scratchpad.format_evidence_for_llm()

    try:
        result = llm.extract(evidence_text, schema)
    except Exception as exc:
        logger.warning("layer3.llm_final_extraction.fallback_to_scratchpad", error=str(exc))
        try:
            default_instance = schema()
        except Exception:
            default_instance = None
        default_values: Dict[str, Any] = default_instance.model_dump() if default_instance is not None else {}
        result = schema.model_validate({**default_values, **scratchpad_values})

    res_dict = result.model_dump()
    merged = dict(res_dict)
    for k, v in scratchpad_values.items():
        if k in merged and (merged[k] in ("", None)) and v not in ("", None):
            merged[k] = v
    if merged != res_dict:
        result = schema.model_validate(merged)

    if graph_out is not None:
        graph_out["strategy"] = "page_scan"
        graph_out["stats"] = {"total_nodes": 0, "total_edges": 0, "strategy": "page_scan"}
        graph_out["scratchpad"] = scratchpad.snapshot()
        graph_out["provenance"] = scratchpad.provenance()
    return result


async def extract_with_graph_memory_ordered(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE A: ORDERED CONTEXT MODE.
    Processes pages in sequence: Page N can observe canonical memory produced by pages 1..N-1.
    LLM inference executes outside the canonical merge lock.
    Enforces the hardened lifecycle: Observe -> PageDelta -> Validate -> Checkpoint -> Merge -> Reconcile -> Resolve.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        get_page_checkpoint,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)

    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    logger.info(
        "layer3.graph_memory_ordered.start",
        total_pages=total_pages,
        doc_id=doc_id,
        anchor_types=manager.anchor_types,
    )

    pages_succeeded: List[int] = []
    pages_failed: List[int] = []

    for page_data in pages:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")

        if not str(page_md or "").strip():
            pages_succeeded.append(page_num)
            continue

        # Check durable checkpoint for crash recovery
        replayed_delta: Optional[PageDelta] = None
        replayed_checksum: Optional[str] = None
        if doc_id:
            cp = get_page_checkpoint(doc_id, page_num, pipeline_version=pipeline_version)
            if cp and cp.get("status") in ("DELTA_PERSISTED", "COMPLETED", "MERGED") and cp.get("page_delta_json"):
                try:
                    replayed_delta = PageDelta.from_dict(cp["page_delta_json"])
                    replayed_checksum = cp.get("delta_checksum")
                    logger.info(
                        "page.checkpoint.recovered",
                        doc_id=doc_id,
                        page_number=page_num,
                        checksum=replayed_checksum,
                    )
                except Exception as exc:
                    logger.warning("page.checkpoint.recovery_failed", page=page_num, error=str(exc))

        if replayed_delta is not None:
            # Replay already verified delta idempotently
            await manager.ingest_page_delta(replayed_delta, delta_checksum=replayed_checksum)
            pages_succeeded.append(page_num)
            continue

        # 1. Fetch targeted canonical context (Page N sees memory from pages 1..N-1)
        context = await manager.get_context_for_page(page_num)

        if doc_id:
            save_page_checkpoint(doc_id, page_num, status="PROCESSING", pipeline_version=pipeline_version)

        # 2. Build isolated PageDelta (LLM runs outside lock)
        try:
            delta = await agent.build_page_delta_async(
                page_number=page_num,
                markdown=page_md,
                total_pages=total_pages,
                context=context,
            )
        except Exception as exc:
            logger.error("layer3.ordered.worker_failed", page_number=page_num, error=str(exc))
            pages_failed.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", error_info=str(exc), pipeline_version=pipeline_version)
            continue

        # 3. Validate PageDelta
        val_res = validate_page_delta(delta)
        if not val_res.is_valid:
            err_msg = f"Validation failed: {', '.join(val_res.errors)}"
            logger.error("page.validation_failed", page_number=page_num, errors=val_res.errors)
            pages_failed.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", error_info=err_msg, pipeline_version=pipeline_version)
            continue

        delta_dict = delta.to_dict()
        checksum = compute_page_delta_checksum(delta_dict)

        # 4. Durable checkpoint BEFORE canonical mutation
        if doc_id:
            save_page_checkpoint(
                doc_id,
                page_num,
                status="DELTA_PERSISTED",
                page_delta_json=delta_dict,
                delta_checksum=checksum,
                pipeline_version=pipeline_version,
            )

        # 5. Canonical ingestion under manager lock
        await manager.ingest_page_delta(delta, delta_checksum=checksum)

        # 6. Mark page completed
        if doc_id:
            save_page_checkpoint(doc_id, page_num, status="COMPLETED", pipeline_version=pipeline_version)
        pages_succeeded.append(page_num)

    # 7. Reconcile cross-page references
    await manager.reconcile_cross_references()

    # 8. Schema resolution
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), strategy="graph_memory_ordered")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_ordered"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        graph_out["unresolved_reference_count"] = manager.unresolved_count
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

    return result


def extract_with_graph_memory(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Sequential / Ordered Graph-Memory Extraction Strategy.
    Wraps Mode A (Ordered Context Mode) synchronously for backward compatibility.
    """
    return _run_coroutine_sync(
        extract_with_graph_memory_ordered(
            pages_md=pages_md,
            schema=schema,
            llm=llm,
            graph_out=graph_out,
            doc_id=doc_id,
            **kwargs,
        )
    )


async def extract_with_graph_memory_concurrent(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE B: CONCURRENT DELTA MODE.
    Processes pages concurrently up to concurrency_limit with isolated PageDelta generation.
    SEMANTIC GUARANTEE:
    Concurrent workers receive a dispatch-time snapshot context; they do NOT receive dynamically
    changing canonical memory during parallel inference (concurrency != fresh shared memory).
    Cross-page references and consistency are recovered through serialized canonical merge
    under manager lock and post-merge reconciliation.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        list_page_checkpoints,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

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
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    logger.info(
        "layer3.graph_memory_concurrent.start",
        total_pages=total_pages,
        concurrency_limit=concurrency_limit,
        doc_id=doc_id,
        anchor_types=manager.anchor_types,
    )

    # Checkpoint recovery: check existing durable checkpoints
    recovered_pages: Set[int] = set()
    if doc_id:
        checkpoints = list_page_checkpoints(doc_id, pipeline_version=pipeline_version)
        for cp in checkpoints:
            if cp.get("status") in ("DELTA_PERSISTED", "COMPLETED", "MERGED") and cp.get("page_delta_json"):
                try:
                    replayed_delta = PageDelta.from_dict(cp["page_delta_json"])
                    chk = cp.get("delta_checksum")
                    await manager.ingest_page_delta(replayed_delta, delta_checksum=chk)
                    recovered_pages.add(cp["page_number"])
                    logger.info(
                        "page.checkpoint.recovered",
                        doc_id=doc_id,
                        page_number=cp["page_number"],
                        checksum=chk,
                    )
                except Exception as exc:
                    logger.warning("page.checkpoint.replay_failed", page=cp.get("page_number"), error=str(exc))

    semaphore = asyncio.Semaphore(concurrency_limit)
    pages_succeeded: List[int] = list(recovered_pages)
    pages_failed: List[int] = []
    failed_errors: Dict[int, str] = {}

    async def _process_page(page_data: Dict[str, Any]) -> None:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")

        if page_num in recovered_pages:
            return  # Already recovered from checkpoint

        if not str(page_md or "").strip():
            pages_succeeded.append(page_num)
            return

        async with semaphore:
            # 1. Fetch targeted dispatch-time context (snapshot at dispatch)
            context = await manager.get_context_for_page(page_num)

            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="PROCESSING", pipeline_version=pipeline_version)

            # 2. Build isolated PageDelta (LLM runs concurrently outside manager lock)
            try:
                delta = await agent.build_page_delta_async(
                    page_number=page_num,
                    markdown=page_md,
                    total_pages=total_pages,
                    context=context,
                )
            except Exception as exc:
                logger.error("layer3.concurrent.worker_failed", page_number=page_num, error=str(exc))
                pages_failed.append(page_num)
                failed_errors[page_num] = str(exc)
                if doc_id:
                    save_page_checkpoint(doc_id, page_num, status="FAILED", error_info=str(exc), pipeline_version=pipeline_version)
                return

        # 3. Validate PageDelta
        val_res = validate_page_delta(delta)
        if not val_res.is_valid:
            err_msg = f"Validation failed: {', '.join(val_res.errors)}"
            logger.error("page.validation_failed", page_number=page_num, errors=val_res.errors)
            pages_failed.append(page_num)
            failed_errors[page_num] = err_msg
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", error_info=err_msg, pipeline_version=pipeline_version)
            return

        delta_dict = delta.to_dict()
        checksum = compute_page_delta_checksum(delta_dict)

        # 4. Durable checkpoint BEFORE canonical merge
        if doc_id:
            save_page_checkpoint(
                doc_id,
                page_num,
                status="DELTA_PERSISTED",
                page_delta_json=delta_dict,
                delta_checksum=checksum,
                pipeline_version=pipeline_version,
            )

        # 5. Ingest delta into central GraphMemory (canonical merge under lock)
        await manager.ingest_page_delta(delta, delta_checksum=checksum)

        # 6. Mark page completed
        if doc_id:
            save_page_checkpoint(doc_id, page_num, status="COMPLETED", pipeline_version=pipeline_version)
        pages_succeeded.append(page_num)

    # Launch concurrent worker tasks for non-recovered pages
    tasks = [_process_page(p) for p in pages if p.get("page_number") not in recovered_pages]
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

    # 7. Reconcile cross-page anaphoric references post-merge
    await manager.reconcile_cross_references()

    # 8. Synthesize final target schema strictly from the complete Graph Memory
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), strategy="graph_memory_concurrent")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

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
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

    return result


async def extract_with_graph_memory_batched(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    epoch_size: int = 4,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE C: EPOCH / BATCH MODE (Optional Strategy).
    Partitions pages into sequential epochs of size epoch_size.
    Within each epoch, workers execute concurrently; between epochs, PageDeltas are merged
    so the next epoch observes fresh canonical memory produced by all previous epochs.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

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
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    semaphore = asyncio.Semaphore(concurrency_limit)
    pages_succeeded: List[int] = []
    pages_failed: List[int] = []

    # Partition into epochs
    epochs = [pages[i : i + epoch_size] for i in range(0, len(pages), epoch_size)]

    for epoch_idx, epoch_pages in enumerate(epochs):
        logger.info(
            "layer3.epoch.start",
            epoch_index=epoch_idx + 1,
            total_epochs=len(epochs),
            page_count=len(epoch_pages),
        )

        epoch_deltas: List[Tuple[PageDelta, str]] = []

        async def _process_epoch_page(page_data: Dict[str, Any]) -> None:
            page_num = page_data.get("page_number", 1)
            page_md = page_data.get("markdown", "")

            if not str(page_md or "").strip():
                pages_succeeded.append(page_num)
                return

            async with semaphore:
                context = await manager.get_context_for_page(page_num)
                try:
                    delta = await agent.build_page_delta_async(
                        page_number=page_num,
                        markdown=page_md,
                        total_pages=total_pages,
                        context=context,
                    )
                except Exception as exc:
                    logger.error("layer3.epoch.worker_failed", page=page_num, error=str(exc))
                    pages_failed.append(page_num)
                    return

            val_res = validate_page_delta(delta)
            if not val_res.is_valid:
                logger.error("page.validation_failed", page=page_num, errors=val_res.errors)
                pages_failed.append(page_num)
                return

            delta_dict = delta.to_dict()
            chk = compute_page_delta_checksum(delta_dict)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="DELTA_PERSISTED", page_delta_json=delta_dict, delta_checksum=chk)
            epoch_deltas.append((delta, chk))

        tasks = [_process_epoch_page(p) for p in epoch_pages]
        if tasks:
            await asyncio.gather(*tasks)

        # Merge epoch deltas into canonical memory before next epoch begins
        for d, chk in epoch_deltas:
            await manager.ingest_page_delta(d, delta_checksum=chk)
            if doc_id:
                save_page_checkpoint(doc_id, d.page_number, status="COMPLETED")
            pages_succeeded.append(d.page_number)

    await manager.reconcile_cross_references()
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), strategy="graph_memory_batched")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_batched"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        graph_out["unresolved_reference_count"] = manager.unresolved_count
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

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
    doc_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Unified entry point for Layer 3 extraction with strategy switch.
    Supported strategies:
    - "page_scan": Plain sequential page scanning with Scratchpad.
    - "graph_memory" / "graph_memory_ordered": Mode A (Ordered Context Mode).
    - "graph_memory_concurrent": Mode B (Concurrent Delta Mode).
    - "graph_memory_batched" / "graph_memory_epoch": Mode C (Epoch / Batch Mode).
    """
    from src.config.settings import settings

    selected_strategy = strategy or getattr(settings, "layer3_strategy", "graph_memory")

    if selected_strategy in ("graph_memory_concurrent",):
        return _run_coroutine_sync(
            extract_with_graph_memory_concurrent(
                pages_md=pages_md,
                schema=schema,
                llm=llm,
                graph_out=graph_out,
                doc_id=doc_id,
                **kwargs,
            )
        )
    elif selected_strategy in ("graph_memory_batched", "graph_memory_epoch"):
        return _run_coroutine_sync(
            extract_with_graph_memory_batched(
                pages_md=pages_md,
                schema=schema,
                llm=llm,
                graph_out=graph_out,
                doc_id=doc_id,
                **kwargs,
            )
        )
    elif selected_strategy in ("graph_memory", "graph_memory_ordered"):
        return extract_with_graph_memory(
            pages_md=pages_md,
            schema=schema,
            llm=llm,
            graph_out=graph_out,
            doc_id=doc_id,
            **kwargs,
        )
    return extract_by_page_scan(pages_md, schema, llm, graph_out=graph_out, **kwargs)
