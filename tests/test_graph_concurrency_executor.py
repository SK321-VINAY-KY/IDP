"""
File: test_graph_concurrency_executor.py
Purpose: Verify the dedicated LLM thread pool in extract_with_graph_memory_concurrent.

The pool size must follow settings.graph_concurrency_limit (capped at 32), so admitted
pages each get a thread. Threads must be shut down after the build phase. Merge order
and results must not change. Logger correlation_id must reach the worker thread.
No network, no Bedrock: the LLM is a MagicMock with a fake extract function.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from src.ai.layer3_extraction import extractor
from src.utils.logger import correlation_id


class PatientClaimSchema(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Name of the patient")
    hospital_name: Optional[str] = Field(default=None, description="Name of the hospital")


def async_test(coro_fn):
    """Run an async test inside asyncio.run without requiring pytest-asyncio."""
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro_fn(*args, **kwargs))
    return wrapper


def _payload(page_number: int) -> Dict[str, Any]:
    return {
        "entities": [{"id": f"p{page_number}_e1", "type": "Patient", "value": f"Patient {page_number}"}],
        "relationships": [],
        "reference_resolutions": [],
        "quality": "clean",
    }


def _make_llm(extract_fn) -> MagicMock:
    llm = MagicMock()
    llm.extract_graph_from_page.side_effect = extract_fn
    llm.resolve_schema_from_graph.return_value = PatientClaimSchema(patient_name="Patient 1")
    return llm


def _pages(n: int) -> List[Dict[str, Any]]:
    return [{"markdown": f"Page content {i}", "page_number": i} for i in range(1, n + 1)]


def _llm_threads_alive() -> List[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("graph-llm")]


# -------------------------------------------------------------------------
# (a) Concurrency reaches the limit, not the small default pool size.
# -------------------------------------------------------------------------
@async_test
async def test_a_concurrency_reaches_limit(monkeypatch):
    from src.config.settings import settings
    limit = 16
    monkeypatch.setattr(settings, "graph_concurrency_limit", limit)

    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}
    barrier = threading.Barrier(limit, timeout=5)

    def fake_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        with lock:
            state["calls"] += 1
            idx = state["calls"]
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            if idx <= limit:
                barrier.wait()  # first `limit` calls must all be in flight together
            time.sleep(0.01)
        finally:
            with lock:
                state["active"] -= 1
        return _payload(page_number)

    # Simulate the 2-vCPU default executor (~6 threads) that asyncio.to_thread would use.
    small_default = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    asyncio.get_running_loop().set_default_executor(small_default)

    graph_out: Dict[str, Any] = {}
    try:
        await extractor.extract_with_graph_memory_concurrent(
            pages_md=_pages(20),
            schema=PatientClaimSchema,
            llm=_make_llm(fake_extract),
            owner="_admin",
            graph_out=graph_out,
        )
    finally:
        small_default.shutdown(wait=True)

    assert state["peak"] == limit, f"expected peak concurrency {limit}, got {state['peak']}"
    assert graph_out["failed_pages"] == []
    assert graph_out["pages_succeeded"] == 20


# -------------------------------------------------------------------------
# (b) Executor is shut down after the build (no leaked threads).
# -------------------------------------------------------------------------
@async_test
async def test_b_executor_shut_down_no_leaked_threads(monkeypatch):
    from src.config.settings import settings
    monkeypatch.setattr(settings, "graph_concurrency_limit", 4)

    def fake_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        return _payload(page_number)

    graph_out: Dict[str, Any] = {}
    await extractor.extract_with_graph_memory_concurrent(
        pages_md=_pages(6),
        schema=PatientClaimSchema,
        llm=_make_llm(fake_extract),
        owner="_admin",
        graph_out=graph_out,
    )

    assert _llm_threads_alive() == []


# -------------------------------------------------------------------------
# (c) Pool size is the concurrency limit, capped at 32.
# -------------------------------------------------------------------------
class _RecordingExecutor(concurrent.futures.ThreadPoolExecutor):
    sizes: List[Optional[int]] = []

    def __init__(self, *args, **kwargs):
        if kwargs.get("thread_name_prefix") == "graph-llm":
            _RecordingExecutor.sizes.append(kwargs.get("max_workers"))
        super().__init__(*args, **kwargs)


def _run_with_limit(monkeypatch, limit: int, n_pages: int) -> List[Optional[int]]:
    from src.config.settings import settings
    monkeypatch.setattr(settings, "graph_concurrency_limit", limit)
    _RecordingExecutor.sizes = []
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _RecordingExecutor)

    def fake_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        return _payload(page_number)

    asyncio.run(
        extractor.extract_with_graph_memory_concurrent(
            pages_md=_pages(n_pages),
            schema=PatientClaimSchema,
            llm=_make_llm(fake_extract),
            owner="_admin",
        )
    )
    return list(_RecordingExecutor.sizes)


def test_c_pool_size_follows_limit_capped_at_32(monkeypatch):
    assert _run_with_limit(monkeypatch, limit=4, n_pages=6) == [4]
    assert _run_with_limit(monkeypatch, limit=64, n_pages=3) == [32]


# -------------------------------------------------------------------------
# (d) Merge order and results are unchanged even when completion order is reversed.
# -------------------------------------------------------------------------
def _run_graph(monkeypatch, limit: int, sleep_by_page: bool, n_pages: int = 6) -> Dict[str, Any]:
    from src.config.settings import settings
    monkeypatch.setattr(settings, "graph_concurrency_limit", limit)

    def fake_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        if sleep_by_page:
            # Page 1 finishes last, page n finishes first.
            time.sleep(0.005 * (n_pages + 1 - page_number))
        return _payload(page_number)

    graph_out: Dict[str, Any] = {}
    asyncio.run(
        extractor.extract_with_graph_memory_concurrent(
            pages_md=_pages(n_pages),
            schema=PatientClaimSchema,
            llm=_make_llm(fake_extract),
            owner="_admin",
            graph_out=graph_out,
        )
    )
    return graph_out


def test_d_merge_order_and_results_unchanged(monkeypatch):
    reversed_out = _run_graph(monkeypatch, limit=6, sleep_by_page=True)
    serial_out = _run_graph(monkeypatch, limit=1, sleep_by_page=False)

    # Nodes are merged in ascending page order: source_pages must be non-decreasing.
    node_pages = [n["source_pages"][0] for n in reversed_out["snapshot"]["nodes"]]
    assert node_pages == sorted(node_pages)

    assert reversed_out["stats"] == serial_out["stats"]
    assert reversed_out["snapshot"] == serial_out["snapshot"]
    assert reversed_out["pages_succeeded"] == serial_out["pages_succeeded"] == 6
    assert reversed_out["failed_pages"] == serial_out["failed_pages"] == []


# -------------------------------------------------------------------------
# (e) Logger correlation_id is visible inside the worker thread.
# -------------------------------------------------------------------------
@async_test
async def test_e_correlation_id_propagates_to_worker_thread(monkeypatch):
    from src.config.settings import settings
    monkeypatch.setattr(settings, "graph_concurrency_limit", 4)

    seen: List[Optional[str]] = []
    seen_lock = threading.Lock()

    def fake_extract(page_md, schema_fields, existing_nodes, page_number, total_pages):
        with seen_lock:
            seen.append(correlation_id.get())
        return _payload(page_number)

    token = correlation_id.set("cid-test")
    try:
        await extractor.extract_with_graph_memory_concurrent(
            pages_md=_pages(5),
            schema=PatientClaimSchema,
            llm=_make_llm(fake_extract),
            owner="_admin",
        )
    finally:
        correlation_id.reset(token)

    assert len(seen) == 5
    assert all(cid == "cid-test" for cid in seen), seen
