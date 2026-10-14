"""
File: eval/context_instrumentation.py
Purpose: Close-out item 3 — measure per-page CONTEXT behaviour in concurrent mode,
         metadata-only (counts + dispatch position; NO document content).

Hooks GraphMemoryManager.get_context_for_page to record, per call:
  - page_number
  - number of graph nodes returned as context (len)
  - dispatch position (the order in which get_context_for_page was called this run)

Reports, over N runs of a page set: how many pages got non-empty context and the
node-count distribution. Works on whatever code is checked out (before/after the
merge fix) so it can be run under a git worktree for the before-commit measurement.

Also supports a settings flag (graph_concurrent_build_empty_context) if present:
when on, the build phase sees an empty graph (reproducible). This harness only
MEASURES; it does not change production behaviour.

Usage:
    python -m eval.context_instrumentation --pages 12 --runs 3 --out <path>
    python -m eval.context_instrumentation --pages 76 --runs 3 --out <path>
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# reuse the credential + logging-rollover setup
from eval.variance_attribution import _ensure_aws_creds, _patch_logging_rollover  # noqa
from eval.baseline_compare import (
    ADMIN_OWNER, CHANDER_MD, TARGETED_PAGES, _git_commit_and_status,
)
from scripts.benchmark_concurrency_modes import load_document_metadata, load_markdown_pages
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager


class ContextRecorder:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._dispatch_counter = 0

    def record(self, page_number: int, n_nodes: int):
        with self._lock:
            pos = self._dispatch_counter
            self._dispatch_counter += 1
            self.calls.append({"page": page_number, "context_nodes": n_nodes,
                               "dispatch_position": pos})


@contextlib.contextmanager
def hook_context(recorder: ContextRecorder):
    orig = GraphMemoryManager.get_context_for_page

    async def hooked(self_mgr, page_number):
        ctx = await orig(self_mgr, page_number)
        try:
            recorder.record(page_number, len(ctx))
        except Exception:
            pass
        return ctx

    GraphMemoryManager.get_context_for_page = hooked  # type: ignore
    try:
        yield
    finally:
        GraphMemoryManager.get_context_for_page = orig  # type: ignore


def _pages(mode: str):
    allp = load_markdown_pages(CHANDER_MD)
    if mode == "12":
        return [p for p in allp if p.get("page_number") in TARGETED_PAGES]
    return list(allp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", choices=["12", "76"], default="12")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    git_info = _git_commit_and_status()
    pages = _pages(args.pages)
    schema, _gt = load_document_metadata(CHANDER_MD)
    client = get_extraction_client()

    runs = []
    for i in range(args.runs):
        rec = ContextRecorder()
        with hook_context(rec):
            asyncio.run(extract_with_graph_memory_concurrent(
                pages_md=pages, schema=schema, llm=client, owner=ADMIN_OWNER, doc_id=None))
        n_calls = len(rec.calls)
        non_empty = sum(1 for c in rec.calls if c["context_nodes"] > 0)
        counts = [c["context_nodes"] for c in rec.calls]
        dist: Dict[str, int] = {}
        for c in counts:
            dist[str(c)] = dist.get(str(c), 0) + 1
        runs.append({
            "run": i + 1,
            "context_calls": n_calls,
            "pages_with_nonempty_context": non_empty,
            "node_count_distribution": dist,
            "max_context_nodes": max(counts) if counts else 0,
            "mean_context_nodes": round(statistics.mean(counts), 2) if counts else 0,
            "per_call": rec.calls,
        })
        print(f"  run {i+1}: calls={n_calls} non_empty={non_empty} "
              f"max_nodes={runs[-1]['max_context_nodes']} dist={dist}", flush=True)

    report = {"git": git_info, "page_mode": args.pages, "runs": args.runs, "run_data": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote -> {args.out} (commit {git_info['commit'][:10]} {git_info['working_tree']})")


if __name__ == "__main__":
    main()
