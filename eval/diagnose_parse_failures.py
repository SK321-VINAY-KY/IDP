"""
File: eval/diagnose_parse_failures.py
Purpose: STEP 1 DIAGNOSIS (no behavior change) for audit finding #1. Runs the graph
         extraction arm at an 8000-token graph-extraction budget on the 12-page Chander
         subset and classifies every graph-extraction call that production would treat
         as parse_failed, using METADATA-ONLY features. It NEVER logs, prints, stores,
         or persists any prompt text, document text, or model output text. Only derived
         booleans / lengths / counts leave this module.

How it stays non-invasive:
  * It wraps client._call_converse to stash, per worker thread and IN MEMORY ONLY, the
    most recent graph_extraction Converse response. The original call is executed
    unchanged and its result returned verbatim — behavior is identical.
  * It wraps client.extract_graph_from_page to run the ORIGINAL method (so production's
    own parse + salvage path decides the entity count), then re-derives the SAME raw
    text production saw and classifies it only when production would log parse_failed
    (json.loads fails AND the substring-{...} salvage also fails).
  * The stashed response is dropped immediately after feature extraction.

Classification features (all metadata-only):
  - stop_reason, output_tokens
  - starts_with_fence (after lstrip)
  - pre_json_len / post_json_len : chars before first '{' / after last '}'
  - braces_balanced : count('{') == count('}')
  - parses_after_fence_strip : json.loads(strip_fences(raw)) succeeds
  - multiple_top_level_objects : >1 '}{' boundary at depth 0 (approx via scan)
  - has_reasoning_block : a reasoningContent block exists in the response
  - salvage_entities : entities the production path recovered for this call
  - typical_entities_for_page : median recovered on clean calls for the same page

Usage:
    python -m eval.diagnose_parse_failures --target-failures 10 --max-runs 10 \
        --out dataset_output/parse_failure_diagnosis.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))

from eval.baseline_compare import ADMIN_OWNER, CHANDER_MD, DATASET_OUT, TARGETED_PAGES
from eval.baseline_fairness import GRAPH_MAXTOKENS_D, graph_maxtokens_override

from src.adapters.llm import llm_metrics as m
from src.adapters.llm.bedrock_client import (
    _extract_text_from_converse_response,
    _strip_fences,
)
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent
from src.ai.layer3_extraction.storage import delete_page_checkpoints
from scripts.benchmark_concurrency_modes import load_document_metadata, load_markdown_pages


# --------------------------------------------------------------------------- #
# Metadata-only feature extraction (NEVER returns content)
# --------------------------------------------------------------------------- #

def _count_salvageable(raw: str) -> Optional[int]:
    """Reproduce production's parse: returns entity count if json.loads OR substring
    salvage succeeds, else None (= production parse_failed). No content leaves."""
    stripped = _strip_fences(raw)
    for candidate in (stripped,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return len(parsed.get("entities", []) or [])
            return 0
        except Exception:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(stripped[start:end + 1])
            if isinstance(parsed, dict):
                return len(parsed.get("entities", []) or [])
            return 0
        except Exception:
            pass
    return None  # production parse_failed


def _has_reasoning_block(resp: Dict[str, Any]) -> bool:
    try:
        content = resp.get("output", {}).get("message", {}).get("content", [])
        return any(isinstance(b, dict) and "reasoningContent" in b for b in content)
    except Exception:
        return False


def _approx_top_level_objects(s: str) -> int:
    """Count depth-0 closures to approximate # of top-level {...} objects. Metadata only."""
    depth = 0
    closed = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                closed += 1
    return closed


def classify_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Return ONLY metadata features for one graph-extraction response."""
    raw = _extract_text_from_converse_response(resp) or ""
    stripped = _strip_fences(raw)
    lstripped = raw.lstrip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    usage = resp.get("usage") or {}
    out_tok = usage.get("outputTokens")
    salvage = _count_salvageable(raw)
    parses_after_fence = None
    json_err_class = None   # metadata only: JSONDecodeError msg CLASS, not content
    char_class_at_err = None
    try:
        json.loads(stripped)
        parses_after_fence = True
    except json.JSONDecodeError as je:
        parses_after_fence = False
        # Record only the error-kind phrase and the CHARACTER CLASS at the failure
        # position (never the character itself nor surrounding text).
        msg = str(je).split(":")[0].strip().lower()
        json_err_class = msg[:60]
        try:
            ch = stripped[je.pos] if 0 <= je.pos < len(stripped) else ""
            if ch == "":
                char_class_at_err = "eof"
            elif ch.isspace():
                char_class_at_err = "whitespace"
            elif ch in "{}[],:\"":
                char_class_at_err = "structural"
            elif ord(ch) < 0x20:
                char_class_at_err = "control_char"
            elif ch.isalnum():
                char_class_at_err = "alnum"
            else:
                char_class_at_err = "other_punct"
        except Exception:
            char_class_at_err = None
    except Exception:
        parses_after_fence = False
    return {
        "stop_reason": resp.get("stopReason"),
        "output_tokens": out_tok,
        "raw_len": len(raw),
        "starts_with_fence": lstripped.startswith("```"),
        "pre_json_len": first_brace if first_brace != -1 else len(stripped),
        "post_json_len": (len(stripped) - 1 - last_brace) if last_brace != -1 else 0,
        "braces_balanced": stripped.count("{") == stripped.count("}"),
        "open_braces": stripped.count("{"),
        "close_braces": stripped.count("}"),
        "parses_after_fence_strip": parses_after_fence,
        "approx_top_level_objects": _approx_top_level_objects(stripped),
        "has_reasoning_block": _has_reasoning_block(resp),
        "json_err_class": json_err_class,       # e.g. "expecting ',' delimiter"
        "char_class_at_err": char_class_at_err,  # structural/alnum/control_char/...
        "salvage_entities": salvage,          # None => production parse_failed
        "is_parse_failed": salvage is None,
    }


# --------------------------------------------------------------------------- #
# Instrumentation (in-memory stash; original behavior preserved)
# --------------------------------------------------------------------------- #

class Diagnoser:
    def __init__(self, client: Any):
        self.client = client
        self._lock = threading.Lock()
        self._stash: Dict[int, Dict[str, Any]] = {}  # thread_id -> last graph resp
        self.failed_features: List[Dict[str, Any]] = []
        # per (page) entity counts on clean (parsed) calls, to compute typical
        self.clean_entities_by_page: Dict[int, List[int]] = {}
        self._orig_call = client._call_converse
        self._orig_graph = client.extract_graph_from_page

    def install(self) -> None:
        orig_call = self._orig_call
        orig_graph = self._orig_graph
        stash = self._stash
        lock = self._lock

        def call_wrapper(*args, **kwargs):
            resp = orig_call(*args, **kwargs)
            if kwargs.get("_purpose") == m.PURPOSE_GRAPH_EXTRACTION:
                with lock:
                    stash[threading.get_ident()] = resp
            return resp

        def graph_wrapper(*args, **kwargs):
            result = orig_graph(*args, **kwargs)  # UNCHANGED behavior
            page_number = kwargs.get("page_number", 0)
            with lock:
                resp = stash.pop(threading.get_ident(), None)
            if resp is not None:
                feats = classify_response(resp)
                feats["page_number"] = page_number
                n_ent = len(result.get("entities", []) or []) if isinstance(result, dict) else 0
                if feats["is_parse_failed"]:
                    self.failed_features.append(feats)
                else:
                    self.clean_entities_by_page.setdefault(page_number, []).append(n_ent)
            return result

        self.client._call_converse = call_wrapper          # type: ignore
        self.client.extract_graph_from_page = graph_wrapper  # type: ignore

    def uninstall(self) -> None:
        self.client._call_converse = self._orig_call          # type: ignore
        self.client.extract_graph_from_page = self._orig_graph  # type: ignore

    def enrich_typical(self) -> None:
        """Attach typical (median) clean entity count for each failed call's page."""
        for f in self.failed_features:
            pg = f.get("page_number")
            clean = self.clean_entities_by_page.get(pg, [])
            f["typical_entities_for_page"] = (
                round(statistics.median(clean), 1) if clean else None
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-failures", type=int, default=10)
    ap.add_argument("--max-runs", type=int, default=10)
    ap.add_argument("--out", type=str, default=str(DATASET_OUT / "parse_failure_diagnosis.json"))
    args = ap.parse_args()

    client = get_extraction_client()
    all_pages = load_markdown_pages(CHANDER_MD)
    pages = [p for p in all_pages if p.get("page_number") in TARGETED_PAGES]
    schema, _gt = load_document_metadata(CHANDER_MD)

    diag = Diagnoser(client)
    diag.install()
    runs_done = 0
    per_run_parse_fail = []
    try:
        for i in range(args.max_runs):
            doc_id = f"diag_graph_run{i+1}"
            coll = m.get_collector()
            coll.reset()
            m.set_run_context(owner=ADMIN_OWNER, doc_id=doc_id)
            try:
                delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
            except Exception:
                pass
            with graph_maxtokens_override(client, GRAPH_MAXTOKENS_D):
                graph_out: Dict[str, Any] = {}
                asyncio.run(extract_with_graph_memory_concurrent(
                    pages_md=pages, schema=schema, llm=client,
                    graph_out=graph_out, doc_id=doc_id, owner=ADMIN_OWNER,
                ))
            try:
                delete_page_checkpoints(doc_id, owner=ADMIN_OWNER)
            except Exception:
                pass
            summ = coll.summarize(owner=ADMIN_OWNER, doc_id=doc_id)
            m.clear_run_context()
            runs_done += 1
            per_run_parse_fail.append(summ["parse_failures"])
            print(f"  run {i+1}: parse_failures(metrics)={summ['parse_failures']} "
                  f"captured_failed={len(diag.failed_features)} "
                  f"max_tokens_stops={summ['max_tokens_stops']}", flush=True)
            if len(diag.failed_features) >= args.target_failures:
                break
    finally:
        diag.uninstall()

    diag.enrich_typical()

    # ---- Aggregate counts per category (metadata only) ----
    ff = diag.failed_features
    n = len(ff)

    def count(pred) -> int:
        return sum(1 for f in ff if pred(f))

    categories = {
        "total_parse_failed_classified": n,
        "stop_reason_max_tokens": count(lambda f: f["stop_reason"] == "max_tokens"),
        "stop_reason_end_turn": count(lambda f: f["stop_reason"] == "end_turn"),
        "stop_reason_other": count(lambda f: f["stop_reason"] not in ("max_tokens", "end_turn")),
        "starts_with_fence": count(lambda f: f["starts_with_fence"]),
        "has_pre_json_text": count(lambda f: f["pre_json_len"] > 0),
        "has_post_json_text": count(lambda f: f["post_json_len"] > 0),
        "braces_unbalanced": count(lambda f: not f["braces_balanced"]),
        "parses_after_fence_strip": count(lambda f: f["parses_after_fence_strip"]),
        "multiple_top_level_objects": count(lambda f: f["approx_top_level_objects"] > 1),
        "has_reasoning_block": count(lambda f: f["has_reasoning_block"]),
    }

    # Distribution of the JSON error-kind and char-class at failure (metadata only).
    err_dist: Dict[str, int] = {}
    charcls_dist: Dict[str, int] = {}
    for f in ff:
        ek = f.get("json_err_class") or "none"
        err_dist[ek] = err_dist.get(ek, 0) + 1
        cc = f.get("char_class_at_err") or "none"
        charcls_dist[cc] = charcls_dist.get(cc, 0) + 1
    categories["json_err_class_distribution"] = err_dist
    categories["char_class_at_err_distribution"] = charcls_dist

    report = {
        "runs_done": runs_done,
        "per_run_parse_failures_metrics": per_run_parse_fail,
        "graph_max_tokens": GRAPH_MAXTOKENS_D,
        "pages": sorted(TARGETED_PAGES),
        "category_counts": categories,
        "failed_calls_features": ff,  # metadata only, no content
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nClassified {n} parse_failed graph-extraction calls across {runs_done} runs.")
    print("CATEGORY COUNTS:")
    for k, v in categories.items():
        print(f"  {k}: {v}")
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
