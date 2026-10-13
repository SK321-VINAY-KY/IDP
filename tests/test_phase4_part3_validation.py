"""Phase 4 Part 3 — honest formal validation on Chander Kochhar graph100.
RETAINED as a permanent validation harness.

Four requirements:
 (1) evidence_groundedness() before (keyword-only) vs after (hybrid)
 (2) paraphrase pass/fail: OLD keyword-only vs NEW hybrid (BFS-subgraph reachability)
 (3) deterministic fallback still works when Bedrock is DOWN
 (4) real per-query latency: retrieval + full query() — measured, not estimated

Runnable:
  - as a script:  python tests/test_phase4_part3_validation.py   (full report)
  - under pytest: the DB-only metrics (1)(2)(3) run when the DB is reachable and
    skip otherwise; latency (4) only prints under script mode.
"""
import os, re, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sqlalchemy import create_engine, text
from src.config.settings import settings
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
from scripts.eval_metrics import evidence_groundedness

QSET = [
    ("how much did the hospital bill come to",        "189771"),
    ("what room was the patient kept in",             "TWIN"),
    ("who is covering the medical costs",             "ICICI"),
    ("which company is the health cover with",        "ICICI"),
    ("what is the sum the hospital charged overall",  "189771"),
    ("what was wrong with the patient medically",     "breathless"),
    ("where is the patient based",                    "COZI"),
    ("who should be contacted in an emergency",       "022 2546"),
]


def _load_creds():
    for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        os.environ.pop(_k, None)
    try:
        n = 0
        with open(os.path.join(_ROOT, ".env2"), encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r'\s*\$Env:(AWS_\w+)\s*=\s*"([^"]*)"', line)
                if m:
                    os.environ[m.group(1)] = m.group(2); n += 1
        os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
        return n >= 3
    except Exception:
        return False


def _load_graph(graph_id: int = 100):
    try:
        eng = create_engine(
            settings.database_url.replace('postgresql://', 'postgresql+psycopg2://'),
            connect_args={'connect_timeout': 5},
        )
        with eng.connect() as c:
            row = c.execute(text("select graph_json from document_graphs where id=:i"),
                            {"i": graph_id}).fetchone()
        if not row:
            return None
        g = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return GraphMemory.from_dict(g)
    except Exception:
        return None


def _answer_in(nodes, needle):
    nd = needle.lower()
    for n in nodes:
        if nd in (n.value or '').lower():
            return True
        for ev in (n.evidence or []):
            if nd in (ev.text or '').lower():
                return True
    return False


def _run_groundedness(graph, svc):
    print("=" * 92)
    print("(1) evidence_groundedness() BEFORE (keyword-only) vs AFTER (hybrid) — retrieved subgraph")
    print("=" * 92)
    before, after = [], []
    for q, _ in QSET:
        kw = svc.find_seed_nodes(graph, q)
        kw_nodes, _, _ = svc.traverse_subgraph(graph, kw) if kw else ([], [], 0)
        vec = svc.find_vector_seed_nodes(graph, q)
        merged, _ = svc._merge_seed_sets(kw, vec, 10)
        h_nodes, _, _ = svc.traverse_subgraph(graph, merged) if merged else ([], [], 0)
        gb = evidence_groundedness([n.to_dict() for n in kw_nodes])
        ga = evidence_groundedness([n.to_dict() for n in h_nodes])
        before.append(gb["grounded_pct"]); after.append(ga["grounded_pct"])
        print("  Q=%-45s | before %5.1f%% (%d) -> after %5.1f%% (%d)" % (
            q[:45], gb["grounded_pct"], gb["total_nodes"], ga["grounded_pct"], ga["total_nodes"]))
    avg_b = round(sum(before) / len(before), 1)
    avg_a = round(sum(after) / len(after), 1)
    print("  AVG groundedness: before %.1f%%  after %.1f%%" % (avg_b, avg_a))
    return avg_b, avg_a


def _run_paraphrase(graph, svc):
    print("\n" + "=" * 92)
    print("(2) PARAPHRASE PASS/FAIL — OLD keyword-only vs NEW hybrid (BFS-subgraph reachability)")
    print("=" * 92)
    old_pass = new_pass = rescues = regress = 0
    for q, needle in QSET:
        kw = svc.find_seed_nodes(graph, q)
        kw_nodes, _, _ = svc.traverse_subgraph(graph, kw) if kw else ([], [], 0)
        vec = svc.find_vector_seed_nodes(graph, q)
        merged, _ = svc._merge_seed_sets(kw, vec, 10)
        h_nodes, _, _ = svc.traverse_subgraph(graph, merged) if merged else ([], [], 0)
        o = _answer_in(kw_nodes, needle); n = _answer_in(h_nodes, needle)
        old_pass += o; new_pass += n
        tag = "RESCUE" if (n and not o) else ("REGRESS" if (o and not n) else "")
        if tag == "RESCUE": rescues += 1
        if tag == "REGRESS": regress += 1
        print("  [%s->%s] %-45s %s" % ("P" if o else "F", "P" if n else "F", q[:45], tag))
    print("  OLD keyword-only: %d/%d  NEW hybrid: %d/%d  | rescues=%d regressions=%d" % (
        old_pass, len(QSET), new_pass, len(QSET), rescues, regress))
    return old_pass, new_pass, rescues, regress


def _run_fallback(graph, svc):
    print("\n" + "=" * 92)
    print("(3) DETERMINISTIC FALLBACK with Bedrock DOWN (no creds) — must still answer")
    print("=" * 92)
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        os.environ.pop(k, None)
    ok = True
    for q, _ in QSET[:3]:
        try:
            r = svc.query(graph, q, verify_grounding=True)
            answered = bool(r.answer) and "could not be found" not in r.answer.lower()
            print("  Q=%-45s | answered=%s mode=%s grounding_ran=%s" % (
                q[:45], answered, r.mode, bool(r.grounding)))
            ok = ok and answered
        except Exception as e:
            ok = False
            print("  Q=%-45s | EXCEPTION %s" % (q[:45], str(e)[:50]))
    print("  FALLBACK SAFETY NET intact: %s" % ok)
    return ok


def _run_latency(graph, svc):
    print("\n" + "=" * 92)
    print("(4) LATENCY per query (real measured seconds)")
    print("=" * 92)
    print("  Bedrock creds loaded: %s" % _load_creds())
    import numpy as np
    t = time.perf_counter(); _ = svc.find_vector_seed_nodes(graph, "warmup"); warm = time.perf_counter() - t
    print("  one-time embedding model load + first encode: %.2fs" % warm)
    ret_times, full_times = [], []
    for q, _ in QSET:
        t = time.perf_counter(); svc.find_vector_seed_nodes(graph, q); ret_times.append(time.perf_counter() - t)
        t = time.perf_counter(); svc.query(graph, q, verify_grounding=True); full_times.append(time.perf_counter() - t)
    def stat(xs): return (round(np.mean(xs), 3), round(np.min(xs), 3), round(np.max(xs), 3))
    print("  per-query vector retrieval (warm):  mean/min/max = %.3f / %.3f / %.3f s" % stat(ret_times))
    print("  per-query FULL query():             mean/min/max = %.3f / %.3f / %.3f s" % stat(full_times))


def test_phase4_part3_paraphrase_and_fallback():
    """pytest entry (DB-dependent): hybrid >= keyword with no regressions, >=1 rescue,
    and the deterministic fallback answers with Bedrock down."""
    import pytest
    graph = _load_graph(100)
    if graph is None:
        pytest.skip("local DB / document_graphs id=100 not reachable")
    svc = GraphQueryService()
    _run_groundedness(graph, svc)
    old_pass, new_pass, rescues, regress = _run_paraphrase(graph, svc)
    fallback_ok = _run_fallback(graph, svc)
    assert regress == 0
    assert new_pass >= old_pass and rescues >= 1
    assert fallback_ok is True


if __name__ == "__main__":
    _g = _load_graph(100)
    if _g is None:
        print("SKIP: local DB / graph id=100 not reachable")
    else:
        _svc = GraphQueryService()
        _run_groundedness(_g, _svc)
        _run_paraphrase(_g, _svc)
        _run_fallback(_g, _svc)
        _run_latency(_g, _svc)
