"""Phase 4 Part 1.4 paraphrase validation harness: keyword-only vs hybrid
(keyword+vector) seed-finding on Chander Kochhar's graph. No Bedrock needed
(retrieval is local). RETAINED as a permanent validation harness.

For each paraphrased question we compare:
  OLD = find_seed_nodes() only  (keyword/synonym)
  NEW = find_seed_nodes() + find_vector_seed_nodes() merged
measuring whether the node holding the KNOWN-CORRECT answer survives into the
BFS-traversed subgraph (the end-to-end signal that actually feeds the answer).

Runnable two ways:
  - as a script:  python tests/test_phase4_paraphrase.py   (prints full report)
  - under pytest: skips gracefully if the local DB / graph id=100 isn't reachable
    (same live-dependency-skip convention as tests/test_query_bot_e2e.py).
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import create_engine, text
from src.config.settings import settings
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService


def _load_graph(graph_id: int = 100):
    """Load the Chander Kochhar graph from the local DB. Returns None if the DB
    or the row isn't reachable, so pytest can skip rather than error."""
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


# Paraphrased questions whose correct answers are known from the corrected GT.
# Each: (paraphrase, known-correct answer substring to look for in a found node).
CASES = [
    ("how much did the hospital bill come to",            "189771"),      # total_hospital_bill_amount=189771.25
    ("what is the patient's unique hospital id",          "145304681"),   # patient_uhid (NOTE: absent from graph100)
    ("how much did the insurer agree to pay",             "80698"),       # sanctioned_amount
    ("what room was the patient kept in",                 "TWIN"),        # room_category=TWIN SHARING
    ("who is covering the medical costs",                 "ICICI"),       # insurance_company=ICICI LOMBARD
    ("which company is the health cover with",            "ICICI"),       # insurance_company (paraphrase 2)
    ("what sort of ward was she admitted to",             "TWIN"),        # room_category (paraphrase 2)
    ("what is the sum the hospital charged overall",      "189771"),      # total bill (paraphrase 2)
]


def found_answer(nodes, needle):
    nd = needle.lower()
    for n in nodes:
        if nd in (n.value or '').lower():
            return n
    return None


def _run(graph, svc):
    print("=" * 100)
    print("PHASE 4 PART 1.4 PARAPHRASE TEST  (graph id=100, Chander Kochhar, %d nodes)" % len(graph.nodes))
    print("=" * 100)

    summary = []
    for q, needle in CASES:
        kw = svc.find_seed_nodes(graph, q)
        vec = svc.find_vector_seed_nodes(graph, q)
        merged, telem = svc._merge_seed_sets(kw, vec, max_total=10)

        # END-TO-END signal: does the answer node survive into the BFS-traversed
        # subgraph? That is what actually feeds the answer, not just the seed set.
        kw_nodes, _, _ = svc.traverse_subgraph(graph, kw) if kw else ([], [], 0)
        new_nodes, _, _ = svc.traverse_subgraph(graph, merged) if merged else ([], [], 0)
        kw_bfs_hit = found_answer(kw_nodes, needle)
        new_bfs_hit = found_answer(new_nodes, needle)

        print("\nQ: %r   (looking for answer containing %r)" % (q, needle))
        print("  OLD keyword-only: %d seeds -> %d BFS nodes | answer in subgraph: %s" % (
            len(kw), len(kw_nodes), "YES" if kw_bfs_hit else "NO"))
        print("  NEW hybrid:       %d seeds -> %d BFS nodes | answer in subgraph: %s" % (
            len(merged), len(new_nodes), "YES -> " + new_bfs_hit.value[:40] if new_bfs_hit else "NO"))
        vtop = [(round(s, 3), (n.value or '')[:40]) for n, s in sorted(vec, key=lambda x: x[1], reverse=True)[:3]]
        print("  vector top hits:  %s" % vtop)
        if new_bfs_hit and not kw_bfs_hit:
            verdict = "*** VECTOR RESCUED (keyword subgraph missed, hybrid reached) ***"
        elif kw_bfs_hit and new_bfs_hit:
            verdict = "both reached (no regression)"
        elif not kw_bfs_hit and not new_bfs_hit:
            verdict = "neither reached"
        else:
            verdict = "REGRESSION?! keyword reached, hybrid lost"
        print("  VERDICT: %s" % verdict)
        summary.append((q, needle, bool(kw_bfs_hit), bool(new_bfs_hit), verdict))

    print("\n" + "=" * 100)
    print("SUMMARY")
    rescued = sum(1 for *_, v in summary if "RESCUED" in v)
    regress = sum(1 for *_, v in summary if "REGRESSION" in v)
    for q, needle, k, n, v in summary:
        print("  [%s->%s] %s" % ("Y" if k else "N", "Y" if n else "N", q))
    print("\nVECTOR-RESCUED cases: %d | regressions: %d" % (rescued, regress))
    return rescued, regress


def test_phase4_paraphrase_hybrid_rescues():
    """pytest entry: hybrid must rescue >=1 paraphrase vs keyword-only and never regress."""
    import pytest
    graph = _load_graph(100)
    if graph is None:
        pytest.skip("local DB / document_graphs id=100 (Chander Kochhar) not reachable")
    svc = GraphQueryService()
    rescued, regress = _run(graph, svc)
    assert regress == 0, "hybrid retrieval regressed a keyword-reachable answer"
    assert rescued >= 1, "expected at least one vector rescue vs keyword-only"


if __name__ == "__main__":
    _g = _load_graph(100)
    if _g is None:
        print("SKIP: local DB / graph id=100 not reachable")
    else:
        _run(_g, GraphQueryService())
