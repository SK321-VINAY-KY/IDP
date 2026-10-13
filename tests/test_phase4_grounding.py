"""Phase 4 Part 2 grounding validation harness on Chander Kochhar graph100.
RETAINED as a permanent validation harness.

Covers:
  (A) direct check_grounding unit cases - grounded vs hallucinated (no DB/creds)
  (B) full query() with grounding, Bedrock-OFF (deterministic + fallback safety net)
  (C) full query() with grounding, Bedrock-ON (optional LLM entailment pass)

Runnable two ways:
  - as a script:  python tests/test_phase4_grounding.py   (prints full A/B/C report)
  - under pytest: the deterministic unit cases (A) run always; the DB/Bedrock
    sections skip gracefully when unavailable.
"""
import os, re, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import create_engine, text
from src.config.settings import settings
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
from src.ai.layer3_extraction.graph_agent.grounding_checker import check_grounding

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Deterministic unit cases — need neither DB nor Bedrock.
EV = [
    {"page": 10, "evidence": "Net Bill Amount 189,771.25", "node": "189771.25"},
    {"page": 9,  "evidence": "Sanctioned 80698", "node": "80698"},
    {"page": 6,  "evidence": "Bill. Class : TWIN SHARING", "node": "TWIN SHARING"},
]
UNIT_CASES = [
    ("The total hospital bill was 189,771.25.",        True,  "grounded: paraphrase around shared number"),
    ("The sanctioned amount is 80698.",                True,  "grounded: near-verbatim"),
    ("The patient was kept in a TWIN SHARING room.",   True,  "grounded: key token present"),
    ("The patient paid a deductible of 50000.",        False, "HALLUCINATED: 50000 not in evidence"),
    ("The hospital is located in Chennai.",            False, "HALLUCINATED: no such evidence"),
]


def _load_creds():
    """Parse .env2 (PowerShell $Env: form) into os.environ. Clears ambient AWS_*
    first so only .env2 is used. Returns True if 3 keys were set."""
    for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        os.environ.pop(_k, None)
    try:
        got = 0
        with open(os.path.join(_ROOT, ".env2"), encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r'\s*\$Env:(AWS_\w+)\s*=\s*"([^"]*)"', line)
                if m:
                    os.environ[m.group(1)] = m.group(2); got += 1
        os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
        return got >= 3
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


def _run_unit():
    print("=" * 90)
    print("(A) DIRECT check_grounding UNIT CASES (deterministic only, use_llm=False)")
    print("=" * 90)
    caught = 0
    for claim, expect_grounded, note in UNIT_CASES:
        r = check_grounding(claim, EV, use_llm=False)
        v = r["claims"][0]
        ok = (v["grounded"] == expect_grounded)
        caught += ok
        print("  [%s] expect=%s got=%s score=%.3f  %s" % (
            "OK" if ok else "XX", "grounded" if expect_grounded else "unverified",
            v["status"], v["score"], note))
    print("  UNIT: %d correct / %d" % (caught, len(UNIT_CASES)))
    return caught


def _run_bedrock_off(graph, svc):
    print("\n" + "=" * 90)
    print("(B) FULL query() WITH GROUNDING, BEDROCK OFF -> deterministic + fallback safety net")
    print("=" * 90)
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        os.environ.pop(k, None)
    t0 = time.perf_counter()
    res = svc.query(graph, "what was the total hospital bill", verify_grounding=True)
    dt = time.perf_counter() - t0
    print("  mode=%s answer=%r" % (res.mode, res.answer[:70]))
    print("  grounding: %s/%s grounded | method_summary=%s" % (
        res.grounding.get("grounded_count"), res.grounding.get("total_claims"),
        res.grounding.get("method_summary")))
    print("  fallback safety net worked (answer non-empty, no exception): %s | %.2fs" % (bool(res.answer), dt))
    return bool(res.answer)


def _run_bedrock_on(graph, svc):
    print("\n" + "=" * 90)
    print("(C) FULL query() WITH GROUNDING, BEDROCK ON -> LLM entailment pass active")
    print("=" * 90)
    if not _load_creds():
        print("  SKIP: .env2 creds not loadable")
        return
    for q in ["what was the total hospital bill", "who is the insurance provider"]:
        t0 = time.perf_counter()
        res = svc.query(graph, q, verify_grounding=True)
        dt = time.perf_counter() - t0
        print("\n  Q=%r (%.2fs)" % (q, dt))
        print("  answer=%r" % res.answer[:80])
        gr = res.grounding
        print("  grounding: %s/%s grounded | methods=%s" % (
            gr.get("grounded_count"), gr.get("total_claims"), gr.get("method_summary")))
        for cl in gr.get("claims", []):
            cites = cl.get("citations", [])
            cp = ("p%s" % cites[0]["page"]) if cites else "-"
            print("     [%s via %s %.2f %s] %s" % (cl["status"], cl["method"], cl["score"], cp, cl["claim"][:55]))


def test_phase4_grounding_unit_cases():
    """pytest entry (deterministic, no DB/creds): overlap heuristic must correctly
    ground the 3 real claims and flag the 2 hallucinations."""
    caught = _run_unit()
    assert caught == len(UNIT_CASES), "grounding heuristic misclassified a unit case"


def test_phase4_grounding_fallback_bedrock_off():
    """pytest entry: with Bedrock unavailable, query()+grounding must still answer
    (deterministic fallback safety net) without raising."""
    import pytest
    graph = _load_graph(100)
    if graph is None:
        pytest.skip("local DB / document_graphs id=100 not reachable")
    svc = GraphQueryService()
    assert _run_bedrock_off(graph, svc) is True


if __name__ == "__main__":
    _run_unit()
    _g = _load_graph(100)
    if _g is None:
        print("\nSKIP (B)(C): local DB / graph id=100 not reachable")
    else:
        _svc = GraphQueryService()
        _run_bedrock_off(_g, _svc)
        _run_bedrock_on(_g, _svc)
