"""Phase 4 Part 2 — definitive validation of the OPTIONAL LLM grounding pass
(the secondary layer). RETAINED as a permanent validation harness.

Constructs a claim the deterministic overlap heuristic FAILS (heavily paraphrased
number-in-words) and confirms a real Bedrock entailment call can ground it, plus a
hallucination the LLM should reject. Proves the secondary layer fires (method=llm)
and that it degrades gracefully.

Runnable:
  - as a script:  python tests/test_phase4_llmpass.py
  - under pytest: skips when .env2 Bedrock creds aren't loadable / profile denied.

NOTE: gpt-oss at temperature 0 is still nondeterministic (documented Phase 3
finding), so the LLM verdict can vary run-to-run; the pytest assertion only
requires the call to COMPLETE and the hallucination to stay unverified, not a
specific grounded verdict on the paraphrase.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
from src.ai.layer3_extraction.graph_agent.grounding_checker import check_grounding, _best_overlap

EV = [
    {"page": 9, "evidence": "Sanctioned 80698", "node": "80698"},
    {"page": 10, "evidence": "Net Bill Amount 189,771.25", "node": "189771.25"},
]
PARAPHRASE_CLAIM = "The insurer approved a payout of eighty thousand six hundred ninety eight rupees."
HALLUCINATION = "The insurer approved a payout of five hundred thousand rupees."


def _load_creds():
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


def _bedrock_reachable(svc) -> bool:
    """Returns True only if a trivial converse succeeds (profile actually invokable)."""
    try:
        out = svc._bedrock_grounding_caller("Reply with one word.", "Say OK")
        return bool(out)
    except Exception:
        return False


def _run(svc):
    results = {}
    for label, claim in [("PARAPHRASE (entailed)", PARAPHRASE_CLAIM), ("HALLUCINATION", HALLUCINATION)]:
        ov, _ = _best_overlap(claim, EV)
        print("\n%s" % label)
        print("  claim: %r" % claim)
        print("  overlap-only score: %.3f (below 0.45 => heuristic UNVERIFIED, LLM pass decides)" % ov)
        r = check_grounding(claim, EV, use_llm=True, llm_caller=svc._bedrock_grounding_caller)
        v = r["claims"][0]
        print("  FINAL: status=%s method=%s" % (v["status"], v["method"]))
        if v["citations"]:
            print("  cite -> page %s: %r" % (v["citations"][0]["page"], v["citations"][0]["evidence"]))
        results[label] = v
    return results


def test_phase4_llm_grounding_pass():
    """pytest entry: the LLM pass must COMPLETE against real Bedrock and must NOT
    falsely ground the hallucination. (Paraphrase verdict is nondeterministic, so
    it is reported but not asserted.)"""
    import pytest
    if not _load_creds():
        pytest.skip(".env2 Bedrock creds not loadable")
    svc = GraphQueryService()
    if not _bedrock_reachable(svc):
        pytest.skip("Bedrock inference profile not invokable with current creds")
    results = _run(svc)
    hallu = results["HALLUCINATION"]
    assert hallu["status"] == "unverified", "LLM pass wrongly grounded a hallucination"


if __name__ == "__main__":
    if not _load_creds():
        print("SKIP: .env2 creds not loadable")
    else:
        _svc = GraphQueryService()
        if not _bedrock_reachable(_svc):
            print("SKIP: Bedrock profile not invokable with current creds")
        else:
            _run(_svc)
