"""
Cross-process determinism test for the concurrent graph merge (close-out item 1).

Builds the graph from the SAME synthetic deltas in separate subprocesses under at
least 5 different PYTHONHASHSEED values and compares full graph fingerprints
(nodes, edges, canonical ids, evidence order, insertion order). Set iteration of
string ids (e.g. _alias_index[val_norm], lookup_alias's next(iter(...))) varies with
the hash seed, so this test fails if any merge-time match iterates a set of ids
without an order-stabilizing sort.

Synthetic fixtures only — NO document content.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# This worker script is executed in a fresh subprocess per hash seed. It builds the
# graph deterministically (sorted merge) and prints a full fingerprint as JSON.
WORKER = r'''
import asyncio, json, sys
from pathlib import Path
ROOT = Path(r"__ROOT__")
sys.path.insert(0, str(ROOT))
from pydantic import BaseModel, Field
from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
from src.ai.layer3_extraction.graph_agent.models import GraphNode, PageDelta, Evidence
from src.ai.layer3_extraction.extractor import merge_page_deltas_sorted

class S(BaseModel):
    field_a: str = Field(default="", description="a")
    field_b: str = Field(default="", description="b")

def node(nid, value, label, page, ntype="Value", aliases=None):
    return GraphNode(
        id=nid, type=ntype, label=label, value=value,
        properties={}, source_pages=[page], confidence=0.9, status="ASSERTED",
        category="EXPLICIT", aliases=list(aliases or []),
        evidence=[Evidence(page_number=page, text="ev-"+nid, confidence=0.9)],
    )

# Deltas crafted so several distinct nodes share a normalized alias value, forcing
# the merge-time _alias_index[val_norm] set to hold multiple ids (iteration order
# of that set is what varies by hash seed).
def deltas():
    return {
        1: PageDelta(page_number=1, entities=[
            node("p1_x", "SHARED VALUE", "label_one", 1, "TypeA"),
            node("p1_y", "ALPHA", "label_a", 1, "TypeA", aliases=["shared value"]),
        ]),
        2: PageDelta(page_number=2, entities=[
            node("p2_x", "shared value", "label_two", 2, "TypeB"),
            node("p2_z", "BETA", "label_b", 2, "TypeB", aliases=["shared value"]),
        ]),
        3: PageDelta(page_number=3, entities=[
            node("p3_x", "Shared Value", "label_three", 3, "TypeA"),
        ]),
    }

async def build():
    mgr = GraphMemoryManager(anchor_types=[], anchor_max_k=5, schema=S, pipeline_version="v1")
    ds = deltas()
    await merge_page_deltas_sorted(mgr, [PageDelta.from_dict(ds[p].to_dict()) for p in ds])
    await mgr.reconcile_cross_references()
    return mgr.graph

# Direct probe of the merge-time graph primitive find_reuse_candidate: several
# distinct nodes share one normalized value+type, so _alias_index[val_norm] holds
# multiple ids. The returned candidate must NOT depend on set-iteration order.
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory as _GM
_probe = _GM()
for _nid in ["z_node", "a_node", "m_node", "q_node", "b_node"]:
    _probe.create_node(GraphNode(id=_nid, type="TypeA", label="L", value="Shared Value",
                                 properties={}, source_pages=[1], confidence=0.9,
                                 status="ASSERTED", category="EXPLICIT"))
_reuse = _probe.find_reuse_candidate(node_type="TypeA", value="shared value", label="L")
_reuse_pick = _reuse.id if _reuse else None

g = asyncio.run(build())
nodes = []
for n in g.nodes.values():
    nodes.append({
        "id": n.id, "value": " ".join(n.value.lower().split()), "label": n.label,
        "type": n.type, "pages": sorted(n.source_pages), "status": n.status,
        "ev": [e.text for e in n.evidence],
    })
edges = [{"id": e.id, "src": e.source_node, "tgt": e.target_node, "rel": e.relationship}
         for e in g.edges.values()]
fp = {
    "reuse_pick": _reuse_pick,                      # find_reuse_candidate's choice
    "insertion_order": [n["id"] for n in nodes],   # dict order = merge insertion order
    "nodes": nodes,
    "edges": edges,
}
print("FINGERPRINT:" + json.dumps(fp, sort_keys=False))
'''


def _run_worker(seed: int) -> dict:
    script = WORKER.replace("__ROOT__", str(ROOT).replace("\\", "\\\\"))
    env = {"PYTHONHASHSEED": str(seed)}
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=full_env, cwd=str(ROOT), timeout=180,
    )
    assert proc.returncode == 0, f"worker (seed={seed}) failed:\n{proc.stderr[-2000:]}"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("FINGERPRINT:")), None)
    assert line, f"no fingerprint from worker (seed={seed}):\n{proc.stdout[-1000:]}"
    return json.loads(line[len("FINGERPRINT:"):])


def test_graph_identical_across_pythonhashseeds():
    seeds = [0, 1, 7, 42, 1234, 99999]  # >= 5 distinct seeds
    fps = [_run_worker(s) for s in seeds]
    base = fps[0]
    base_blob = json.dumps(base, sort_keys=False)
    for s, fp in zip(seeds[1:], fps[1:]):
        assert json.dumps(fp, sort_keys=False) == base_blob, (
            f"graph differs under PYTHONHASHSEED={s}: "
            f"insertion_order base={base['insertion_order']} vs {fp['insertion_order']}"
        )
    # canonical ids reproducible across all seeds
    id_sets = {tuple(sorted(n["id"] for n in fp["nodes"])) for fp in fps}
    assert len(id_sets) == 1, "canonical id sets differ across hash seeds"


def test_find_reuse_candidate_is_seed_stable():
    """find_reuse_candidate must return the SAME node id across hash seeds. Fails on
    set-iteration (next match in _alias_index[val_norm] set order)."""
    seeds = [0, 1, 7, 42, 1234, 99999]
    picks = {_run_worker(s)["reuse_pick"] for s in seeds}
    assert len(picks) == 1, f"find_reuse_candidate picked different nodes across seeds: {picks}"
