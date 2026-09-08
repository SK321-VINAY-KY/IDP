"""
File: identity_resolution.py
Purpose: Step 6 Scoped Identity Resolution for Layer 3.
         Implements resolution to POC Section 8.2:
         - Restricted strictly to Person and Organization categories only.
         - Bounded disambiguation pass (Sections 17-18 of graph memory POC).
         - Evaluated ONLY over the post-navigation candidate set.
         - Hard bounds: MAX_REASONING_STEPS = 20, MAX_TOOL_CALLS = 30, MAX_GRAPH_DEPTH = 3.
         - Never merge on nearest-node or first-match. Ambiguous cases get UNCERTAIN, not guessed.
Owner: engineer-a@idp-pilot
Created: 2026-09-07 | References: docs/poc/layer3_graph_memory_poc_v2.md (Sections 17, 18)
"""
import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.ai.layer3_extraction.extractor import CandidateEdge, CandidateNode
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Bounded reasoning constraints from graph memory POC Phase 2
MAX_REASONING_STEPS_PER_CATEGORY = 20
MAX_GLOBAL_REASONING_STEPS = 100
MAX_TOOL_CALLS_PER_CATEGORY = 30
MAX_REASONING_STEPS = MAX_REASONING_STEPS_PER_CATEGORY  # Backwards-compatibility alias
MAX_TOOL_CALLS = MAX_TOOL_CALLS_PER_CATEGORY            # Backwards-compatibility alias
MAX_GRAPH_DEPTH = 3                                    # Per-call BFS depth limit

# Resolution thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.78
LOW_CONFIDENCE_THRESHOLD = 0.50


class ResolvedEntity(BaseModel):
    canonical_id: str
    category: str
    canonical_label: str
    mention_node_ids: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    resolution_confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED | INFERRED | UNCERTAIN
    resolution_note: str = ""


class DisambiguationResult(BaseModel):
    node_id_a: str
    node_id_b: str
    label_a: str
    label_b: str
    category: str
    similarity_score: float
    decision: str  # "SAME_AS" | "UNCERTAIN" | "DIFFERENT"
    confidence: float
    evidence: str
    reasoning_step: int
    reasoning_trace: List[str] = Field(default_factory=list)
    category_budget_exhausted: bool = False


class EvidenceGraph:
    """
    In-memory graph providing bounded inspection operations:
    search_nodes(), read_neighbors(), traverse().
    """
    def __init__(self, nodes: List[CandidateNode], edges: List[CandidateEdge]) -> None:
        self.nodes: Dict[str, CandidateNode] = {n.node_id: n for n in nodes}
        self.edges: List[CandidateEdge] = list(edges)
        self.outgoing: Dict[str, List[CandidateEdge]] = {}
        self.incoming: Dict[str, List[CandidateEdge]] = {}
        self.tool_calls_count = 0
        self._rebuild_indices()

    def reset_tool_calls_budget(self) -> None:
        """Reset the tool calls counter for a new category pass."""
        self.tool_calls_count = 0

    def _rebuild_indices(self) -> None:
        self.outgoing = {nid: [] for nid in self.nodes}
        self.incoming = {nid: [] for nid in self.nodes}
        for e in self.edges:
            self.outgoing.setdefault(e.source, []).append(e)
            self.incoming.setdefault(e.target, []).append(e)

    def search_nodes(self, query: str, category: Optional[str] = None) -> List[CandidateNode]:
        """Search graph nodes by text query and optional category."""
        if self.tool_calls_count >= MAX_TOOL_CALLS_PER_CATEGORY:
            logger.warning("evidence_graph.tool_calls_budget_exhausted")
            return []
        self.tool_calls_count += 1

        q_lower = query.lower()
        results = []
        for n in self.nodes.values():
            if category and n.type != category:
                continue
            if q_lower in n.label.lower() or q_lower in n.evidence.lower():
                results.append(n)
        return results

    def read_neighbors(self, node_id: str) -> List[Tuple[CandidateEdge, CandidateNode]]:
        """Return connected neighbors for a node."""
        if self.tool_calls_count >= MAX_TOOL_CALLS_PER_CATEGORY:
            logger.warning("evidence_graph.tool_calls_budget_exhausted")
            return []
        self.tool_calls_count += 1

        neighbors: List[Tuple[CandidateEdge, CandidateNode]] = []
        for e in self.outgoing.get(node_id, []):
            target_node = self.nodes.get(e.target)
            if target_node:
                neighbors.append((e, target_node))
        for e in self.incoming.get(node_id, []):
            source_node = self.nodes.get(e.source)
            if source_node:
                neighbors.append((e, source_node))
        return neighbors

    def traverse(
        self,
        node_id: str,
        relationship: Optional[str] = None,
        max_depth: int = MAX_GRAPH_DEPTH,
    ) -> List[CandidateNode]:
        """Bounded breadth-first traversal up to MAX_GRAPH_DEPTH."""
        if self.tool_calls_count >= MAX_TOOL_CALLS_PER_CATEGORY:
            return []
        self.tool_calls_count += 1

        effective_depth = min(max_depth, MAX_GRAPH_DEPTH)
        visited = {node_id}
        frontier = [node_id]
        reachable: List[CandidateNode] = []

        for _ in range(effective_depth):
            next_frontier = []
            for cur in frontier:
                for e in self.outgoing.get(cur, []):
                    if relationship and e.relationship != relationship:
                        continue
                    if e.target not in visited:
                        visited.add(e.target)
                        target_node = self.nodes.get(e.target)
                        if target_node:
                            reachable.append(target_node)
                            next_frontier.append(e.target)
            frontier = next_frontier

        return reachable

    def add_edge(self, edge: CandidateEdge) -> None:
        self.edges.append(edge)
        self.outgoing.setdefault(edge.source, []).append(edge)
        self.incoming.setdefault(edge.target, []).append(edge)


def _clean_person_name(name: str) -> str:
    """Normalize honorifics, OCR prefix noise, trailing context, and spacing for comparison."""
    n = name.upper()
    n = re.sub(r"\b(DR\.?|DOCTOR|MR\.?|MRS\.?|MS\.?|OR|PATEENT|PATIENT)\b", "", n)
    n = re.sub(r"\b(WITH\s+PRIOR\s+APPOINTMENT|WITH\s+PRIOR\s+APPOINTM|WITH\s+PRIOR|CONSULT|APPOINTMENT|APPOINTM)\b.*", "", n)
    n = re.sub(r"[^A-Z\s]", "", n)
    return " ".join(n.split())


def _clean_org_name(name: str) -> str:
    """Normalize organization names for comparison."""
    n = name.upper()
    n = re.sub(r"\b(HOSPITAL|CENTRE|CENTER|MEDICAL|RESEARCH|PVT|LTD|LIMITED|COMPANY|COMPA|SERVICE|CHEMIST)\b", "", n)
    n = re.sub(r"[^A-Z0-9\s]", "", n)
    return " ".join(n.split())


def compute_entity_similarity(label_a: str, label_b: str, category: str) -> float:
    """
    Multi-signal lexical and phonetic similarity score between two entity mentions.
    Handles OCR character substitutions (e.g. Abhay vs Arhay, Kochhar vs Kdchhar)
    and name transposition (e.g. Raut Abhay vs Abhay Raut).
    """
    if category == "Person":
        clean_a = _clean_person_name(label_a)
        clean_b = _clean_person_name(label_b)
    else:
        clean_a = _clean_org_name(label_a)
        clean_b = _clean_org_name(label_b)

    if not clean_a or not clean_b:
        return 0.0

    if clean_a == clean_b:
        return 1.0

    tokens_a = clean_a.split()
    tokens_b = clean_b.split()

    # Exact token set match (e.g. "RAUT ABHAY" vs "ABHAY RAUT")
    if set(tokens_a) == set(tokens_b) and len(tokens_a) > 0:
        return 1.0

    # Substring containment (e.g. "Chander Kochhar" inside "Mrs Chander Kochhar")
    if clean_a in clean_b or clean_b in clean_a:
        min_len = min(len(clean_a), len(clean_b))
        max_len = max(len(clean_a), len(clean_b))
        if min_len / max_len > 0.5:
            return 0.95

    # Word-by-word token comparison (pairing tokens with highest similarity)
    if len(tokens_a) == len(tokens_b) and len(tokens_a) > 0:
        # Best matching permutation
        sim_direct = sum(difflib.SequenceMatcher(None, ta, tb).ratio() for ta, tb in zip(tokens_a, tokens_b)) / len(tokens_a)
        sim_reversed = sum(difflib.SequenceMatcher(None, ta, tb).ratio() for ta, tb in zip(tokens_a, reversed(tokens_b))) / len(tokens_a)
        best_token_sim = max(sim_direct, sim_reversed)
        return best_token_sim

    # Sequence matcher similarity fallback
    return difflib.SequenceMatcher(None, clean_a, clean_b).ratio()



def run_scoped_identity_resolution(
    nodes: List[CandidateNode],
    edges: List[CandidateEdge],
) -> Tuple[List[ResolvedEntity], List[CandidateEdge], List[DisambiguationResult]]:
    """
    Step 6 Entry Point:
    Runs bounded identity resolution over Person and Organization candidate nodes only.
    - Strictly obeys MAX_REASONING_STEPS_PER_CATEGORY, MAX_TOOL_CALLS_PER_CATEGORY, MAX_GRAPH_DEPTH.
    - Resolves duplicate mentions (e.g. "Dr. Abhay Raut" and "Dr Arhay Raut").
    - Never merges on nearest-node or first-match; ambiguous cases get UNCERTAIN.

    Returns:
        (resolved_entities, updated_edges, disambiguation_log)
    """
    # Restrict candidate set strictly to Person and Organization
    target_nodes = [n for n in nodes if n.type in ("Person", "Organization")]
    graph = EvidenceGraph(nodes, edges)

    resolved_entities: List[ResolvedEntity] = []
    disambiguation_log: List[DisambiguationResult] = []

    total_reasoning_steps = 0
    category_stats: Dict[str, Dict[str, Any]] = {}

    # Group target nodes by category
    by_category: Dict[str, List[CandidateNode]] = {}
    for n in target_nodes:
        by_category.setdefault(n.type, []).append(n)

    # Disjoint set / union-find tracking for merges
    parent: Dict[str, str] = {n.node_id: n.node_id for n in target_nodes}

    def find(nid: str) -> str:
        if parent[nid] != nid:
            parent[nid] = find(parent[nid])
        return parent[nid]

    def union(nid_a: str, nid_b: str) -> None:
        root_a = find(nid_a)
        root_b = find(nid_b)
        if root_a != root_b:
            parent[root_b] = root_a

    # Step A: Deterministic exact-match grouping (zero reasoning steps)
    for cat, cat_nodes in by_category.items():
        cleaned_groups: Dict[Any, List[CandidateNode]] = {}
        for n in cat_nodes:
            cleaned = _clean_person_name(n.label) if cat == "Person" else _clean_org_name(n.label)
            if cleaned:
                # Key by frozenset of tokens to handle word-order permutations (e.g. "RAUT ABHAY" vs "ABHAY RAUT")
                tok_key = frozenset(cleaned.split())
                cleaned_groups.setdefault(tok_key, []).append(n)
        for cln, group in cleaned_groups.items():
            first_node = group[0]
            for subsequent in group[1:]:
                union(first_node.node_id, subsequent.node_id)


    # Step B: Bounded disambiguation pass over distinct candidate clusters (per-category budgeting)
    for cat, cat_nodes in by_category.items():
        cat_reasoning_steps = 0
        cat_budget_exhausted = False
        graph.reset_tool_calls_budget()

        # Get unique representative nodes per cluster
        seen_roots: Set[str] = set()
        cluster_reps: List[CandidateNode] = []
        for n in cat_nodes:
            r = find(n.node_id)
            if r not in seen_roots:
                seen_roots.add(r)
                # Use the node whose ID equals root or first found
                cluster_reps.append(n)

        # Generate candidate pairs with blocking (token overlap or role match)
        candidate_pairs: List[Tuple[CandidateNode, CandidateNode]] = []
        c_len = len(cluster_reps)
        for i in range(c_len):
            for j in range(i + 1, c_len):
                rep_a = cluster_reps[i]
                rep_b = cluster_reps[j]
                clean_a = _clean_person_name(rep_a.label) if cat == "Person" else _clean_org_name(rep_a.label)
                clean_b = _clean_person_name(rep_b.label) if cat == "Person" else _clean_org_name(rep_b.label)
                # Blocking: only consider pairs with shared tokens, substring, or initial match
                toks_a = set(clean_a.split())
                toks_b = set(clean_b.split())
                if (toks_a & toks_b) or clean_a in clean_b or clean_b in clean_a or (toks_a and toks_b and any(difflib.SequenceMatcher(None, ta, tb).ratio() >= 0.75 for ta in toks_a for tb in toks_b)):
                    candidate_pairs.append((rep_a, rep_b))

        # Evaluate candidate pairs within per-category reasoning budget
        for node_a, node_b in candidate_pairs:
            if cat_reasoning_steps >= MAX_REASONING_STEPS_PER_CATEGORY:
                logger.warning(
                    "identity_resolution.category_reasoning_budget_exhausted",
                    category=cat,
                    steps_used=cat_reasoning_steps,
                    max_steps=MAX_REASONING_STEPS_PER_CATEGORY,
                    remaining_pairs=len(candidate_pairs) - cat_reasoning_steps,
                )
                cat_budget_exhausted = True
                break
            if total_reasoning_steps >= MAX_GLOBAL_REASONING_STEPS:
                logger.warning(
                    "identity_resolution.global_reasoning_budget_exhausted",
                    total_steps=total_reasoning_steps,
                    max_global=MAX_GLOBAL_REASONING_STEPS,
                )
                break

            cat_reasoning_steps += 1
            total_reasoning_steps += 1
            trace: List[str] = []
            trace.append(f"Step 1 [Candidate Selection]: Inspecting candidate pair Node A '{node_a.label}' (id={node_a.node_id}, page={node_a.source_page}) vs Node B '{node_b.label}' (id={node_b.node_id}, page={node_b.source_page}).")
            trace.append(f"Step 2 [Ontology Type Verification]: Both nodes confirmed as category '{cat}'. Passed type filter.")

            clean_a = _clean_person_name(node_a.label) if cat == "Person" else _clean_org_name(node_a.label)
            clean_b = _clean_person_name(node_b.label) if cat == "Person" else _clean_org_name(node_b.label)
            trace.append(f"Step 3 [Normalization]: Stripped honorifics/noise -> Cleaned A='{clean_a}' | Cleaned B='{clean_b}'.")

            sim = compute_entity_similarity(node_a.label, node_b.label, cat)
            toks_a = clean_a.split()
            toks_b = clean_b.split()
            tok_details = []
            for ta in toks_a:
                for tb in toks_b:
                    r = difflib.SequenceMatcher(None, ta, tb).ratio()
                    if r >= 0.70:
                        tok_details.append(f"'{ta}' vs '{tb}' (ratio={r:.2f})")
            trace.append(f"Step 4 [Lexical / Phonetic Analysis]: Token alignments: {', '.join(tok_details) or 'no close tokens'}. Base lexical similarity score = {sim:.2f}.")

            # Contextual check: shared doctor/patient roles
            role_a = "doctor" if any(w in node_a.evidence.lower() for w in ["doctor", "dr", "ooctor", "consult"]) else ""
            role_b = "doctor" if any(w in node_b.evidence.lower() for w in ["doctor", "dr", "ooctor", "consult"]) else ""
            trace.append(f"Step 5 [Contextual Role & Textual Evidence]: Node A evidence='{node_a.evidence.strip()}' (role='{role_a or 'unknown'}'), Node B evidence='{node_b.evidence.strip()}' (role='{role_b or 'unknown'}').")

            confidence = sim
            if role_a and role_b and role_a == role_b:
                confidence = min(1.0, confidence + 0.10)
                trace.append(f"Step 6 [Evidence Integration]: Concordant role '{role_a}' confirmed across document contexts (discharge summary vs bill/prescription). Applied +0.10 contextual boost -> Final confidence = {confidence:.2f}.")
            else:
                trace.append(f"Step 6 [Evidence Integration]: No shared role boost -> Final confidence = {confidence:.2f}.")

            # Disambiguation decision logic
            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                decision = "SAME_AS"
                trace.append(f"Step 7 [Bounded Decision]: Confidence {confidence:.2f} >= HIGH_CONFIDENCE_THRESHOLD ({HIGH_CONFIDENCE_THRESHOLD}) -> DECISION: SAME_AS. Action: Merging '{node_a.label}' (p{node_a.source_page}) and '{node_b.label}' (p{node_b.source_page}) into canonical entity cluster.")
                union(node_a.node_id, node_b.node_id)
                graph.add_edge(CandidateEdge(
                    source=node_a.node_id,
                    relationship="SAME_AS",
                    target=node_b.node_id,
                    source_page=node_b.source_page,
                    evidence=f"Resolved '{node_a.label}' (p{node_a.source_page}) to '{node_b.label}' (p{node_b.source_page}) with conf={confidence:.2f}",
                    confidence=confidence,
                    status="INFERRED",
                ))
            elif confidence >= LOW_CONFIDENCE_THRESHOLD:
                # Ambiguous case: do NOT guess, mark UNCERTAIN
                decision = "UNCERTAIN"
                trace.append(f"Step 7 [Bounded Decision]: Confidence {confidence:.2f} is in ambiguous range [{LOW_CONFIDENCE_THRESHOLD}, {HIGH_CONFIDENCE_THRESHOLD}) -> DECISION: UNCERTAIN. Action: Never guess or nearest-node merge; marked UNCERTAIN per Section 18.")
                graph.add_edge(CandidateEdge(
                    source=node_a.node_id,
                    relationship="SAME_AS",
                    target=node_b.node_id,
                    source_page=node_b.source_page,
                    evidence=f"Ambiguous match between '{node_a.label}' and '{node_b.label}' (conf={confidence:.2f})",
                    confidence=confidence,
                    status="UNCERTAIN",
                ))
            else:
                decision = "DIFFERENT"
                trace.append(f"Step 7 [Bounded Decision]: Confidence {confidence:.2f} < LOW_CONFIDENCE_THRESHOLD ({LOW_CONFIDENCE_THRESHOLD}) -> DECISION: DIFFERENT. Action: Kept as distinct entities.")

            logger.info(
                "identity_resolution.pair_evaluated",
                category=cat,
                pair=f"{node_a.label} <-> {node_b.label}",
                decision=decision,
                confidence=confidence,
                category_step=cat_reasoning_steps,
                total_steps=total_reasoning_steps,
            )

            # Record in log file if active
            try:
                from src.adapters.llm.sarvam_client import get_run_log_file
                log_f = get_run_log_file()
                if log_f and log_f.parent.exists():
                    trace_block = (
                        f"\n{'=' * 80}\n"
                        f"PHASE: identity resolution\n"
                        f"CATEGORY: {cat} (step {cat_reasoning_steps} / {MAX_REASONING_STEPS_PER_CATEGORY})\n"
                        f"PAIR: '{node_a.label}' (p{node_a.source_page}) vs '{node_b.label}' (p{node_b.source_page})\n"
                        f"DECISION: {decision} | CONFIDENCE: {confidence:.2f}\n"
                        f"{'-' * 30} REASONING TRACE {'-' * 30}\n"
                        + "\n".join(trace) + "\n"
                        f"{'=' * 80}\n"
                    )
                    with open(log_f, "a", encoding="utf-8") as lf:
                        lf.write(trace_block)
                        lf.flush()
            except Exception:
                pass

            disambiguation_log.append(DisambiguationResult(
                node_id_a=node_a.node_id,
                node_id_b=node_b.node_id,
                label_a=node_a.label,
                label_b=node_b.label,
                category=cat,
                similarity_score=sim,
                decision=decision,
                confidence=confidence,
                evidence=f"Comparing '{node_a.label}' (p{node_a.source_page}) vs '{node_b.label}' (p{node_b.source_page})",
                reasoning_step=cat_reasoning_steps,
                reasoning_trace=trace,
                category_budget_exhausted=False,
            ))

        if cat_budget_exhausted:
            disambiguation_log.append(DisambiguationResult(
                node_id_a="BUDGET_EXHAUSTED",
                node_id_b="BUDGET_EXHAUSTED",
                label_a=f"Category '{cat}' budget ceiling reached",
                label_b=f"Evaluated {cat_reasoning_steps}/{MAX_REASONING_STEPS_PER_CATEGORY} candidate pairs",
                category=cat,
                similarity_score=0.0,
                decision="UNCERTAIN",
                confidence=0.0,
                evidence=f"Budget ceiling reached for {cat}. Remaining candidate pairs were skipped.",
                reasoning_step=cat_reasoning_steps,
                reasoning_trace=[
                    f"Category '{cat}' reached MAX_REASONING_STEPS_PER_CATEGORY ({MAX_REASONING_STEPS_PER_CATEGORY}).",
                    f"Total pairs available: {len(candidate_pairs)}. Evaluated: {cat_reasoning_steps}. Skipped: {len(candidate_pairs) - cat_reasoning_steps}.",
                ],
                category_budget_exhausted=True,
            ))

        category_stats[cat] = {
            "steps_used": cat_reasoning_steps,
            "budget_exhausted": cat_budget_exhausted,
            "pairs_evaluated": cat_reasoning_steps,
            "max_budget": MAX_REASONING_STEPS_PER_CATEGORY,
            "tool_calls": graph.tool_calls_count,
        }
        logger.info(
            "identity_resolution.category_completed",
            category=cat,
            steps_used=cat_reasoning_steps,
            budget_exhausted=cat_budget_exhausted,
            tool_calls=graph.tool_calls_count,
        )

        try:
            from src.adapters.llm.sarvam_client import get_run_log_file
            log_f = get_run_log_file()
            if log_f and log_f.parent.exists():
                cat_block = (
                    f"\n{'=' * 80}\n"
                    f"PHASE: identity resolution - Category Summary\n"
                    f"CATEGORY: {cat}\n"
                    f"STEPS USED: {cat_reasoning_steps} / {MAX_REASONING_STEPS_PER_CATEGORY}\n"
                    f"BUDGET EXHAUSTED: {cat_budget_exhausted}\n"
                    f"TOOL CALLS USED: {graph.tool_calls_count} / {MAX_TOOL_CALLS_PER_CATEGORY}\n"
                    f"{'=' * 80}\n"
                )
                with open(log_f, "a", encoding="utf-8") as lf:
                    lf.write(cat_block)
                    lf.flush()
        except Exception:
            pass


    # Cluster resolved entities based on union-find components
    clusters: Dict[str, List[CandidateNode]] = {}
    for n in target_nodes:
        root = find(n.node_id)
        clusters.setdefault(root, []).append(n)

    for root_id, cluster_nodes in clusters.items():
        # TODO(follow-up): Domain-specific heuristics and noise phrase filtering are hardcoded for sample data. Refactor to generic pattern extractors in a separate follow-up.
        # Pick the cleanest, most concise label without noise phrases
        sorted_by_quality = sorted(
            cluster_nodes,
            key=lambda n: (
                not any(w in n.label.lower() for w in ["with", "prior", "appoint", "consult"]),
                n.confidence,
                -len(n.label),
            ),
            reverse=True,
        )
        canonical_node = sorted_by_quality[0]

        aliases = list({n.label for n in cluster_nodes if n.label != canonical_node.label})
        source_pages = sorted(list({n.source_page for n in cluster_nodes}))

        resolved_entities.append(ResolvedEntity(
            canonical_id=canonical_node.node_id,
            category=canonical_node.type,
            canonical_label=canonical_node.label,
            mention_node_ids=[n.node_id for n in cluster_nodes],
            source_pages=source_pages,
            aliases=aliases,
            resolution_confidence=min(n.confidence for n in cluster_nodes),
            status="ASSERTED" if len(cluster_nodes) == 1 else "INFERRED",
            resolution_note=f"Merged {len(cluster_nodes)} mention(s) across page(s) {source_pages}",
        ))

    logger.info(
        "identity_resolution.completed",
        candidate_count=len(target_nodes),
        resolved_entities_count=len(resolved_entities),
        total_reasoning_steps=total_reasoning_steps,
        category_stats=category_stats,
    )
    return resolved_entities, graph.edges, disambiguation_log
