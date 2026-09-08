"""
File: identifier_matching.py
Purpose: Deterministic, zero-LLM-cost matching pass for Identifier-type graph nodes.
         Resolves duplicate and OCR-garbled/truncated identifier mentions (phone numbers,
         HS numbers, admission numbers, policy numbers, bill/voucher numbers)
         without routing through Step 6's Person/Organization LLM disambiguation.

Design & Constraints (Approved Plan v2):
  1. Strict Kind-Gating:
     - Detects kind: 'phone', 'hs_no', 'admission', 'policy', 'bill', or 'unknown'.
     - If kind_a and kind_b are both known and kind_a != kind_b:
       Instant reject (KIND_MISMATCH, DIFFERENT, conf=0.0). No cross-system comparison.
     - If one or both kinds are unknown:
       Fuzzy rules (prefix/suffix, edit distance) are strictly PROHIBITED.
       Only exact match >= 8 chars permitted (state=INFERRED, conf=0.85).
  2. Raised Exact-Match Threshold (Rule A):
     - Exact match >= 6 chars (same kind) -> state=ASSERTED, conf=1.0.
     - Exact match 4-5 chars (same kind) -> state=INFERRED, conf=0.75.
     - Exact match < 4 chars -> DIFFERENT, no merge.
  3. Prefix/Suffix Overlap >= 7 digits (Rule B):
     - Same kind, numeric, overlap >= 7 -> state=INFERRED, conf=overlap_len / max(len(a), len(b)).
     - Preserves BOTH source values and BOTH source pages in node and edge evidence.
  4. Edit Distance 1-2 on same length >= 6 (Rule C):
     - Same kind, numeric, edit_dist in {1, 2} -> state=INFERRED, conf=(len - dist) / len.
  5. Explicit UNCERTAIN Near-Miss Band:
     - Prefix overlap exactly 6 digits on strings >= 8 digits -> UNCERTAIN edge created,
       cluster NOT merged.
     - Edit distance exactly 3 on strings >= 8 digits -> UNCERTAIN edge created,
       cluster NOT merged.
  6. Non-matching pairs (overlap <= 5, edit dist >= 4):
     - DIFFERENT, zero graph edges created, no clutter.

Owner: engineer-a@idp-pilot
Created: 2026-09-07
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.ai.layer3_extraction.extractor import CandidateEdge, CandidateNode
from src.ai.layer3_extraction.identity_resolution import ResolvedEntity
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREFIX_PATTERN = re.compile(
    r"^(phone|hs\s*no|admission\s*no|policy\s*no|bill\s*no|voucher\s*no|invoice\s*no|receipt\s*no|contact\s*no|tel|mob|ipd|uhid|opd)[\s.:#-]*",
    re.IGNORECASE,
)


class IdentifierMatchResult(BaseModel):
    node_id_a: str
    node_id_b: str
    label_a: str
    label_b: str
    kind_a: Optional[str]
    kind_b: Optional[str]
    normalized_a: str
    normalized_b: str
    match_type: str  # "EXACT" | "EXACT_SHORT" | "EXACT_UNKNOWN_KIND" | "PREFIX_SUFFIX" | "EDIT_DISTANCE" | "AMBIGUOUS_PREFIX" | "AMBIGUOUS_EDIT_DIST" | "KIND_MISMATCH" | "NO_MATCH"
    decision: str    # "SAME_AS" | "DIFFERENT" | "UNCERTAIN"
    confidence: float
    state: str       # "ASSERTED" | "INFERRED" | "UNCERTAIN" | "NONE"
    reason: str


def detect_identifier_kind(text: str) -> Optional[str]:
    """Identify category hint from label prefixes and context."""
    t = text.lower()
    if any(k in t for k in ["phone", "contact", "mobile", "tel"]):
        return "phone"
    if any(k in t for k in ["hs no", "hs num", "hs_no", "hs#"]):
        return "hs_no"
    if any(k in t for k in ["admission", "ipd", "uhid", "opd"]):
        return "admission"
    if any(k in t for k in ["policy", "claim"]):
        return "policy"
    if any(k in t for k in ["bill", "invoice", "voucher", "receipt"]):
        return "bill"
    return None


def normalize_identifier_value(label: str) -> Tuple[str, bool, Optional[str]]:
    """
    Normalize identifier value:
    - Detect kind (phone, hs_no, admission, policy, bill).
    - Strip prefix labels.
    - If numeric-looking: strip non-digits.
    - If alphanumeric: strip whitespace/punctuation, keep letters.
    Returns: (normalized_str, is_numeric, detected_kind)
    """
    kind = detect_identifier_kind(label)
    clean = label.strip()

    # Strip recognized prefix words
    clean = PREFIX_PATTERN.sub("", clean).strip()

    # If label still has a colon where prefix has no digits (e.g. "Invoice: 1234")
    if ":" in clean:
        prefix_part, rest_part = clean.split(":", 1)
        if not any(c.isdigit() for c in prefix_part):
            clean = rest_part.strip()

    num_digits = sum(1 for c in clean if c.isdigit())
    num_letters = sum(1 for c in clean if c.isalpha())

    # Pure numeric strings (e.g. phone numbers, HS numbers, admission codes)
    if num_letters == 0 and num_digits > 0:
        norm = "".join(c for c in clean if c.isdigit())
        return norm, True, kind

    # Alphanumeric identifiers (e.g. policy numbers, alphanumeric vouchers)
    if num_letters > 0 and num_digits > 0:
        norm = "".join(c.upper() for c in clean if c.isalnum())
        return norm, False, kind

    # Fallback: remove non-alphanumerics
    norm = "".join(c.upper() for c in clean if c.isalnum())
    return norm, False, kind


def run_identifier_matching(
    nodes: List[CandidateNode],
    edges: List[CandidateEdge],
) -> Tuple[List[ResolvedEntity], List[CandidateEdge], List[IdentifierMatchResult]]:
    """
    Execute deterministic, zero-LLM comparison over all type=Identifier nodes
    with strict kind-gating, raised Rule A thresholds, and an explicit UNCERTAIN near-miss band.
    """
    identifier_nodes = [n for n in nodes if n.type == "Identifier"]
    new_edges = list(edges)
    match_log: List[IdentifierMatchResult] = []

    if not identifier_nodes:
        logger.info("identifier_matching.no_identifier_nodes_found")
        return [], new_edges, match_log

    logger.info("identifier_matching.start", identifier_nodes_count=len(identifier_nodes))

    # Pre-normalize all identifier nodes
    parsed_meta: Dict[str, Tuple[str, bool, Optional[str]]] = {}
    for node in identifier_nodes:
        parsed_meta[node.node_id] = normalize_identifier_value(node.label)

    # Union-Find data structure for clustering
    parent: Dict[str, str] = {n.node_id: n.node_id for n in identifier_nodes}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Track cluster state (ASSERTED vs INFERRED)
    cluster_edge_states: Dict[str, Set[str]] = {n.node_id: set() for n in identifier_nodes}
    cluster_edge_confs: Dict[str, List[float]] = {n.node_id: [] for n in identifier_nodes}

    # Pairwise comparison
    for i in range(len(identifier_nodes)):
        node_a = identifier_nodes[i]
        norm_a, is_num_a, kind_a = parsed_meta[node_a.node_id]

        for j in range(i + 1, len(identifier_nodes)):
            node_b = identifier_nodes[j]
            norm_b, is_num_b, kind_b = parsed_meta[node_b.node_id]

            match_type = "NO_MATCH"
            decision = "DIFFERENT"
            confidence = 0.0
            state = "NONE"
            reason = ""

            # -------------------------------------------------------------
            # ITEM 1: CRITICAL KIND-GATING
            # -------------------------------------------------------------
            if kind_a is not None and kind_b is not None and kind_a != kind_b:
                match_type = "KIND_MISMATCH"
                decision = "DIFFERENT"
                confidence = 0.0
                state = "NONE"
                reason = (
                    f"Kind mismatch: kind_a='{kind_a}' vs kind_b='{kind_b}'. "
                    f"Cross-reference system comparison rejected."
                )

            elif kind_a is None or kind_b is None:
                # UNKNOWN KIND POLICY:
                # Prohibit fuzzy matching (Rules B & C). Only allow exact match on high-entropy strings (>= 8 chars).
                if norm_a == norm_b and len(norm_a) >= 8:
                    match_type = "EXACT_UNKNOWN_KIND"
                    decision = "SAME_AS"
                    confidence = 0.85
                    state = "INFERRED"
                    reason = (
                        f"Exact match on unclassified identifier with high entropy "
                        f"(len={len(norm_a)} >= 8): '{norm_a}'"
                    )
                else:
                    match_type = "NO_MATCH"
                    decision = "DIFFERENT"
                    confidence = 0.0
                    state = "NONE"
                    reason = (
                        f"Unknown identifier kind for pair ({kind_a}, {kind_b}) — "
                        f"fuzzy matching barred, length insufficient for unclassified exact match"
                    )

            else:
                # kind_a == kind_b and both are known!
                len_a, len_b = len(norm_a), len(norm_b)
                max_len = max(len_a, len_b)

                # ---------------------------------------------------------
                # ITEM 2: RAISED RULE A EXACT MATCH
                # ---------------------------------------------------------
                if norm_a == norm_b:
                    if len_a >= 6:
                        # Full confidence ASSERTED exact match
                        match_type = "EXACT"
                        decision = "SAME_AS"
                        confidence = 1.0
                        state = "ASSERTED"
                        reason = f"Exact normalized match on kind '{kind_a}' (len={len_a} >= 6): '{norm_a}'"
                    elif 4 <= len_a <= 5:
                        # Short exact match with known matching kind -> lower confidence INFERRED
                        match_type = "EXACT_SHORT"
                        decision = "SAME_AS"
                        confidence = 0.75
                        state = "INFERRED"
                        reason = f"Short exact match on kind '{kind_a}' (len={len_a} in [4, 5]): '{norm_a}'"
                    else:
                        # len < 4: Too low entropy, accidental collision risk
                        match_type = "NO_MATCH"
                        decision = "DIFFERENT"
                        confidence = 0.0
                        state = "NONE"
                        reason = f"Exact match rejected due to low entropy (len={len_a} < 4): '{norm_a}'"

                elif is_num_a and is_num_b:
                    # Numeric fuzzy comparisons (Rules B, C and Near-Miss UNCERTAIN Band)
                    # Common prefix length
                    prefix_len = 0
                    for ca, cb in zip(norm_a, norm_b):
                        if ca == cb:
                            prefix_len += 1
                        else:
                            break

                    # Common suffix length
                    suffix_len = 0
                    for ca, cb in zip(reversed(norm_a), reversed(norm_b)):
                        if ca == cb:
                            suffix_len += 1
                        else:
                            break

                    # Substring containment
                    sub_len = 0
                    if norm_a in norm_b:
                        sub_len = len_a
                    elif norm_b in norm_a:
                        sub_len = len_b

                    max_overlap = max(prefix_len, suffix_len, sub_len)

                    # -----------------------------------------------------
                    # RULE B: Numeric Prefix/Suffix match >= 7 digits
                    # -----------------------------------------------------
                    if max_overlap >= 7:
                        match_type = "PREFIX_SUFFIX"
                        decision = "SAME_AS"
                        confidence = round(max_overlap / max_len, 3)
                        state = "INFERRED"
                        reason = (
                            f"Numeric prefix/suffix overlap of {max_overlap} >= 7 digits "
                            f"on kind '{kind_a}' (overlap={max_overlap}/{max_len})"
                        )

                    # -----------------------------------------------------
                    # RULE C: Edit distance in {1, 2} on same length >= 6
                    # -----------------------------------------------------
                    elif len_a == len_b and len_a >= 6 and 1 <= sum(1 for ca, cb in zip(norm_a, norm_b) if ca != cb) <= 2:
                        edit_dist = sum(1 for ca, cb in zip(norm_a, norm_b) if ca != cb)
                        match_type = "EDIT_DISTANCE"
                        decision = "SAME_AS"
                        confidence = round((len_a - edit_dist) / len_a, 3)
                        state = "INFERRED"
                        reason = (
                            f"Numeric edit distance {edit_dist} <= 2 on same-length "
                            f"string of {len_a} chars on kind '{kind_a}'"
                        )

                    # -----------------------------------------------------
                    # ITEM 3: EXPLICIT UNCERTAIN NEAR-MISS BAND
                    # -----------------------------------------------------
                    elif max_overlap == 6 and max_len >= 8:
                        # Ambiguous borderline prefix near-miss
                        match_type = "AMBIGUOUS_PREFIX"
                        decision = "UNCERTAIN"
                        confidence = round(6 / max_len, 3)
                        state = "UNCERTAIN"
                        reason = (
                            f"Borderline prefix overlap of exactly 6 digits on {max_len}-digit string "
                            f"(kind '{kind_a}') — marked UNCERTAIN audit edge"
                        )

                    elif len_a == len_b and len_a >= 8 and sum(1 for ca, cb in zip(norm_a, norm_b) if ca != cb) == 3:
                        # Ambiguous borderline edit distance near-miss
                        match_type = "AMBIGUOUS_EDIT_DIST"
                        decision = "UNCERTAIN"
                        confidence = round((len_a - 3) / len_a, 3)
                        state = "UNCERTAIN"
                        reason = (
                            f"Borderline edit distance of exactly 3 on {len_a}-digit string "
                            f"(kind '{kind_a}') — marked UNCERTAIN audit edge"
                        )

                    else:
                        # -------------------------------------------------
                        # CLEAR NON-MATCH: DIFFERENT
                        # -------------------------------------------------
                        match_type = "NO_MATCH"
                        decision = "DIFFERENT"
                        confidence = 0.0
                        state = "NONE"
                        reason = (
                            f"Below threshold on kind '{kind_a}': overlap={max_overlap} (<=5), "
                            f"lens=({len_a}, {len_b})"
                        )

                else:
                    match_type = "NO_MATCH"
                    decision = "DIFFERENT"
                    confidence = 0.0
                    state = "NONE"
                    reason = f"Non-matching alphanumeric identifiers '{norm_a}' vs '{norm_b}'"

            # -------------------------------------------------------------
            # Graph Edge and Cluster Updates
            # -------------------------------------------------------------
            if decision == "SAME_AS":
                evidence_str = (
                    f"Source A (p{node_a.source_page}): '{node_a.label}' | "
                    f"Source B (p{node_b.source_page}): '{node_b.label}' | "
                    f"Rule: {match_type} (conf={confidence:.3f})"
                )
                new_edge = CandidateEdge(
                    source=node_a.node_id,
                    relationship="SAME_AS",
                    target=node_b.node_id,
                    source_page=node_b.source_page,
                    evidence=evidence_str,
                    confidence=confidence,
                    status=state,
                )
                new_edges.append(new_edge)
                union(node_a.node_id, node_b.node_id)
                cluster_edge_states[node_a.node_id].add(state)
                cluster_edge_states[node_b.node_id].add(state)
                cluster_edge_confs[node_a.node_id].append(confidence)
                cluster_edge_confs[node_b.node_id].append(confidence)

            elif decision == "UNCERTAIN":
                # Create UNCERTAIN audit edge per Section 18 of Graph Memory POC,
                # but DO NOT merge into cluster (no union call)
                evidence_str = (
                    f"Source A (p{node_a.source_page}): '{node_a.label}' | "
                    f"Source B (p{node_b.source_page}): '{node_b.label}' | "
                    f"Near-Miss: {match_type} (conf={confidence:.3f})"
                )
                new_edge = CandidateEdge(
                    source=node_a.node_id,
                    relationship="SAME_AS",
                    target=node_b.node_id,
                    source_page=node_b.source_page,
                    evidence=evidence_str,
                    confidence=confidence,
                    status="UNCERTAIN",
                )
                new_edges.append(new_edge)

            # Log comparison result
            match_res = IdentifierMatchResult(
                node_id_a=node_a.node_id,
                node_id_b=node_b.node_id,
                label_a=node_a.label,
                label_b=node_b.label,
                kind_a=kind_a,
                kind_b=kind_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                match_type=match_type,
                decision=decision,
                confidence=confidence,
                state=state,
                reason=reason,
            )
            match_log.append(match_res)

            if decision in ("SAME_AS", "UNCERTAIN"):
                logger.info(
                    "identifier_matching.pair_evaluated",
                    pair=f"{node_a.label} (p{node_a.source_page}) <-> {node_b.label} (p{node_b.source_page})",
                    kinds=f"{kind_a} vs {kind_b}",
                    decision=decision,
                    match_type=match_type,
                    confidence=confidence,
                    state=state,
                )
                # Write to active run log file for positive/ambiguous matches
                try:
                    from src.adapters.llm.sarvam_client import get_run_log_file
                    log_f = get_run_log_file()
                    if log_f and log_f.parent.exists():
                        trace_block = (
                            f"\n{'=' * 80}\n"
                            f"PHASE: identifier matching\n"
                            f"PAIR: '{node_a.label}' (p{node_a.source_page}, kind={kind_a}) vs "
                            f"'{node_b.label}' (p{node_b.source_page}, kind={kind_b})\n"
                            f"NORMALIZED: '{norm_a}' vs '{norm_b}'\n"
                            f"MATCH TYPE: {match_type} | DECISION: {decision} | STATE: {state} | CONFIDENCE: {confidence:.3f}\n"
                            f"REASON: {reason}\n"
                            f"{'=' * 80}\n"
                        )
                        with open(log_f, "a", encoding="utf-8") as lf:
                            lf.write(trace_block)
                            lf.flush()
                except Exception:
                    pass
            else:
                logger.debug(
                    "identifier_matching.pair_evaluated",
                    pair=f"{node_a.label} (p{node_a.source_page}) <-> {node_b.label} (p{node_b.source_page})",
                    kinds=f"{kind_a} vs {kind_b}",
                    decision=decision,
                    match_type=match_type,
                    confidence=confidence,
                    state=state,
                )

    # Group nodes into clusters
    clusters: Dict[str, List[CandidateNode]] = {}
    for n in identifier_nodes:
        root = find(n.node_id)
        clusters.setdefault(root, []).append(n)

    resolved_entities: List[ResolvedEntity] = []

    for root_id, cluster_nodes in clusters.items():
        # Pick canonical node: longest normalized value, then highest confidence, then earlier page
        sorted_nodes = sorted(
            cluster_nodes,
            key=lambda n: (
                len(parsed_meta[n.node_id][0]),
                n.confidence,
                -n.source_page,
            ),
            reverse=True,
        )
        canonical_node = sorted_nodes[0]

        # Gather cluster states and confidences
        all_states: Set[str] = set()
        all_confs: List[float] = []
        for n in cluster_nodes:
            all_states.update(cluster_edge_states[n.node_id])
            all_confs.extend(cluster_edge_confs[n.node_id])

        if len(cluster_nodes) == 1:
            cluster_status = "ASSERTED"
            cluster_confidence = canonical_node.confidence
        else:
            cluster_status = "INFERRED" if "INFERRED" in all_states else "ASSERTED"
            cluster_confidence = min(all_confs) if all_confs else canonical_node.confidence

        # Record BOTH source values and BOTH source pages in node evidence
        if len(cluster_nodes) > 1:
            sources_summary = " ; ".join(
                f"Page {n.source_page}: '{n.label}' (conf={n.confidence:.3f})"
                for n in cluster_nodes
            )
            for n in cluster_nodes:
                n.evidence = (
                    f"Merged identifier mentions: [{sources_summary}] | "
                    f"Original OCR: {n.evidence}"
                )

        aliases = sorted(list({n.label for n in cluster_nodes if n.label != canonical_node.label}))
        source_pages = sorted(list({n.source_page for n in cluster_nodes}))

        note = (
            f"Merged {len(cluster_nodes)} identifier mention(s) across page(s) {source_pages}: "
            f"[{', '.join(repr(n.label) for n in cluster_nodes)}]"
        )

        resolved_entities.append(
            ResolvedEntity(
                canonical_id=canonical_node.node_id,
                category="Identifier",
                canonical_label=canonical_node.label,
                mention_node_ids=[n.node_id for n in cluster_nodes],
                source_pages=source_pages,
                aliases=aliases,
                resolution_confidence=cluster_confidence,
                status=cluster_status,
                resolution_note=note,
            )
        )

    logger.info(
        "identifier_matching.completed",
        candidate_count=len(identifier_nodes),
        resolved_entities_count=len(resolved_entities),
        comparisons_made=len(match_log),
        new_edges_created=len(new_edges) - len(edges),
    )

    return resolved_entities, new_edges, match_log
