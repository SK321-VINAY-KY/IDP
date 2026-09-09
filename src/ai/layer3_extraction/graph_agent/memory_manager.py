"""
File: memory_manager.py
Purpose: Central Blackboard / Memory Manager for concurrent Layer 3 graph extraction.
Coordinates concurrent page workers, merges PageDeltas canonically,
remaps worker-local IDs to canonical IDs, rewrites edge endpoints,
and reconciles cross-page references post-merge.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import Evidence, GraphEdge, GraphNode
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UnresolvedRef:
    """An anaphoric or cross-reference mention that could not be resolved during page extraction."""
    source_node_id: str
    phrase: str
    target_type: str
    page_number: int
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "phrase": self.phrase,
            "target_type": self.target_type,
            "page_number": self.page_number,
            "rationale": self.rationale,
        }


@dataclass
class PageDelta:
    """
    Immutable extraction delta emitted by an independent concurrent page worker.
    Contains local entities, relationships, unresolved references, and local ID maps.
    Must NOT mutate GraphMemory directly.
    """
    page_number: int
    entities: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphEdge] = field(default_factory=list)
    unresolved_references: List[UnresolvedRef] = field(default_factory=list)
    raw_id_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "entities": [n.to_dict() for n in self.entities],
            "relationships": [e.to_dict() for e in self.relationships],
            "unresolved_references": [r.to_dict() for r in self.unresolved_references],
            "raw_id_map": dict(self.raw_id_map),
        }


# Semantic keyword-to-anchor mapping covering medical, financial, legal, business domains
ANCHOR_KEYWORD_MAP: List[Tuple[re.Pattern, str]] = [
    # Healthcare
    (re.compile(r"(patient|subject|admittee)", re.I), "Patient"),
    (re.compile(r"(doctor|physician|surgeon|specialist|clinician)", re.I), "Doctor"),
    (re.compile(r"(hospital|clinic|facility|provider|institution)", re.I), "Hospital"),
    # Insurance & Claims
    (re.compile(r"(policy|coverage)", re.I), "Policy"),
    (re.compile(r"(claim|preauth)", re.I), "Claim"),
    # Financial / Banking / Commerce
    (re.compile(r"(account|bank|portfolio)", re.I), "Account"),
    (re.compile(r"(customer|client|buyer|purchaser)", re.I), "Customer"),
    (re.compile(r"(vendor|merchant|seller|supplier)", re.I), "Vendor"),
    (re.compile(r"(invoice|bill|receipt)", re.I), "Invoice"),
    (re.compile(r"(transaction|payment|disbursement|transfer)", re.I), "Transaction"),
    (re.compile(r"(order|purchase_order)", re.I), "Order"),
    # Legal / Contracts
    (re.compile(r"(plaintiff|claimant)", re.I), "Plaintiff"),
    (re.compile(r"(defendant|respondent)", re.I), "Defendant"),
    (re.compile(r"(case|litigation|lawsuit|matter)", re.I), "Case"),
    (re.compile(r"(contract|agreement|accord)", re.I), "Contract"),
    (re.compile(r"(court|tribunal|jurisdiction)", re.I), "Court"),
    # Corporate / Employment
    (re.compile(r"(employee|staff|personnel|worker)", re.I), "Employee"),
    (re.compile(r"(employer|organization|company|corporation|firm)", re.I), "Organization"),
]


def derive_anchor_types_from_schema(schema: type[BaseModel]) -> List[str]:
    """
    Derive anchor entity types dynamically from target schema fields using
    deterministic semantic heuristics across medical, financial, legal, etc. schemas.
    Does NOT hardcode healthcare-only concepts.
    """
    if not schema or not hasattr(schema, "model_fields"):
        return []

    derived: List[str] = []
    seen: Set[str] = set()

    for field_name in schema.model_fields.keys():
        clean_name = field_name.strip()
        matched = False
        for pattern, anchor_type in ANCHOR_KEYWORD_MAP:
            if pattern.search(clean_name):
                if anchor_type not in seen:
                    derived.append(anchor_type)
                    seen.add(anchor_type)
                matched = True
                break

        if not matched:
            # Suffix-based generic entity derivation: e.g. "claimant_name" -> "Claimant"
            m_name = re.match(r"^([a-zA-Z0-9]+)_(?:name|id|number|num|code)$", clean_name, re.I)
            if m_name:
                stem = m_name.group(1).capitalize()
                if len(stem) > 2 and stem not in seen:
                    derived.append(stem)
                    seen.add(stem)

    # Graceful fallback: if no semantic match, pick up to 3 non-empty field names capitalized
    if not derived:
        for fname in list(schema.model_fields.keys())[:3]:
            cand = re.sub(r"[^a-zA-Z0-9]+", " ", fname).strip().title().replace(" ", "")
            if cand and cand not in seen:
                derived.append(cand)
                seen.add(cand)

    return derived


class GraphMemoryManager:
    """
    Central Blackboard / Memory Manager for concurrent Layer 3 extraction.

    Responsibilities:
    - Owns one GraphMemory instance.
    - Provides targeted anchor context for page workers.
    - Ingests PageDeltas with serialized canonical merge and entity fusion.
    - Scopes worker-local ID to canonical ID remapping by (page_number, local_id).
    - Rewrites relationship endpoints to canonical IDs.
    - Collects unresolved references across pages.
    - Reconciles cross-page references post-merge.
    """

    def __init__(
        self,
        anchor_types: Optional[List[str]] = None,
        anchor_max_k: int = 5,
        schema: Optional[type[BaseModel]] = None,
    ) -> None:
        self.graph = GraphMemory()
        self.anchor_max_k = max(1, int(anchor_max_k))
        self._schema = schema

        if anchor_types:
            self.anchor_types = [str(a).strip() for a in anchor_types if str(a).strip()]
        elif schema is not None:
            self.anchor_types = derive_anchor_types_from_schema(schema)
        else:
            self.anchor_types = []

        self._unresolved: List[UnresolvedRef] = []
        self._lock = asyncio.Lock()
        # Scoped mapping: (page_number, local_id_or_key) -> canonical_id
        self._id_remap: Dict[Tuple[int, str], str] = {}

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    async def get_context_for_page(self, page_number: int) -> List[Dict[str, Any]]:
        """
        Return targeted anchor context only, rather than dumping the entire graph.
        Anchor ranking considers:
        1. Configured/derived anchor type match
        2. Node category (EXPLICIT > CONTEXTUAL > REFERENCE_SUPPORT)
        3. Node confidence
        4. Page coverage / proximity to page_number
        """
        async with self._lock:
            if not self.graph._nodes:
                return []

            norm_anchors = {self._normalize(a) for a in self.anchor_types}
            candidates: List[Tuple[float, GraphNode]] = []

            for node in self.graph._nodes.values():
                type_norm = self._normalize(node.type)
                label_norm = self._normalize(node.label)

                is_anchor_match = (
                    (type_norm in norm_anchors)
                    or (label_norm in norm_anchors)
                    or any(a in type_norm or a in label_norm for a in norm_anchors)
                ) if norm_anchors else True

                # Calculate ranking score
                score = 0.0
                if is_anchor_match:
                    score += 10.0

                if node.category == "EXPLICIT":
                    score += 5.0
                elif node.category == "CONTEXTUAL":
                    score += 2.0

                score += node.confidence * 2.0

                if node.source_pages:
                    max_p = max(node.source_pages)
                    # Proximity score: closer or earlier pages score higher
                    page_diff = abs(page_number - max_p)
                    score += max(0.0, 3.0 - (page_diff * 0.5))

                candidates.append((score, node))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top_nodes = [node for _, node in candidates[:self.anchor_max_k]]
            return [n.to_dict() for n in top_nodes]

    async def ingest_page_delta(self, delta: PageDelta) -> None:
        """
        Canonically ingest a PageDelta into the central GraphMemory under lock.
        1. Resolves and deduplicates entities against canonical GraphMemory.
        2. Maintains scoped (page_number, local_id) -> canonical_id mapping.
        3. Rewrites relationship endpoints using canonical IDs.
        4. Collects unresolved references for post-merge resolution.
        """
        async with self._lock:
            page_num = delta.page_number

            # ---------------------------------------------------------
            # Step 1: Ingest and Canonicalize Entities (Nodes)
            # ---------------------------------------------------------
            for ent in delta.entities:
                raw_id = ent.id
                ent_type = ent.type or "Entity"
                ent_label = ent.label or ent_type
                ent_val = str(ent.value or "").strip()
                if not ent_val:
                    continue

                canonical_id: Optional[str] = None

                # 1A. Check if an explicit reused_node_id was specified
                reused_id = ent.properties.get("reused_node_id")
                if reused_id and self.graph.get_node(reused_id):
                    canonical_id = reused_id
                    self.graph.update_node(
                        canonical_id,
                        source_page=page_num,
                        evidence=ent.evidence,
                        confidence=ent.confidence,
                        aliases=ent.aliases,
                    )
                    if ent.category == "EXPLICIT":
                        cand_node = self.graph.get_node(canonical_id)
                        if cand_node:
                            cand_node.category = "EXPLICIT"
                else:
                    # 1B. Check existing GraphMemory reuse candidates
                    reuse_cand = self.graph.find_reuse_candidate(
                        node_type=ent_type,
                        value=ent_val,
                        label=ent_label,
                    )
                    if not reuse_cand:
                        # Fallback alias lookup
                        reuse_cand = self.graph.lookup_alias(ent_val)

                    if reuse_cand and reuse_cand.id:
                        canonical_id = reuse_cand.id
                        self.graph.update_node(
                            canonical_id,
                            source_page=page_num,
                            evidence=ent.evidence,
                            confidence=ent.confidence,
                            aliases=ent.aliases,
                        )
                        if ent.category == "EXPLICIT":
                            reuse_cand.category = "EXPLICIT"
                    else:
                        # 1C. Create brand new canonical node
                        if raw_id and raw_id not in self.graph._nodes:
                            node_id = raw_id
                        else:
                            node_id = f"node_{page_num}_{uuid.uuid4().hex[:6]}"

                        pages = list(ent.source_pages) if ent.source_pages else [page_num]
                        if page_num not in pages:
                            pages.append(page_num)

                        props = dict(ent.properties)
                        props["raw_id"] = raw_id

                        new_node = GraphNode(
                            id=node_id,
                            type=ent_type,
                            label=ent_label,
                            value=ent_val,
                            properties=props,
                            aliases=list(ent.aliases),
                            evidence=list(ent.evidence),
                            source_pages=pages,
                            confidence=ent.confidence,
                            status=ent.status or "ASSERTED",
                            category=ent.category or "CONTEXTUAL",
                        )
                        self.graph.create_node(new_node)
                        canonical_id = node_id

                # Register scoped mappings for this page worker
                self._id_remap[(page_num, raw_id)] = canonical_id
                self._id_remap[(page_num, self._normalize(raw_id))] = canonical_id
                self._id_remap[(page_num, self._normalize(ent_val))] = canonical_id
                if ent_label:
                    self._id_remap[(page_num, self._normalize(ent_label))] = canonical_id

                # Register any aliases on canonical node
                for a in ent.aliases:
                    if a:
                        self.graph.add_alias(canonical_id, a)
                        self._id_remap[(page_num, self._normalize(a))] = canonical_id

            # Incorporate worker's raw_id_map if provided
            for raw_k, local_val in delta.raw_id_map.items():
                target_canonical = (
                    self._id_remap.get((page_num, local_val))
                    or self._id_remap.get((page_num, self._normalize(local_val)))
                )
                if target_canonical:
                    self._id_remap[(page_num, raw_k)] = target_canonical
                    self._id_remap[(page_num, self._normalize(raw_k))] = target_canonical

            # ---------------------------------------------------------
            # Step 2: Ingest and Rewrite Relationships (Edges)
            # ---------------------------------------------------------
            for edge in delta.relationships:
                raw_src = str(edge.source_node or "").strip()
                raw_tgt = str(edge.target_node or "").strip()
                if not raw_src or not raw_tgt:
                    continue

                # Remap source node
                canon_src = (
                    self._id_remap.get((page_num, raw_src))
                    or self._id_remap.get((page_num, self._normalize(raw_src)))
                )
                if not canon_src and raw_src in self.graph._nodes:
                    canon_src = raw_src
                if not canon_src:
                    cands = self.graph.search_nodes(query=raw_src, limit=1)
                    if cands:
                        canon_src = cands[0].id

                # Remap target node
                canon_tgt = (
                    self._id_remap.get((page_num, raw_tgt))
                    or self._id_remap.get((page_num, self._normalize(raw_tgt)))
                )
                if not canon_tgt and raw_tgt in self.graph._nodes:
                    canon_tgt = raw_tgt
                if not canon_tgt:
                    cands = self.graph.search_nodes(query=raw_tgt, limit=1)
                    if cands:
                        canon_tgt = cands[0].id

                # Validate both endpoints are canonical and non-reflexive
                if canon_src and canon_tgt and canon_src != canon_tgt:
                    if canon_src in self.graph._nodes and canon_tgt in self.graph._nodes:
                        edge_id = f"edge_{page_num}_{uuid.uuid4().hex[:6]}"
                        canon_edge = GraphEdge(
                            id=edge_id,
                            source_node=canon_src,
                            target_node=canon_tgt,
                            relationship=edge.relationship or "RELATED_TO",
                            evidence=edge.evidence or "",
                            source_page=page_num,
                            confidence=edge.confidence,
                            status=edge.status or "ASSERTED",
                            properties=dict(edge.properties),
                        )
                        try:
                            # Preserves NetworkX MultiDiGraph parallel edges
                            self.graph.create_edge(canon_edge)
                        except Exception as exc:
                            logger.debug("manager.edge_creation_skipped", error=str(exc))
                else:
                    logger.debug(
                        "manager.edge_endpoint_unresolved_dropped",
                        page=page_num,
                        raw_src=raw_src,
                        raw_tgt=raw_tgt,
                        resolved_src=canon_src,
                        resolved_tgt=canon_tgt,
                    )

            # ---------------------------------------------------------
            # Step 3: Collect Unresolved References for Post-Merge
            # ---------------------------------------------------------
            self._unresolved.extend(delta.unresolved_references)

    async def reconcile_cross_references(self) -> None:
        """
        Reconcile cross-page anaphoric references (e.g. "the above patient")
        after ALL page workers have completed and deltas are merged.
        Ranks candidates deterministically using target type, page proximity,
        confidence, and category. Ambiguous cases are left unresolved.
        """
        async with self._lock:
            if not self._unresolved:
                return

            for ref in self._unresolved:
                phrase = str(ref.phrase or "").strip()
                target_type = str(ref.target_type or "").strip()
                page_num = ref.page_number

                if not phrase and not target_type:
                    continue

                norm_tgt_type = self._normalize(target_type)
                candidates: List[Tuple[float, GraphNode]] = []

                for node in self.graph._nodes.values():
                    node_type_norm = self._normalize(node.type)
                    node_label_norm = self._normalize(node.label)

                    type_match = False
                    if norm_tgt_type:
                        if norm_tgt_type in (node_type_norm, node_label_norm):
                            type_match = True
                        elif norm_tgt_type in ("patient", "person") and node_type_norm in ("patient", "person"):
                            type_match = True
                        elif norm_tgt_type in ("doctor", "surgeon", "physician") and node_type_norm in ("doctor", "surgeon", "physician"):
                            type_match = True
                        elif norm_tgt_type in ("hospital", "clinic", "organization") and node_type_norm in ("hospital", "clinic", "organization"):
                            type_match = True
                    else:
                        type_match = True

                    if not type_match:
                        continue

                    # Score candidate
                    score = 1.0

                    # Proximity scoring: prefer entities established on or before this page
                    pages_before_or_at = [p for p in node.source_pages if p <= page_num]
                    if pages_before_or_at:
                        closest_page = max(pages_before_or_at)
                        page_distance = page_num - closest_page
                        score += 3.0 / (1.0 + page_distance)
                    else:
                        # Entity only seen on later page: penalize
                        score -= 2.0

                    # Category priority
                    if node.category == "EXPLICIT":
                        score += 2.0
                    elif node.category == "CONTEXTUAL":
                        score += 1.0

                    # Confidence
                    score += node.confidence

                    candidates.append((score, node))

                if not candidates:
                    logger.debug("manager.reconcile.no_candidate", phrase=phrase, target_type=target_type)
                    continue

                candidates.sort(key=lambda x: x[0], reverse=True)
                top_score, best_candidate = candidates[0]

                # Check for ambiguity: if second candidate exists with identical score
                if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.05:
                    logger.info(
                        "manager.reconcile.ambiguous_skipped",
                        phrase=phrase,
                        target_type=target_type,
                        cand1=best_candidate.id,
                        cand2=candidates[1][1].id,
                    )
                    continue  # Ambiguous, leave unresolved

                if top_score < 1.5:
                    continue  # Low confidence, leave unresolved

                # Resolve reference: connect source node to canonical target
                canon_src = (
                    self._id_remap.get((page_num, ref.source_node_id))
                    or self._id_remap.get((page_num, self._normalize(ref.source_node_id)))
                    or ref.source_node_id
                )

                if canon_src and canon_src in self.graph._nodes and canon_src != best_candidate.id:
                    ref_edge = GraphEdge(
                        id=f"edge_ref_{page_num}_{uuid.uuid4().hex[:6]}",
                        source_node=canon_src,
                        target_node=best_candidate.id,
                        relationship="REFERS_TO",
                        evidence=ref.rationale or phrase,
                        source_page=page_num,
                        confidence=round(min(1.0, top_score / 5.0), 3),
                        status="INFERRED",
                    )
                    self.graph.create_edge(ref_edge)

                # Register alias on best candidate
                if phrase:
                    self.graph.add_alias(best_candidate.id, phrase)

                logger.info(
                    "manager.reconcile.resolved",
                    phrase=phrase,
                    target_type=target_type,
                    resolved_node=best_candidate.id,
                    target_value=best_candidate.value,
                    score=top_score,
                )

    @property
    def unresolved_count(self) -> int:
        return len(self._unresolved)
