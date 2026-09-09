"""
File: graph_memory.py
Purpose: In-memory dynamic graph memory engine using NetworkX.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

from src.ai.layer3_extraction.graph_agent.models import Evidence, GraphEdge, GraphNode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphMemory:
    """
    In-memory graph store holding document entities, facts, and relationships.
    Survives for the entire multi-page document processing session.
    Uses NetworkX MultiDiGraph to preserve multiple distinct relationships
    between the same source and target nodes without overwriting.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        # Inverted index for quick entity reuse and lookup
        self._type_index: Dict[str, Set[str]] = {}
        self._alias_index: Dict[str, Set[str]] = {}  # normalized value -> set of node_ids

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @property
    def nodes(self) -> Dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> Dict[str, GraphEdge]:
        return self._edges

    def lookup_alias(self, alias: str) -> Optional[GraphNode]:
        """Resolve an alias or reference mention to an existing GraphNode."""
        alias_norm = self._normalize(alias)
        if alias_norm in self._alias_index:
            node_ids = self._alias_index[alias_norm]
            if node_ids:
                return self.get_node(next(iter(node_ids)))

        # Fallback check across node aliases in case alias was added directly to GraphNode
        for node in self._nodes.values():
            for a in getattr(node, "aliases", []):
                if self._normalize(a) == alias_norm:
                    if alias_norm not in self._alias_index:
                        self._alias_index[alias_norm] = set()
                    self._alias_index[alias_norm].add(node.id)
                    return node
        return None

    def add_alias(self, node_id: str, alias: str) -> None:
        """Register an alias for a node and update index."""
        node = self.get_node(node_id)
        if node:
            node.add_alias(alias)
            alias_key = self._normalize(alias)
            if alias_key:
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(node_id)

    def create_node(
        self,
        node_or_id: Union[GraphNode, str, None] = None,
        node_type: Optional[str] = None,
        label: Optional[str] = None,
        value: Optional[str] = None,
        *,
        node_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        evidence: Optional[Union[Evidence, List[Evidence]]] = None,
        source_page: Optional[int] = None,
        source_pages: Optional[List[int]] = None,
        confidence: float = 1.0,
        status: str = "EXTRACTED",
        category: str = "CONTEXTUAL",
        **kwargs: Any,
    ) -> GraphNode:
        """Add a new node to the graph and update inverted indexes."""
        if isinstance(node_or_id, GraphNode):
            node = node_or_id
        else:
            actual_id = node_id or (node_or_id if isinstance(node_or_id, str) else "")
            if not actual_id:
                raise ValueError("Node must have a non-empty id.")
            pages = list(source_pages or [])
            if source_page is not None and source_page not in pages:
                pages.append(source_page)
            ev_list: List[Evidence] = []
            if evidence:
                if isinstance(evidence, list):
                    ev_list.extend(evidence)
                else:
                    ev_list.append(evidence)
            node = GraphNode(
                id=actual_id,
                type=node_type or kwargs.get("type", "Entity"),
                label=label or "",
                value=value or "",
                properties=properties or {},
                aliases=aliases or [],
                evidence=ev_list,
                source_pages=pages or [1],
                confidence=confidence,
                status=status,
                category=category,
            )

        if not node.id:
            raise ValueError("Node must have a non-empty id.")
        if node.id in self._nodes:
            logger.debug("graph.node_already_exists_updating", node_id=node.id)
            return self.update_node(node.id, node.to_dict())

        self._nodes[node.id] = node
        self.graph.add_node(node.id, data=node)

        # Update type index
        type_key = self._normalize(node.type)
        if type_key not in self._type_index:
            self._type_index[type_key] = set()
        self._type_index[type_key].add(node.id)

        # Update alias/value index for primary value
        val_key = self._normalize(node.value)
        if val_key:
            if val_key not in self._alias_index:
                self._alias_index[val_key] = set()
            self._alias_index[val_key].add(node.id)

        # Update alias/value index for known aliases
        for alias in getattr(node, "aliases", []):
            alias_key = self._normalize(alias)
            if alias_key:
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(node.id)

        logger.debug(
            "graph.node.created",
            node_id=node.id,
            node_type=node.type,
            value=node.value,
            category=node.category,
        )
        return node

    def update_node(
        self,
        node_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GraphNode:
        """Update properties, evidence, or pages on an existing node."""
        node = self.get_node(node_id)
        if not node:
            raise KeyError(f"Node {node_id} does not exist in graph memory.")

        upd = dict(updates or {})
        upd.update(kwargs)

        if "value" in upd and upd["value"]:
            old_val_key = self._normalize(node.value)
            new_val_key = self._normalize(upd["value"])
            if old_val_key in self._alias_index and node_id in self._alias_index[old_val_key]:
                self._alias_index[old_val_key].remove(node_id)
            node.value = str(upd["value"])
            if new_val_key not in self._alias_index:
                self._alias_index[new_val_key] = set()
            self._alias_index[new_val_key].add(node_id)

        if "label" in upd and upd["label"]:
            node.label = str(upd["label"])

        if "confidence" in upd and upd["confidence"] is not None:
            node.confidence = max(node.confidence, float(upd["confidence"]))

        if "status" in upd and upd["status"]:
            node.status = str(upd["status"])

        if "source_pages" in upd:
            for p in upd["source_pages"]:
                node.add_page(int(p))
        if "source_page" in upd and upd["source_page"] is not None:
            node.add_page(int(upd["source_page"]))

        if "evidence" in upd:
            for ev in upd["evidence"]:
                if isinstance(ev, dict):
                    node.add_evidence(
                        page_number=int(ev.get("page_number", 1)),
                        text=str(ev.get("text", "")),
                        confidence=float(ev.get("confidence", 1.0)),
                    )
                else:
                    node.add_evidence(page_number=ev.page_number, text=ev.text, confidence=ev.confidence)

        if "properties" in upd and isinstance(upd["properties"], dict):
            node.properties.update(upd["properties"])

        if "aliases" in upd and upd["aliases"]:
            for a in upd["aliases"]:
                node.add_alias(a)
                a_key = self._normalize(a)
                if a_key:
                    if a_key not in self._alias_index:
                        self._alias_index[a_key] = set()
                    self._alias_index[a_key].add(node_id)

        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve node by ID."""
        return self._nodes.get(node_id)

    def create_edge(
        self,
        edge_or_id: Union[GraphEdge, str, None] = None,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relationship: Optional[str] = None,
        *,
        edge_id: Optional[str] = None,
        evidence: Optional[str] = None,
        source_page: int = 1,
        confidence: float = 1.0,
        status: str = "EXTRACTED",
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GraphEdge:
        """Add a directed relationship between two nodes in the graph."""
        if isinstance(edge_or_id, GraphEdge):
            edge = edge_or_id
        else:
            actual_id = edge_id or (edge_or_id if isinstance(edge_or_id, str) else "")
            src = source_node or kwargs.get("source", "")
            tgt = target_node or kwargs.get("target", "")
            rel = relationship or kwargs.get("rel", "RELATED_TO")
            edge = GraphEdge(
                id=actual_id,
                source_node=src,
                target_node=tgt,
                relationship=rel,
                evidence=evidence or kwargs.get("text", ""),
                source_page=source_page,
                confidence=confidence,
                status=status,
                properties=properties or {},
            )

        if edge.source_node not in self._nodes:
            raise KeyError(f"Source node {edge.source_node} does not exist in graph memory.")
        if edge.target_node not in self._nodes:
            raise KeyError(f"Target node {edge.target_node} does not exist in graph memory.")

        if not edge.id:
            edge.id = f"{edge.source_node}__{edge.relationship.lower()}__{edge.target_node}"

        self._edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_node,
            edge.target_node,
            key=edge.id,
            id=edge.id,
            relationship=edge.relationship,
            data=edge,
        )

        logger.debug(
            "graph.edge.created",
            edge_id=edge.id,
            source=edge.source_node,
            rel=edge.relationship,
            target=edge.target_node,
            status=edge.status,
        )
        return edge

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    def get_edges(
        self,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> List[GraphEdge]:
        """Filter edges by source, target, and/or relationship type."""
        results = []
        rel_norm = self._normalize(relationship) if relationship else None
        for edge in self._edges.values():
            if source_node and edge.source_node != source_node:
                continue
            if target_node and edge.target_node != target_node:
                continue
            if rel_norm and self._normalize(edge.relationship) != rel_norm:
                continue
            results.append(edge)
        return results

    def read_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[GraphEdge, GraphNode]]:
        """
        Return incoming, outgoing, or both neighbor edges and adjacent nodes.
        direction: 'out', 'in', or 'both'
        """
        if node_id not in self._nodes:
            return []

        neighbors = []
        if direction in ("out", "both"):
            for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                edge: Optional[GraphEdge] = edge_data.get("data")
                target_node = self._nodes.get(target_id)
                if edge and target_node:
                    neighbors.append((edge, target_node))

        if direction in ("in", "both"):
            for source_id, _, edge_data in self.graph.in_edges(node_id, data=True):
                edge: Optional[GraphEdge] = edge_data.get("data")
                source_node = self._nodes.get(source_id)
                if edge and source_node:
                    neighbors.append((edge, source_node))

        return neighbors

    def search_nodes(
        self,
        query: Optional[str] = None,
        node_type: Optional[str] = None,
        label: Optional[str] = None,
        page: Optional[int] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """
        Search for entities by query (fuzzy substring in value/label), node_type, label, or page.
        """
        candidates: List[Tuple[float, GraphNode]] = []
        query_norm = self._normalize(query) if query else None
        type_norm = self._normalize(node_type) if node_type else None
        label_norm = self._normalize(label) if label else None

        for node in self._nodes.values():
            if page is not None and page not in node.source_pages:
                continue

            if type_norm and self._normalize(node.type) != type_norm:
                continue

            if label_norm and self._normalize(node.label) != label_norm:
                continue

            score = 0.0
            if query_norm:
                val_norm = self._normalize(node.value)
                lbl_norm = self._normalize(node.label)
                if query_norm == val_norm:
                    score = 1.0
                elif query_norm in val_norm:
                    score = 0.8
                elif val_norm in query_norm and len(val_norm) > 2:
                    score = 0.7
                elif query_norm in lbl_norm or lbl_norm in query_norm:
                    score = 0.5
                else:
                    continue
            else:
                score = 0.5

            candidates.append((score, node))

        candidates.sort(key=lambda x: (x[0], x[1].confidence, max(x[1].source_pages or [0])), reverse=True)
        return [node for _, node in candidates[:limit]]

    def find_reuse_candidate(
        self,
        node_type: str,
        value: str,
        label: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ) -> Optional[GraphNode]:
        """
        High-confidence check for entity reuse to avoid duplicate nodes,
        without aggressive or false merging.
        """
        val_norm = self._normalize(value)
        if not val_norm:
            return None

        # 1. Exact normalized value match in same or compatible type
        if val_norm in self._alias_index:
            for cand_id in self._alias_index[val_norm]:
                cand = self.get_node(cand_id)
                if cand and (self._normalize(cand.type) == self._normalize(node_type) or not node_type):
                    return cand

        # 2. Person name partial/honorific matching (e.g. "Dr. Arun Sharma" vs "Arun Sharma")
        if self._normalize(node_type) in ("person", "doctor", "surgeon", "patient"):
            clean_name = re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?)\s+", "", val_norm)
            for node in self._nodes.values():
                if self._normalize(node.type) in ("person", "doctor", "surgeon", "patient"):
                    cand_clean = re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?)\s+", "", self._normalize(node.value))
                    if clean_name == cand_clean and len(clean_name) > 3:
                        return node

        return None

    def get_bounded_summary(self, schema_fields: Optional[List[str]] = None, max_nodes: int = 40) -> str:
        """
        Generate a compact, readable summary of existing graph memory for context
        in agent prompts without overwhelming token limits.
        """
        if not self._nodes:
            return "No entities in graph memory yet."

        lines = ["Existing Graph Entities:"]
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (1 if n.category == "EXPLICIT" else 0, max(n.source_pages or [0])),
            reverse=True,
        )

        for n in sorted_nodes[:max_nodes]:
            pages_str = f"P{','.join(map(str, n.source_pages))}" if n.source_pages else "P?"
            lines.append(f"- [{n.id}] {n.type} ({n.label}): \"{n.value}\" [{pages_str}]")

        if self._edges:
            lines.append("\nKey Relationships:")
            edge_count = 0
            for edge in list(self._edges.values())[-25:]:  # show most recent 25 edges
                src = self.get_node(edge.source_node)
                tgt = self.get_node(edge.target_node)
                if src and tgt:
                    src_val = src.value or src.id
                    tgt_val = tgt.value or tgt.id
                    lines.append(f"- ({src_val}) -[{edge.relationship}]-> ({tgt_val}) [P{edge.source_page}]")
                    edge_count += 1

        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of graph memory."""
        type_counts: Dict[str, int] = {}
        for n in self._nodes.values():
            type_counts[n.type] = type_counts.get(n.type, 0) + 1

        rel_counts: Dict[str, int] = {}
        for e in self._edges.values():
            rel_counts[e.relationship] = rel_counts.get(e.relationship, 0) + 1

        all_pages = set()
        for n in self._nodes.values():
            all_pages.update(n.source_pages)

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "pages_covered": sorted(list(all_pages)),
            "node_types": type_counts,
            "relationships": rel_counts,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Full serializable snapshot of the graph."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "stats": self.stats(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GraphMemory into a lossless serializable dictionary."""
        return self.snapshot()

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore graph state from snapshot dictionary."""
        restored = self.from_dict(snapshot)
        self.graph = restored.graph
        self._nodes = restored._nodes
        self._edges = restored._edges
        self._type_index = restored._type_index
        self._alias_index = restored._alias_index

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphMemory:
        """Losslessly reconstruct a GraphMemory instance from a serialized dictionary."""
        gm = cls()
        for ndata in data.get("nodes", []):
            node = GraphNode.from_dict(ndata)
            gm.create_node(node)
        for edata in data.get("edges", []):
            edge = GraphEdge.from_dict(edata)
            try:
                gm.create_edge(edge)
            except Exception as exc:
                logger.debug("graph.from_dict.edge_failed", error=str(exc), edge_id=edata.get("id"))
        return gm

