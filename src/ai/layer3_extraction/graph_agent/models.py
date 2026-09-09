"""
File: models.py
Purpose: Data structures for Graph-Based Contextual Extraction in Layer 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    page_number: int
    text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        return cls(
            page_number=int(data.get("page_number", 1)),
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 1.0)),
        )


NodeEvidence = Evidence


@dataclass
class GraphNode:
    id: str
    type: str  # Generic type: Person, Organization, Event, Date, Amount, Identifier, Admission, etc.
    label: str
    value: str
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN
    category: str = "CONTEXTUAL"  # EXPLICIT, CONTEXTUAL, REFERENCE_SUPPORT

    def add_page(self, page_number: int) -> None:
        if page_number not in self.source_pages:
            self.source_pages.append(page_number)
            self.source_pages.sort()

    def add_evidence(self, page_number: int, text: str, confidence: float = 1.0) -> None:
        self.add_page(page_number)
        if text and not any(e.text == text and e.page_number == page_number for e in self.evidence):
            self.evidence.append(Evidence(page_number=page_number, text=text, confidence=confidence))

    def add_alias(self, alias: str) -> None:
        clean = alias.strip()
        if clean and clean not in self.aliases:
            self.aliases.append(clean)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "value": self.value,
            "properties": self.properties,
            "aliases": list(self.aliases),
            "evidence": [e.to_dict() for e in self.evidence],
            "source_pages": self.source_pages,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphNode:
        ev_list = [Evidence.from_dict(e) if isinstance(e, dict) else e for e in data.get("evidence", [])]
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "Entity")),
            label=str(data.get("label", data.get("type", "Entity"))),
            value=str(data.get("value", "")),
            properties=data.get("properties", {}),
            aliases=list(data.get("aliases", [])),
            evidence=ev_list,
            source_pages=list(data.get("source_pages", [])),
            confidence=float(data.get("confidence", 1.0)),
            status=str(data.get("status", "ASSERTED")),
            category=str(data.get("category", "CONTEXTUAL")),
        )


@dataclass
class GraphEdge:
    id: str
    source_node: str
    target_node: str
    relationship: str  # Generic: HAS, BELONGS_TO, RELATED_TO, PERFORMED_BY, UNDERWENT, FOR, etc.
    evidence: str = ""
    source_page: int = 1
    confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relationship": self.relationship,
            "evidence": self.evidence,
            "source_page": self.source_page,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphEdge:
        return cls(
            id=str(data.get("id", "")),
            source_node=str(data["source_node"]),
            target_node=str(data["target_node"]),
            relationship=str(data["relationship"]),
            evidence=str(data.get("evidence", "")),
            source_page=int(data.get("source_page", 1)),
            confidence=float(data.get("confidence", 1.0)),
            status=str(data.get("status", "ASSERTED")),
            properties=data.get("properties", {}),
        )


@dataclass
class PageProcessingResult:
    page_number: int
    nodes_created: int = 0
    nodes_reused: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    references_resolved: int = 0
    ambiguities_encountered: int = 0
    raw_llm_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "nodes_created": self.nodes_created,
            "nodes_reused": self.nodes_reused,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "references_resolved": self.references_resolved,
            "ambiguities_encountered": self.ambiguities_encountered,
        }
