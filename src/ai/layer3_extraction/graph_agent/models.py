"""
File: models.py
Purpose: Data structures for Graph-Based Contextual Extraction in Layer 3.
Hardened with explicit uncertainty models, candidate evaluation structures,
merge audit records, page validation, and serialization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ResolutionStatus(str, Enum):
    CONFIRMED_MATCH = "CONFIRMED_MATCH"
    UNCERTAIN_MATCH = "UNCERTAIN_MATCH"
    NEW_ENTITY = "NEW_ENTITY"
    REJECTED_CONTRADICTION = "REJECTED_CONTRADICTION"


class TypeCompatibility(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class PageProcessingState(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DELTA_CREATED = "DELTA_CREATED"
    DELTA_PERSISTED = "DELTA_PERSISTED"
    MERGED = "MERGED"
    RECONCILED = "RECONCILED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN, EXTRACTED
    category: str = "CONTEXTUAL"  # EXPLICIT, CONTEXTUAL, REFERENCE_SUPPORT
    # Hardened uncertainty dimensions:
    extraction_confidence: float = 1.0  # text extraction certainty from source document
    resolution_confidence: Optional[float] = None  # canonical identity certainty (None for NEW_ENTITY or unestablished resolution)
    evidence_strength: float = 1.0      # supporting textual evidence strength

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
            "extraction_confidence": round(self.extraction_confidence, 3),
            "resolution_confidence": round(self.resolution_confidence, 3) if self.resolution_confidence is not None else None,
            "evidence_strength": round(self.evidence_strength, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphNode:
        ev_list = [Evidence.from_dict(e) if isinstance(e, dict) else e for e in data.get("evidence", [])]
        conf = float(data.get("confidence", 1.0))
        res_conf = data.get("resolution_confidence")
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "Entity")),
            label=str(data.get("label", data.get("type", "Entity"))),
            value=str(data.get("value", "")),
            properties=data.get("properties", {}),
            aliases=list(data.get("aliases", [])),
            evidence=ev_list,
            source_pages=list(data.get("source_pages", [])),
            confidence=conf,
            status=str(data.get("status", "ASSERTED")),
            category=str(data.get("category", "CONTEXTUAL")),
            extraction_confidence=float(data.get("extraction_confidence", conf)),
            resolution_confidence=float(res_conf) if res_conf is not None else None,
            evidence_strength=float(data.get("evidence_strength", 1.0)),
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
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN, EXTRACTED
    properties: Dict[str, Any] = field(default_factory=dict)
    # Hardened uncertainty dimensions:
    extraction_confidence: float = 1.0    # certainty of mention extraction
    relationship_confidence: float = 1.0  # support for the semantic link between nodes
    evidence_strength: float = 1.0        # strength of textual evidence for this relation

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
            "extraction_confidence": round(self.extraction_confidence, 3),
            "relationship_confidence": round(self.relationship_confidence, 3),
            "evidence_strength": round(self.evidence_strength, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphEdge:
        conf = float(data.get("confidence", 1.0))
        return cls(
            id=str(data.get("id", "")),
            source_node=str(data["source_node"]),
            target_node=str(data["target_node"]),
            relationship=str(data["relationship"]),
            evidence=str(data.get("evidence", "")),
            source_page=int(data.get("source_page", 1)),
            confidence=conf,
            status=str(data.get("status", "ASSERTED")),
            properties=data.get("properties", {}),
            extraction_confidence=float(data.get("extraction_confidence", conf)),
            relationship_confidence=float(data.get("relationship_confidence", conf)),
            evidence_strength=float(data.get("evidence_strength", 1.0)),
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnresolvedRef:
        return cls(
            source_node_id=str(data.get("source_node_id", "")),
            phrase=str(data.get("phrase", "")),
            target_type=str(data.get("target_type", "")),
            page_number=int(data.get("page_number", 1)),
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class PageDelta:
    """
    Immutable extraction delta emitted by an independent concurrent page worker.
    Contains local entities, relationships, unresolved references, and local ID maps.
    Represents observations and proposals; MUST NOT contain canonical mutation commands.
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageDelta:
        entities = [GraphNode.from_dict(n) if isinstance(n, dict) else n for n in data.get("entities", [])]
        relationships = [GraphEdge.from_dict(e) if isinstance(e, dict) else e for e in data.get("relationships", [])]
        unresolved = [
            UnresolvedRef.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("unresolved_references", [])
        ]
        return cls(
            page_number=int(data.get("page_number", 1)),
            entities=entities,
            relationships=relationships,
            unresolved_references=unresolved,
            raw_id_map=dict(data.get("raw_id_map", {})),
        )


@dataclass
class CandidateEvaluation:
    """Evaluation of a canonical node candidate for an incoming entity observation."""
    candidate_id: str
    candidate_node: GraphNode
    candidate_score: float  # Explicitly a heuristic ranking score [0.0 - 1.0], not a calibrated probability
    matching_signals: List[str] = field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_reasons: List[str] = field(default_factory=list)
    compatibility: str = TypeCompatibility.UNKNOWN.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_score": round(self.candidate_score, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_detected": self.contradiction_detected,
            "contradiction_reasons": list(self.contradiction_reasons),
            "compatibility": self.compatibility,
        }


@dataclass
class ResolutionDecision:
    """Explicit decision made by GraphMemoryManager for an entity observation."""
    status: ResolutionStatus
    canonical_id: Optional[str] = None
    winning_candidate: Optional[CandidateEvaluation] = None
    all_candidates: List[CandidateEvaluation] = field(default_factory=list)
    candidate_score: float = 0.0
    second_candidate_score: float = 0.0
    decision_margin: float = 0.0
    matching_signals: List[str] = field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_reasons: List[str] = field(default_factory=list)
    resolution_method: str = "NONE"
    rationale: str = ""
    algorithm_version: str = "entity_resolution_v2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, ResolutionStatus) else str(self.status),
            "canonical_id": self.canonical_id,
            "candidate_score": round(self.candidate_score, 4),
            "second_candidate_score": round(self.second_candidate_score, 4),
            "decision_margin": round(self.decision_margin, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_detected": self.contradiction_detected,
            "contradiction_reasons": list(self.contradiction_reasons),
            "resolution_method": self.resolution_method,
            "rationale": self.rationale,
            "algorithm_version": self.algorithm_version,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
        }


@dataclass
class MergeAuditRecord:
    """Durable audit record explaining why two observations were merged canonically."""
    source_node_id: str
    source_page: int
    source_value: str
    canonical_node_id: str
    canonical_value: str
    resolution_method: str
    candidate_score: float
    second_candidate_score: float = 0.0
    decision_margin: float = 0.0
    matching_signals: List[str] = field(default_factory=list)
    contradiction_result: str = "PASSED_HARD_GATES"
    contradiction_reasons: List[str] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    algorithm_version: str = "entity_resolution_v2"
    pipeline_version: str = "v1"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "source_page": self.source_page,
            "source_value": self.source_value,
            "canonical_node_id": self.canonical_node_id,
            "canonical_value": self.canonical_value,
            "resolution_method": self.resolution_method,
            "candidate_score": round(self.candidate_score, 4),
            "second_candidate_score": round(self.second_candidate_score, 4),
            "decision_margin": round(self.decision_margin, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_result": self.contradiction_result,
            "contradiction_reasons": list(self.contradiction_reasons),
            "source_pages": list(self.source_pages),
            "evidence": list(self.evidence),
            "algorithm_version": self.algorithm_version,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.timestamp,
        }


@dataclass
class PageDeltaValidationResult:
    """Result of pre-ingestion PageDelta structural and semantic validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_page_delta(delta: Any) -> PageDeltaValidationResult:
    """
    Validate a PageDelta before durable checkpointing and canonical memory ingestion.
    Enforces:
    1. Valid delta instance and page number >= 1
    2. Local entity IDs unique within the delta
    3. Entities have non-empty value and type
    4. Relationships have non-empty endpoints and relationship types
    5. Confidences within [0.0, 1.0]
    6. Valid status values
    7. No destructive canonical mutation commands
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(delta, PageDelta):
        return PageDeltaValidationResult(
            is_valid=False,
            errors=[f"Expected PageDelta instance, got {type(delta).__name__}"],
        )

    if delta.page_number is None or delta.page_number < 1:
        errors.append(f"Invalid page_number: {delta.page_number}")

    seen_node_ids: Set[str] = set()
    for idx, ent in enumerate(delta.entities):
        if not isinstance(ent, GraphNode):
            errors.append(f"Entity at index {idx} is not a GraphNode instance")
            continue
        if not ent.id or not ent.id.strip():
            errors.append(f"Entity at index {idx} has an empty or null id")
        elif ent.id in seen_node_ids:
            errors.append(f"Duplicate entity id '{ent.id}' in PageDelta for page {delta.page_number}")
        else:
            seen_node_ids.add(ent.id)

        if not ent.value or not ent.value.strip():
            errors.append(f"Entity '{ent.id}' has empty value")
        if not ent.type or not ent.type.strip():
            errors.append(f"Entity '{ent.id}' has empty type")

        if ent.confidence < 0.0 or ent.confidence > 1.0:
            errors.append(f"Entity '{ent.id}' confidence {ent.confidence} out of range [0.0, 1.0]")
        if ent.extraction_confidence < 0.0 or ent.extraction_confidence > 1.0:
            errors.append(f"Entity '{ent.id}' extraction_confidence {ent.extraction_confidence} out of range [0.0, 1.0]")
        if ent.resolution_confidence is not None and (ent.resolution_confidence < 0.0 or ent.resolution_confidence > 1.0):
            errors.append(f"Entity '{ent.id}' resolution_confidence {ent.resolution_confidence} out of range [0.0, 1.0]")

        if ent.status not in ("ASSERTED", "INFERRED", "UNCERTAIN", "EXTRACTED"):
            warnings.append(f"Entity '{ent.id}' has non-standard status: '{ent.status}'")

        # Guard: PageDelta must not carry destructive commands
        props = getattr(ent, "properties", {}) or {}
        for destructive_key in ("delete_node", "merge_override", "drop_graph", "truncate"):
            if destructive_key in props:
                errors.append(f"Destructive mutation command '{destructive_key}' rejected in PageDelta")

    for idx, edge in enumerate(delta.relationships):
        if not isinstance(edge, GraphEdge):
            errors.append(f"Relationship at index {idx} is not a GraphEdge instance")
            continue
        if not edge.source_node or not edge.source_node.strip():
            errors.append(f"Edge at index {idx} missing source_node")
        if not edge.target_node or not edge.target_node.strip():
            errors.append(f"Edge at index {idx} missing target_node")
        if not edge.relationship or not edge.relationship.strip():
            errors.append(f"Edge at index {idx} missing relationship type")

        if edge.confidence < 0.0 or edge.confidence > 1.0:
            errors.append(f"Edge at index {idx} confidence {edge.confidence} out of range [0.0, 1.0]")
        if edge.relationship_confidence < 0.0 or edge.relationship_confidence > 1.0:
            errors.append(f"Edge at index {idx} relationship_confidence {edge.relationship_confidence} out of range [0.0, 1.0]")

    return PageDeltaValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )
