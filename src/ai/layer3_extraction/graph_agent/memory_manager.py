"""
File: memory_manager.py
Purpose: Central Blackboard / Memory Manager for concurrent Layer 3 graph extraction.
Coordinates concurrent page workers, merges PageDeltas canonically,
remaps worker-local IDs to canonical IDs, rewrites edge endpoints,
and reconciles cross-page references post-merge.

Hardened with:
- False-merge protection (candidate generation, hard contradiction gates, heuristic scoring, ambiguity margins)
- Observation preservation and merge audit trails
- Explicit uncertainty states (ASSERTED, INFERRED, UNCERTAIN)
- Deterministic idempotency and checksum conflict detection
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import (
    CandidateEvaluation,
    Evidence,
    GraphEdge,
    GraphNode,
    MergeAuditRecord,
    PageDelta,
    PageDeltaValidationResult,
    PageProcessingState,
    ResolutionDecision,
    ResolutionStatus,
    TypeCompatibility,
    UnresolvedRef,
    validate_page_delta,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

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

# Type compatibility clusters
CLUSTER_PERSON: Set[str] = {
    "person", "doctor", "physician", "surgeon", "specialist", "clinician",
    "patient", "subject", "admittee", "employee", "staff", "worker",
    "customer", "client", "buyer", "purchaser", "plaintiff", "claimant",
    "defendant", "respondent", "user", "individual", "witness",
}

CLUSTER_ORGANIZATION: Set[str] = {
    "hospital", "clinic", "facility", "provider", "institution",
    "organization", "company", "corporation", "firm", "employer",
    "vendor", "merchant", "seller", "supplier", "court", "tribunal",
    "bank", "agency", "department",
}

CLUSTER_VALUE_CONCEPT: Set[str] = {
    "date", "time", "amount", "currency", "price", "fee", "cost",
    "identifier", "uhid", "ssn", "pan", "account_no", "policy_no",
    "admission", "record", "note", "diagnosis", "procedure",
}


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
            m_name = re.match(r"^([a-zA-Z0-9]+)_(?:name|id|number|num|code)$", clean_name, re.I)
            if m_name:
                stem = m_name.group(1).capitalize()
                if len(stem) > 2 and stem not in seen:
                    derived.append(stem)
                    seen.add(stem)

    if not derived:
        for fname in list(schema.model_fields.keys())[:3]:
            cand = re.sub(r"[^a-zA-Z0-9]+", " ", fname).strip().title().replace(" ", "")
            if cand and cand not in seen:
                derived.append(cand)
                seen.add(cand)

    return derived


class GraphMemoryManager:
    """
    Central Blackboard / Memory Manager for Layer 3 extraction.

    Guarantees:
    - Candidate -> Verify -> Commit resolution pipeline.
    - LLM proposals (reused_node_id) never silently force canonical merges.
    - Hard contradiction gates block false merges even with high textual similarity.
    - Scores are explicit heuristic ranking scores, not calibrated probabilities.
    - Ambiguous candidates remain unresolved (status=UNCERTAIN).
    - Committed merges retain complete audit provenance and source observations.
    - Idempotent ingestion per document/page.
    """

    def __init__(
        self,
        anchor_types: Optional[List[str]] = None,
        anchor_max_k: int = 5,
        schema: Optional[type[BaseModel]] = None,
        match_threshold: float = 0.85,
        ambiguity_margin: float = 0.15,
        pipeline_version: str = "v1",
        algorithm_version: str = "entity_resolution_v2",
    ) -> None:
        self.graph = GraphMemory()
        self.anchor_max_k = max(1, anchor_max_k)
        self._schema = schema
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.pipeline_version = pipeline_version
        self.algorithm_version = algorithm_version

        if anchor_types:
            self.anchor_types = [a.strip() for a in anchor_types if a.strip()]
        elif schema is not None:
            self.anchor_types = derive_anchor_types_from_schema(schema)
        else:
            self.anchor_types = []

        self._unresolved: List[UnresolvedRef] = []
        self._lock = asyncio.Lock()
        # Scoped mapping: (page_number, local_id_or_key) -> canonical_id
        self._id_remap: Dict[Tuple[int, str], str] = {}
        # In-memory audit ledger of all canonical merges
        self._merge_ledger: List[MergeAuditRecord] = []
        # Idempotency tracking: page_number -> delta checksum
        self._ingested_deltas: Dict[int, str] = {}
        # Checksum conflict records: list of conflicts encountered
        self._checksum_conflicts: List[Dict[str, Any]] = []

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def clean_person_name(self, name: str) -> str:
        """Strip common honorifics and titles from person names."""
        norm = self._normalize(name)
        return re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?|md|esq\.?)\s+", "", norm).strip()

    @property
    def merge_ledger(self) -> List[MergeAuditRecord]:
        return list(self._merge_ledger)

    def get_merge_ledger(self) -> List[MergeAuditRecord]:
        return list(self._merge_ledger)

    @property
    def checksum_conflicts(self) -> List[Dict[str, Any]]:
        return list(self._checksum_conflicts)

    def check_type_compatibility(self, type_a: str, type_b: str) -> TypeCompatibility:
        """
        Generic entity type compatibility check.
        Distinguishes COMPATIBLE, INCOMPATIBLE, and UNKNOWN without brittle single-string inequality.
        """
        a = self._normalize(type_a)
        b = self._normalize(type_b)

        if not a or not b or a == b or a == "entity" or b == "entity":
            return TypeCompatibility.COMPATIBLE

        # Cross-cluster incompatibilities (e.g. Person vs Hospital, Date vs Amount)
        in_person_a = a in CLUSTER_PERSON
        in_person_b = b in CLUSTER_PERSON
        in_org_a = a in CLUSTER_ORGANIZATION
        in_org_b = b in CLUSTER_ORGANIZATION
        in_val_a = a in CLUSTER_VALUE_CONCEPT
        in_val_b = b in CLUSTER_VALUE_CONCEPT

        if in_person_a and (in_org_b or in_val_b):
            return TypeCompatibility.INCOMPATIBLE
        if in_org_a and (in_person_b or in_val_b):
            return TypeCompatibility.INCOMPATIBLE
        if in_person_b and (in_org_a or in_val_a):
            return TypeCompatibility.INCOMPATIBLE
        if in_org_b and (in_person_a or in_val_a):
            return TypeCompatibility.INCOMPATIBLE

        # Intra-person explicit opposing roles
        doctor_roles = {"doctor", "physician", "surgeon", "specialist", "clinician"}
        patient_roles = {"patient", "subject", "admittee"}
        if (a in doctor_roles and b in patient_roles) or (a in patient_roles and b in doctor_roles):
            return TypeCompatibility.INCOMPATIBLE

        legal_plaintiff = {"plaintiff", "claimant"}
        legal_defendant = {"defendant", "respondent"}
        if (a in legal_plaintiff and b in legal_defendant) or (a in legal_defendant and b in legal_plaintiff):
            return TypeCompatibility.INCOMPATIBLE

        # Within same cluster
        if (in_person_a and in_person_b) or (in_org_a and in_org_b) or (in_val_a and in_val_b):
            return TypeCompatibility.COMPATIBLE

        return TypeCompatibility.UNKNOWN

    def check_hard_contradictions(
        self,
        incoming: GraphNode,
        candidate: GraphNode,
    ) -> Tuple[bool, List[str]]:
        """
        Deterministic hard contradiction gates evaluated BEFORE any candidate merge.
        Returns (has_contradiction, reasons).
        """
        reasons: List[str] = []

        # 1. Type compatibility gate
        compat = self.check_type_compatibility(incoming.type, candidate.type)
        if compat == TypeCompatibility.INCOMPATIBLE:
            reasons.append(
                f"INCOMPATIBLE_TYPES: incoming '{incoming.type}' is incompatible with candidate '{candidate.type}'"
            )

        # 2. Explicit conflicting identifiers gate (UHID, SSN, PAN, Account No, etc.)
        id_keys = (
            "uhid", "ssn", "pan", "account_no", "account_number",
            "policy_no", "policy_number", "case_number", "id_number",
            "identifier", "tax_id", "mrn", "ipd_no", "ref_no", "npi",
        )
        for key in id_keys:
            val_in = str(incoming.properties.get(key, "")).strip().lower()
            val_cand = str(candidate.properties.get(key, "")).strip().lower()
            if val_in and val_cand and val_in != val_cand:
                reasons.append(
                    f"CONFLICTING_IDENTIFIERS: '{key}' differs ('{val_in}' != '{val_cand}')"
                )

        # Identifier node value conflict (when entity itself is an identifier)
        incoming_is_id = (
            self._normalize(incoming.type) in ("identifier", "uhid", "ssn", "pan", "account", "policy")
            or "id" in self._normalize(incoming.label)
        )
        candidate_is_id = (
            self._normalize(candidate.type) in ("identifier", "uhid", "ssn", "pan", "account", "policy")
            or "id" in self._normalize(candidate.label)
        )
        if incoming_is_id and candidate_is_id:
            if self._normalize(incoming.value) != self._normalize(candidate.value):
                reasons.append(
                    f"CONFLICTING_IDENTIFIER_VALUES: '{incoming.value}' != '{candidate.value}'"
                )

        # 3. Strong identity attributes (DOB, Gender)
        dob_in = (
            incoming.properties.get("dob")
            or incoming.properties.get("birth_date")
            or incoming.properties.get("date_of_birth")
        )
        dob_cand = (
            candidate.properties.get("dob")
            or candidate.properties.get("birth_date")
            or candidate.properties.get("date_of_birth")
        )
        if dob_in and dob_cand and str(dob_in).strip().lower() != str(dob_cand).strip().lower():
            reasons.append(f"CONFLICTING_DOB: '{dob_in}' != '{dob_cand}'")

        gender_in = incoming.properties.get("gender") or incoming.properties.get("sex")
        gender_cand = candidate.properties.get("gender") or candidate.properties.get("sex")
        if gender_in and gender_cand and str(gender_in).strip().lower() != str(gender_cand).strip().lower():
            reasons.append(f"CONFLICTING_GENDER: '{gender_in}' != '{gender_cand}'")

        # 4. Conflicting distinct person names (e.g. Arun Sharma vs Ramesh Sharma)
        val_in = self._normalize(incoming.value)
        val_cand = self._normalize(candidate.value)
        cand_aliases = [self._normalize(a) for a in candidate.aliases]
        in_aliases = [self._normalize(a) for a in incoming.aliases]

        # If incoming value is an explicit alias of candidate (or vice versa), it is not a contradiction
        is_alias_match = (val_in in cand_aliases) or (val_cand in in_aliases)

        incoming_person = self._normalize(incoming.type) in CLUSTER_PERSON
        candidate_person = self._normalize(candidate.type) in CLUSTER_PERSON
        if not is_alias_match and (incoming_person or candidate_person):
            clean_in = self.clean_person_name(incoming.value)
            clean_cand = self.clean_person_name(candidate.value)
            if clean_in and clean_cand:
                tokens_in = [t for t in clean_in.split() if len(t) > 0]
                tokens_cand = [t for t in clean_cand.split() if len(t) > 0]
                if len(tokens_in) >= 2 and len(tokens_cand) >= 2:
                    t_in_0 = tokens_in[0].rstrip(".")
                    t_cand_0 = tokens_cand[0].rstrip(".")
                    # An initial matching the first letter (e.g. "A." vs "Arun") is compatible
                    is_initial = (len(t_in_0) == 1 or len(t_cand_0) == 1) and (t_in_0[0] == t_cand_0[0])
                    if not is_initial and t_in_0 != t_cand_0:
                        reasons.append(
                            f"CONFLICTING_PERSON_NAMES: first name '{tokens_in[0]}' != '{tokens_cand[0]}' in '{clean_in}' vs '{clean_cand}'"
                        )

        return (len(reasons) > 0, reasons)

    def score_candidate(
        self,
        incoming: GraphNode,
        candidate: GraphNode,
        proposed_reused_id: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """
        Calculate a heuristic candidate ranking score [0.0 - 1.0].
        Explicitly a heuristic ranking score, NOT a calibrated probability.
        """
        val_in = self._normalize(incoming.value)
        val_cand = self._normalize(candidate.value)
        signals: List[str] = []
        score = 0.0

        # Exact normalized match
        if val_in == val_cand:
            score = 1.0
            signals.append("EXACT_VALUE")
        # Alias match
        elif (
            val_in in [self._normalize(a) for a in candidate.aliases]
            or any(self._normalize(a) == val_cand for a in incoming.aliases)
        ):
            score = 0.95
            signals.append("ALIAS_MATCH")
        # Clean person name match (honorifics stripped)
        elif self.clean_person_name(incoming.value) == self.clean_person_name(candidate.value) and len(self.clean_person_name(incoming.value)) > 3:
            score = 0.90
            signals.append("NORMALIZED_NAME_MATCH")
        # Substring / containment match
        elif (val_in in val_cand or val_cand in val_in) and min(len(val_in), len(val_cand)) > 3:
            score = 0.75
            signals.append("SUBSTRING_MATCH")
        else:
            # Token overlap / containment heuristic
            tok_in = set(val_in.split())
            tok_cand = set(val_cand.split())
            if tok_in and tok_cand:
                inter = tok_in & tok_cand
                if tok_in.issubset(tok_cand) or tok_cand.issubset(tok_in):
                    # e.g. "Robert Smith" in "Robert James Smith"
                    score = 0.85
                    signals.append("TOKEN_CONTAINMENT")
                elif inter:
                    union = tok_in | tok_cand
                    jaccard = len(inter) / len(union)
                    score = 0.50 + (0.35 * jaccard)
                    signals.append(f"TOKEN_OVERLAP_{round(jaccard, 2)}")

        # Signal boost: LLM suggested this node as candidate
        if proposed_reused_id and candidate.id == proposed_reused_id:
            score = min(1.0, score + 0.05)
            signals.append("LLM_PROPOSED_REUSE")

        # Signal boost: Exact type match
        if self._normalize(incoming.type) == self._normalize(candidate.type):
            score = min(1.0, score + 0.05)
            signals.append("EXACT_TYPE_MATCH")

        return (round(min(1.0, max(0.0, score)), 4), signals)

    def resolve_entity(
        self,
        incoming: GraphNode,
        page_num: int,
    ) -> ResolutionDecision:
        """
        Execute Candidate Generation -> Hard Contradiction Gates -> Scoring -> Margin Evaluation -> Decision.
        Does NOT mutate the canonical graph itself.
        """
        raw_val = (incoming.value or "").strip()
        if not raw_val:
            return ResolutionDecision(
                status=ResolutionStatus.NEW_ENTITY,
                rationale="Empty value, cannot resolve",
                algorithm_version=self.algorithm_version,
            )

        val_norm = self._normalize(raw_val)
        proposed_reused_id = incoming.properties.get("reused_node_id")

        # ---------------------------------------------------------
        # Step A: Candidate Generation
        # ---------------------------------------------------------
        candidate_nodes: Dict[str, GraphNode] = {}

        # 1. Proposed reused_node_id from LLM (examined as a candidate only!)
        if proposed_reused_id and proposed_reused_id in self.graph._nodes:
            cand = self.graph.get_node(proposed_reused_id)
            if cand:
                candidate_nodes[cand.id] = cand

        # 2. Inverted index alias lookup
        if val_norm in self.graph._alias_index:
            for cid in self.graph._alias_index[val_norm]:
                cand = self.graph.get_node(cid)
                if cand:
                    candidate_nodes[cand.id] = cand

        for a in incoming.aliases:
            a_norm = self._normalize(a)
            if a_norm in self.graph._alias_index:
                for cid in self.graph._alias_index[a_norm]:
                    cand = self.graph.get_node(cid)
                    if cand:
                        candidate_nodes[cand.id] = cand

        # 3. Existing reuse candidate logic
        reuse_cand = self.graph.find_reuse_candidate(
            node_type=incoming.type,
            value=incoming.value,
            label=incoming.label,
        )
        if reuse_cand and reuse_cand.id:
            candidate_nodes[reuse_cand.id] = reuse_cand

        # 4. Search nodes by query
        search_cands = self.graph.search_nodes(query=raw_val, limit=5)
        for cand in search_cands:
            candidate_nodes[cand.id] = cand

        # 5. Search by tokens if multi-word query (e.g. "Robert Smith" -> "Robert", "Smith")
        words = [w for w in val_norm.split() if len(w) > 3]
        for w in words:
            for cand in self.graph.search_nodes(query=w, limit=5):
                candidate_nodes[cand.id] = cand

        logger.debug(
            "entity_resolution.candidate_generated",
            incoming_id=incoming.id,
            incoming_value=raw_val,
            candidate_count=len(candidate_nodes),
            candidate_ids=list(candidate_nodes.keys()),
        )

        if not candidate_nodes:
            return ResolutionDecision(
                status=ResolutionStatus.NEW_ENTITY,
                rationale="No candidate nodes found in existing memory",
                algorithm_version=self.algorithm_version,
            )

        # ---------------------------------------------------------
        # Step B & C: Contradiction Gates & Heuristic Scoring
        # ---------------------------------------------------------
        evaluations: List[CandidateEvaluation] = []
        for cand_id, cand_node in candidate_nodes.items():
            has_contradiction, reasons = self.check_hard_contradictions(incoming, cand_node)
            score, signals = self.score_candidate(incoming, cand_node, proposed_reused_id)
            compat = self.check_type_compatibility(incoming.type, cand_node.type)

            evaluations.append(
                CandidateEvaluation(
                    candidate_id=cand_id,
                    candidate_node=cand_node,
                    candidate_score=score,
                    matching_signals=signals,
                    contradiction_detected=has_contradiction,
                    contradiction_reasons=reasons,
                    compatibility=compat.value,
                )
            )

        # Filter out candidates blocked by hard contradictions
        valid_evals = [e for e in evaluations if not e.contradiction_detected]
        contradictory_evals = [e for e in evaluations if e.contradiction_detected]

        for c_eval in contradictory_evals:
            logger.info(
                "entity_resolution.contradiction",
                incoming_id=incoming.id,
                incoming_value=incoming.value,
                candidate_id=c_eval.candidate_id,
                candidate_value=c_eval.candidate_node.value,
                candidate_score=c_eval.candidate_score,
                reasons=c_eval.contradiction_reasons,
            )

        if not valid_evals:
            # All candidates blocked by hard contradictions
            all_reasons = [r for c in contradictory_evals for r in c.contradiction_reasons]
            return ResolutionDecision(
                status=ResolutionStatus.REJECTED_CONTRADICTION,
                all_candidates=evaluations,
                contradiction_detected=True,
                contradiction_reasons=all_reasons,
                rationale="All candidate matches rejected by hard contradiction gates",
                algorithm_version=self.algorithm_version,
            )

        # ---------------------------------------------------------
        # Step D: Ambiguity Margin & Decision Evaluation
        # ---------------------------------------------------------
        valid_evals.sort(key=lambda x: x.candidate_score, reverse=True)
        top = valid_evals[0]
        second = valid_evals[1] if len(valid_evals) > 1 else None

        top_score = top.candidate_score
        second_score = second.candidate_score if second else 0.0
        decision_margin = round(top_score - second_score, 4)

        if top_score < self.match_threshold:
            if top_score >= 0.50:
                # Moderate score but below match threshold -> leave uncertain
                return ResolutionDecision(
                    status=ResolutionStatus.UNCERTAIN_MATCH,
                    winning_candidate=top,
                    all_candidates=evaluations,
                    candidate_score=top_score,
                    second_candidate_score=second_score,
                    decision_margin=decision_margin,
                    matching_signals=top.matching_signals,
                    resolution_method="SCORE_BELOW_THRESHOLD",
                    rationale=f"Top candidate score {top_score} below match threshold {self.match_threshold}",
                    algorithm_version=self.algorithm_version,
                )
            else:
                return ResolutionDecision(
                    status=ResolutionStatus.NEW_ENTITY,
                    winning_candidate=None,
                    all_candidates=evaluations,
                    candidate_score=top_score,
                    second_candidate_score=second_score,
                    decision_margin=decision_margin,
                    matching_signals=top.matching_signals,
                    resolution_method="LOW_SIMILARITY",
                    rationale=f"Top candidate score {top_score} indicates distinct entity",
                    algorithm_version=self.algorithm_version,
                )

        # Top score is >= match_threshold; check decision margin
        if second is not None and decision_margin < self.ambiguity_margin:
            # Ambiguous close candidates
            return ResolutionDecision(
                status=ResolutionStatus.UNCERTAIN_MATCH,
                winning_candidate=top,
                all_candidates=evaluations,
                candidate_score=top_score,
                second_candidate_score=second_score,
                decision_margin=decision_margin,
                matching_signals=top.matching_signals,
                resolution_method="INSUFFICIENT_MARGIN",
                rationale=f"Decision margin {decision_margin} below required ambiguity margin {self.ambiguity_margin}",
                algorithm_version=self.algorithm_version,
            )

        # Confirmed match
        method = top.matching_signals[0] if top.matching_signals else "CONFIRMED_HEURISTIC"
        return ResolutionDecision(
            status=ResolutionStatus.CONFIRMED_MATCH,
            canonical_id=top.candidate_id,
            winning_candidate=top,
            all_candidates=evaluations,
            candidate_score=top_score,
            second_candidate_score=second_score,
            decision_margin=decision_margin,
            matching_signals=top.matching_signals,
            resolution_method=method,
            rationale=f"Confirmed match to {top.candidate_id} with score {top_score} and margin {decision_margin}",
            algorithm_version=self.algorithm_version,
        )

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
                    page_diff = abs(page_number - max_p)
                    score += max(0.0, 3.0 - (page_diff * 0.5))

                candidates.append((score, node))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top_nodes = [node for _, node in candidates[:self.anchor_max_k]]
            return [n.to_dict() for n in top_nodes]

    async def ingest_page_delta(
        self,
        delta: PageDelta,
        delta_checksum: Optional[str] = None,
    ) -> bool:
        """
        Canonically ingest a PageDelta into the central GraphMemory under lock.
        1. Checks idempotency: if page was already ingested with the same checksum, safely no-ops.
        2. Resolves entities via candidate -> contradiction check -> score -> margin -> decision pipeline.
        3. Fuses confirmed entities while preserving source pages, evidence, and audit logs.
        4. Maintains scoped (page_number, local_id) -> canonical_id mapping.
        5. Rewrites relationship endpoints to canonical IDs.
        6. Collects unresolved references for post-merge resolution.
        """
        async with self._lock:
            page_num = delta.page_number

            # Check idempotency
            if page_num in self._ingested_deltas:
                stored_checksum = self._ingested_deltas[page_num]
                if delta_checksum and stored_checksum and delta_checksum == stored_checksum:
                    logger.info(
                        "page.retry.idempotent_noop",
                        page_number=page_num,
                        checksum=delta_checksum,
                    )
                    return True
                elif delta_checksum and stored_checksum and delta_checksum != stored_checksum:
                    self._checksum_conflicts.append({
                        "page_number": page_num,
                        "stored_checksum": stored_checksum,
                        "incoming_checksum": delta_checksum,
                    })
                    logger.warning(
                        "page.retry.checksum_conflict",
                        page_number=page_num,
                        stored_checksum=stored_checksum,
                        incoming_checksum=delta_checksum,
                    )
                    return False

            # ---------------------------------------------------------
            # Step 1: Ingest and Canonicalize Entities (Nodes)
            # ---------------------------------------------------------
            for ent in delta.entities:
                raw_id = ent.id
                ent_val = (ent.value or "").strip()
                if not ent_val:
                    continue

                # Run resolution pipeline
                decision = self.resolve_entity(ent, page_num=page_num)
                canonical_id: Optional[str] = None

                if decision.status == ResolutionStatus.CONFIRMED_MATCH and decision.canonical_id:
                    canonical_id = decision.canonical_id
                    # Fused merge: preserve source pages, evidence, aliases, observations
                    target_node = self.graph.get_node(canonical_id)
                    if target_node:
                        target_node.add_page(page_num)
                        for ev in ent.evidence:
                            target_node.add_evidence(ev.page_number, ev.text, ev.confidence)
                        for al in ent.aliases:
                            target_node.add_alias(al)
                        if ent.category == "EXPLICIT":
                            target_node.category = "EXPLICIT"

                        # Promote UNCERTAIN node to ASSERTED if incoming evidence is ASSERTED
                        if target_node.status == "UNCERTAIN" and ent.status == "ASSERTED":
                            target_node.status = "ASSERTED"
                            # If incoming has a longer/more complete value, update canonical value and register old as alias
                            if len(ent.value) > len(target_node.value):
                                old_val = target_node.value
                                self.graph.update_node(target_node.id, {"value": ent.value})
                                target_node.add_alias(old_val)

                        # Preserve raw observation
                        obs_list = target_node.properties.setdefault("observations", [])
                        obs_list.append({
                            "raw_id": raw_id,
                            "page_number": page_num,
                            "value": ent.value,
                            "label": ent.label,
                            "evidence": [e.to_dict() for e in ent.evidence],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                        # Update confidence metrics
                        target_node.resolution_confidence = decision.candidate_score
                        target_node.extraction_confidence = min(target_node.extraction_confidence, ent.extraction_confidence)
                        target_node.confidence = max(target_node.confidence, ent.confidence)

                        # Record audit
                        audit_rec = MergeAuditRecord(
                            source_node_id=raw_id,
                            source_page=page_num,
                            source_value=ent.value,
                            canonical_node_id=canonical_id,
                            canonical_value=target_node.value,
                            resolution_method=decision.resolution_method,
                            candidate_score=decision.candidate_score,
                            second_candidate_score=decision.second_candidate_score,
                            decision_margin=decision.decision_margin,
                            matching_signals=decision.matching_signals,
                            contradiction_result="PASSED_HARD_GATES",
                            contradiction_reasons=[],
                            source_pages=list(target_node.source_pages),
                            evidence=[e.to_dict() for e in target_node.evidence],
                            algorithm_version=self.algorithm_version,
                            pipeline_version=self.pipeline_version,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        self._merge_ledger.append(audit_rec)
                        target_node.properties.setdefault("merge_audit", []).append(audit_rec.to_dict())

                        logger.info(
                            "entity_resolution.confirmed",
                            source_id=raw_id,
                            source_value=ent.value,
                            canonical_id=canonical_id,
                            method=decision.resolution_method,
                            score=decision.candidate_score,
                            margin=decision.decision_margin,
                        )

                elif decision.status == ResolutionStatus.UNCERTAIN_MATCH:
                    # Ambiguous / uncertain match: do NOT merge into candidate
                    node_id = f"node_unc_{page_num}_{uuid.uuid4().hex[:6]}"
                    pages = list(ent.source_pages) if ent.source_pages else [page_num]
                    props = dict(ent.properties)
                    props["raw_id"] = raw_id
                    props["uncertain_candidate_id"] = (
                        decision.winning_candidate.candidate_id if decision.winning_candidate else None
                    )
                    props["candidate_score"] = decision.candidate_score
                    props["decision_margin"] = decision.decision_margin
                    props["rationale"] = decision.rationale

                    unc_node = GraphNode(
                        id=node_id,
                        type=ent.type,
                        label=ent.label,
                        value=ent_val,
                        properties=props,
                        aliases=list(ent.aliases),
                        evidence=list(ent.evidence),
                        source_pages=pages,
                        confidence=ent.confidence,
                        status="UNCERTAIN",
                        category=ent.category or "CONTEXTUAL",
                        extraction_confidence=ent.extraction_confidence,
                        resolution_confidence=decision.candidate_score,
                        evidence_strength=ent.evidence_strength,
                    )
                    self.graph.create_node(unc_node)
                    canonical_id = node_id

                    logger.info(
                        "entity_resolution.uncertain",
                        source_id=raw_id,
                        source_value=ent.value,
                        created_node_id=node_id,
                        competing_candidate=(
                            decision.winning_candidate.candidate_id if decision.winning_candidate else None
                        ),
                        margin=decision.decision_margin,
                    )

                else:
                    # NEW_ENTITY or REJECTED_CONTRADICTION
                    if raw_id and raw_id not in self.graph._nodes:
                        node_id = raw_id
                    else:
                        node_id = f"node_{page_num}_{uuid.uuid4().hex[:6]}"

                    pages = list(ent.source_pages) if ent.source_pages else [page_num]
                    props = dict(ent.properties)
                    props["raw_id"] = raw_id
                    if decision.status == ResolutionStatus.REJECTED_CONTRADICTION:
                        props["contradiction_reasons"] = decision.contradiction_reasons

                    new_node = GraphNode(
                        id=node_id,
                        type=ent.type,
                        label=ent.label,
                        value=ent_val,
                        properties=props,
                        aliases=list(ent.aliases),
                        evidence=list(ent.evidence),
                        source_pages=pages,
                        confidence=ent.confidence,
                        status=ent.status or "ASSERTED",
                        category=ent.category or "CONTEXTUAL",
                        extraction_confidence=ent.extraction_confidence,
                        resolution_confidence=None,  # NEW_ENTITY does not imply identity resolution against an existing entity
                        evidence_strength=ent.evidence_strength,
                    )
                    self.graph.create_node(new_node)
                    canonical_id = node_id

                    logger.info(
                        "entity_resolution.new_entity",
                        node_id=node_id,
                        type=ent.type,
                        value=ent_val,
                        status=new_node.status,
                        contradiction_detected=decision.contradiction_detected,
                    )

                # Register scoped mappings for this page worker
                self._id_remap[(page_num, raw_id)] = canonical_id
                self._id_remap[(page_num, self._normalize(raw_id))] = canonical_id
                self._id_remap[(page_num, self._normalize(ent_val))] = canonical_id
                if ent.label:
                    self._id_remap[(page_num, self._normalize(ent.label))] = canonical_id

                for a in ent.aliases:
                    if a:
                        self.graph.add_alias(canonical_id, a)
                        self._id_remap[(page_num, self._normalize(a))] = canonical_id

            # Incorporate worker's raw_id_map
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
                raw_src = (edge.source_node or "").strip()
                raw_tgt = (edge.target_node or "").strip()
                if not raw_src or not raw_tgt:
                    continue

                canon_src = (
                    self._id_remap.get((page_num, raw_src))
                    or self._id_remap.get((page_num, self._normalize(raw_src)))
                )
                if not canon_src and raw_src in self.graph._nodes:
                    canon_src = raw_src

                canon_tgt = (
                    self._id_remap.get((page_num, raw_tgt))
                    or self._id_remap.get((page_num, self._normalize(raw_tgt)))
                )
                if not canon_tgt and raw_tgt in self.graph._nodes:
                    canon_tgt = raw_tgt

                if not canon_src or not canon_tgt:
                    logger.warning(
                        "graph.relationship.endpoint_unresolved",
                        page_number=page_num,
                        relationship=edge.relationship,
                        raw_src=raw_src,
                        raw_tgt=raw_tgt,
                        resolved_src=canon_src,
                        resolved_tgt=canon_tgt,
                    )
                    continue

                if canon_src not in self.graph._nodes or canon_tgt not in self.graph._nodes:
                    logger.warning(
                        "graph.relationship.endpoint_not_in_graph",
                        page_number=page_num,
                        relationship=edge.relationship,
                        raw_src=raw_src,
                        raw_tgt=raw_tgt,
                        resolved_src=canon_src,
                        resolved_tgt=canon_tgt,
                    )
                    continue

                if canon_src == canon_tgt:
                    logger.debug(
                        "graph.relationship.reflexive_skipped",
                        page_number=page_num,
                        node_id=canon_src,
                        relationship=edge.relationship,
                    )
                    continue

                # Deduplicate: avoid creating duplicate edges on page retries
                existing_edges = self.graph.get_edges(
                    source_node=canon_src,
                    target_node=canon_tgt,
                    relationship=edge.relationship,
                )
                duplicate = any(e.source_page == page_num for e in existing_edges)
                if not duplicate:
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
                        extraction_confidence=edge.extraction_confidence,
                        relationship_confidence=edge.relationship_confidence,
                        evidence_strength=edge.evidence_strength,
                    )
                    try:
                        self.graph.create_edge(canon_edge)
                    except Exception as exc:
                        logger.debug("manager.edge_creation_skipped", error=str(exc))

            # ---------------------------------------------------------
            # Step 3: Collect Unresolved References for Post-Merge
            # ---------------------------------------------------------
            self._unresolved.extend(delta.unresolved_references)
            self._ingested_deltas[page_num] = delta_checksum or ""
            return True

    async def reconcile_cross_references(self) -> None:
        """
        Reconcile cross-page anaphoric references post-merge.
        Ranks candidates deterministically using target type, page proximity,
        confidence, and category, and enforces hard contradiction and ambiguity checks.
        Ambiguous cases are left unresolved.
        """
        async with self._lock:
            if not self._unresolved:
                return

            logger.info("graph.reconciliation.started", unresolved_count=len(self._unresolved))

            remaining_unresolved: List[UnresolvedRef] = []

            for ref in self._unresolved:
                phrase = (ref.phrase or "").strip()
                target_type = (ref.target_type or "").strip()
                page_num = ref.page_number

                if not phrase and not target_type:
                    remaining_unresolved.append(ref)
                    continue

                norm_tgt_type = self._normalize(target_type)
                candidates: List[Tuple[float, GraphNode]] = []

                for node in self.graph._nodes.values():
                    node_type_norm = self._normalize(node.type)
                    node_label_norm = self._normalize(node.label)

                    type_compat = self.check_type_compatibility(target_type, node.type)
                    if type_compat == TypeCompatibility.INCOMPATIBLE:
                        continue

                    type_match = False
                    if norm_tgt_type:
                        if norm_tgt_type in (node_type_norm, node_label_norm):
                            type_match = True
                        elif type_compat == TypeCompatibility.COMPATIBLE:
                            type_match = True
                    else:
                        type_match = True

                    if not type_match:
                        continue

                    score = 1.0
                    pages_before_or_at = [p for p in node.source_pages if p <= page_num]
                    if pages_before_or_at:
                        closest_page = max(pages_before_or_at)
                        page_distance = page_num - closest_page
                        score += 3.0 / (1.0 + page_distance)
                    else:
                        score -= 2.0

                    if node.category == "EXPLICIT":
                        score += 2.0
                    elif node.category == "CONTEXTUAL":
                        score += 1.0

                    score += node.confidence
                    candidates.append((score, node))

                if not candidates:
                    logger.info("graph.reconciliation.unresolved", phrase=phrase, reason="no_candidates")
                    remaining_unresolved.append(ref)
                    continue

                candidates.sort(key=lambda x: x[0], reverse=True)
                top_score, best_candidate = candidates[0]

                # Ambiguity check
                if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < self.ambiguity_margin:
                    logger.info(
                        "graph.reconciliation.unresolved",
                        phrase=phrase,
                        target_type=target_type,
                        reason="ambiguous_candidates",
                        cand1=best_candidate.id,
                        cand2=candidates[1][1].id,
                    )
                    remaining_unresolved.append(ref)
                    continue

                if top_score < 1.5:
                    logger.info(
                        "graph.reconciliation.unresolved",
                        phrase=phrase,
                        target_type=target_type,
                        reason="low_confidence",
                        score=top_score,
                    )
                    remaining_unresolved.append(ref)
                    continue

                canon_src = (
                    self._id_remap.get((page_num, ref.source_node_id))
                    or self._id_remap.get((page_num, self._normalize(ref.source_node_id)))
                    or ref.source_node_id
                )

                if canon_src and canon_src in self.graph._nodes and canon_src != best_candidate.id:
                    existing_edges = self.graph.get_edges(
                        source_node=canon_src,
                        target_node=best_candidate.id,
                        relationship="REFERS_TO",
                    )
                    duplicate = any(e.source_page == page_num for e in existing_edges)
                    if not duplicate:
                        ref_edge = GraphEdge(
                            id=f"edge_ref_{page_num}_{uuid.uuid4().hex[:6]}",
                            source_node=canon_src,
                            target_node=best_candidate.id,
                            relationship="REFERS_TO",
                            evidence=ref.rationale or phrase,
                            source_page=page_num,
                            confidence=round(min(1.0, top_score / 5.0), 3),
                            status="INFERRED",
                            relationship_confidence=round(min(1.0, top_score / 5.0), 3),
                            extraction_confidence=1.0,
                        )
                        self.graph.create_edge(ref_edge)
                elif not (canon_src and canon_src in self.graph._nodes):
                    remaining_unresolved.append(ref)
                    continue

                if phrase:
                    self.graph.add_alias(best_candidate.id, phrase)

                logger.info(
                    "graph.reconciliation.resolved",
                    phrase=phrase,
                    target_type=target_type,
                    resolved_node=best_candidate.id,
                    target_value=best_candidate.value,
                    score=top_score,
                )

            self._unresolved = remaining_unresolved

    @property
    def unresolved_count(self) -> int:
        return len(self._unresolved)
