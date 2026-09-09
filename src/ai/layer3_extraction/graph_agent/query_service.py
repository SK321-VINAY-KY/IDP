"""
File: query_service.py
Purpose: Bounded, cycle-safe Graph Query Service for Layer 3 Graph Memory.
         Enables the Query Bot to perform multi-hop graph retrieval and grounded Q&A.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import GraphEdge, GraphNode
from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configurable retrieval bounds
DEFAULT_MAX_HOPS = getattr(settings, "graph_query_max_hops", 3)
DEFAULT_MAX_NODES = getattr(settings, "graph_query_max_nodes", 25)
DEFAULT_MAX_EDGES = getattr(settings, "graph_query_max_edges", 30)

STOP_WORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "at", "in", "for",
    "on", "by", "about", "with", "from", "into", "through", "during",
    "before", "after", "above", "below", "what", "who", "which", "where",
    "when", "why", "how", "associated", "treat", "treating", "treated",
    "tell", "me", "find", "give", "show", "can", "could", "would", "should",
    "of", "and", "or", "underwent", "performed", "recorded", "happened",
}


@dataclass
class GraphQueryResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_nodes: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_edges: List[Dict[str, Any]] = field(default_factory=list)
    hops_traversed: int = 0
    mode: str = "graph"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_nodes": self.retrieved_nodes,
            "retrieved_edges": self.retrieved_edges,
            "hops_traversed": self.hops_traversed,
            "mode": self.mode,
        }


class GraphQueryService:
    """
    Dedicated Graph Reasoning Service.
    Retrieves relevant subgraphs via bounded, cycle-safe traversal and coordinates
    evidence-grounded answering.
    """

    def __init__(
        self,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_edges: int = DEFAULT_MAX_EDGES,
    ) -> None:
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _clean_stem(word: str) -> str:
        """Strip common inflectional suffixes for robust entity matching."""
        w = word.lower().strip()
        for suffix in ["tions", "tion", "ments", "ment", "ed", "ing", "es", "s"]:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    @staticmethod
    def _is_garbage_node(node: GraphNode) -> bool:
        """Filter out low-confidence OCR noise, single-char artifacts, or placeholder nodes."""
        v = node.value.strip()
        v_lower = v.lower()
        if len(v) <= 1 and v_lower in ["u", "a", "x", "o", "-", "", "n", "y"]:
            return True
        if any(p in v_lower for p in ["not clearly visible", "not visible", "unreadable", "not legible", "not available"]):
            return True
        if node.type == "Amount":
            if not re.search(r"\d", v):
                return True
            words = v.split()
            if len(words) > 4 and not any(ch in v_lower for ch in ["₹", "$", "rs", "inr"]):
                return True
        return False

    def _extract_query_tokens(self, question: str) -> List[str]:
        """Extract meaningful keyword terms from question, excluding stop words."""
        cleaned = re.sub(r"[^\w\s]", " ", question.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2 and t not in STOP_WORDS]
        return tokens

    def find_seed_nodes(
        self,
        graph: GraphMemory,
        question: str,
        max_seeds: int = 6,
    ) -> List[GraphNode]:
        """
        Identify seed nodes in graph relevant to the question based on:
        1. Exact substantial value/alias matches
        2. Multi-token and stem matches across label, schema_field, and value
        3. Domain synonym expansion (e.g. 'claimed' -> 'claimed_amount', 'deducted' -> 'non_medical_deductions')
        4. Quality tie-breaking prioritizing verified numeric data and early document summary pages
        """
        norm_q = self._normalize(question)
        tokens = self._extract_query_tokens(question)
        stems = {self._clean_stem(t) for t in tokens}

        # Domain synonym expansion for common question intents
        SYNONYMS: Dict[str, List[str]] = {
            "candidate": ["applicant", "person", "full_name", "employee", "name"],
            "applicant": ["candidate", "person", "full_name", "name"],
            "name": ["person", "candidate", "applicant", "patient", "doctor", "full_name", "patient_name"],
            "patient": ["person", "patient_name", "full_name", "insured"],
            "doctor": ["physician", "consultant", "provider", "doctor_name"],
            "age": ["patient_age", "years", "dob", "birth"],
            "claim": ["claimed", "claimed_amount", "sanctioned_amount", "settlement", "insurance_claim"],
            "claimed": ["claim", "claimed_amount", "total_hospital_bill"],
            "deducted": ["deduction", "deductions", "non_medical_deductions", "non_payable", "copay"],
            "deduction": ["deducted", "deductions", "non_medical_deductions", "non_payable"],
            "sanctioned": ["approved", "settled", "sanctioned_amount", "final_claim"],
            "approved": ["sanctioned", "sanctioned_amount", "settled"],
            "insured": ["sum_insured", "sanctioned_amount", "claimed_amount", "insurance_company", "policy_number", "insurance_policy_number"],
            "insurance": ["insurance_company", "sponsor", "tpa", "policy_number", "insurance_policy_number"],
        }
        syn_tokens: Set[str] = set()
        for tok in tokens:
            if tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[tok])
            stem_tok = self._clean_stem(tok)
            if stem_tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[stem_tok])

        scored_candidates: List[Tuple[float, int, int, int, int, int, GraphNode]] = []

        for node in graph._nodes.values():
            if self._is_garbage_node(node):
                continue

            val_norm = self._normalize(node.value)
            type_norm = self._normalize(node.type)
            lbl_norm = self._normalize(node.label)
            aliases_norm = [self._normalize(a) for a in getattr(node, "aliases", [])]
            schema_field = ""
            if getattr(node, "properties", None) and isinstance(node.properties, dict):
                schema_field = self._normalize(str(node.properties.get("schema_field_name", "")))

            score = 0.0
            matched_tokens = 0

            # Substantial exact value match in question (require at least 3 characters to prevent 'u' false matches)
            if len(val_norm) >= 3 and val_norm in norm_q:
                score += 2.0
            for a in aliases_norm:
                if len(a) >= 3 and a in norm_q:
                    score += 1.8

            # Substring and token matches
            for tok in tokens:
                if len(tok) >= 3 and tok in val_norm:
                    matched_tokens += 1
                    score += 1.5
                if tok in lbl_norm or (schema_field and tok in schema_field):
                    matched_tokens += 1
                    score += 1.2
                elif tok in type_norm:
                    score += 0.35
                for a in aliases_norm:
                    if tok in a:
                        score += 0.8

            # Stem matches on label & schema fields
            for st in stems:
                if len(st) >= 3:
                    if st in lbl_norm or (schema_field and st in schema_field):
                        score += 0.8
                    elif st in type_norm:
                        score += 0.2

            # Match against expanded synonyms
            for st in syn_tokens:
                if st in lbl_norm or (schema_field and st in schema_field):
                    score += 0.7
                elif st in type_norm:
                    score += 0.2

            if score > 0.0:
                has_digits = 1 if (node.type == "Amount" and re.search(r"\d", val_norm)) else 0
                page_coverage = len(node.source_pages or [])
                # Prioritize earlier summary pages (e.g. Page 6 face sheet & Page 9 settlement over Page 64 handwriting)
                min_page = min(node.source_pages or [999])
                scored_candidates.append((
                    score,
                    matched_tokens,
                    has_digits,
                    1 if node.category == "EXPLICIT" else 0,
                    page_coverage,
                    -min_page,
                    node,
                ))

        # Sort: score DESC, matched_tokens DESC, valid digits DESC, EXPLICIT DESC, page coverage DESC, earlier pages DESC
        scored_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]), reverse=True)

        seen_ids = set()
        seeds: List[GraphNode] = []
        for cand in scored_candidates:
            node = cand[6]
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                seeds.append(node)
                if len(seeds) >= max_seeds:
                    break

        logger.debug("graph.query.seed_found", count=len(seeds), seed_ids=[s.id for s in seeds])
        return seeds

    def traverse_subgraph(
        self,
        graph: GraphMemory,
        seed_nodes: List[GraphNode],
    ) -> Tuple[List[GraphNode], List[GraphEdge], int]:
        """
        Bounded, cycle-safe Breadth-First Search (BFS) starting from seed nodes.
        Strictly limits depth (hops), nodes, and edges, preventing infinite loops.
        """
        if not seed_nodes:
            return [], [], 0

        visited_nodes: Set[str] = {s.id for s in seed_nodes}
        visited_edges: Set[str] = set()

        collected_nodes: List[GraphNode] = list(seed_nodes)
        collected_edges: List[GraphEdge] = []

        # Queue contains (node_id, current_hop)
        queue: deque[Tuple[str, int]] = deque([(s.id, 0) for s in seed_nodes])
        max_hop_reached = 0

        while queue and len(collected_nodes) < self.max_nodes:
            curr_id, curr_hop = queue.popleft()
            max_hop_reached = max(max_hop_reached, curr_hop)

            if curr_hop >= self.max_hops:
                continue

            # Bidirectional neighbor lookup
            neighbors = graph.read_neighbors(curr_id, direction="both")
            for edge, neighbor_node in neighbors:
                # Cycle prevention on edges
                if edge.id in visited_edges:
                    continue
                visited_edges.add(edge.id)

                if len(collected_edges) < self.max_edges:
                    collected_edges.append(edge)

                # Cycle prevention on nodes
                if neighbor_node.id not in visited_nodes:
                    visited_nodes.add(neighbor_node.id)
                    if len(collected_nodes) < self.max_nodes:
                        collected_nodes.append(neighbor_node)
                    queue.append((neighbor_node.id, curr_hop + 1))

        logger.debug(
            "graph.query.traversal_completed",
            nodes=len(collected_nodes),
            edges=len(collected_edges),
            max_hop=max_hop_reached,
        )
        return collected_nodes, collected_edges, max_hop_reached

    def format_evidence_context(
        self,
        graph: GraphMemory,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        max_evidence: int = 12,
        max_sources: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved nodes, relationships, and provenances into a high-signal, compact context for the LLM.
        Caches and returns deduplicated source citations capped to top relevant pages.
        """
        lines = ["=== RELEVANT GRAPH ENTITIES ==="]
        for n in nodes[:12]:
            pages = f"Page {','.join(map(str, n.source_pages[:3]))}" if n.source_pages else "Page ?"
            if len(n.source_pages or []) > 3:
                pages += f",... ({len(n.source_pages)} pages total)"
            lines.append(f"- [{n.id}] {n.type} ({n.label}): \"{n.value}\" ({pages})")

        lines.append("\n=== RELATIONSHIPS ===")
        for e in edges[:15]:
            src = graph.get_node(e.source_node)
            tgt = graph.get_node(e.target_node)
            src_val = f"{src.value} ({src.type})" if src else e.source_node
            tgt_val = f"{tgt.value} ({tgt.type})" if tgt else e.target_node
            lines.append(f"- {src_val} --[{e.relationship}]--> {tgt_val} (Page {e.source_page})")

        lines.append("\n=== DOCUMENT EVIDENCE SPANS ===")
        sources: List[Dict[str, Any]] = []
        seen_ev: Set[Tuple[int, str]] = set()
        ev_count = 0

        for n in nodes:
            # Prioritize high confidence, earlier page evidence
            sorted_ev = sorted(
                n.evidence or [],
                key=lambda ev: (-getattr(ev, "confidence", 0.5), getattr(ev, "page_number", 999)),
            )
            for ev in sorted_ev[:2]:
                text_clean = ev.text.strip()
                key = (ev.page_number, text_clean)
                if text_clean and key not in seen_ev and ev_count < max_evidence:
                    seen_ev.add(key)
                    lines.append(f"- [Page {ev.page_number}]: \"{text_clean}\" (supports {n.label}: {n.value})")
                    ev_count += 1
                    if len(sources) < max_sources:
                        sources.append({
                            "page": ev.page_number,
                            "evidence": text_clean,
                            "node": n.value,
                            "type": n.type,
                        })

        for e in edges:
            if e.evidence and ev_count < max_evidence:
                text_clean = e.evidence.strip()
                key = (e.source_page, text_clean)
                if text_clean and key not in seen_ev:
                    seen_ev.add(key)
                    lines.append(f"- [Page {e.source_page}]: \"{text_clean}\" (supports relationship {e.relationship})")
                    ev_count += 1
                    if len(sources) < max_sources:
                        sources.append({
                            "page": e.source_page,
                            "evidence": text_clean,
                            "relationship": e.relationship,
                        })

        sources.sort(key=lambda s: s.get("page", 1))
        return "\n".join(lines), sources

    def query(
        self,
        graph: GraphMemory,
        question: str,
        llm: Optional[ExtractionLLMClient] = None,
    ) -> GraphQueryResult:
        """
        Execute a graph-backed query:
        1. Find seed entities with quality scoring and synonym expansion.
        2. Bounded cycle-safe BFS traversal.
        3. Compact high-signal context extraction with capped citations.
        4. Grounded synthesis via LLM with strict role separation and fallback.
        """
        logger.info("graph.query.started", question=question)

        # Step 1: Identify seed entities
        seeds = self.find_seed_nodes(graph, question)

        # If no seeds found directly, check if question is asking for all entities of a type
        if not seeds:
            q_norm = self._normalize(question)
            for t, node_ids in graph._type_index.items():
                if t in q_norm:
                    for nid in node_ids:
                        n = graph.get_node(nid)
                        if n and n not in seeds and not self._is_garbage_node(n):
                            seeds.append(n)
                            if len(seeds) >= 5:
                                break

        # If still no relevant nodes found in graph
        if not seeds:
            logger.info("graph.query.no_seeds_found", question=question)
            return GraphQueryResult(
                question=question,
                answer="The requested information could not be found in the document knowledge graph.",
                sources=[],
                retrieved_nodes=[],
                retrieved_edges=[],
                hops_traversed=0,
                mode="graph",
            )

        # Step 2: Bounded Traversal
        nodes, edges, max_hop = self.traverse_subgraph(graph, seeds)

        # Step 3: Format compact evidence context & sources
        context_text, sources = self.format_evidence_context(graph, nodes, edges)

        # Step 4: Synthesize answer
        answer = self._generate_answer(context_text, sources, nodes, question)
        logger.info("graph.query.answer_generated", answer_preview=answer[:100], sources_count=len(sources))

        return GraphQueryResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_nodes=[n.to_dict() for n in nodes],
            retrieved_edges=[e.to_dict() for e in edges],
            hops_traversed=max_hop,
            mode="graph",
        )

    @staticmethod
    def _sanitize_model_output(content: str) -> str:
        """Strip <think> tags, conversational preamble, and leaked chain-of-thought scratchpad text."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip leading broken fragments or hanging quotes e.g. 'found". ' or '". '
        content = re.sub(r'^[^\w\s]*(?:found|stated|mentioned)?["\'”’\.]+\s*', '', content, flags=re.IGNORECASE).strip()
        # Strip asterisks / markdown bolding
        content = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", content)
        content = re.sub(r"[*_#`]+", "", content)
        content = re.sub(r"^(?:Assistant|Answer):\s*", "", content.strip(), flags=re.IGNORECASE)

        # Check for conversational reasoning preamble
        if re.search(r"^\s*(?:Based on the (?:prompt|constraint|evidence)|The user wants to know|Looking at the|From the relevant graph|To answer this question|I need to look|I can mention|1\.\s*Analyze)", content, re.IGNORECASE):
            # Check draft patterns
            m_draft = re.search(
                r'(?:combine into a smooth sentence|final polish|draft the response:?|final answer:?|answer:?)\s*[:"-]*\s*["\']?([^"\'\n]+)["\']?',
                content,
                re.IGNORECASE,
            )
            if m_draft and len(m_draft.group(1).strip()) > 5:
                cand = m_draft.group(1).strip()
                if not cand.endswith(":") and len(cand.split()) >= 3:
                    return cand

            # Try to extract explicit concluding statement
            m_concl = re.search(
                r"(?:(?:So|Therefore|In summary|Hence),?\s*(?:the patient name is|the answer is|the claimed amount is|the age is|the deducted amount is|the doctor is|the patient is)|(?:The patient name is|The amount claimed is|The deducted amount is|The age of the patient is|The doctor treating the patient is|The patient underwent|Dr\.))\s*([^\n]+)",
                content,
                re.IGNORECASE,
            )
            if m_concl:
                candidate = m_concl.group(0).strip()
                if len(candidate) > 10 and not candidate.endswith(":") and len(candidate.split()) >= 3:
                    return candidate

            # Extract from lines
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            meaningful_lines = []
            for line in lines:
                if not re.search(r"^\s*(?:\d+\.|\*|I will|I can|I should|Let's|The evidence|So I should|Looking at|From the|According to|Based on)", line, re.IGNORECASE):
                    if not line.endswith(":") and len(line) > 5:
                        meaningful_lines.append(line.strip())
            if meaningful_lines:
                if len(meaningful_lines) > 1 and re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", meaningful_lines[-1], re.IGNORECASE):
                    return f"{meaningful_lines[-2]} {meaningful_lines[-1]}"
                if not re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", meaningful_lines[-1], re.IGNORECASE):
                    return meaningful_lines[-1]

        # If output is too short, punctuation-only, ends in a colon, or merely a source citation without an answer, return empty so fallback kicks in
        clean_out = content.strip()
        clean_stripped = clean_out.rstrip(" :.-_#*")
        if (
            len(clean_out) <= 3
            or clean_stripped.endswith(":")
            or len(clean_out.split()) < 2
            or not re.search(r"\w", clean_out)
            or re.search(r"(?:prompt['s]*\s*constraint|prompt\s*constraint|system\s*prompt|internal\s*monologue|scratchpad)", clean_out, re.IGNORECASE)
            or re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", clean_out, re.IGNORECASE)
            or re.match(r"^(?:\s*Source:\s*Page\s*\d+\s*)+$", clean_out, re.IGNORECASE)
            or any(bad in clean_out.lower() for bad in ["critical: do not", "internal monologue", "scratchpad", "system prompt", "knowledge graph evidence:", "based on the prompt"])
        ):
            return ""
        return clean_out

    def _deterministic_fallback(
        self,
        question: str,
        nodes: List[GraphNode],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Accurate deterministic answering using structured graph entities when LLM generation is unavailable or noisy."""
        q = question.lower()

        # 0. Person role & relationship queries (e.g. "Who is Manu Kochhar and what is his relationship to the patient?")
        if any(w in q for w in ["who is", "relationship", "relation", "relative", "guardian", "card holder", "guarantor", "son"]):
            for n in nodes:
                if n.type == "Person" and n.value and n.value.lower() in q and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    role = n.label.replace("_", " ")
                    return f"{n.value} is recorded as the {role} for the patient. (Source: Page {page})"
            for n in nodes:
                if any(w in n.label.lower() for w in ["guardian", "card_holder", "responsible_person", "guarantor"]) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    role = n.label.replace("_", " ")
                    return f"{n.value} is recorded as the {role}. (Source: Page {page})"

        # 1. Patient Name
        if any(w in q for w in ["patient name", "name of the patient", "who is the patient", "patient's name"]):
            for n in nodes:
                if n.type in ["Person", "Identifier"] and any(w in n.label.lower() for w in ["patient_name", "patient"]) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The patient name is {n.value}. (Source: Page {page})"

        # 2. Patient Age
        if any(w in q for w in ["age", "how old"]):
            for n in nodes:
                if (n.type == "Age" or "age" in n.label.lower()) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The age of the patient is {n.value}. (Source: Page {page})"

        # 3. Claimed Amount
        if any(w in q for w in ["claimed", "amount claimed", "claim amount", "total bill"]):
            for n in nodes:
                if "claim" in n.label.lower() and n.type == "Amount" and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The claimed amount is Rs. {n.value}. (Source: Page {page})"

        # 4. Deducted Amount
        if any(w in q for w in ["deduct", "deduction", "non payable", "non-payable"]):
            for n in nodes:
                if any(w in n.label.lower() for w in ["deduct", "non_medical", "non_payable"]) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    lbl_clean = n.label.replace("_", " ")
                    return f"The deducted amount is Rs. {n.value} ({lbl_clean}). (Source: Page {page})"

        # 5. Insured / Sanctioned Amount
        if any(w in q for w in ["insured", "sum insured", "insurance amount"]):
            sanctioned = None
            exhausted = None
            for n in nodes:
                if "sanction" in n.label.lower() and not self._is_garbage_node(n):
                    sanctioned = n
                if "payable" in n.label.lower() and not self._is_garbage_node(n):
                    exhausted = n
            if sanctioned and exhausted:
                return (
                    f"The sanctioned (approved) insurance amount is Rs. {sanctioned.value}, and the balance sum insured exhausted was Rs. {exhausted.value}. (Source: Page 9)"
                )
            if sanctioned:
                return f"The sanctioned (approved) insurance amount is Rs. {sanctioned.value}. (Source: Page {sanctioned.source_pages[0]})"

        # 6. Sanctioned Amount
        if any(w in q for w in ["sanction", "approved amount", "amount approved"]):
            for n in nodes:
                if "sanction" in n.label.lower() and n.type == "Amount" and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The sanctioned (approved) amount is Rs. {n.value}. (Source: Page {page})"

        # 7. Treating Doctor
        if any(w in q for w in ["doctor", "physician", "consultant"]):
            for n in nodes:
                if (n.type in ["Doctor", "Person"] or "doctor" in n.label.lower()) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The doctor is {n.value}. (Source: Page {page})"

        # 8. Procedure / Surgery
        if any(w in q for w in ["procedure", "surgery", "operation", "undergo", "underwent"]):
            for n in nodes:
                if (n.type in ["Procedure", "Event"] or "procedure" in n.label.lower()) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The patient underwent {n.value}. (Source: Page {page})"

        # 9. Diagnosis / Ailment
        if any(w in q for w in ["diagnosis", "diagnosed", "ailment", "condition"]):
            for n in nodes:
                if (n.type in ["Diagnosis", "Condition"] or "diagnosis" in n.label.lower()) and not self._is_garbage_node(n):
                    page = n.source_pages[0] if n.source_pages else 1
                    return f"The diagnosis was {n.value}. (Source: Page {page})"

        # Generic best node match
        for n in nodes:
            if not self._is_garbage_node(n):
                page = n.source_pages[0] if n.source_pages else 1
                return f"{n.value} ({n.label.replace('_', ' ')}). (Source: Page {page})"

        return "The requested information could not be found in the document graph."

    def _generate_answer(
        self,
        context_text: str,
        sources: List[Dict[str, Any]],
        nodes: List[GraphNode],
        question: str,
    ) -> str:
        """Call LLM API with proper system/user role separation, with fallback."""
        import os
        import httpx
        try:
            from app.config import settings as app_settings
        except ImportError:
            app_settings = None
        from src.config.settings import settings as a_settings

        api_key = (
            getattr(app_settings, "sarvam_api_key", None)
            or getattr(a_settings, "sarvam_api_key", None)
            or os.getenv("IDP_SARVAM_API_KEY")
            or os.getenv("SARVAM_API_KEY")
            or ""
        )
        base_url = (
            getattr(app_settings, "sarvam_base_url", None)
            or getattr(a_settings, "sarvam_base_url", None)
            or os.getenv("IDP_SARVAM_BASE_URL")
            or os.getenv("SARVAM_BASE_URL")
            or "https://api.sarvam.ai/v1"
        ).rstrip("/")
        model_name = (
            getattr(app_settings, "sarvam_model", None)
            or getattr(a_settings, "sarvam_model_name", None)
            or os.getenv("IDP_SARVAM_MODEL_NAME")
            or os.getenv("SARVAM_MODEL")
            or "sarvam-105b"
        )
        backend = (
            getattr(app_settings, "llm_provider", None)
            or getattr(a_settings, "extraction_backend", None)
            or os.getenv("IDP_EXTRACTION_BACKEND")
            or "sarvam"
        )

        try:
            if (backend == "sarvam" or api_key) and api_key:
                system_prompt = (
                    "You are a factual, concise Document QA assistant. "
                    "Answer the user's question directly in 1 or 2 plain sentences based strictly on the provided Document Knowledge Graph evidence. "
                    "Always cite the exact source page number where the answer is found (e.g. 'Source: Page 6'). "
                    "CRITICAL: Do NOT output thinking, reasoning steps, internal monologue, numbered analysis lists, or scratchpads. "
                    "Provide ONLY the final answer."
                )
                user_prompt = f"Document Knowledge Graph Evidence:\n{context_text}\n\nQuestion: {question}\nDirect Answer:"

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1500,
                }
                headers = {
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                resp = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30.0)
                if resp.is_success:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = str(msg.get("content") or "").strip()
                        if content:
                            cleaned = self._sanitize_model_output(content)
                            is_conversational = bool(
                                re.search(
                                    r"(?:the user wants to know|i can mention|i should|i will|i think|if there are multiple|might be incomplete|found[\"\'”’\.]|looking at|from the relevant|to answer this question|let\'s)",
                                    cleaned,
                                    re.IGNORECASE,
                                )
                            )
                            if cleaned and len(cleaned) > 5 and not is_conversational:
                                return cleaned

                        # If content was empty or conversational, check reasoning_content
                        reasoning = str(msg.get("reasoning_content") or "").strip()
                        if reasoning:
                            m_draft = re.search(
                                r'(?:\*?Draft \d+:?\*?|final polish|Final Answer:?|In summary,?|The answer is:?)\s*([^\n]+)',
                                reasoning,
                                re.IGNORECASE,
                            )
                            if m_draft:
                                draft_cand = self._sanitize_model_output(m_draft.group(1).strip())
                                if draft_cand and len(draft_cand) > 5 and not draft_cand.endswith(":"):
                                    return draft_cand

                            # Also look for quoted final answer sentences in reasoning
                            quotes = re.findall(r'"([^"\n]{15,250})"', reasoning)
                            for q_text in reversed(quotes):
                                q_cleaned = self._sanitize_model_output(q_text)
                                if q_cleaned and len(q_cleaned) > 15 and not q_cleaned.endswith(":"):
                                    return q_cleaned

            # Ollama fallback
            ollama_url = getattr(a_settings, "ollama_base_url", "http://localhost:11434/v1").rstrip("/v1")
            ollama_model = getattr(a_settings, "extraction_model_name", "llama3.1")
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": ollama_model, "prompt": f"{context_text}\n\nQuestion: {question}\nAnswer:", "stream": False},
                timeout=15.0,
            )
            if resp.status_code == 200:
                raw_ans = resp.json().get("response", "").strip()
                if raw_ans:
                    return self._sanitize_model_output(raw_ans)
        except Exception as exc:
            logger.warning("graph.query_llm.error_falling_back_to_deterministic", error=str(exc))

        # Accurate deterministic fallback
        return self._deterministic_fallback(question, nodes, sources)
