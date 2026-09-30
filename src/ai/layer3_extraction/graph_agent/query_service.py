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
    "when", "why", "how", "associated",
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
        # Universal cross-domain synonym expansion for common question intents
        SYNONYMS: Dict[str, List[str]] = {
            # Identity & People
            "name": ["person", "candidate", "applicant", "patient", "employee", "author", "client", "customer", "vendor", "full_name", "contact_person"],
            "person": ["candidate", "applicant", "employee", "patient", "client", "customer", "doctor", "signatory", "guardian", "officer"],
            "candidate": ["applicant", "person", "employee", "full_name", "name"],
            "applicant": ["candidate", "person", "full_name", "name"],
            "patient": ["person", "patient_name", "insured", "policyholder"],
            "doctor": ["physician", "consultant", "provider", "specialist", "surgeon", "treating_doctor", "reference_doctor", "dr"],
            "doctors": ["doctor", "physician", "consultant", "provider", "specialist", "dr"],
            "physician": ["doctor", "consultant", "specialist"],
            "consultant": ["doctor", "physician", "specialist", "advisor"],
            "vendor": ["supplier", "seller", "provider", "merchant", "company", "issuer"],
            "customer": ["client", "buyer", "recipient", "purchaser", "consumer", "insured", "patient"],
            "employer": ["company", "organization", "workplace", "firm"],
            "employee": ["staff", "worker", "candidate", "person"],
            "age": ["dob", "birth", "years", "date_of_birth"],
            # Organizations & Institutions
            "company": ["organization", "vendor", "issuer", "bank", "hospital", "firm", "insurer", "institution", "agency", "supplier"],
            "organization": ["company", "institution", "agency", "firm", "hospital", "bank", "vendor"],
            "hospital": ["clinic", "healthcare", "facility", "provider", "institution"],
            "bank": ["financial_institution", "lender", "bank_name", "issuer"],
            "insurer": ["insurance_company", "sponsor", "tpa", "underwriter", "carrier"],
            "insurance": ["insurance_company", "sponsor", "tpa", "policy", "coverage", "carrier"],
            # Financials, Invoices, Billing
            "amount": ["total", "sum", "cost", "price", "fee", "rate", "subtotal", "balance", "charge", "value", "payable", "paid"],
            "total": ["gross", "net", "subtotal", "grand_total", "total_amount", "total_bill", "overall_amount", "sum"],
            "bill": ["invoice", "statement", "receipt", "charge", "total_bill", "fee"],
            "invoice": ["bill", "statement", "receipt", "tax_invoice", "proforma"],
            "claimed": ["claim", "claimed_amount", "total_bill", "requested_amount"],
            "claim": ["claimed_amount", "settlement", "sanctioned_amount", "insurance_claim"],
            "payable": ["due", "net_payable", "amount_due", "balance_due", "paid_by_insured", "patient_payable", "outstanding"],
            "pocket": ["out_of_pocket", "patient_payable", "copay", "non_payable", "deduction", "uncovered"],
            "paid": ["settled", "disbursed", "remitted", "cleared", "amount_paid"],
            "tax": ["gst", "vat", "sales_tax", "cgst", "sgst", "igst", "service_tax", "cess", "withholding"],
            "discount": ["rebate", "concession", "reduction", "deduction"],
            "deduction": ["deducted", "deductions", "non_medical", "non_payable", "disallowed", "copay", "withholding"],
            "deductions": ["deducted", "deduction", "non_medical", "non_payable", "disallowed", "copay", "withholding"],
            "deducted": ["deduction", "deductions", "disallowed", "non_payable"],
            "sanctioned": ["approved", "settled", "sanctioned_amount", "authorized_amount", "final_claim"],
            "approved": ["sanctioned", "authorized", "settled", "granted"],
            "fee": ["cost", "charge", "price", "rate", "tariff", "tuition", "premium"],
            "salary": ["wages", "compensation", "stipend", "pay", "ctc", "remuneration"],
            # Identifiers & Numbers
            "number": ["id", "identifier", "code", "reference", "ref_no", "no"],
            "id": ["identifier", "number", "code", "reference"],
            "policy": ["policy_number", "insurance_policy", "policy_no", "contract_number"],
            "account": ["account_number", "acc_no", "bank_account"],
            # Dates & Temporal
            "date": ["time", "period", "day", "month", "year", "timestamp", "validity", "expiry"],
            "expiry": ["expiration", "valid_until", "validity", "due_date"],
            # Items & Details
            "item": ["line_item", "product", "description", "service", "goods", "article"],
            "items": ["line_items", "products", "services", "goods", "articles"],
            "service": ["procedure", "operation", "treatment", "item", "offering"],
            "skills": ["technologies", "competencies", "tools", "expertise", "qualifications"],
            "education": ["degree", "qualification", "university", "college", "school", "academic"],
            "experience": ["work_experience", "employment", "history", "career"],
            "address": ["location", "place", "city", "state", "country", "pin", "zipcode", "residence"],
        }
        syn_tokens: Set[str] = set()
        for tok in tokens:
            if tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[tok])
            stem_tok = self._clean_stem(tok)
            if stem_tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[stem_tok])

        # Dynamic Interrogative Target Focus:
        # Detect target focus by scanning query tokens & expanded synonyms against all entity types and node labels present in the graph memory.
        target_focus_types: Set[str] = set()
        target_focus_labels: Set[str] = set()

        for t in getattr(graph, "_type_index", {}).keys():
            t_norm = self._normalize(t)
            if t_norm in norm_q or any(tok in t_norm or self._clean_stem(tok) in t_norm for tok in tokens):
                target_focus_types.add(t_norm)
            for syn in syn_tokens:
                if syn in t_norm or t_norm in syn:
                    target_focus_types.add(t_norm)

        for n in graph._nodes.values():
            lbl_norm = self._normalize(n.label)
            if any(tok in lbl_norm or self._clean_stem(tok) in lbl_norm for tok in tokens if len(tok) >= 3):
                target_focus_labels.add(lbl_norm)

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

            # Target focus boost: ensures the actual entity being asked for is prioritized
            if target_focus_types or target_focus_labels:
                if type_norm in target_focus_types or any(tfl in lbl_norm for tfl in target_focus_labels) or (schema_field and any(tfl in schema_field for tfl in target_focus_labels)):
                    score += 5.0
                    matched_tokens += 2

            # Substantial exact value match in question
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

        # Cluster seeds to prevent any single entity type from dominating all slots
        seen_ids = set()
        seeds: List[GraphNode] = []
        type_counts: Dict[str, int] = {}

        # Pass 1: Add target focus nodes first
        if target_focus_types or target_focus_labels:
            for cand in scored_candidates:
                node = cand[6]
                ntype = node.type.lower()
                nlbl = node.label.lower()
                if node.id not in seen_ids:
                    if ntype in target_focus_types or any(tfl in nlbl for tfl in target_focus_labels):
                        seen_ids.add(node.id)
                        seeds.append(node)
                        type_counts[ntype] = type_counts.get(ntype, 0) + 1
                        if len(seeds) >= max_seeds:
                            break

        # Pass 2: Add other high-scoring seeds with at most 2 per entity type
        for cand in scored_candidates:
            node = cand[6]
            ntype = node.type.lower()
            if node.id not in seen_ids:
                if type_counts.get(ntype, 0) < 2 or len(seeds) < 3:
                    seen_ids.add(node.id)
                    seeds.append(node)
                    type_counts[ntype] = type_counts.get(ntype, 0) + 1
                    if len(seeds) >= max_seeds:
                        break

        # Pass 3: Fill any remaining slots
        if len(seeds) < max_seeds:
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
        max_evidence: int = 16,
        max_sources: int = 8,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved nodes, relationships, and provenances into a high-signal, compact context for the LLM.
        Ensures entity diversity so citations aren't crowded out by a single high-frequency entity.
        """
        lines = ["=== RELEVANT GRAPH ENTITIES ==="]
        for n in nodes[:15]:
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

        # Pass 1: Select 1 top evidence span per node to guarantee entity diversity
        for n in nodes:
            if ev_count >= max_evidence:
                break
            sorted_ev = sorted(
                n.evidence or [],
                key=lambda ev: (-getattr(ev, "confidence", 0.5), getattr(ev, "page_number", 999)),
            )
            for ev in sorted_ev[:1]:
                text_clean = ev.text.strip()
                key = (ev.page_number, text_clean)
                if text_clean and key not in seen_ev:
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

        # Pass 2: Select a second evidence span for key nodes up to limits
        for n in nodes:
            if ev_count >= max_evidence:
                break
            sorted_ev = sorted(
                n.evidence or [],
                key=lambda ev: (-getattr(ev, "confidence", 0.5), getattr(ev, "page_number", 999)),
            )
            for ev in sorted_ev[1:2]:
                text_clean = ev.text.strip()
                key = (ev.page_number, text_clean)
                if text_clean and key not in seen_ev:
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

        # Edge evidence
        for e in edges:
            if ev_count >= max_evidence:
                break
            if e.evidence:
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
            for t, node_ids in getattr(graph, "_type_index", {}).items():
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
                r"(?:(?:So|Therefore|In summary|Hence|In conclusion),?\s*([^\n]+))",
                content,
                re.IGNORECASE,
            )
            if m_concl:
                candidate = m_concl.group(1).strip()
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
            or any(bad in clean_out.lower() for bad in ["critical: do not", "internal monologue", "scratchpad", "system prompt", "knowledge graph evidence:", "based on the prompt", "e.g. 'source: page", "e.g. \"source: page", "found (e.g."])
            or clean_out.lower().startswith("found (")
        ):
            return ""
        return clean_out

    def _deterministic_fallback(
        self,
        question: str,
        nodes: List[GraphNode],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Universal, graph-grounded deterministic answering when LLM generation is unavailable."""
        q = question.lower()
        tokens = self._extract_query_tokens(question)
        stems = {self._clean_stem(t) for t in tokens}

        if not nodes:
            return "The requested information could not be found in the document graph."

        valid_nodes = [n for n in nodes if not self._is_garbage_node(n)]
        if not valid_nodes:
            return "The requested information could not be found in the document graph."

        # Dynamic scoring of nodes based on query token & stem overlap
        scored: List[Tuple[float, GraphNode]] = []
        for n in valid_nodes:
            val_norm = self._normalize(n.value)
            lbl_norm = self._normalize(n.label)
            type_norm = self._normalize(n.type)
            score = 0.0

            for tok in tokens:
                if len(tok) >= 3:
                    if tok in val_norm:
                        score += 2.0
                    if tok in lbl_norm:
                        score += 1.8
                    if tok in type_norm:
                        score += 0.8
            for st in stems:
                if len(st) >= 3:
                    if st in lbl_norm:
                        score += 1.2
                    if st in val_norm:
                        score += 1.0

            # Match against evidence text
            for ev in (n.evidence or []):
                ev_norm = self._normalize(getattr(ev, "text", ""))
                for tok in tokens:
                    if len(tok) >= 4 and tok in ev_norm:
                        score += 0.5

            if score > 0.0:
                scored.append((score, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [s[1] for s in scored] if scored else valid_nodes

        # 1. Multi-entity list queries (e.g. "which all doctors", "what are the items", "list all skills...")
        if any(w in q for w in ["which all", "all", "list", "what are the", "what are all"]):
            primary_type = top_nodes[0].type
            matching_list = [n for n in top_nodes if n.type == primary_type]
            seen_values = set()
            unique_list = []
            for n in matching_list:
                v_clean = n.value.strip().lower()
                if v_clean not in seen_values:
                    seen_values.add(v_clean)
                    page = n.source_pages[0] if n.source_pages else 1
                    lbl = n.label.replace("_", " ")
                    unique_list.append(f"{n.value} ({lbl}, Source: Page {page})")
            if unique_list:
                return f"The {primary_type.lower()}s recorded in the document are: {'; '.join(unique_list)}."

        # 2. Comparative / multi-part queries (e.g. payable from X and from Y, totals and subtotals)
        if len(top_nodes) >= 2 and any(w in q for w in ["and", "from", "between", "payable", "versus", "vs"]):
            items_formatted = []
            seen_v = set()
            for n in top_nodes[:3]:
                if n.value.strip().lower() not in seen_v:
                    seen_v.add(n.value.strip().lower())
                    page = n.source_pages[0] if n.source_pages else 1
                    items_formatted.append(f"{n.label.replace('_', ' ')} is {n.value} (Source: Page {page})")
            if items_formatted:
                return f"Based on the document: {', and '.join(items_formatted)}."

        # 3. Best single entity match
        best_node = top_nodes[0]
        page = best_node.source_pages[0] if best_node.source_pages else 1
        lbl_clean = best_node.label.replace("_", " ")
        return f"The {lbl_clean} is {best_node.value}. (Source: Page {page})"

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
            or "bedrock"
        ).lower()
        bedrock_region = (
            getattr(app_settings, "bedrock_region", None)
            or getattr(a_settings, "bedrock_region", None)
            or os.getenv("IDP_BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or "ap-south-1"
        )
        bedrock_model_id = (
            getattr(app_settings, "bedrock_model_id", None)
            or getattr(a_settings, "bedrock_model_id", None)
            or os.getenv("IDP_BEDROCK_MODEL_ID")
            or os.getenv("BEDROCK_MODEL_ID")
            or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
        )

        system_prompt = (
            "You are a factual, concise Document QA assistant. "
            "Answer the user's question directly, accurately, and concisely in 1 or 2 plain sentences based strictly on the provided Document Knowledge Graph evidence. "
            "Always cite the exact source page number where the answer is found (e.g. 'Source: Page 4').\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. Grounding: Answer strictly based on the provided entities, properties, values, and relationships. Never guess or hallucinate.\n"
            "2. Entity Disambiguation: When multiple entities, amounts, parties (e.g. patient vs insurer, buyer vs vendor, employer vs candidate), or line items/deductions are asked, clearly distinguish each respective role and amount.\n"
            "3. Multi-Entity Listings: When asked to list items, persons, services, or figures (e.g. 'which all', 'what are the'), list all distinct matching entities found in the evidence.\n"
            "4. Conciseness: Do NOT output thinking, reasoning steps, internal monologue, numbered analysis lists, or scratchpads. Provide ONLY the final direct answer."
        )
        user_prompt = f"Document Knowledge Graph Evidence:\n{context_text}\n\nQuestion: {question}\nDirect Answer:"

        try:
            if backend == "bedrock" or (not api_key and bedrock_model_id):
                import boto3
                profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
                if profile:
                    b_client = boto3.Session(profile_name=profile, region_name=bedrock_region).client("bedrock-runtime", region_name=bedrock_region)
                else:
                    b_client = boto3.client("bedrock-runtime", region_name=bedrock_region)

                kwargs = {
                    "modelId": bedrock_model_id,
                    "system": [{"text": system_prompt}],
                    "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                    "inferenceConfig": {"temperature": 0.0, "maxTokens": 1500},
                }
                effort = getattr(app_settings, "bedrock_reasoning_effort", None) or getattr(a_settings, "bedrock_reasoning_effort", None) or "low"
                if effort and ("gpt-oss" in bedrock_model_id.lower() or "qdtz23c8eis1" in bedrock_model_id):
                    kwargs["additionalModelRequestFields"] = {"reasoning_effort": effort}

                try:
                    b_resp = b_client.converse(**kwargs)
                except Exception:
                    kwargs.pop("additionalModelRequestFields", None)
                    b_resp = b_client.converse(**kwargs)

                content = b_resp.get("output", {}).get("message", {}).get("content", [])
                text = next((block["text"] for block in content if isinstance(block, dict) and "text" in block), "")
                if text:
                    cleaned = self._sanitize_model_output(text)
                    if cleaned and len(cleaned) > 5:
                        return cleaned

            elif (backend == "sarvam" or api_key) and api_key:
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
        except Exception as exc:
            logger.warning("graph.query_llm.error_falling_back_to_deterministic", error=str(exc))

        # Universal accurate deterministic fallback
        return self._deterministic_fallback(question, nodes, sources)
