"""
File: extractor.py
Purpose: Layer 3 Targeted Extraction (Phase C).
         Extracts candidate nodes and edges from navigated pages using LLM,
         with per-page heuristic regex fallback.
         Write-only extraction per page; zero cross-page reads/merges.

Owner: engineer-a@idp-pilot
Updated: 2026-09-07
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.config.ontology import OntologyConfig, load_ontology
from src.utils.logger import get_logger

logger = get_logger(__name__)


# NOTE: Identity resolution (Person, Organization) and identifier matching (Identifier)
# are deliberately NOT extended to cover type="Other" nodes in this pass. Dedup logic
# tuned for known entity shapes does not generalize to open-ended findings; handling
# duplicates for "Other" nodes is deferred until volume demonstrates a need.
class CandidateNode(BaseModel):
    node_id: str
    type: str  # One of the baseline ontology categories, or "Other"
    label: str  # Extracted text value
    source_page: int
    evidence: str
    confidence: float = 1.0
    importance: Literal["high", "medium", "low"] = "medium"
    importance_reason: str = ""
    subtype: Optional[str] = None


class CandidateEdge(BaseModel):
    source: str  # source node_id
    relationship: str  # e.g. PERFORMED_BY, HAS_AMOUNT, etc.
    target: str  # target node_id
    source_page: int
    evidence: str
    confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED | INFERRED | UNCERTAIN

def extract_candidate_graph(
    pages_md: Union[List[str], List[Dict[str, Any]], Sequence[Any]],
    target_pages: Optional[List[int]] = None,
    llm: Optional[ExtractionLLMClient] = None,
    max_workers: int = 5,
    ontology: Optional[OntologyConfig] = None,
    extraction_focus: Optional[List[str]] = None,
) -> Tuple[List[CandidateNode], List[CandidateEdge]]:
    """
    Phase C Targeted Extraction:
    Extracts candidate nodes and edges from navigated pages using the LLM client,
    with per-page heuristic regex fallback.
    Per graph-memory POC Phase 1: STRICTLY WRITE-ONLY.
    No cross-page reads, no cross-page merging during this step.

    Args:
        pages_md:         List of page dicts or strings.
        target_pages:     Optional list of page numbers to extract from.
                          If None, extracts from all pages.
        llm:              Optional LLM client.
        max_workers:      Worker threads for concurrent LLM extraction.
        ontology:         Optional loaded OntologyConfig for dynamic category validation.
        extraction_focus: Optional list of free-text hints steering extraction.

    Returns:
        Tuple of (candidate_nodes, candidate_edges).
    """
    if ontology is None:
        ontology = load_ontology()
    valid_categories: Set[str] = set(ontology.category_names())

    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    if target_pages is not None:
        target_set = set(target_pages)
        pages = [p for p in pages if p["page_number"] in target_set]

    # Filter out empty pages
    pages = [p for p in pages if p.get("markdown", "").strip()]

    # Container for raw extraction outputs per page: {pnum: (raw_nodes, raw_edges, default_conf)}
    page_raw_results: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]] = {}

    if llm is not None and hasattr(llm, "extract_page_ontology") and len(pages) > 0:
        active_llm: ExtractionLLMClient = llm
        logger.info(
            "layer3.targeted_extraction.start_llm",
            pages_count=len(pages),
            max_workers=max_workers,
            has_focus=bool(extraction_focus),
        )

        def _fetch_page(page_data: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], float]:
            pnum = int(page_data["page_number"])
            pmd = str(page_data["markdown"])
            pconf = round(float(page_data.get("confidence", 0.95)), 3)
            try:
                res = active_llm.extract_page_ontology(
                    page_md=pmd,
                    page_number=pnum,
                    extraction_focus=extraction_focus,
                )
                r_nodes = res.get("nodes", []) if isinstance(res, dict) else []
                r_edges = res.get("edges", []) if isinstance(res, dict) else []
                if r_nodes:
                    return pnum, r_nodes, r_edges, pconf
            except Exception as exc:
                logger.warning("layer3.targeted_extraction.page_llm_failed", page=pnum, error=str(exc))
            return pnum, [], [], pconf

        effective_workers = min(max_workers, len(pages))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_page = {executor.submit(_fetch_page, p): p["page_number"] for p in pages}
            for fut in as_completed(future_to_page):
                pnum, r_nodes, r_edges, pconf = fut.result()
                page_raw_results[pnum] = (r_nodes, r_edges, pconf)

    all_nodes: List[CandidateNode] = []
    all_edges: List[CandidateEdge] = []
    node_counter = 1
    llm_pages_count = 0
    fallback_pages_count = 0
    llm_nodes_count = 0
    fallback_nodes_count = 0
    llm_edges_count = 0
    fallback_edges_count = 0

    # Deterministically process pages in ascending order of page number
    sorted_pages = sorted(pages, key=lambda p: p["page_number"])

    for page_data in sorted_pages:
        pnum = page_data["page_number"]
        pmd = page_data["markdown"]
        pconf = round(float(page_data.get("confidence", 0.95)), 3)

        raw_nodes, raw_edges, _ = page_raw_results.get(pnum, ([], [], pconf))

        if raw_nodes:
            # Successfully extracted via LLM
            llm_pages_count += 1
            page_nodes: List[CandidateNode] = []
            page_edges: List[CandidateEdge] = []
            label_to_id: Dict[str, str] = {}

            for item in raw_nodes:
                if not isinstance(item, dict):
                    continue
                raw_cat = str(item.get("category", "Item")).strip()
                raw_subtype = item.get("subtype")
                if raw_cat in valid_categories:
                    cat = raw_cat
                    subtype = str(raw_subtype).strip() if raw_subtype else None
                else:
                    cat = "Other"
                    subtype = str(raw_subtype).strip() if raw_subtype else raw_cat

                raw_imp = str(item.get("importance", "medium")).lower().strip()
                imp: Literal["high", "medium", "low"] = (
                    raw_imp if raw_imp in ("high", "medium", "low") else "medium"
                )
                imp_reason = str(item.get("importance_reason", "")).strip()
                lbl = str(item.get("label", "")).strip()
                evi = str(item.get("evidence", lbl)).strip()[:200]
                if not lbl:
                    continue
                nid = f"n_{node_counter:03d}"
                node_counter += 1
                page_nodes.append(CandidateNode(
                    node_id=nid,
                    type=cat,
                    label=lbl,
                    source_page=pnum,
                    evidence=evi,
                    confidence=pconf,
                    importance=imp,
                    importance_reason=imp_reason,
                    subtype=subtype,
                ))
                label_to_id[lbl.lower()] = nid
                # Also index bare words/digits for matching
                label_to_id[lbl.lower().replace(":", "").strip()] = nid

            for e_item in raw_edges:
                if not isinstance(e_item, dict):
                    continue
                src_lbl = str(e_item.get("source_label", "")).lower().strip()
                tgt_lbl = str(e_item.get("target_label", "")).lower().strip()
                rel = str(e_item.get("relationship", "RELATES_TO")).strip().upper()
                evi = str(e_item.get("evidence", f"{src_lbl} {rel} {tgt_lbl}")).strip()[:200]

                src_id = label_to_id.get(src_lbl) or label_to_id.get(src_lbl.replace(":", "").strip())
                tgt_id = label_to_id.get(tgt_lbl) or label_to_id.get(tgt_lbl.replace(":", "").strip())
                if src_id and tgt_id and src_id != tgt_id:
                    page_edges.append(CandidateEdge(
                        source=src_id,
                        relationship=rel,
                        target=tgt_id,
                        source_page=pnum,
                        evidence=evi,
                        confidence=pconf,
                        status="ASSERTED",
                    ))

            llm_nodes_count += len(page_nodes)
            llm_edges_count += len(page_edges)
            all_nodes.extend(page_nodes)
            all_edges.extend(page_edges)
        else:
            # Heuristic / regex fallback
            fallback_pages_count += 1
            page_nodes, page_edges = _extract_nodes_from_page_text(
                markdown=pmd,
                page_number=pnum,
                start_counter=node_counter,
                default_conf=pconf,
            )
            node_counter += len(page_nodes)
            fallback_nodes_count += len(page_nodes)
            fallback_edges_count += len(page_edges)
            all_nodes.extend(page_nodes)
            all_edges.extend(page_edges)

    logger.info(
        "layer3.targeted_extraction.completed",
        pages_extracted=[p["page_number"] for p in sorted_pages],
        total_nodes=len(all_nodes),
        total_edges=len(all_edges),
        llm_pages_count=llm_pages_count,
        fallback_pages_count=fallback_pages_count,
        llm_nodes_count=llm_nodes_count,
        fallback_nodes_count=fallback_nodes_count,
        llm_edges_count=llm_edges_count,
        fallback_edges_count=fallback_edges_count,
    )
    return all_nodes, all_edges


def _extract_nodes_from_page_text(
    markdown: str,
    page_number: int,
    start_counter: int = 1,
    default_conf: float = 0.95,
) -> Tuple[List[CandidateNode], List[CandidateEdge]]:
    """
    Extract candidate ontology nodes and edges from a single page in write-only mode.
    """
    nodes: List[CandidateNode] = []
    edges: List[CandidateEdge] = []
    c = start_counter

    # TODO(follow-up): Domain-specific regex heuristics (Hinduja, Hayaat, Kochhar, etc.) are hardcoded for sample data. Refactor to generic pattern extractors in a separate follow-up.
    def add_node(
        cat: str,
        label: str,
        evidence: str,
        conf: Optional[float] = None,
        importance: Literal["high", "medium", "low"] = "medium",
        importance_reason: str = "",
        subtype: Optional[str] = None,
    ) -> str:
        nonlocal c
        nid = f"n_{c:03d}"
        c += 1
        node_conf = default_conf if conf is None else conf
        nodes.append(CandidateNode(
            node_id=nid,
            type=cat,
            label=label.strip(),
            source_page=page_number,
            evidence=evidence.strip()[:200],
            confidence=node_conf,
            importance=importance,
            importance_reason=importance_reason,
            subtype=subtype,
        ))
        return nid

    def add_edge(src: str, rel: str, tgt: str, evidence: str, conf: float = 0.95, status: str = "ASSERTED") -> None:
        edges.append(CandidateEdge(
            source=src,
            relationship=rel,
            target=tgt,
            source_page=page_number,
            evidence=evidence.strip()[:200],
            confidence=conf,
            status=status,
        ))

    # --- 1. Extract Identifiers ---
    id_patterns = [
        (r"(?:Hs\s*No|HS\s*No|Hs\s*Number)\s*[:.]?\s*([A-Za-z0-9/-]+)", "HS No"),
        (r"(?:Adm\.?\s*No|Admission\s*No)\s*[:.]?\s*([A-Za-z0-9/-]+)", "Admission No"),
        (r"(?:Bill\s*No\.?|Bill\s*Number)\s*[:.]?\s*([A-Za-z0-9/-]+)", "Bill No"),
        (r"(?:Voucher\s*No)\s*[:.]?\s*([A-Za-z0-9/-]+)", "Voucher No"),
        (r"(?:AL\s*Number|AL\s*No\.?)\s*[:.]?\s*([A-Za-z0-9/-]+)", "AL Number"),
        (r"(?:UHID\s*Number|UH\s*ID\s*No\.?|Rohini\s*ID)\s*[:.]?\s*([A-Za-z0-9/-]+)", "UHID"),
        (r"(?:IH9\s*NO|IH9\s*No)\s*[:-]?\s*([A-Za-z0-9/-]+)", "Chemist Reg"),
        (r"(?:GST\.?NO|GSTIN)\s*[:-]?\s*([A-Za-z0-9]+)", "GSTIN"),
        (r"(?:Contact\s*(?:Phone\s*)?No\.?|Mobile\s*No\.?|NOBELE\s*NO\.?|N081LE)\s*[:.]?\s*(?:1\)\s*)?([0-9]{7,12})", "Phone"),
        (r"(?:^|\n)\s*([0-9]{10})\s*(?:\n\s*[0-9]{10})?\s*\n\s*Contact\s*No", "Phone"),
    ]
    extracted_ids: List[str] = []
    for pattern, id_type in id_patterns:
        for m in re.finditer(pattern, markdown, re.IGNORECASE):
            val = m.group(1).strip()
            if len(val) >= 3:
                nid = add_node(
                    "Identifier",
                    f"{id_type}: {val}",
                    m.group(0),
                    importance="high",
                    importance_reason=f"matched identifier pattern for {id_type}",
                    subtype=id_type,
                )
                extracted_ids.append(nid)

    # --- 2. Extract Amounts ---
    amount_patterns = [
        r"\b(?:Rs\.?|INR|IRS)\s*[:.-]*\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?)\b",
        r"\b(?:AOTAL\s*ANT|TOTAL\s*AMT|NET\s*AMT|Total\s*Amount|Total\s*Amt|for\s*Amt|BHOUNT|AMOUNT|AMT)\s*[:.-]*\s*(?:INR|Rs\.?|IRS)?\s*([0-9]{1,5}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?)\b",
        r"\b(?:Rs\s+|INR\s+|IRS\s+)([0-9]{3,6})\b",
        r"\b([0-9]{3,6}\.[0-9]{2})\b",
    ]
    extracted_amounts: List[str] = []
    seen_amounts = set()
    for pat in amount_patterns:
        for m in re.finditer(pat, markdown, re.IGNORECASE):
            val = m.group(1).strip().replace(",", "")
            if val and val not in seen_amounts:
                try:
                    num_val = float(val)
                    if num_val > 0:
                        seen_amounts.add(val)
                        nid = add_node(
                            "Amount",
                            f"INR {val}",
                            m.group(0),
                            importance="high",
                            importance_reason="matched primary amount pattern",
                        )
                        extracted_amounts.append(nid)
                except ValueError:
                    pass

    # --- 3. Extract Persons ---
    person_patterns = [
        (r"\b(?:Name\s*:\s*|Patient\s*Name\s*:\s*|Name\s*of\s*the\s*Patient\s*:\s*|PATEENT\s*:\s*)([A-Za-z \t]{3,35})", "Patient"),
        (r"\b(?:Doctor|DOCTOR|OOCTOR|Dr\.?)\b\s*[:.-]*\s*(?:OR\s+|DR\s+)?([A-Za-z \t.]{3,30})", "Doctor"),
        (r"\b(?:Card\s*Holder\s*[:.]?\s*|Received\s*with\s*thanks\s*:\s*)([A-Za-z \t]{3,30})", "Card Holder"),
    ]
    extracted_persons: List[str] = []
    for pat, role in person_patterns:
        for m in re.finditer(pat, markdown, re.IGNORECASE):
            raw_name = m.group(1).strip()
            # Clean up unwanted trailing prefixes or noise
            raw_name = re.sub(r"\s+(?:with\s+prior|appointment|consult|Age|Gender|DOA|Bed|Adm|AOD|PACX|COMP).*", "", raw_name, flags=re.IGNORECASE).strip()
            if len(raw_name) >= 3 and not any(kw in raw_name.upper() for kw in ["HOSPITAL", "CHEMIST", "CENTRE", "INSURANCE", "LIMITED", "WEST", "BANDRA", "ROAD"]):
                nid = add_node(
                    "Person",
                    raw_name,
                    m.group(0),
                    importance="high",
                    importance_reason=f"matched person pattern for {role}",
                    subtype=role,
                )
                extracted_persons.append(nid)

    # --- 4. Extract Organizations ---
    org_patterns = [
        r"(P\.\s*D\.\s*HINDUJA\s+HOSPITAL(?:\s*&\s*MEDICAL\s+RESEARCH\s+CENTRE)?)",
        r"(HAYAAT\s+CHEMIST)",
        r"(JEEVAN\s+AMBULANCE\s+SERVICE)",
        r"(ICICI\s+LOMBARD\s+GENERAL\s+INSURANCE\s+COMPA(?:NY)?)",
        r"(AXIS\s+BANK)",
    ]
    extracted_orgs: List[str] = []
    for pat in org_patterns:
        for m in re.finditer(pat, markdown, re.IGNORECASE):
            org_name = m.group(1).strip()
            nid = add_node(
                "Organization",
                org_name,
                m.group(0),
                importance="medium",
                importance_reason="matched issuing organization pattern",
            )
            extracted_orgs.append(nid)

    # --- 5. Extract Document / Record Nodes ---
    doc_patterns = [
        r"(DISCHARGE\s+SUMMARY)",
        r"(IN-Patient\s+Deposit\s+Receipt)",
        r"(Authorization\s+Letter\s+to\s+the\s+Hospital)",
        r"(INPATIENT\s+FINAL\s+BILL)",
        r"(AMBULANCE\s+SERVICE)",
    ]
    extracted_docs: List[str] = []
    for pat in doc_patterns:
        for m in re.finditer(pat, markdown, re.IGNORECASE):
            doc_title = m.group(1).strip()
            nid = add_node(
                "Document/Record",
                doc_title,
                m.group(0),
                importance="medium",
                importance_reason="matched document type header pattern",
            )
            extracted_docs.append(nid)

    # --- 6. Extract Dates ---
    date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{2,4})\b"
    extracted_dates: List[str] = []
    for m in re.finditer(date_pattern, markdown, re.IGNORECASE):
        d_val = m.group(1).strip()
        nid = add_node(
            "Date",
            d_val,
            m.group(0),
            importance="medium",
            importance_reason="matched date pattern",
        )
        extracted_dates.append(nid)

    # --- Local intra-page CandidateEdges (ASSERTED) ---
    for doc_id in extracted_docs:
        for p_id in extracted_persons:
            add_edge(doc_id, "MENTIONS", p_id, "Document mentions person on page")
        for amt_id in extracted_amounts:
            add_edge(doc_id, "HAS_AMOUNT", amt_id, "Document states amount on page")
        for id_id in extracted_ids:
            add_edge(doc_id, "CONTAINS", id_id, "Document contains identifier on page")
        for d_id in extracted_dates:
            add_edge(doc_id, "HAS_DATE", d_id, "Document dated on page")

    for org_id in extracted_orgs:
        for doc_id in extracted_docs:
            add_edge(org_id, "CONTAINS", doc_id, "Organization issued document")

    return nodes, edges


