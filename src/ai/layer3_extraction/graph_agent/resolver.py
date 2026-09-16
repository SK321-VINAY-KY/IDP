"""
File: resolver.py
Purpose: Synthesize final dynamic Pydantic schema from GraphMemory.
Hardened to:
- Distinguish ASSERTED (primary/direct evidence), SUPPORTED INFERRED (supporting evidence),
  and UNCERTAIN (candidate/unverified evidence) in evidence formatting.
- Prevent UNCERTAIN facts from silently contaminating authoritative schema output.
- Guard fallback reconciliation from treating uncertain nodes as authoritative candidates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_graph_evidence_for_resolution(graph: GraphMemory, schema_fields: List[Dict[str, str]]) -> str:
    """
    Format all entities, values, relationships, and provenances from the graph
    into a structured prompt context for final schema resolution.
    Explicitly labels evidence status so the LLM distinguishes authoritative from unverified facts.
    """
    lines = [
        "=== EXTRACTED GRAPH ENTITIES ===",
        "NOTE: Prefer [ASSERTED] primary evidence and [SUPPORTED INFERRED] evidence.",
        "Do NOT treat [UNCERTAIN - UNVERIFIED] candidates as authoritative facts unless supported by explicit evidence.\n",
    ]
    nodes = list(graph._nodes.values())
    if not nodes:
        lines.append("No entities were extracted into graph memory.")
    else:
        for n in nodes:
            pages = f"Page {','.join(map(str, n.source_pages))}" if n.source_pages else "Page ?"
            ev = f' | Evidence: "{n.evidence[0].text}"' if n.evidence else ""
            category_tag = f" [{n.category}]"

            status_upper = (n.status or "ASSERTED").upper()
            if status_upper == "ASSERTED":
                status_label = "[ASSERTED - Primary Evidence]"
            elif status_upper == "INFERRED":
                status_label = f"[SUPPORTED INFERRED - Conf: {n.resolution_confidence:.2f}]"
            elif status_upper == "UNCERTAIN":
                status_label = "[UNCERTAIN - UNVERIFIED Candidate]"
            else:
                status_label = f"[{status_upper}]"

            lines.append(
                f"- [{n.id}] {status_label} {n.type} ({n.label}): \"{n.value}\" ({pages}){category_tag}{ev}"
            )

    lines.append("\n=== ENTITY RELATIONSHIPS ===")
    edges = list(graph._edges.values())
    if not edges:
        lines.append("No explicit relationships recorded.")
    else:
        for e in edges:
            src = graph.get_node(e.source_node)
            tgt = graph.get_node(e.target_node)
            src_str = f"{src.value} ({src.type})" if src else e.source_node
            tgt_str = f"{tgt.value} ({tgt.type})" if tgt else e.target_node
            ev = f' | Evidence: "{e.evidence}"' if e.evidence else ""

            e_status = (e.status or "ASSERTED").upper()
            if e_status == "ASSERTED":
                rel_status = "ASSERTED"
            elif e_status == "INFERRED":
                rel_status = f"SUPPORTED INFERRED (Conf: {e.relationship_confidence:.2f})"
            elif e_status == "UNCERTAIN":
                rel_status = "UNCERTAIN - UNVERIFIED"
            else:
                rel_status = e_status

            lines.append(
                f"- {src_str} --[{e.relationship} (Status: {rel_status})]-> {tgt_str} (Page {e.source_page}){ev}"
            )

    return "\n".join(lines)


def resolve_schema_from_graph(
    graph: GraphMemory,
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
) -> BaseModel:
    """
    Produce final hydrated dynamic Pydantic BaseModel instance from GraphMemory.
    Does NOT reread or concatenate raw page markdown — graph memory is the single source of truth.
    Authoritative resolution preferentially consumes ASSERTED and SUPPORTED INFERRED evidence,
    and protects output from UNCERTAIN fact contamination.
    """
    schema_fields = [
        {"name": k, "description": v.description or k}
        for k, v in schema.model_fields.items()
    ]
    field_names = set(schema.model_fields.keys())

    evidence_text = format_graph_evidence_for_resolution(graph, schema_fields)
    logger.info(
        "graph.schema_resolution.start",
        fields=list(field_names),
        evidence_lines=len(evidence_text.splitlines()),
    )

    try:
        result = llm.resolve_schema_from_graph(evidence_text, schema)
    except Exception as exc:
        logger.warning("graph.schema_resolution.llm_failed_fallback_to_graph", error=str(exc))
        try:
            default_instance = schema()
        except Exception:
            default_instance = None
        default_values: Dict[str, Any] = default_instance.model_dump() if default_instance is not None else {}
        result = schema.model_validate(default_values)

    res_dict = result.model_dump()
    merged = dict(res_dict)

    # Post-resolution reconciliation:
    # If any schema field is empty in the LLM output, look for authoritative graph matches.
    # Only nodes explicitly tagged with schema_field_name matching the target field are used.
    # UNCERTAIN nodes must NOT contaminate authoritative output.
    for fname in field_names:
        if merged.get(fname) in ("", None):
            # Search for nodes explicitly tagged with this field name (excluding UNCERTAIN)
            for node in graph._nodes.values():
                if node.status == "UNCERTAIN":
                    continue
                if node.properties.get("schema_field_name") == fname and node.value:
                    merged[fname] = node.value
                    logger.debug("graph.resolver.reconciled_tagged", field=fname, value=node.value)
                    break

    if merged != res_dict:
        result = schema.model_validate(merged)

    logger.info(
        "graph.extraction.completed",
        resolved_fields={k: v for k, v in result.model_dump().items() if v not in ("", None)},
        graph_stats=graph.stats(),
    )
    return result
