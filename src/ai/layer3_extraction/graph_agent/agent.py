"""
File: agent.py
Purpose: Graph Extraction Agent processing pages sequentially and maintaining contextual graph memory.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.graph_agent.evidence_resolver import resolve_evidence_to_blocks
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import PageDelta, UnresolvedRef
from src.ai.layer3_extraction.graph_agent.models import (
    Evidence,
    GraphEdge,
    GraphNode,
)
from src.ai.schemas.document_ir import Block
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_evidence_list(
    page_number: int,
    ent_evidence: str,
    ent_conf: float,
    blocks: Optional[List[Block]] = None,
) -> List[Evidence]:
    if not ent_evidence:
        return []
    block_ids, bbox = [], None
    if blocks:
        block_ids, bbox = resolve_evidence_to_blocks(ent_evidence, blocks)
    return [
        Evidence(
            page_number=page_number,
            text=ent_evidence,
            confidence=ent_conf,
            block_ids=block_ids,
            bbox=bbox,
        )
    ]


class GraphExtractionAgent:
    """
    Graph Extraction Agent.
    Considers the schema as a priority signal, but also captures contextually
    important entities, resolves cross-page references, and maintains traceable evidence.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        graph: GraphMemory,
        llm: ExtractionLLMClient,
    ) -> None:
        self.schema = schema
        self.graph = graph
        self.llm = llm

        # Cache target schema fields once
        self.schema_fields: List[Dict[str, str]] = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        self.schema_field_names: Set[str] = set(schema.model_fields.keys())
        self.llm_call_count = 0
        self.session_id_map: Dict[str, str] = {}

    async def build_page_delta_async(
        self,
        page_number: int,
        markdown: str,
        total_pages: int,
        context: List[Dict[str, Any]],
        blocks: Optional[List[Block]] = None,
    ) -> PageDelta:
        """
        Extract page graph asynchronously into an isolated PageDelta.
        STATELESS with respect to self.graph: MUST NOT mutate self.graph.
        Calls the synchronous LLM via asyncio.to_thread().
        """
        clean_md = (markdown or "").strip()
        if not clean_md:
            logger.debug("graph.page_async.skip_empty", page_number=page_number)
            return PageDelta(page_number=page_number)

        logger.info(
            "graph.page_async.process_start",
            page_number=page_number,
            total_pages=total_pages,
            schema_fields=list(self.schema_field_names),
            context_nodes_count=len(context),
        )
        self.llm_call_count += 1
        llm_output = await asyncio.to_thread(
            self.llm.extract_graph_from_page,
            page_md=clean_md,
            schema_fields=self.schema_fields,
            existing_nodes=context,
            page_number=page_number,
            total_pages=total_pages,
        )

        entities = llm_output.get("entities", [])
        relationships = llm_output.get("relationships", [])
        ref_resolutions = llm_output.get("reference_resolutions", [])

        delta_entities: List[GraphNode] = []
        delta_edges: List[GraphEdge] = []
        unresolved: List[UnresolvedRef] = []
        raw_id_map: Dict[str, str] = {}

        # 1. Parse Entities
        for idx, ent in enumerate(entities):
            raw_id = ent.get("id") or f"p{page_number}_e{idx + 1}"
            reused_id = ent.get("reused_node_id")
            ent_type = ent.get("type") or "Entity"
            ent_label = ent.get("label") or ent_type
            ent_value = str(ent.get("value") or "").strip()
            ent_evidence = str(ent.get("evidence") or "")
            ent_conf = float(ent.get("confidence", 1.0))
            is_schema = bool(ent.get("is_schema_field", False))
            schema_fname = ent.get("schema_field_name") or ""

            if not ent_value:
                continue

            if schema_fname in self.schema_field_names:
                is_schema = True

            category = "EXPLICIT" if is_schema else "CONTEXTUAL"

            ev_list = _build_evidence_list(page_number, ent_evidence, ent_conf, blocks)
            props: Dict[str, Any] = {"raw_id": raw_id}
            if schema_fname:
                props["schema_field_name"] = schema_fname
            if reused_id:
                props["reused_node_id"] = reused_id

            node = GraphNode(
                id=raw_id,
                type=ent_type,
                label=ent_label,
                value=ent_value,
                properties=props,
                aliases=list(ent.get("aliases", [])),
                evidence=ev_list,
                source_pages=[page_number],
                confidence=ent_conf,
                status="ASSERTED",
                category=category,
            )
            delta_entities.append(node)
            raw_id_map[raw_id] = raw_id
            if ent_value:
                raw_id_map[ent_value.lower()] = raw_id
            if ent_label:
                raw_id_map[ent_label.lower()] = raw_id

        # 2. Parse Relationships
        for edge_data in relationships:
            raw_src = str(edge_data.get("source_node", "")).strip()
            raw_tgt = str(edge_data.get("target_node", "")).strip()
            rel = str(edge_data.get("relationship", "RELATED_TO")).strip().upper() or "RELATED_TO"
            ev_text = str(edge_data.get("evidence", ""))
            edge_conf = float(edge_data.get("confidence", 1.0))
            edge_status = str(edge_data.get("status", "ASSERTED")).upper()
            if edge_status not in ("ASSERTED", "INFERRED", "UNCERTAIN"):
                edge_status = "ASSERTED"

            if raw_src and raw_tgt:
                edge_id = f"edge_{page_number}_{uuid.uuid4().hex[:6]}"
                graph_edge = GraphEdge(
                    id=edge_id,
                    source_node=raw_src,
                    target_node=raw_tgt,
                    relationship=rel,
                    evidence=ev_text,
                    source_page=page_number,
                    confidence=edge_conf,
                    status=edge_status,
                )
                delta_edges.append(graph_edge)

        # 3. Parse Reference Resolutions
        for ref in ref_resolutions:
            phrase = str(ref.get("phrase", "")).strip()
            target_id = str(ref.get("resolved_to_node_id", "")).strip()
            target_type = str(ref.get("target_type") or ref.get("type") or "").strip()
            rationale = str(ref.get("rationale", "")).strip()

            if not phrase and not target_id:
                continue

            # Check if reference is resolved locally to an entity in this delta
            resolved_local = (
                raw_id_map.get(target_id)
                or raw_id_map.get(target_id.lower())
                if target_id
                else None
            )

            # Check if reference points to a node in the initial context
            resolved_context: Optional[str] = (
                next((str(c["id"]) for c in context if c.get("id") == target_id), None)
                if target_id
                else None
            )

            canon_target = resolved_local or resolved_context
            if canon_target is not None:
                src_node_id = (
                    raw_id_map.get(phrase.lower())
                    or raw_id_map.get(phrase)
                    or f"ref_src_{page_number}_{uuid.uuid4().hex[:4]}"
                )
                delta_edges.append(
                    GraphEdge(
                        id=f"edge_ref_{page_number}_{uuid.uuid4().hex[:6]}",
                        source_node=src_node_id,
                        target_node=canon_target,
                        relationship="REFERS_TO",
                        evidence=rationale or phrase,
                        source_page=page_number,
                        confidence=0.9,
                        status="INFERRED",
                    )
                )
            else:
                unresolved_src = (
                    raw_id_map.get(phrase.lower())
                    or raw_id_map.get(phrase)
                    or phrase
                )
                if not target_type:
                    words = phrase.split()
                    target_type = words[-1].capitalize() if words else "Entity"

                unresolved.append(
                    UnresolvedRef(
                        source_node_id=unresolved_src,
                        phrase=phrase,
                        target_type=target_type,
                        page_number=page_number,
                        rationale=rationale,
                    )
                )

        return PageDelta(
            page_number=page_number,
            entities=delta_entities,
            relationships=delta_edges,
            unresolved_references=unresolved,
            raw_id_map=raw_id_map,
        )
