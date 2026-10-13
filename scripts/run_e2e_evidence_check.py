"""
End-to-end verification script for DocumentIR Phase 2 field-level evidence resolution.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.layer2_conversion.digital import convert_digital_page
from src.ai.schemas.page import PageOutput
from src.ai.layer3_extraction.page_loader import load_pages_with_confidence
from src.ai.layer3_extraction.extractor import extract_with_graph_memory_concurrent


def main():
    # 1. Real Layer 2 conversion producing real PageOutputs with DocumentIR blocks
    md1, conf1, blocks1 = convert_digital_page("tests/fixtures/Vinay_Resume.pdf", 1)
    md2, conf2, blocks2 = convert_digital_page("tests/fixtures/Vinay_Resume.pdf", 2)

    p1 = PageOutput(
        page_number=1,
        markdown=md1,
        confidence=conf1,
        engines_used=["docling"],
        escalated=False,
        escalation_attempts=0,
        low_confidence=False,
        blocks=blocks1,
    )
    p2 = PageOutput(
        page_number=2,
        markdown=md2,
        confidence=conf2,
        engines_used=["docling"],
        escalated=False,
        escalation_attempts=0,
        low_confidence=False,
        blocks=blocks2,
    )

    # 2. page_loader contracts
    pages_data = load_pages_with_confidence([p1, p2])
    print(f"PageLoader output: {len(pages_data)} pages")
    print(f"  Page 1 blocks: {len(pages_data[0]['blocks'])}")
    print(f"  Page 2 blocks: {len(pages_data[1]['blocks'])}")

    # 3. Target Schema
    class ResumeSchema(BaseModel):
        applicant_name: str = Field(description="Name of applicant", default="")
        skills: list[str] = Field(description="Technical skills", default_factory=list)
        education: str = Field(description="Education details", default="")

    # 4. Mock LLM that returns real graph extractions
    mock_llm = MagicMock()

    def mock_extract_graph(page_md, schema_fields, existing_nodes, page_number, total_pages):
        if page_number == 1:
            return {
                "entities": [
                    {
                        "id": "p1_e1",
                        "type": "Person",
                        "label": "Applicant",
                        "value": "Vinay K",
                        "evidence": "Motivated AIML student in CSE with hands-on experience in Deep Learning",
                        "confidence": 0.98,
                        "is_schema_field": True,
                        "schema_field_name": "applicant_name",
                    },
                    {
                        "id": "p1_e2",
                        "type": "Organization",
                        "label": "Company",
                        "value": "Rashi Peripherals Limited",
                        "evidence": "Rashi Peripherals Limited",
                        "confidence": 0.95,
                    },
                    {
                        "id": "p1_e3",
                        "type": "Role",
                        "label": "Internship",
                        "value": "Machine Learning(LLM/SLM) Intern",
                        "evidence": "Machine Learning(LLM/SLM) Intern - Bangalore, India",
                        "confidence": 0.95,
                    },
                    {
                        "id": "p1_e4",
                        "type": "Project",
                        "label": "ASR Project",
                        "value": "ASR model on Jetson Orin",
                        "evidence": "Developed and deployed an ASR (Automated Speech Recognition) model on Jetson Orin for real-time implementation.",
                        "confidence": 0.95,
                    },
                ],
                "relationships": [
                    {
                        "source_node": "p1_e1",
                        "target_node": "p1_e2",
                        "relationship": "WORKED_AT",
                        "evidence": "Rashi Peripherals Limited",
                    },
                ],
                "reference_resolutions": [],
            }
        else:
            return {
                "entities": [
                    {
                        "id": "p2_e1",
                        "type": "Skill",
                        "label": "SQL",
                        "value": "SQL",
                        "evidence": ": SQL, Python, Java, Machine Learning, NLP, Computer Vision,",
                        "confidence": 0.95,
                        "is_schema_field": True,
                        "schema_field_name": "skills",
                    },
                    {
                        "id": "p2_e2",
                        "type": "Skill",
                        "label": "Docker",
                        "value": "Docker",
                        "evidence": ": Docker, Kubernetes, MLflow, Jetson Nano/Orin, Tableau, Power BI",
                        "confidence": 0.95,
                    },
                    {
                        "id": "p2_e3",
                        "type": "Event",
                        "label": "Hackathon",
                        "value": "Techflix LA De Costa",
                        "evidence": "Won 4th Prize as Team Lead in Techflix LA De Costa, a National Level Hackathon hosted at college.",
                        "confidence": 0.95,
                    },
                ],
                "relationships": [],
                "reference_resolutions": [],
            }

    mock_llm.extract_graph_from_page.side_effect = mock_extract_graph
    mock_llm.resolve_schema_from_graph.return_value = ResumeSchema(
        applicant_name="Vinay K",
        skills=["SQL", "Docker"],
        education="B.E. CSE",
    )

    # 5. Run Mode B extraction end-to-end
    graph_out = {}
    res = asyncio.run(
        extract_with_graph_memory_concurrent(
            pages_md=pages_data,
            schema=ResumeSchema,
            llm=mock_llm,
            graph_out=graph_out,
            owner="_admin",
        )
    )

    print("\n=== End-to-End Extraction Graph Result ===")
    g_dict = graph_out.get("snapshot") or graph_out.get("graph") or {}
    nodes = g_dict.get("nodes", [])
    print(f"Total Canonical Graph Nodes: {len(nodes)}")

    resolved_count = 0
    total_nodes = len(nodes)
    examples = []

    for n in nodes:
        ev_list = n.get("evidence", [])
        has_blocks = any(len(e.get("block_ids", [])) > 0 for e in ev_list)
        if has_blocks:
            resolved_count += 1
        for e in ev_list:
            examples.append({
                "node_id": n.get("id"),
                "label": n.get("label"),
                "value": n.get("value"),
                "evidence_text": e.get("text"),
                "block_ids": e.get("block_ids"),
                "bbox": e.get("bbox"),
            })

    print(f"Resolved entities with non-empty block_ids: {resolved_count}/{total_nodes} ({resolved_count/total_nodes:.1%})")
    print("\n=== Sample Resolved Evidence Spans ===")
    for ex in examples:
        lbl = ex["label"]
        val = ex["value"]
        txt = ex["evidence_text"]
        bids = ex["block_ids"]
        bb = ex["bbox"]
        print(f"Node: {lbl} ({val!r})")
        print(f"  Evidence Text : {txt!r}")
        print(f"  Resolved Blocks: {bids}")
        print(f"  Union BBox    : {bb}")
        print()


if __name__ == "__main__":
    main()
