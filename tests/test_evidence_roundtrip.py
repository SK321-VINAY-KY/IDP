import json
import pytest
from pathlib import Path
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import (
    GraphNode,
    Evidence,
    GraphEdge,
)
from src.ai.layer3_extraction.storage import save_document_graph, get_document_graph


def test_graph_evidence_db_roundtrip():
    # 1. Create a GraphMemory with explicit block_ids and bbox
    evidence = Evidence(
        page_number=1,
        text="Sample entity text verified from page 1",
        confidence=0.95,
        block_ids=["p1-docling-1", "p1-docling-2"],
        bbox={"x0": 12.5, "y0": 34.2, "x1": 200.0, "y1": 85.0},
    )
    node1 = GraphNode(
        id="person:vinay",
        type="Person",
        label="Vinay",
        value="Vinay",
        confidence=0.98,
        source_pages=[1],
        status="ASSERTED",
        evidence=[evidence],
    )
    node2 = GraphNode(
        id="org:rashi",
        type="Organization",
        label="Rashi Peripherals",
        value="Rashi Peripherals",
        confidence=0.95,
        source_pages=[1],
        status="ASSERTED",
        evidence=[],
    )
    edge = GraphEdge(
        id="edge:1",
        source_node="person:vinay",
        target_node="org:rashi",
        relationship="EMPLOYED_AT",
        confidence=0.9,
        source_page=1,
        evidence="Employed at Rashi",
    )
    graph = GraphMemory()
    graph.create_node(node1)
    graph.create_node(node2)
    graph.create_edge(edge)

    # Convert to dict
    graph_dict = graph.to_dict()

    # 2. Persist to DB
    doc_id = "test_roundtrip_doc.pdf"
    job_id = "job_roundtrip_test"
    saved_id = save_document_graph(
        doc_id=doc_id,
        graph_dict=graph_dict,
        job_id=job_id,
        owner="test_user",
        schema_id="test_schema",
    )
    assert saved_id is not None, "Failed to save graph to DB"

    # 3. Reload from DB
    loaded_dict = get_document_graph(doc_id=doc_id, job_id=job_id)
    assert loaded_dict is not None, "Failed to load graph from DB"

    reloaded_graph = GraphMemory.from_dict(loaded_dict)
    reloaded_node = reloaded_graph.get_node("person:vinay")
    assert reloaded_node is not None
    assert len(reloaded_node.evidence) == 1
    rel_ev = reloaded_node.evidence[0]

    # Verify block_ids and bbox
    assert rel_ev.block_ids == ["p1-docling-1", "p1-docling-2"]
    assert isinstance(rel_ev.block_ids, list)
    assert rel_ev.bbox is not None
    assert isinstance(rel_ev.bbox, dict)
    assert rel_ev.bbox["x0"] == 12.5
    assert rel_ev.bbox["y0"] == 34.2
    assert rel_ev.bbox["x1"] == 200.0
    assert rel_ev.bbox["y1"] == 85.0


def test_graph_evidence_disk_json_roundtrip(tmp_path: Path):
    # Test disk serialization (.graph.json)
    evidence = Evidence(
        page_number=2,
        text="Project lead on DeepTrust",
        confidence=0.92,
        block_ids=["p2-docling-15", "p2-docling-16"],
        bbox={"x0": 50.0, "y0": 100.0, "x1": 400.0, "y1": 150.0},
    )
    node = GraphNode(
        id="project:deeptrust",
        type="Project",
        label="DeepTrust",
        value="DeepTrust",
        confidence=0.95,
        source_pages=[2],
        status="ASSERTED",
        evidence=[evidence],
    )
    graph = GraphMemory()
    graph.create_node(node)

    json_path = tmp_path / "test_doc.graph.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    reloaded_graph = GraphMemory.from_dict(loaded_json)
    node_ev = reloaded_graph.get_node("project:deeptrust").evidence[0]
    assert node_ev.block_ids == ["p2-docling-15", "p2-docling-16"]
    assert isinstance(node_ev.block_ids, list)
    assert node_ev.bbox is not None
    assert isinstance(node_ev.bbox, dict)
    assert node_ev.bbox["x0"] == 50.0
    assert node_ev.bbox["y0"] == 100.0
    assert node_ev.bbox["x1"] == 400.0
    assert node_ev.bbox["y1"] == 150.0
