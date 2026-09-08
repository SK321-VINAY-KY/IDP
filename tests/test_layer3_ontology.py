"""
Tests for Step 1: Static Type Ontology.
Validates that ontology.yaml loads with zero LLM calls and contains the
fixed 9 categories, 14 edge types, and 3 relationship states.
"""
import pytest
from src.config.ontology import (
    load_ontology,
    get_categories,
    get_edge_types,
    get_relationship_states,
)


def test_ontology_loads_without_llm():
    ontology = load_ontology()
    assert ontology is not None
    assert ontology.version == "1.0"


def test_ontology_categories():
    categories = get_categories()
    expected_categories = [
        "Person",
        "Organization",
        "Location",
        "Document/Record",
        "Event",
        "Amount",
        "Identifier",
        "Date",
        "Item",
    ]
    assert len(categories) == 9
    for exp in expected_categories:
        assert exp in categories, f"Expected category {exp} missing from ontology"


def test_ontology_edge_types():
    edges = get_edge_types()
    expected_edges = [
        "CONTAINS",
        "MENTIONS",
        "REFERS_TO",
        "SAME_AS",
        "BELONGS_TO",
        "HAS_VALUE",
        "HAS_DATE",
        "HAS_AMOUNT",
        "HAS_ADMISSION",
        "HAS_CLAIM",
        "HAS_PROCEDURE",
        "PERFORMED_BY",
        "RELATES_TO",
        "CONTINUES",
    ]
    assert len(edges) == 14
    for exp in expected_edges:
        assert exp in edges, f"Expected edge type {exp} missing from ontology"


def test_relationship_states():
    states = get_relationship_states()
    assert "ASSERTED" in states
    assert "INFERRED" in states
    assert "UNCERTAIN" in states
