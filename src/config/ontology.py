"""
File: ontology.py
Purpose: Static Type Ontology loader for Layer 3.
         Replaces dynamic schema discovery (schema_chatbot_v2) with a fixed,
         zero-LLM top-level taxonomy and edge definitions loaded once at startup.
Owner: engineer-a@idp-pilot
Created: 2026-09-07
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


ONTOLOGY_PATH = Path(__file__).parent / "ontology.yaml"


class CategoryDefinition(BaseModel):
    name: str = ""
    description: str = ""
    examples: List[str] = Field(default_factory=list)


class EdgeTypeDefinition(BaseModel):
    name: str = ""
    description: str = ""


class OntologyConfig(BaseModel):
    version: str = "1.0"
    categories: Dict[str, CategoryDefinition] = Field(default_factory=dict)
    edge_types: Dict[str, EdgeTypeDefinition] = Field(default_factory=dict)
    relationship_states: List[str] = Field(default_factory=list)

    def category_names(self) -> List[str]:
        return list(self.categories.keys())

    def edge_type_names(self) -> List[str]:
        return list(self.edge_types.keys())


@lru_cache(maxsize=1)
def load_ontology(path: Optional[Path] = None) -> OntologyConfig:
    """
    Load and cache the static type ontology from YAML.
    Zero runtime LLM calls.
    """
    yaml_file = path or ONTOLOGY_PATH
    if not yaml_file.exists():
        raise FileNotFoundError(f"Ontology file not found at: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_categories = data.get("categories", {})
    categories: Dict[str, CategoryDefinition] = {}
    for cat_name, cat_val in raw_categories.items():
        if isinstance(cat_val, dict):
            categories[cat_name] = CategoryDefinition(
                name=cat_name,
                description=cat_val.get("description", ""),
                examples=cat_val.get("examples", []),
            )
        else:
            categories[cat_name] = CategoryDefinition(name=cat_name, description=str(cat_val))

    raw_edges = data.get("edge_types", {})
    edge_types: Dict[str, EdgeTypeDefinition] = {}
    for edge_name, edge_val in raw_edges.items():
        if isinstance(edge_val, dict):
            edge_types[edge_name] = EdgeTypeDefinition(
                name=edge_name,
                description=edge_val.get("description", ""),
            )
        else:
            edge_types[edge_name] = EdgeTypeDefinition(name=edge_name, description=str(edge_val))

    rel_states = data.get("relationship_states", ["ASSERTED", "INFERRED", "UNCERTAIN"])

    return OntologyConfig(
        version=str(data.get("version", "1.0")),
        categories=categories,
        edge_types=edge_types,
        relationship_states=rel_states,
    )


def get_categories() -> List[str]:
    return load_ontology().category_names()


def get_edge_types() -> List[str]:
    return load_ontology().edge_type_names()


def get_relationship_states() -> List[str]:
    return load_ontology().relationship_states
