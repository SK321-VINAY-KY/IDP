"""
File: loader.py
Purpose: Load and render versioned prompt templates for Layer 3.
Owner: genai-platform@shellkode
Created: 2026-08-20 | Deps: jinja2, pyyaml
"""
import yaml
from functools import lru_cache
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIG_PATH = Path(__file__).parent / "configs" / "prompt_config.yaml"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


@lru_cache(maxsize=1)
def _prompt_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_prompt(prompt_name: str, **context) -> str:
    config = _prompt_config()[prompt_name]
    template = _env().get_template(config["template"])
    return template.render(**context)


def prompt_params(prompt_name: str) -> dict:
    config = _prompt_config()[prompt_name]
    return {"temperature": config["temperature"], "max_tokens": config["max_tokens"]}