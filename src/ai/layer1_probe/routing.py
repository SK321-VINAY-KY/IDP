"""
File: routing.py
Purpose: Script x table  ->  engine + OCR language.
         The single place that encodes the routing table; edit here, not in the handler.
Deps: stdlib
"""
from __future__ import annotations

from src.ai.layer1_probe.types import (
    ProbeConfig, RouteChoice, SCRIPT_TO_PADDLE_LANG,
    ROUTE_PRINTED, ROUTE_SKIP, ROUTE_VLM,
)


def decide_route(*, script: str, script_conf: float, mixed: bool,
                 is_blank: bool, has_table: bool, cfg: ProbeConfig) -> RouteChoice:
    if is_blank:
        return RouteChoice(route=ROUTE_SKIP, reason="blank_page")
    if script in ("unknown", "other") or script_conf < cfg.min_script_conf:
        return RouteChoice(route=ROUTE_VLM, reason=f"script_unresolved({script},{script_conf:.2f})")
    if mixed:
        return RouteChoice(route=ROUTE_VLM, reason="mixed_scripts")
    if script == "latin":
        return RouteChoice(route=ROUTE_PRINTED, ocr_lang=SCRIPT_TO_PADDLE_LANG["latin"],
                           use_structure=has_table, reason="latin_print")
    # Future: dedicated per-script OCR (Telugu/Devanagari/Tamil/...). For now every
    # non-Latin detected script escalates to the VLM.
    return RouteChoice(route=ROUTE_VLM, reason=f"{script}_to_vlm_no_dedicated_ocr")
