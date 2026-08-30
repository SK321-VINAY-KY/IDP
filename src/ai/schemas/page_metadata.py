"""
File: page_metadata.py
Purpose: Structured metadata for Engineer A page processing.
Owner: engineer-a@idp-pilot
Created: 2026-08-20 | Deps: stdlib only (dataclasses, json)

This module stores page-level metadata independently from logging.
Metadata covers:
    - Page inspection
    - Detected capabilities
    - Routing decision
    - Engine execution
    - Confidence
    - Quality checks
    - Escalation history
    - Final processing result

Architectural role:

    page_metadata.py
           │
           │ stores state
           ▼
    PageMetadata
           │
           ├──────────► PageOutput / Engineer B
           │
           └──────────► logger.py
                             │
                             ▼
                        pipeline.log

PageMetadata is the source of structured page state.
logger.py records processing events.
They are intentionally separate — do not merge them.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json


# ============================================================
# Escalation
# ============================================================

@dataclass
class EscalationRecord:
    """Records one escalation from one processing engine to another."""

    from_engine: str
    to_engine: str
    reason: str
    from_confidence: Optional[float] = None
    threshold: Optional[float] = None
    attempt: int = 1
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Engine Result
# ============================================================

@dataclass
class EngineResult:
    """Records the result produced by a single engine execution."""

    engine: str
    confidence: Optional[float] = None
    success: bool = True
    latency_ms: Optional[float] = None
    output_type: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Page Metadata
# ============================================================

@dataclass
class PageMetadata:
    """
    Complete metadata describing the processing lifecycle of one document page.

    Populated incrementally as the page passes through each pipeline stage:
        1. After inspect_page()          — inspection fields
        2. After capability detection    — capabilities list
        3. After routing                 — routing fields + engine_plan
        4. After each engine run         — engine_results.append(...)
        5. After quality check           — quality fields
        6. After escalation (if any)     — escalation_history.append(...)
        7. After final result            — final_* fields

    Usage:
        metadata = PageMetadata(document_name="sdg_goals.pdf", page_number=1)
        metadata.add_capability("has_digital_text")
        metadata.set_routing(["docling"], "capability_based")
        metadata.add_engine_result("docling", confidence=0.97, latency_ms=520)
        metadata.set_final_result("docling", 0.97, success=True, total_latency_ms=520)
        print(metadata.to_json())
    """

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page_number: Optional[int] = None

    # --------------------------------------------------------
    # Inspection metadata
    # --------------------------------------------------------
    char_count: Optional[int] = None
    image_coverage: Optional[float] = None
    is_scanned: Optional[bool] = None
    primary_script: Optional[str] = None
    complexity_score: Optional[float] = None

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------
    classification: Optional[str] = None
    classification_confidence: Optional[float] = None

    # --------------------------------------------------------
    # Capability detection
    # --------------------------------------------------------
    capabilities: List[str] = field(default_factory=list)
    # Example: ["has_digital_text", "has_handwriting", "has_figures"]

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------
    routing_mode: Optional[str] = None
    engine_plan: List[str] = field(default_factory=list)
    selected_engine: Optional[str] = None
    route_confidence: Optional[float] = None

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------
    engine_results: List[EngineResult] = field(default_factory=list)

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------
    quality_passed: Optional[bool] = None
    quality_confidence: Optional[float] = None
    quality_reason: Optional[str] = None

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------
    escalated: bool = False
    escalation_count: int = 0
    escalation_history: List[EscalationRecord] = field(default_factory=list)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------
    final_engine: Optional[str] = None
    final_confidence: Optional[float] = None
    processing_success: Optional[bool] = None
    total_latency_ms: Optional[float] = None

    # --------------------------------------------------------
    # Additional metadata
    # --------------------------------------------------------
    extra: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # Methods
    # ============================================================

    def add_capability(self, capability: str) -> None:
        """Add a detected capability without creating duplicates."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def set_routing(
        self,
        engine_plan: List[str],
        routing_mode: str,
        selected_engine: Optional[str] = None,
        route_confidence: Optional[float] = None,
    ) -> None:
        """Store the routing decision."""
        self.engine_plan = engine_plan
        self.routing_mode = routing_mode
        self.selected_engine = selected_engine
        self.route_confidence = route_confidence

    def add_engine_result(
        self,
        engine: str,
        confidence: Optional[float] = None,
        success: bool = True,
        latency_ms: Optional[float] = None,
        output_type: Optional[str] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the result of an engine execution."""
        self.engine_results.append(EngineResult(
            engine=engine,
            confidence=confidence,
            success=success,
            latency_ms=latency_ms,
            output_type=output_type,
            error=error,
            details=details or {},
        ))

    def add_escalation(
        self,
        from_engine: str,
        to_engine: str,
        reason: str,
        from_confidence: Optional[float] = None,
        threshold: Optional[float] = None,
        attempt: int = 1,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an engine escalation and update escalation counters."""
        self.escalation_history.append(EscalationRecord(
            from_engine=from_engine,
            to_engine=to_engine,
            reason=reason,
            from_confidence=from_confidence,
            threshold=threshold,
            attempt=attempt,
            details=details or {},
        ))
        self.escalated = True
        self.escalation_count += 1

    def set_quality_result(
        self,
        passed: bool,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Store the quality-gate result."""
        self.quality_passed = passed
        self.quality_confidence = confidence
        self.quality_reason = reason

    def set_final_result(
        self,
        engine: Optional[str],
        confidence: Optional[float],
        success: bool,
        total_latency_ms: Optional[float] = None,
    ) -> None:
        """Store the final page processing result."""
        self.final_engine = engine
        self.final_confidence = confidence
        self.processing_success = success
        self.total_latency_ms = total_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata into a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metadata into a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
