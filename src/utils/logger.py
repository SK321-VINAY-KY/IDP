"""
File: logger.py
Purpose: Structured JSON logging with correlation-ID propagation.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Updated: 2026-08-20 (IST timestamp, event key, propagation guard)
Deps: stdlib logging, json, zoneinfo

Logging events emitted by the pipeline:
    document.started
    page.inspected
    page.routed
    page.processed
    page.quality_checked
    page.escalated
    page.completed
    document.completed
"""
import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

# ============================================================
# Configuration
# ============================================================

SERVICE_NAME = "idp-pipeline-a"
LOG_FILE     = "logs/pipeline.log"
LOG_TIMEZONE = ZoneInfo("Asia/Kolkata")   # IST — all timestamps in local time

# ============================================================
# Correlation ID
# ============================================================

# One correlation ID per pipeline execution context.
# Set at document-entry level; all downstream log calls pick it up automatically.
correlation_id: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(value: str) -> None:
    """Set the correlation ID for the current pipeline execution."""
    correlation_id.set(value)


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID."""
    return correlation_id.get()


# ============================================================
# JSON Formatter
# ============================================================

class JSONFormatter(logging.Formatter):
    """
    Converts Python logging records into structured JSON.

    The logger itself remains generic — page/engine-specific
    information is supplied through **kwargs / extra fields.
    Those components decide what to log; this class only handles how.
    """

    # Standard LogRecord attributes that must not be copied into the
    # final JSON object (they are either redundant or internal to Python).
    _STANDARD_FIELDS: frozenset = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "lineno", "funcName", "created",
        "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "getMessage", "exc_info",
        "exc_text", "stack_info", "message", "taskName",
    })

    # Core keys already written explicitly — never overwrite them with extras.
    _CORE_KEYS: frozenset = frozenset({
        "timestamp", "level", "service", "logger", "event", "correlation_id",
    })

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        # Timezone-aware IST timestamp with millisecond precision.
        timestamp = datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")

        log_entry: Dict[str, Any] = {
            "timestamp":      timestamp,
            "level":          record.levelname,
            "service":        self.service_name,
            "logger":         record.name,
            "event":          record.getMessage(),   # "event" not "message"
            "correlation_id": getattr(record, "correlation_id", correlation_id.get()),
        }

        # Append application-specific structured fields from extra={...}
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_FIELDS and key not in self._CORE_KEYS:
                log_entry[key] = value

        # Include exception traceback if present.
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ============================================================
# Pipeline Logger
# ============================================================

class PipelineLogger:
    """
    Generic structured logger used across Engineer A's pipeline.

    Does NOT know about PyMuPDF, Docling, PaddleOCR, or any routing
    logic — those components decide what fields to include.
    This class only handles transport and formatting.
    """

    def __init__(self, name: str, service_name: str = SERVICE_NAME) -> None:
        self.logger = logging.getLogger(name)
        self.service_name = service_name
        self._setup_logger()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def _setup_logger(self) -> None:
        # Guard against duplicate handlers when get_logger() is called
        # multiple times for the same logger name.
        if self.logger.handlers:
            return

        self.logger.setLevel(logging.INFO)
        # Prevent records from propagating to the root logger (avoids
        # double-printing in environments that configure a root handler).
        self.logger.propagate = False

        formatter = JSONFormatter(self.service_name)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Rotating file handler — survives long-running batch jobs
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=10_000_000,   # 10 MB per file
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as exc:
            # Logging must never crash the document pipeline.
            # Fall back to console-only silently.
            self.logger.warning(
                "logger.file_handler_failed",
                extra={"error": str(exc)},
            )

    # --------------------------------------------------------
    # Internal dispatch
    # --------------------------------------------------------

    def _log(self, level: int, event: str, **fields: Any) -> None:
        fields["correlation_id"] = correlation_id.get()
        fields["service"]        = self.service_name
        self.logger.log(level, event, extra=fields)

    # --------------------------------------------------------
    # Public log-level methods
    # --------------------------------------------------------

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)


# ============================================================
# Logger factory
# ============================================================

def get_logger(name: str) -> PipelineLogger:
    """
    Return a configured PipelineLogger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("page.inspected", page_number=1, char_count=0)
    """
    return PipelineLogger(name)
