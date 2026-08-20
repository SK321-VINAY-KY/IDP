"""
File: logger.py
Purpose: Structured JSON logging with correlation-ID propagation (framework standard).
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: stdlib logging, json
"""

import logging
import json
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class JSONFormatter(logging.Formatter):
    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", correlation_id.get()),
        }
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            ):
                log_entry[key] = value
        return json.dumps(log_entry, default=str)


class PipelineLogger:
    def __init__(self, name: str, service_name: str = "idp-pipeline-a"):
        self.logger = logging.getLogger(name)
        self.service_name = service_name
        self._setup_logger()

    def _setup_logger(self) -> None:
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter(self.service_name))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _log(self, level: int, message: str, extra: Dict[str, Any]) -> None:
        extra.update(
            {"correlation_id": correlation_id.get(), "service": self.service_name}
        )
        self.logger.log(level, message, extra=extra)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, kwargs)


def get_logger(name: str) -> PipelineLogger:
    return PipelineLogger(name)
