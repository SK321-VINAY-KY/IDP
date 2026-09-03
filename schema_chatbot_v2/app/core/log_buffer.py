"""
In-memory ring buffer log handler to capture recent system logs for admin inspection.
"""
from __future__ import annotations

import collections
import logging

_log_buffer: collections.deque = collections.deque(maxlen=2000)

_STANDARD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "lineno", "funcName", "created",
    "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "getMessage", "exc_info",
    "exc_text", "stack_info", "message", "taskName", "service", "asctime",
})


class BufferFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_FIELDS and not k.startswith("_") and v is not None
        }
        if extras:
            # Format concise extras like: doc_id=168.pdf page=1 confidence=0.98
            details = " ".join(f"{k}={v}" for k, v in list(extras.items())[:6] if not isinstance(v, (dict, list)))
            if details:
                return f"{base} | {details}"
        return base


class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append(self.format(record))
        except Exception:
            pass


buf_handler = BufferHandler()
buf_handler.setFormatter(BufferFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(buf_handler)


def attach_buf_handler(target_logger: logging.Logger) -> None:
    """Attach buf_handler to any logger so its logs reach the web admin console."""
    if buf_handler not in target_logger.handlers:
        target_logger.addHandler(buf_handler)

