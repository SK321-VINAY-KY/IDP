"""
In-memory ring buffer log handler to capture recent system logs for admin inspection.
"""
from __future__ import annotations

import collections
import logging

_log_buffer: collections.deque = collections.deque(maxlen=2000)


class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append(self.format(record))
        except Exception:
            pass


buf_handler = BufferHandler()
buf_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(buf_handler)
