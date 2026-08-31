"""
In-memory structured user activity log for administrative auditing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

_activity_log: List[Dict[str, Any]] = []


def log_activity(username: str, action: str, detail: Optional[Dict[str, Any]] = None) -> None:
    """
    Appends a timestamped structured activity log entry.
    """
    _activity_log.append({
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "username": username,
        "action": action,
        "detail": detail or {},
    })


def get_activity_logs(username: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Returns recent activity log entries, optionally filtered by username.
    """
    logs = _activity_log
    if username and username.strip():
        target = username.strip()
        logs = [e for e in logs if e.get("username") == target]
    return logs[-limit:]


def clear_activity_logs() -> None:
    """Helper for test cleanup."""
    global _activity_log
    _activity_log.clear()
