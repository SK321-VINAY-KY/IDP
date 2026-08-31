from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.models.auth_models import SystemOverview, UserActivityRecord, UserStats
from app.storage.user_store import get_user_store

_IST = ZoneInfo("Asia/Kolkata")
AUDIT_LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._activities: List[UserActivityRecord] = []
        self._load_from_disk()

    def _now_str(self) -> str:
        return datetime.now(_IST).isoformat(timespec="seconds")

    def _load_from_disk(self) -> None:
        if not AUDIT_LOG_FILE.exists():
            return
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        self._activities.append(UserActivityRecord(**record))
        except Exception:
            pass

    def log_activity(
        self,
        username: str,
        role: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> UserActivityRecord:
        record = UserActivityRecord(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            timestamp=self._now_str(),
            username=username,
            role=role,
            action=action,
            details=details or {},
            status=status,
        )

        with self._lock:
            self._activities.insert(0, record)  # most recent first
            try:
                AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.model_dump()) + "\n")
            except Exception:
                pass

        user_store = get_user_store()
        user_store.update_last_active(username)
        return record

    def get_activities(
        self,
        limit: int = 100,
        username: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[UserActivityRecord]:
        with self._lock:
            items = self._activities
            if username:
                items = [a for a in items if a.username.lower() == username.lower()]
            if action:
                items = [a for a in items if a.action.upper() == action.upper()]
            return items[:limit]

    def get_user_stats(self, username: str) -> Optional[UserStats]:
        user_store = get_user_store()
        u = user_store.get_user(username)
        if not u:
            return None

        stats = UserStats(
            username=u.username,
            role=u.role.value,
            full_name=u.full_name,
            created_at=u.created_at,
            last_active=u.last_active,
        )

        with self._lock:
            for act in self._activities:
                if act.username.lower() != username.lower():
                    continue
                if act.action == "DOCUMENT_UPLOAD":
                    stats.documents_uploaded += act.details.get("count", 1)
                elif act.action == "SCHEMA_CONFIRM":
                    stats.schemas_created += 1
                elif act.action == "JOB_START":
                    stats.jobs_executed += 1
                elif act.action == "JOB_COMPLETE":
                    stats.jobs_succeeded += act.details.get("success_count", 1)
                    stats.jobs_failed += act.details.get("failure_count", 0)

        return stats

    def get_all_users_stats(self) -> List[UserStats]:
        user_store = get_user_store()
        users = user_store.list_users()
        result = []
        for u in users:
            s = self.get_user_stats(u.username)
            if s:
                result.append(s)
        return result

    def get_system_overview(self) -> SystemOverview:
        users_stats = self.get_all_users_stats()
        total_users = len(users_stats)
        total_docs = sum(s.documents_uploaded for s in users_stats)
        total_schemas = sum(s.schemas_created for s in users_stats)
        total_jobs = sum(s.jobs_executed for s in users_stats)
        total_succ = sum(s.jobs_succeeded for s in users_stats)
        total_fail = sum(s.jobs_failed for s in users_stats)

        total_runs = total_succ + total_fail
        success_rate = round((total_succ / total_runs * 100.0) if total_runs > 0 else 100.0, 1)

        return SystemOverview(
            total_users=total_users,
            active_users_today=total_users,
            total_documents_uploaded=total_docs,
            total_schemas_created=total_schemas,
            total_jobs_executed=total_jobs,
            total_jobs_succeeded=total_succ,
            total_jobs_failed=total_fail,
            success_rate=success_rate,
        )


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
