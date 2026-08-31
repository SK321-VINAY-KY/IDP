from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query

from app.core.auth import require_admin
from app.models.auth_models import SystemOverview, User, UserActivityRecord, UserStats
from app.storage.audit_log import get_audit_logger
from app.storage.user_store import get_user_store

router = APIRouter(prefix="/admin", tags=["Admin"])

ROOT_DIR = Path(__file__).resolve().parents[3]
LOGS_DIR = ROOT_DIR / "logs"
CHATBOT_LOGS_DIR = ROOT_DIR / "schema_chatbot_v2" / "logs"
SCHEMA_REGISTRY = ROOT_DIR / "schema_registry"
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "dataset_output"


@router.get("/overview", response_model=SystemOverview)
def get_overview(admin: User = Depends(require_admin)) -> SystemOverview:
    audit = get_audit_logger()
    return audit.get_system_overview()


@router.get("/users", response_model=List[UserStats])
def get_users_summary(admin: User = Depends(require_admin)) -> List[UserStats]:
    audit = get_audit_logger()
    return audit.get_all_users_stats()


@router.get("/activity", response_model=List[UserActivityRecord])
def get_activity_feed(
    username: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    admin: User = Depends(require_admin),
) -> List[UserActivityRecord]:
    audit = get_audit_logger()
    return audit.get_activities(limit=limit, username=username, action=action)


@router.get("/user/{username}")
def get_user_details(
    username: str,
    admin: User = Depends(require_admin),
) -> Dict[str, Any]:
    audit = get_audit_logger()
    user_store = get_user_store()
    user = user_store.get_user(username)
    if not user:
        return {"error": f"User '{username}' not found"}

    stats = audit.get_user_stats(username)
    activities = audit.get_activities(limit=50, username=username)

    # Find schemas created by user (or attributed)
    user_schemas = []
    for f in sorted(SCHEMA_REGISTRY.glob("schema_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("created_by") == username or (not data.get("created_by") and username == "user1"):
                user_schemas.append({
                    "file": f.name,
                    "schema_id": data.get("schema_id"),
                    "document_type": (data.get("schema") or {}).get("document_type") or data.get("document_type"),
                    "confirmed_at": data.get("confirmed_at"),
                    "field_count": len((data.get("schema") or {}).get("fields", [])),
                })
        except Exception:
            pass

    return {
        "user": user,
        "stats": stats,
        "recent_activity": activities,
        "schemas": user_schemas,
    }


@router.get("/logs")
def get_system_logs(
    limit: int = Query(default=150, ge=10, le=1000),
    admin: User = Depends(require_admin),
) -> Dict[str, Any]:
    lines = []
    log_files = [
        CHATBOT_LOGS_DIR / "audit_trail.jsonl",
        LOGS_DIR / "pipeline.log",
        CHATBOT_LOGS_DIR / "app.log",
    ]

    for lf in log_files:
        if lf.exists():
            try:
                content = lf.read_text(encoding="utf-8").strip().splitlines()
                lines.extend(content[-limit:])
            except Exception:
                pass

    return {
        "total_lines": len(lines),
        "logs": lines[-limit:],
    }
