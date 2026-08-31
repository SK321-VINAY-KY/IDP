"""
Admin routes for user management and administrative operations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.activity_log import get_activity_logs
from app.core.auth import require_admin
from app.core.log_buffer import _log_buffer
from app.storage.user_store import Role, User, get_user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    role: Role = Role.USER


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(
    req: CreateUserRequest,
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Creates a new user account (admin-only).
    """
    store = get_user_store()
    try:
        user = store.create(username=req.username, password=req.password, role=req.role)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/users")
def list_all_users(_: User = Depends(require_admin)) -> List[Dict[str, Any]]:
    """
    Lists all users in the system (admin-only).
    """
    store = get_user_store()
    return [
        {
            "user_id": u.user_id,
            "username": u.username,
            "role": u.role.value,
        }
        for u in store.list_users()
    ]


@router.get("/logs/system")
def get_system_logs(
    limit: int = 500,
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Returns recent system log lines from the in-memory ring buffer (admin-only).
    """
    return {"lines": list(_log_buffer)[-limit:]}


@router.get("/logs/users")
def get_user_logs(
    username: str = "",
    limit: int = 500,
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Returns structured user activity logs, optionally filtered by username (admin-only).
    """
    return {"logs": get_activity_logs(username=username, limit=limit)}
