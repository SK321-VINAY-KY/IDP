from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    NORMAL = "normal"


class User(BaseModel):
    username: str
    role: UserRole = UserRole.NORMAL
    full_name: str
    created_at: str
    last_active: str
    token: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    user: User
    message: str = "Logged in successfully"


class RegisterRequest(BaseModel):
    username: str
    full_name: str
    role: UserRole = UserRole.NORMAL
    password: Optional[str] = None


class UserActivityRecord(BaseModel):
    event_id: str
    timestamp: str
    username: str
    role: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"


class UserStats(BaseModel):
    username: str
    role: str
    full_name: str
    documents_uploaded: int = 0
    schemas_created: int = 0
    jobs_executed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    last_active: str
    created_at: str


class SystemOverview(BaseModel):
    total_users: int
    active_users_today: int
    total_documents_uploaded: int
    total_schemas_created: int
    total_jobs_executed: int
    total_jobs_succeeded: int
    total_jobs_failed: int
    success_rate: float
