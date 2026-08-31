from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import get_current_user
from app.models.auth_models import LoginRequest, LoginResponse, RegisterRequest, User, UserRole
from app.storage.audit_log import get_audit_logger
from app.storage.user_store import get_user_store

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    user_store = get_user_store()
    audit = get_audit_logger()
    username = req.username.strip().lower()

    user = user_store.get_user(username)
    if not user:
        # If user doesn't exist, automatically register as normal user for demo ease
        user = user_store.create_or_update_user(
            username=username,
            full_name=req.username.capitalize(),
            role=UserRole.NORMAL,
        )

    audit.log_activity(
        username=user.username,
        role=user.role.value,
        action="LOGIN",
        details={"ip": "127.0.0.1", "client": "web_console"},
    )

    return LoginResponse(
        token=user.token or f"token_{user.username}",
        user=user,
        message=f"Welcome back, {user.full_name} ({user.role.value})",
    )


@router.post("/register", response_model=LoginResponse)
def register(req: RegisterRequest) -> LoginResponse:
    user_store = get_user_store()
    audit = get_audit_logger()
    username = req.username.strip().lower()

    if user_store.get_user(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{username}' already exists",
        )

    user = user_store.create_or_update_user(
        username=username,
        full_name=req.full_name,
        role=req.role,
    )

    audit.log_activity(
        username=user.username,
        role=user.role.value,
        action="REGISTER",
        details={"full_name": user.full_name, "role": user.role.value},
    )

    return LoginResponse(
        token=user.token or f"token_{user.username}",
        user=user,
        message=f"User '{user.username}' created successfully.",
    )


@router.get("/me", response_model=User)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.get("/users", response_model=List[User])
def list_available_users() -> List[User]:
    user_store = get_user_store()
    return user_store.list_users()
