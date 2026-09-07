"""
Authentication routes for user signup, login, logout, and current-user identity retrieval.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.core.activity_log import log_activity
from app.core.auth import authenticate_user, create_access_token, get_current_user
from app.storage.user_store import Role, User, get_user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupRequest(BaseModel):
    full_name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="Password (min 8 characters)")
    confirm_password: Optional[str] = Field(None, description="Confirm password for client validation")


class LoginJsonRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    identifier: Optional[str] = None
    password: str = Field(..., min_length=1)


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest) -> Dict[str, Any]:
    """
    Registers a new standard user account.
    Self-signups are strictly assigned role = 'user'.
    """
    full_name = req.full_name.strip()
    email = req.email.strip().lower()
    password = req.password

    if not full_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Full name is required",
        )

    if not EMAIL_REGEX.match(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid email address",
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long",
        )

    if req.confirm_password is not None and req.confirm_password != password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match",
        )

    store = get_user_store()

    # Generate clean username from email or full_name
    base_username = email.split("@")[0].lower()
    username = base_username
    # If username exists, disambiguate with suffix
    existing = store.get_by_username(username)
    suffix = 1
    while existing and existing.email != email:
        username = f"{base_username}{suffix}"
        existing = store.get_by_username(username)
        suffix += 1

    try:
        user = store.create(
            username=username,
            password=password,
            role=Role.USER,  # Public signup strictly defaults to role = user
            full_name=full_name,
            email=email,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    access_token = create_access_token(user)
    log_activity(user.username, "signup", {"email": user.email, "role": user.role.value})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role.value,
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "username": user.username,
    }


@router.post("/login")
async def login(
    request: Request,
    form_data: Optional[OAuth2PasswordRequestForm] = Depends(lambda: None),
) -> Dict[str, Any]:
    """
    Authenticates user credentials (accepts either form data or JSON) and issues a JWT Bearer token.
    """
    identifier = ""
    password = ""

    # Check for JSON body
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            identifier = body.get("identifier") or body.get("email") or body.get("username") or ""
            password = body.get("password") or ""
        except Exception:
            pass
    elif form_data:
        identifier = form_data.username
        password = form_data.password
    else:
        # Fallback to form parsing
        try:
            form = await request.form()
            identifier = str(form.get("username") or form.get("email") or form.get("identifier") or "")
            password = str(form.get("password") or "")
        except Exception:
            pass

    if not identifier or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email/Username and password are required",
        )

    user = authenticate_user(identifier, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user)
    log_activity(user.username, "login", {"role": user.role.value, "email": user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role.value,
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "username": user.username,
    }


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Logs out the currently authenticated user.
    """
    log_activity(current_user.username, "logout", {"role": current_user.role.value})
    return {"status": "ok", "message": "Logged out successfully"}


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns the identity and role profile of the currently authenticated user.
    """
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name or current_user.username,
        "role": current_user.role.value,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }

