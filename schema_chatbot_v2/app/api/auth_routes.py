"""
Authentication routes for user login and current-user identity retrieval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.activity_log import log_activity
from app.core.auth import authenticate_user, create_access_token, get_current_user
from app.storage.user_store import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    """
    Authenticates user credentials and issues a JWT Bearer token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user)
    log_activity(user.username, "login", {"role": user.role.value})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role.value,
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns the identity and role of the currently authenticated user.
    """
    return {
        "username": current_user.username,
        "role": current_user.role.value,
    }
