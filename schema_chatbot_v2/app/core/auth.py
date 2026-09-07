"""
Authentication and authorization utilities (JWT creation, verification, and role dependencies).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import settings
from app.storage.user_store import Role, User, get_user_store, verify_password

logger = logging.getLogger(__name__)

SECRET_KEY = settings.jwt_secret
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_expire_minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def authenticate_user(identifier: str, password: str) -> Optional[User]:
    """
    Authenticates a user by email or username and password. Returns the User if valid, None otherwise.
    """
    if not identifier or not password:
        return None
    store = get_user_store()
    user = store.get_by_email_or_username(identifier)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generates a JWT access token containing 'sub' (username), 'email', 'full_name', 'role', 'user_id', and 'exp'.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": user.username,
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role.value,
        "exp": expire,
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Decodes the JWT bearer token, retrieves the user from the store, and returns the User object.
    Raises 401 Unauthorized if token is invalid, expired, or user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    store = get_user_store()
    user = store.get_by_username(username)
    if user is None:
        user = store.get_by_email_or_username(username)
    if user is None:
        raise credentials_exception
    return user


def require_role(*allowed_roles: Union[Role, str]):
    """
    Reusable role-based permission guard factory.
    Returns a FastAPI dependency that checks if the authenticated user has one of the allowed roles.
    Example:
        @router.get("/admin-only", dependencies=[Depends(require_role(Role.ADMIN))])
    """
    valid_roles = {r.value if isinstance(r, Role) else str(r) for r in allowed_roles}

    def _role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role.value not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: User role '{current_user.role.value}' lacks required permission.",
            )
        return current_user

    return _role_checker


# Predefined RBAC dependencies for convenience and backward compatibility
require_admin = require_role(Role.ADMIN)
require_user = require_role(Role.ADMIN, Role.USER)

