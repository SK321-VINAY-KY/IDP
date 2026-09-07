"""
User persistence and authentication storage, supporting PostgreSQL database persistence
via SQLAlchemy with an in-memory fallback for testing and development.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure environment variables from .env are loaded
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_password_hash(password: str) -> str:
    """
    Hashes a plain text password using standard library PBKDF2-HMAC-SHA256
    with a random cryptographic salt (zero external binary dependencies).
    """
    salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"pbkdf2:sha256:100000${salt}${pw_hash.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a stored PBKDF2 hash using constant-time comparison.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        if hashed_password.startswith("pbkdf2:"):
            parts = hashed_password.split("$")
            if len(parts) != 3:
                return False
            params, salt, expected_hash = parts
            _, algo, iters = params.split(":")
            actual_hash = hashlib.pbkdf2_hmac(
                algo,
                plain_password.encode("utf-8"),
                salt.encode("utf-8"),
                int(iters),
            ).hex()
            return hmac.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False
    return False


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"


class User(BaseModel):
    user_id: str
    username: str
    email: str = ""
    full_name: str = ""
    hashed_password: str
    role: Role = Role.USER
    created_at: Optional[datetime] = None


class UserStore(ABC):
    @abstractmethod
    def create(
        self,
        username: str,
        password: str,
        role: Role = Role.USER,
        full_name: str = "",
        email: str = "",
    ) -> User: ...

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[User]: ...

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[User]: ...

    @abstractmethod
    def get_by_email_or_username(self, identifier: str) -> Optional[User]: ...

    @abstractmethod
    def get(self, user_id: str) -> Optional[User]: ...

    @abstractmethod
    def save(self, user: User) -> None: ...

    @abstractmethod
    def delete(self, user_id: str) -> None: ...

    @abstractmethod
    def list_users(self) -> List[User]: ...


class InMemoryUserStore(UserStore):
    """
    In-memory user storage implementation for development and testing.
    """

    def __init__(self):
        self._users_by_id: Dict[str, User] = {}
        self._users_by_username: Dict[str, User] = {}
        self._users_by_email: Dict[str, User] = {}

    def create(
        self,
        username: str,
        password: str,
        role: Role = Role.USER,
        full_name: str = "",
        email: str = "",
    ) -> User:
        username_clean = (username or "").strip()
        email_clean = (email or "").strip().lower()
        full_name_clean = (full_name or "").strip() or username_clean

        if not username_clean:
            if email_clean:
                username_clean = email_clean.split("@")[0]
            else:
                raise ValueError("username cannot be empty")

        if not password or not str(password).strip():
            raise ValueError("password cannot be empty")

        if username_clean in self._users_by_username:
            raise ValueError(f"user with username '{username_clean}' already exists")

        if email_clean and email_clean in self._users_by_email:
            raise ValueError(f"user with email '{email_clean}' already exists")

        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                raise ValueError(f"invalid role '{role}'")

        user = User(
            user_id=str(uuid.uuid4()),
            username=username_clean,
            email=email_clean,
            full_name=full_name_clean,
            hashed_password=get_password_hash(password),
            role=role,
            created_at=datetime.now(timezone.utc),
        )
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user
        if user.email:
            self._users_by_email[user.email] = user
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        if not username:
            return None
        return self._users_by_username.get(username.strip())

    def get_by_email(self, email: str) -> Optional[User]:
        if not email:
            return None
        return self._users_by_email.get(email.strip().lower())

    def get_by_email_or_username(self, identifier: str) -> Optional[User]:
        if not identifier:
            return None
        clean_id = identifier.strip()
        # Try exact email first
        user = self._users_by_email.get(clean_id.lower())
        if user:
            return user
        # Try username
        return self._users_by_username.get(clean_id)

    def get(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        return self._users_by_id.get(user_id)

    def save(self, user: User) -> None:
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user
        if user.email:
            self._users_by_email[user.email.lower()] = user

    def delete(self, user_id: str) -> None:
        user = self._users_by_id.pop(user_id, None)
        if user:
            self._users_by_username.pop(user.username, None)
            if user.email:
                self._users_by_email.pop(user.email.lower(), None)

    def list_users(self) -> List[User]:
        return list(self._users_by_id.values())


class PostgresUserStore(UserStore):
    """
    PostgreSQL-backed user storage implementation with SQLAlchemy.
    Falls back gracefully to SQLite if PostgreSQL is not reachable.
    """

    def __init__(self):
        try:
            from src.ai.layer3_extraction.storage import init_db
            init_db()
        except Exception as exc:
            logger.warning("user_store.db_init_warning", error=str(exc))

    def _to_user(self, rec) -> Optional[User]:
        if not rec:
            return None
        try:
            role_enum = Role(rec.role)
        except Exception:
            role_enum = Role.USER

        return User(
            user_id=rec.id,
            username=rec.username,
            email=rec.email or "",
            full_name=rec.full_name or rec.username,
            hashed_password=rec.password_hash,
            role=role_enum,
            created_at=rec.created_at,
        )

    def create(
        self,
        username: str,
        password: str,
        role: Role = Role.USER,
        full_name: str = "",
        email: str = "",
    ) -> User:
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        username_clean = (username or "").strip()
        email_clean = (email or "").strip().lower()
        full_name_clean = (full_name or "").strip() or username_clean

        if not username_clean:
            if email_clean:
                username_clean = email_clean.split("@")[0]
            else:
                raise ValueError("username cannot be empty")

        if not password or not str(password).strip():
            raise ValueError("password cannot be empty")

        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                raise ValueError(f"invalid role '{role}'")

        session = SessionLocal()
        try:
            # Check unique username
            existing_user = session.query(UserRecord).filter_by(username=username_clean).first()
            if existing_user:
                raise ValueError(f"user with username '{username_clean}' already exists")

            # Check unique email
            if email_clean:
                existing_email = session.query(UserRecord).filter_by(email=email_clean).first()
                if existing_email:
                    raise ValueError(f"user with email '{email_clean}' already exists")

            user_id = str(uuid.uuid4())
            rec = UserRecord(
                id=user_id,
                username=username_clean,
                email=email_clean or f"{username_clean}@example.com",
                full_name=full_name_clean,
                password_hash=get_password_hash(password),
                role=role.value,
                created_at=datetime.now(timezone.utc),
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return self._to_user(rec)
        finally:
            session.close()

    def get_by_username(self, username: str) -> Optional[User]:
        if not username:
            return None
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            rec = session.query(UserRecord).filter_by(username=username.strip()).first()
            return self._to_user(rec)
        finally:
            session.close()

    def get_by_email(self, email: str) -> Optional[User]:
        if not email:
            return None
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            rec = session.query(UserRecord).filter_by(email=email.strip().lower()).first()
            return self._to_user(rec)
        finally:
            session.close()

    def get_by_email_or_username(self, identifier: str) -> Optional[User]:
        if not identifier:
            return None
        clean_id = identifier.strip()
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            # Try email match
            rec = session.query(UserRecord).filter_by(email=clean_id.lower()).first()
            if not rec:
                # Try username match
                rec = session.query(UserRecord).filter_by(username=clean_id).first()
            return self._to_user(rec)
        finally:
            session.close()

    def get(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            rec = session.query(UserRecord).filter_by(id=user_id).first()
            return self._to_user(rec)
        finally:
            session.close()

    def save(self, user: User) -> None:
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            rec = session.query(UserRecord).filter_by(id=user.user_id).first()
            if rec:
                rec.username = user.username
                rec.email = user.email
                rec.full_name = user.full_name
                rec.password_hash = user.hashed_password
                rec.role = user.role.value
                session.commit()
        finally:
            session.close()

    def delete(self, user_id: str) -> None:
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            rec = session.query(UserRecord).filter_by(id=user_id).first()
            if rec:
                session.delete(rec)
                session.commit()
        finally:
            session.close()

    def list_users(self) -> List[User]:
        from src.ai.layer3_extraction.storage import SessionLocal, UserRecord

        session = SessionLocal()
        try:
            recs = session.query(UserRecord).all()
            return [u for u in (self._to_user(r) for r in recs) if u is not None]
        finally:
            session.close()


# Process-wide singleton for the user store.
_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    """
    Returns the singleton UserStore instance backed by PostgreSQL / SQLite.
    On first call, seeds one admin account from ADMIN_USERNAME / ADMIN_PASSWORD / ADMIN_EMAIL
    env vars (default admin@example.com / changeme).
    """
    global _user_store
    if _user_store is None:
        try:
            store = PostgresUserStore()
        except Exception as exc:
            logger.warning("user_store.fallback_to_in_memory", error=str(exc))
            store = InMemoryUserStore()

        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        admin_pass = os.getenv("ADMIN_PASSWORD", "changeme")
        admin_name = os.getenv("ADMIN_NAME", "System Administrator")

        # Check if admin already exists
        existing_admin = store.get_by_username(admin_user) or store.get_by_email(admin_email)
        if not existing_admin:
            if os.getenv("ADMIN_USERNAME") is None or os.getenv("ADMIN_PASSWORD") is None:
                logger.warning(
                    "ADMIN_USERNAME and/or ADMIN_PASSWORD environment variables not set; "
                    "seeding default admin account '%s' (%s) with default password.",
                    admin_user,
                    admin_email,
                )
            store.create(
                username=admin_user,
                password=admin_pass,
                role=Role.ADMIN,
                full_name=admin_name,
                email=admin_email,
            )
        _user_store = store
    return _user_store


