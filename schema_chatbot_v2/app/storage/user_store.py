"""
User persistence and authentication storage, abstracted behind an interface
so the in-memory implementation can be swapped for a database backend later.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

# Ensure environment variables from .env are loaded
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


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
    hashed_password: str
    role: Role = Role.USER


class UserStore(ABC):
    @abstractmethod
    def create(self, username: str, password: str, role: Role = Role.USER) -> User: ...

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[User]: ...

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

    def create(self, username: str, password: str, role: Role = Role.USER) -> User:
        username_clean = (username or "").strip()
        if not username_clean:
            raise ValueError("username cannot be empty")
        if not password or not str(password).strip():
            raise ValueError("password cannot be empty")
        if username_clean in self._users_by_username:
            raise ValueError(f"user with username '{username_clean}' already exists")

        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                raise ValueError(f"invalid role '{role}'")

        user = User(
            user_id=str(uuid.uuid4()),
            username=username_clean,
            hashed_password=get_password_hash(password),
            role=role,
        )
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        if not username:
            return None
        return self._users_by_username.get(username.strip())

    def get(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        return self._users_by_id.get(user_id)

    def save(self, user: User) -> None:
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user

    def delete(self, user_id: str) -> None:
        user = self._users_by_id.pop(user_id, None)
        if user:
            self._users_by_username.pop(user.username, None)

    def list_users(self) -> List[User]:
        return list(self._users_by_id.values())


class JSONFileUserStore(UserStore):
    """
    File-backed user storage implementation that persists users to a JSON file on disk.
    Loads existing users on startup and writes to disk on every create, save, and delete.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()
        self._users_by_id: Dict[str, User] = {}
        self._users_by_username: Dict[str, User] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.file_path.exists():
            return
        try:
            content = self.file_path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("User store file %s is empty; initializing empty store.", self.file_path)
                return
            data = json.loads(content)
            if not isinstance(data, list):
                logger.warning(
                    "User store file %s does not contain a list; initializing empty store.",
                    self.file_path,
                )
                return
            for item in data:
                try:
                    user = User(**item)
                    self._users_by_id[user.user_id] = user
                    self._users_by_username[user.username] = user
                except Exception as exc:
                    logger.warning("Skipping invalid user entry in %s: %s", self.file_path, exc)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt JSON in %s (%s); initializing empty store.", self.file_path, exc)
        except Exception as exc:
            logger.error("Error reading %s: %s", self.file_path, exc)

    def _save_to_disk(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                user.model_dump(mode="json")
                if hasattr(user, "model_dump")
                else json.loads(user.json())
                for user in self._users_by_id.values()
            ]
            temp_file = self.file_path.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(self.file_path)
        except Exception as exc:
            logger.error("Failed to write users to %s: %s", self.file_path, exc)

    def create(self, username: str, password: str, role: Role = Role.USER) -> User:
        username_clean = (username or "").strip()
        if not username_clean:
            raise ValueError("username cannot be empty")
        if not password or not str(password).strip():
            raise ValueError("password cannot be empty")
        if username_clean in self._users_by_username:
            raise ValueError(f"user with username '{username_clean}' already exists")

        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                raise ValueError(f"invalid role '{role}'")

        user = User(
            user_id=str(uuid.uuid4()),
            username=username_clean,
            hashed_password=get_password_hash(password),
            role=role,
        )
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user
        self._save_to_disk()
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        if not username:
            return None
        return self._users_by_username.get(username.strip())

    def get(self, user_id: str) -> Optional[User]:
        if not user_id:
            return None
        return self._users_by_id.get(user_id)

    def save(self, user: User) -> None:
        self._users_by_id[user.user_id] = user
        self._users_by_username[user.username] = user
        self._save_to_disk()

    def delete(self, user_id: str) -> None:
        user = self._users_by_id.pop(user_id, None)
        if user:
            self._users_by_username.pop(user.username, None)
            self._save_to_disk()

    def list_users(self) -> List[User]:
        return list(self._users_by_id.values())


# Process-wide singleton for the user store.
_user_store: Optional[UserStore] = None


def reset_user_store() -> None:
    """Reset the singleton instance (primarily for tests)."""
    global _user_store
    _user_store = None


def get_user_store(file_path: Optional[str | Path] = None) -> UserStore:
    """
    Returns the singleton UserStore instance. Defaults to JSONFileUserStore (backed by data/users.json).
    Seeds the admin account from ADMIN_USERNAME / ADMIN_PASSWORD only if the store is empty.
    """
    global _user_store
    if _user_store is None:
        if file_path is None:
            raw_path = os.getenv("USER_STORE_FILE")
            if raw_path:
                path = Path(raw_path)
            else:
                path = Path(__file__).resolve().parent.parent.parent / "data" / "users.json"
        else:
            path = Path(file_path)

        store = JSONFileUserStore(path)
        if not store.list_users():
            admin_user = os.getenv("ADMIN_USERNAME", "admin")
            admin_pass = os.getenv("ADMIN_PASSWORD", "changeme")

            if os.getenv("ADMIN_USERNAME") is None or os.getenv("ADMIN_PASSWORD") is None:
                logger.warning(
                    "ADMIN_USERNAME and/or ADMIN_PASSWORD environment variables not set; "
                    "seeding default admin account '%s' with default password.",
                    admin_user,
                )

            store.create(username=admin_user, password=admin_pass, role=Role.ADMIN)
        _user_store = store
    return _user_store

