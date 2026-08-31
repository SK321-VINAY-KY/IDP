from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from app.models.auth_models import User, UserRole

_IST = ZoneInfo("Asia/Kolkata")
USER_STORE_FILE = Path(__file__).resolve().parents[2] / "logs" / "users.json"


class UserStore:
    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        self._tokens: Dict[str, str] = {}  # token -> username
        self._init_defaults()
        self._load_from_disk()

    def _now_str(self) -> str:
        return datetime.now(_IST).isoformat(timespec="seconds")

    def _init_defaults(self) -> None:
        now = self._now_str()
        default_users = [
            User(
                username="admin",
                role=UserRole.ADMIN,
                full_name="System Administrator",
                created_at="2026-08-30T09:00:00+05:30",
                last_active=now,
                token="token_admin_supersecret",
            ),
            User(
                username="user1",
                role=UserRole.NORMAL,
                full_name="Normal User 1 (Operations)",
                created_at="2026-08-30T10:00:00+05:30",
                last_active=now,
                token="token_user1_secret",
            ),
            User(
                username="user2",
                role=UserRole.NORMAL,
                full_name="Normal User 2 (Auditing)",
                created_at="2026-08-30T11:00:00+05:30",
                last_active=now,
                token="token_user2_secret",
            ),
        ]
        for u in default_users:
            self._users[u.username] = u
            if u.token:
                self._tokens[u.token] = u.username

    def _load_from_disk(self) -> None:
        if not USER_STORE_FILE.exists():
            return
        try:
            data = json.loads(USER_STORE_FILE.read_text(encoding="utf-8"))
            for item in data:
                u = User(**item)
                self._users[u.username] = u
                if u.token:
                    self._tokens[u.token] = u.username
        except Exception:
            pass

    def _save_to_disk(self) -> None:
        try:
            USER_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [u.model_dump() for u in self._users.values()]
            USER_STORE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username)

    def get_user_by_token(self, token: str) -> Optional[User]:
        username = self._tokens.get(token)
        if username:
            return self._users.get(username)
        return None

    def list_users(self) -> List[User]:
        return list(self._users.values())

    def update_last_active(self, username: str) -> None:
        if username in self._users:
            self._users[username].last_active = self._now_str()
            self._save_to_disk()

    def create_or_update_user(
        self,
        username: str,
        full_name: str,
        role: UserRole = UserRole.NORMAL,
    ) -> User:
        username = username.strip().lower()
        now = self._now_str()
        token = f"token_{username}_{datetime.now().timestamp():.0f}"
        if username in self._users:
            u = self._users[username]
            u.full_name = full_name
            u.role = role
            u.last_active = now
            u.token = token
        else:
            u = User(
                username=username,
                role=role,
                full_name=full_name,
                created_at=now,
                last_active=now,
                token=token,
            )
            self._users[username] = u
        self._tokens[token] = username
        self._save_to_disk()
        return u


_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store
