"""
Pytest configuration and test fixtures for schema_chatbot_v2.
"""
import os
import pytest
from app.storage.user_store import reset_user_store
from app.storage.session_store import reset_session_store

@pytest.fixture(autouse=True)
def isolate_test_stores(monkeypatch):
    """
    Ensure each test starts with fresh, isolated in-memory stores
    unless the test explicitly overrides USER_STORE_TYPE / USER_STORE_FILE.
    """
    if "USER_STORE_FILE" not in os.environ and "USER_STORE_TYPE" not in os.environ:
        monkeypatch.setenv("USER_STORE_TYPE", "memory")
    reset_user_store()
    reset_session_store()
    yield
    reset_user_store()
    reset_session_store()
