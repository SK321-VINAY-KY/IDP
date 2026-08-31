import pytest
from app.storage.user_store import InMemoryUserStore, Role, get_user_store, verify_password


def test_in_memory_user_store_crud():
    store = InMemoryUserStore()
    user = store.create(username="alice", password="alicepassword", role=Role.USER)
    assert user.username == "alice"
    assert user.role == Role.USER
    assert user.user_id is not None
    assert verify_password("alicepassword", user.hashed_password)
    assert not verify_password("wrongpassword", user.hashed_password)

    # Retrieval
    assert store.get_by_username("alice") == user
    assert store.get(user.user_id) == user
    assert store.get_by_username("nonexistent") is None
    assert store.get("nonexistent_id") is None


def test_user_store_duplicate_username_raises():
    store = InMemoryUserStore()
    store.create(username="bob", password="password123")
    with pytest.raises(ValueError, match="already exists"):
        store.create(username="bob", password="anotherpassword")


def test_singleton_seeds_admin():
    store = get_user_store()
    admin = store.get_by_username("admin")
    assert admin is not None
    assert admin.role == Role.ADMIN
    assert verify_password("changeme", admin.hashed_password)


def test_user_store_validations_and_delete():
    store = InMemoryUserStore()

    # Empty username
    with pytest.raises(ValueError, match="username cannot be empty"):
        store.create(username="", password="password123")

    # Empty password
    with pytest.raises(ValueError, match="password cannot be empty"):
        store.create(username="valid_user", password="")

    # Invalid role
    with pytest.raises(ValueError, match="invalid role"):
        store.create(username="valid_user", password="password123", role="superuser")

    # String role coercion
    u = store.create(username="charles", password="password123", role="admin")
    assert u.role == Role.ADMIN

    # Whitespace in get_by_username
    assert store.get_by_username(" charles ") == u

    # Save
    u.role = Role.USER
    store.save(u)
    assert store.get_by_username("charles").role == Role.USER

    # Delete
    store.delete(u.user_id)
    assert store.get(u.user_id) is None
    assert store.get_by_username("charles") is None
