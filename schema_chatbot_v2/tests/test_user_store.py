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


def test_json_file_user_store_persists_across_restarts(tmp_path):
    from app.storage.user_store import JSONFileUserStore
    file_path = tmp_path / "users.json"

    # 1. Start with fresh store
    store1 = JSONFileUserStore(file_path)
    u1 = store1.create(username="persisted_user", password="secretpassword", role=Role.USER)
    assert u1.username == "persisted_user"
    assert file_path.exists()

    # Verify passwords stored on disk are hashed, never plaintext
    content = file_path.read_text(encoding="utf-8")
    assert "secretpassword" not in content
    assert "pbkdf2:sha256:" in content

    # 2. Simulate server restart with a new instance pointing to same file
    store2 = JSONFileUserStore(file_path)
    loaded_user = store2.get_by_username("persisted_user")
    assert loaded_user is not None
    assert loaded_user.user_id == u1.user_id
    assert loaded_user.role == Role.USER
    assert verify_password("secretpassword", loaded_user.hashed_password)

    # 3. Update user and verify persistence
    loaded_user.role = Role.ADMIN
    store2.save(loaded_user)

    store3 = JSONFileUserStore(file_path)
    assert store3.get_by_username("persisted_user").role == Role.ADMIN

    # 4. Delete user and verify removal
    store3.delete(loaded_user.user_id)
    store4 = JSONFileUserStore(file_path)
    assert store4.get_by_username("persisted_user") is None


def test_json_file_user_store_corrupt_and_empty_files(tmp_path):
    from app.storage.user_store import JSONFileUserStore

    # Non-existent file
    store_nonexistent = JSONFileUserStore(tmp_path / "does_not_exist.json")
    assert store_nonexistent.list_users() == []

    # Empty file
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   \n", encoding="utf-8")
    store_empty = JSONFileUserStore(empty_file)
    assert store_empty.list_users() == []

    # Corrupt JSON file
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not: valid, json", encoding="utf-8")
    store_corrupt = JSONFileUserStore(corrupt_file)
    assert store_corrupt.list_users() == []

    # Can still create users after starting from corrupt file
    u = store_corrupt.create("recovered", "password123")
    assert store_corrupt.get_by_username("recovered") is not None


def test_get_user_store_does_not_overwrite_existing_users_on_restart(tmp_path, monkeypatch):
    from app.storage.user_store import reset_user_store

    users_file = tmp_path / "users.json"
    monkeypatch.setenv("USER_STORE_FILE", str(users_file))

    # First boot: seeds admin
    reset_user_store()
    store1 = get_user_store()
    admin = store1.get_by_username("admin")
    assert admin is not None
    assert admin.role == Role.ADMIN

    # Create additional user
    store1.create("engineer_bob", "bobpass", role=Role.USER)

    # Second boot: simulates restart
    reset_user_store()
    # Change env admin pass to ensure it doesn't overwrite existing users
    monkeypatch.setenv("ADMIN_PASSWORD", "brand_new_pass_that_should_not_overwrite")
    store2 = get_user_store()
    assert store2.get_by_username("engineer_bob") is not None
    reloaded_admin = store2.get_by_username("admin")
    assert reloaded_admin is not None
    # Original admin password hash remains, not overwritten
    assert verify_password("changeme", reloaded_admin.hashed_password)
    assert not verify_password("brand_new_pass_that_should_not_overwrite", reloaded_admin.hashed_password)


def test_json_file_session_store_persists(tmp_path):
    from app.storage.session_store import JSONFileSessionStore
    sess_file = tmp_path / "sessions.json"

    store1 = JSONFileSessionStore(sess_file)
    sess = store1.create()
    sess.turn_count = 5
    sess.owner = "test_user"
    store1.save(sess)

    # Reload from disk in a new store instance
    store2 = JSONFileSessionStore(sess_file)
    loaded = store2.get(sess.session_id)
    assert loaded is not None
    assert loaded.turn_count == 5
    assert loaded.owner == "test_user"

