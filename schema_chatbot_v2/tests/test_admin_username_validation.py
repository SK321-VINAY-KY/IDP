import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store
from app.api.pipeline_routes import OUTPUT_DIR, resolve_owner_output_dir
from fastapi import HTTPException


@pytest.fixture
def client():
    return TestClient(app)


def get_admin_auth_header(client):
    store = get_user_store()
    user = store.get_by_username("admin_val_test")
    if not user:
        store.create(username="admin_val_test", password="adminpassword", role=Role.ADMIN)
    resp = client.post("/auth/login", data={"username": "admin_val_test", "password": "adminpassword"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_admin_username_validation_rejects_invalid_usernames(client):
    admin_auth = get_admin_auth_header(client)

    invalid_usernames = [
        ".",
        "..",
        "ab",  # too short (< 3)
        "a" * 65,  # too long (> 64)
        "user/evil",  # slash
        "user\\evil",  # backslash
        "user evil",  # space
        "user@name",  # invalid character @
        "user#name",  # invalid character #
    ]

    for bad_user in invalid_usernames:
        resp = client.post(
            "/admin/users",
            headers=admin_auth,
            json={"username": bad_user, "password": "validpassword123", "role": "user"},
        )
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for invalid username '{bad_user}', got {resp.status_code}: {resp.text}"
        )

    # Valid username
    valid_resp = client.post(
        "/admin/users",
        headers=admin_auth,
        json={"username": "valid_user-1.2", "password": "validpassword123", "role": "user"},
    )
    assert valid_resp.status_code == 201
    assert valid_resp.json()["username"] == "valid_user-1.2"


def test_owner_directory_resolve_and_is_relative_to():
    # Valid owner directory resolves within OUTPUT_DIR
    valid_dir = resolve_owner_output_dir("valid_owner")
    assert valid_dir.is_relative_to(OUTPUT_DIR.resolve())
    assert valid_dir == (OUTPUT_DIR / "valid_owner").resolve()

    # Traversal attempts must raise HTTPException 400
    for bad_owner in ["../traversal", "../../evil", "foo/../../bar"]:
        with pytest.raises(HTTPException) as exc_info:
            resolve_owner_output_dir(bad_owner)
        assert exc_info.value.status_code == 400
        assert "Path traversal detected" in exc_info.value.detail
