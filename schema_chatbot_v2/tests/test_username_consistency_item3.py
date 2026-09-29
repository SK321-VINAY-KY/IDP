import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import InMemoryUserStore, Role, get_user_store
from app.api.pipeline_routes import OUTPUT_DIR, resolve_owner_output_dir
from fastapi import HTTPException


@pytest.fixture
def client():
    return TestClient(app)


def test_user_store_lowercase_creation_and_lookup():
    store = InMemoryUserStore()
    user = store.create(username="MixedCaseUser", password="password123", role=Role.USER)
    
    # Username should be lowercased on creation
    assert user.username == "mixedcaseuser"

    # Lookup should succeed regardless of case
    assert store.get_by_username("mixedcaseuser") == user
    assert store.get_by_username("MixedCaseUser") == user
    assert store.get_by_username("MIXEDCASEUSER") == user
    assert store.get_by_username("  MiXeDcAsEuSeR  ") == user

    # Creating a duplicate with different casing should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        store.create(username="MIXEDCASEUSER", password="otherpassword", role=Role.USER)
    assert "already exists" in str(exc_info.value)


def test_admin_create_user_consecutive_dots_rejected_and_lowercase(client):
    # Ensure admin user exists
    store = get_user_store()
    if not store.get_by_username("admin_item3"):
        store.create(username="admin_item3", password="adminpassword", role=Role.ADMIN)
    
    login_resp = client.post("/auth/login", data={"username": "admin_item3", "password": "adminpassword"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Reject consecutive dots
    for bad_username in ["user..name", "alice..bob", "..test", "test.."]:
        resp = client.post(
            "/admin/users",
            headers=headers,
            json={"username": bad_username, "password": "validpassword123", "role": "user"},
        )
        assert resp.status_code in (400, 422), (
            f"Expected rejection for consecutive dots '{bad_username}', got {resp.status_code}"
        )

    # Mixed case should be lowercased
    resp_create = client.post(
        "/admin/users",
        headers=headers,
        json={"username": "TestUserMixedCase", "password": "validpassword123", "role": "user"},
    )
    assert resp_create.status_code == 201
    created = resp_create.json()
    assert created["username"] == "testusermixedcase"

    # User can login with any case
    resp_login = client.post(
        "/auth/login",
        data={"username": "TESTUSERMIXEDCASE", "password": "validpassword123"},
    )
    assert resp_login.status_code == 200


def test_resolve_owner_output_dir_consistency():
    # Consecutive dots must raise 400
    for bad_owner in ["user..name", "../traversal", "alice..bob"]:
        with pytest.raises(HTTPException) as exc_info:
            resolve_owner_output_dir(bad_owner)
        assert exc_info.value.status_code == 400

    # Normalization to lowercase
    resolved = resolve_owner_output_dir("UpperUser")
    assert resolved == (OUTPUT_DIR / "upperuser").resolve()
