import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.user_store import Role, get_user_store


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_login_admin_success(client):
    # Seeded admin is admin/changeme or admin@example.com by default
    resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"

    # Test /auth/me with the token
    token = data["access_token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == "admin"
    assert me_data["role"] == "admin"
    assert "user_id" in me_data


def test_auth_login_via_email(client):
    # Test logging in with admin email
    resp = client.post("/auth/login", json={"email": "admin@example.com", "password": "changeme"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "admin"
    assert "access_token" in data


def test_auth_signup_success(client):
    import uuid
    uid = uuid.uuid4().hex[:6]
    test_email = f"janedoe_{uid}@example.com"
    req_data = {
        "full_name": "Jane Doe",
        "email": test_email,
        "password": "strongPassword123",
        "confirm_password": "strongPassword123",
    }
    resp = client.post("/auth/signup", json=req_data)
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == test_email
    assert data["full_name"] == "Jane Doe"
    assert data["role"] == "user"  # Strictly defaulted to user
    assert "access_token" in data

    # Verify user can immediately access /auth/me
    token = data["access_token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == test_email
    assert me_resp.json()["role"] == "user"


def test_auth_signup_validation_errors(client):
    # Password too short (< 8 chars)
    resp = client.post("/auth/signup", json={
        "full_name": "Short Pass",
        "email": "short@example.com",
        "password": "short",
    })
    assert resp.status_code == 400
    assert "at least 8 characters" in resp.json()["detail"]

    # Invalid email format
    resp = client.post("/auth/signup", json={
        "full_name": "Bad Email",
        "email": "not-an-email",
        "password": "validPassword123",
    })
    assert resp.status_code == 400
    assert "valid email" in resp.json()["detail"]

    # Password mismatch
    resp = client.post("/auth/signup", json={
        "full_name": "Mismatch Pass",
        "email": "mismatch@example.com",
        "password": "validPassword123",
        "confirm_password": "differentPassword123",
    })
    assert resp.status_code == 400
    assert "Passwords do not match" in resp.json()["detail"]


def test_auth_signup_duplicate_email(client):
    import uuid
    uid = uuid.uuid4().hex[:6]
    test_email = f"dup_{uid}@example.com"
    # Create first
    resp1 = client.post("/auth/signup", json={
        "full_name": "Duplicate User",
        "email": test_email,
        "password": "validPassword123",
    })
    assert resp1.status_code == 201

    # Create again
    resp2 = client.post("/auth/signup", json={
        "full_name": "Duplicate User Two",
        "email": test_email,
        "password": "validPassword123",
    })
    assert resp2.status_code == 400
    assert "already exists" in resp2.json()["detail"]


def test_auth_login_invalid_password(client):
    resp = client.post("/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert resp.status_code == 401
    assert "Incorrect" in resp.json()["detail"]


def test_auth_login_nonexistent_user(client):
    resp = client.post("/auth/login", data={"username": "does_not_exist", "password": "any"})
    assert resp.status_code == 401


def test_auth_me_without_token(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_auth_me_with_invalid_token(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.value"})
    assert resp.status_code == 401


def test_auth_logout(client):
    # Login as admin
    login_resp = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
    token = login_resp.json()["access_token"]

    resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_regular_user_auth(client):
    import uuid
    uid = uuid.uuid4().hex[:6]
    test_username = f"user_{uid}"
    test_email = f"user_{uid}@example.com"
    store = get_user_store()
    store.create(
        username=test_username,
        password="charliepassword",
        role=Role.USER,
        full_name="Charlie B",
        email=test_email,
    )

    resp = client.post("/auth/login", data={"username": test_username, "password": "charliepassword"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "user"

    token = data["access_token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == test_username
    assert me_resp.json()["role"] == "user"


def test_static_app_delivery(client):
    """Verifies that the frontend static app is properly served with all auth and RBAC elements."""
    resp = client.get("/app/")
    assert resp.status_code == 200
    html = resp.text

    # Verify Auth Layer Elements
    assert 'id="auth-container"' in html
    assert 'id="login-card"' in html
    assert 'id="signup-card"' in html
    assert 'id="login-identifier"' in html
    assert 'id="login-password"' in html
    assert 'id="signup-fullname"' in html
    assert 'id="signup-email"' in html
    assert 'id="signup-password"' in html
    assert 'id="signup-confirm-password"' in html
    assert 'id="admin-dashboard-modal"' in html
    assert 'id="forgot-password-modal"' in html

    # Verify Header RBAC Elements
    assert 'id="admin-nav-btn"' in html
    assert 'id="user-menu-container"' in html
    assert 'id="user-dropdown-menu"' in html
    assert 'id="user-avatar-circle"' in html


def test_complete_user_signup_and_rbac_flow(client):
    """
    Simulates complete user registration, authentication, and RBAC restriction.
    """
    import uuid
    uid = uuid.uuid4().hex[:8]
    full_name = f"Jane Doe {uid}"
    email = f"jane_{uid}@example.com"
    password = "MySecurePassword123"

    # 1. Sign up new account
    signup_resp = client.post("/auth/signup", json={
        "full_name": full_name,
        "email": email,
        "password": password,
        "confirm_password": password,
    })
    assert signup_resp.status_code == 201
    signup_data = signup_resp.json()
    assert signup_data["email"] == email
    assert signup_data["full_name"] == full_name
    assert signup_data["role"] == "user"  # Public signup MUST be role 'user'
    assert "access_token" in signup_data

    user_token = signup_data["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}

    # 2. Verify identity via /auth/me
    me_resp = client.get("/auth/me", headers=user_headers)
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == email
    assert me_data["full_name"] == full_name
    assert me_data["role"] == "user"

    # 3. Verify user is FORBIDDEN from admin-only routes
    for admin_route in ["/admin/users", "/admin/logs/system", "/admin/logs/users"]:
        res = client.get(admin_route, headers=user_headers)
        assert res.status_code == 403, f"Expected 403 for {admin_route}, got {res.status_code}"

    # 4. Verify user can access user-allowed private workspace routes
    doc_resp = client.get("/me/documents", headers=user_headers)
    assert doc_resp.status_code == 200
    assert "documents" in doc_resp.json()

    # 5. Log out user
    logout_resp = client.post("/auth/logout", headers=user_headers)
    assert logout_resp.status_code == 200


