import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_cors_disallowed_origin_gets_no_allow_origin_header():
    client = TestClient(app)
    resp = client.options(
        "/auth/login",
        headers={
            "Origin": "https://malicious.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    # A disallowed origin must not receive Access-Control-Allow-Origin
    assert "access-control-allow-origin" not in resp.headers, (
        f"Disallowed origin received allow-origin header: {resp.headers.get('access-control-allow-origin')}"
    )


def test_cors_allowed_origin_gets_allow_origin_header():
    client = TestClient(app)
    resp = client.options(
        "/auth/login",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_production_rejects_wildcard_cors():
    from app.main import get_cors_origins
    with patch.dict(os.environ, {"APP_ENV": "production", "CORS_ORIGINS": "*"}):
        with pytest.raises(ValueError, match="CORS|wildcard|production"):
            get_cors_origins()


def test_production_requires_cors_origins():
    from app.main import get_cors_origins
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=False):
        if "CORS_ORIGINS" in os.environ:
            del os.environ["CORS_ORIGINS"]
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            get_cors_origins()
