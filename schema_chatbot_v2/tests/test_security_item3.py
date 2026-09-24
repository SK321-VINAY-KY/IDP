import os
from unittest.mock import patch
import pytest

from app.config import Settings


def test_production_raises_on_default_jwt_secret():
    env = {
        "APP_ENV": "production",
        "JWT_SECRET": "idp-schema-pipeline-dev-secret-key-change-me",
        "ADMIN_PASSWORD": "ValidSecureAdminPassword123!",
        "DATABASE_URL": "postgresql://postgres:SuperSecretDbPass999!@localhost:5432/idp",
    }
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(ValueError, match="JWT_SECRET"):
            Settings()


def test_production_raises_on_short_jwt_secret():
    env = {
        "APP_ENV": "production",
        "JWT_SECRET": "short-secret-key",
        "ADMIN_PASSWORD": "ValidSecureAdminPassword123!",
        "DATABASE_URL": "postgresql://postgres:SuperSecretDbPass999!@localhost:5432/idp",
    }
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(ValueError, match="32"):
            Settings()


def test_production_raises_on_default_admin_password():
    env = {
        "APP_ENV": "production",
        "JWT_SECRET": "a" * 32,
        "ADMIN_PASSWORD": "changeme",
        "DATABASE_URL": "postgresql://postgres:SuperSecretDbPass999!@localhost:5432/idp",
    }
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(ValueError, match="ADMIN_PASSWORD"):
            Settings()


def test_production_raises_on_weak_database_password():
    for weak_pass in ["password", "12345", "changeme"]:
        env = {
            "APP_ENV": "production",
            "JWT_SECRET": "a" * 32,
            "ADMIN_PASSWORD": "ValidSecureAdminPassword123!",
            "DATABASE_URL": f"postgresql://postgres:{weak_pass}@localhost:5432/idp",
            "IDP_DATABASE_URL": f"postgresql://postgres:{weak_pass}@localhost:5432/idp",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="database|password|DATABASE_URL"):
                Settings()


def test_production_starts_with_valid_credentials():
    env = {
        "APP_ENV": "production",
        "JWT_SECRET": "a_secure_production_jwt_secret_with_32_plus_chars",
        "ADMIN_PASSWORD": "ValidSecureAdminPassword123!",
        "DATABASE_URL": "postgresql://postgres:SuperSecretDbPass999!@localhost:5432/idp",
        "IDP_DATABASE_URL": "postgresql://postgres:SuperSecretDbPass999!@localhost:5432/idp",
    }
    with patch.dict(os.environ, env, clear=False):
        s = Settings()
        assert s.app_env == "production"
        assert s.jwt_secret == "a_secure_production_jwt_secret_with_32_plus_chars"


def test_development_starts_with_default_credentials():
    env = {
        "APP_ENV": "development",
        "JWT_SECRET": "idp-schema-pipeline-dev-secret-key-change-me",
        "ADMIN_PASSWORD": "changeme",
        "DATABASE_URL": "postgresql://postgres:password@localhost:5432/idp",
        "IDP_DATABASE_URL": "postgresql://postgres:password@localhost:5432/idp",
    }
    with patch.dict(os.environ, env, clear=False):
        s = Settings()
        assert s.app_env == "development"
