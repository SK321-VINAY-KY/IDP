import os
import pytest

from app.config import resolve_app_env, Settings


def test_app_env_unset_raises_error(monkeypatch):
    """APP_ENV unset without AWS runtime must fail fast with ValueError."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("IDP_APP_ENV", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)

    with pytest.raises(ValueError, match="APP_ENV"):
        resolve_app_env()

    with pytest.raises(ValueError, match="APP_ENV"):
        Settings()


def test_app_env_aws_ecs_defaults_to_production(monkeypatch):
    """When running on ECS with unset APP_ENV, default to production."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("IDP_APP_ENV", raising=False)
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI", "http://169.254.170.2/v2/metadata")

    assert resolve_app_env() == "production"

    # Enforces production credential checks
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings()

    # With production credentials, successfully instantiates as production
    s = Settings(
        jwt_secret="A" * 32,
        admin_password="SuperSecureAdminPassword999!",
        database_url="postgresql://postgres:StrongDbPass888!@localhost:5432/idp",
    )
    assert s.app_env == "production"


def test_app_env_aws_lambda_defaults_to_production(monkeypatch):
    """When running on AWS Lambda with unset APP_ENV, default to production."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("IDP_APP_ENV", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI", raising=False)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "idp-layer1-routing")

    assert resolve_app_env() == "production"

    # Enforces production credential checks
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings()


def test_app_env_explicit_development(monkeypatch):
    """Explicit APP_ENV=development is honored."""
    monkeypatch.setenv("APP_ENV", "development")

    assert resolve_app_env() == "development"
    s = Settings(database_url="postgresql://postgres:password@localhost:5432/idp")
    assert s.app_env == "development"
