import os
import sys
import pytest


def test_lambda_handlers_import_under_aws_lambda_env(monkeypatch):
    """
    Verify that in an AWS Lambda runtime (AWS_LAMBDA_FUNCTION_NAME set),
    unrelated web app secrets (DATABASE_URL, JWT_SECRET, ADMIN_PASSWORD)
    do NOT block lambda_layer1_handler and engine handlers from importing.
    """
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "test-layer1-routing")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("IDP_APP_ENV", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("IDP_DATABASE_URL", raising=False)

    # 1. Import Layer 1 Lambda handler
    from src.api.lambda_layer1_handler import handler as l1_handler, lambda_handler, lambda_settings
    assert callable(l1_handler)
    assert callable(lambda_handler)
    assert lambda_settings is not None

    # 2. Import Docker Engine handlers
    for p in ["docker/docling", "docker/paddle_printed", "docker/paddle_handwritten"]:
        sys.path.insert(0, p)
        try:
            import handler as engine_handler
            assert callable(getattr(engine_handler, "lambda_handler", None))
        finally:
            sys.path.pop(0)


def test_shared_settings_does_not_skip_database_url_even_if_on_lambda(monkeypatch):
    """
    Verify that the shared Settings in src/config/settings.py does NOT skip
    the DATABASE_URL production check just because AWS_LAMBDA_FUNCTION_NAME is set.
    Any web process running on Lambda must still provide valid production database credentials.
    """
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "test-web-process-on-lambda")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("IDP_DATABASE_URL", raising=False)

    from src.config.settings import Settings, LambdaSettings

    # Shared Settings MUST raise because DATABASE_URL has default/weak password
    with pytest.raises(ValueError, match="Insecure/default database password in DATABASE_URL"):
        Settings()

    # But the Lambda handlers' own config (LambdaSettings) skips database_url
    lambda_cfg = LambdaSettings()
    assert lambda_cfg.app_env == "production"


def test_webapp_settings_still_strictly_enforces_production(monkeypatch):
    """
    Verify that the web app (schema_chatbot_v2) STILL rejects default
    credentials and missing secrets in production/staging.
    """
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    sys.path.insert(0, "schema_chatbot_v2")
    try:
        with pytest.raises(ValueError, match="JWT_SECRET"):
            from app.config import Settings
            Settings()
    finally:
        if "schema_chatbot_v2" in sys.path:
            sys.path.remove("schema_chatbot_v2")
