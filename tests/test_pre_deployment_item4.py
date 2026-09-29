import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT_DIR / ".env.example"


def test_env_example_removals_and_additions():
    assert ENV_EXAMPLE.is_file(), ".env.example must exist"
    content = ENV_EXAMPLE.read_text(encoding="utf-8")

    # 1. AWS_LAMBDA_FUNCTION_NAME and REDIS_URL must NOT be present
    assert "AWS_LAMBDA_FUNCTION_NAME" not in content, "AWS_LAMBDA_FUNCTION_NAME should be removed from .env.example"
    assert "REDIS_URL" not in content, "REDIS_URL is unused and should be removed from .env.example"

    # 2. Added rate limiting & upload size variables
    assert "MAX_UPLOAD_SIZE_BYTES" in content, "MAX_UPLOAD_SIZE_BYTES should be present in .env.example"
    assert "RATE_LIMIT_EXTRACT_MAX_REQUESTS" in content, "RATE_LIMIT_EXTRACT_MAX_REQUESTS should be present in .env.example"
    assert "RATE_LIMIT_EXTRACT_WINDOW_SECONDS" in content, "RATE_LIMIT_EXTRACT_WINDOW_SECONDS should be present in .env.example"

    # 3. Added Gemini keys
    assert "IDP_GEMINI_API_KEY" in content, "IDP_GEMINI_API_KEY should be present in .env.example"
    assert "IDP_GEMINI_BASE_URL" in content, "IDP_GEMINI_BASE_URL should be present in .env.example"

    # 4. VLM model name present
    assert "IDP_VLM_MODEL_NAME" in content, "IDP_VLM_MODEL_NAME should be present in .env.example"
