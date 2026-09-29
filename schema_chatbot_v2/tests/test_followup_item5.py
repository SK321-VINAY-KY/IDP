import pytest
from fastapi import HTTPException
from app.api.pipeline_routes import resolve_strict_schema_file, _SCHEMA_ID_REGEX


def test_schema_id_regex_uses_fullmatch():
    """Verify that fullmatch rejects strings with trailing newlines that regex.match accepts."""
    # Proof of python regex subtlety:
    assert _SCHEMA_ID_REGEX.match("schema_valid\n") is not None  # $ matches before newline in match()
    assert _SCHEMA_ID_REGEX.fullmatch("schema_valid\n") is None  # fullmatch strictly rejects

    with pytest.raises(HTTPException) as exc:
        resolve_strict_schema_file("schema_valid\n")
    assert exc.value.status_code == 400
    assert "Invalid schema_id format" in exc.value.detail


def test_schema_id_regex_accepts_valid():
    """Valid schema_id without invalid characters is handled without regex error."""
    # Will fail with 404 because file doesn't exist in registry, but NOT 400 format error
    with pytest.raises(HTTPException) as exc:
        resolve_strict_schema_file("nonexistent_schema_123")
    assert exc.value.status_code == 404
