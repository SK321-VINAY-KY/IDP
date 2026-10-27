"""
File: test_bedrock_client_timeouts.py
Purpose: Verify the explicit botocore timeout Config on the Layer 3 Bedrock client.

Connect/read timeouts AND bounded retries are configured: read_timeout bounds a
single attempt and max_attempts (default 2, standard mode) bounds the attempt count,
so a dead call's worst case is ~2 x read_timeout instead of the botocore legacy
5-attempt default. boto3 is replaced with a recording fake, so no AWS call,
credential lookup, or network access happens.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from botocore.config import Config

from src.adapters.llm import bedrock_client
from src.adapters.llm.bedrock_client import BedrockExtractionClient

_TIMEOUT_ENVS = (
    "AWS_PROFILE",
    "IDP_AWS_PROFILE",
    "IDP_BEDROCK_CONNECT_TIMEOUT",
    "IDP_BEDROCK_READ_TIMEOUT",
    "IDP_BEDROCK_MAX_ATTEMPTS",
)


class _FakeBoto3:
    """Records client()/Session() kwargs. Returns MagicMock clients."""

    def __init__(self) -> None:
        self.client_calls: List[Dict[str, Any]] = []
        self.session_calls: List[Dict[str, Any]] = []
        self.session_client_calls: List[Dict[str, Any]] = []
        outer = self

        class _Session:
            def __init__(self, **kwargs: Any) -> None:
                outer.session_calls.append(kwargs)

            def client(self, service_name: str, **kwargs: Any) -> MagicMock:
                outer.session_client_calls.append({"service_name": service_name, **kwargs})
                return MagicMock(name="session_client")

        self.Session = _Session

    def client(self, service_name: str, **kwargs: Any) -> MagicMock:
        self.client_calls.append({"service_name": service_name, **kwargs})
        return MagicMock(name="client")


@pytest.fixture
def fake_boto3(monkeypatch):
    for name in _TIMEOUT_ENVS:
        monkeypatch.delenv(name, raising=False)
    fake = _FakeBoto3()
    monkeypatch.setattr(bedrock_client, "boto3", fake)
    return fake


def _make_client() -> BedrockExtractionClient:
    return BedrockExtractionClient(model_id="m", region="ap-south-1")


def _assert_timeout_config(config: Any, read: float, connect: float, attempts: int = 2) -> None:
    assert isinstance(config, Config)
    assert config.read_timeout == read
    assert config.connect_timeout == connect
    # Retries are now BOUNDED (pulled forward): max_attempts caps the attempt count
    # so the read_timeout is actually effective.
    assert config.retries == {"max_attempts": attempts, "mode": "standard"}


def test_a_default_branch_passes_timeout_config(fake_boto3):
    _make_client().client

    assert len(fake_boto3.client_calls) == 1
    call = fake_boto3.client_calls[0]
    assert call["service_name"] == "bedrock-runtime"
    assert "retries" not in call
    _assert_timeout_config(call["config"], read=240, connect=10)


def test_b_session_branch_passes_timeout_config(fake_boto3, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    _make_client().client

    assert fake_boto3.session_calls == [{"profile_name": "test-profile", "region_name": "ap-south-1"}]
    assert len(fake_boto3.session_client_calls) == 1
    call = fake_boto3.session_client_calls[0]
    assert call["service_name"] == "bedrock-runtime"
    assert "retries" not in call
    _assert_timeout_config(call["config"], read=240, connect=10)


def test_c_env_overrides_are_honored(fake_boto3, monkeypatch):
    monkeypatch.setenv("IDP_BEDROCK_READ_TIMEOUT", "300")
    monkeypatch.setenv("IDP_BEDROCK_CONNECT_TIMEOUT", "5")
    _make_client().client

    _assert_timeout_config(fake_boto3.client_calls[0]["config"], read=300, connect=5)


@pytest.mark.parametrize("bad_value", ["abc", "0"])
def test_d_invalid_read_override_falls_back_to_default(fake_boto3, monkeypatch, bad_value):
    monkeypatch.setenv("IDP_BEDROCK_READ_TIMEOUT", bad_value)
    _make_client().client  # must not raise

    _assert_timeout_config(fake_boto3.client_calls[0]["config"], read=240, connect=10)


def test_e_bounded_retries_default_is_two(fake_boto3):
    _make_client().client
    cfg = fake_boto3.client_calls[0]["config"]
    assert cfg.retries == {"max_attempts": 2, "mode": "standard"}


def test_f_max_attempts_env_override_is_honored(fake_boto3, monkeypatch):
    monkeypatch.setenv("IDP_BEDROCK_MAX_ATTEMPTS", "3")
    _make_client().client
    _assert_timeout_config(fake_boto3.client_calls[0]["config"], read=240, connect=10, attempts=3)


@pytest.mark.parametrize("bad_value", ["abc", "0", "-1"])
def test_g_invalid_max_attempts_falls_back_to_default(fake_boto3, monkeypatch, bad_value):
    monkeypatch.setenv("IDP_BEDROCK_MAX_ATTEMPTS", bad_value)
    _make_client().client  # must not raise
    _assert_timeout_config(fake_boto3.client_calls[0]["config"], read=240, connect=10, attempts=2)
