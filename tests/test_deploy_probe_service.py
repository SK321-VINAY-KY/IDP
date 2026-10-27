"""Unit tests for scripts/deploy_probe_service.py — pure helpers + a mocked
--dry-run run. No boto3/network.
"""
import json
import sys
from pathlib import Path

import pytest


def _first_json_object(text: str) -> dict:
    """Extract the first balanced top-level JSON object from mixed stdout."""
    start = text.index("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no balanced JSON object found")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_probe_service as ds  # noqa: E402


def _live_taskdef():
    """Shape mirrors deploy_idp_fargate.register_task's described result."""
    return {
        "taskDefinitionArn": "arn:aws:ecs:ap-south-1:111:task-definition/idp-layer3-task:3",
        "family": "idp-layer3-task",
        "revision": 3,
        "status": "ACTIVE",
        "requiresAttributes": [{"name": "x"}],
        "compatibilities": ["FARGATE"],
        "registeredAt": "2026-01-01",
        "registeredBy": "arn:aws:iam::111:user/x",
        "networkMode": "awsvpc",
        "cpu": "2048",
        "memory": "4096",
        "requiresCompatibilities": ["FARGATE"],
        "executionRoleArn": "arn:aws:iam::111:role/idp-ecs-task-execution-role",
        "taskRoleArn": "arn:aws:iam::111:role/idp-ecs-task-role",
        "containerDefinitions": [
            {
                "name": "idp-web-console",
                "image": "111.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:latest",
                "essential": True,
                "environment": [
                    {"name": "APP_ENV", "value": "production"},
                    {"name": "JWT_SECRET", "value": "super-secret-jwt"},
                    {"name": "ADMIN_PASSWORD", "value": "AdminSecPass!"},
                    {"name": "DATABASE_URL", "value": "postgresql://u:p4ss@host:5432/idp"},
                ],
            }
        ],
    }


NEW_IMAGE = "111.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-203001020304-abc1234"


def test_strip_for_register_removes_rejected_keys():
    stripped = ds.strip_for_register(_live_taskdef())
    for key in ds._REGISTER_REJECTED:
        assert key not in stripped
    # kept fields survive
    assert stripped["family"] == "idp-layer3-task"
    assert stripped["executionRoleArn"].endswith("idp-ecs-task-execution-role")
    assert stripped["taskRoleArn"].endswith("idp-ecs-task-role")
    assert stripped["cpu"] == "2048"


def test_apply_probe_changes_flips_only_image_and_flag():
    out = ds.apply_probe_changes(_live_taskdef(), NEW_IMAGE, "true")
    cd = out["containerDefinitions"][0]
    assert cd["image"] == NEW_IMAGE
    env = {e["name"]: e["value"] for e in cd["environment"]}
    assert env["IDP_PROBE_ENABLED"] == "true"
    # original envs preserved
    assert env["APP_ENV"] == "production"
    # N_SAMPLES never added
    assert "IDP_PROBE_N_SAMPLES" not in env
    # register-rejected fields stripped
    assert "taskDefinitionArn" not in out


def test_apply_probe_changes_flag_off():
    out = ds.apply_probe_changes(_live_taskdef(), NEW_IMAGE, "false")
    env = {e["name"]: e["value"] for e in out["containerDefinitions"][0]["environment"]}
    assert env["IDP_PROBE_ENABLED"] == "false"


def test_upsert_env_replaces_existing():
    container = {"environment": [{"name": "IDP_PROBE_ENABLED", "value": "false"}]}
    ds.upsert_env(container, "IDP_PROBE_ENABLED", "true")
    assert container["environment"] == [{"name": "IDP_PROBE_ENABLED", "value": "true"}]


def test_mask_secrets():
    env = [
        {"name": "APP_ENV", "value": "production"},
        {"name": "JWT_SECRET", "value": "super-secret-jwt"},
        {"name": "ADMIN_PASSWORD", "value": "AdminSecPass!"},
        {"name": "DATABASE_URL", "value": "postgresql://u:p4ss@host:5432/idp"},
    ]
    masked = {e["name"]: e["value"] for e in ds.mask_secrets(env)}
    assert masked["APP_ENV"] == "production"
    assert masked["JWT_SECRET"] == "****"
    assert masked["ADMIN_PASSWORD"] == "****"
    assert "p4ss" not in masked["DATABASE_URL"]
    assert "****" in masked["DATABASE_URL"]
    assert masked["DATABASE_URL"].startswith("postgresql://u:")


def test_assert_not_latest_rejects_latest_image():
    with pytest.raises(ValueError):
        ds.assert_not_latest(NEW_IMAGE.replace("probe-203001020304-abc1234", "latest"))


def test_ecs_tags_are_mandatory_agents_tags():
    keys = {t["key"]: t["value"] for t in ds.ECS_TAGS}
    assert keys["createdby"] == "vinay.k@shellkode"
    assert keys["purpose"] == "internal"
    assert keys["customer"] == "idp-internal-35385"


class _FakeEcs:
    def __init__(self, taskdef):
        self._td = taskdef
        self.registered = None
        self.updated = None

    def describe_services(self, cluster, services):
        return {"services": [{"taskDefinition": self._td["taskDefinitionArn"]}]}

    def describe_task_definition(self, taskDefinition):
        return {"taskDefinition": self._td}

    def register_task_definition(self, **kwargs):
        self.registered = kwargs
        return {"taskDefinition": {"taskDefinitionArn": self._td["taskDefinitionArn"] + "-new"}}

    def update_service(self, **kwargs):
        self.updated = kwargs
        return {"service": {"deployments": [{"status": "PRIMARY", "id": "ecs-svc/dep1"}]}}

    def get_waiter(self, name):
        class _W:
            def wait(self, **kwargs):
                return None
        return _W()


def test_dry_run_masks_and_writes_no_rollback(monkeypatch, tmp_path, capsys):
    fake = _FakeEcs(_live_taskdef())

    class _Session:
        def __init__(self, region_name=None):
            pass

        def client(self, name):
            return fake

    import boto3
    monkeypatch.setattr(boto3, "Session", _Session)
    # redirect rollback writes into tmp just in case
    monkeypatch.setattr(ds, "ROOT", tmp_path)

    rc = ds.main(["--image", NEW_IMAGE, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    # register JSON printed to stdout with IDP_PROBE_ENABLED=true and masked secrets
    payload = _first_json_object(out)
    env = {e["name"]: e["value"] for e in payload["containerDefinitions"][0]["environment"]}
    assert env["IDP_PROBE_ENABLED"] == "true"
    assert env["JWT_SECRET"] == "****"
    assert "IDP_PROBE_N_SAMPLES" not in env
    # no rollback file written in dry-run, no AWS mutation
    assert list(tmp_path.glob("rollback_*.json")) == []
    assert fake.registered is None
    assert fake.updated is None


def test_dry_run_flag_off(monkeypatch, tmp_path, capsys):
    fake = _FakeEcs(_live_taskdef())

    class _Session:
        def __init__(self, region_name=None):
            pass

        def client(self, name):
            return fake

    import boto3
    monkeypatch.setattr(boto3, "Session", _Session)
    monkeypatch.setattr(ds, "ROOT", tmp_path)

    rc = ds.main(["--image", NEW_IMAGE, "--flag-off", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = _first_json_object(out)
    env = {e["name"]: e["value"] for e in payload["containerDefinitions"][0]["environment"]}
    assert env["IDP_PROBE_ENABLED"] == "false"


def test_rejects_latest_image_arg(monkeypatch):
    with pytest.raises(ValueError):
        ds.main(["--image", NEW_IMAGE.replace("probe-203001020304-abc1234", "latest"),
                 "--dry-run"])


def test_write_rollback_file_roundtrip(tmp_path):
    from datetime import datetime, timezone
    td = _live_taskdef()
    path = ds.write_rollback_file(tmp_path, datetime(2030, 1, 1, tzinfo=timezone.utc),
                                  td["taskDefinitionArn"], td)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["taskDefinitionArn"] == td["taskDefinitionArn"]
    assert data["taskDefinition"]["family"] == "idp-layer3-task"
