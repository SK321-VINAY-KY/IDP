"""Unit tests for scripts/deploy_probe_image.py — pure/import-safe helpers only,
no docker/aws/network.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_probe_image as d  # noqa: E402


def test_import_has_no_side_effects():
    # Mirrors constants resolved at import; nothing network/aws happened.
    assert d.REPO == "idp-layer3-app"
    assert d.REGION == "ap-south-1"


def test_build_tag_format_clean():
    now = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)
    tag = d.build_tag(now, "abc1234", dirty=False)
    assert tag == "probe-203001020304-abc1234"


def test_build_tag_dirty_suffix():
    now = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)
    tag = d.build_tag(now, "abc1234", dirty=True)
    assert tag.endswith("-dirty")
    assert tag == "probe-203001020304-abc1234-dirty"


def test_assert_not_latest_rejects_bare():
    with pytest.raises(ValueError):
        d.assert_not_latest("latest")


def test_assert_not_latest_rejects_suffix():
    with pytest.raises(ValueError):
        d.assert_not_latest("acct.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:latest")


def test_assert_not_latest_allows_probe_tag():
    d.assert_not_latest("probe-203001020304-abc1234")  # no raise


def test_mask_login_command():
    cmd = "docker login -u AWS -p SUPERSECRET host"
    assert "SUPERSECRET" not in d.mask_login_command(cmd, "SUPERSECRET")
    assert "****" in d.mask_login_command(cmd, "SUPERSECRET")


def test_parse_porcelain_splits_modified_and_untracked():
    porc = " M src/a.py\n?? scripts/new.py\nA  staged.py"
    modified, untracked = d.parse_porcelain(porc)
    assert "src/a.py" in modified
    assert "staged.py" in modified
    assert untracked == ["scripts/new.py"]


def test_build_labels_truncates_long_lists():
    many = [f"file_{i}.py" for i in range(500)]
    labels = d.build_labels("abc1234", True, many, [])
    joined = "\n".join(labels)
    assert "org.idp.git_sha=abc1234" in joined
    assert "org.idp.dirty=true" in joined
    modified_label = [l for l in labels if l.startswith("org.idp.modified_files=")][0]
    # truncated to the cap (+ key prefix), not the full 500-file string
    assert len(modified_label) < len("org.idp.modified_files=") + d._LABEL_TRUNC + 2


def test_dockerignore_does_not_exclude_onnx():
    patterns = ["*.bin", "*.safetensors", "logs/", ".env*"]
    assert d.dockerignore_excludes(patterns, "src/ai/layer1_probe/models/lipika_script.onnx") is False
    assert d.dockerignore_excludes(patterns, "src/ai/layer1_probe/models/lipika_script.labels.json") is False


def test_dockerignore_excludes_known_patterns():
    patterns = ["*.bin", "logs/", ".env*"]
    assert d.dockerignore_excludes(patterns, "model.bin") is True
    assert d.dockerignore_excludes(patterns, "logs/pipeline.log") is True


def _make_fake_context(tmp_path: Path, with_onnx=True, with_labels=True,
                       shipped_default=None):
    models = tmp_path / "src" / "ai" / "layer1_probe" / "models"
    models.mkdir(parents=True)
    if with_labels:
        meta = {"labels": ["latin"], }
        if shipped_default:
            meta["shipped_default"] = shipped_default
        (models / "lipika_script.labels.json").write_text(json.dumps(meta), encoding="utf-8")
    if with_onnx:
        (models / "lipika_script.onnx").write_bytes(b"\x00onnx-fp32")
        (models / "lipika_script.int8.onnx").write_bytes(b"\x00onnx-int8")
    # a dockerignore that excludes bins but not onnx
    (tmp_path / ".dockerignore").write_text("*.bin\nlogs/\n", encoding="utf-8")
    return tmp_path


def test_verify_models_ok(tmp_path):
    ctx = _make_fake_context(tmp_path)
    records = d.verify_models_in_context(ctx)
    paths = {r["path"] for r in records}
    assert "src/ai/layer1_probe/models/lipika_script.onnx" in paths
    assert "src/ai/layer1_probe/models/lipika_script.labels.json" in paths
    for r in records:
        assert r["size"] > 0
        assert len(r["sha256"]) == 64


def test_verify_models_picks_int8_default(tmp_path):
    ctx = _make_fake_context(tmp_path, shipped_default="int8")
    records = d.verify_models_in_context(ctx)
    paths = {r["path"] for r in records}
    assert "src/ai/layer1_probe/models/lipika_script.int8.onnx" in paths


def test_verify_models_missing_raises(tmp_path):
    ctx = _make_fake_context(tmp_path, with_onnx=False)
    with pytest.raises(FileNotFoundError):
        d.verify_models_in_context(ctx)


def test_verify_models_excluded_raises(tmp_path):
    ctx = _make_fake_context(tmp_path)
    (ctx / ".dockerignore").write_text("*.onnx\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        d.verify_models_in_context(ctx)


def test_image_uri_and_build_command():
    uri = d.image_uri("123456789012", "ap-south-1", "idp-layer3-app",
                      "probe-203001020304-abc1234")
    assert uri == "123456789012.dkr.ecr.ap-south-1.amazonaws.com/idp-layer3-app:probe-203001020304-abc1234"
    cmd = d.build_command("schema_chatbot_v2/Dockerfile", uri, ["org.idp.git_sha=abc1234"])
    assert "--provenance=false" in cmd
    assert "--platform linux/amd64" in cmd
    assert "-f schema_chatbot_v2/Dockerfile" in cmd
    assert uri in cmd
