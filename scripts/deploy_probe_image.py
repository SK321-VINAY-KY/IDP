"""
Build + push the Layer 3 app image (idp-layer3-app) carrying the in-process
probe, with a traceable, NON-:latest tag.

DO NOT confuse this with scripts/deploy_idp_fargate.py — that file registers the
ECS task def / service and has NO build logic. The build/push parameters mirror
scripts/deploy_idp_ecr_images.py (the file that actually builds idp-layer3-app):
  - build context  = repo ROOT        (deploy_idp_ecr_images.py:23  ROOT = parents[1])
  - Dockerfile     = schema_chatbot_v2/Dockerfile  (deploy_idp_ecr_images.py:37 IMAGES)
  - build command  = docker build --provenance=false --platform linux/amd64
                     -f {dockerfile} -t {tag} .     (deploy_idp_ecr_images.py:90)
  - ECR repo       = idp-layer3-app                 (deploy_idp_ecr_images.py:37)
  - region         = ap-south-1                     (deploy_idp_fargate.py:21 / ecr:21)
  - account assert = sts.get_caller_identity()      (deploy_idp_ecr_images.py:72)
  - ECR login      = ecr.get_authorization_token()  (deploy_idp_ecr_images.py:79-82)

Nothing runs at import time: every side effect lives in a function or under the
`if __name__ == "__main__"` guard, so the pure helpers are unit-testable.

Tag  : probe-<UTC yyyymmddHHMM>-<short git sha>[-dirty]
Guard : refuses any tag that is / resolves to ``latest``.

Account / region / repo come from runtime lookup (boto3 STS) or the mirrored
fargate constants, NEVER from new hard-coded inline values. No secrets are
printed: the ECR login password is masked in the emitted command.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Repo root = parent of scripts/ (mirrors deploy_idp_ecr_images.py:23).
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants — prefer importing from the fargate script (import-safe: that module
# builds no boto3 client at import). Fall back to literals mirrored from the
# source with a cite if the import fails in a stripped checkout.
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    import deploy_idp_fargate as _fargate  # type: ignore

    REGION = _fargate.REGION          # deploy_idp_fargate.py:21
    EXPECTED_ACCOUNT = _fargate.ACCOUNT  # deploy_idp_fargate.py:22
    REPO = _fargate.REPO              # deploy_idp_fargate.py:35  ("idp-layer3-app")
except Exception:  # pragma: no cover - exercised only in a stripped checkout
    REGION = "ap-south-1"             # mirror deploy_idp_fargate.py:21
    EXPECTED_ACCOUNT = "530084039784"  # mirror deploy_idp_fargate.py:22
    REPO = "idp-layer3-app"           # mirror deploy_idp_fargate.py:35

# Build parameters mirror deploy_idp_ecr_images.py (NOT deploy_idp_fargate.py).
DOCKERFILE = "schema_chatbot_v2/Dockerfile"   # deploy_idp_ecr_images.py:37
PLATFORM = "linux/amd64"                       # deploy_idp_ecr_images.py:90
BUILD_FLAGS = "--provenance=false"             # deploy_idp_ecr_images.py:90

# AGENTS.md mandatory tags (ECR TagResource uses Key/Value capitalisation,
# mirroring deploy_idp_ecr_images.py:25-29 TAGS).
MANDATORY_TAGS = [
    {"Key": "createdby", "Value": "vinay.k@shellkode"},
    {"Key": "purpose", "Value": "internal"},
    {"Key": "customer", "Value": "idp-internal-35385"},
]

# Probe runtime model files that MUST ship inside the image (the probe loads the
# ONNX weights from here; labels.json is the index->script map). Relative to ROOT.
MODEL_DIR = Path("src/ai/layer1_probe/models")
ONNX_FP32 = MODEL_DIR / "lipika_script.onnx"
ONNX_INT8 = MODEL_DIR / "lipika_script.int8.onnx"
LABELS_JSON = MODEL_DIR / "lipika_script.labels.json"

_LABEL_TRUNC = 512  # cap the modified/untracked file-list labels


# ---------------------------------------------------------------------------
# Pure, side-effect-free helpers (unit-tested without docker/aws/network)
# ---------------------------------------------------------------------------
def build_tag(now: datetime, sha: str, dirty: bool) -> str:
    """probe-<UTC yyyymmddHHMM>-<short sha>[-dirty]. No I/O."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%d%H%M")
    tag = f"probe-{stamp}-{sha}"
    if dirty:
        tag += "-dirty"
    return tag


def assert_not_latest(tag: str) -> None:
    """Refuse a tag that is exactly ``latest`` or resolves to ``…:latest``."""
    bare = tag.rsplit(":", 1)[-1] if ":" in tag else tag
    if bare.strip().lower() == "latest":
        raise ValueError(f"refusing to use the 'latest' tag (got {tag!r})")


def mask_login_command(cmd: str, password: str) -> str:
    """Replace the real ECR password in a `docker login` command with ``****``."""
    if password:
        return cmd.replace(password, "****")
    return cmd


def _truncate(text: str, limit: int = _LABEL_TRUNC) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"  # ellipsis


def parse_porcelain(porcelain: str) -> tuple[list[str], list[str]]:
    """Split `git status --porcelain` output into (modified, untracked) paths.

    Untracked = lines starting with '??'; everything else with a status code is
    treated as modified/staged. Pure parsing, no git call.
    """
    modified: list[str] = []
    untracked: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        code, _, path = line.partition(" ")
        path = line[3:] if len(line) > 3 else path
        if line.startswith("??"):
            untracked.append(path.strip())
        else:
            modified.append(path.strip())
    return modified, untracked


def build_labels(sha: str, dirty: bool, modified: list[str], untracked: list[str]) -> list[str]:
    """Docker --label values carrying provenance. File lists are truncated."""
    return [
        f"org.idp.git_sha={sha}",
        f"org.idp.dirty={'true' if dirty else 'false'}",
        f"org.idp.modified_files={_truncate(','.join(modified))}",
        f"org.idp.untracked_files={_truncate(','.join(untracked))}",
    ]


def dockerignore_excludes(patterns: list[str], rel_path: str) -> bool:
    """True if a .dockerignore pattern would exclude ``rel_path``.

    Minimal matcher for the patterns this repo uses: exact, suffix globs like
    ``*.onnx``, and directory prefixes like ``logs/``. Good enough to assert the
    model files are NOT excluded (the only use here).
    """
    import fnmatch

    name = rel_path.replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    for raw in patterns:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        p = p.rstrip("/")
        if p.startswith("*."):
            if fnmatch.fnmatch(base, p):
                return True
        elif "/" in p:
            if name == p or name.startswith(p + "/"):
                return True
        else:
            if base == p or name == p or name.startswith(p + "/"):
                return True
    return False


def _read_dockerignore(root: Path) -> list[str]:
    di = root / ".dockerignore"
    if not di.exists():
        return []
    return di.read_text(encoding="utf-8").splitlines()


def sha256_and_size(path: Path) -> tuple[str, int]:
    """Stream the file (handles the ~111MB fp32 model without loading it whole)."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def default_model_relpath(root: Path) -> Path:
    """Which ONNX file ships as the runtime default.

    Mirrors src/ai/layer1_probe/script_classifier.py::default_model_path: int8
    iff the labels sidecar says shipped_default == 'int8', else fp32.
    """
    import json

    labels = root / LABELS_JSON
    try:
        meta = json.loads(labels.read_text(encoding="utf-8"))
        if meta.get("shipped_default") == "int8":
            return ONNX_INT8
    except Exception:
        pass
    return ONNX_FP32


def verify_models_in_context(root: Path) -> list[dict]:
    """Assert the model + labels exist in the build context and are NOT excluded
    by .dockerignore. Returns per-file {path, size, sha256} records (also used
    for printing). Raises if a required file is missing or excluded.
    """
    patterns = _read_dockerignore(root)
    default_onnx = default_model_relpath(root)

    required = [default_onnx, LABELS_JSON]
    # Include the other onnx too if present (both ship), but only the default +
    # labels are hard-required.
    records: list[dict] = []
    for rel in required:
        abs_path = root / rel
        rel_str = str(rel).replace("\\", "/")
        if not abs_path.exists():
            raise FileNotFoundError(f"required model file missing from build context: {rel_str}")
        if dockerignore_excludes(patterns, rel_str):
            raise RuntimeError(f".dockerignore excludes required model file: {rel_str}")
        digest, size = sha256_and_size(abs_path)
        records.append({"path": rel_str, "size": size, "sha256": digest})
    return records


def image_uri(account: str, region: str, repo: str, tag: str) -> str:
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}"


def build_command(dockerfile: str, uri: str, labels: list[str]) -> str:
    label_flags = " ".join(f'--label "{lbl}"' for lbl in labels)
    return (
        f"docker build {BUILD_FLAGS} --platform {PLATFORM} "
        f"-f {dockerfile} {label_flags} -t {uri} ."
    )


def push_command(uri: str) -> str:
    return f"docker push {uri}"


# ---------------------------------------------------------------------------
# Thin git wrappers (I/O; mocked in tests)
# ---------------------------------------------------------------------------
def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def git_short_sha(root: Path) -> str:
    return _git(root, "rev-parse", "--short", "HEAD")


def git_porcelain(root: Path) -> str:
    return _git(root, "status", "--porcelain")


# ---------------------------------------------------------------------------
# Orchestration (I/O; exercised only via --dry-run / real run, not in unit tests)
# ---------------------------------------------------------------------------
def _run(cmd: str) -> None:
    print(f"[EXEC] {cmd}", file=sys.stderr)
    r = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact login/build/push commands + URI, run nothing")
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args(argv)

    # --- provenance from git ---
    sha = git_short_sha(ROOT)
    porcelain = git_porcelain(ROOT)
    dirty = bool(porcelain.strip())
    modified, untracked = parse_porcelain(porcelain)

    tag = build_tag(datetime.now(timezone.utc), sha, dirty)
    assert_not_latest(tag)
    labels = build_labels(sha, dirty, modified, untracked)

    # --- verify models are in the build context and not .dockerignore'd ---
    records = verify_models_in_context(ROOT)
    for rec in records:
        print(f"[model] {rec['path']}  size={rec['size']}B  sha256={rec['sha256']}",
              file=sys.stderr)

    if args.dry_run:
        # Do not touch AWS at all in dry-run: use the mirrored account constant
        # and clearly label it as a placeholder, no STS / ECR calls.
        account = EXPECTED_ACCOUNT
        uri = image_uri(account, args.region, REPO, tag)
        endpoint = f"{account}.dkr.ecr.{args.region}.amazonaws.com"
        login_cmd = f"docker login -u AWS -p <ECR_TOKEN> {endpoint}"
        print("[dry-run] would run (account from mirrored constant; no AWS call made):",
              file=sys.stderr)
        print(mask_login_command(login_cmd, "<ECR_TOKEN>"), file=sys.stderr)
        print(build_command(DOCKERFILE, uri, labels), file=sys.stderr)
        print(push_command(uri), file=sys.stderr)
        assert_not_latest(uri)
        print(uri)  # the one stdout line
        return 0

    # --- real run: resolve account + ECR login via boto3, then build+push ---
    import boto3  # imported inside main so import stays side-effect-free

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(f"WRONG ACCOUNT {account} (expected {EXPECTED_ACCOUNT})")

    ecr = session.client("ecr")
    # Best-effort apply AGENTS.md tags to the repo (mirrors how
    # deploy_idp_ecr_images.py tags the repo at creation; here we only tag an
    # existing repo and never create it).
    try:
        repo_arn = ecr.describe_repositories(repositoryNames=[REPO])["repositories"][0]["repositoryArn"]
        ecr.tag_resource(resourceArn=repo_arn, tags=MANDATORY_TAGS)
    except Exception as exc:  # non-fatal: push does not require tagging
        print(f"[warn] could not tag ECR repo {REPO}: {exc}", file=sys.stderr)

    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, pwd = base64.b64decode(auth["authorizationToken"]).decode().split(":")
    endpoint = auth["proxyEndpoint"]
    uri = image_uri(account, args.region, REPO, tag)
    assert_not_latest(uri)

    login_cmd = f"docker login -u {user} -p {pwd} {endpoint}"
    print(f"[EXEC] {mask_login_command(login_cmd, pwd)}", file=sys.stderr)
    r = subprocess.run(login_cmd, shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError("docker login failed")

    _run(build_command(DOCKERFILE, uri, labels))
    _run(push_command(uri))

    print(uri)  # ONLY the final image URI on stdout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
