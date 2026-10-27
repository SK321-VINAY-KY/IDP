"""
Deploy the probe image to the live ECS service by cloning the CURRENT task
definition, flipping ONLY the container image and the IDP_PROBE_ENABLED env var,
then updating the service. A rollback snapshot is written BEFORE anything changes.

Nothing runs at import time (all side effects under main()/functions) so the
pure helpers are unit-testable without boto3/network.

What it does (real run):
  1. describe-services -> current taskDefinition ARN
  2. describe-task-definition -> full live task def
  3. write rollback_<UTC ts>.json {current ARN + full current task def JSON}
  4. clone the def, set container image = --image, upsert IDP_PROBE_ENABLED
     (true, or false with --flag-off); leave IDP_PROBE_N_SAMPLES untouched
  5. strip the fields register-task-definition rejects
  6. register-task-definition (with AGENTS.md mandatory tags)
  7. update-service to the new revision, wait services-stable
  8. print new revision ARN + deployment id

--dry-run prints the exact register JSON (secrets MASKED) + the update-service
call and mutates nothing (no rollback file written).

No hard-coded account / subnet / role: everything comes from the looked-up live
task def. Cluster/service default to idp-cluster / idp-layer3-service (mirrors
deploy_idp_fargate.py:31-32).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# Import-safe constants from the fargate script (no boto3 client built at its
# import). Fall back to mirrored literals in a stripped checkout.
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    import deploy_idp_fargate as _fargate  # type: ignore

    REGION = _fargate.REGION              # deploy_idp_fargate.py:21
    DEFAULT_CLUSTER = _fargate.CLUSTER    # deploy_idp_fargate.py:31
    DEFAULT_SERVICE = _fargate.SERVICE    # deploy_idp_fargate.py:32
    TASK_FAMILY = _fargate.TASK_FAMILY    # deploy_idp_fargate.py:33
except Exception:  # pragma: no cover
    REGION = "ap-south-1"                 # mirror deploy_idp_fargate.py:21
    DEFAULT_CLUSTER = "idp-cluster"       # mirror deploy_idp_fargate.py:31
    DEFAULT_SERVICE = "idp-layer3-service"  # mirror deploy_idp_fargate.py:32
    TASK_FAMILY = "idp-layer3-task"       # mirror deploy_idp_fargate.py:33

# Reuse the latest-guard from the image script (duplicated here as a fallback so
# this module stands alone in a stripped checkout — see deploy_probe_image.py).
try:
    from deploy_probe_image import assert_not_latest  # type: ignore
except Exception:  # pragma: no cover
    def assert_not_latest(tag: str) -> None:
        bare = tag.rsplit(":", 1)[-1] if ":" in tag else tag
        if bare.strip().lower() == "latest":
            raise ValueError(f"refusing to use the 'latest' tag (got {tag!r})")

PROBE_FLAG = "IDP_PROBE_ENABLED"

# AGENTS.md mandatory tags. ECS register_task_definition uses lowercase
# key/value (mirrors deploy_idp_fargate.py:44 ECS_TAGS).
ECS_TAGS = [
    {"key": "createdby", "value": "vinay.k@shellkode.com"},
    {"key": "purpose", "value": "internal"},
    {"key": "customer", "value": "idp-internal-35385"},
]

# Fields register_task_definition rejects when cloning a described task def.
_REGISTER_REJECTED = (
    "taskDefinitionArn", "revision", "status", "requiresAttributes",
    "compatibilities", "registeredAt", "registeredBy", "deregisteredAt",
)

# Env var keys whose values are secrets and must be masked in any printout.
_SECRET_KEYS = {"JWT_SECRET", "ADMIN_PASSWORD", "IDP_JWT_SECRET", "IDP_ADMIN_PASSWORD"}
_MASK = "****"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no AWS)
# ---------------------------------------------------------------------------
def _mask_url_password(value: str) -> str:
    """Mask the password in a URL like scheme://user:pass@host/… -> user:****@."""
    return re.sub(r"(://[^:/@\s]+:)([^@/\s]+)(@)", r"\1" + _MASK + r"\3", value)


def mask_secrets(env_list: list[dict]) -> list[dict]:
    """Return a COPY of an ECS `environment` list with secret values masked.

    Masks known secret keys outright and the password inside any *_URL /
    DATABASE_URL value. Non-secret envs are left intact.
    """
    masked: list[dict] = []
    for item in env_list:
        name = item.get("name", "")
        value = item.get("value", "")
        if name in _SECRET_KEYS:
            value = _MASK
        elif name.endswith("_URL") or "DATABASE_URL" in name:
            value = _mask_url_password(value)
        masked.append({"name": name, "value": value})
    return masked


def mask_taskdef_for_print(taskdef: dict) -> dict:
    """Deep-copy a task def and mask secrets in every container's environment."""
    td = copy.deepcopy(taskdef)
    for cd in td.get("containerDefinitions", []):
        if "environment" in cd:
            cd["environment"] = mask_secrets(cd["environment"])
    return td


def strip_for_register(taskdef: dict) -> dict:
    """Return a register_task_definition-ready copy: drop the read-only fields
    AWS rejects, keep everything else the live def carried."""
    td = copy.deepcopy(taskdef)
    for key in _REGISTER_REJECTED:
        td.pop(key, None)
    return td


def _select_container(containers: list[dict], name: Optional[str] = None) -> dict:
    """The container to mutate: by name if given, else the web console container
    (deploy_idp_fargate.py:200 'idp-web-console'), else the sole container."""
    if name:
        for cd in containers:
            if cd.get("name") == name:
                return cd
        raise KeyError(f"container {name!r} not found in task def")
    for cd in containers:
        if cd.get("name") == "idp-web-console":
            return cd
    if len(containers) == 1:
        return containers[0]
    raise KeyError("could not identify the target container (multiple present)")


def upsert_env(container: dict, name: str, value: str) -> None:
    """Replace the env var in-place if present, else append it."""
    env = container.setdefault("environment", [])
    for item in env:
        if item.get("name") == name:
            item["value"] = value
            return
    env.append({"name": name, "value": value})


def apply_probe_changes(taskdef: dict, image: str, flag_value: str,
                        container_name: Optional[str] = None) -> dict:
    """Clone the live task def; flip ONLY the container image + IDP_PROBE_ENABLED.

    IDP_PROBE_N_SAMPLES is intentionally NOT touched (its default applies).
    Returns a register-ready dict (read-only fields already stripped).
    """
    cloned = copy.deepcopy(taskdef)
    container = _select_container(cloned.get("containerDefinitions", []), container_name)
    container["image"] = image
    upsert_env(container, PROBE_FLAG, flag_value)
    return strip_for_register(cloned)


def _rollback_filename(now: datetime) -> str:
    return f"rollback_{now.astimezone(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"


def register_request_json(register_ready: dict, tags: list[dict]) -> dict:
    """Shape of the register_task_definition kwargs (for printing/inspection)."""
    req = copy.deepcopy(register_ready)
    req["tags"] = tags
    return req


# ---------------------------------------------------------------------------
# I/O helpers (mocked in tests)
# ---------------------------------------------------------------------------
def write_rollback_file(root: Path, now: datetime, current_arn: str, current_td: dict) -> Path:
    path = root / _rollback_filename(now)
    payload = {"taskDefinitionArn": current_arn, "taskDefinition": current_td}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    try:
        import os
        os.chmod(path, 0o600)  # best-effort restrictive perms (no-op on Windows)
    except Exception:
        pass
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True,
                    help="full ECR URI incl. probe tag (rejects :latest)")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER)
    ap.add_argument("--service", default=DEFAULT_SERVICE)
    ap.add_argument("--container", default=None,
                    help="container name to mutate (default: idp-web-console / sole container)")
    ap.add_argument("--flag-off", action="store_true",
                    help="set IDP_PROBE_ENABLED=false (deploy image, probe OFF)")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    assert_not_latest(args.image)
    flag_value = "false" if args.flag_off else "true"

    import boto3  # inside main: import stays side-effect free

    ecs = boto3.Session(region_name=args.region).client("ecs")

    # 1-2. Look up the CURRENT task def of the service.
    svc = ecs.describe_services(cluster=args.cluster, services=[args.service])
    services = svc.get("services", [])
    if not services:
        raise RuntimeError(f"service {args.service} not found in cluster {args.cluster}")
    current_arn = services[0]["taskDefinition"]
    current_td = ecs.describe_task_definition(taskDefinition=current_arn)["taskDefinition"]

    register_ready = apply_probe_changes(current_td, args.image, flag_value, args.container)
    req = register_request_json(register_ready, ECS_TAGS)

    if args.dry_run:
        print("=== DRY RUN: register_task_definition request (secrets masked) ===",
              file=sys.stderr)
        print(json.dumps(mask_taskdef_for_print(req), indent=2, default=str))
        print("=== DRY RUN: update_service call it would make ===", file=sys.stderr)
        print(json.dumps({
            "cluster": args.cluster,
            "service": args.service,
            "taskDefinition": "<new revision ARN from register above>",
        }, indent=2))
        print(f"[dry-run] current task def ARN (would be saved to rollback file): {current_arn}",
              file=sys.stderr)
        print("[dry-run] no rollback file written, no AWS mutation performed.",
              file=sys.stderr)
        return 0

    # 3. Rollback snapshot FIRST (faithful, unmasked — needed to re-register).
    rollback_path = write_rollback_file(ROOT, datetime.now(timezone.utc), current_arn, current_td)
    print(f"[rollback] wrote {rollback_path}", file=sys.stderr)
    print("[WARNING] rollback file contains UNMASKED env secrets — do NOT commit it; "
          "add rollback_*.json to .gitignore.", file=sys.stderr)

    # 5-6. Register the new revision.
    new_td = ecs.register_task_definition(**req)["taskDefinition"]
    new_arn = new_td["taskDefinitionArn"]
    print(f"[ecs] registered {new_arn} ({PROBE_FLAG}={flag_value})", file=sys.stderr)

    # 7. Update the service and wait for stable.
    upd = ecs.update_service(cluster=args.cluster, service=args.service, taskDefinition=new_arn)
    deployment_id = None
    for dep in upd.get("service", {}).get("deployments", []):
        if dep.get("status") == "PRIMARY":
            deployment_id = dep.get("id")
            break
    print(f"[ecs] update_service issued (deployment {deployment_id}); waiting for stable…",
          file=sys.stderr)
    ecs.get_waiter("services_stable").wait(cluster=args.cluster, services=[args.service])

    print(json.dumps({"new_task_definition_arn": new_arn, "deployment_id": deployment_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
