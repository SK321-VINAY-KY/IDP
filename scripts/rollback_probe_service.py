"""
Roll the ECS service back to a previous task definition — either the ARN saved
in a rollback_<ts>.json snapshot (written by deploy_probe_service.py) or an ARN
passed directly. If the saved revision was deregistered, re-register it from the
stored JSON (same strip logic) and roll to the new revision.

Nothing runs at import time. --dry-run mutates nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# Reuse helpers/constants from deploy_probe_service (import-safe: no side effects
# at import). Fall back to literals mirrored from deploy_idp_fargate.py.
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    from deploy_probe_service import (  # type: ignore
        DEFAULT_CLUSTER, DEFAULT_SERVICE, REGION, ECS_TAGS,
        strip_for_register, mask_taskdef_for_print,
    )
except Exception:  # pragma: no cover
    REGION = "ap-south-1"
    DEFAULT_CLUSTER = "idp-cluster"
    DEFAULT_SERVICE = "idp-layer3-service"
    ECS_TAGS = [
        {"key": "createdby", "value": "vinay.k@shellkode"},
        {"key": "purpose", "value": "internal"},
        {"key": "customer", "value": "idp-internal-35385"},
    ]

    def strip_for_register(td):  # minimal fallback
        import copy
        t = copy.deepcopy(td)
        for k in ("taskDefinitionArn", "revision", "status", "requiresAttributes",
                  "compatibilities", "registeredAt", "registeredBy", "deregisteredAt"):
            t.pop(k, None)
        return t

    def mask_taskdef_for_print(td):
        return td


def load_rollback_file(path: Path) -> tuple[str, dict]:
    """Return (saved ARN, saved full task def JSON) from a rollback snapshot."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["taskDefinitionArn"], data.get("taskDefinition", {})


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollback-file", help="path to a rollback_<ts>.json snapshot")
    src.add_argument("--task-definition-arn", help="task def ARN to roll back to")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER)
    ap.add_argument("--service", default=DEFAULT_SERVICE)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    saved_td: dict = {}
    if args.rollback_file:
        target_arn, saved_td = load_rollback_file(Path(args.rollback_file))
    else:
        target_arn = args.task_definition_arn

    if args.dry_run:
        print("=== DRY RUN: update_service call it would make ===", file=sys.stderr)
        print(json.dumps({
            "cluster": args.cluster,
            "service": args.service,
            "taskDefinition": target_arn,
        }, indent=2))
        if saved_td:
            print("=== DRY RUN: snapshot task def (secrets masked; re-registered only "
                  "if the revision is gone) ===", file=sys.stderr)
            print(json.dumps(mask_taskdef_for_print(saved_td), indent=2, default=str),
                  file=sys.stderr)
        print("[dry-run] no AWS mutation performed.", file=sys.stderr)
        return 0

    import boto3  # inside main: import stays side-effect free

    ecs = boto3.Session(region_name=args.region).client("ecs")

    # If the target revision no longer exists, re-register from the saved JSON.
    use_arn = target_arn
    try:
        ecs.describe_task_definition(taskDefinition=target_arn)
    except Exception as exc:
        if not saved_td:
            raise RuntimeError(
                f"target task def {target_arn} not found and no snapshot JSON to "
                f"re-register from: {exc}"
            )
        print(f"[rollback] {target_arn} is gone; re-registering from snapshot…",
              file=sys.stderr)
        req = strip_for_register(saved_td)
        req["tags"] = ECS_TAGS
        use_arn = ecs.register_task_definition(**req)["taskDefinition"]["taskDefinitionArn"]

    upd = ecs.update_service(cluster=args.cluster, service=args.service, taskDefinition=use_arn)
    deployment_id = None
    for dep in upd.get("service", {}).get("deployments", []):
        if dep.get("status") == "PRIMARY":
            deployment_id = dep.get("id")
            break
    print(f"[ecs] update_service issued (deployment {deployment_id}); waiting for stable…",
          file=sys.stderr)
    ecs.get_waiter("services_stable").wait(cluster=args.cluster, services=[args.service])

    print(json.dumps({"restored_task_definition_arn": use_arn, "deployment_id": deployment_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
