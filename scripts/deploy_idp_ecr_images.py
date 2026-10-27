"""
Stage 2: create (tagged) ECR repos and FRESHLY build + push all container images
into account 530084039784 (ap-south-1). No reuse of prior images.

Repos/images:
  idp-layer2-docling            <- docker/docling/Dockerfile
  idp-layer2-paddle-printed     <- docker/paddle_printed/Dockerfile
  idp-layer2-paddle-handwritten <- docker/paddle_handwritten/Dockerfile
  idp-layer3-app                <- schema_chatbot_v2/Dockerfile

Reads AWS creds from the environment (set from .env2).
"""
import argparse
import base64
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-south-1"
TAGS = [
    {"Key": "createdby", "Value": "vinay.k@shellkode.com"},
    {"Key": "customer", "Value": "idp-internal-35385"},
    {"Key": "purpose", "Value": "internal"},
]

IMAGES = {
    "idp-layer2-docling": "docker/docling/Dockerfile",
    "idp-layer2-paddle-printed": "docker/paddle_printed/Dockerfile",
    "idp-layer2-paddle-handwritten": "docker/paddle_handwritten/Dockerfile",
    "idp-layer3-app": "schema_chatbot_v2/Dockerfile",
}


def run(cmd):
    print(f"[EXEC] {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}")


def ensure_repo(ecr, name):
    try:
        uri = ecr.describe_repositories(repositoryNames=[name])["repositories"][0]["repositoryUri"]
        print(f"[ecr] exists: {uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            uri = ecr.create_repository(
                repositoryName=name,
                imageScanningConfiguration={"scanOnPush": True},
                tags=TAGS,
            )["repository"]["repositoryUri"]
            print(f"[ecr] created: {uri}")
        else:
            raise
    return uri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-sep subset of repo names to build", default="")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    acct = session.client("sts").get_caller_identity()["Account"]
    assert acct == "530084039784", f"WRONG ACCOUNT {acct}"
    ecr = session.client("ecr")

    # docker login to ECR
    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, pwd = base64.b64decode(auth["authorizationToken"]).decode().split(":")
    endpoint = auth["proxyEndpoint"]
    run(f"docker login -u {user} -p {pwd} {endpoint}")

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    results = {}
    for repo, dockerfile in IMAGES.items():
        if only and repo not in only:
            continue
        uri = ensure_repo(ecr, repo)
        tag = f"{uri}:latest"
        print(f"\n=== build+push {repo} ===")
        run(f"docker build --provenance=false --platform linux/amd64 -f {dockerfile} -t {tag} .")
        run(f"docker push {tag}")
        results[repo] = tag
        print(f"[done] {repo} -> {tag}")

    print("\n=== IMAGES PUSHED ===")
    for k, v in results.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
