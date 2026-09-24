"""
Deployment automation script for IDP Layer 2:
- Builds Docker container images for Docling, PaddleOCR Printed, and PaddleOCR Handwritten
- Creates and tags Amazon ECR repositories
- Pushes container images to ECR
- Deploys container-based AWS Lambda functions
- Creates and tags the AWS Step Functions Map state machine
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[1]
ASL_FILE = ROOT_DIR / "step_functions" / "state_machine.asl.json"


def run_command(cmd, cwd=None):
    print(f"  [EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [ERROR] {res.stderr.strip()}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res.stdout.strip()


def parse_tags(tags_str: str | None) -> dict[str, str]:
    raw = tags_str or os.getenv("IDP_AWS_TAGS") or "createdby=vinay.k@shellkode.com,customer=internal"
    tags = {}
    for part in raw.strip().split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


def ensure_ecr_repo(ecr_client, repo_name: str, tags: dict[str, str]) -> str:
    print(f"  Configuring ECR repository '{repo_name}'...")
    try:
        resp = ecr_client.describe_repositories(repositoryNames=[repo_name])
        repo_uri = resp["repositories"][0]["repositoryUri"]
        print(f"    ECR repo exists: {repo_uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            create_kwargs = {
                "repositoryName": repo_name,
                "imageScanningConfiguration": {"scanOnPush": True},
            }
            if tags:
                create_kwargs["tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]
            resp = ecr_client.create_repository(**create_kwargs)
            repo_uri = resp["repository"]["repositoryUri"]
            print(f"    Created ECR repo: {repo_uri}")
        else:
            raise
    return repo_uri


def ensure_stepfunctions_role(iam_client, role_name: str, account_id: str, region: str, tags: dict[str, str]) -> str:
    print(f"  Configuring Step Functions IAM role '{role_name}'...")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "states.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        resp = iam_client.get_role(RoleName=role_name)
        role_arn = resp["Role"]["Arn"]
        print(f"    Role exists: {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            create_kwargs = {
                "RoleName": role_name,
                "AssumeRolePolicyDocument": json.dumps(trust_policy),
                "Description": "IAM role for IDP Layer 2 Step Functions Orchestrator",
            }
            if tags:
                create_kwargs["Tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]
            resp = iam_client.create_role(**create_kwargs)
            role_arn = resp["Role"]["Arn"]
            print(f"    Created role: {role_arn}")
            time.sleep(5)
        else:
            raise

    # Attach policy to invoke Lambdas
    lambda_invoke_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": [
                    f"arn:aws:lambda:{region}:{account_id}:function:idp-engine-*"
                ],
            }
        ],
    }
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="idp-stepfunctions-lambda-invoke",
        PolicyDocument=json.dumps(lambda_invoke_policy),
    )
    return role_arn


def main():
    parser = argparse.ArgumentParser(description="Deploy IDP Layer 2 Container Engines & Step Functions")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE", "106611079163_SK-ML-Sandbox-team-Ps"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    parser.add_argument("--tags", default="createdby=vinay.k@shellkode.com,customer=internal")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    tags = parse_tags(args.tags)

    print("==============================================================")
    print(f"  IDP Layer 2 Deployment: Account {account_id} ({args.region})")
    print(f"  Tags: {tags}")
    print("==============================================================")

    # 1. ECR Repositories
    print("\n--- [1/4] Ensuring ECR Repositories ---")
    ecr = session.client("ecr")
    repos = {
        "docling": ensure_ecr_repo(ecr, "idp-layer2-docling", tags),
        "paddle_printed": ensure_ecr_repo(ecr, "idp-layer2-paddle-printed", tags),
        "paddle_handwritten": ensure_ecr_repo(ecr, "idp-layer2-paddle-handwritten", tags),
    }

    # 2. Step Functions IAM Role
    print("\n--- [2/4] Ensuring Step Functions IAM Role ---")
    iam = session.client("iam")
    sf_role_arn = ensure_stepfunctions_role(iam, "idp-stepfunctions-role", account_id, args.region, tags)

    print("\n--- ECR Repositories & Role Prepared ---")
    for name, uri in repos.items():
        print(f"  {name}: {uri}")
    print(f"  Step Functions Role ARN: {sf_role_arn}")
    print("\nNext step: Build and push container images to ECR, then deploy Lambda functions.")


if __name__ == "__main__":
    main()
