"""
Deployment automation script for IDP Layer 3 (Schema Discovery & Web Console):
- Builds & pushes Docker container image to Amazon ECR (106611079163.dkr.ecr.ap-south-1.amazonaws.com/idp-schema-chatbot)
- Registers the ECS Fargate Task Definition (idp-schema-chatbot)
- Deploys/Updates the ECS Fargate Service (idp-schema-chatbot-service) in idp-cluster
- Waits for healthy deployment, queries public ENI IP, and verifies /health
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRATCH_DIR = ROOT_DIR / "scratch"
TASK_DEF_FILE = SCRATCH_DIR / "ecs_task_definition.json"

DEFAULT_REGION = "ap-south-1"
DEFAULT_PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
DEFAULT_CLUSTER = "idp-cluster"
DEFAULT_SERVICE = "idp-schema-chatbot-service"
DEFAULT_ECR_REPO = "idp-schema-chatbot"
DEFAULT_SECURITY_GROUP = "sg-0f34993cb19a0b7f2"
DEFAULT_SUBNETS = [
    "subnet-0d12237b6f3ea123d",
    "subnet-0bbf97bb99124d9a3",
    "subnet-09853bf0b0f48720f",
]


def run_command(cmd, cwd=None, capture: bool = False):
    print(f"  [EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    if capture:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            print(f"  [ERROR] {res.stderr.strip()}", flush=True)
            raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
        return res.stdout.strip()
    else:
        res = subprocess.run(cmd, cwd=cwd, shell=True)
        if res.returncode != 0:
            raise RuntimeError(f"Command failed with code {res.returncode}")
        return ""


def get_session(profile: str | None, region: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def docker_build_and_push(
    session: boto3.Session,
    region: str,
    account_id: str,
    repo_name: str = DEFAULT_ECR_REPO,
    image_tag: str = "latest",
) -> str:
    print(f"\n[0/3] Building & Pushing Docker Image to ECR ({repo_name}:{image_tag})...")
    ecr = session.client("ecr")
    auth_token = ecr.get_authorization_token()
    auth_data = auth_token["authorizationData"][0]
    token = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
    username, password = token.split(":")
    endpoint = auth_data["proxyEndpoint"]

    login_cmd = f"docker login -u {username} -p {password} {endpoint}"
    run_command(login_cmd)

    repo_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}"
    full_image = f"{repo_uri}:{image_tag}"
    build_cmd = f"docker build --platform linux/amd64 -f schema_chatbot_v2/Dockerfile -t {full_image} ."
    run_command(build_cmd, cwd=str(ROOT_DIR))

    push_cmd = f"docker push {full_image}"
    run_command(push_cmd)
    print(f"  Successfully pushed image to {full_image}")
    return full_image


def register_task_definition(ecs_client, task_def_path: Path) -> str:
    print(f"\n[1/3] Registering ECS Task Definition from {task_def_path}...")
    with open(task_def_path, "r", encoding="utf-8") as f:
        task_def_data = json.load(f)

    resp = ecs_client.register_task_definition(
        family=task_def_data["family"],
        networkMode=task_def_data["networkMode"],
        requiresCompatibilities=task_def_data["requiresCompatibilities"],
        cpu=task_def_data["cpu"],
        memory=task_def_data["memory"],
        executionRoleArn=task_def_data["executionRoleArn"],
        taskRoleArn=task_def_data["taskRoleArn"],
        containerDefinitions=task_def_data["containerDefinitions"],
        tags=[
            {"key": "createdby", "value": "vinay.k@shellkode.com"},
            {"key": "customer", "value": "internal"},
            {"key": "purpose", "value": "internal"},
        ],
    )
    task_def_arn = resp["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered Task Definition: {task_def_arn}")
    return task_def_arn


def deploy_ecs_service(
    ecs_client,
    cluster_name: str,
    service_name: str,
    task_def_arn: str,
    subnets: list[str],
    security_groups: list[str],
) -> str:
    print(f"\n[2/3] Deploying ECS Fargate Service '{service_name}' on cluster '{cluster_name}'...")
    try:
        resp = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        active_services = [s for s in resp.get("services", []) if s.get("status") == "ACTIVE"]
        if active_services:
            print(f"  Updating existing service '{service_name}' with new task definition...")
            up_resp = ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=task_def_arn,
                forceNewDeployment=True,
            )
            svc_arn = up_resp["service"]["serviceArn"]
            print(f"  Service update dispatched: {svc_arn}")
            return svc_arn
    except ClientError as e:
        print(f"  Notice checking service: {e}")

    print(f"  Creating new ECS Fargate service '{service_name}'...")
    create_resp = ecs_client.create_service(
        cluster=cluster_name,
        serviceName=service_name,
        taskDefinition=task_def_arn,
        desiredCount=1,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnets,
                "securityGroups": security_groups,
                "assignPublicIp": "ENABLED",
            }
        },
        tags=[
            {"key": "createdby", "value": "vinay.k@shellkode.com"},
            {"key": "customer", "value": "internal"},
        ],
    )
    svc_arn = create_resp["service"]["serviceArn"]
    print(f"  Created ECS Service: {svc_arn}")
    return svc_arn


def wait_and_get_public_ip(
    ecs_client,
    ec2_client,
    cluster_name: str,
    service_name: str,
    task_def_arn: str | None = None,
    timeout_s: int = 300,
) -> str | None:
    print(f"\n[3/3] Waiting for Fargate task to reach RUNNING state (timeout: {timeout_s}s)...")
    start_time = time.time()
    task_arn = None

    while time.time() - start_time < timeout_s:
        task_list = ecs_client.list_tasks(cluster=cluster_name, serviceName=service_name)
        arns = task_list.get("taskArns", [])
        if arns:
            desc = ecs_client.describe_tasks(cluster=cluster_name, tasks=arns)
            tasks = desc.get("tasks", [])
            for t in tasks:
                status = t.get("lastStatus")
                t_def = t.get("taskDefinitionArn")
                print(f"  Task {t['taskArn'].split('/')[-1]} ({t_def.split('/')[-1] if t_def else 'unknown'}) status: {status}")
                if status == "RUNNING" and (task_def_arn is None or t_def == task_def_arn):
                    task_arn = t["taskArn"]
                    break
            if task_arn:
                break
        time.sleep(10)

    if not task_arn:
        print("  Timed out waiting for task to reach RUNNING state.")
        return None

    # Retrieve ENI to find Public IP
    desc = ecs_client.describe_tasks(cluster=cluster_name, tasks=[task_arn])
    containers = desc["tasks"][0]["containers"]
    eni_id = None
    for detail in desc["tasks"][0].get("attachments", []):
        for d in detail.get("details", []):
            if d.get("name") == "networkInterfaceId":
                eni_id = d.get("value")
                break

    if eni_id:
        eni_desc = ec2_client.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
        public_ip = eni_desc["NetworkInterfaces"][0].get("Association", {}).get("PublicIp")
        print(f"\n=======================================================")
        print(f"SUCCESS: Layer 3 Web Console is running in AWS Fargate!")
        print(f"Task ARN  : {task_arn}")
        print(f"Public IP : {public_ip}")
        print(f"URL       : http://{public_ip}:8000/app")
        print(f"Health API: http://{public_ip}:8000/health")
        print(f"=======================================================")
        return public_ip

    print("  Could not locate public IP for task.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy IDP Layer 3 Web Platform to AWS ECS Fargate")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build and push to ECR")
    args = parser.parse_args()

    session = get_session(args.profile, args.region)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    ecs = session.client("ecs")
    ec2 = session.client("ec2")

    if not args.skip_build:
        docker_build_and_push(session, args.region, account_id)

    task_def_arn = register_task_definition(ecs, TASK_DEF_FILE)
    deploy_ecs_service(
        ecs,
        args.cluster,
        args.service,
        task_def_arn,
        DEFAULT_SUBNETS,
        [DEFAULT_SECURITY_GROUP],
    )
    wait_and_get_public_ip(ecs, ec2, args.cluster, args.service, task_def_arn=task_def_arn)


if __name__ == "__main__":
    main()
