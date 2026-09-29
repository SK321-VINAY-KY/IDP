"""
File: deploy_ecs_fargate.py
Purpose: Complete deployment of IDP Layer 3 Web Console & Chatbot to AWS ECS Fargate
         behind an Application Load Balancer (ALB) with mandatory resource tagging.
Tags:
  createdby=vinay.k@shellkode.com
  customer=internal
  purpose=internal
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
DEFAULT_PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
DEFAULT_REGION = "ap-south-1"
REPO_NAME = "idp-layer3-app"
CLUSTER_NAME = "idp-cluster"
SERVICE_NAME = "idp-layer3-service"
TASK_FAMILY = "idp-layer3-task"
ALB_NAME = "idp-web-alb"
TG_NAME = "idp-web-tg"

DEFAULT_TAGS = {
    "createdby": "vinay.k@shellkode.com",
    "customer": "internal",
    "purpose": "internal",
}


def run_command(cmd, cwd=None):
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"  [EXEC] {cmd_str}")
    res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [ERROR] {res.stderr.strip()}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res.stdout.strip()


def parse_tags(tags_str: str | None) -> dict[str, str]:
    raw = tags_str or os.getenv("IDP_AWS_TAGS") or "createdby=vinay.k@shellkode.com,customer=internal,purpose=internal"
    tags = {}
    for part in raw.strip().split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


def to_aws_tags(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in tags.items()]


# ── 1. ECR Repository & Docker Image Build / Push ──────────────────────────

def ensure_ecr_repo(ecr_client, repo_name: str, tags: dict[str, str]) -> str:
    print(f"\n--- [1/7] Ensuring ECR Repository '{repo_name}' ---")
    try:
        resp = ecr_client.describe_repositories(repositoryNames=[repo_name])
        repo_uri = resp["repositories"][0]["repositoryUri"]
        print(f"  ECR repo exists: {repo_uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            resp = ecr_client.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={"scanOnPush": True},
                tags=to_aws_tags(tags),
            )
            repo_uri = resp["repository"]["repositoryUri"]
            print(f"  Created ECR repo: {repo_uri}")
        else:
            raise

    # Ensure repository tags
    try:
        arn = ecr_client.describe_repositories(repositoryNames=[repo_name])["repositories"][0]["repositoryArn"]
        ecr_client.tag_resource(resourceArn=arn, tags=to_aws_tags(tags))
    except Exception as exc:
        print(f"  Notice: ECR tagging update: {exc}")

    return repo_uri


def docker_build_and_push(session, region: str, repo_uri: str, image_tag: str = "latest") -> str:
    print("\n--- [2/7] Docker Build & Push to ECR ---")
    ecr = session.client("ecr")
    auth_token = ecr.get_authorization_token()
    auth_data = auth_token["authorizationData"][0]
    token = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
    username, password = token.split(":")
    endpoint = auth_data["proxyEndpoint"]

    print("  Logging in to Amazon ECR...")
    run_command(f"docker login -u {username} -p {password} {endpoint}")

    full_image_uri = f"{repo_uri}:{image_tag}"
    print(f"  Building Docker image: {full_image_uri}...")
    run_command(f"docker build -f schema_chatbot_v2/Dockerfile -t {full_image_uri} .", cwd=str(ROOT_DIR))

    print(f"  Pushing image to ECR: {full_image_uri}...")
    run_command(f"docker push {full_image_uri}")
    print(f"  [SUCCESS] Pushed image {full_image_uri}")
    return full_image_uri


# ── 2. IAM Roles for ECS ───────────────────────────────────────────────────

def ensure_ecs_execution_role(iam_client, role_name: str, tags: dict[str, str]) -> str:
    print(f"\n--- [3/7] Ensuring ECS Task Execution IAM Role '{role_name}' ---")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        resp = iam_client.get_role(RoleName=role_name)
        role_arn = resp["Role"]["Arn"]
        print(f"  Role exists: {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            resp = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="ECS Task Execution Role for IDP Layer 3 Web Console",
                Tags=to_aws_tags(tags),
            )
            role_arn = resp["Role"]["Arn"]
            print(f"  Created role: {role_arn}")
            time.sleep(5)
        else:
            raise

    # Attach AmazonECSTaskExecutionRolePolicy
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    return role_arn


# ── 3. CloudWatch Log Group ────────────────────────────────────────────────

def ensure_cloudwatch_logs(logs_client, log_group_name: str, tags: dict[str, str]):
    try:
        logs_client.create_log_group(
            logGroupName=log_group_name,
            tags=tags,
        )
        print(f"  Created CloudWatch log group: {log_group_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
            print(f"  CloudWatch log group already exists: {log_group_name}")
            try:
                logs_client.tag_log_group(logGroupName=log_group_name, tags=tags)
            except Exception:
                pass
        else:
            raise


# ── 4. Networking: VPC, Subnets & Security Groups ──────────────────────────

def setup_networking(ec2_client, tags: dict[str, str]) -> tuple[str, list[str], str, str]:
    print("\n--- [4/7] Configuring VPC & Security Groups ---")
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        vpcs = ec2_client.describe_vpcs()["Vpcs"]
    if not vpcs:
        raise RuntimeError("No VPC found in target region.")

    vpc_id = vpcs[0]["VpcId"]
    print(f"  Using VPC: {vpc_id}")

    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    subnet_ids = [s["SubnetId"] for s in subnets]
    if len(subnet_ids) < 2:
        raise RuntimeError(f"ALB requires at least 2 subnets across AZs; found {len(subnet_ids)}.")
    print(f"  Available Subnets: {subnet_ids[:3]}")

    # Security Group for ALB (inbound 80 from everywhere)
    alb_sg_name = "idp-alb-sg"
    try:
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [alb_sg_name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )
        if resp["SecurityGroups"]:
            alb_sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"  ALB Security Group exists: {alb_sg_id}")
        else:
            create_resp = ec2_client.create_security_group(
                GroupName=alb_sg_name,
                Description="Security Group for IDP Application Load Balancer",
                VpcId=vpc_id,
                TagSpecifications=[{"ResourceType": "security-group", "Tags": to_aws_tags(tags)}],
            )
            alb_sg_id = create_resp["GroupId"]
            ec2_client.authorize_security_group_ingress(
                GroupId=alb_sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTP public access"}],
                    }
                ],
            )
            print(f"  Created ALB Security Group: {alb_sg_id}")
    except ClientError as e:
        print(f"  Notice during ALB SG setup: {e}")
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [alb_sg_name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )
        alb_sg_id = resp["SecurityGroups"][0]["GroupId"]

    # Security Group for ECS Task (inbound 8000 from ALB SG)
    ecs_sg_name = "idp-ecs-task-sg"
    try:
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [ecs_sg_name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )
        if resp["SecurityGroups"]:
            ecs_sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"  ECS Task Security Group exists: {ecs_sg_id}")
        else:
            create_resp = ec2_client.create_security_group(
                GroupName=ecs_sg_name,
                Description="Security Group for IDP ECS Fargate Task",
                VpcId=vpc_id,
                TagSpecifications=[{"ResourceType": "security-group", "Tags": to_aws_tags(tags)}],
            )
            ecs_sg_id = create_resp["GroupId"]
            ec2_client.authorize_security_group_ingress(
                GroupId=ecs_sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "UserIdGroupPairs": [{"GroupId": alb_sg_id, "Description": "Allow from ALB"}],
                    }
                ],
            )
            print(f"  Created ECS Task Security Group: {ecs_sg_id}")
    except ClientError as e:
        print(f"  Notice during ECS SG setup: {e}")
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [ecs_sg_name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )
        ecs_sg_id = resp["SecurityGroups"][0]["GroupId"]

    return vpc_id, subnet_ids, alb_sg_id, ecs_sg_id


# ── 5. Application Load Balancer & Target Group ─────────────────────────────

def ensure_alb_and_target_group(
    elbv2_client, vpc_id: str, subnet_ids: list[str], alb_sg_id: str, tags: dict[str, str]
) -> tuple[str, str, str]:
    print("\n--- [5/7] Configuring Application Load Balancer & Target Group ---")
    aws_tags = to_aws_tags(tags)

    # 1. Target Group
    try:
        resp = elbv2_client.describe_target_groups(Names=[TG_NAME])
        tg_arn = resp["TargetGroups"][0]["TargetGroupArn"]
        print(f"  Target Group exists: {tg_arn}")
    except ClientError as e:
        if "TargetGroupNotFound" in str(e):
            resp = elbv2_client.create_target_group(
                Name=TG_NAME,
                Protocol="HTTP",
                Port=8000,
                VpcId=vpc_id,
                TargetType="ip",
                HealthCheckProtocol="HTTP",
                HealthCheckPort="8000",
                HealthCheckPath="/health",
                HealthCheckIntervalSeconds=15,
                HealthCheckTimeoutSeconds=5,
                HealthyThresholdCount=2,
                UnhealthyThresholdCount=3,
                Tags=aws_tags,
            )
            tg_arn = resp["TargetGroups"][0]["TargetGroupArn"]
            print(f"  Created Target Group: {tg_arn}")
        else:
            raise

    # 2. Load Balancer
    try:
        resp = elbv2_client.describe_load_balancers(Names=[ALB_NAME])
        alb_arn = resp["LoadBalancers"][0]["LoadBalancerArn"]
        alb_dns = resp["LoadBalancers"][0]["DNSName"]
        print(f"  ALB exists: {alb_arn}")
        print(f"  ALB DNS: http://{alb_dns}")
    except ClientError as e:
        if "LoadBalancerNotFound" in str(e):
            resp = elbv2_client.create_load_balancer(
                Name=ALB_NAME,
                Subnets=subnet_ids[:3],
                SecurityGroups=[alb_sg_id],
                Scheme="internet-facing",
                Type="application",
                IpAddressType="ipv4",
                Tags=aws_tags,
            )
            alb_arn = resp["LoadBalancers"][0]["LoadBalancerArn"]
            alb_dns = resp["LoadBalancers"][0]["DNSName"]
            print(f"  Created ALB: {alb_arn}")
            print(f"  ALB DNS: http://{alb_dns}")
        else:
            raise

    # 3. Listener on Port 80
    listeners = elbv2_client.describe_listeners(LoadBalancerArn=alb_arn)["Listeners"]
    listener_80 = next((l for l in listeners if l["Port"] == 80), None)
    if not listener_80:
        elbv2_client.create_listener(
            LoadBalancerArn=alb_arn,
            Protocol="HTTP",
            Port=80,
            DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
            Tags=aws_tags,
        )
        print("  Created ALB HTTP listener on port 80 forwarding to target group.")

    return alb_arn, alb_dns, tg_arn


# ── 6. ECS Cluster, Task Definition & Fargate Service ──────────────────────

def ensure_ecs_cluster(ecs_client, cluster_name: str, tags: dict[str, str]) -> str:
    print(f"\n--- [6/7] Ensuring ECS Cluster '{cluster_name}' ---")
    try:
        resp = ecs_client.describe_clusters(clusters=[cluster_name])
        existing = [c for c in resp["clusters"] if c["status"] == "ACTIVE"]
        if existing:
            cluster_arn = existing[0]["clusterArn"]
            print(f"  ECS Cluster exists: {cluster_arn}")
            return cluster_arn
    except Exception:
        pass

    resp = ecs_client.create_cluster(
        clusterName=cluster_name,
        tags=to_aws_tags(tags),
    )
    cluster_arn = resp["cluster"]["clusterArn"]
    print(f"  Created ECS Cluster: {cluster_arn}")
    return cluster_arn


def register_task_definition(
    ecs_client, execution_role_arn: str, image_uri: str, region: str, tags: dict[str, str]
) -> str:
    print(f"  Registering ECS Fargate Task Definition '{TASK_FAMILY}'...")
    log_group = f"/ecs/{REPO_NAME}"
    container_def = {
        "name": "idp-web-console",
        "image": image_uri,
        "essential": True,
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "environment": [
            {"name": "APP_ENV", "value": "production"},
            {"name": "PORT", "value": "8000"},
            {"name": "PYTHONPATH", "value": "/app/schema_chatbot_v2:/app"},
            {"name": "LOG_LEVEL", "value": "INFO"},
        ],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": region,
                "awslogs-stream-prefix": "web",
            },
        },
        "healthCheck": {
            "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
            "interval": 15,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 30,
        },
    }

    resp = ecs_client.register_task_definition(
        family=TASK_FAMILY,
        executionRoleArn=execution_role_arn,
        networkMode="awsvpc",
        containerDefinitions=[container_def],
        requiresCompatibilities=["FARGATE"],
        cpu="1024",     # 1 vCPU
        memory="2048",  # 2 GB RAM
        tags=to_aws_tags(tags),
    )
    task_def_arn = resp["taskDefinition"]["taskDefinitionArn"]
    print(f"  Registered Task Definition: {task_def_arn}")
    return task_def_arn


def ensure_fargate_service(
    ecs_client,
    cluster_name: str,
    service_name: str,
    task_def_arn: str,
    tg_arn: str,
    subnet_ids: list[str],
    ecs_sg_id: str,
    tags: dict[str, str],
):
    print(f"\n--- [7/7] Deploying ECS Fargate Service '{service_name}' ---")
    aws_tags = to_aws_tags(tags)
    try:
        resp = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        active = [s for s in resp["services"] if s["status"] == "ACTIVE"]
    except Exception:
        active = []

    if active:
        print(f"  Service '{service_name}' exists. Updating task definition...")
        ecs_client.update_service(
            cluster=cluster_name,
            service=service_name,
            taskDefinition=task_def_arn,
            forceNewDeployment=True,
        )
        print(f"  [SUCCESS] Triggered service update for {service_name}.")
    else:
        print(f"  Creating new Fargate Service '{service_name}'...")
        ecs_client.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_def_arn,
            launchType="FARGATE",
            desiredCount=1,
            loadBalancers=[
                {
                    "targetGroupArn": tg_arn,
                    "containerName": "idp-web-console",
                    "containerPort": 8000,
                }
            ],
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": subnet_ids[:3],
                    "securityGroups": [ecs_sg_id],
                    "assignPublicIp": "ENABLED",
                }
            },
            tags=aws_tags,
        )
        print(f"  [SUCCESS] Created ECS Fargate Service: {service_name}")


# ── Main Entrypoint ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy IDP Layer 3 to AWS ECS Fargate behind ALB")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    parser.add_argument(
        "--tags",
        default="createdby=vinay.k@shellkode.com,customer=internal,purpose=internal",
        help="Comma-separated tags e.g. 'createdby=...,customer=...,purpose=...'",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build and push")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    tags = parse_tags(args.tags)

    print("==============================================================")
    print(f"  IDP Layer 3 ECS Fargate Deployment: Account {account_id} ({args.region})")
    print(f"  Mandatory Tags: {tags}")
    print("==============================================================")

    # 1. ECR
    ecr = session.client("ecr")
    repo_uri = ensure_ecr_repo(ecr, REPO_NAME, tags)

    # 2. Build & Push
    image_uri = f"{repo_uri}:latest"
    if not args.skip_build:
        image_uri = docker_build_and_push(session, args.region, repo_uri, "latest")

    # 3. IAM Role & CloudWatch
    iam = session.client("iam")
    execution_role_arn = ensure_ecs_execution_role(iam, "idp-ecs-task-execution-role", tags)

    logs = session.client("logs")
    ensure_cloudwatch_logs(logs, f"/ecs/{REPO_NAME}", tags)

    # 4. Networking
    ec2 = session.client("ec2")
    vpc_id, subnet_ids, alb_sg_id, ecs_sg_id = setup_networking(ec2, tags)

    # 5. ALB & Target Group
    elbv2 = session.client("elbv2")
    alb_arn, alb_dns, tg_arn = ensure_alb_and_target_group(elbv2, vpc_id, subnet_ids, alb_sg_id, tags)

    # 6. ECS Cluster & Task Def
    ecs = session.client("ecs")
    cluster_arn = ensure_ecs_cluster(ecs, CLUSTER_NAME, tags)
    task_def_arn = register_task_definition(ecs, execution_role_arn, image_uri, args.region, tags)

    # 7. Fargate Service
    ensure_fargate_service(
        ecs,
        CLUSTER_NAME,
        SERVICE_NAME,
        task_def_arn,
        tg_arn,
        subnet_ids,
        ecs_sg_id,
        tags,
    )

    print("\n==============================================================")
    print("  Deployment Succeeded!")
    print(f"  Application Load Balancer URL: http://{alb_dns}")
    print(f"  Health Endpoint:              http://{alb_dns}/health")
    print(f"  Web UI / Dashboard:           http://{alb_dns}/app")
    print("==============================================================")


if __name__ == "__main__":
    main()
