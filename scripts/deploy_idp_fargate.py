"""
Stage 4: create ECS cluster + Fargate task def + service (ALB) in account
530084039784 (ap-south-1), all tagged.

Task env injects the single inference profile as BOTH LLM and VLM, the RDS URL,
the S3 bucket, and the 4 Lambda ARNs. Requires the L3 app image pushed (Stage 2)
and the Lambdas created (Stage 3) — pass the lambda ARNs via --lambdas-arns.

Reads AWS creds from env (.env2).
"""
import argparse
import base64
import json
import os
import time

import boto3
from botocore.exceptions import ClientError

REGION = "ap-south-1"
ACCOUNT = "530084039784"
VPC_ID = "vpc-0571f21ae29ec7a1d"
SUBNETS = ["subnet-05a694d5be10434f3", "subnet-00af2fd62f81c622e", "subnet-054bd923089c27d5a"]
DB_ENDPOINT = os.getenv("IDP_DB_ENDPOINT", "idp-postgres-db.c30agqwq4v2n.ap-south-1.rds.amazonaws.com")
# Secrets are read from the environment — never hardcode them (this file is committed to git).
DB_USER = os.getenv("IDP_DB_USER", "idp_admin")
DB_PASSWORD = os.environ["IDP_DB_PASSWORD"]  # required; set before running
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_ENDPOINT}:5432/idp"
INFERENCE_PROFILE = "arn:aws:bedrock:ap-south-1:530084039784:application-inference-profile/24n3l11zddro"
S3_BUCKET = "idp-internal-35385"
ECR = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
L3_IMAGE = f"{ECR}/idp-layer3-app:latest"

CLUSTER = "idp-cluster"
SERVICE = "idp-layer3-service"
TASK_FAMILY = "idp-layer3-task"
REPO = "idp-layer3-app"
ALB_NAME = "idp-web-alb"
TG_NAME = "idp-web-tg"

TAGS = {
    "createdby": "vinay.k@shellkode.com",
    "customer": "idp-internal-35385",
    "purpose": "internal",
}
TAGLIST = [{"Key": k, "Value": v} for k, v in TAGS.items()]      # for most services
ECS_TAGS = [{"key": k, "value": v} for k, v in TAGS.items()]     # ECS uses lowercase


def run(cmd):
    print(f"[EXEC] {cmd}")
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {r.stderr}")
    return r.stdout.strip()


def ensure_exec_role(iam):
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    name = "idp-ecs-task-execution-role"
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        print(f"[iam] exec role exists: {arn}")
        iam.tag_role(RoleName=name, Tags=TAGLIST)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="IDP ECS task execution role", Tags=TAGLIST)["Role"]["Arn"]
        print(f"[iam] created exec role: {arn}")
        time.sleep(6)
    iam.attach_role_policy(RoleName=name, PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy")
    return arn


def ensure_task_role(iam):
    """Task role: Bedrock invoke, S3 r/w, Lambda invoke (L1+L2), CloudWatch."""
    name = "idp-ecs-task-role"
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        print(f"[iam] task role exists: {arn}")
        iam.tag_role(RoleName=name, Tags=TAGLIST)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="IDP ECS task role (Bedrock+S3+Lambda)", Tags=TAGLIST)["Role"]["Arn"]
        print(f"[iam] created task role: {arn}")
        time.sleep(6)
    pol = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
             "Resource": [INFERENCE_PROFILE, f"arn:aws:bedrock:{REGION}:{ACCOUNT}:application-inference-profile/*"]},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
             "Resource": [f"arn:aws:s3:::{S3_BUCKET}", f"arn:aws:s3:::{S3_BUCKET}/*"]},
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"],
             "Resource": [f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-layer1-router",
                          f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-*"]},
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "*"},
        ],
    }
    iam.put_role_policy(RoleName=name, PolicyName="idp-task-permissions", PolicyDocument=json.dumps(pol))
    print("[iam] task role policy attached")
    return arn


def ensure_logs(logs):
    grp = f"/ecs/{REPO}"
    try:
        logs.create_log_group(logGroupName=grp, tags=TAGS)
        print(f"[logs] created {grp}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
            print(f"[logs] exists {grp}")
        else:
            raise


def ensure_networking(ec2):
    vpcs = ec2.describe_vpcs(VpcIds=[VPC_ID])["Vpcs"]
    assert vpcs, "target VPC not found"
    print(f"[net] vpc {VPC_ID}")
    # ALB SG
    alb_sg = "idp-alb-sg"
    res = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [alb_sg]}, {"Name": "vpc-id", "Values": [VPC_ID]}])
    if res.get("SecurityGroups"):
        alb_sgid = res["SecurityGroups"][0]["GroupId"]
    else:
        alb_sgid = ec2.create_security_group(GroupName=alb_sg, Description="IDP ALB SG", VpcId=VPC_ID, TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGLIST}])["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=alb_sgid, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    print(f"[net] alb-sg {alb_sgid}")
    # ECS SG
    ecs_sg = "idp-ecs-task-sg"
    res = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [ecs_sg]}, {"Name": "vpc-id", "Values": [VPC_ID]}])
    if res.get("SecurityGroups"):
        ecs_sgid = res["SecurityGroups"][0]["GroupId"]
    else:
        ecs_sgid = ec2.create_security_group(GroupName=ecs_sg, Description="IDP ECS task SG", VpcId=VPC_ID, TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGLIST}])["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=ecs_sgid, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "UserIdGroupPairs": [{"GroupId": alb_sgid}]}])
    print(f"[net] ecs-sg {ecs_sgid}")
    return alb_sgid, ecs_sgid


def ensure_alb(elbv2, alb_sgid):
    # target group
    try:
        tg_arn = elbv2.describe_target_groups(Names=[TG_NAME])["TargetGroups"][0]["TargetGroupArn"]
        print(f"[alb] tg exists {tg_arn}")
    except ClientError:
        tg_arn = elbv2.create_target_group(Name=TG_NAME, Protocol="HTTP", Port=8000, VpcId=VPC_ID, TargetType="ip",
                                          HealthCheckProtocol="HTTP", HealthCheckPort="8000", HealthCheckPath="/health",
                                          HealthCheckIntervalSeconds=15, HealthCheckTimeoutSeconds=5,
                                          HealthyThresholdCount=2, UnhealthyThresholdCount=3, Tags=TAGLIST)["TargetGroups"][0]["TargetGroupArn"]
        print(f"[alb] created tg {tg_arn}")
    # ALB
    try:
        lb = elbv2.describe_load_balancers(Names=[ALB_NAME])["LoadBalancers"][0]
        alb_arn, dns = lb["LoadBalancerArn"], lb["DNSName"]
        print(f"[alb] exists {alb_arn}")
    except ClientError:
        lb = elbv2.create_load_balancer(Name=ALB_NAME, Subnets=SUBNETS, SecurityGroups=[alb_sgid],
                                       Scheme="internet-facing", Type="application", IpAddressType="ipv4", Tags=TAGLIST)["LoadBalancers"][0]
        alb_arn, dns = lb["LoadBalancerArn"], lb["DNSName"]
        print(f"[alb] created {alb_arn}")
    # listener
    if not any(l.get("Port") == 80 for l in elbv2.describe_listeners(LoadBalancerArn=alb_arn)["Listeners"]):
        elbv2.create_listener(LoadBalancerArn=alb_arn, Protocol="HTTP", Port=80,
                             DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}], Tags=TAGLIST)
        print("[alb] created listener 80")
    return alb_arn, dns, tg_arn


def register_task(ecs, exec_arn, task_arn):
    env = [
        {"name": "APP_ENV", "value": "production"},
        {"name": "PYTHONPATH", "value": "/app/schema_chatbot_v2:/app"},
        {"name": "IDP_APP_ENV", "value": "production"},
        {"name": "AWS_DEFAULT_REGION", "value": REGION},
        {"name": "BEDROCK_REGION", "value": REGION},
        {"name": "IDP_BEDROCK_REGION", "value": REGION},
        {"name": "LLM_PROVIDER", "value": "bedrock"},
        {"name": "IDP_LLM_PROVIDER", "value": "bedrock"},
        {"name": "EXTRACTION_BACKEND", "value": "bedrock"},
        {"name": "IDP_EXTRACTION_BACKEND", "value": "bedrock"},
        {"name": "BEDROCK_MODEL_ID", "value": INFERENCE_PROFILE},
        {"name": "IDP_BEDROCK_MODEL_ID", "value": INFERENCE_PROFILE},
        {"name": "BEDROCK_VLM_MODEL_ID", "value": INFERENCE_PROFILE},
        {"name": "IDP_BEDROCK_VLM_MODEL_ID", "value": INFERENCE_PROFILE},
        {"name": "VLM_MODEL_NAME", "value": INFERENCE_PROFILE},
        {"name": "IDP_VLM_MODEL_NAME", "value": INFERENCE_PROFILE},
        {"name": "BEDROCK_REASONING_EFFORT", "value": "low"},
        {"name": "LAYER3_STRATEGY", "value": "graph_memory_concurrent"},
        {"name": "IDP_LAYER3_STRATEGY", "value": "graph_memory_concurrent"},
        {"name": "DATABASE_URL", "value": DB_URL},
        {"name": "IDP_DATABASE_URL", "value": DB_URL},
        {"name": "IDP_S3_BUCKET", "value": S3_BUCKET},
        {"name": "S3_BUCKET_NAME", "value": S3_BUCKET},
        {"name": "PADDLE_PREFER_LAMBDA", "value": "true"},
        {"name": "IDP_PADDLE_PREFER_LAMBDA", "value": "true"},
        {"name": "PADDLE_LAMBDA_REGION", "value": REGION},
        {"name": "IDP_PADDLE_LAMBDA_REGION", "value": REGION},
        {"name": "PADDLE_PRINTED_LAMBDA_FUNCTION", "value": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-paddle-printed"},
        {"name": "IDP_PADDLE_PRINTED_LAMBDA_FUNCTION", "value": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-paddle-printed"},
        {"name": "PADDLE_HANDWRITTEN_LAMBDA_FUNCTION", "value": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-paddle-handwritten"},
        {"name": "IDP_PADDLE_HANDWRITTEN_LAMBDA_FUNCTION", "value": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:idp-engine-paddle-handwritten"},
        {"name": "ENABLE_VLM_DIRECT", "value": "false"},
        {"name": "IDP_ENABLE_VLM_DIRECT", "value": "false"},
        {"name": "ROUTING_MODE", "value": "capability_based"},
        {"name": "IDP_ROUTING_MODE", "value": "capability_based"},
        {"name": "GRAPH_CONCURRENCY_LIMIT", "value": "16"},
        {"name": "PAGE_CONCURRENCY_LIMIT", "value": "25"},
        {"name": "ESCALATION_CONFIDENCE_THRESHOLD", "value": "0.70"},
        {"name": "JWT_SECRET", "value": os.environ["IDP_JWT_SECRET"]},
        {"name": "ADMIN_PASSWORD", "value": os.environ["IDP_ADMIN_PASSWORD"]},
        {"name": "USER_STORE_TYPE", "value": "postgres"},
        {"name": "ALLOW_FILE_USER_STORE_IN_PROD", "value": "true"},
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "CORS_ORIGINS", "value": "http://idp-web-alb-1724249754.ap-south-1.elb.amazonaws.com,http://localhost:8000,http://127.0.0.1:8000"},
    ]
    cd = {
        "name": "idp-web-console", "image": L3_IMAGE, "essential": True,
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "environment": env,
        "logConfiguration": {"logDriver": "awslogs", "options": {"awslogs-group": f"/ecs/{REPO}", "awslogs-region": REGION, "awslogs-stream-prefix": "web"}},
        "healthCheck": {"command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"], "interval": 15, "timeout": 5, "retries": 3, "startPeriod": 30},
    }
    resp = ecs.register_task_definition(family=TASK_FAMILY, executionRoleArn=exec_arn, taskRoleArn=task_arn,
                                        networkMode="awsvpc", containerDefinitions=[cd],
                                        requiresCompatibilities=["FARGATE"], cpu="2048", memory="8192", tags=ECS_TAGS)
    arn = resp["taskDefinition"]["taskDefinitionArn"]
    print(f"[ecs] task def registered: {arn}")
    return arn


def ensure_service(ecs, task_arn, tg_arn, ecs_sgid):
    try:
        resp = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])
        active = [s for s in resp["services"] if s["status"] == "ACTIVE"]
    except Exception:
        active = []
    if active:
        ecs.update_service(cluster=CLUSTER, service=SERVICE, taskDefinition=task_arn, forceNewDeployment=True)
        print(f"[ecs] service {SERVICE} updated")
    else:
        ecs.create_service(cluster=CLUSTER, serviceName=SERVICE, taskDefinition=task_arn, launchType="FARGATE",
                          desiredCount=1, loadBalancers=[{"targetGroupArn": tg_arn, "containerName": "idp-web-console", "containerPort": 8000}],
                          networkConfiguration={"awsvpcConfiguration": {"subnets": SUBNETS, "securityGroups": [ecs_sgid], "assignPublicIp": "ENABLED"}},
                          tags=ECS_TAGS)
        print(f"[ecs] service {SERVICE} created")


def main():
    session = boto3.Session(region_name=REGION)
    acct = session.client("sts").get_caller_identity()["Account"]
    assert acct == ACCOUNT, f"WRONG ACCOUNT {acct}"
    iam = session.client("iam")
    logs = session.client("logs")
    ec2 = session.client("ec2")
    elbv2 = session.client("elbv2")
    ecs = session.client("ecs")

    print(f"=== Stage 4 Fargate :: {ACCOUNT} {REGION} ===")
    ensure_logs(logs)
    exec_arn = ensure_exec_role(iam)
    task_arn = ensure_task_role(iam)
    alb_sgid, ecs_sgid = ensure_networking(ec2)
    alb_arn, dns, tg_arn = ensure_alb(elbv2, alb_sgid)

    try:
        ecs.describe_clusters(clusters=[CLUSTER])["clusters"][0]
        print(f"[ecs] cluster exists {CLUSTER}")
        ecs.tag_resource(resourceArn=f"arn:aws:ecs:{REGION}:{ACCOUNT}:cluster/{CLUSTER}", tags=ECS_TAGS)
    except Exception:
        ecs.create_cluster(clusterName=CLUSTER, tags=ECS_TAGS)
        print(f"[ecs] cluster created {CLUSTER}")

    td_arn = register_task(ecs, exec_arn, task_arn)
    ensure_service(ecs, td_arn, tg_arn, ecs_sgid)
    print(f"\n=== DONE ===\nALB: http://{dns}\nHealth: http://{dns}/health\nUI: http://{dns}/app")


if __name__ == "__main__":
    main()
