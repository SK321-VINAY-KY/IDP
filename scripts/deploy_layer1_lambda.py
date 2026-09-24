"""
Automated deployment script for IDP Layer 1 AWS Lambda function ('idp-layer1-router').
Supports multi-account deployments, mandatory resource tagging, custom S3 buckets,
custom regions, and named AWS profiles.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRATCH_DIR = ROOT_DIR / "scratch"
BUILD_DIR = SCRATCH_DIR / "lambda_layer1_build"
ZIP_PATH = SCRATCH_DIR / "layer1_lambda.zip"
S3_PACKAGE_KEY = "lambda-deployments/layer1_lambda.zip"


def run_command(cmd, cwd=None):
    print(f"  [EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [ERROR] {res.stderr.strip()}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr}")
    return res.stdout.strip()


def parse_tags(tags_str: str | None) -> dict[str, str]:
    """
    Parses tags from CLI string or environment variable:
    Formats supported:
      "Project=IDP,Environment=dev,Owner=Vinay,CostCenter=101"
      '{"Project": "IDP", "Environment": "dev"}'
    """
    raw = tags_str or os.getenv("IDP_AWS_TAGS") or ""
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    tags = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


def get_session(profile: str | None, region: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def validate_aws_environment(
    session: boto3.Session, bucket_name: str | None, region: str, tags: dict[str, str]
) -> tuple[str, str]:
    print("\n--- [Step 1/6] Validating AWS Environment ---")
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]
    arn = identity["Arn"]
    print(f"  Caller ARN:  {arn}")
    print(f"  Account ID:  {account_id}")
    print(f"  Region:      {region}")

    target_bucket = bucket_name or os.getenv("IDP_S3_BUCKET") or f"idp-documents-{account_id}"
    print(f"  Target S3:   s3://{target_bucket}")

    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=target_bucket)
        print(f"  S3 Bucket '{target_bucket}' exists and is accessible.")
    except ClientError as e:
        error_code = str(e.response["Error"]["Code"])
        if error_code in ("404", "NoSuchBucket"):
            print(f"  Bucket '{target_bucket}' does not exist. Creating bucket in {region}...")
            if region == "us-east-1":
                s3.create_bucket(Bucket=target_bucket)
            else:
                s3.create_bucket(
                    Bucket=target_bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            print(f"  Bucket '{target_bucket}' created successfully.")
        else:
            print(f"  Note: Bucket check returned: {e}. Proceeding.")

    # Apply S3 bucket tagging if tags provided
    if tags:
        try:
            s3.put_bucket_tagging(
                Bucket=target_bucket,
                Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in tags.items()]},
            )
            print(f"  Applied tags to S3 bucket '{target_bucket}': {tags}")
        except Exception as e:
            print(f"  Note: S3 bucket tagging returned: {e}")

    return account_id, target_bucket


def build_package():
    print("\n--- [Step 2/6] Building Linux-compatible Layer 1 Package ---")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and install Linux manylinux wheels for Python 3.11 (if not already cached)
    if not (BUILD_DIR / "pymupdf").exists():
        print("  Downloading Linux x86_64 wheels for PyMuPDF, Pydantic, Dotenv...")
        pip_cmd = (
            f'"{sys.executable}" -m pip install '
            f"--platform manylinux2014_x86_64 "
            f"--target \"{BUILD_DIR}\" "
            f"--implementation cp "
            f"--python-version 3.11 "
            f"--only-binary=:all: "
            f"pymupdf pydantic pydantic-settings python-dotenv"
        )
        run_command(pip_cmd)
    else:
        print("  Using cached Linux x86_64 dependency wheels in build directory.")

    # 2. Copy source modules required by Layer 1
    print("  Copying Layer 1 source code...")
    modules_to_copy = [
        ("src/api/__init__.py", "src/api/__init__.py"),
        ("src/api/lambda_layer1_handler.py", "src/api/lambda_layer1_handler.py"),
        ("src/__init__.py", "src/__init__.py"),
        ("src/ai/__init__.py", "src/ai/__init__.py"),
        ("src/ai/layer1_routing/__init__.py", "src/ai/layer1_routing/__init__.py"),
        ("src/ai/layer1_routing/inspect.py", "src/ai/layer1_routing/inspect.py"),
        ("src/ai/layer1_routing/router.py", "src/ai/layer1_routing/router.py"),
        ("src/ai/layer1_routing/capability_router.py", "src/ai/layer1_routing/capability_router.py"),
        ("src/ai/layer1_routing/capability_types.py", "src/ai/layer1_routing/capability_types.py"),
        ("src/ai/schemas/__init__.py", "src/ai/schemas/__init__.py"),
        ("src/ai/schemas/page.py", "src/ai/schemas/page.py"),
        ("src/config/__init__.py", "src/config/__init__.py"),
        ("src/config/settings.py", "src/config/settings.py"),
        ("src/utils/__init__.py", "src/utils/__init__.py"),
        ("src/utils/logger.py", "src/utils/logger.py"),
    ]

    for src_rel, dest_rel in modules_to_copy:
        src_path = ROOT_DIR / src_rel
        dest_path = BUILD_DIR / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
        else:
            dest_path.touch()

    # 3. Create ZIP archive
    print(f"  Compressing into {ZIP_PATH}...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    uncompressed_size = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in BUILD_DIR.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(BUILD_DIR)
                zf.write(file, arcname)
                uncompressed_size += file.stat().st_size

    zip_size = ZIP_PATH.stat().st_size
    print(f"  Package Uncompressed Size: {uncompressed_size / (1024 * 1024):.2f} MB")
    print(f"  Package Compressed Size:   {zip_size / (1024 * 1024):.2f} MB")

    return zip_size, uncompressed_size


def upload_to_s3(session: boto3.Session, bucket_name: str):
    print(f"\n--- [Step 3/6] Archiving Package to S3 ---")
    s3 = session.client("s3")
    dest_uri = f"s3://{bucket_name}/{S3_PACKAGE_KEY}"
    print(f"  Uploading {ZIP_PATH.name} -> {dest_uri}...")
    try:
        s3.upload_file(str(ZIP_PATH), bucket_name, S3_PACKAGE_KEY)
        print(f"  Upload to {bucket_name} complete.")
    except Exception as e:
        print(f"  Warning: S3 archive upload failed: {e}. (Lambda will still deploy via direct ZipFile payload)")


def ensure_cloudwatch_log_group(session: boto3.Session, function_name: str, tags: dict[str, str]):
    logs = session.client("logs")
    log_group_name = f"/aws/lambda/{function_name}"
    print(f"  Configuring CloudWatch log group '{log_group_name}'...")
    try:
        kwargs = {"logGroupName": log_group_name}
        if tags:
            kwargs["tags"] = tags
        logs.create_log_group(**kwargs)
        print(f"  Created CloudWatch log group '{log_group_name}' with tags: {tags}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
            if tags:
                try:
                    logs.tag_log_group(logGroupName=log_group_name, tags=tags)
                    print(f"  Tagged existing CloudWatch log group '{log_group_name}'.")
                except Exception as tag_err:
                    print(f"  Note on log group tagging: {tag_err}")
        else:
            print(f"  Note on log group creation: {e}")


def ensure_iam_role(session: boto3.Session, role_name: str, bucket_name: str, tags: dict[str, str]) -> str:
    print(f"\n--- [Step 4/6] Configuring IAM Execution Role '{role_name}' ---")
    iam = session.client("iam")

    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        role_resp = iam.get_role(RoleName=role_name)
        role_arn = role_resp["Role"]["Arn"]
        print(f"  Role already exists: {role_arn}")
        if tags:
            try:
                iam.tag_role(RoleName=role_name, Tags=[{"Key": k, "Value": v} for k, v in tags.items()])
                print(f"  Updated tags on role '{role_name}': {tags}")
            except Exception as tag_err:
                print(f"  Note on role tagging: {tag_err}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"  Creating role '{role_name}' with tags...")
            create_role_kwargs = {
                "RoleName": role_name,
                "AssumeRolePolicyDocument": json.dumps(assume_role_policy),
                "Description": "Execution role for IDP Layer 1 Lambda Router",
            }
            if tags:
                create_role_kwargs["Tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]
            role_resp = iam.create_role(**create_role_kwargs)
            role_arn = role_resp["Role"]["Arn"]
            print(f"  Created role: {role_arn} with tags: {tags}")
            time.sleep(5)
        else:
            raise

    # Attach AWSLambdaBasicExecutionRole
    basic_policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    iam.attach_role_policy(RoleName=role_name, PolicyArn=basic_policy_arn)
    print("  Attached managed policy: AWSLambdaBasicExecutionRole")

    # Attach S3 Read Policy for target bucket
    s3_read_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="idp-s3-read-access",
        PolicyDocument=json.dumps(s3_read_policy),
    )
    print(f"  Configured inline policy for s3://{bucket_name}/* read access.")
    print("  Waiting 5 seconds for IAM role propagation...")
    time.sleep(5)

    return role_arn


def deploy_lambda_function(
    session: boto3.Session, function_name: str, role_arn: str, tags: dict[str, str]
) -> dict:
    print(f"\n--- [Step 5/6] Deploying AWS Lambda Function '{function_name}' ---")
    lambda_client = session.client("lambda")

    with open(ZIP_PATH, "rb") as f:
        zip_bytes = f.read()

    function_exists = False
    try:
        lambda_client.get_function(FunctionName=function_name)
        function_exists = True
        print(f"  Found existing function '{function_name}'. Updating code...")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            function_exists = False
        else:
            raise

    if function_exists:
        # Update function code directly using ZipFile bytes
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_bytes,
            Architectures=["x86_64"],
        )
        print("  Waiting for function code update to complete...")
        waiter = lambda_client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=function_name)

        # Update function configuration
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Role=role_arn,
            Handler="src.api.lambda_layer1_handler.lambda_handler",
            Runtime="python3.11",
            Timeout=60,
            MemorySize=512,
            Environment={"Variables": {"IDP_ROUTING_MODE": "capability_based"}},
        )
        print("  Waiting for function config update to complete...")
        waiter.wait(FunctionName=function_name)

        if tags:
            try:
                fn_resp = lambda_client.get_function(FunctionName=function_name)
                lambda_client.tag_resource(Resource=fn_resp["Configuration"]["FunctionArn"], Tags=tags)
                print(f"  Updated tags on Lambda function: {tags}")
            except Exception as tag_err:
                print(f"  Note on Lambda tagging: {tag_err}")
    else:
        # Create function using direct ZipFile upload WITH tags
        print(f"  Creating new Lambda function '{function_name}' with tags ({len(zip_bytes) / (1024 * 1024):.2f} MB)...")
        create_kwargs = {
            "FunctionName": function_name,
            "Runtime": "python3.11",
            "Role": role_arn,
            "Handler": "src.api.lambda_layer1_handler.lambda_handler",
            "Code": {
                "ZipFile": zip_bytes,
            },
            "Description": "IDP Layer 1 Page Inspection and Capability Router",
            "Timeout": 60,
            "MemorySize": 512,
            "Environment": {"Variables": {"IDP_ROUTING_MODE": "capability_based"}},
            "Architectures": ["x86_64"],
        }
        if tags:
            create_kwargs["Tags"] = tags
        lambda_client.create_function(**create_kwargs)
        print("  Waiting for function to become active...")
        waiter = lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=function_name)

    resp = lambda_client.get_function(FunctionName=function_name)
    conf = resp["Configuration"]
    print(f"  Function State:  {conf.get('State')}")
    print(f"  Runtime:         {conf.get('Runtime')}")
    print(f"  Memory Size:     {conf.get('MemorySize')} MB")
    print(f"  Timeout:         {conf.get('Timeout')} s")
    print(f"  Code Size:       {conf.get('CodeSize') / (1024 * 1024):.2f} MB")
    print(f"  Function ARN:    {conf.get('FunctionArn')}")

    return conf


def main():
    parser = argparse.ArgumentParser(description="Deploy IDP Layer 1 to AWS Lambda with Tagging Support")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE"), help="AWS CLI profile name")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"), help="Target AWS region")
    parser.add_argument("--bucket", default=os.getenv("IDP_S3_BUCKET"), help="Target S3 bucket for documents and packages")
    parser.add_argument("--function-name", default=os.getenv("IDP_LAMBDA_NAME", "idp-layer1-router"), help="Lambda function name")
    parser.add_argument("--role-name", default="idp-layer1-lambda-role", help="IAM role name")
    parser.add_argument(
        "--tags",
        default=os.getenv("IDP_AWS_TAGS"),
        help="Mandatory AWS resource tags as Key=Value pairs comma-separated, e.g.: 'Project=IDP,Environment=dev,Owner=Vinay'",
    )
    args = parser.parse_args()

    tags = parse_tags(args.tags)
    session = get_session(args.profile, args.region)

    print("==============================================================")
    print(f"  IDP Layer 1 Lambda Deployment: {args.function_name} ({args.region})")
    if args.profile:
        print(f"  AWS Profile:   {args.profile}")
    if tags:
        print(f"  Resource Tags: {tags}")
    else:
        print("  Resource Tags: (none specified)")
    print("==============================================================")

    account_id, bucket_name = validate_aws_environment(session, args.bucket, args.region, tags)
    zip_size, uncomp_size = build_package()
    upload_to_s3(session, bucket_name)
    ensure_cloudwatch_log_group(session, args.function_name, tags)
    role_arn = ensure_iam_role(session, args.role_name, bucket_name, tags)
    conf = deploy_lambda_function(session, args.function_name, role_arn, tags)

    print("\n--- [Step 6/6] Deployment Finished Successfully! ---")
    print(f"Function Name: {conf['FunctionName']}")
    print(f"Function ARN:  {conf['FunctionArn']}")
    print(f"S3 Bucket:     s3://{bucket_name}")
    if tags:
        print(f"Applied Tags:  {tags}")
    print(f"\nNext step: Run test script:")
    profile_flag = f"--profile {args.profile} " if args.profile else ""
    print(f"  python scripts/test_layer1_lambda.py {profile_flag}--bucket {bucket_name} --region {args.region}")


if __name__ == "__main__":
    main()
