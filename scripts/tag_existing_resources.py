"""
File: tag_existing_resources.py
Purpose: Apply standard tags across all IDP AWS resources in account 106611079163.
Tags:
  createdby=vinay.k@shellkode.com
  customer=internal
  purpose=internal
"""
import argparse
import os
import boto3
from botocore.exceptions import ClientError

DEFAULT_PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
DEFAULT_REGION = "ap-south-1"
DEFAULT_TAGS = {
    "createdby": "vinay.k@shellkode.com",
    "customer": "internal",
    "purpose": "internal",
}


def tag_s3_bucket(s3_client, bucket_name: str, tags: dict[str, str]):
    print(f"Tagging S3 bucket: {bucket_name}...")
    try:
        tag_set = [{"Key": k, "Value": v} for k, v in tags.items()]
        s3_client.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={"TagSet": tag_set},
        )
        print(f"  [SUCCESS] S3 bucket {bucket_name} tagged.")
    except ClientError as e:
        print(f"  [WARNING] Could not tag S3 bucket {bucket_name}: {e}")


def tag_lambda_function(lambda_client, function_name: str, tags: dict[str, str]):
    print(f"Tagging Lambda function: {function_name}...")
    try:
        resp = lambda_client.get_function(FunctionName=function_name)
        arn = resp["Configuration"]["FunctionArn"]
        lambda_client.tag_resource(Resource=arn, Tags=tags)
        print(f"  [SUCCESS] Lambda {function_name} tagged.")
    except ClientError as e:
        print(f"  [WARNING] Could not tag Lambda {function_name}: {e}")


def tag_step_function(sfn_client, state_machine_arn: str, tags: dict[str, str]):
    print(f"Tagging Step Function: {state_machine_arn}...")
    try:
        tag_list = [{"key": k, "value": v} for k, v in tags.items()]
        sfn_client.tag_resource(resourceArn=state_machine_arn, tags=tag_list)
        print(f"  [SUCCESS] Step Function tagged.")
    except ClientError as e:
        print(f"  [WARNING] Could not tag Step Function: {e}")


def tag_ecr_repository(ecr_client, repo_name: str, tags: dict[str, str]):
    print(f"Tagging ECR repository: {repo_name}...")
    try:
        resp = ecr_client.describe_repositories(repositoryNames=[repo_name])
        arn = resp["repositories"][0]["repositoryArn"]
        tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
        ecr_client.tag_resource(resourceArn=arn, tags=tag_list)
        print(f"  [SUCCESS] ECR repository {repo_name} tagged.")
    except ClientError as e:
        print(f"  [WARNING] Could not tag ECR {repo_name}: {e}")


def tag_iam_role(iam_client, role_name: str, tags: dict[str, str]):
    print(f"Tagging IAM role: {role_name}...")
    try:
        tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
        iam_client.tag_role(RoleName=role_name, Tags=tag_list)
        print(f"  [SUCCESS] IAM role {role_name} tagged.")
    except ClientError as e:
        print(f"  [WARNING] Could not tag IAM role {role_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tag all IDP AWS resources")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    print("==============================================================")
    print(f"Applying tags to IDP resources in Account {account_id} ({args.region})")
    print(f"Tags: {DEFAULT_TAGS}")
    print("==============================================================")

    s3 = session.client("s3")
    tag_s3_bucket(s3, f"idp-documents-{account_id}", DEFAULT_TAGS)

    lam = session.client("lambda")
    for fn in ["idp-layer1-router", "idp-engine-docling", "idp-engine-paddle-printed", "idp-engine-paddle-handwritten"]:
        tag_lambda_function(lam, fn, DEFAULT_TAGS)

    sfn = session.client("stepfunctions")
    sfn_arn = f"arn:aws:states:{args.region}:{account_id}:stateMachine:idp-document-processing"
    tag_step_function(sfn, sfn_arn, DEFAULT_TAGS)

    ecr = session.client("ecr")
    for r in ["idp-layer2-docling", "idp-layer2-paddle-printed", "idp-layer2-paddle-handwritten", "idp-layer3-app"]:
        tag_ecr_repository(ecr, r, DEFAULT_TAGS)

    iam = session.client("iam")
    for role in ["idp-layer1-lambda-role", "idp-stepfunctions-role"]:
        tag_iam_role(iam, role, DEFAULT_TAGS)

    print("\n--- Tagging operation complete! ---")


if __name__ == "__main__":
    main()
