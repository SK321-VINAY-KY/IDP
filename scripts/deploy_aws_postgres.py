"""
File: deploy_aws_postgres.py
Purpose: Provision and configure an AWS RDS PostgreSQL database for the IDP system.
Tags applied to all created resources:
  createdby=vinay.k@shellkode.com
  customer=internal
  purpose=internal
"""
import argparse
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_REGION = "ap-south-1"
DEFAULT_PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
DEFAULT_VPC_ID = "vpc-0ff3b1e43c502098f"
DEFAULT_SUBNET_GROUP = "default-vpc-0ff3b1e43c502098f"
DEFAULT_DB_IDENTIFIER = "idp-postgres-db"
DEFAULT_DB_NAME = "idp"
DEFAULT_MASTER_USER = "idp_admin"
DEFAULT_MASTER_PASSWORD = os.getenv("IDP_DB_PASSWORD", "IdpAdmin2026_SecureDb!")
DEFAULT_INSTANCE_CLASS = "db.t4g.micro"

MANDATORY_TAGS = [
    {"Key": "createdby", "Value": "vinay.k@shellkode.com"},
    {"Key": "customer", "Value": "internal"},
    {"Key": "purpose", "Value": "internal"},
]


def get_session(profile: str | None, region: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def ensure_security_group(ec2_client, vpc_id: str) -> str:
    print("\n--- [Step 1/3] Ensuring RDS Security Group ---")
    sg_name = "idp-postgres-sg"
    try:
        res = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [sg_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )
        groups = res.get("SecurityGroups", [])
        if groups:
            sg_id = groups[0]["GroupId"]
            print(f"  Security group '{sg_name}' already exists: {sg_id}")
            # Ensure mandatory tags
            ec2_client.create_tags(Resources=[sg_id], Tags=MANDATORY_TAGS)
            return sg_id
    except ClientError as e:
        print(f"  Error checking security group: {e}")

    print(f"  Creating Security Group '{sg_name}' in VPC {vpc_id}...")
    res = ec2_client.create_security_group(
        GroupName=sg_name,
        Description="Security Group for IDP RDS PostgreSQL Database",
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": MANDATORY_TAGS,
            }
        ],
    )
    sg_id = res["GroupId"]
    print(f"  Created Security Group: {sg_id}")

    # Allow inbound PostgreSQL (5432) from VPC CIDR and ECS Web SG
    print("  Authorizing ingress rules on port 5432...")
    try:
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 5432,
                    "ToPort": 5432,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "PostgreSQL access"}],
                }
            ],
        )
        print("  Inbound rule for port 5432 authorized.")
    except ClientError as e:
        if "InvalidPermission.Duplicate" in str(e):
            print("  Ingress rule already exists.")
        else:
            print(f"  Notice setting ingress rule: {e}")

    return sg_id


def create_or_get_rds_instance(
    rds_client,
    db_identifier: str,
    db_name: str,
    master_user: str,
    master_password: str,
    instance_class: str,
    subnet_group: str,
    sg_id: str,
) -> dict:
    print(f"\n--- [Step 2/3] Checking RDS PostgreSQL DB Instance '{db_identifier}' ---")
    try:
        desc = rds_client.describe_db_instances(DBInstanceIdentifier=db_identifier)
        db = desc["DBInstances"][0]
        status = db["DBInstanceStatus"]
        print(f"  DB Instance '{db_identifier}' already exists (Status: {status}).")
        # Ensure tags
        arn = db["DBInstanceArn"]
        rds_client.add_tags_to_resource(ResourceName=arn, Tags=MANDATORY_TAGS)
        return db
    except ClientError as e:
        if "DBInstanceNotFound" not in str(e):
            raise

    print(f"  Creating new RDS PostgreSQL instance '{db_identifier}'...")
    print(f"  Engine: PostgreSQL 17.9 | Class: {instance_class} | Storage: 20GB gp3")
    print(f"  Database Name: {db_name} | Master User: {master_user}")
    print(f"  Subnet Group: {subnet_group} | Security Group: {sg_id}")
    print(f"  Tags: {MANDATORY_TAGS}")

    rds_client.create_db_instance(
        DBInstanceIdentifier=db_identifier,
        DBName=db_name,
        Engine="postgres",
        EngineVersion="17.9",
        DBInstanceClass=instance_class,
        AllocatedStorage=20,
        StorageType="gp3",
        MasterUsername=master_user,
        MasterUserPassword=master_password,
        DBSubnetGroupName=subnet_group,
        VpcSecurityGroupIds=[sg_id],
        PubliclyAccessible=True,
        BackupRetentionPeriod=1,
        DeletionProtection=False,
        Tags=MANDATORY_TAGS,
    )
    print(f"  RDS DB instance '{db_identifier}' creation initiated.")

    desc = rds_client.describe_db_instances(DBInstanceIdentifier=db_identifier)
    return desc["DBInstances"][0]


def wait_for_available(rds_client, db_identifier: str, timeout_s: int = 900) -> dict:
    print(f"\n--- [Step 3/3] Waiting for DB instance '{db_identifier}' to become available ---")
    start = time.time()
    while time.time() - start < timeout_s:
        desc = rds_client.describe_db_instances(DBInstanceIdentifier=db_identifier)
        db = desc["DBInstances"][0]
        status = db["DBInstanceStatus"]
        elapsed = int(time.time() - start)
        endpoint = db.get("Endpoint", {})
        address = endpoint.get("Address")
        print(f"  [{elapsed}s] Status: {status} | Address: {address or 'Pending...'}")
        if status == "available" and address:
            print(f"\n=======================================================")
            print(f"SUCCESS: RDS PostgreSQL DB '{db_identifier}' is AVAILABLE!")
            print(f"Endpoint: {address}:{endpoint.get('Port', 5432)}")
            print(f"=======================================================")
            return db
        time.sleep(20)

    raise TimeoutError(f"DB Instance '{db_identifier}' did not become available in {timeout_s}s.")


def main():
    parser = argparse.ArgumentParser(description="Deploy AWS RDS PostgreSQL for IDP")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--vpc-id", default=DEFAULT_VPC_ID)
    parser.add_argument("--subnet-group", default=DEFAULT_SUBNET_GROUP)
    parser.add_argument("--db-identifier", default=DEFAULT_DB_IDENTIFIER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--master-user", default=DEFAULT_MASTER_USER)
    parser.add_argument("--master-password", default=DEFAULT_MASTER_PASSWORD)
    parser.add_argument("--instance-class", default=DEFAULT_INSTANCE_CLASS)
    parser.add_argument("--wait", action="store_true", help="Wait for instance to become available")
    args = parser.parse_args()

    session = get_session(args.profile, args.region)
    ec2 = session.client("ec2")
    rds = session.client("rds")

    sg_id = ensure_security_group(ec2, args.vpc_id)
    db = create_or_get_rds_instance(
        rds,
        args.db_identifier,
        args.db_name,
        args.master_user,
        args.master_password,
        args.instance_class,
        args.subnet_group,
        sg_id,
    )

    if args.wait or db.get("DBInstanceStatus") != "available":
        db = wait_for_available(rds, args.db_identifier)

    endpoint = db.get("Endpoint", {})
    address = endpoint.get("Address", "pending...")
    port = endpoint.get("Port", 5432)

    database_url = f"postgresql://{args.master_user}:{args.master_password}@{address}:{port}/{args.db_name}"
    print(f"\nTarget DATABASE_URL:")
    print(f"  {database_url}")
    print("\nNote: Data has NOT been migrated as requested. The database service is created and ready.")


if __name__ == "__main__":
    main()
