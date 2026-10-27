"""
Stage 1 of the fresh IDP deploy into account 530084039784 (ap-south-1):
  - S3 document bucket (idp-documents-<account>)
  - RDS PostgreSQL (idp-postgres-db) with a DB subnet group in the default VPC
All resources tagged. Reads AWS creds from the environment (set from .env2).
Idempotent: re-running reuses existing resources.
"""
import argparse
import os
import time

import boto3
from botocore.exceptions import ClientError

REGION = "ap-south-1"
VPC_ID = "vpc-0571f21ae29ec7a1d"
SUBNET_IDS = [
    "subnet-05a694d5be10434f3",
    "subnet-00af2fd62f81c622e",
    "subnet-054bd923089c27d5a",
]
DB_IDENTIFIER = "idp-postgres-db"
DB_NAME = "idp"
DB_USER = "idp_admin"
DB_PASSWORD = os.environ["IDP_DB_PASSWORD"]  # required; never hardcode (file is committed to git)
DB_INSTANCE_CLASS = "db.t4g.micro"
DB_SUBNET_GROUP = "idp-db-subnet-group"

TAGS = {
    "createdby": "vinay.k@shellkode.com",
    "customer": "idp-internal-35385",
    "purpose": "internal",
}
TAGLIST = [{"Key": k, "Value": v} for k, v in TAGS.items()]


def ensure_s3(session, account_id):
    # The SCP blocks s3:CreateBucket via CLI (no tag-on-create param), so we
    # REUSE the pre-created, tagged bucket 'idp-internal-35385'. Just verify access.
    s3 = session.client("s3")
    bucket = os.getenv("IDP_S3_BUCKET", "idp-internal-35385")
    s3.head_bucket(Bucket=bucket)
    print(f"[s3] using existing bucket: {bucket}")
    return bucket


def ensure_rds_sg(session):
    ec2 = session.client("ec2")
    name = "idp-postgres-sg"
    res = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [name]},
        {"Name": "vpc-id", "Values": [VPC_ID]},
    ])
    if res.get("SecurityGroups"):
        sg_id = res["SecurityGroups"][0]["GroupId"]
        print(f"[rds-sg] exists: {sg_id}")
        ec2.create_tags(Resources=[sg_id], Tags=TAGLIST)
        return sg_id
    res = ec2.create_security_group(
        GroupName=name, Description="IDP RDS PostgreSQL SG", VpcId=VPC_ID,
        TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGLIST}],
    )
    sg_id = res["GroupId"]
    print(f"[rds-sg] created: {sg_id}")
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "PostgreSQL"}],
            }],
        )
        print("[rds-sg] ingress 5432 authorized")
    except ClientError as e:
        if "InvalidPermission.Duplicate" in str(e):
            print("[rds-sg] ingress already present")
        else:
            raise
    return sg_id


def ensure_db_subnet_group(session):
    rds = session.client("rds")
    try:
        rds.describe_db_subnet_groups(DBSubnetGroupName=DB_SUBNET_GROUP)
        print(f"[rds-subnet] exists: {DB_SUBNET_GROUP}")
        return DB_SUBNET_GROUP
    except ClientError as e:
        if "DBSubnetGroupNotFoundFault" not in str(e):
            raise
    rds.create_db_subnet_group(
        DBSubnetGroupName=DB_SUBNET_GROUP,
        DBSubnetGroupDescription="IDP RDS subnet group (default VPC)",
        SubnetIds=SUBNET_IDS,
        Tags=TAGLIST,
    )
    print(f"[rds-subnet] created: {DB_SUBNET_GROUP}")
    return DB_SUBNET_GROUP


def ensure_rds(session, sg_id, subnet_group):
    rds = session.client("rds")
    try:
        desc = rds.describe_db_instances(DBInstanceIdentifier=DB_IDENTIFIER)
        db = desc["DBInstances"][0]
        print(f"[rds] exists: {DB_IDENTIFIER} status={db['DBInstanceStatus']}")
        rds.add_tags_to_resource(ResourceName=db["DBInstanceArn"], Tags=TAGLIST)
        return db
    except ClientError as e:
        if "DBInstanceNotFound" not in str(e):
            raise
    print(f"[rds] creating {DB_IDENTIFIER} (postgres, {DB_INSTANCE_CLASS}, 20GB gp3)...")
    rds.create_db_instance(
        DBInstanceIdentifier=DB_IDENTIFIER, DBName=DB_NAME, Engine="postgres",
        DBInstanceClass=DB_INSTANCE_CLASS, AllocatedStorage=20, StorageType="gp3",
        MasterUsername=DB_USER, MasterUserPassword=DB_PASSWORD,
        DBSubnetGroupName=subnet_group, VpcSecurityGroupIds=[sg_id],
        PubliclyAccessible=True, BackupRetentionPeriod=1, DeletionProtection=False,
        Tags=TAGLIST,
    )
    print(f"[rds] creation initiated: {DB_IDENTIFIER}")
    return rds.describe_db_instances(DBInstanceIdentifier=DB_IDENTIFIER)["DBInstances"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    print(f"=== Stage 1 S3+RDS :: account {account_id} region {REGION} ===")
    assert account_id == "530084039784", f"WRONG ACCOUNT {account_id}"

    bucket = ensure_s3(session, account_id)
    sg_id = ensure_rds_sg(session)
    subnet_group = ensure_db_subnet_group(session)
    db = ensure_rds(session, sg_id, subnet_group)

    if args.wait:
        rds = session.client("rds")
        start = time.time()
        while time.time() - start < 1200:
            db = rds.describe_db_instances(DBInstanceIdentifier=DB_IDENTIFIER)["DBInstances"][0]
            ep = db.get("Endpoint", {})
            print(f"  [{int(time.time()-start)}s] status={db['DBInstanceStatus']} addr={ep.get('Address')}")
            if db["DBInstanceStatus"] == "available" and ep.get("Address"):
                break
            time.sleep(20)

    ep = db.get("Endpoint", {})
    addr = ep.get("Address", "pending")
    print(f"S3_BUCKET={bucket}")
    print(f"DB_ENDPOINT={addr}:{ep.get('Port', 5432)}")
    print(f"DATABASE_URL=postgresql://{DB_USER}:{DB_PASSWORD}@{addr}:{ep.get('Port',5432)}/{DB_NAME}")


if __name__ == "__main__":
    main()
