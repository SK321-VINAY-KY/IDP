"""
Stage 3: create the IAM execution role + the 4 Lambda functions in account
530084039784 (ap-south-1), all tagged.

  idp-layer1-router            (ZIP, python3.11, PyMuPDF)   <- built here
  idp-engine-docling           (container image from ECR)
  idp-engine-paddle-printed    (container image from ECR)
  idp-engine-paddle-handwritten(container image from ECR)

Container images must already be pushed (Stage 2). The L1 ZIP is built here
using manylinux wheels. Reads AWS creds from env (.env2).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch_lambda_build"
ZIP_PATH = ROOT / "scratch_layer1_lambda.zip"
REGION = "ap-south-1"
ACCOUNT = "530084039784"
BUCKET = "idp-internal-35385"

TAGS = {
    "createdby": "vinay.k@shellkode.com",
    "customer": "idp-internal-35385",
    "purpose": "internal",
}
ECR = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"

CONTAINER_LAMBDAS = {
    "idp-engine-docling": f"{ECR}/idp-layer2-docling:latest",
    "idp-engine-paddle-printed": f"{ECR}/idp-layer2-paddle-printed:latest",
    "idp-engine-paddle-handwritten": f"{ECR}/idp-layer2-paddle-handwritten:latest",
}

EXEC_ROLE = "idp-lambda-exec-role"


def run(cmd):
    print(f"[EXEC] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {r.stderr}")
    return r.stdout.strip()


def ensure_exec_role(iam):
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    try:
        arn = iam.get_role(RoleName=EXEC_ROLE)["Role"]["Arn"]
        print(f"[iam] role exists: {arn}")
        iam.tag_role(RoleName=EXEC_ROLE, Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        arn = iam.create_role(
            RoleName=EXEC_ROLE,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="IDP Lambda execution role (L1 router + L2 engines)",
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )["Role"]["Arn"]
        print(f"[iam] created role: {arn}")
        time.sleep(8)
    iam.attach_role_policy(RoleName=EXEC_ROLE, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    s3pol = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                       "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]}],
    }
    iam.put_role_policy(RoleName=EXEC_ROLE, PolicyName="idp-s3-access", PolicyDocument=json.dumps(s3pol))
    print("[iam] attached basic-exec + s3 access")
    time.sleep(5)
    return arn


def build_layer1_zip():
    print("[l1] building ZIP package (manylinux wheels)...")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not (SCRATCH / "pymupdf").exists() and not list(SCRATCH.glob("fitz*")):
        run(f'"{sys.executable}" -m pip install --platform manylinux2014_x86_64 '
            f'--target "{SCRATCH}" --implementation cp --python-version 3.11 '
            f'--only-binary=:all: pymupdf pydantic pydantic-settings python-dotenv')
    mods = [
        "src/api/__init__.py", "src/api/lambda_layer1_handler.py", "src/__init__.py",
        "src/ai/__init__.py", "src/ai/layer1_routing/__init__.py",
        "src/ai/layer1_routing/inspect.py", "src/ai/layer1_routing/router.py",
        "src/ai/layer1_routing/pipeline.py", "src/ai/layer1_routing/capability_router.py",
        "src/ai/layer1_routing/capability_types.py", "src/ai/schemas/__init__.py",
        "src/ai/schemas/page.py", "src/ai/schemas/document_ir.py", "src/ai/schemas/page_metadata.py",
        "src/config/__init__.py", "src/config/settings.py",
        "src/utils/__init__.py", "src/utils/logger.py",
    ]
    for rel in mods:
        src = ROOT / rel
        dst = SCRATCH / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.touch()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SCRATCH.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(SCRATCH))
    print(f"[l1] zip size {ZIP_PATH.stat().st_size/1e6:.1f} MB")


def deploy_zip_lambda(lam, role_arn):
    fn = "idp-layer1-router"
    zip_bytes = ZIP_PATH.read_bytes()
    try:
        lam.get_function(FunctionName=fn)
        print(f"[l1] updating {fn} code...")
        lam.update_function_code(FunctionName=fn, ZipFile=zip_bytes, Architectures=["x86_64"])
        lam.get_waiter("function_updated_v2").wait(FunctionName=fn)
        lam.update_function_configuration(
            FunctionName=fn, Role=role_arn,
            Handler="src.api.lambda_layer1_handler.lambda_handler",
            Runtime="python3.11", Timeout=120, MemorySize=1024,
            Environment={"Variables": {"IDP_ROUTING_MODE": "capability_based"}},
        )
        lam.get_waiter("function_updated_v2").wait(FunctionName=fn)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        print(f"[l1] creating {fn}...")
        lam.create_function(
            FunctionName=fn, Runtime="python3.11", Role=role_arn,
            Handler="src.api.lambda_layer1_handler.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="IDP Layer 1 Page Inspection & Capability Router",
            Timeout=120, MemorySize=1024, Architectures=["x86_64"],
            Environment={"Variables": {"IDP_ROUTING_MODE": "capability_based"}},
            Tags=TAGS,
        )
        lam.get_waiter("function_active_v2").wait(FunctionName=fn)
    arn = lam.get_function(FunctionName=fn)["Configuration"]["FunctionArn"]
    print(f"[l1] {fn} -> {arn}")
    return arn


def deploy_container_lambda(lam, role_arn, fn, image_uri):
    try:
        lam.get_function(FunctionName=fn)
        print(f"[l2] updating {fn} image...")
        lam.update_function_code(FunctionName=fn, ImageUri=image_uri)
        lam.get_waiter("function_updated_v2").wait(FunctionName=fn)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        print(f"[l2] creating {fn} from {image_uri}...")
        lam.create_function(
            FunctionName=fn, Role=role_arn,
            PackageType="Image", Code={"ImageUri": image_uri},
            Description=f"IDP Layer 2 engine: {fn}",
            Timeout=300, MemorySize=3008, Architectures=["x86_64"],
            Tags=TAGS,
        )
        lam.get_waiter("function_active_v2").wait(FunctionName=fn)
    arn = lam.get_function(FunctionName=fn)["Configuration"]["FunctionArn"]
    print(f"[l2] {fn} -> {arn}")
    return arn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-l1", action="store_true")
    ap.add_argument("--only", default="", help="subset of container lambda names")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    acct = session.client("sts").get_caller_identity()["Account"]
    assert acct == ACCOUNT, f"WRONG ACCOUNT {acct}"
    iam = session.client("iam")
    lam = session.client("lambda")

    role_arn = ensure_exec_role(iam)
    arns = {}
    if not args.skip_l1:
        build_layer1_zip()
        arns["idp-layer1-router"] = deploy_zip_lambda(lam, role_arn)

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    for fn, img in CONTAINER_LAMBDAS.items():
        if only and fn not in only:
            continue
        arns[fn] = deploy_container_lambda(lam, role_arn, fn, img)

    print("\n=== LAMBDAS ===")
    for k, v in arns.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
