"""
Script to build, push, and update PaddleOCR Lambda functions without UVDoc.
"""
import base64
import boto3
import os
import subprocess
import sys
import time

def run(cmd, cwd=None):
    print(f"[EXEC] {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")

def main():
    profile = os.getenv("AWS_PROFILE", "106611079163_SK-ML-Sandbox-team-Ps")
    region = os.getenv("AWS_REGION", "ap-south-1")
    session = boto3.Session(profile_name=profile, region_name=region)
    ecr = session.client("ecr")
    lam = session.client("lambda")

    # 1. ECR Login
    print("[1/4] Logging into ECR...")
    auth_data = ecr.get_authorization_token()["authorizationData"][0]
    token = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
    username, password = token.split(":")
    endpoint = auth_data["proxyEndpoint"]
    run(f"docker login -u {username} -p {password} {endpoint}")

    # 2. Build & Push paddle_printed
    printed_tag = "106611079163.dkr.ecr.ap-south-1.amazonaws.com/idp-layer2-paddle-printed:latest"
    print(f"\n[2/4] Building & pushing {printed_tag}...")
    run(f"docker build --provenance=false --platform linux/amd64 -t {printed_tag} docker/paddle_printed")
    run(f"docker push {printed_tag}")

    print("Updating Lambda function 'idp-engine-paddle-printed' code...")
    res = lam.update_function_code(FunctionName="idp-engine-paddle-printed", ImageUri=printed_tag)
    print("Update dispatched:", res.get("FunctionArn"))

    # 3. Build & Push paddle_handwritten
    handwritten_tag = "106611079163.dkr.ecr.ap-south-1.amazonaws.com/idp-layer2-paddle-handwritten:latest"
    print(f"\n[3/4] Building & pushing {handwritten_tag}...")
    run(f"docker build --provenance=false --platform linux/amd64 -t {handwritten_tag} docker/paddle_handwritten")
    run(f"docker push {handwritten_tag}")

    print("Updating Lambda function 'idp-engine-paddle-handwritten' code...")
    res = lam.update_function_code(FunctionName="idp-engine-paddle-handwritten", ImageUri=handwritten_tag)
    print("Update dispatched:", res.get("FunctionArn"))

    # 4. Wait for both functions to reach Active/Successful state
    print("\n[4/4] Waiting for Lambda functions to finish updating...")
    for fn in ["idp-engine-paddle-printed", "idp-engine-paddle-handwritten"]:
        for _ in range(30):
            conf = lam.get_function_configuration(FunctionName=fn)
            state = conf.get("LastUpdateStatus")
            print(f"  {fn} LastUpdateStatus: {state}")
            if state == "Successful":
                break
            time.sleep(4)

    print("\nSUCCESS: Both PaddleOCR Lambda functions updated to latest image without UVDoc!")

if __name__ == "__main__":
    main()
