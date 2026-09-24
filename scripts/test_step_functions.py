"""
File: test_step_functions.py
Purpose: Test AWS Step Functions orchestration for Layer 2 OCR/Conversion engines.
"""
import json
import time
import boto3

PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
REGION = "ap-south-1"
STATE_MACHINE_ARN = "arn:aws:states:ap-south-1:106611079163:stateMachine:idp-document-processing"

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
sfn_client = session.client("stepfunctions")

def run_test(name: str, payload: dict):
    print(f"\n==========================================")
    print(f"Starting Step Functions Execution: {name}")
    print(f"==========================================")
    print("Input Payload:")
    print(json.dumps(payload, indent=2))
    
    execution_name = f"test_{int(time.time())}_{name[:10]}"
    response = sfn_client.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=execution_name,
        input=json.dumps(payload),
    )
    execution_arn = response["executionArn"]
    print(f"\nExecution ARN: {execution_arn}")
    print("Polling execution status...")
    
    while True:
        desc = sfn_client.describe_execution(executionArn=execution_arn)
        status = desc["status"]
        print(f"Status: {status} ...")
        if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
            break
        time.sleep(3)
        
    print(f"\nExecution finished with status: {status}")
    if status == "SUCCEEDED":
        output = json.loads(desc["output"])
        print("\nWorkflow Final Output:")
        print(json.dumps(output, indent=2))
        return output
    else:
        print(f"Execution Error / Cause: {desc.get('cause', 'Unknown')}")
        return None

if __name__ == "__main__":
    # Test 1: Real Layer 1 routing output from test.pdf
    with open("scratch/layer1_manifest.json", "r") as f:
        l1_data = json.load(f)["body"]

    test1_payload = {
        "document_id": l1_data["document_name"],
        "s3_bucket": l1_data["s3_bucket"],
        "s3_key": l1_data["s3_key"],
        "pages": [
            {"page_number": p["page_number"], "route": p["route"]}
            for p in l1_data["pages"]
        ]
    }
    
    out1 = run_test("real_layer1_docling", test1_payload)
    
    # Test 2: Heterogeneous Multi-Engine Routing (Page 1 -> Docling, Page 2 -> PaddleOCR Printed)
    test2_payload = {
        "document_id": "test.pdf",
        "s3_bucket": "idp-documents-106611079163",
        "s3_key": "documents/test.pdf",
        "pages": [
            {"page_number": 1, "route": "docling"},
            {"page_number": 2, "route": "paddleocr_printed"}
        ]
    }
    out2 = run_test("multi_engine_parallel", test2_payload)
