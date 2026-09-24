"""
Integration test for Layer 1 AWS Lambda function ('idp-layer1-router').
Uploads sample PDF to S3, invokes Lambda, captures performance metrics, and verifies routing output.
Supports custom AWS profiles, buckets, regions, and functions.
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import boto3

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT_DIR / "dataset" / "test.pdf"


def run_cloud_test(profile: str | None, region: str, bucket_name: str, function_name: str, pdf_path: Path):
    print(f"=== Layer 1 AWS Lambda Live Integration Test ===")
    print(f"Target Function: {function_name}")
    print(f"Target Region:   {region}")
    print(f"Target Bucket:   {bucket_name}")
    print(f"Test PDF:        {pdf_path}")
    if profile:
        print(f"AWS Profile:     {profile}")

    if not pdf_path.exists():
        print(f"ERROR: Local test PDF not found at {pdf_path}")
        sys.exit(1)

    session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(region_name=region)
    s3_client = session.client("s3")
    lambda_client = session.client("lambda")

    test_s3_key = f"documents/{pdf_path.name}"

    # 1. Upload sample PDF to S3
    print(f"\n[1/3] Uploading {pdf_path.name} ({pdf_path.stat().st_size} bytes) to s3://{bucket_name}/{test_s3_key}...")
    with open(pdf_path, "rb") as f:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_s3_key,
            Body=f.read(),
            ContentType="application/pdf",
        )
    print(f"      Upload to {bucket_name} completed successfully.")

    # 2. Invoke Lambda Function (Direct Invocation)
    payload = {
        "s3_bucket": bucket_name,
        "s3_key": test_s3_key,
    }
    print(f"\n[2/3] Invoking AWS Lambda '{function_name}'...")
    t0 = time.monotonic()
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        LogType="Tail",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    wall_duration_ms = (time.monotonic() - t0) * 1000

    # 3. Parse execution logs and metrics from CloudWatch LogResult header
    log_result_b64 = response.get("LogResult", "")
    log_text = base64.b64decode(log_result_b64).decode("utf-8", errors="replace") if log_result_b64 else ""

    print(f"\n=== AWS CloudWatch Execution Metrics ===")
    report_lines = [line for line in log_text.splitlines() if line.startswith("REPORT")]
    if report_lines:
        print(f"  {report_lines[-1]}")
    print(f"  Client Wall Clock Latency: {wall_duration_ms:.1f} ms")

    # 4. Parse Lambda Response Payload
    raw_body = response["Payload"].read().decode("utf-8")
    result = json.loads(raw_body)

    status_code = result.get("statusCode")
    print(f"\n[3/3] Validating Response (HTTP Status: {status_code})...")

    if status_code != 200:
        print(f"ERROR: Lambda invocation returned status {status_code}")
        print(f"Body: {result.get('body')}")
        print("\nFull CloudWatch Tail Logs:")
        print(log_text)
        sys.exit(1)

    body = result.get("body", {})
    total_pages = body.get("total_pages", 0)
    pages = body.get("pages", [])

    print(f"\n--- Document Routing Results ---")
    print(f"Document Name: {body.get('document_name')}")
    print(f"Total Pages:   {total_pages}")
    print(f"Routing Mode:  {body.get('routing_mode')}")

    assert total_pages > 0, "Expected at least 1 page in document."
    assert len(pages) == total_pages, f"Page array length ({len(pages)}) does not match total_pages ({total_pages})"

    for page in pages:
        p_num = page.get("page_number")
        route = page.get("route")
        caps = page.get("capabilities", [])
        tasks = page.get("tasks", [])
        prof = page.get("profile", {})
        needs_vlm = page.get("needs_vlm")

        task_str = " -> ".join(f"{t['engine']}(pri={t['priority']})" for t in tasks)
        print(f"\n  [Page {p_num}]")
        print(f"    - Route:        {route}")
        print(f"    - Tasks:        {task_str}")
        print(f"    - Capabilities: {caps}")
        print(f"    - Needs VLM:    {needs_vlm}")
        print(f"    - Profile:      chars={prof.get('char_count')}, img_cov={prof.get('image_coverage')}, complexity={prof.get('complexity_score')}, script={prof.get('primary_script')}")

    print(f"\n=======================================================")
    print(f"  ALL CHECKS PASSED: Layer 1 Lambda is fully operational!")
    print(f"=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Test Layer 1 AWS Lambda function")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE"), help="AWS CLI profile name")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"), help="Target AWS region")
    parser.add_argument("--bucket", default=os.getenv("IDP_S3_BUCKET", "s3-bucket-demo-rag"), help="Target S3 bucket")
    parser.add_argument("--function-name", default=os.getenv("IDP_LAMBDA_NAME", "idp-layer1-router"), help="Lambda function name")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to local test PDF")
    args = parser.parse_args()

    run_cloud_test(
        profile=args.profile,
        region=args.region,
        bucket_name=args.bucket,
        function_name=args.function_name,
        pdf_path=Path(args.pdf),
    )


if __name__ == "__main__":
    main()
