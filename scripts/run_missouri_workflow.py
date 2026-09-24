"""
File: run_missouri_workflow.py
Purpose: Run Layer 2 Step Functions orchestration on the 8-page Missouri Public Water Systems document.
"""
import json
import time
import boto3

PROFILE = "106611079163_SK-ML-Sandbox-team-Ps"
REGION = "ap-south-1"
STATE_MACHINE_ARN = "arn:aws:states:ap-south-1:106611079163:stateMachine:idp-document-processing"

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
sfn_client = session.client("stepfunctions")

def main():
    with open("scratch/layer1_missouri_result.json", "r") as f:
        l1_body = json.load(f)["body"]

    # Build Step Functions input payload
    # Pages 1, 2, 4, 5, 6, 7, 8 -> paddleocr_printed
    # Page 3 (signature page) -> paddleocr_handwritten
    pages = []
    for p in l1_body["pages"]:
        num = p["page_number"]
        route = "paddleocr_handwritten" if num == 3 else "paddleocr_printed"
        pages.append({"page_number": num, "route": route})

    sfn_payload = {
        "document_id": l1_body["document_name"],
        "s3_bucket": l1_body["s3_bucket"],
        "s3_key": l1_body["s3_key"],
        "pages": pages,
    }

    print("=================================================================")
    print("Starting Layer 2 Step Functions Execution on 8-Page Missouri Document")
    print("=================================================================")
    print(f"Document: {sfn_payload['document_id']} ({len(pages)} pages)")
    print(f"Routing Plan:")
    for p in pages:
        print(f"  Page {p['page_number']}: {p['route']}")

    t0_start = time.time()
    exec_name = f"missouri_8p_{int(t0_start)}"
    resp = sfn_client.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=exec_name,
        input=json.dumps(sfn_payload),
    )
    exec_arn = resp["executionArn"]
    print(f"\nExecution ARN: {exec_arn}")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0_start))}")
    print("\nOrchestrating Map State parallel executions (Concurrency = 8)...")

    prev_status = None
    while True:
        desc = sfn_client.describe_execution(executionArn=exec_arn)
        status = desc["status"]
        elapsed = round(time.time() - t0_start, 1)
        if status != prev_status:
            print(f"[{elapsed}s] Status changed to: {status}")
            prev_status = status
        else:
            print(f"[{elapsed}s] Still running...")

        if status in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"):
            break
        time.sleep(5)

    total_duration = round(time.time() - t0_start, 2)
    print(f"\nExecution finished in {total_duration}s with status: {status}")

    if status == "SUCCEEDED":
        out = json.loads(desc["output"])
        with open("scratch/missouri_step_functions_output.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print("\n=================================================================")
        print("LAYER 2 PARALLEL EXTRACTION REPORT")
        print("=================================================================")
        print(f"Document ID: {out['document_id']}")
        print(f"S3 Location: s3://{out['s3_bucket']}/{out['s3_key']}")
        print(f"Total Pages Processed: {out['total_pages']}")
        print(f"Total Wall-Clock Latency: {total_duration}s")
        print(f"Concurrency Achieved: {len(out['results'])} concurrent executions")
        print("-----------------------------------------------------------------")
        print(f"{'Page':<6}{'Engine':<24}{'Status':<10}{'Confidence':<14}{'Lines':<8}{'Latency (ms)'}")
        print("-----------------------------------------------------------------")
        for r in out.get("results", []):
            print(f"{r.get('page_number'):<6}{r.get('engine'):<24}{r.get('status'):<10}{r.get('confidence', 0.0):<14.4f}{r.get('lines_extracted', 0):<8}{r.get('latency_ms', 0):.2f}")
        print("-----------------------------------------------------------------")
    else:
        print(f"Error: {desc.get('cause')}")

if __name__ == "__main__":
    main()
