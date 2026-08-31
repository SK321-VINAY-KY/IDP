"""
CLI test client - talks to the running FastAPI server over HTTP.

Usage:
    python -m cli.client
    python -m cli.client --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description="Schema Discovery Chatbot CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--show-schema", action="store_true", help="Print the live schema JSON each turn")
    parser.add_argument(
        "--from-documents",
        nargs="+",
        metavar="PDF",
        help="Start from 2-5 sample PDFs instead of the text interview, e.g. --from-documents a.pdf b.pdf c.pdf",
    )
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=120)

    if args.from_documents:
        files = [("files", (path, open(path, "rb"), "application/pdf")) for path in args.from_documents]
        try:
            resp = client.post("/schema/infer", files=files)
        finally:
            for _, (_, fh, _) in files:
                fh.close()
        if resp.status_code != 200:
            print(f"[error {resp.status_code}] {resp.text}", file=sys.stderr)
            sys.exit(1)
    else:
        resp = client.post("/chat", json={})

    data = resp.json()
    session_id = data["session_id"]
    print(f"[session {session_id}]")
    print(f"Bot: {data['message']}")

    if args.show_schema and data.get("schema"):
        print("--- schema so far ---")
        print(json.dumps(data["schema"], indent=2))
        print("---------------------")

    if data.get("completed"):
        print(f"\n✅ Done. schema_id = {data['schema_id']}")
        return

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_message.lower() in ("quit", "exit"):
            break

        resp = client.post("/chat", json={"session_id": session_id, "message": user_message})
        if resp.status_code != 200:
            print(f"[error {resp.status_code}] {resp.text}", file=sys.stderr)
            continue
        data = resp.json()
        print(f"Bot: {data['message']}")

        if args.show_schema and data.get("schema"):
            print("--- schema so far ---")
            print(json.dumps(data["schema"], indent=2))
            print("---------------------")

        if data.get("completed"):
            print(f"\n✅ Done. schema_id = {data['schema_id']}")
            break


if __name__ == "__main__":
    main()
