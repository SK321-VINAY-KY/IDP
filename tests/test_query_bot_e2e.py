import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"


def test_query_bot_e2e():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Verify static assets
        r_html = client.get("/app/index.html")
        assert r_html.status_code == 200, f"HTML failed: {r_html.status_code}"
        assert "app.js?v=21" in r_html.text, "index.html does not contain app.js?v=21"

        r_js = client.get("/app/app.js?v=21")
        assert r_js.status_code == 200, f"JS failed: {r_js.status_code}"
        assert "/api/query-bot/documents" in r_js.text, "app.js does not contain /api/query-bot/documents"

        # 2. Login as admin
        login_res = client.post("/auth/login", data={"username": "admin", "password": "changeme"})
        assert login_res.status_code == 200, f"Login failed: {login_res.status_code}"
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Fetch query bot documents
        docs_res = client.get("/api/query-bot/documents", headers=headers)
        assert docs_res.status_code == 200, f"Docs failed: {docs_res.status_code}"
        data = docs_res.json()
        docs = data["documents"]
        assert len(docs) > 0, "No documents returned"

        doc_ids = [d["doc_id"] for d in docs]
        assert "Chander Kochhar 01_compressed 2.pdf" in doc_ids, "Chander Kochhar missing from documents list"

        ck_doc = next(d for d in docs if d["doc_id"] == "Chander Kochhar 01_compressed 2.pdf")
        assert ck_doc["has_graph"] is True
        assert ck_doc["strategy"] == "graph_memory"

        # 4. Ask question via Query Bot (Graph Memory)
        ask_res = client.post(
            "/api/query-bot/ask",
            json={
                "question": "What is the patient name?",
                "doc_id": "Chander Kochhar 01_compressed 2.pdf",
            },
            headers=headers,
        )
        assert ask_res.status_code == 200, f"Ask failed: {ask_res.status_code}"
        ans_data = ask_res.json()
        assert ans_data["success"] is True
        assert ans_data["mode"] == "graph"
        assert "CHANDER KOCHHAR" in ans_data["answer"].upper()
        assert len(ans_data["sources"]) > 0

        # 5. Ask question on JSON Fallback document (e.g. Vinay_Resume.pdf)
        if "Vinay_Resume.pdf" in doc_ids:
            fb_res = client.post(
                "/api/query-bot/ask",
                json={
                    "question": "What is this document?",
                    "doc_id": "Vinay_Resume.pdf",
                },
                headers=headers,
            )
            assert fb_res.status_code == 200
            fb_data = fb_res.json()
            assert fb_data["mode"] in ("graph", "json_fallback")
            assert len(fb_data.get("answer", "")) > 0
