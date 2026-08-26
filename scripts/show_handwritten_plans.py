from src.ai.layer1_routing.inspect import inspect_page
from src.ai.layer1_routing.router import capabilities_from_profile, build_engine_plan
from src.adapters.llm.factory import get_llm_client
import pymupdf
import os

PDF = os.path.join("tests", "fixtures", "camscanner_handwritten.pdf")


llm_client = get_llm_client()
print(f"Using LLM provider: {type(llm_client).__name__}")


def main():
    doc = pymupdf.open(PDF)
    print(f"Document: {PDF} | pages={len(doc)}")
    for i, page in enumerate(doc):
        page_number = i + 1
        profile = inspect_page(page, page_number)
        caps = capabilities_from_profile(profile)
        plan = build_engine_plan(caps)
        print(f"\nPage {page_number}")
        print("  profile:", {k: v for k, v in profile.model_dump().items() if k in ('char_count','image_coverage','is_scanned','primary_script')})
        print("  capabilities:", caps.model_dump())
        print("  plan:", [t.model_dump() for t in plan])

    doc.close()


if __name__ == '__main__':
    main()
