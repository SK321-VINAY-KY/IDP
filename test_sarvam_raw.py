import sys
sys.path.insert(0, 'backend')
from src.adapters.llm.sarvam_client import SarvamExtractionClient
from src.ai.layer3_extraction.prompts.loader import render_prompt, prompt_params
import json

client = SarvamExtractionClient()
schema_fields = [
    {"name": "full_name", "description": ""},
    {"name": "address", "description": ""},
    {"name": "phone_number", "description": ""},
    {"name": "contact_email", "description": ""},
    {"name": "summary", "description": ""},
    {"name": "skills", "description": ""},
    {"name": "work_history", "description": ""},
    {"name": "education", "description": ""},
]
md = open('dataset_output/165.md', encoding='utf-8').read()

prompt = render_prompt(
    "page_field_check",
    page_md=md,
    schema_fields=schema_fields,
    page_number=1,
    total_pages=1,
)
params = prompt_params("page_field_check")

print("Requesting with max_tokens:", params["max_tokens"])
resp = client.raw_client.chat.completions.create(
    model=client.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=params["temperature"],
    max_tokens=params["max_tokens"],
    extra_body={"reasoning_effort": None},
)
choice = resp.choices[0]
print("Finish reason:", choice.finish_reason)
print("Length of content:", len(choice.message.content or ""))
print("Raw content:\n", choice.message.content)
