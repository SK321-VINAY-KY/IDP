from src.ai.layer3_extraction.page_loader import load_pages
from src.ai.layer3_extraction.router import route_and_extract
from src.ai.layer3_extraction.schema_validation import extract_with_retry
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.schemas.extraction_schema import DocumentExtraction

pages = load_pages("sdg_goals_output")
print(f"Loaded {len(pages)} pages")

llm = get_extraction_client()
result = extract_with_retry(
    lambda: route_and_extract(pages, DocumentExtraction, llm),
    DocumentExtraction,
)
print(result.model_dump_json(indent=2))