import time
from src.ai.layer3_extraction.page_loader import load_pages_from_fixture
from src.ai.layer3_extraction.router import route_and_extract
from src.ai.layer3_extraction.schema_validation import extract_with_retry
from src.adapters.llm.extraction_factory import get_extraction_client
from src.ai.schemas.extraction_schema import DocumentExtraction
from src.ai.layer3_extraction.storage import init_db, save_processing_result
from src.config.settings import settings

init_db()

pages = load_pages_from_fixture("sdg_goals_output")
print(f"Loaded {len(pages)} pages")

llm = get_extraction_client()

start = time.time()
result = extract_with_retry(
    lambda: route_and_extract(pages, DocumentExtraction, llm),
    DocumentExtraction,
)
elapsed = round(time.time() - start, 2)

# Placeholder values below (None) until real Layer 1/2 PageOutput data is wired in —
# these columns exist and are ready, just unpopulated when testing from fixtures.
result_id = save_processing_result(
    doc_id="sdg_goals_output",
    page_count=len(pages),
    routes_used=None,
    engines_used=None,
    primary_scripts=None,
    avg_confidence=None,
    min_confidence=None,
    low_confidence_page_count=None,
    escalated_page_count=None,
    total_escalation_attempts=None,
    max_complexity_score=None,
    has_images=None,
    schema_name="DocumentExtraction",
    result_json=result.model_dump(),
    llm_provider=settings.extraction_backend,
    model_name=settings.extraction_model_name if settings.extraction_backend == "ollama" else settings.sarvam_model_name,
    strategy_used="B_pageindex" if len(pages) >= settings.short_doc_page_limit else "A_short",
    processing_time_seconds=elapsed,
)
print(f"Saved to database, row id={result_id}, took {elapsed}s")
print(result.model_dump_json(indent=2))