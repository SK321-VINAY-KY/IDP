from src.ai.schemas.extraction_schema import DocumentExtraction


def test_extraction_defaults_are_non_null_strings():
    payload = DocumentExtraction()

    assert payload.document_title == ""
    assert payload.total_goals == ""
    assert payload.total_targets == ""
    assert payload.first_goal_title == ""
    assert payload.last_goal_title == ""
