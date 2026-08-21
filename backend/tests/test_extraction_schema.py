from src.ai.schemas.extraction_schema import DocumentExtraction


def test_extraction_defaults_are_non_null_strings():
    payload = DocumentExtraction()

    assert payload.document_title == ""
    assert payload.total_goals == ""
    assert payload.total_targets == ""
    assert payload.first_goal_title == ""
    assert payload.last_goal_title == ""


def test_extraction_values_are_completed_from_source_text():
    result = DocumentExtraction(last_goal_title="Goal 14")
    content = """[Page 1]\n5th SDG Youth Summer Camp - SDG Resource Document\nThe 17 Sustainable Development Goals (SDGs) and 169 Targets\nGoal 1. End poverty in all its forms everywhere\nGoal 17. Strengthen the means of implementation and revitalize the Global Partnership\n"""

    completed = DocumentExtraction.complete_from_text(result, content)

    assert completed.document_title == "5th SDG Youth Summer Camp - SDG Resource Document"
    assert completed.total_goals == "17"
    assert completed.total_targets == "169"
    assert completed.first_goal_title == "End poverty in all its forms everywhere"
    assert completed.last_goal_title == "Strengthen the means of implementation and revitalize the Global Partnership"
