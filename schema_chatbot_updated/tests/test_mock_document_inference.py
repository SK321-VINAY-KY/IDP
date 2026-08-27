"""
Tests for MockLLMAdapter.infer_schema_from_pdfs - the offline, zero-dependency
stand-in for real OCR+inference (see the docstring on that method for the
tiny plain-text format it expects instead of real PDF bytes).
"""
from app.llm.mock_adapter import MockLLMAdapter

SAMPLE_1 = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
total_amount: number: yes
notes: string: no
"""

SAMPLE_2 = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
total_amount: number: yes
"""

# Same fields, but "total_amount" disagrees on type across samples.
SAMPLE_3_INCONSISTENT = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
total_amount: string: yes
"""


def test_infers_document_type_and_fields_across_samples():
    adapter = MockLLMAdapter()
    proposal = adapter.infer_schema_from_pdfs([SAMPLE_1, SAMPLE_2])

    assert proposal.extraction_failed is False
    assert proposal.document_type == "invoice"

    by_name = {f.name: f for f in proposal.fields}
    assert by_name["invoice_number"].seen_in_samples == 2
    assert by_name["invoice_number"].total_samples == 2
    assert by_name["invoice_number"].required is True
    assert by_name["invoice_number"].type == "string"

    # "notes" only appeared in one of the two samples.
    assert by_name["notes"].seen_in_samples == 1


def test_inconsistent_type_across_samples_is_left_null():
    adapter = MockLLMAdapter()
    proposal = adapter.infer_schema_from_pdfs([SAMPLE_1, SAMPLE_3_INCONSISTENT])

    by_name = {f.name: f for f in proposal.fields}
    assert by_name["total_amount"].type is None
    assert by_name["total_amount"].notes == "inconsistent across samples"


def test_non_utf8_bytes_fail_gracefully():
    adapter = MockLLMAdapter()
    proposal = adapter.infer_schema_from_pdfs([b"\xff\xfe\x00\x01", SAMPLE_1])
    assert proposal.extraction_failed is True
    assert proposal.failure_reason is not None
