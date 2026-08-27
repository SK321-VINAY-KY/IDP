import pytest

from app.core.conversation_manager import ConversationManager
from app.llm.mock_adapter import MockLLMAdapter
from app.storage.session_store import InMemorySessionStore


@pytest.fixture
def manager():
    return ConversationManager(llm=MockLLMAdapter(), store=InMemorySessionStore())


def test_full_happy_path_invoice(manager):
    r = manager.start_session()
    assert r.state == "ASK_DOCUMENT_TYPE"
    sid = r.session_id

    r = manager.handle_message(sid, "invoices from suppliers")
    assert r.state == "ASK_FIELDS"

    r = manager.handle_message(sid, "invoice number, vendor name and total amount")
    assert r.state == "FIELD_DETAILS"

    # answer gaps one by one until confirmation
    for _ in range(20):
        if r.state == "CONFIRMATION":
            break
        gap_field = manager.store.get(sid).schema_state.next_gap()
        if gap_field.attribute == "type":
            r = manager.handle_message(sid, "text")
        elif gap_field.attribute == "required":
            r = manager.handle_message(sid, "always")
        elif gap_field.attribute == "item_type":
            r = manager.handle_message(sid, "text")

    assert r.state == "CONFIRMATION"
    assert r.schema is not None
    assert len(r.schema["fields"]) == 3

    r = manager.handle_message(sid, "yes")
    assert r.completed is True
    assert r.schema_id is not None
    assert r.schema_id.startswith("schema_")


def test_correction_removes_field_mid_interview(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "insurance claims")
    manager.handle_message(sid, "customer name and gst")

    assert "gst" in manager.store.get(sid).schema_state.fields

    r = manager.handle_message(sid, "remove gst")
    assert "gst" not in manager.store.get(sid).schema_state.fields


def test_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        manager.handle_message("does-not-exist", "hello")


# ---- document intake (upload N samples instead of the text interview) ----
# Samples are plain-text-as-bytes here since MockLLMAdapter has no real
# OCR/PDF support by design - see MockLLMAdapter.infer_schema_from_pdfs.

_SAMPLE_A = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
total_amount: number: yes
"""

_SAMPLE_B = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
total_amount: number: yes
"""

_SAMPLE_C_WITH_GAP = b"""
document_type: invoice
invoice_number: string: yes
vendor_name: string: yes
notes:: 
"""


def test_start_from_documents_with_no_gaps_goes_to_confirmation(manager):
    r = manager.start_from_documents([_SAMPLE_A, _SAMPLE_B])
    assert r.state == "CONFIRMATION"
    assert r.schema["document_type"] == "invoice"
    assert {f["name"] for f in r.schema["fields"]} == {"invoice_number", "vendor_name", "total_amount"}

    r = manager.handle_message(r.session_id, "yes")
    assert r.completed is True
    assert r.schema_id is not None


def test_start_from_documents_with_ambiguous_field_asks_about_it(manager):
    r = manager.start_from_documents([_SAMPLE_A, _SAMPLE_C_WITH_GAP])
    # "notes" only shows up in one sample and has no type/required info,
    # so it should be routed into the normal gap-driven interview.
    assert r.state == "FIELD_DETAILS"
    gap = manager.store.get(r.session_id).schema_state.next_gap()
    assert gap.field_name == "notes"


def test_start_from_documents_wrong_sample_count_raises(manager):
    with pytest.raises(ValueError):
        manager.start_from_documents([_SAMPLE_A])  # only 1, need 2-5


def test_start_from_documents_falls_back_to_text_interview_on_failure(manager):
    r = manager.start_from_documents([b"\xff\xfe", _SAMPLE_A])
    assert r.state == "ASK_DOCUMENT_TYPE"
    assert r.schema is None


def test_rejecting_confirmation_returns_to_ask_fields(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount")
    # answer the one gap
    while r.state != "CONFIRMATION":
        gap = manager.store.get(sid).schema_state.next_gap()
        answer = "number" if gap.attribute == "type" else "always"
        r = manager.handle_message(sid, answer)

    r = manager.handle_message(sid, "no")
    assert r.state == "ASK_FIELDS"
    assert r.completed is False


def test_completed_session_rejects_further_messages(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount")
    while r.state != "CONFIRMATION":
        gap = manager.store.get(sid).schema_state.next_gap()
        answer = "number" if gap.attribute == "type" else "always"
        r = manager.handle_message(sid, answer)
    r = manager.handle_message(sid, "yes")
    assert r.completed is True

    r = manager.handle_message(sid, "add another field")
    assert "already completed" in r.message
