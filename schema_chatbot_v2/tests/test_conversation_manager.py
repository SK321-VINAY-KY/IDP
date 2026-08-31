import pytest

from app.core.conversation_manager import ConversationManager
from app.llm.mock_adapter import MockLLMAdapter
from app.storage.session_store import InMemorySessionStore


@pytest.fixture
def manager():
    return ConversationManager(llm=MockLLMAdapter(), store=InMemorySessionStore())


def _fill_all_gaps(manager, sid, r):
    """Drives the REVIEW loop to confirmation by answering whatever gap is
    open next, same helper role the old per-gap tests used - but now
    everything (including document type) lives in the one REVIEW state."""
    for _ in range(20):
        schema_state = manager.store.get(sid).schema_state
        if schema_state.document_type and not schema_state.all_gaps():
            return r
        gap = schema_state.next_gap()
        if gap is None:
            r = manager.handle_message(sid, "invoices from suppliers")
            continue
        if gap.attribute == "type":
            r = manager.handle_message(sid, "text")
        elif gap.attribute == "required":
            r = manager.handle_message(sid, "always")
        elif gap.attribute == "item_type":
            r = manager.handle_message(sid, "text")
    return r


def test_full_happy_path_invoice(manager):
    r = manager.start_session()
    assert r.state == "REVIEW"
    sid = r.session_id

    r = manager.handle_message(sid, "invoices from suppliers")
    assert manager.store.get(sid).schema_state.document_type == "invoices_from_suppliers"

    r = manager.handle_message(sid, "invoice number, vendor name and total amount")
    assert len(manager.store.get(sid).schema_state.fields) == 3

    r = _fill_all_gaps(manager, sid, r)
    assert manager.store.get(sid).schema_state.all_gaps() == []
    assert r.schema is not None
    assert len(r.schema["fields"]) == 3

    r = manager.handle_message(sid, "yes")
    assert r.completed is True
    assert r.schema_id is not None
    assert r.schema_id.startswith("schema_")


def test_one_message_sets_document_type_and_fields_together(manager):
    # The whole point of the collapsed REVIEW state: a single message that
    # both names the document type and lists fields should be credited for
    # both, not just whichever one the old rigid state happened to expect.
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount and date")
    session = manager.store.get(sid)
    assert session.schema_state.document_type == "receipts"
    assert set(session.schema_state.fields) == {"amount", "date"}


def test_one_message_answers_two_gaps_at_once(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    manager.handle_message(sid, "amount")
    # "amount" opens two gaps: type, then required. One reply answers both.
    r = manager.handle_message(sid, "text, always")
    field = manager.store.get(sid).schema_state.fields["amount"]
    assert field.type == "string"
    assert field.required is True


def test_correction_removes_field_mid_interview(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "insurance claims")
    manager.handle_message(sid, "customer name and gst")

    assert "gst" in manager.store.get(sid).schema_state.fields

    r = manager.handle_message(sid, "remove gst")
    assert "gst" not in manager.store.get(sid).schema_state.fields


def test_retype_correction_applies_immediately(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    manager.handle_message(sid, "total")
    manager.handle_message(sid, "text, always")
    assert manager.store.get(sid).schema_state.fields["total"].type == "string"

    manager.handle_message(sid, "make total a number")
    assert manager.store.get(sid).schema_state.fields["total"].type == "number"


def test_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        manager.handle_message("does-not-exist", "hello")


# ---- document intake (upload N samples instead of, or alongside, the
# text interview) ----
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
    assert r.state == "REVIEW"
    assert r.schema["document_type"] == "invoice"
    assert {f["name"] for f in r.schema["fields"]} == {"invoice_number", "vendor_name", "total_amount"}
    assert manager.store.get(r.session_id).schema_state.all_gaps() == []

    r = manager.handle_message(r.session_id, "yes")
    assert r.completed is True
    assert r.schema_id is not None


def test_start_from_documents_with_ambiguous_field_leaves_a_gap(manager):
    r = manager.start_from_documents([_SAMPLE_A, _SAMPLE_C_WITH_GAP])
    # "notes" only shows up in one sample and has no type/required info,
    # so it lands as an ordinary open gap in the same REVIEW loop.
    gap = manager.store.get(r.session_id).schema_state.next_gap()
    assert gap.field_name == "notes"


def test_start_from_documents_wrong_sample_count_raises(manager):
    with pytest.raises(ValueError):
        manager.start_from_documents([_SAMPLE_A])  # only 1, need 2-5


def test_start_from_documents_falls_back_to_text_interview_on_failure(manager):
    r = manager.start_from_documents([b"\xff\xfe", _SAMPLE_A])
    assert r.state == "REVIEW"
    assert r.schema["document_type"] is None


def test_start_from_documents_can_resume_an_existing_session(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "we process invoices")

    r = manager.start_from_documents([_SAMPLE_A, _SAMPLE_B], session_id=sid)
    assert r.session_id == sid
    # The document type from chat is kept; samples add the fields on top of
    # the same session instead of starting a new one.
    assert manager.store.get(sid).schema_state.document_type == "we_process_invoices"
    assert {"invoice_number", "vendor_name", "total_amount"} <= set(manager.store.get(sid).schema_state.fields)


def test_start_from_documents_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        manager.start_from_documents([_SAMPLE_A, _SAMPLE_B], session_id="does-not-exist")


def test_rejecting_confirmation_stays_in_review_for_more_edits(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount")
    r = _fill_all_gaps(manager, sid, r)

    r = manager.handle_message(sid, "no")
    assert r.state == "REVIEW"
    assert r.completed is False

    # Still fully conversational from here - can keep adding fields.
    r = manager.handle_message(sid, "also add a due date")
    assert "due_date" in manager.store.get(sid).schema_state.fields


def test_confirmation_with_open_gap_does_not_complete(manager):
    # A stray "yes" should never complete the schema while something is
    # still unresolved - this is the "stuck loop" risk turned into a guard
    # instead of a dead end.
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount")  # opens gaps: type, required
    r = manager.handle_message(sid, "yes")
    assert r.completed is False
    assert r.state == "REVIEW"


def test_completed_session_rejects_further_messages(manager):
    r = manager.start_session()
    sid = r.session_id
    manager.handle_message(sid, "receipts")
    r = manager.handle_message(sid, "amount")
    r = _fill_all_gaps(manager, sid, r)
    r = manager.handle_message(sid, "yes")
    assert r.completed is True

    r = manager.handle_message(sid, "add another field")
    assert "already completed" in r.message
