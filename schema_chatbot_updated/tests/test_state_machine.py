import pytest

from app.core.state_machine import ConversationState, InvalidTransition, transition


def test_happy_path_transitions():
    s = ConversationState.START
    s = transition(s, "start")
    assert s == ConversationState.ASK_DOCUMENT_TYPE
    s = transition(s, "document_type_captured")
    assert s == ConversationState.ASK_FIELDS
    s = transition(s, "fields_captured_with_gaps")
    assert s == ConversationState.FIELD_DETAILS
    s = transition(s, "no_gaps_left")
    assert s == ConversationState.CONFIRMATION
    s = transition(s, "confirmed_valid")
    assert s == ConversationState.COMPLETED


def test_confirmation_rejected_goes_back_to_fields():
    s = ConversationState.CONFIRMATION
    s = transition(s, "rejected")
    assert s == ConversationState.ASK_FIELDS


def test_confirmation_invalid_goes_back_to_field_details():
    s = ConversationState.CONFIRMATION
    s = transition(s, "confirmed_invalid")
    assert s == ConversationState.FIELD_DETAILS


def test_invalid_event_raises():
    with pytest.raises(InvalidTransition):
        transition(ConversationState.START, "confirmed_valid")


def test_completed_is_terminal():
    with pytest.raises(InvalidTransition):
        transition(ConversationState.COMPLETED, "start")


def test_document_intake_with_gaps_goes_to_field_details():
    s = ConversationState.START
    s = transition(s, "start_from_documents")
    assert s == ConversationState.DOCUMENT_INTAKE
    s = transition(s, "documents_captured_with_gaps")
    assert s == ConversationState.FIELD_DETAILS


def test_document_intake_no_gaps_goes_straight_to_confirmation():
    s = transition(ConversationState.DOCUMENT_INTAKE, "documents_captured_no_gaps")
    assert s == ConversationState.CONFIRMATION


def test_document_intake_failure_falls_back_to_text_interview():
    s = transition(ConversationState.DOCUMENT_INTAKE, "inference_failed")
    assert s == ConversationState.ASK_DOCUMENT_TYPE
