import pytest

from app.core.state_machine import ConversationState, InvalidTransition, transition


def test_start_to_review_via_start():
    s = transition(ConversationState.START, "start")
    assert s == ConversationState.REVIEW


def test_start_to_review_via_start_from_documents():
    s = transition(ConversationState.START, "start_from_documents")
    assert s == ConversationState.REVIEW


def test_review_to_completed_via_confirmed_valid():
    s = transition(ConversationState.REVIEW, "confirmed_valid")
    assert s == ConversationState.COMPLETED


def test_review_has_no_other_transitions():
    # Everything else (adding/editing/removing fields, answering gaps, a
    # rejected confirmation) happens *inside* REVIEW without a state
    # transition - only a fully-valid confirmation moves the state.
    with pytest.raises(InvalidTransition):
        transition(ConversationState.REVIEW, "rejected")


def test_invalid_event_raises():
    with pytest.raises(InvalidTransition):
        transition(ConversationState.START, "confirmed_valid")


def test_completed_is_terminal():
    with pytest.raises(InvalidTransition):
        transition(ConversationState.COMPLETED, "start")
