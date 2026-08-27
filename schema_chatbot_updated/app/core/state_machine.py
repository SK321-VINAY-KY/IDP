"""
Deterministic state machine for the interview.

The LLM never sets `state` directly. It proposes structured updates
(new fields, attribute answers, corrections); this module is the only
thing allowed to move `state` forward. This is the "LLM understands,
code decides" boundary from the design doc.
"""
from __future__ import annotations

from enum import Enum


class ConversationState(str, Enum):
    START = "START"
    ASK_DOCUMENT_TYPE = "ASK_DOCUMENT_TYPE"
    ASK_FIELDS = "ASK_FIELDS"
    FIELD_DETAILS = "FIELD_DETAILS"
    CONFIRMATION = "CONFIRMATION"
    COMPLETED = "COMPLETED"
    # Alternate entry point: user uploaded 2-5 sample documents instead of
    # starting the plain-text interview. One inference pass populates
    # SchemaState directly, then control hands off into the *same*
    # FIELD_DETAILS/CONFIRMATION states the text interview uses - so any
    # remaining gaps or user edits are handled by existing machinery, not
    # new code.
    DOCUMENT_INTAKE = "DOCUMENT_INTAKE"


# Explicit transition table instead of if/elif chains, so it doesn't
# calcify into spaghetti once corrections/interrupts are added (see the
# scaling concern raised before building this).
_TRANSITIONS = {
    ConversationState.START: {
        "start": ConversationState.ASK_DOCUMENT_TYPE,
        "start_from_documents": ConversationState.DOCUMENT_INTAKE,
    },
    ConversationState.DOCUMENT_INTAKE: {
        # Same landing states the text interview reaches after ASK_FIELDS -
        # gap-filling and confirmation are shared, unchanged code from here.
        "documents_captured_with_gaps": ConversationState.FIELD_DETAILS,
        "documents_captured_no_gaps": ConversationState.CONFIRMATION,
        # OCR/inference failed or didn't produce a usable document_type -
        # degrade to the normal chat interview rather than dead-ending.
        "inference_failed": ConversationState.ASK_DOCUMENT_TYPE,
    },
    ConversationState.ASK_DOCUMENT_TYPE: {
        "document_type_captured": ConversationState.ASK_FIELDS,
    },
    ConversationState.ASK_FIELDS: {
        "fields_captured_with_gaps": ConversationState.FIELD_DETAILS,
        "fields_captured_no_gaps": ConversationState.CONFIRMATION,
        "no_fields_yet": ConversationState.ASK_FIELDS,  # stay
    },
    ConversationState.FIELD_DETAILS: {
        "gap_remaining": ConversationState.FIELD_DETAILS,  # stay, ask next gap
        "no_gaps_left": ConversationState.CONFIRMATION,
    },
    ConversationState.CONFIRMATION: {
        "confirmed_valid": ConversationState.COMPLETED,
        "confirmed_invalid": ConversationState.FIELD_DETAILS,
        "rejected": ConversationState.ASK_FIELDS,
        "correction_made": ConversationState.FIELD_DETAILS,
    },
    ConversationState.COMPLETED: {},
}


class InvalidTransition(Exception):
    pass


def transition(current: ConversationState, event: str) -> ConversationState:
    allowed = _TRANSITIONS.get(current, {})
    if event not in allowed:
        raise InvalidTransition(f"No transition for event {event!r} from state {current!r}")
    return allowed[event]
