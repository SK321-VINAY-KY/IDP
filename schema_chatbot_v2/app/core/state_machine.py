"""
Deterministic state machine for the interview.

The LLM never sets `state` directly. It proposes structured updates (a
document type, field operations, a confirmation); this module is the only
thing allowed to move `state` forward. This is the "LLM understands, code
decides" boundary from the design doc.

Collapsed from the original five-state linear interview
(ASK_DOCUMENT_TYPE -> ASK_FIELDS -> FIELD_DETAILS -> CONFIRMATION, plus a
separate DOCUMENT_INTAKE entry point) into one open REVIEW state that both
entry points land in and never leave until the schema is actually
confirmed valid. The old states each only accepted one kind of answer,
which is what made the interview feel hardcoded - a message that named a
document type AND listed fields AND answered a gap could only ever be
credited for whichever one thing the active state was listening for.
REVIEW instead hands the LLM the whole schema + every open gap + any
validation errors on every turn and accepts a batch of operations back, so
"how much of what you said gets used" no longer depends on which state you
happened to be in.
"""
from __future__ import annotations

from enum import Enum


class ConversationState(str, Enum):
    START = "START"
    # The one open working-schema loop. Both the plain-text interview and
    # document upload land here; from here on out there is no meaningful
    # difference between "still filling in document_type", "still adding
    # fields", "still answering gaps", and "reviewing before confirming" -
    # they're all just "the schema isn't confirmed valid yet".
    REVIEW = "REVIEW"
    COMPLETED = "COMPLETED"


_TRANSITIONS = {
    ConversationState.START: {
        "start": ConversationState.REVIEW,
        "start_from_documents": ConversationState.REVIEW,
    },
    ConversationState.REVIEW: {
        # The only way out of REVIEW: the user confirmed AND the schema has
        # no open gaps and no validator errors at that moment (checked by
        # ConversationManager before this event is ever fired - see
        # _finish_review_turn). Everything else - adding/editing/removing
        # fields, answering gaps, a rejected confirmation, a correction
        # after confirming - is handled inside REVIEW without a transition,
        # which is what makes it a real back-and-forth instead of a forced
        # march through checkpoints.
        "confirmed_valid": ConversationState.COMPLETED,
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
