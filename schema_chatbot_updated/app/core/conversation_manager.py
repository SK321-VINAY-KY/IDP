"""
The orchestrator. This is the only place that:
  - decides what to ask next
  - decides when the state machine advances
  - applies LLM-proposed changes to SchemaState

The LLM is consulted for two narrow things (interpreting a message,
phrasing a question) and never for "what do we do next".
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from app.core.schema_state import SchemaState
from app.core.state_machine import ConversationState, InvalidTransition, transition
from app.core.validator import validate_schema
from app.llm.base import ExtractionResult, LLMAdapter, SchemaProposal
from app.llm.prompts import (
    confirmation_prompt,
    document_intake_failed_message,
    document_intake_intro,
    document_type_question,
    fields_question,
)
from app.storage.session_store import Session, SessionStore

MIN_DOCUMENT_SAMPLES = 2
MAX_DOCUMENT_SAMPLES = 5


@dataclass
class TurnResult:
    session_id: str
    message: str
    state: str
    schema: Optional[dict]
    completed: bool
    schema_id: Optional[str] = None
    errors: Optional[list] = None


class ConversationManager:
    def __init__(self, llm: LLMAdapter, store: SessionStore):
        self.llm = llm
        self.store = store

    # ---- public API ----

    def start_session(self) -> TurnResult:
        session = self.store.create()
        session.state = transition(session.state, "start")
        self.store.save(session)
        return TurnResult(
            session_id=session.session_id,
            message=document_type_question(),
            state=session.state.value,
            schema=None,
            completed=False,
        )

    def start_from_documents(self, samples: List[bytes]) -> TurnResult:
        """
        Alternate entry point: instead of the plain-text interview, the user
        uploaded 2-5 sample PDFs. One inference pass proposes document_type
        + fields; any field left ambiguous by the samples (see
        FieldObservation in app.llm.base) falls through to the exact same
        gap-driven interview handle_message() already runs.
        """
        if not (MIN_DOCUMENT_SAMPLES <= len(samples) <= MAX_DOCUMENT_SAMPLES):
            raise ValueError(f"expected {MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample documents, got {len(samples)}")

        session = self.store.create()
        session.state = transition(session.state, "start_from_documents")

        proposal = self.llm.infer_schema_from_pdfs(samples)
        return self._apply_schema_proposal(session, proposal)

    def _apply_schema_proposal(self, session: Session, proposal: SchemaProposal) -> TurnResult:
        if proposal.extraction_failed or not proposal.document_type:
            session.state = transition(session.state, "inference_failed")
            self.store.save(session)
            return self._as_result(session, document_intake_failed_message(proposal.failure_reason))

        session.schema_state.set_document_type(proposal.document_type)
        for obs in proposal.fields:
            session.schema_state.add_field(
                obs.name,
                type=obs.type,
                required=obs.required,
                currency=obs.currency,
                item_type=obs.item_type,
                pattern=obs.pattern,
            )

        gap = session.schema_state.next_gap()
        if gap is None:
            session.state = transition(session.state, "documents_captured_no_gaps")
            self.store.save(session)
            return self._as_result(session, confirmation_prompt(session.schema_state.human_summary()))

        session.state = transition(session.state, "documents_captured_with_gaps")
        self.store.save(session)
        question = self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
        return self._as_result(session, document_intake_intro() + question)

    def handle_message(self, session_id: str, user_message: str) -> TurnResult:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session.completed:
            return self._as_result(session, "This schema is already completed. Start a new session to build another.")

        session.turn_count += 1

        # 1. Ask the LLM to interpret this message given current state.
        #    Pass the current gap (if any) so the adapter knows which
        #    field/attribute a bare answer like "always" or "text" refers to.
        extraction = self.llm.extract(
            state=session.state.value,
            user_message=user_message,
            context=self._build_context(session, session.schema_state.next_gap()),
        )

        # 2. Corrections can arrive at any state - apply them first,
        #    deterministically, regardless of what else is going on.
        removed_any = self._apply_removals(session, extraction)

        # 3. Route by current state.
        if session.state == ConversationState.ASK_DOCUMENT_TYPE:
            return self._handle_ask_document_type(session, extraction)

        if session.state == ConversationState.ASK_FIELDS:
            return self._handle_ask_fields(session, extraction)

        if session.state == ConversationState.FIELD_DETAILS:
            return self._handle_field_details(session, extraction, removed_any)

        if session.state == ConversationState.CONFIRMATION:
            return self._handle_confirmation(session, extraction, removed_any)

        # Shouldn't get here.
        return self._as_result(session, "I'm not sure how to proceed - could you rephrase that?")

    # ---- per-state handlers ----

    def _handle_ask_document_type(self, session: Session, extraction: ExtractionResult) -> TurnResult:
        if extraction.extraction_failed or not extraction.document_type:
            self.store.save(session)
            return self._as_result(session, "Sorry, what type of documents are these? (e.g. invoices, receipts)")

        session.schema_state.set_document_type(extraction.document_type)
        session.state = transition(session.state, "document_type_captured")
        self.store.save(session)
        return self._as_result(session, fields_question())

    def _handle_ask_fields(self, session: Session, extraction: ExtractionResult) -> TurnResult:
        if extraction.extraction_failed or (not extraction.new_fields and not session.schema_state.has_fields()):
            self.store.save(session)
            return self._as_result(
                session,
                "I didn't catch any fields there. What information would you like to extract? "
                "For example: customer name, date, total amount.",
            )

        for proposal in extraction.new_fields:
            session.schema_state.add_field(
                proposal.name,
                type=proposal.type,
                required=proposal.required,
                currency=proposal.currency,
                item_type=proposal.item_type,
            )

        gap = session.schema_state.next_gap()
        if gap is None:
            session.state = transition(session.state, "fields_captured_no_gaps")
            self.store.save(session)
            return self._as_result(session, confirmation_prompt(session.schema_state.human_summary()))

        session.state = transition(session.state, "fields_captured_with_gaps")
        self.store.save(session)
        question = self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
        return self._as_result(session, question)

    def _handle_field_details(self, session: Session, extraction: ExtractionResult, removed_any: bool) -> TurnResult:
        applied = False
        for answer in extraction.field_answers:
            if session.schema_state.update_field_attribute(answer.field_name, answer.attribute, answer.value):
                applied = True

        if not applied and not removed_any:
            gap = session.schema_state.next_gap()
            self.store.save(session)
            if gap is None:
                # A correction resolved the only remaining gap.
                session.state = transition(session.state, "no_gaps_left")
                self.store.save(session)
                return self._as_result(session, confirmation_prompt(session.schema_state.human_summary()))
            question = self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
            return self._as_result(session, "Sorry, I didn't quite get that. " + question)

        gap = session.schema_state.next_gap()
        if gap is None:
            session.state = transition(session.state, "no_gaps_left")
            self.store.save(session)
            return self._as_result(session, confirmation_prompt(session.schema_state.human_summary()))

        session.state = transition(session.state, "gap_remaining")
        self.store.save(session)
        question = self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
        return self._as_result(session, question)

    def _handle_confirmation(self, session: Session, extraction: ExtractionResult, removed_any: bool) -> TurnResult:
        if removed_any:
            gap = session.schema_state.next_gap()
            if gap is not None:
                session.state = transition(session.state, "correction_made")
                self.store.save(session)
                question = self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
                return self._as_result(session, f"Got it, removed that. {question}")
            self.store.save(session)
            return self._as_result(session, confirmation_prompt(session.schema_state.human_summary()))

        if extraction.confirmation is True:
            errors = validate_schema(session.schema_state)
            if errors:
                session.state = transition(session.state, "confirmed_invalid")
                self.store.save(session)
                gap = session.schema_state.next_gap()
                msg = "I found some issues before we can finish:\n" + "\n".join(f"- {e}" for e in errors)
                if gap:
                    msg += "\n\n" + self.llm.phrase_question(gap.field_name, gap.attribute, self._build_context(session, gap))
                return self._as_result(session, msg, errors=errors)

            session.state = transition(session.state, "confirmed_valid")
            session.completed = True
            session.schema_id = f"schema_{uuid.uuid4().hex[:12]}"
            self.store.save(session)
            return self._as_result(session, "Schema created successfully.")

        if extraction.confirmation is False:
            session.state = transition(session.state, "rejected")
            self.store.save(session)
            return self._as_result(session, "No problem - what would you like to add, remove, or change?")

        self.store.save(session)
        return self._as_result(session, "Just to confirm - is the schema above correct? (yes/no)")

    # ---- helpers ----

    def _apply_removals(self, session: Session, extraction: ExtractionResult) -> bool:
        removed_any = False
        for name in extraction.removals:
            if session.schema_state.remove_field(name):
                removed_any = True
        return removed_any

    def _build_context(self, session: Session, gap=None) -> dict:
        ctx = {"schema": session.schema_state.to_json_schema()}
        if gap is not None:
            ctx["current_gap"] = {"field_name": gap.field_name, "attribute": gap.attribute}
        return ctx

    def _as_result(self, session: Session, message: str, errors: Optional[list] = None) -> TurnResult:
        return TurnResult(
            session_id=session.session_id,
            message=message,
            state=session.state.value,
            schema=session.schema_state.to_json_schema() if session.state != ConversationState.ASK_DOCUMENT_TYPE else None,
            completed=session.completed,
            schema_id=session.schema_id,
            errors=errors,
        )
