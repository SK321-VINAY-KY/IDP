"""
The orchestrator. This is the only place that:
  - decides when the state machine advances
  - applies LLM-proposed changes to SchemaState

The LLM is consulted for two things (interpreting a message and, now,
composing the reply text) and never for "what do we do next" - that's
still fully deterministic here.

This is the collapsed REVIEW-loop design: there is one open state
(app.core.state_machine.ConversationState.REVIEW) instead of a chain of
single-purpose states. Every turn - regardless of whether the user is
naming a document type, listing fields, answering a gap, correcting
something, or confirming - goes through the same handle_message() path,
which shows the LLM the whole schema + every open gap + any validation
errors and applies whatever batch of operations comes back. See
app.core.state_machine for why this replaces the old five-state chain.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.schema_state import SchemaState, normalize_type
from app.core.state_machine import ConversationState, transition
from app.core.validator import validate_schema
from app.llm.base import ExtractionResult, LLMAdapter, SchemaProposal
from app.llm.prompts import (
    confirmation_prompt,
    document_intake_failed_message,
    document_intake_intro,
    document_type_question,
    fallback_question,
    greeting_message,
)
from app.storage.session_store import Session, SessionStore

SCHEMA_REGISTRY_DIR = Path(__file__).resolve().parents[3] / "schema_registry"

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

    # ---- schema registry persistence ----

    def _persist_confirmed_schema(self, session: Session) -> Path:
        """
        Writes the fully-confirmed schema to schema_registry/<schema_id>.json
        on disk. Called exactly once, immediately after a successful
        confirmation. The persisted record contains the schema, an id, a
        timestamp, and the document_type so downstream pipeline code can
        locate and reuse it by id without re-reading the session.
        """
        SCHEMA_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_id": session.schema_id,
            "document_type": session.schema_state.document_type,
            "confirmed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": session.session_id,
            "turn_count_at_confirm": session.turn_count,
            "schema": session.schema_state.to_json_schema(),
            "sample_documents": getattr(session, "sample_documents", None) or [],
            "owner": session.owner,
        }
        path = SCHEMA_REGISTRY_DIR / f"{session.schema_id}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return path

    # ---- public API ----

    def start_session(self) -> TurnResult:
        session = self.store.create()
        session.state = transition(session.state, "start")
        self.store.save(session)
        return TurnResult(
            session_id=session.session_id,
            message=greeting_message(),
            state=session.state.value,
            schema=session.schema_state.to_json_schema(),
            completed=False,
        )

    def start_from_documents(self, samples: List[bytes], session_id: Optional[str] = None) -> TurnResult:
        """
        Alternate entry point: instead of (or in addition to) the plain-text
        interview, the user uploads 2-5 sample PDFs. One inference pass
        proposes document_type + fields; anything left ambiguous by the
        samples (see FieldObservation in app.llm.base) becomes an ordinary
        open gap in the same REVIEW loop handle_message() runs - there's no
        separate gap-filling code path for the document-upload entry point.

        Pass session_id to feed samples into a session that's already
        mid-conversation (e.g. the user started by chatting, then decided to
        also upload samples) rather than always starting a fresh session.
        """
        if not (MIN_DOCUMENT_SAMPLES <= len(samples) <= MAX_DOCUMENT_SAMPLES):
            raise ValueError(f"expected {MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample documents, got {len(samples)}")

        if session_id is not None:
            session = self.store.get(session_id)
            if session is None:
                raise KeyError(f"unknown session_id {session_id!r}")
            if session.completed:
                raise ValueError("session is already completed")
        else:
            session = self.store.create()

        if session.state == ConversationState.START:
            session.state = transition(session.state, "start_from_documents")

        proposal = self.llm.infer_schema_from_pdfs(samples)
        return self._apply_schema_proposal(session, proposal)

    def _apply_schema_proposal(self, session: Session, proposal: SchemaProposal) -> TurnResult:
        if proposal.extraction_failed or not proposal.document_type:
            self.store.save(session)
            return self._as_result(session, document_intake_failed_message(proposal.failure_reason))

        if not session.schema_state.document_type:
            session.schema_state.set_document_type(proposal.document_type)
        notes: List[str] = []
        for obs in proposal.fields:
            normalized_type, note = normalize_type(obs.type)
            if note:
                notes.append(note)
            session.schema_state.add_field(
                obs.name,
                type=normalized_type,
                required=obs.required,
                currency=obs.currency,
                item_type=obs.item_type,
                pattern=obs.pattern,
            )

        self.store.save(session)
        gaps = session.schema_state.all_gaps()
        errors = validate_schema(session.schema_state)
        body = self._review_fallback_reply(session, gaps, False, errors, notes)
        return self._as_result(session, document_intake_intro() + body)

    def handle_message(self, session_id: str, user_message: str) -> TurnResult:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session.completed:
            return self._as_result(session, "This schema is already completed. Start a new session to build another.")

        session.turn_count += 1

        # Every turn sees the WHOLE picture - full schema, every open gap,
        # every validation error - not just whatever single thing the old
        # per-state handlers were listening for. That's what lets one
        # message do several things at once.
        extraction = self.llm.extract(
            state=session.state.value,
            user_message=user_message,
            context=self._build_context(session),
        )

        if extraction.extraction_failed:
            self.store.save(session)
            return self._as_result(session, "Sorry, I had trouble understanding that - could you rephrase?")

        if extraction.document_type:
            session.schema_state.set_document_type(extraction.document_type)

        notes = session.schema_state.apply_operations(extraction.operations) if extraction.operations else []

        return self._finish_review_turn(session, extraction, notes)

    def update_schema_manually(self, session_id: str, document_type: Optional[str], fields: List[Dict[str, Any]]) -> TurnResult:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session.completed:
            raise ValueError("session is already completed")

        session.turn_count += 1
        new_state = SchemaState()
        if document_type:
            new_state.set_document_type(document_type)

        for f in fields:
            name = f.get("name")
            if not name or not str(name).strip():
                continue
            raw_type = str(f.get("type", "string")).strip()
            item_type = f.get("item_type")
            if "[" in raw_type and "]" in raw_type:
                item_type = raw_type[raw_type.find("[") + 1 : raw_type.find("]")].strip()
                raw_type = "array"

            norm_type, _ = normalize_type(raw_type)
            new_state.add_field(
                name,
                type=norm_type or "string",
                required=bool(f.get("required", False)),
                description=f.get("description"),
                item_type=item_type,
                pattern=f.get("pattern"),
                currency=f.get("currency"),
            )

        session.schema_state = new_state
        if session.state == ConversationState.START:
            session.state = ConversationState.REVIEW

        self.store.save(session)
        errors = validate_schema(session.schema_state)
        return self._as_result(session, "Schema updated directly from UI editor.", errors=errors)

    # ---- the one review turn ----

    def _finish_review_turn(self, session: Session, extraction: ExtractionResult, notes: List[str]) -> TurnResult:
        gaps = session.schema_state.all_gaps()
        doc_type_missing = session.schema_state.document_type is None
        errors = validate_schema(session.schema_state)

        still_open = bool(gaps or doc_type_missing or errors)

        if extraction.confirmation is True and not still_open:
            session.state = transition(session.state, "confirmed_valid")
            session.completed = True
            session.schema_id = f"schema_{uuid.uuid4().hex[:12]}"
            self.store.save(session)
            try:
                self._persist_confirmed_schema(session)
                persist_note = (
                    f"\n\nSaved to schema_registry/{session.schema_id}.json."
                )
            except Exception as exc:  # pragma: no cover - defensive
                persist_note = (
                    f"\n\n(Warning: could not write to schema_registry: {exc})"
                )
            try:
                from app.core.activity_log import log_activity
                log_activity(
                    session.owner or "unknown",
                    "schema_confirmed",
                    {"schema_id": session.schema_id, "document_type": session.schema_state.document_type},
                )
            except Exception:
                pass
            message = (extraction.reply or "Schema created successfully.") + persist_note
            return self._as_result(session, message)

        self.store.save(session)

        if extraction.confirmation is True and still_open:
            # The LLM (or user) tried to confirm, but something's still
            # open - never let a confirmation complete the schema out from
            # under an unresolved gap/error. Explain what's still missing
            # instead of silently ignoring the "yes".
            message = extraction.reply or (
                "Almost there, but a couple of things still need sorting out first.\n\n"
                + self._review_fallback_reply(session, gaps, doc_type_missing, errors, notes)
            )
            return self._as_result(session, message)

        if extraction.needs_clarification and not extraction.reply:
            reason = f" {extraction.clarification_reason}" if extraction.clarification_reason else ""
            return self._as_result(session, f"Sorry, I didn't quite follow that.{reason} Could you rephrase?")

        message = extraction.reply or self._review_fallback_reply(session, gaps, doc_type_missing, errors, notes)
        return self._as_result(session, message)

    # ---- helpers ----

    def _review_fallback_reply(
        self,
        session: Session,
        gaps: list,
        doc_type_missing: bool,
        errors: List[str],
        notes: List[str],
    ) -> str:
        """
        Deterministic fallback used whenever the LLM doesn't supply `reply`
        itself (provider call failed, or extraction_failed/needs_clarification
        with no reply) - so a response is never a hard dependency on the
        provider being up, same safety net phrase_question() used to be.
        """
        parts: List[str] = []
        if notes:
            parts.append(" ".join(n[0].upper() + n[1:] + ("." if not n.endswith(".") else "") for n in notes))

        if doc_type_missing:
            parts.append(document_type_question())
            return "\n\n".join(parts)

        if errors:
            parts.append("A few things still need fixing:\n" + "\n".join(f"- {e}" for e in errors))

        if gaps:
            gap = gaps[0]
            parts.append(fallback_question(gap.field_name, gap.attribute))
        elif not errors:
            parts.append(confirmation_prompt(session.schema_state.human_summary()))

        return "\n\n".join(parts)

    def _build_context(self, session: Session) -> dict:
        gaps = session.schema_state.all_gaps()
        return {
            "schema": session.schema_state.to_json_schema(),
            "document_type_missing": session.schema_state.document_type is None,
            "gaps": [{"field_name": g.field_name, "attribute": g.attribute} for g in gaps],
            "validation_errors": validate_schema(session.schema_state),
        }

    def _as_result(self, session: Session, message: str, errors: Optional[list] = None) -> TurnResult:
        return TurnResult(
            session_id=session.session_id,
            message=message,
            state=session.state.value,
            schema=session.schema_state.to_json_schema(),
            completed=session.completed,
            schema_id=session.schema_id,
            errors=errors,
        )
