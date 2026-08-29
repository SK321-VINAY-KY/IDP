"""
LLM adapter interface.

Every provider (Ollama today, Bedrock later, Mock for tests) implements
this same interface. The conversation manager never talks to a provider
SDK directly - only to this interface. Swapping Ollama for Bedrock later
should mean changing one line of config, not touching conversation logic.

Design boundary: the LLM's job is understanding + phrasing. It returns
*proposed* changes; it never mutates SchemaState or ConversationState
itself. app.core.conversation_manager decides what to actually do with
what comes back here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FieldOp(BaseModel):
    """
    One add/update/remove operation on a field, proposed by the LLM.

    Replaces the old split between NewFieldProposal (one shot, on add only)
    and FieldAnswer (one attribute, on an already-open gap): a single
    ExtractionResult.operations list can now add a field, correct another
    field's type, and remove a third, all from one user message, regardless
    of which fields already exist or which gaps are currently open. Applied
    by SchemaState.apply_operations - the LLM proposes, that method decides
    what actually lands (including normalizing/rejecting "type").
    """
    op: str  # "add" | "update" | "remove"
    field_name: str
    type: Optional[str] = None
    required: Optional[bool] = None
    currency: Optional[str] = None
    item_type: Optional[str] = None
    pattern: Optional[str] = None
    description: Optional[str] = None


class ExtractionResult(BaseModel):
    """
    What the LLM proposes after seeing one user message, in the single open
    REVIEW loop. The model is shown the *entire* current schema, every open
    gap, and any validation errors on every turn (see
    build_extraction_user_prompt) - so a message can answer several gaps,
    add/edit/remove fields, and/or confirm, all in the same reply. Which of
    those actually happen is still decided deterministically by
    ConversationManager/SchemaState, never by the LLM directly.
    """
    document_type: Optional[str] = None
    operations: List[FieldOp] = Field(default_factory=list)
    confirmation: Optional[bool] = None  # yes/no, only meaningful if the user was asked to confirm
    # The LLM composes the actual reply text shown to the user, so responses
    # read like a normal conversation instead of picking from prompts.py
    # templates every time. None (or extraction_failed/needs_clarification)
    # falls back to a deterministic template - see
    # ConversationManager._review_fallback_reply - so a reply is never a
    # hard dependency on the provider being up.
    reply: Optional[str] = None
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    # Set to True if the adapter itself failed to produce valid structured
    # output (bad JSON, provider error, etc.) so the caller can degrade
    # gracefully instead of silently applying garbage.
    extraction_failed: bool = False


class FieldObservation(BaseModel):
    """
    One field as observed across the sample documents given to
    infer_schema_from_pdfs. Mirrors FieldSpec's mutable attributes so
    ConversationManager can feed it straight into SchemaState.add_field()
    with no translation layer.

    The adapter should leave "type"/"required"/etc. as None whenever the
    samples disagree (e.g. present in 2 of 3, or formatted differently) -
    that's what makes SchemaState.next_gap() pick the field up automatically
    and route it through the normal interview instead of silently guessing.
    """
    name: str
    type: Optional[str] = None
    required: Optional[bool] = None
    currency: Optional[str] = None
    item_type: Optional[str] = None
    pattern: Optional[str] = None
    # How many of the N samples this field was actually observed in.
    # total_samples is filled in by the adapter (== len(samples)), not
    # trusted from the model, since it's a fact we already know deterministically.
    seen_in_samples: int = 0
    total_samples: int = 0
    notes: Optional[str] = None  # e.g. "amount format varies: Rs.1,200 vs 1200.00"


class SchemaProposal(BaseModel):
    """
    What an adapter proposes after reading 2-5 sample documents in one shot,
    via infer_schema_from_pdfs. Applied by ConversationManager the same way
    ExtractionResult is - the LLM never touches SchemaState directly.
    """
    document_type: Optional[str] = None
    fields: List[FieldObservation] = Field(default_factory=list)
    # Same degrade-gracefully contract as ExtractionResult.extraction_failed:
    # set this instead of raising, so the caller can fall back to the plain
    # text interview (ASK_DOCUMENT_TYPE) rather than erroring out.
    extraction_failed: bool = False
    failure_reason: Optional[str] = None


class LLMAdapter(ABC):
    @abstractmethod
    def extract(
        self,
        state: str,
        user_message: str,
        context: Dict[str, Any],
    ) -> ExtractionResult:
        """
        Interpret one user message given the full current schema, every open
        gap, and any validation errors (all passed via `context` - see
        build_extraction_user_prompt). `state` is passed through for
        logging/prompt context but the REVIEW loop no longer branches
        behavior on it the way the old per-state handlers did. Must not
        raise on malformed model output - catch it and return
        ExtractionResult(extraction_failed=True) instead.
        """
        raise NotImplementedError

    @abstractmethod
    def phrase_question(self, gap_field: str, gap_attribute: str, context: Dict[str, Any]) -> str:
        """
        Turn a (field, missing_attribute) pair into a natural clarifying
        question. Should have a deterministic template fallback if the
        provider call fails - this method must not raise.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_schema_from_pdfs(self, samples: List[bytes]) -> SchemaProposal:
        """
        Look across 2-5 sample PDF documents of the same document type (raw
        PDF bytes, one entry per sample) and propose a first-pass SchemaState:
        a document type plus a field list. Providers without a way to read
        documents (e.g. a text-only local model) should return
        SchemaProposal(extraction_failed=True, failure_reason=...) rather
        than raising, so the caller can fall back to the plain-text interview.
        """
        raise NotImplementedError
