"""
File: grounding_checker.py
Purpose: Phase 4 Part 2 — verify that each claim in a synthesized Query Bot
         answer actually traces back to specific retrieved evidence, BEFORE the
         answer is returned to the user.

Design (confirmed with user), mirrors the project's layered-fallback philosophy:
  - PRIMARY check: deterministic text-overlap, same spirit + same engine
    (difflib.SequenceMatcher, normalized text, a min-ratio floor) as Phase 2's
    evidence_resolver.py. Always available — no LLM, no AWS, no new dependency.
    This is the floor that always runs.
  - SECONDARY check (optional): a single cheap, targeted Bedrock call that
    judges per-claim entailment against the evidence, used to UPGRADE a claim
    the heuristic marked unverified (catches paraphrase/semantic grounding the
    overlap heuristic misses). Degrades gracefully to primary-only if Bedrock is
    unavailable (expired creds, network, etc.) — never raises.

Policy on ungrounded claims: FLAG as "unverified" in the response, never
silently drop. A wrong/absent citation is worse than an honest "unverified".

Each claim carries its citations (page + evidence span text, block_id where the
evidence span has one — stored data currently has none, but the field is wired
through so it lights up once Phase 1/2 block_ids are persisted).

Deps: stdlib (re, difflib). boto3 only imported lazily inside the optional pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Overlap ratio floor to call a claim grounded by the deterministic check.
# Aligned in spirit with evidence_resolver.MIN_MATCH_RATIO (0.55) but a touch
# lower: a claim is a paraphrase/summary of evidence, not a verbatim quote, so
# we require a weaker literal overlap here. Tunable.
GROUNDING_MIN_RATIO = 0.45

# Claims shorter than this (after normalization) are treated as non-substantive
# connective text (e.g. "The answer is") and are not independently graded.
MIN_CLAIM_CHARS = 12


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def split_claims(answer: str) -> List[str]:
    """Split a synthesized answer into claim-sized units. Sentence/clause
    granularity: split on sentence terminators and semicolons, drop the trailing
    source-citation parenthetical the synthesizer appends (e.g. '(Source: Page 4)')
    so the citation text itself isn't graded as a claim."""
    if not answer:
        return []
    # Strip trailing "(Source: Page N)" style citations from the gradable text.
    cleaned = re.sub(r"\(?\s*source:?\s*page\s*\d+[^)]*\)?", "", answer, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+|;\s+", cleaned)
    claims = []
    for p in parts:
        p = p.strip()
        if len(_normalize(p)) >= MIN_CLAIM_CHARS:
            claims.append(p)
    # Fall back to the whole answer if splitting produced nothing substantive.
    if not claims and len(_normalize(cleaned)) >= MIN_CLAIM_CHARS:
        claims.append(cleaned.strip())
    return claims


@dataclass
class ClaimVerdict:
    claim: str
    grounded: bool
    method: str                       # "overlap" | "llm" | "none"
    score: float                      # best overlap ratio (0-1) for the heuristic
    citations: List[Dict[str, Any]] = field(default_factory=list)  # page/evidence/block_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "grounded": self.grounded,
            "status": "grounded" if self.grounded else "unverified",
            "method": self.method,
            "score": round(self.score, 3),
            "citations": self.citations,
        }


# Content-word stop list for token overlap (claims are paraphrases/summaries of
# evidence, so function words shouldn't count toward grounding).
_STOP = {
    "the", "a", "an", "is", "was", "were", "are", "be", "of", "to", "in", "on",
    "for", "and", "or", "at", "by", "with", "as", "that", "this", "it", "its",
    "from", "patient", "hospital",  # domain-ubiquitous; too common to be evidential
}


def _content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9.,/-]*", _normalize(text))
    out = set()
    for t in toks:
        # Strip leading/trailing punctuation (e.g. a trailing sentence period on
        # "189771.25." that otherwise blocks the match with "189771.25").
        t = t.strip(".,/-")
        if not t:
            continue
        # Normalize numbers by removing thousands separators so
        # "189,771.25" == "189771.25".
        t = t.replace(",", "")
        if len(t) < 2 and not t.isdigit():
            continue
        if t in _STOP:
            continue
        out.add(t)
    return out


def _best_overlap(claim: str, evidence_items: List[Dict[str, Any]]) -> tuple[float, Optional[Dict[str, Any]]]:
    """Return (best_ratio, best_evidence_item) for a claim vs. the retrieved
    evidence spans.

    Uses the MAX of two signals, both normalized to "how much of the claim is
    supported by this evidence span":
      1. Contiguous longest-match ratio (difflib, same engine as Phase 2's
         evidence_resolver) — strong for near-verbatim claims.
      2. Content-token coverage: fraction of the claim's content tokens (numbers
         with separators normalized, stop/ubiquitous words removed) present in
         the evidence span — robust when the claim paraphrases the evidence
         around a shared key value (e.g. claim "the total bill was 189,771.25"
         vs evidence "Net Bill Amount 189,771.25").
    Taking the max keeps verbatim matches strong while rescuing paraphrases that
    share the evidential tokens."""
    norm_claim = _normalize(claim)
    if not norm_claim:
        return 0.0, None
    claim_tokens = _content_tokens(claim)
    best_ratio = 0.0
    best_item: Optional[Dict[str, Any]] = None
    for item in evidence_items:
        ev_text = _normalize(item.get("evidence", ""))
        if not ev_text:
            continue
        # Signal 1: contiguous span.
        m = SequenceMatcher(None, ev_text, norm_claim, autojunk=False)
        match = m.find_longest_match(0, len(ev_text), 0, len(norm_claim))
        contig = match.size / max(len(norm_claim), 1)
        # Signal 2: content-token coverage.
        ev_tokens = _content_tokens(ev_text)
        token_cov = (len(claim_tokens & ev_tokens) / len(claim_tokens)) if claim_tokens else 0.0
        ratio = max(contig, token_cov)
        if ratio > best_ratio:
            best_ratio = ratio
            best_item = item
    return best_ratio, best_item


def _citation_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a retrieved-evidence source item into a per-claim citation,
    surfacing block_id when the evidence span carries one (currently none in
    stored data; wired through for when Phase 1/2 block_ids are persisted)."""
    return {
        "page": item.get("page"),
        "evidence": item.get("evidence"),
        "block_ids": item.get("block_ids", []) or [],
        "node": item.get("node"),
    }


def check_grounding(
    answer: str,
    evidence_items: List[Dict[str, Any]],
    use_llm: bool = True,
    llm_caller=None,
) -> Dict[str, Any]:
    """
    Verify each claim in `answer` against `evidence_items` (the retrieved
    sources, each a dict with at least {page, evidence}).

    Returns a dict:
      {
        "claims": [ClaimVerdict.to_dict(), ...],
        "all_grounded": bool,
        "grounded_count": int,
        "total_claims": int,
        "unverified_claims": [claim strings],
        "method_summary": {"overlap": n, "llm": n, "none": n},
      }

    PRIMARY deterministic overlap always runs. The optional LLM pass only runs
    for claims the heuristic left ungrounded, only if use_llm and a caller is
    provided, and never raises (any failure => heuristic verdict stands).
    """
    claims = split_claims(answer)
    verdicts: List[ClaimVerdict] = []

    for claim in claims:
        ratio, item = _best_overlap(claim, evidence_items)
        if ratio >= GROUNDING_MIN_RATIO and item is not None:
            verdicts.append(ClaimVerdict(
                claim=claim, grounded=True, method="overlap", score=ratio,
                citations=[_citation_from_item(item)],
            ))
        else:
            verdicts.append(ClaimVerdict(
                claim=claim, grounded=False, method="none", score=ratio,
                citations=[_citation_from_item(item)] if item else [],
            ))

    # SECONDARY optional LLM pass — only for the ungrounded claims.
    if use_llm and llm_caller is not None:
        ungrounded = [v for v in verdicts if not v.grounded]
        if ungrounded:
            try:
                upgraded = _llm_grounding_pass(ungrounded, evidence_items, llm_caller)
                for v, (is_grounded, cite) in zip(ungrounded, upgraded):
                    if is_grounded:
                        v.grounded = True
                        v.method = "llm"
                        if cite:
                            v.citations = [cite]
            except Exception as exc:
                logger.warning("grounding.llm_pass_failed_degrading", error=str(exc))

    grounded = [v for v in verdicts if v.grounded]
    method_summary = {"overlap": 0, "llm": 0, "none": 0}
    for v in verdicts:
        method_summary[v.method if v.grounded else "none"] = method_summary.get(
            v.method if v.grounded else "none", 0
        ) + 1

    return {
        "claims": [v.to_dict() for v in verdicts],
        "all_grounded": len(grounded) == len(verdicts) and len(verdicts) > 0,
        "grounded_count": len(grounded),
        "total_claims": len(verdicts),
        "unverified_claims": [v.claim for v in verdicts if not v.grounded],
        "method_summary": method_summary,
    }


def _llm_grounding_pass(
    ungrounded_verdicts: List[ClaimVerdict],
    evidence_items: List[Dict[str, Any]],
    llm_caller,
) -> List[tuple[bool, Optional[Dict[str, Any]]]]:
    """One targeted Bedrock call judging entailment of each still-ungrounded
    claim against the evidence. llm_caller(system, user) -> str. Returns a list
    parallel to ungrounded_verdicts of (is_grounded, citation_or_None).

    Any parsing ambiguity resolves to NOT grounded (conservative: we would
    rather flag 'unverified' than wrongly assert grounding)."""
    ev_lines = []
    for i, item in enumerate(evidence_items):
        ev_lines.append(f"[E{i}] (Page {item.get('page')}) {item.get('evidence', '')}")
    evidence_block = "\n".join(ev_lines) if ev_lines else "(no evidence)"

    claim_lines = [f"[C{i}] {v.claim}" for i, v in enumerate(ungrounded_verdicts)]
    claims_block = "\n".join(claim_lines)

    system = (
        "You are a strict grounding verifier. For each claim, decide if it is "
        "DIRECTLY supported by the provided evidence spans. Answer ONLY with a "
        "JSON array, one object per claim, in order: "
        '[{"claim_index": int, "grounded": true|false, "evidence_index": int|null}]. '
        "Mark grounded=true ONLY if a specific evidence span states the claim. "
        "If unsure, grounded=false. No prose, JSON only."
    )
    user = f"EVIDENCE SPANS:\n{evidence_block}\n\nCLAIMS:\n{claims_block}\n\nJSON:"

    raw = llm_caller(system, user)  # may raise; caller wraps in try/except
    results: List[tuple[bool, Optional[Dict[str, Any]]]] = [(False, None)] * len(ungrounded_verdicts)
    if not raw:
        return results
    # Extract the JSON array.
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return results
    import json
    try:
        parsed = json.loads(raw[start:end + 1])
    except Exception:
        return results
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        ci = obj.get("claim_index")
        if not isinstance(ci, int) or ci < 0 or ci >= len(results):
            continue
        is_g = bool(obj.get("grounded"))
        cite = None
        ei = obj.get("evidence_index")
        if is_g and isinstance(ei, int) and 0 <= ei < len(evidence_items):
            cite = _citation_from_item(evidence_items[ei])
        results[ci] = (is_g, cite)
    return results
