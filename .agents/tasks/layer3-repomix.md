This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/ai/layer3_extraction/**/*.py, src/ai/layer3_extraction/prompts/templates/**/*.j2, src/ai/layer3_extraction/prompts/configs/**/*.yaml, schema_chatbot_v2/app/api/pipeline_routes.py, schema_chatbot_v2/app/api/routes.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/*.pyo, **/migrations/**, repomix-output.*, repomix-layer3.*, repomix-layer12.*, repomix*.xml, repomix*.md
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
schema_chatbot_v2/
  app/
    api/
      pipeline_routes.py
      routes.py
src/
  ai/
    layer3_extraction/
      graph_agent/
        __init__.py
        agent.py
        evidence_resolver.py
        evidence_retriever.py
        graph_memory.py
        grounding_checker.py
        memory_manager.py
        models.py
        query_service.py
        resolver.py
      prompts/
        configs/
          prompt_config.yaml
        templates/
          extraction_system.j2
          graph_page_ingestion.j2
          graph_schema_resolution.j2
          navigation.j2
          page_field_check.j2
          page_summary.j2
        __init__.py
        loader.py
      __init__.py
      chunker.py
      extractor.py
      ir_store.py
      page_loader.py
      s3_documents.py
      schema_validation.py
      scratchpad.py
      storage.py
```

# Files

## File: src/ai/layer3_extraction/graph_agent/__init__.py
```python
"""
Layer 3 Graph-Memory Extraction Strategy Package.
"""
from src.ai.layer3_extraction.graph_agent.models import (
    Evidence,
    GraphEdge,
    GraphNode,
    PageProcessingResult,
)
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
from src.ai.layer3_extraction.graph_agent.query_service import (
    GraphQueryService,
    GraphQueryResult,
)

__all__ = [
    "Evidence",
    "GraphEdge",
    "GraphNode",
    "PageProcessingResult",
    "GraphMemory",
    "GraphExtractionAgent",
    "resolve_schema_from_graph",
    "GraphQueryService",
    "GraphQueryResult",
]
```

## File: src/ai/layer3_extraction/graph_agent/evidence_resolver.py
```python
"""
File: evidence_resolver.py
Purpose: Phase 2 of DocumentIR — resolve an LLM-provided evidence text
         snippet (from PageDelta extraction) back to the DocumentIR
         block(s) it came from, so GraphNode/Evidence can carry block_ids
         + bbox instead of just free text.

         Deliberately NOT a single-block match. The extraction prompt
         asks the LLM for "exact verbatim text snippet" at SENTENCE
         granularity, but PaddleOCR blocks are per OCR LINE (see
         paddle_block_rollup.py) and Docling blocks are per paragraph.
         A sentence will often span 2+ line-blocks. This resolves a
         CONTIGUOUS SPAN of blocks, not one block_id.

         Deterministic, no LLM call, no new dependency beyond stdlib
         difflib. Accuracy here is exactly the kind of thing the Phase 3
         eval harness should measure — this is a best-effort heuristic,
         not guaranteed-correct, and degrades gracefully (empty
         block_ids, not an exception) when no good match is found.

Owner: TBD
Deps: stdlib only (difflib, re)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

try:
    from src.ai.schemas.document_ir import Block
except ImportError:  # pragma: no cover - import path varies by deployment context
    Block = None  # type: ignore


# Minimum match ratio (0-1) to accept a resolution at all. Below this,
# we'd rather return no block_ids than a wrong one — a wrong citation is
# worse than an absent one for anything downstream (highlighting, review).
MIN_MATCH_RATIO = 0.55


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def resolve_evidence_to_blocks(
    evidence_text: str,
    page_blocks: List["Block"],
) -> Tuple[List[str], Optional[dict]]:
    """
    Find the contiguous span of page_blocks (in reading order) whose
    concatenated text best matches evidence_text.

    Returns (block_ids, union_bbox_dict). block_ids is [] and
    union_bbox_dict is None when no match clears MIN_MATCH_RATIO —
    callers should treat that as "no resolution", not an error.
    """
    norm_evidence = _normalize(evidence_text)
    if not norm_evidence or not page_blocks:
        return [], None

    ordered = sorted(
        page_blocks,
        key=lambda b: (b.reading_order if b.reading_order is not None else 1_000_000),
    )

    # Build one normalized concatenated string for the page, tracking which
    # block each character range belongs to, so a match in the concatenated
    # string can be mapped back to the specific blocks it spans.
    concatenated = ""
    char_to_block_idx: List[int] = []  # parallel array: char_to_block_idx[i] = index into `ordered`
    for idx, block in enumerate(ordered):
        block_text = _normalize(block.text)
        if not block_text:
            continue
        if concatenated:
            concatenated += " "
            char_to_block_idx.append(-1)  # separator belongs to no block
        start = len(concatenated)
        concatenated += block_text
        char_to_block_idx.extend([idx] * (len(concatenated) - start))

    if not concatenated:
        return [], None

    matcher = SequenceMatcher(None, concatenated, norm_evidence, autojunk=False)
    match = matcher.find_longest_match(0, len(concatenated), 0, len(norm_evidence))

    if match.size == 0:
        return [], None

    match_ratio = match.size / max(len(norm_evidence), 1)
    if match_ratio < MIN_MATCH_RATIO:
        return [], None

    matched_block_indices = sorted({
        char_to_block_idx[i]
        for i in range(match.a, match.a + match.size)
        if 0 <= i < len(char_to_block_idx) and char_to_block_idx[i] != -1
    })
    if not matched_block_indices:
        return [], None

    matched_blocks = [ordered[i] for i in matched_block_indices]
    block_ids = [b.block_id for b in matched_blocks]

    boxed = [b.bbox for b in matched_blocks if b.bbox is not None]
    union_bbox = None
    if boxed:
        union_bbox = {
            "x0": min(b.x0 for b in boxed),
            "y0": min(b.y0 for b in boxed),
            "x1": max(b.x1 for b in boxed),
            "y1": max(b.y1 for b in boxed),
        }

    return block_ids, union_bbox
```

## File: src/ai/layer3_extraction/graph_agent/evidence_retriever.py
```python
"""
File: evidence_retriever.py
Purpose: Phase 4 — semantic (vector) retrieval over graph node EVIDENCE SPANS,
         added ALONGSIDE the existing keyword seed search in query_service.py.

Why evidence spans (not DocumentIR blocks): DocumentIR blocks are never
persisted (they are transient in Layer 2). The persisted, citable unit at the
same page-level precision is the GraphNode Evidence span (text + page_number),
which is exactly what the evidence resolver and GraphQueryResult.sources already
use. Embedding spans keeps retrieval citable at the precision Phase 2 achieved.

Design (confirmed with user, Phase 4 Part 1):
- Model: sentence-transformers/all-MiniLM-L6-v2
  pinned revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 (384-dim).
- Store: in-process flat numpy matrix + cosine similarity. No pgvector
  (extension unavailable on the local Postgres 18.6), no new infra.
- Scope: one index per GraphMemory instance, built lazily, cached by a cheap
  content signature so repeated queries on the same graph don't re-embed.

This module adds a NEW candidate source. It does not modify find_seed_nodes,
traverse_subgraph, or _deterministic_fallback.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import GraphNode
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pinned model + revision (see requirements.txt).
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_DIM = 384

# Lazy singletons — the model loads once per process (~90MB), the first encode
# is the only slow call. Guarded so concurrent queries don't double-load.
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the embedding model once per process. Returns None if the library
    or weights are unavailable, so callers can degrade gracefully to keyword-only."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBED_MODEL_NAME, revision=EMBED_MODEL_REVISION)
            logger.info(
                "evidence_retriever.model_loaded",
                model=EMBED_MODEL_NAME,
                revision=EMBED_MODEL_REVISION,
            )
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning("evidence_retriever.model_load_failed", error=str(exc))
            _model = None
    return _model


def _embed(texts: List[str]) -> Optional[np.ndarray]:
    """Encode texts to L2-normalized float32 vectors. None on failure."""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        vecs = model.encode(
            texts,
            normalize_embeddings=True,  # unit vectors -> dot product == cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        logger.warning("evidence_retriever.embed_failed", error=str(exc))
        return None


@dataclass
class EvidenceSpan:
    """One embedded evidence span, carrying back-pointers for citation + seeding."""
    node_id: str
    text: str
    page_number: int
    confidence: float


class EvidenceSpanIndex:
    """In-process cosine-similarity index over a graph's evidence spans.

    Built from a GraphMemory. Each row is one Evidence span (deduped by
    (node_id, page, normalized-text)). search() returns the owning GraphNodes
    so results can be merged into the existing BFS seed set.
    """

    def __init__(self, graph: GraphMemory) -> None:
        self._graph = graph
        self._spans: List[EvidenceSpan] = []
        self._matrix: Optional[np.ndarray] = None
        self._available = False
        self._build()

    def _build(self) -> None:
        # KNOWN LIMITATION (Phase 4, verified on Chander Kochhar graph100):
        # Terse numeric-value evidence spans — e.g. a bare amount span like
        # "Sanctioned 80698" or "80698" — embed poorly against semantic
        # paraphrase queries ("how much did the insurer agree to pay"), because
        # the span carries almost no textual context for the embedding model to
        # match the question's intent against. In testing, such paraphrases
        # matched the insurer *name* span far higher than the terse amount span.
        # Vector retrieval therefore helps text/entity-name answers much more
        # than terse numeric answers.
        # FUTURE IMPROVEMENT (not doing now): prepend the owning node's field
        # label / schema description to the span text before embedding
        # (e.g. "sanctioned_amount approved claim amount: 80698") so numeric
        # spans gain the context the model needs. Left as-is this phase.
        spans: List[EvidenceSpan] = []
        seen: set[Tuple[str, int, str]] = set()
        for node in graph_nodes(self._graph):
            for ev in getattr(node, "evidence", None) or []:
                text = (getattr(ev, "text", "") or "").strip()
                if len(text) < 3:
                    continue
                page = int(getattr(ev, "page_number", 0) or 0)
                key = (node.id, page, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    EvidenceSpan(
                        node_id=node.id,
                        text=text,
                        page_number=page,
                        confidence=float(getattr(ev, "confidence", 1.0) or 1.0),
                    )
                )

        if not spans:
            logger.info("evidence_retriever.index_empty", doc_spans=0)
            return

        matrix = _embed([s.text for s in spans])
        if matrix is None:
            # Model unavailable -> index stays unavailable; caller degrades to keyword-only.
            return

        self._spans = spans
        self._matrix = matrix
        self._available = True
        logger.info("evidence_retriever.index_built", spans=len(spans), dim=matrix.shape[1])

    @property
    def available(self) -> bool:
        return self._available

    @property
    def span_count(self) -> int:
        return len(self._spans)

    def search(
        self,
        question: str,
        top_k: int = 8,
        min_score: float = 0.30,
    ) -> List[Tuple[GraphNode, float, EvidenceSpan]]:
        """Return up to top_k (node, cosine_score, span) tuples whose evidence
        span is most semantically similar to the question. Deduped so each
        GraphNode appears at most once (keeping its best-scoring span)."""
        if not self._available or self._matrix is None:
            return []
        q = _embed([question])
        if q is None:
            return []
        # cosine == dot product (both sides normalized)
        scores = (self._matrix @ q[0]).astype(float)
        order = np.argsort(-scores)

        results: List[Tuple[GraphNode, float, EvidenceSpan]] = []
        seen_nodes: set[str] = set()
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                break
            span = self._spans[idx]
            if span.node_id in seen_nodes:
                continue
            node = self._graph.get_node(span.node_id)
            if node is None:
                continue
            seen_nodes.add(span.node_id)
            results.append((node, score, span))
            if len(results) >= top_k:
                break
        return results


def graph_nodes(graph: GraphMemory) -> List[GraphNode]:
    """Access nodes via the public property if present, else the internal dict.
    Isolated here so query_service doesn't reach into graph internals for this."""
    nodes_attr = getattr(graph, "nodes", None)
    if isinstance(nodes_attr, dict):
        return list(nodes_attr.values())
    if callable(nodes_attr):
        try:
            val = nodes_attr()
            if isinstance(val, dict):
                return list(val.values())
        except Exception:
            pass
    internal = getattr(graph, "_nodes", {})
    return list(internal.values())


# Per-graph index cache keyed by a cheap content signature, so repeated queries
# against the same loaded graph reuse the embeddings instead of re-encoding.
_index_cache: Dict[int, Tuple[int, EvidenceSpanIndex]] = {}
_cache_lock = threading.Lock()


def get_index_for_graph(graph: GraphMemory) -> EvidenceSpanIndex:
    """Return a cached EvidenceSpanIndex for this graph, building it once.
    Keyed by id(graph)+node-count so a mutated/reloaded graph rebuilds."""
    sig = len(graph_nodes(graph))
    key = id(graph)
    with _cache_lock:
        cached = _index_cache.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        index = EvidenceSpanIndex(graph)
        _index_cache[key] = (sig, index)
        return index
```

## File: src/ai/layer3_extraction/graph_agent/grounding_checker.py
```python
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
```

## File: src/ai/layer3_extraction/prompts/templates/extraction_system.j2
```jinja
You are extracting structured data from a document. Read the ENTIRE
document text below carefully before answering.

Target schema fields and their descriptions:
{% for field in schema_fields %}
- {{ field.name }}{% if field.description %}: {{ field.description }}{% endif %}

{% endfor %}

Rules:
- Extract each field strictly according to its description.
- Field names that include "first" mean the first occurrence in document order.
- Field names that include "last" mean the last occurrence in document order.
- If the exact value is stated in the text, extract it exactly as written.
- If a value can be reasonably inferred from context (e.g., "17 Sustainable
  Development Goals" implies total_goals = "17"), extract that inferred value.
- If a field's information genuinely does not appear anywhere in the text,
  set it to null. Do not guess or fabricate values.

Respond with ONLY the structured JSON output — no explanation, no preamble.
```

## File: src/ai/layer3_extraction/prompts/templates/graph_schema_resolution.j2
```jinja
You are synthesizing final structured schema fields using a Contextual Document Knowledge Graph.

================================================================================
TARGET SCHEMA FIELDS
================================================================================
{% for field in schema_fields %}
- {{ field.name }}{% if field.description %}: {{ field.description }}{% endif %}
{% endfor %}

================================================================================
DOCUMENT KNOWLEDGE GRAPH EVIDENCE
================================================================================
{{ graph_evidence }}

================================================================================
RULES & INSTRUCTIONS
================================================================================
1. Extract the exact value for each target schema field using the entities, relationships, and provenances in the graph.
2. Follow entity relationship chains to resolve values (e.g. Patient -> Admission -> Claim -> Approved Amount).
3. If an entity was resolved across multiple pages (e.g. "the above patient" linked to the patient on Page 1), use the fully resolved entity and its associated facts.
4. If a field was not found anywhere in the graph, set it to an empty string "".
5. Output ONLY a valid JSON object matching the target schema field names as keys. No markdown fences, no preamble.
```

## File: src/ai/layer3_extraction/prompts/templates/navigation.j2
```jinja
Document page summaries:
{{ page_summaries }}

Target fields to extract: {{ schema_fields }}

For each field, list the page numbers (as INTEGERS, not strings) where
information for that field is likely to appear. If a field's information
could be spread across most or all pages, include all relevant page numbers.

Return ONLY valid JSON in this exact format, no explanation:
{"field_name": [1, 2, 3]}
```

## File: src/ai/layer3_extraction/prompts/templates/page_summary.j2
```jinja
Summarize this page's topic and content in {{ max_words }} words or fewer.
Preserve any specific labels, numbers, dates, titles, or names EXACTLY as
written — do not paraphrase these.

Page:
{{ page_md }}
```

## File: src/ai/layer3_extraction/prompts/__init__.py
```python

```

## File: src/ai/layer3_extraction/prompts/loader.py
```python
"""
File: loader.py
Purpose: Load and render versioned prompt templates for Layer 3.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Deps: jinja2, pyyaml
"""
import yaml
from functools import lru_cache
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIG_PATH = Path(__file__).parent / "configs" / "prompt_config.yaml"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


@lru_cache(maxsize=1)
def _prompt_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_prompt(prompt_name: str, **context) -> str:
    config = _prompt_config()[prompt_name]
    template = _env().get_template(config["template"])
    return template.render(**context)


def prompt_params(prompt_name: str) -> dict:
    config = _prompt_config()[prompt_name]
    return {"temperature": config["temperature"], "max_tokens": config["max_tokens"]}
```

## File: src/ai/layer3_extraction/__init__.py
```python

```

## File: src/ai/layer3_extraction/chunker.py
```python
"""
File: chunker.py
Purpose: Phase 4, commit 2 — turn persisted DocumentIR (PageIR + Block) into
         retrieval Chunks. A chunk is the unit the embedder/lexical index
         (later commits) will carry, and the unit a citation resolves to.

PURE / DETERMINISTIC / OFFLINE:
  No model, no AWS, no network, stdlib + pydantic only. The same input
  produces byte-identical output across runs and across PYTHONHASHSEED
  values: blocks are ordered by a stable (reading_order, block_id) key,
  there is no iteration over sets of strings, and ids are positional.

TENANT ISOLATION:
  Every public function that takes `owner` requires it with NO default, and
  the owner is carried on every Chunk. Chunk ids are positional and content
  free (c{page}-{seq}), so an id never leaks document text.

CONTENT SAFETY:
  This module copies block text into Chunk.text (that is its job), but it
  never logs or prints that text. Callers that build reports must emit counts
  and ids only.

Owner: Phase 4 retrieval. Deps: pydantic, stdlib (re).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.ai.schemas.document_ir import BBox, Block, BlockType, PageIR, TableCell

# ---------------------------------------------------------------------------
# Tunables (documented defaults; all overridable via chunk_page_irs(..., config=))
# ---------------------------------------------------------------------------

# Target max characters for a coalesced text chunk. Paragraph/heading/line
# blocks are packed in reading order until adding the next block would exceed
# this, then a new chunk starts. ~400 chars ≈ a few sentences — small enough
# to localise a citation, large enough to keep a label with its value.
DEFAULT_TARGET_CHARS = 400

# A block shorter than this (after strip) is a candidate "label" that should be
# glued to the following value block so e.g. "Sanctioned Amount" and its number
# land in one chunk. See _is_label_like for the full rule.
LABEL_MAX_CHARS = 40


class ChunkConfig(BaseModel):
    """Chunker tunables. Kept explicit (not kwargs) so a config is reproducible
    and can be logged by id/values without any document content."""

    target_chars: int = DEFAULT_TARGET_CHARS
    label_max_chars: int = LABEL_MAX_CHARS
    prepend_table_header: bool = True


class Chunk(BaseModel):
    """One retrieval unit. Never spans a page boundary. block_ids is the ordered
    list of source Block.block_id values this chunk was built from, so a citation
    can map back to exact blocks + bbox."""

    chunk_id: str                       # positional, content-free: "c{page}-{seq}"
    owner: str
    doc_id: str
    page: int
    block_ids: List[str]
    bbox: Optional[BBox] = None         # union of source block bboxes; None if none had one
    text: str = ""
    source_engine: str = ""
    kind: str = "other"                 # paragraph | heading | table_row | list | other
    row_index: Optional[int] = None     # set only when kind == "table_row"


# ---------------------------------------------------------------------------
# Ordering & small helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•·]|\(?\d+[.)]|[a-z]\))\s+", re.IGNORECASE)


def _order_key(block: Block) -> Tuple[int, str]:
    """Stable ordering: reading_order first (blocks with none sort last), then
    block_id as a deterministic tie-break. No set iteration, no hash reliance."""
    ro = block.reading_order if block.reading_order is not None else 1_000_000
    return (ro, block.block_id or "")


def _union_bbox(boxes: List[Optional[BBox]]) -> Optional[BBox]:
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    return BBox(
        x0=min(b.x0 for b in present),
        y0=min(b.y0 for b in present),
        x1=max(b.x1 for b in present),
        y1=max(b.y1 for b in present),
    )


def _is_label_like(block: Block) -> bool:
    """Heuristic (no model) for a block that is a field label whose value is the
    next block in reading order. The LENGTH gate is applied by the caller (it
    owns the configurable cap); this tests only the SHAPE:
      - it does NOT end in sentence punctuation (. ! ?), and
      - EITHER it ends with ':' (classic "Label:")
        OR it has no internal '.' and <= 6 words (e.g. "Sanctioned Amount").
    Headings are never labels. Deterministic string tests only."""
    t = (block.text or "").strip()
    if not t:
        return False
    if block.type == BlockType.HEADING:
        return False  # headings get their own chunk, not glued to a value
    if t[-1] in ".!?":
        return False
    if t.endswith(":"):
        return True
    return ("." not in t) and (len(t.split()) <= 6)


def _kind_for(block: Block) -> str:
    if block.type == BlockType.HEADING:
        return "heading"
    if _LIST_PREFIX.match(block.text or ""):
        return "list"
    if block.type in (BlockType.PARAGRAPH, BlockType.LINE, BlockType.CAPTION):
        return "paragraph"
    return "other"


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------

def _table_grid(cells: List[TableCell]) -> Dict[int, Dict[int, TableCell]]:
    """Map (row -> col -> cell). Later cells at the same (row,col) win, which is
    deterministic given the input order; row/col spans are preserved on the cell."""
    grid: Dict[int, Dict[int, TableCell]] = {}
    for c in cells:
        grid.setdefault(c.row, {})[c.column] = c
    return grid


def _row_text(grid_row: Dict[int, TableCell]) -> str:
    """Join a row's cells left-to-right by column index. Empty cells become
    empty strings so column alignment is preserved; separator is ' | '."""
    if not grid_row:
        return ""
    max_col = max(grid_row.keys())
    parts = [(grid_row.get(c).text.strip() if grid_row.get(c) else "") for c in range(max_col + 1)]
    return " | ".join(parts)


def _has_header(grid: Dict[int, Dict[int, TableCell]]) -> bool:
    """A table has a usable header when row 0 exists and has >=2 non-empty cells
    and there is at least one data row below it. Conservative on purpose: a bad
    prepend corrupts every row chunk."""
    if 0 not in grid:
        return False
    nonempty = sum(1 for cell in grid[0].values() if (cell.text or "").strip())
    data_rows = [r for r in grid.keys() if r > 0]
    return nonempty >= 2 and len(data_rows) >= 1


def _table_row_chunks(
    block: Block,
    page: int,
    owner: str,
    doc_id: str,
    seq_start: int,
    config: ChunkConfig,
) -> Tuple[List[Chunk], int]:
    """One chunk per data row; header row text prepended for context when a
    header is detected. When the TABLE block has no `cells`, fall back to a
    single chunk from block.text (so a table still produces something)."""
    seq = seq_start
    if not block.cells:
        # No structured cells: treat the whole table as one text chunk.
        text = (block.text or "").strip()
        if not text:
            return [], seq
        chunk = Chunk(
            chunk_id=f"c{page}-{seq}",
            owner=owner, doc_id=doc_id, page=page,
            block_ids=[block.block_id],
            bbox=block.bbox,
            text=text,
            source_engine=block.source_engine,
            kind="table_row",
            row_index=None,
        )
        return [chunk], seq + 1

    grid = _table_grid(block.cells)
    header_text = ""
    if config.prepend_table_header and _has_header(grid):
        header_text = _row_text(grid[0])

    chunks: List[Chunk] = []
    data_rows = sorted(r for r in grid.keys() if (not header_text or r > 0))
    for r in data_rows:
        body = _row_text(grid[r])
        if not body.strip():
            continue  # skip fully empty rows
        text = f"{header_text}\n{body}" if header_text else body
        cell_boxes = [c.bbox for c in grid[r].values()]
        # include header cell boxes in the union so the citation covers context too
        if header_text:
            cell_boxes += [c.bbox for c in grid[0].values()]
        chunks.append(Chunk(
            chunk_id=f"c{page}-{seq}",
            owner=owner, doc_id=doc_id, page=page,
            block_ids=[block.block_id],
            bbox=_union_bbox(cell_boxes),
            text=text,
            source_engine=block.source_engine,
            kind="table_row",
            row_index=r,
        ))
        seq += 1
    return chunks, seq


# ---------------------------------------------------------------------------
# Long-block splitting (never mid-word)
# ---------------------------------------------------------------------------

def _split_long_text(text: str, cap: int) -> List[str]:
    """Split an over-cap single block at sentence boundaries, falling back to
    whitespace. Never splits inside a word. Returns >=1 pieces, each <= cap
    where possible (a single word longer than cap is emitted whole)."""
    text = text.strip()
    if len(text) <= cap:
        return [text]
    pieces: List[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) <= cap:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(sentence) <= cap:
            buf = sentence
        else:
            # sentence itself too long: split on whitespace, never mid-word
            word_buf = ""
            for word in sentence.split():
                cand = f"{word_buf} {word}".strip() if word_buf else word
                if len(cand) <= cap:
                    word_buf = cand
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    word_buf = word  # a lone over-cap word is kept whole
            if word_buf:
                buf = word_buf
    if buf:
        pieces.append(buf)
    return pieces or [text]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_page_irs(
    owner: str,
    doc_id: str,
    pages: List[PageIR],
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """Turn persisted PageIR objects into deterministic Chunks.

    owner is REQUIRED (no default). A chunk never crosses a page boundary.
    Text blocks are coalesced in reading order up to config.target_chars;
    label-like blocks are kept with the following value block; tables produce
    one chunk per data row with the header prepended. Empty-text blocks are
    skipped. Blocks with no bbox still chunk (null bbox)."""
    if not owner or not str(owner).strip():
        raise ValueError("owner is required and cannot be empty")
    cfg = config or ChunkConfig()
    all_chunks: List[Chunk] = []

    for page_ir in sorted(pages, key=lambda p: p.page_number):
        page = page_ir.page_number
        ordered = sorted(page_ir.blocks, key=_order_key)
        seq = 0
        i = 0
        # pending coalescing buffer for text/heading/list blocks
        buf_ids: List[str] = []
        buf_boxes: List[Optional[BBox]] = []
        buf_text_parts: List[str] = []
        buf_engine = ""
        buf_kind = "paragraph"

        def _flush() -> None:
            nonlocal seq, buf_ids, buf_boxes, buf_text_parts, buf_engine, buf_kind
            if not buf_ids:
                return
            text = "\n".join(tp for tp in buf_text_parts if tp)
            all_chunks.append(Chunk(
                chunk_id=f"c{page}-{seq}",
                owner=owner, doc_id=doc_id, page=page,
                block_ids=list(buf_ids),
                bbox=_union_bbox(buf_boxes),
                text=text,
                source_engine=buf_engine,
                kind=buf_kind,
            ))
            seq += 1
            buf_ids = []
            buf_boxes = []
            buf_text_parts = []
            buf_engine = ""
            buf_kind = "paragraph"

        while i < len(ordered):
            block = ordered[i]
            btext = (block.text or "").strip()

            # Tables: flush any pending text, then emit per-row chunks.
            if block.type == BlockType.TABLE:
                _flush()
                row_chunks, seq = _table_row_chunks(block, page, owner, doc_id, seq, cfg)
                all_chunks.extend(row_chunks)
                i += 1
                continue

            if not btext:  # skip empty-text blocks
                i += 1
                continue

            # Headings get their own chunk (flush pending, emit, continue).
            if block.type == BlockType.HEADING:
                _flush()
                all_chunks.append(Chunk(
                    chunk_id=f"c{page}-{seq}",
                    owner=owner, doc_id=doc_id, page=page,
                    block_ids=[block.block_id],
                    bbox=block.bbox,
                    text=btext,
                    source_engine=block.source_engine,
                    kind="heading",
                ))
                seq += 1
                i += 1
                continue

            # Label+value gluing: a short label-like block binds to the NEXT
            # block in reading order (whatever its length) in the same chunk.
            is_label = len(btext) <= cfg.label_max_chars and _is_label_like(block)
            if is_label and (i + 1) < len(ordered):
                nxt = ordered[i + 1]
                nxt_text = (nxt.text or "").strip()
                if nxt_text and nxt.type not in (BlockType.TABLE, BlockType.HEADING):
                    _flush()  # don't bleed previous paragraph into the label pair
                    all_chunks.append(Chunk(
                        chunk_id=f"c{page}-{seq}",
                        owner=owner, doc_id=doc_id, page=page,
                        block_ids=[block.block_id, nxt.block_id],
                        bbox=_union_bbox([block.bbox, nxt.bbox]),
                        text=f"{btext} {nxt_text}",
                        source_engine=block.source_engine or nxt.source_engine,
                        kind=_kind_for(nxt),
                    ))
                    seq += 1
                    i += 2
                    continue

            # Over-cap single block: emit as its own chunk(s), same block_ids.
            if len(btext) > cfg.target_chars:
                _flush()
                for piece in _split_long_text(btext, cfg.target_chars):
                    all_chunks.append(Chunk(
                        chunk_id=f"c{page}-{seq}",
                        owner=owner, doc_id=doc_id, page=page,
                        block_ids=[block.block_id],
                        bbox=block.bbox,
                        text=piece,
                        source_engine=block.source_engine,
                        kind=_kind_for(block),
                    ))
                    seq += 1
                i += 1
                continue

            # Normal coalescing: would adding this block exceed the cap?
            current_len = sum(len(tp) for tp in buf_text_parts) + len(buf_text_parts)
            if buf_ids and (current_len + len(btext) > cfg.target_chars):
                _flush()
            if not buf_ids:
                buf_engine = block.source_engine
                buf_kind = _kind_for(block)
            buf_ids.append(block.block_id)
            buf_boxes.append(block.bbox)
            buf_text_parts.append(btext)
            i += 1

        _flush()

    return all_chunks


def chunk_id_to_blocks(
    chunk_id: str,
    chunks: List[Chunk],
) -> Tuple[List[str], Optional[BBox]]:
    """Map a chunk_id back to its (block_ids, union bbox) for citations.
    Returns ([], None) when the id is not present (never raises)."""
    for c in chunks:
        if c.chunk_id == chunk_id:
            return list(c.block_ids), c.bbox
    return [], None
```

## File: src/ai/layer3_extraction/s3_documents.py
```python
"""
File: s3_documents.py
Purpose: S3-backed storage for source PDF documents.

Source PDFs live in S3 (bucket from settings.s3_bucket) under a documents
prefix, instead of the local `dataset/` folder. Shared (admin) documents use:

    <prefix>/<filename>.pdf

Per-user documents are isolated under:

    <prefix>/_users/<username>/<filename>.pdf

All functions degrade gracefully (return empty / None / False) when the bucket
is not configured or an S3 error occurs, so callers can treat "no S3" the same
as "no documents".

Owner: engineer-a@idp-pilot
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import List, Optional

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_USERS_SEGMENT = "_users"


def is_configured() -> bool:
    """True if an S3 bucket is configured for document storage."""
    return bool(getattr(settings, "s3_bucket", "") or "")


def _client():
    import boto3  # imported lazily so environments without boto3 still import this module
    region = getattr(settings, "s3_region", None) or "ap-south-1"
    return boto3.client("s3", region_name=region)


def _prefix() -> str:
    p = (getattr(settings, "s3_documents_prefix", "documents") or "documents").strip("/")
    return p


def _shared_prefix() -> str:
    return f"{_prefix()}/"


def _user_prefix(username: str) -> str:
    uname = (username or "").strip().lower()
    return f"{_prefix()}/{_USERS_SEGMENT}/{uname}/"


def _shared_key(filename: str) -> str:
    return f"{_shared_prefix()}{filename}"


def _user_key(username: str, filename: str) -> str:
    return f"{_user_prefix(username)}{filename}"


def _is_pdf_key(key: str) -> bool:
    return key.lower().endswith(".pdf")


def _list_objects(prefix: str, recursive: bool = False) -> List[dict]:
    """
    List object summaries under a prefix. When recursive is False, only objects
    directly under the prefix (no deeper "folders") are returned.
    """
    if not is_configured():
        return []
    bucket = settings.s3_bucket
    out: List[dict] = []
    try:
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key == prefix or key.endswith("/"):
                    continue
                out.append({
                    "key": key,
                    "name": key.rsplit("/", 1)[-1],
                    "size": obj.get("Size", 0),
                    "modified": (obj.get("LastModified") or datetime.now(timezone.utc)).isoformat(),
                })
    except Exception as exc:
        logger.warning("s3_documents.list_failed", prefix=prefix, error=str(exc))
    return out


def list_shared_documents() -> List[dict]:
    """List shared (admin dataset) PDFs directly under the documents prefix."""
    return [d for d in _list_objects(_shared_prefix(), recursive=False) if _is_pdf_key(d["key"])]


def list_user_documents(username: str) -> List[dict]:
    """List a single user's PDFs under their isolated prefix."""
    return [d for d in _list_objects(_user_prefix(username), recursive=False) if _is_pdf_key(d["key"])]


def upload_shared_document(filename: str, data: bytes, content_type: str = "application/pdf") -> Optional[str]:
    """Upload a shared document to S3. Returns the object key, or None on failure."""
    return _put(_shared_key(filename), data, content_type)


def upload_user_document(username: str, filename: str, data: bytes, content_type: str = "application/pdf") -> Optional[str]:
    """Upload a per-user document to S3. Returns the object key, or None on failure."""
    return _put(_user_key(username, filename), data, content_type)


def _put(key: str, data: bytes, content_type: str) -> Optional[str]:
    if not is_configured():
        logger.warning("s3_documents.put_skipped_no_bucket", key=key)
        return None
    try:
        _client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type or "application/pdf",
        )
        logger.info("s3_documents.uploaded", key=key, size=len(data))
        return key
    except Exception as exc:
        logger.error("s3_documents.put_failed", key=key, error=str(exc))
        return None


def download_shared_document(filename: str) -> Optional[bytes]:
    return _get(_shared_key(filename))


def download_user_document(username: str, filename: str) -> Optional[bytes]:
    return _get(_user_key(username, filename))


def download_bytes_by_key(key: str) -> Optional[bytes]:
    return _get(key)


def _get(key: str) -> Optional[bytes]:
    if not is_configured():
        return None
    try:
        resp = _client().get_object(Bucket=settings.s3_bucket, Key=key)
        return resp["Body"].read()
    except Exception as exc:
        logger.warning("s3_documents.get_failed", key=key, error=str(exc))
        return None


def shared_document_exists(filename: str) -> bool:
    return _exists(_shared_key(filename))


def user_document_exists(username: str, filename: str) -> bool:
    return _exists(_user_key(username, filename))


def _exists(key: str) -> bool:
    if not is_configured():
        return False
    try:
        _client().head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except Exception:
        return False


def delete_shared_document(filename: str) -> bool:
    return _delete(_shared_key(filename))


def delete_user_document(username: str, filename: str) -> bool:
    return _delete(_user_key(username, filename))


def _delete(key: str) -> bool:
    if not is_configured():
        return False
    try:
        _client().delete_object(Bucket=settings.s3_bucket, Key=key)
        logger.info("s3_documents.deleted", key=key)
        return True
    except Exception as exc:
        logger.error("s3_documents.delete_failed", key=key, error=str(exc))
        return False
```

## File: src/ai/layer3_extraction/schema_validation.py
```python
"""
File: schema_validation.py
Purpose: Retry/repair loop for extraction validation failures.
Owner: engineer-b@idp-pilot
Created: 2026-08-20
"""
from typing import Callable, Optional
from pydantic import BaseModel, ValidationError

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_with_retry(extract_fn: Callable[[], BaseModel], schema: type[BaseModel]) -> BaseModel:
    last_error: Optional[Exception] = None
    for attempt in range(1, settings.max_extraction_retries + 2):
        try:
            result = extract_fn()
            schema.model_validate(result.model_dump())
            if attempt > 1:
                logger.info("extraction.recovered_after_retry", attempt=attempt)
            return result
        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning("extraction.validation_failed", attempt=attempt, error=str(e))

    logger.error("extraction.failed_after_retries", attempts=settings.max_extraction_retries + 1)
    raise last_error
```

## File: src/ai/layer3_extraction/prompts/templates/graph_page_ingestion.j2
```jinja
You are an expert Contextual Knowledge Graph Extraction Agent reading Page {{ page_number }} of {{ total_pages }} in a document. The document may belong to ANY domain (e.g. resumes, invoices, contracts, financial statements, medical records, insurance claims, shipping manifests, academic transcripts, government forms, correspondence). Do NOT assume a specific domain — infer it from the content on the page.

================================================================================
TARGET USER SCHEMA (PRIORITY SIGNAL)
================================================================================
The user explicitly wants to extract the following fields from this document:
{% for field in schema_fields %}
- {{ field.name }}{% if field.description %}: {{ field.description }}{% endif %}
{% endfor %}

================================================================================
EXISTING GRAPH MEMORY (FROM PREVIOUS PAGES)
================================================================================
{{ existing_graph_context }}

================================================================================
CURRENT PAGE CONTENT (PAGE {{ page_number }})
================================================================================
{{ page_md }}

================================================================================
INSTRUCTIONS
================================================================================
1. CORE PRINCIPLE:
   - The user schema represents what the user explicitly cares about (Priority Signal). It is the anchor, not the boundary.
   - Do NOT restrict extraction exclusively to schema fields. Capture the surrounding information needed to understand, verify, and connect the schema fields.
   - Categorize every piece of information as follows:
     * CATEGORY A (Explicitly Requested): Information directly corresponding to a schema field. MUST be extracted with is_schema_field=true and schema_field_name set to the matching field.

   DISAMBIGUATING schema_field_name (apply whenever several candidates could match one field):
   - Assign schema_field_name to a value ONLY when it genuinely matches that field's description above. Read the field description carefully, including any "usually labeled ..." hints and any "NOT ..." exclusions, and honor them.
   - When several identifier-shaped values (reference / account / voucher / letter / registration / sample numbers) or several amount-shaped values (requested vs sanctioned vs total vs balance, across columns of a table) could fill the same field, pick the ONE whose on-page label most closely matches the field's description. Prefer a value sitting next to an explicit matching label over one guessed from position or proximity.
   - If no candidate clearly matches the field's description, leave schema_field_name empty (is_schema_field=false) rather than forcing a guess. A missed tag is recoverable downstream; a confident wrong tag is not.
     * CATEGORY B (Contextually Important): Any entity, attribute, value, date, quantity, identifier, event, role, status, or relationship that explains, qualifies, connects, disambiguates, scopes, or provides provenance for the schema information — even if the user did not list it. Extract these with is_schema_field=false.
     * CATEGORY C (Irrelevant): Isolated boilerplate or incidental facts with no bearing on the extraction objective (e.g. page footers, printer marks, generic legal disclaimers, building directions, cafeteria hours). IGNORE them.

   HOW TO DECIDE CATEGORY B (domain-agnostic reasoning):
   Ask, for each candidate, "would this help a careful reader trust, locate, or interpret a schema field?" If yes, extract it. Useful contextual signals that recur across domains include:
     - People & roles: who is named and in what capacity (author, applicant, signer, issuer, recipient, counterparty, approver, dependent, witness, officer, agent).
     - Organizations & places: companies, institutions, departments, branches, jurisdictions, addresses, locations tied to an entity or event.
     - Identifiers: any reference, account, policy, invoice, case, order, employee, tax, registration, or document number that keys a record.
     - Dates & time spans: issue/effective/expiry/start/end/transaction dates and periods that scope a value or event.
     - Quantities & amounts: totals, subtotals, rates, balances, counts, measures, durations, percentages, and their currencies/units.
     - Events & actions: things that happened or were performed, requested, approved, rejected, shipped, filed, or completed.
     - Status & qualifiers: approved/pending/void/active/draft/final flags, conditions, exclusions, or notes that change how a value should be read.
     - Line items & components: individual rows, entries, or sub-records that roll up into a schema field.
   Treat these as GUIDANCE, not a fixed list — extract the equivalent context for whatever domain this document actually is.

2. CROSS-PAGE REFERENCES & ANAPHORA:
   - If this page refers back to an entity introduced earlier using a referring expression (e.g. "the above", "the said", "this applicant", "the aforementioned company", "the same account", "the recipient", "it", "they", "the former/latter"):
     Look at the EXISTING GRAPH MEMORY above and resolve the reference to the specific existing node ID.
     Record this in "reference_resolutions".
     Connect any new values, events, or relationships on this page directly to that resolved existing node rather than creating a new one.

3. ENTITY REUSE & DEDUPLICATION:
   - If an entity on this page is the same real-world thing as one already in the graph — even if written differently (e.g. a full name vs. an initialed name, a company vs. its abbreviation, a role-prefixed name vs. the bare name) — set reused_node_id to that existing node ID instead of creating a duplicate.
   - Only merge when confidence is high and nothing contradicts the match (conflicting identifiers, dates, or clearly different names mean they are DISTINCT). If uncertain or ambiguous, keep them separate and let later resolution decide.

4. EVIDENCE & PROVENANCE:
   - Provide the exact verbatim text snippet from THIS page in "evidence" for every entity and relationship.
   - Status must be ASSERTED (directly stated on the page), INFERRED (logically deduced from the page or graph), or UNCERTAIN (plausible but not confirmed).

NOTES ON THE OUTPUT TYPES BELOW:
   - "type", "label", and "relationship" values below are OPEN vocabularies. The listed options are common examples, not an exhaustive enum. Choose the most accurate, domain-appropriate value; invent a clear PascalCase/UPPER_SNAKE term if none of the examples fit. Prefer a generic type (e.g. "Entity", "Value", "Event") over forcing a wrong specific one.

Respond with ONLY valid JSON in this exact structure:
{
  "entities": [
    {
      "id": "<unique_local_id_e.g._p{{ page_number }}_node_1>",
      "reused_node_id": "<existing_node_id_or_null>",
      "type": "<Person|Organization|Location|Event|Date|Amount|Quantity|Identifier|Document|Status|LineItem|Entity|...>",
      "label": "<short_role_or_meaning_e.g._Applicant|Issuer|Invoice_Number|Effective_Date|Total_Amount>",
      "value": "<extracted_value>",
      "evidence": "<exact_sentence_from_page>",
      "confidence": 0.95,
      "is_schema_field": false,
      "schema_field_name": "<schema_field_name_or_empty>"
    }
  ],
  "relationships": [
    {
      "source_node": "<source_id_or_reused_id>",
      "target_node": "<target_id_or_reused_id>",
      "relationship": "<HAS|BELONGS_TO|RELATED_TO|HAS_VALUE|HAS_DATE|HAS_AMOUNT|ISSUED_BY|ISSUED_TO|PERFORMED_BY|PART_OF|FOR|LOCATED_IN|SAME_AS|REFERS_TO|...>",
      "evidence": "<exact_sentence_from_page>",
      "confidence": 0.95,
      "status": "ASSERTED"
    }
  ],
  "reference_resolutions": [
    {
      "phrase": "<referring_expression_e.g._the_said_applicant>",
      "resolved_to_node_id": "<existing_node_id>",
      "rationale": "<brief_reasoning>"
    }
  ]
}
```

## File: src/ai/layer3_extraction/prompts/templates/page_field_check.j2
```jinja
You are extracting structured data from page {{ page_number }} of {{ total_pages }} in a document.

Target schema fields:
{% for field in schema_fields %}
- {{ field.name }}: {{ field.description }}
{% endfor %}

Page content:
{{ page_md }}

Instructions:
- Go through EVERY schema field listed above one by one and check if this page contains it.
- Only include a field in "matches" if an actual value is present on THIS page.
- Do NOT include fields that are not found. Do NOT return "Not found", "N/A", "null", or placeholders.
- Extract exact values as written.

Respond with ONLY this JSON format:
{
  "matches": [
    {"field": "<field_name>", "value": "<extracted_value>"}
  ]
}

If no fields found on this page: {"matches": []}
```

## File: src/ai/layer3_extraction/ir_store.py
```python
"""
File: ir_store.py
Purpose: Phase 4, commit 1 — per-owner persistence of DocumentIR PageIR objects,
         so downstream chunking/retrieval can carry block_ids + bbox and the
         evidence resolver can run against the real persisted blocks.

TENANT ISOLATION:
  Files are written under the SAME owner-partitioned convention already used by
  llm_metrics (dataset_output/_users/<owner>/...). `owner` is REQUIRED (no default)
  on every public function. doc_id is sanitized with the SAME rule llm_metrics uses
  (alnum + "._-" kept, everything else -> "_", capped length), so a doc_id like
  "../../evil" cannot escape the owner directory. A read with the wrong owner looks
  in that owner's directory only and returns None — it cannot reach another owner's,
  an ownerless, or a `_legacy` file (there is no cross-owner or sentinel path here).

FILE FORMAT (deterministic, UTF-8):
  {
    "schema_version": 1,
    "doc_id": "<original doc_id>",
    "pages": [ <PageIR.model_dump()>, ... ]   # sorted by page_number
  }
  Atomic write: a temp file in the same dir is fully written+flushed, then
  os.replace()'d over the target (atomic on Windows and POSIX).

CONTENT SAFETY: IR files contain document text. This module NEVER logs, prints, or
returns that text in warnings — only counts, ids, categories, and the path.

NO MODEL / NO AWS: pure local disk + json. owner is required; nothing here calls
Bedrock, S3, or any network service.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from src.ai.schemas.document_ir import PageIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Bump when the on-disk shape changes; a file with a different version loads as None.
IR_SCHEMA_VERSION = 1

# Root mirrors the existing owner-partitioned convention (see llm_metrics.write_summary:
# dataset_output/_users/<owner>/metrics/). We use a sibling "ir" subdir.
_DATASET_ROOT = Path("dataset_output")
_USERS_SEGMENT = "_users"


def _safe_owner(owner: str) -> str:
    """Normalize owner the same way the metrics path does: strip + lowercase.
    Raises on empty owner (owner is required, no default)."""
    if not owner or not str(owner).strip():
        raise ValueError("owner is required and cannot be empty")
    return str(owner).strip().lower()


def _safe_doc_id(doc_id: str) -> str:
    """Sanitize a doc_id into a single path-safe filename segment, IDENTICAL to the
    rule in llm_metrics.write_summary. Keeps alnum and '._-', maps everything else
    (including '/', '\\', and '..' separators) to '_', and caps length. This makes
    traversal doc_ids like '../../evil' collapse to a harmless flat name."""
    if doc_id is None or not str(doc_id).strip():
        raise ValueError("doc_id is required and cannot be empty")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(doc_id))[:120]


def _ir_dir(owner: str, output_root: Optional[Path] = None) -> Path:
    root = output_root or (_DATASET_ROOT / _USERS_SEGMENT / _safe_owner(owner) / "ir")
    return root


def ir_path(owner: str, doc_id: str, output_root: Optional[Path] = None) -> Path:
    """Return the owner-scoped path for a document's IR file. owner required."""
    # _safe_owner is applied inside _ir_dir when output_root is None; apply here too
    # so an explicit output_root still gets a sanitized, flat doc filename.
    safe_doc = _safe_doc_id(doc_id)
    base = output_root if output_root is not None else _ir_dir(owner)
    return Path(base) / f"{safe_doc}.ir.json"


def save_page_irs(
    owner: str,
    doc_id: str,
    pages: List[PageIR],
    output_root: Optional[Path] = None,
) -> Path:
    """Persist a document's PageIR list under the owner partition. owner REQUIRED.

    Deterministic serialization (pages sorted by page_number, json indent=2,
    sort_keys), atomic write (temp file -> os.replace). Returns the path written.
    Never logs document text."""
    _safe_owner(owner)  # validate (raises on empty)
    path = ir_path(owner, doc_id, output_root=output_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(pages, key=lambda p: p.page_number)
    payload = {
        "schema_version": IR_SCHEMA_VERSION,
        "doc_id": str(doc_id),
        "pages": [p.model_dump() for p in ordered],
    }
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    # Atomic write: temp file in the SAME directory, then os.replace.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".ir_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        raise

    logger.info(
        "ir_store.saved",
        owner=_safe_owner(owner),
        doc_id=_safe_doc_id(doc_id),
        pages=len(ordered),
        blocks=sum(len(p.blocks) for p in ordered),
        path=str(path),
    )
    return path


def load_page_irs(
    owner: str,
    doc_id: str,
    output_root: Optional[Path] = None,
) -> Optional[List[PageIR]]:
    """Load a document's PageIR list for `owner`. Returns None (never raises) when:
      - the file is missing,
      - the file is corrupted / not valid JSON,
      - the schema_version doesn't match IR_SCHEMA_VERSION,
      - the payload shape is invalid.
    Warnings are metadata-only (owner, doc_id, reason) — never the file content."""
    _safe_owner(owner)
    path = ir_path(owner, doc_id, output_root=output_root)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        logger.warning("ir_store.load_corrupted",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="unreadable_or_bad_json")
        return None
    if not isinstance(payload, dict):
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="not_an_object")
        return None
    if payload.get("schema_version") != IR_SCHEMA_VERSION:
        logger.warning("ir_store.load_version_mismatch",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id),
                       reason="schema_version_mismatch", found=payload.get("schema_version"))
        return None
    pages_raw = payload.get("pages")
    if not isinstance(pages_raw, list):
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="pages_not_a_list")
        return None
    try:
        return [PageIR.model_validate(p) for p in pages_raw]
    except Exception:
        logger.warning("ir_store.load_invalid",
                       owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), reason="pageir_validation_failed")
        return None


def delete_page_irs(
    owner: str,
    doc_id: str,
    output_root: Optional[Path] = None,
) -> bool:
    """Delete a document's owner-scoped IR file. owner REQUIRED (no default).

    Returns True if a file was removed, False if there was nothing to remove.
    Never raises on a missing file; a removal error is logged (metadata only)
    and re-raised so a caller can decide — delete endpoints wrap this so a
    failure doesn't break the overall document delete."""
    _safe_owner(owner)
    path = ir_path(owner, doc_id, output_root=output_root)
    if not path.exists():
        return False
    os.remove(str(path))
    logger.info("ir_store.deleted", owner=_safe_owner(owner), doc_id=_safe_doc_id(doc_id), path=str(path))
    return True
```

## File: src/ai/layer3_extraction/scratchpad.py
```python
from typing import Any, Dict, List, Optional


class Scratchpad:
    """
    Multi-page extraction scratchpad.
    Tracks all field candidates found page-by-page across the entire document.
    """
    def __init__(self, schema_field_names: List[str]) -> None:
        self._all_fields: List[str] = list(schema_field_names)
        # field -> list of {"value": val, "page": page_num}
        self._candidates: Dict[str, List[Dict[str, Any]]] = {f: [] for f in self._all_fields}
        # field -> first found hit {"value": val, "page": page_num}
        self._hits: Dict[str, Dict[str, Any]] = {}

    def update(self, page_number: int, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Record matches found on a page.
        Stores every hit in _candidates, and sets the first-seen in _hits.
        """
        updated = []
        negative_patterns = {"not found", "not mentioned", "n/a", "none", "not available", "null", "nil"}
        for m in matches:
            field = m.get("field", "")
            value = m.get("value", "")
            if field not in self._all_fields or value in ("", None):
                continue

            # Skip negative answers like "Not found on this page"
            val_clean = str(value).strip().lower()
            if any(neg in val_clean for neg in negative_patterns) and len(val_clean) < 40:
                continue

            candidate = {"value": value, "page": page_number}
            self._candidates[field].append(candidate)

            if field not in self._hits:
                self._hits[field] = candidate
                updated.append(field)
            else:
                updated.append(field)

        return updated

    @property
    def all_fields(self) -> List[str]:
        return list(self._all_fields)

    @property
    def missing_fields(self) -> List[str]:
        return [f for f in self._all_fields if not self._candidates[f]]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0

    def get_candidates(self, field: str) -> List[Dict[str, Any]]:
        return self._candidates.get(field, [])

    def to_values_dict(self) -> Dict[str, Any]:
        """First-seen candidate value per field."""
        return {name: hit["value"] for name, hit in self._hits.items()}

    def provenance(self) -> Dict[str, int]:
        """Returns {field_name: page_number} for first-seen hits."""
        return {name: hit["page"] for name, hit in self._hits.items()}

    def snapshot(self) -> List[Dict[str, Any]]:
        """Flat list of found fields with page and value."""
        return [
            {"field": name, "value": hit["value"], "page": hit["page"]}
            for name, hit in self._hits.items()
        ]

    def format_evidence_for_llm(self) -> str:
        """
        Format all collected page-by-page candidates into a readable context
        to feed to the LLM for final extraction.
        """
        lines = ["Here are the candidate values found across document pages:"]
        for field in self._all_fields:
            cands = self._candidates.get(field, [])
            if not cands:
                lines.append(f"- {field}: Not found on any scanned page")
            else:
                formatted_cands = [f'"{c["value"]}" (Page {c["page"]})' for c in cands]
                lines.append(f"- {field}: {', '.join(formatted_cands)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        found_count = len(self._all_fields) - len(self.missing_fields)
        return f"Scratchpad({found_count}/{len(self._all_fields)} found, missing={self.missing_fields})"
```

## File: src/ai/layer3_extraction/graph_agent/agent.py
```python
"""
File: agent.py
Purpose: Graph Extraction Agent processing pages sequentially and maintaining contextual graph memory.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.graph_agent.evidence_resolver import resolve_evidence_to_blocks
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.memory_manager import PageDelta, UnresolvedRef
from src.ai.layer3_extraction.graph_agent.models import (
    Evidence,
    GraphEdge,
    GraphNode,
)
from src.ai.schemas.document_ir import Block
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_evidence_list(
    page_number: int,
    ent_evidence: str,
    ent_conf: float,
    blocks: Optional[List[Block]] = None,
) -> List[Evidence]:
    if not ent_evidence:
        return []
    block_ids, bbox = [], None
    if blocks:
        block_ids, bbox = resolve_evidence_to_blocks(ent_evidence, blocks)
    return [
        Evidence(
            page_number=page_number,
            text=ent_evidence,
            confidence=ent_conf,
            block_ids=block_ids,
            bbox=bbox,
        )
    ]


class GraphExtractionAgent:
    """
    Graph Extraction Agent.
    Considers the schema as a priority signal, but also captures contextually
    important entities, resolves cross-page references, and maintains traceable evidence.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        graph: GraphMemory,
        llm: ExtractionLLMClient,
    ) -> None:
        self.schema = schema
        self.graph = graph
        self.llm = llm

        # Cache target schema fields once
        self.schema_fields: List[Dict[str, str]] = [
            {"name": k, "description": v.description or k}
            for k, v in schema.model_fields.items()
        ]
        self.schema_field_names: Set[str] = set(schema.model_fields.keys())
        self.llm_call_count = 0
        self.session_id_map: Dict[str, str] = {}

    async def build_page_delta_async(
        self,
        page_number: int,
        markdown: str,
        total_pages: int,
        context: List[Dict[str, Any]],
        blocks: Optional[List[Block]] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> PageDelta:
        """
        Extract page graph asynchronously into an isolated PageDelta.
        STATELESS with respect to self.graph: MUST NOT mutate self.graph.
        Calls the synchronous LLM in a worker thread: the given executor when
        provided, otherwise asyncio.to_thread() (loop default executor).
        """
        clean_md = (markdown or "").strip()
        if not clean_md:
            logger.debug("graph.page_async.skip_empty", page_number=page_number)
            return PageDelta(page_number=page_number)

        logger.info(
            "graph.page_async.process_start",
            page_number=page_number,
            total_pages=total_pages,
            schema_fields=list(self.schema_field_names),
            context_nodes_count=len(context),
        )
        self.llm_call_count += 1
        if executor is None:
            llm_output = await asyncio.to_thread(
                self.llm.extract_graph_from_page,
                page_md=clean_md,
                schema_fields=self.schema_fields,
                existing_nodes=context,
                page_number=page_number,
                total_pages=total_pages,
            )
        else:
            # copy_context().run keeps the logger's correlation_id visible in the
            # worker thread, as asyncio.to_thread() does.
            call = functools.partial(
                contextvars.copy_context().run,
                self.llm.extract_graph_from_page,
                page_md=clean_md,
                schema_fields=self.schema_fields,
                existing_nodes=context,
                page_number=page_number,
                total_pages=total_pages,
            )
            llm_output = await asyncio.get_running_loop().run_in_executor(executor, call)

        entities = llm_output.get("entities", [])
        relationships = llm_output.get("relationships", [])
        ref_resolutions = llm_output.get("reference_resolutions", [])
        # Audit #1 (2b/2c): propagate extraction quality (clean/partial/degraded)
        # emitted by the client so the orchestrator can checkpoint non-clean pages
        # as something other than COMPLETED.
        page_quality = str(llm_output.get("quality", "clean"))

        delta_entities: List[GraphNode] = []
        delta_edges: List[GraphEdge] = []
        unresolved: List[UnresolvedRef] = []
        raw_id_map: Dict[str, str] = {}

        # 1. Parse Entities
        for idx, ent in enumerate(entities):
            raw_id = ent.get("id") or f"p{page_number}_e{idx + 1}"
            reused_id = ent.get("reused_node_id")
            ent_type = ent.get("type") or "Entity"
            ent_label = ent.get("label") or ent_type
            ent_value = str(ent.get("value") or "").strip()
            ent_evidence = str(ent.get("evidence") or "")
            ent_conf = float(ent.get("confidence", 1.0))
            is_schema = bool(ent.get("is_schema_field", False))
            schema_fname = ent.get("schema_field_name") or ""

            if not ent_value:
                continue

            if schema_fname in self.schema_field_names:
                is_schema = True

            category = "EXPLICIT" if is_schema else "CONTEXTUAL"

            ev_list = _build_evidence_list(page_number, ent_evidence, ent_conf, blocks)
            props: Dict[str, Any] = {"raw_id": raw_id}
            if schema_fname:
                props["schema_field_name"] = schema_fname
            if reused_id:
                props["reused_node_id"] = reused_id

            node = GraphNode(
                id=raw_id,
                type=ent_type,
                label=ent_label,
                value=ent_value,
                properties=props,
                aliases=list(ent.get("aliases", [])),
                evidence=ev_list,
                source_pages=[page_number],
                confidence=ent_conf,
                status="ASSERTED",
                category=category,
            )
            delta_entities.append(node)
            raw_id_map[raw_id] = raw_id
            if ent_value:
                raw_id_map[ent_value.lower()] = raw_id
            if ent_label:
                raw_id_map[ent_label.lower()] = raw_id

        # 2. Parse Relationships
        for edge_data in relationships:
            raw_src = str(edge_data.get("source_node", "")).strip()
            raw_tgt = str(edge_data.get("target_node", "")).strip()
            rel = str(edge_data.get("relationship", "RELATED_TO")).strip().upper() or "RELATED_TO"
            ev_text = str(edge_data.get("evidence", ""))
            edge_conf = float(edge_data.get("confidence", 1.0))
            edge_status = str(edge_data.get("status", "ASSERTED")).upper()
            if edge_status not in ("ASSERTED", "INFERRED", "UNCERTAIN"):
                edge_status = "ASSERTED"

            if raw_src and raw_tgt:
                edge_id = f"edge_{page_number}_{uuid.uuid4().hex[:6]}"
                graph_edge = GraphEdge(
                    id=edge_id,
                    source_node=raw_src,
                    target_node=raw_tgt,
                    relationship=rel,
                    evidence=ev_text,
                    source_page=page_number,
                    confidence=edge_conf,
                    status=edge_status,
                )
                delta_edges.append(graph_edge)

        # 3. Parse Reference Resolutions
        for ref in ref_resolutions:
            phrase = str(ref.get("phrase", "")).strip()
            target_id = str(ref.get("resolved_to_node_id", "")).strip()
            target_type = str(ref.get("target_type") or ref.get("type") or "").strip()
            rationale = str(ref.get("rationale", "")).strip()

            if not phrase and not target_id:
                continue

            # Check if reference is resolved locally to an entity in this delta
            resolved_local = (
                raw_id_map.get(target_id)
                or raw_id_map.get(target_id.lower())
                if target_id
                else None
            )

            # Check if reference points to a node in the initial context
            resolved_context: Optional[str] = (
                next((str(c["id"]) for c in context if c.get("id") == target_id), None)
                if target_id
                else None
            )

            canon_target = resolved_local or resolved_context
            if canon_target is not None:
                src_node_id = (
                    raw_id_map.get(phrase.lower())
                    or raw_id_map.get(phrase)
                    or f"ref_src_{page_number}_{uuid.uuid4().hex[:4]}"
                )
                delta_edges.append(
                    GraphEdge(
                        id=f"edge_ref_{page_number}_{uuid.uuid4().hex[:6]}",
                        source_node=src_node_id,
                        target_node=canon_target,
                        relationship="REFERS_TO",
                        evidence=rationale or phrase,
                        source_page=page_number,
                        confidence=0.9,
                        status="INFERRED",
                    )
                )
            else:
                unresolved_src = (
                    raw_id_map.get(phrase.lower())
                    or raw_id_map.get(phrase)
                    or phrase
                )
                if not target_type:
                    words = phrase.split()
                    target_type = words[-1].capitalize() if words else "Entity"

                unresolved.append(
                    UnresolvedRef(
                        source_node_id=unresolved_src,
                        phrase=phrase,
                        target_type=target_type,
                        page_number=page_number,
                        rationale=rationale,
                    )
                )

        return PageDelta(
            page_number=page_number,
            entities=delta_entities,
            relationships=delta_edges,
            unresolved_references=unresolved,
            raw_id_map=raw_id_map,
            quality=page_quality,
        )
```

## File: src/ai/layer3_extraction/graph_agent/graph_memory.py
```python
"""
File: graph_memory.py
Purpose: In-memory dynamic graph memory engine using NetworkX.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

from src.ai.layer3_extraction.graph_agent.models import Evidence, GraphEdge, GraphNode
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphMemory:
    """
    In-memory graph store holding document entities, facts, and relationships.
    Survives for the entire multi-page document processing session.
    Uses NetworkX MultiDiGraph to preserve multiple distinct relationships
    between the same source and target nodes without overwriting.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        # Inverted index for quick entity reuse and lookup
        self._type_index: Dict[str, Set[str]] = {}
        self._alias_index: Dict[str, Set[str]] = {}  # normalized value -> set of node_ids

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @property
    def nodes(self) -> Dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> Dict[str, GraphEdge]:
        return self._edges

    def lookup_alias(self, alias: str) -> Optional[GraphNode]:
        """Resolve an alias or reference mention to an existing GraphNode."""
        alias_norm = self._normalize(alias)
        if alias_norm in self._alias_index:
            node_ids = self._alias_index[alias_norm]
            if node_ids:
                # sorted() so the resolved node is set-iteration-order independent.
                return self.get_node(sorted(node_ids)[0])

        # Fallback check across node aliases in case alias was added directly to GraphNode
        for node in self._nodes.values():
            for a in getattr(node, "aliases", []):
                if self._normalize(a) == alias_norm:
                    if alias_norm not in self._alias_index:
                        self._alias_index[alias_norm] = set()
                    self._alias_index[alias_norm].add(node.id)
                    return node
        return None

    def add_alias(self, node_id: str, alias: str) -> None:
        """Register an alias for a node and update index."""
        node = self.get_node(node_id)
        if node:
            node.add_alias(alias)
            alias_key = self._normalize(alias)
            if alias_key:
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(node_id)

    def create_node(
        self,
        node_or_id: Union[GraphNode, str, None] = None,
        node_type: Optional[str] = None,
        label: Optional[str] = None,
        value: Optional[str] = None,
        *,
        node_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        evidence: Optional[Union[Evidence, List[Evidence]]] = None,
        source_page: Optional[int] = None,
        source_pages: Optional[List[int]] = None,
        confidence: float = 1.0,
        status: str = "EXTRACTED",
        category: str = "CONTEXTUAL",
        **kwargs: Any,
    ) -> GraphNode:
        """Add a new node to the graph and update inverted indexes."""
        if isinstance(node_or_id, GraphNode):
            node = node_or_id
        else:
            actual_id = node_id or (node_or_id if isinstance(node_or_id, str) else "")
            if not actual_id:
                raise ValueError("Node must have a non-empty id.")
            pages = list(source_pages or [])
            if source_page is not None and source_page not in pages:
                pages.append(source_page)
            ev_list: List[Evidence] = []
            if evidence:
                if isinstance(evidence, list):
                    ev_list.extend(evidence)
                else:
                    ev_list.append(evidence)
            node = GraphNode(
                id=actual_id,
                type=node_type or kwargs.get("type", "Entity"),
                label=label or "",
                value=value or "",
                properties=properties or {},
                aliases=aliases or [],
                evidence=ev_list,
                source_pages=pages or [1],
                confidence=confidence,
                status=status,
                category=category,
            )

        if not node.id:
            raise ValueError("Node must have a non-empty id.")
        if node.id in self._nodes:
            logger.debug("graph.node_already_exists_updating", node_id=node.id)
            return self.update_node(node.id, node.to_dict())

        self._nodes[node.id] = node
        self.graph.add_node(node.id, data=node)

        # Update type index
        type_key = self._normalize(node.type)
        if type_key not in self._type_index:
            self._type_index[type_key] = set()
        self._type_index[type_key].add(node.id)

        # Update alias/value index for primary value
        val_key = self._normalize(node.value)
        if val_key:
            if val_key not in self._alias_index:
                self._alias_index[val_key] = set()
            self._alias_index[val_key].add(node.id)

        # Update alias/value index for known aliases
        for alias in getattr(node, "aliases", []):
            alias_key = self._normalize(alias)
            if alias_key:
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(node.id)

        logger.debug(
            "graph.node.created",
            node_id=node.id,
            node_type=node.type,
            value=node.value,
            category=node.category,
        )
        return node

    def update_node(
        self,
        node_id: str,
        updates: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GraphNode:
        """Update properties, evidence, or pages on an existing node."""
        node = self.get_node(node_id)
        if not node:
            raise KeyError(f"Node {node_id} does not exist in graph memory.")

        upd = dict(updates or {})
        upd.update(kwargs)

        if "value" in upd and upd["value"]:
            old_val_key = self._normalize(node.value)
            new_val_key = self._normalize(upd["value"])
            if old_val_key in self._alias_index and node_id in self._alias_index[old_val_key]:
                self._alias_index[old_val_key].remove(node_id)
            node.value = str(upd["value"])
            if new_val_key not in self._alias_index:
                self._alias_index[new_val_key] = set()
            self._alias_index[new_val_key].add(node_id)

        if "label" in upd and upd["label"]:
            node.label = str(upd["label"])

        if "confidence" in upd and upd["confidence"] is not None:
            node.confidence = max(node.confidence, float(upd["confidence"]))

        if "status" in upd and upd["status"]:
            node.status = str(upd["status"])

        if "source_pages" in upd:
            for p in upd["source_pages"]:
                node.add_page(int(p))
        if "source_page" in upd and upd["source_page"] is not None:
            node.add_page(int(upd["source_page"]))

        if "evidence" in upd:
            for ev in upd["evidence"]:
                if isinstance(ev, dict):
                    node.add_evidence(
                        page_number=int(ev.get("page_number", 1)),
                        text=str(ev.get("text", "")),
                        confidence=float(ev.get("confidence", 1.0)),
                        block_ids=list(ev.get("block_ids", [])) if ev.get("block_ids") else [],
                        bbox=ev.get("bbox"),
                    )
                elif isinstance(ev, Evidence):
                    node.add_evidence(
                        page_number=ev.page_number,
                        text=ev.text,
                        confidence=ev.confidence,
                        block_ids=ev.block_ids,
                        bbox=ev.bbox,
                    )
                else:
                    node.add_evidence(page_number=ev.page_number, text=ev.text, confidence=ev.confidence)

        if "properties" in upd and isinstance(upd["properties"], dict):
            node.properties.update(upd["properties"])

        if "aliases" in upd and upd["aliases"]:
            for a in upd["aliases"]:
                node.add_alias(a)
                a_key = self._normalize(a)
                if a_key:
                    if a_key not in self._alias_index:
                        self._alias_index[a_key] = set()
                    self._alias_index[a_key].add(node_id)

        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve node by ID."""
        return self._nodes.get(node_id)

    def create_edge(
        self,
        edge_or_id: Union[GraphEdge, str, None] = None,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relationship: Optional[str] = None,
        *,
        edge_id: Optional[str] = None,
        evidence: Optional[str] = None,
        source_page: int = 1,
        confidence: float = 1.0,
        status: str = "EXTRACTED",
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GraphEdge:
        """Add a directed relationship between two nodes in the graph."""
        if isinstance(edge_or_id, GraphEdge):
            edge = edge_or_id
        else:
            actual_id = edge_id or (edge_or_id if isinstance(edge_or_id, str) else "")
            src = source_node or kwargs.get("source", "")
            tgt = target_node or kwargs.get("target", "")
            rel = relationship or kwargs.get("rel", "RELATED_TO")
            edge = GraphEdge(
                id=actual_id,
                source_node=src,
                target_node=tgt,
                relationship=rel,
                evidence=evidence or kwargs.get("text", ""),
                source_page=source_page,
                confidence=confidence,
                status=status,
                properties=properties or {},
            )

        if edge.source_node not in self._nodes:
            raise KeyError(f"Source node {edge.source_node} does not exist in graph memory.")
        if edge.target_node not in self._nodes:
            raise KeyError(f"Target node {edge.target_node} does not exist in graph memory.")

        if not edge.id:
            edge.id = f"{edge.source_node}__{edge.relationship.lower()}__{edge.target_node}"

        self._edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_node,
            edge.target_node,
            key=edge.id,
            id=edge.id,
            relationship=edge.relationship,
            data=edge,
        )

        logger.debug(
            "graph.edge.created",
            edge_id=edge.id,
            source=edge.source_node,
            rel=edge.relationship,
            target=edge.target_node,
            status=edge.status,
        )
        return edge

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    def get_edges(
        self,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> List[GraphEdge]:
        """Filter edges by source, target, and/or relationship type."""
        results = []
        rel_norm = self._normalize(relationship) if relationship else None
        for edge in self._edges.values():
            if source_node and edge.source_node != source_node:
                continue
            if target_node and edge.target_node != target_node:
                continue
            if rel_norm and self._normalize(edge.relationship) != rel_norm:
                continue
            results.append(edge)
        return results

    def read_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[GraphEdge, GraphNode]]:
        """
        Return incoming, outgoing, or both neighbor edges and adjacent nodes.
        direction: 'out', 'in', or 'both'
        """
        if node_id not in self._nodes:
            return []

        neighbors = []
        if direction in ("out", "both"):
            for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                edge: Optional[GraphEdge] = edge_data.get("data")
                target_node = self._nodes.get(target_id)
                if edge and target_node:
                    neighbors.append((edge, target_node))

        if direction in ("in", "both"):
            for source_id, _, edge_data in self.graph.in_edges(node_id, data=True):
                edge: Optional[GraphEdge] = edge_data.get("data")
                source_node = self._nodes.get(source_id)
                if edge and source_node:
                    neighbors.append((edge, source_node))

        return neighbors

    def search_nodes(
        self,
        query: Optional[str] = None,
        node_type: Optional[str] = None,
        label: Optional[str] = None,
        page: Optional[int] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """
        Search for entities by query (fuzzy substring in value/label), node_type, label, or page.
        """
        candidates: List[Tuple[float, GraphNode]] = []
        query_norm = self._normalize(query) if query else None
        type_norm = self._normalize(node_type) if node_type else None
        label_norm = self._normalize(label) if label else None

        for node in self._nodes.values():
            if page is not None and page not in node.source_pages:
                continue

            if type_norm and self._normalize(node.type) != type_norm:
                continue

            if label_norm and self._normalize(node.label) != label_norm:
                continue

            score = 0.0
            if query_norm:
                val_norm = self._normalize(node.value)
                lbl_norm = self._normalize(node.label)
                if query_norm == val_norm:
                    score = 1.0
                elif query_norm in val_norm:
                    score = 0.8
                elif val_norm in query_norm and len(val_norm) > 2:
                    score = 0.7
                elif query_norm in lbl_norm or lbl_norm in query_norm:
                    score = 0.5
                else:
                    continue
            else:
                score = 0.5

            candidates.append((score, node))

        candidates.sort(key=lambda x: (x[0], x[1].confidence, max(x[1].source_pages or [0])), reverse=True)
        return [node for _, node in candidates[:limit]]

    def find_reuse_candidate(
        self,
        node_type: str,
        value: str,
        label: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ) -> Optional[GraphNode]:
        """
        High-confidence check for entity reuse to avoid duplicate nodes,
        without aggressive or false merging.
        """
        val_norm = self._normalize(value)
        if not val_norm:
            return None

        # 1. Exact normalized value match in same or compatible type.
        #    Iterate the id set in SORTED order so the chosen reuse candidate does not
        #    depend on set-iteration (PYTHONHASHSEED) order (cross-process determinism).
        if val_norm in self._alias_index:
            for cand_id in sorted(self._alias_index[val_norm]):
                cand = self.get_node(cand_id)
                if cand and (self._normalize(cand.type) == self._normalize(node_type) or not node_type):
                    return cand

        # 2. Person name partial/honorific matching (e.g. "Dr. Arun Sharma" vs "Arun Sharma")
        if self._normalize(node_type) in ("person", "doctor", "surgeon", "patient"):
            clean_name = re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?)\s+", "", val_norm)
            for node in self._nodes.values():
                if self._normalize(node.type) in ("person", "doctor", "surgeon", "patient"):
                    cand_clean = re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?)\s+", "", self._normalize(node.value))
                    if clean_name == cand_clean and len(clean_name) > 3:
                        return node

        return None

    def get_bounded_summary(self, schema_fields: Optional[List[str]] = None, max_nodes: int = 40) -> str:
        """
        Generate a compact, readable summary of existing graph memory for context
        in agent prompts without overwhelming token limits.
        """
        if not self._nodes:
            return "No entities in graph memory yet."

        lines = ["Existing Graph Entities:"]
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: (1 if n.category == "EXPLICIT" else 0, max(n.source_pages or [0])),
            reverse=True,
        )

        for n in sorted_nodes[:max_nodes]:
            pages_str = f"P{','.join(map(str, n.source_pages))}" if n.source_pages else "P?"
            lines.append(f"- [{n.id}] {n.type} ({n.label}): \"{n.value}\" [{pages_str}]")

        if self._edges:
            lines.append("\nKey Relationships:")
            edge_count = 0
            for edge in list(self._edges.values())[-25:]:  # show most recent 25 edges
                src = self.get_node(edge.source_node)
                tgt = self.get_node(edge.target_node)
                if src and tgt:
                    src_val = src.value or src.id
                    tgt_val = tgt.value or tgt.id
                    lines.append(f"- ({src_val}) -[{edge.relationship}]-> ({tgt_val}) [P{edge.source_page}]")
                    edge_count += 1

        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of graph memory."""
        type_counts: Dict[str, int] = {}
        for n in self._nodes.values():
            type_counts[n.type] = type_counts.get(n.type, 0) + 1

        rel_counts: Dict[str, int] = {}
        for e in self._edges.values():
            rel_counts[e.relationship] = rel_counts.get(e.relationship, 0) + 1

        all_pages = set()
        for n in self._nodes.values():
            all_pages.update(n.source_pages)

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "pages_covered": sorted(list(all_pages)),
            "node_types": type_counts,
            "relationships": rel_counts,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Full serializable snapshot of the graph."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "stats": self.stats(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GraphMemory into a lossless serializable dictionary."""
        return self.snapshot()

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore graph state from snapshot dictionary."""
        restored = self.from_dict(snapshot)
        self.graph = restored.graph
        self._nodes = restored._nodes
        self._edges = restored._edges
        self._type_index = restored._type_index
        self._alias_index = restored._alias_index

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphMemory:
        """Losslessly reconstruct a GraphMemory instance from a serialized dictionary."""
        gm = cls()
        for ndata in data.get("nodes", []):
            node = GraphNode.from_dict(ndata)
            gm.create_node(node)
        for edata in data.get("edges", []):
            edge = GraphEdge.from_dict(edata)
            try:
                gm.create_edge(edge)
            except Exception as exc:
                logger.debug("graph.from_dict.edge_failed", error=str(exc), edge_id=edata.get("id"))
        return gm
```

## File: src/ai/layer3_extraction/prompts/configs/prompt_config.yaml
```yaml
extraction:
  template: extraction_system.j2
  temperature: 0.0
  max_tokens: 4000

page_field_check:
  template: page_field_check.j2
  temperature: 0.0
  max_tokens: 4000

page_summary:
  template: page_summary.j2
  temperature: 0.0
  max_tokens: 500

navigation:
  template: navigation.j2
  temperature: 0.0
  max_tokens: 1000

graph_page_ingestion:
  template: graph_page_ingestion.j2
  temperature: 0.0
  max_tokens: 4000

graph_schema_resolution:
  template: graph_schema_resolution.j2
  temperature: 0.0
  max_tokens: 4000
```

## File: src/ai/layer3_extraction/page_loader.py
```python
"""
File: page_loader.py
Purpose: Convert Layer 1/2 PageOutput objects into the list[dict] contract
         that Layer 3's extractor expects.

Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-28
"""
import re
from pathlib import Path
from typing import Dict, List

from src.ai.schemas.page import PageOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s*\|[^>]*-->", re.IGNORECASE)
PAGE_CLOSING_MARKER = re.compile(r"<!--\s*/PAGE(?:\s+\d+)?\s*-->", re.IGNORECASE)


def load_pages_with_confidence(page_outputs: List[PageOutput]) -> List[Dict]:
    """Real pipeline path — kept for API use."""
    sorted_outputs = sorted(page_outputs, key=lambda p: p.page_number)
    low_conf_pages = [p.page_number for p in sorted_outputs if p.low_confidence]
    if low_conf_pages:
        logger.warning("page_loader.low_confidence_pages_included",
                       pages=low_conf_pages, total_pages=len(sorted_outputs))
    return [
        {"markdown": p.markdown, "page_number": p.page_number, "blocks": getattr(p, "blocks", []) or []}
        for p in sorted_outputs
    ]


def load_pages_from_outputs(page_outputs: List[PageOutput]) -> List[str]:
    """Backward-compatible: returns plain list[str]."""
    return [p["markdown"] for p in load_pages_with_confidence(page_outputs)]


def load_pages_from_fixture(doc_id: str) -> List[Dict]:
    """TESTING ONLY. Reads fixture .md file and returns list[dict]."""
    file_path = FIXTURES_DIR / f"{doc_id}.md"
    if not file_path.exists():
        raise FileNotFoundError(f"No fixture found for doc_id={doc_id!r} at {file_path}")

    content = file_path.read_text(encoding="utf-8")
    matches = list(PAGE_MARKER.finditer(content))
    if not matches:
        return [{"markdown": content.strip(), "page_number": 1, "blocks": []}]

    pages = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        raw_slice = content[start:end].strip()
        cleaned_markdown = PAGE_CLOSING_MARKER.split(raw_slice)[0].strip()
        pages.append({
            "markdown":    cleaned_markdown,
            "page_number": int(match.group(1)),
            "blocks":      [],
        })
    return pages


def load_pages(doc_id: str) -> List[Dict]:
    return load_pages_from_fixture(doc_id)
```

## File: src/ai/layer3_extraction/graph_agent/models.py
```python
"""
File: models.py
Purpose: Data structures for Graph-Based Contextual Extraction in Layer 3.
Hardened with explicit uncertainty models, candidate evaluation structures,
merge audit records, page validation, and serialization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ResolutionStatus(str, Enum):
    CONFIRMED_MATCH = "CONFIRMED_MATCH"
    UNCERTAIN_MATCH = "UNCERTAIN_MATCH"
    NEW_ENTITY = "NEW_ENTITY"
    REJECTED_CONTRADICTION = "REJECTED_CONTRADICTION"


class TypeCompatibility(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class PageProcessingState(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DELTA_CREATED = "DELTA_CREATED"
    DELTA_PERSISTED = "DELTA_PERSISTED"
    MERGED = "MERGED"
    RECONCILED = "RECONCILED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Evidence:
    page_number: int
    text: str
    confidence: float = 1.0
    block_ids: List[str] = field(default_factory=list)
    bbox: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "block_ids": self.block_ids,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        return cls(
            page_number=int(data.get("page_number", 1)),
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 1.0)),
            block_ids=data.get("block_ids", []) or [],
            bbox=data.get("bbox"),
        )


NodeEvidence = Evidence


@dataclass
class GraphNode:
    id: str
    type: str  # Generic type: Person, Organization, Event, Date, Amount, Identifier, Admission, etc.
    label: str
    value: str
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN, EXTRACTED
    category: str = "CONTEXTUAL"  # EXPLICIT, CONTEXTUAL, REFERENCE_SUPPORT
    # Hardened uncertainty dimensions:
    extraction_confidence: float = 1.0  # text extraction certainty from source document
    resolution_confidence: Optional[float] = None  # canonical identity certainty (None for NEW_ENTITY or unestablished resolution)
    evidence_strength: float = 1.0      # supporting textual evidence strength

    def add_page(self, page_number: int) -> None:
        if page_number not in self.source_pages:
            self.source_pages.append(page_number)
            self.source_pages.sort()

    def add_evidence(
        self,
        page_number: int,
        text: str,
        confidence: float = 1.0,
        block_ids: Optional[List[str]] = None,
        bbox: Optional[dict] = None,
    ) -> None:
        self.add_page(page_number)
        if text and not any(e.text == text and e.page_number == page_number for e in self.evidence):
            self.evidence.append(
                Evidence(
                    page_number=page_number,
                    text=text,
                    confidence=confidence,
                    block_ids=block_ids or [],
                    bbox=bbox,
                )
            )

    def add_alias(self, alias: str) -> None:
        clean = alias.strip()
        if clean and clean not in self.aliases:
            self.aliases.append(clean)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "value": self.value,
            "properties": self.properties,
            "aliases": list(self.aliases),
            "evidence": [e.to_dict() for e in self.evidence],
            "source_pages": self.source_pages,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "category": self.category,
            "extraction_confidence": round(self.extraction_confidence, 3),
            "resolution_confidence": round(self.resolution_confidence, 3) if self.resolution_confidence is not None else None,
            "evidence_strength": round(self.evidence_strength, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphNode:
        ev_list = [Evidence.from_dict(e) if isinstance(e, dict) else e for e in data.get("evidence", [])]
        conf = float(data.get("confidence", 1.0))
        res_conf = data.get("resolution_confidence")
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "Entity")),
            label=str(data.get("label", data.get("type", "Entity"))),
            value=str(data.get("value", "")),
            properties=data.get("properties", {}),
            aliases=list(data.get("aliases", [])),
            evidence=ev_list,
            source_pages=list(data.get("source_pages", [])),
            confidence=conf,
            status=str(data.get("status", "ASSERTED")),
            category=str(data.get("category", "CONTEXTUAL")),
            extraction_confidence=float(data.get("extraction_confidence", conf)),
            resolution_confidence=float(res_conf) if res_conf is not None else None,
            evidence_strength=float(data.get("evidence_strength", 1.0)),
        )


@dataclass
class GraphEdge:
    id: str
    source_node: str
    target_node: str
    relationship: str  # Generic: HAS, BELONGS_TO, RELATED_TO, PERFORMED_BY, UNDERWENT, FOR, etc.
    evidence: str = ""
    source_page: int = 1
    confidence: float = 1.0
    status: str = "ASSERTED"  # ASSERTED, INFERRED, UNCERTAIN, EXTRACTED
    properties: Dict[str, Any] = field(default_factory=dict)
    # Hardened uncertainty dimensions:
    extraction_confidence: float = 1.0    # certainty of mention extraction
    relationship_confidence: float = 1.0  # support for the semantic link between nodes
    evidence_strength: float = 1.0        # strength of textual evidence for this relation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relationship": self.relationship,
            "evidence": self.evidence,
            "source_page": self.source_page,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "properties": self.properties,
            "extraction_confidence": round(self.extraction_confidence, 3),
            "relationship_confidence": round(self.relationship_confidence, 3),
            "evidence_strength": round(self.evidence_strength, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphEdge:
        conf = float(data.get("confidence", 1.0))
        return cls(
            id=str(data.get("id", "")),
            source_node=str(data["source_node"]),
            target_node=str(data["target_node"]),
            relationship=str(data["relationship"]),
            evidence=str(data.get("evidence", "")),
            source_page=int(data.get("source_page", 1)),
            confidence=conf,
            status=str(data.get("status", "ASSERTED")),
            properties=data.get("properties", {}),
            extraction_confidence=float(data.get("extraction_confidence", conf)),
            relationship_confidence=float(data.get("relationship_confidence", conf)),
            evidence_strength=float(data.get("evidence_strength", 1.0)),
        )


@dataclass
class PageProcessingResult:
    page_number: int
    nodes_created: int = 0
    nodes_reused: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    references_resolved: int = 0
    ambiguities_encountered: int = 0
    raw_llm_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "nodes_created": self.nodes_created,
            "nodes_reused": self.nodes_reused,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "references_resolved": self.references_resolved,
            "ambiguities_encountered": self.ambiguities_encountered,
        }


@dataclass
class UnresolvedRef:
    """An anaphoric or cross-reference mention that could not be resolved during page extraction."""
    source_node_id: str
    phrase: str
    target_type: str
    page_number: int
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "phrase": self.phrase,
            "target_type": self.target_type,
            "page_number": self.page_number,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnresolvedRef:
        return cls(
            source_node_id=str(data.get("source_node_id", "")),
            phrase=str(data.get("phrase", "")),
            target_type=str(data.get("target_type", "")),
            page_number=int(data.get("page_number", 1)),
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class PageDelta:
    """
    Immutable extraction delta emitted by an independent concurrent page worker.
    Contains local entities, relationships, unresolved references, and local ID maps.
    Represents observations and proposals; MUST NOT contain canonical mutation commands.
    """
    page_number: int
    entities: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphEdge] = field(default_factory=list)
    unresolved_references: List[UnresolvedRef] = field(default_factory=list)
    raw_id_map: Dict[str, str] = field(default_factory=dict)
    # Extraction quality for this page (audit #1): "clean" (parsed first try),
    # "partial" (needed the substring salvage path), or "degraded" (truncated after
    # retry, or invalid JSON after one re-ask). Non-clean pages MUST NOT be
    # checkpointed as COMPLETED so a resumed run re-processes them.
    quality: str = "clean"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "entities": [n.to_dict() for n in self.entities],
            "relationships": [e.to_dict() for e in self.relationships],
            "unresolved_references": [r.to_dict() for r in self.unresolved_references],
            "raw_id_map": dict(self.raw_id_map),
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageDelta:
        entities = [GraphNode.from_dict(n) if isinstance(n, dict) else n for n in data.get("entities", [])]
        relationships = [GraphEdge.from_dict(e) if isinstance(e, dict) else e for e in data.get("relationships", [])]
        unresolved = [
            UnresolvedRef.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("unresolved_references", [])
        ]
        return cls(
            page_number=int(data.get("page_number", 1)),
            entities=entities,
            relationships=relationships,
            unresolved_references=unresolved,
            raw_id_map=dict(data.get("raw_id_map", {})),
            quality=str(data.get("quality", "clean")),
        )


@dataclass
class CandidateEvaluation:
    """Evaluation of a canonical node candidate for an incoming entity observation."""
    candidate_id: str
    candidate_node: GraphNode
    candidate_score: float  # Explicitly a heuristic ranking score [0.0 - 1.0], not a calibrated probability
    matching_signals: List[str] = field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_reasons: List[str] = field(default_factory=list)
    compatibility: str = TypeCompatibility.UNKNOWN.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_score": round(self.candidate_score, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_detected": self.contradiction_detected,
            "contradiction_reasons": list(self.contradiction_reasons),
            "compatibility": self.compatibility,
        }


@dataclass
class ResolutionDecision:
    """Explicit decision made by GraphMemoryManager for an entity observation."""
    status: ResolutionStatus
    canonical_id: Optional[str] = None
    winning_candidate: Optional[CandidateEvaluation] = None
    all_candidates: List[CandidateEvaluation] = field(default_factory=list)
    candidate_score: float = 0.0
    second_candidate_score: float = 0.0
    decision_margin: float = 0.0
    matching_signals: List[str] = field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_reasons: List[str] = field(default_factory=list)
    resolution_method: str = "NONE"
    rationale: str = ""
    algorithm_version: str = "entity_resolution_v2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, ResolutionStatus) else str(self.status),
            "canonical_id": self.canonical_id,
            "candidate_score": round(self.candidate_score, 4),
            "second_candidate_score": round(self.second_candidate_score, 4),
            "decision_margin": round(self.decision_margin, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_detected": self.contradiction_detected,
            "contradiction_reasons": list(self.contradiction_reasons),
            "resolution_method": self.resolution_method,
            "rationale": self.rationale,
            "algorithm_version": self.algorithm_version,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
        }


@dataclass
class MergeAuditRecord:
    """Durable audit record explaining why two observations were merged canonically."""
    source_node_id: str
    source_page: int
    source_value: str
    canonical_node_id: str
    canonical_value: str
    resolution_method: str
    candidate_score: float
    second_candidate_score: float = 0.0
    decision_margin: float = 0.0
    matching_signals: List[str] = field(default_factory=list)
    contradiction_result: str = "PASSED_HARD_GATES"
    contradiction_reasons: List[str] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    algorithm_version: str = "entity_resolution_v2"
    pipeline_version: str = "v1"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "source_page": self.source_page,
            "source_value": self.source_value,
            "canonical_node_id": self.canonical_node_id,
            "canonical_value": self.canonical_value,
            "resolution_method": self.resolution_method,
            "candidate_score": round(self.candidate_score, 4),
            "second_candidate_score": round(self.second_candidate_score, 4),
            "decision_margin": round(self.decision_margin, 4),
            "matching_signals": list(self.matching_signals),
            "contradiction_result": self.contradiction_result,
            "contradiction_reasons": list(self.contradiction_reasons),
            "source_pages": list(self.source_pages),
            "evidence": list(self.evidence),
            "algorithm_version": self.algorithm_version,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.timestamp,
        }


@dataclass
class PageDeltaValidationResult:
    """Result of pre-ingestion PageDelta structural and semantic validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_page_delta(delta: Any) -> PageDeltaValidationResult:
    """
    Validate a PageDelta before durable checkpointing and canonical memory ingestion.
    Enforces:
    1. Valid delta instance and page number >= 1
    2. Local entity IDs unique within the delta
    3. Entities have non-empty value and type
    4. Relationships have non-empty endpoints and relationship types
    5. Confidences within [0.0, 1.0]
    6. Valid status values
    7. No destructive canonical mutation commands
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(delta, PageDelta):
        return PageDeltaValidationResult(
            is_valid=False,
            errors=[f"Expected PageDelta instance, got {type(delta).__name__}"],
        )

    if delta.page_number is None or delta.page_number < 1:
        errors.append(f"Invalid page_number: {delta.page_number}")

    seen_node_ids: Set[str] = set()
    for idx, ent in enumerate(delta.entities):
        if not isinstance(ent, GraphNode):
            errors.append(f"Entity at index {idx} is not a GraphNode instance")
            continue
        if not ent.id or not ent.id.strip():
            errors.append(f"Entity at index {idx} has an empty or null id")
        elif ent.id in seen_node_ids:
            errors.append(f"Duplicate entity id '{ent.id}' in PageDelta for page {delta.page_number}")
        else:
            seen_node_ids.add(ent.id)

        if not ent.value or not ent.value.strip():
            errors.append(f"Entity '{ent.id}' has empty value")
        if not ent.type or not ent.type.strip():
            errors.append(f"Entity '{ent.id}' has empty type")

        if ent.confidence < 0.0 or ent.confidence > 1.0:
            errors.append(f"Entity '{ent.id}' confidence {ent.confidence} out of range [0.0, 1.0]")
        if ent.extraction_confidence < 0.0 or ent.extraction_confidence > 1.0:
            errors.append(f"Entity '{ent.id}' extraction_confidence {ent.extraction_confidence} out of range [0.0, 1.0]")
        if ent.resolution_confidence is not None and (ent.resolution_confidence < 0.0 or ent.resolution_confidence > 1.0):
            errors.append(f"Entity '{ent.id}' resolution_confidence {ent.resolution_confidence} out of range [0.0, 1.0]")

        if ent.status not in ("ASSERTED", "INFERRED", "UNCERTAIN", "EXTRACTED"):
            warnings.append(f"Entity '{ent.id}' has non-standard status: '{ent.status}'")

        # Guard: PageDelta must not carry destructive commands
        props = getattr(ent, "properties", {}) or {}
        for destructive_key in ("delete_node", "merge_override", "drop_graph", "truncate"):
            if destructive_key in props:
                errors.append(f"Destructive mutation command '{destructive_key}' rejected in PageDelta")

    for idx, edge in enumerate(delta.relationships):
        if not isinstance(edge, GraphEdge):
            errors.append(f"Relationship at index {idx} is not a GraphEdge instance")
            continue
        if not edge.source_node or not edge.source_node.strip():
            errors.append(f"Edge at index {idx} missing source_node")
        if not edge.target_node or not edge.target_node.strip():
            errors.append(f"Edge at index {idx} missing target_node")
        if not edge.relationship or not edge.relationship.strip():
            errors.append(f"Edge at index {idx} missing relationship type")

        if edge.confidence < 0.0 or edge.confidence > 1.0:
            errors.append(f"Edge at index {idx} confidence {edge.confidence} out of range [0.0, 1.0]")
        if edge.relationship_confidence < 0.0 or edge.relationship_confidence > 1.0:
            errors.append(f"Edge at index {idx} relationship_confidence {edge.relationship_confidence} out of range [0.0, 1.0]")

    return PageDeltaValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
    )
```

## File: src/ai/layer3_extraction/graph_agent/resolver.py
```python
"""
File: resolver.py
Purpose: Synthesize final dynamic Pydantic schema from GraphMemory.
Hardened to:
- Distinguish ASSERTED (primary/direct evidence), SUPPORTED INFERRED (supporting evidence),
  and UNCERTAIN (candidate/unverified evidence) in evidence formatting.
- Prevent UNCERTAIN facts from silently contaminating authoritative schema output.
- Guard fallback reconciliation from treating uncertain nodes as authoritative candidates.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_graph_evidence_for_resolution(graph: GraphMemory, schema_fields: List[Dict[str, str]]) -> str:
    """
    Format all entities, values, relationships, and provenances from the graph
    into a structured prompt context for final schema resolution.
    Explicitly labels evidence status so the LLM distinguishes authoritative from unverified facts.
    """
    lines = [
        "=== EXTRACTED GRAPH ENTITIES ===",
        "NOTE: Prefer [ASSERTED] primary evidence and [SUPPORTED INFERRED] evidence.",
        "Do NOT treat [UNCERTAIN - UNVERIFIED] candidates as authoritative facts unless supported by explicit evidence.\n",
    ]
    nodes = list(graph._nodes.values())
    if not nodes:
        lines.append("No entities were extracted into graph memory.")
    else:
        for n in nodes:
            pages = f"Page {','.join(map(str, n.source_pages))}" if n.source_pages else "Page ?"
            ev = f' | Evidence: "{n.evidence[0].text}"' if n.evidence else ""
            category_tag = f" [{n.category}]"

            status_upper = (n.status or "ASSERTED").upper()
            if status_upper == "ASSERTED":
                status_label = "[ASSERTED - Primary Evidence]"
            elif status_upper == "INFERRED":
                status_label = f"[SUPPORTED INFERRED - Conf: {n.resolution_confidence:.2f}]"
            elif status_upper == "UNCERTAIN":
                status_label = "[UNCERTAIN - UNVERIFIED Candidate]"
            else:
                status_label = f"[{status_upper}]"

            lines.append(
                f"- [{n.id}] {status_label} {n.type} ({n.label}): \"{n.value}\" ({pages}){category_tag}{ev}"
            )

    lines.append("\n=== ENTITY RELATIONSHIPS ===")
    edges = list(graph._edges.values())
    if not edges:
        lines.append("No explicit relationships recorded.")
    else:
        for e in edges:
            src = graph.get_node(e.source_node)
            tgt = graph.get_node(e.target_node)
            src_str = f"{src.value} ({src.type})" if src else e.source_node
            tgt_str = f"{tgt.value} ({tgt.type})" if tgt else e.target_node
            ev = f' | Evidence: "{e.evidence}"' if e.evidence else ""

            e_status = (e.status or "ASSERTED").upper()
            if e_status == "ASSERTED":
                rel_status = "ASSERTED"
            elif e_status == "INFERRED":
                rel_status = f"SUPPORTED INFERRED (Conf: {e.relationship_confidence:.2f})"
            elif e_status == "UNCERTAIN":
                rel_status = "UNCERTAIN - UNVERIFIED"
            else:
                rel_status = e_status

            lines.append(
                f"- {src_str} --[{e.relationship} (Status: {rel_status})]-> {tgt_str} (Page {e.source_page}){ev}"
            )

    return "\n".join(lines)


def resolve_schema_from_graph(
    graph: GraphMemory,
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
) -> BaseModel:
    """
    Produce final hydrated dynamic Pydantic BaseModel instance from GraphMemory.
    Does NOT reread or concatenate raw page markdown — graph memory is the single source of truth.
    Authoritative resolution preferentially consumes ASSERTED and SUPPORTED INFERRED evidence,
    and protects output from UNCERTAIN fact contamination.
    """
    schema_fields = [
        {"name": k, "description": v.description or k}
        for k, v in schema.model_fields.items()
    ]
    field_names = set(schema.model_fields.keys())

    evidence_text = format_graph_evidence_for_resolution(graph, schema_fields)
    logger.info(
        "graph.schema_resolution.start",
        fields=list(field_names),
        evidence_lines=len(evidence_text.splitlines()),
    )

    try:
        result = llm.resolve_schema_from_graph(evidence_text, schema)
    except Exception as exc:
        logger.warning("graph.schema_resolution.llm_failed_fallback_to_graph", error=str(exc))
        try:
            default_instance = schema()
        except Exception:
            default_instance = None
        default_values: Dict[str, Any] = default_instance.model_dump() if default_instance is not None else {}
        result = schema.model_validate(default_values)

    res_dict = result.model_dump()
    merged = dict(res_dict)

    # Post-resolution reconciliation:
    # If any schema field is empty in the LLM output, look for authoritative graph matches.
    # Graded fallback (first match wins, UNCERTAIN nodes excluded, only empty fields filled):
    #   (i)  exact schema_field_name match (authoritative tag), then
    #   (ii) normalized node-label equals-or-contains the field name
    #        (safety net for values present in the graph but mis-tagged with the
    #         wrong schema_field_name / sfn=null, e.g. dense-table extraction misses).
    # Pass (ii) only runs after pass (i) finds nothing for a given field, so an
    # explicitly tagged node always wins over a label-inferred one.
    def _norm_key(text: str) -> str:
        # Lowercase, drop non-alphanumerics -> comparable token (e.g. "Non_Medical_Deductions" ~ "non medical deductions")
        return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    for fname in field_names:
        if merged.get(fname) not in ("", None):
            continue

        # Pass (i): exact schema_field_name match
        filled = False
        for node in graph._nodes.values():
            if node.status == "UNCERTAIN":
                continue
            # Audit #3: a merged node may carry a UNION of tags in
            # properties['schema_field_names']; the field matches if it is the scalar
            # primary tag OR a member of the union. (Lookup only — scoring unchanged.)
            tag_union = node.properties.get("schema_field_names")
            tag_match = (
                node.properties.get("schema_field_name") == fname
                or (isinstance(tag_union, list) and fname in tag_union)
            )
            if tag_match and node.value:
                merged[fname] = node.value
                logger.debug("graph.resolver.reconciled_tagged", field=fname, value=node.value)
                filled = True
                break
        if filled:
            continue

        # Pass (ii): normalized label equals-or-contains the field name
        fname_key = _norm_key(fname)
        if not fname_key:
            continue
        for node in graph._nodes.values():
            if node.status == "UNCERTAIN":
                continue
            if not node.value:
                continue
            label_key = _norm_key(getattr(node, "label", ""))
            if not label_key:
                continue
            if label_key == fname_key or fname_key in label_key or label_key in fname_key:
                merged[fname] = node.value
                logger.debug(
                    "graph.resolver.reconciled_label_fallback",
                    field=fname,
                    value=node.value,
                    node_label=getattr(node, "label", ""),
                )
                break

    if merged != res_dict:
        result = schema.model_validate(merged)

    logger.info(
        "graph.extraction.completed",
        resolved_fields={k: v for k, v in result.model_dump().items() if v not in ("", None)},
        graph_stats=graph.stats(),
    )
    return result
```

## File: src/ai/layer3_extraction/graph_agent/query_service.py
```python
"""
File: query_service.py
Purpose: Bounded, cycle-safe Graph Query Service for Layer 3 Graph Memory.
         Enables the Query Bot to perform multi-hop graph retrieval and grounded Q&A.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import GraphEdge, GraphNode
from src.ai.layer3_extraction.graph_agent.evidence_retriever import get_index_for_graph
from src.ai.layer3_extraction.graph_agent.grounding_checker import check_grounding
from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configurable retrieval bounds
DEFAULT_MAX_HOPS = getattr(settings, "graph_query_max_hops", 3)
DEFAULT_MAX_NODES = getattr(settings, "graph_query_max_nodes", 25)
DEFAULT_MAX_EDGES = getattr(settings, "graph_query_max_edges", 30)

# Phase 4 — vector (semantic) seed retrieval bounds. Additive to keyword seeds.
DEFAULT_VECTOR_TOP_K = getattr(settings, "graph_query_vector_top_k", 8)
DEFAULT_VECTOR_MIN_SCORE = getattr(settings, "graph_query_vector_min_score", 0.30)
# Cap on the MERGED seed set so downstream BFS bounds stay as before.
DEFAULT_MAX_SEEDS = getattr(settings, "graph_query_max_seeds", 10)

STOP_WORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "at", "in", "for",
    "on", "by", "about", "with", "from", "into", "through", "during",
    "before", "after", "above", "below", "what", "who", "which", "where",
    "when", "why", "how", "associated",
    "tell", "me", "find", "give", "show", "can", "could", "would", "should",
    "of", "and", "or", "underwent", "performed", "recorded", "happened",
}


@dataclass
class GraphQueryResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_nodes: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_edges: List[Dict[str, Any]] = field(default_factory=list)
    hops_traversed: int = 0
    mode: str = "graph"
    # Phase 4 retrieval telemetry (additive; defaults keep old behavior/shape).
    seed_sources: Dict[str, Any] = field(default_factory=dict)
    # Phase 4 Part 2 grounding verification (additive). Empty dict when not run.
    grounding: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_nodes": self.retrieved_nodes,
            "retrieved_edges": self.retrieved_edges,
            "hops_traversed": self.hops_traversed,
            "mode": self.mode,
            "seed_sources": self.seed_sources,
            "grounding": self.grounding,
        }


class GraphQueryService:
    """
    Dedicated Graph Reasoning Service.
    Retrieves relevant subgraphs via bounded, cycle-safe traversal and coordinates
    evidence-grounded answering.
    """

    def __init__(
        self,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_edges: int = DEFAULT_MAX_EDGES,
    ) -> None:
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _clean_stem(word: str) -> str:
        """Strip common inflectional suffixes for robust entity matching."""
        w = word.lower().strip()
        for suffix in ["tions", "tion", "ments", "ment", "ed", "ing", "es", "s"]:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        return w

    @staticmethod
    def _is_garbage_node(node: GraphNode) -> bool:
        """Filter out low-confidence OCR noise, single-char artifacts, or placeholder nodes."""
        v = node.value.strip()
        v_lower = v.lower()
        if len(v) <= 1 and v_lower in ["u", "a", "x", "o", "-", "", "n", "y"]:
            return True
        if any(p in v_lower for p in ["not clearly visible", "not visible", "unreadable", "not legible", "not available"]):
            return True
        if node.type == "Amount":
            if not re.search(r"\d", v):
                return True
            words = v.split()
            if len(words) > 4 and not any(ch in v_lower for ch in ["₹", "$", "rs", "inr"]):
                return True
        return False

    def _extract_query_tokens(self, question: str) -> List[str]:
        """Extract meaningful keyword terms from question, excluding stop words."""
        cleaned = re.sub(r"[^\w\s]", " ", question.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2 and t not in STOP_WORDS]
        return tokens

    def find_seed_nodes(
        self,
        graph: GraphMemory,
        question: str,
        max_seeds: int = 6,
    ) -> List[GraphNode]:
        """
        Identify seed nodes in graph relevant to the question based on:
        1. Exact substantial value/alias matches
        2. Multi-token and stem matches across label, schema_field, and value
        3. Domain synonym expansion (e.g. 'claimed' -> 'claimed_amount', 'deducted' -> 'non_medical_deductions')
        4. Quality tie-breaking prioritizing verified numeric data and early document summary pages
        """
        norm_q = self._normalize(question)
        tokens = self._extract_query_tokens(question)
        stems = {self._clean_stem(t) for t in tokens}

        # Domain synonym expansion for common question intents
        # Universal cross-domain synonym expansion for common question intents
        SYNONYMS: Dict[str, List[str]] = {
            # Identity & People
            "name": ["person", "candidate", "applicant", "patient", "employee", "author", "client", "customer", "vendor", "full_name", "contact_person"],
            "person": ["candidate", "applicant", "employee", "patient", "client", "customer", "doctor", "signatory", "guardian", "officer"],
            "candidate": ["applicant", "person", "employee", "full_name", "name"],
            "applicant": ["candidate", "person", "full_name", "name"],
            "patient": ["person", "patient_name", "insured", "policyholder"],
            "doctor": ["physician", "consultant", "provider", "specialist", "surgeon", "treating_doctor", "reference_doctor", "dr"],
            "doctors": ["doctor", "physician", "consultant", "provider", "specialist", "dr"],
            "physician": ["doctor", "consultant", "specialist"],
            "consultant": ["doctor", "physician", "specialist", "advisor"],
            "vendor": ["supplier", "seller", "provider", "merchant", "company", "issuer"],
            "customer": ["client", "buyer", "recipient", "purchaser", "consumer", "insured", "patient"],
            "employer": ["company", "organization", "workplace", "firm"],
            "employee": ["staff", "worker", "candidate", "person"],
            "age": ["dob", "birth", "years", "date_of_birth"],
            # Organizations & Institutions
            "company": ["organization", "vendor", "issuer", "bank", "hospital", "firm", "insurer", "institution", "agency", "supplier"],
            "organization": ["company", "institution", "agency", "firm", "hospital", "bank", "vendor"],
            "hospital": ["clinic", "healthcare", "facility", "provider", "institution"],
            "bank": ["financial_institution", "lender", "bank_name", "issuer"],
            "insurer": ["insurance_company", "sponsor", "tpa", "underwriter", "carrier"],
            "insurance": ["insurance_company", "sponsor", "tpa", "policy", "coverage", "carrier"],
            # Financials, Invoices, Billing
            "amount": ["total", "sum", "cost", "price", "fee", "rate", "subtotal", "balance", "charge", "value", "payable", "paid"],
            "total": ["gross", "net", "subtotal", "grand_total", "total_amount", "total_bill", "overall_amount", "sum"],
            "bill": ["invoice", "statement", "receipt", "charge", "total_bill", "fee"],
            "invoice": ["bill", "statement", "receipt", "tax_invoice", "proforma"],
            "claimed": ["claim", "claimed_amount", "total_bill", "requested_amount"],
            "claim": ["claimed_amount", "settlement", "sanctioned_amount", "insurance_claim"],
            "payable": ["due", "net_payable", "amount_due", "balance_due", "paid_by_insured", "patient_payable", "outstanding"],
            "pocket": ["out_of_pocket", "patient_payable", "copay", "non_payable", "deduction", "uncovered"],
            "paid": ["settled", "disbursed", "remitted", "cleared", "amount_paid"],
            "tax": ["gst", "vat", "sales_tax", "cgst", "sgst", "igst", "service_tax", "cess", "withholding"],
            "discount": ["rebate", "concession", "reduction", "deduction"],
            "deduction": ["deducted", "deductions", "non_medical", "non_payable", "disallowed", "copay", "withholding"],
            "deductions": ["deducted", "deduction", "non_medical", "non_payable", "disallowed", "copay", "withholding"],
            "deducted": ["deduction", "deductions", "disallowed", "non_payable"],
            "sanctioned": ["approved", "settled", "sanctioned_amount", "authorized_amount", "final_claim"],
            "approved": ["sanctioned", "authorized", "settled", "granted"],
            "fee": ["cost", "charge", "price", "rate", "tariff", "tuition", "premium"],
            "salary": ["wages", "compensation", "stipend", "pay", "ctc", "remuneration"],
            # Identifiers & Numbers
            "number": ["id", "identifier", "code", "reference", "ref_no", "no"],
            "id": ["identifier", "number", "code", "reference"],
            "policy": ["policy_number", "insurance_policy", "policy_no", "contract_number"],
            "account": ["account_number", "acc_no", "bank_account"],
            # Dates & Temporal
            "date": ["time", "period", "day", "month", "year", "timestamp", "validity", "expiry"],
            "expiry": ["expiration", "valid_until", "validity", "due_date"],
            # Items & Details
            "item": ["line_item", "product", "description", "service", "goods", "article"],
            "items": ["line_items", "products", "services", "goods", "articles"],
            "service": ["procedure", "operation", "treatment", "item", "offering"],
            "skills": ["technologies", "competencies", "tools", "expertise", "qualifications"],
            "education": ["degree", "qualification", "university", "college", "school", "academic"],
            "experience": ["work_experience", "employment", "history", "career"],
            "address": ["location", "place", "city", "state", "country", "pin", "zipcode", "residence"],
        }
        syn_tokens: Set[str] = set()
        for tok in tokens:
            if tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[tok])
            stem_tok = self._clean_stem(tok)
            if stem_tok in SYNONYMS:
                syn_tokens.update(SYNONYMS[stem_tok])

        # Dynamic Interrogative Target Focus:
        # Detect target focus by scanning query tokens & expanded synonyms against all entity types and node labels present in the graph memory.
        target_focus_types: Set[str] = set()
        target_focus_labels: Set[str] = set()

        for t in getattr(graph, "_type_index", {}).keys():
            t_norm = self._normalize(t)
            if t_norm in norm_q or any(tok in t_norm or self._clean_stem(tok) in t_norm for tok in tokens):
                target_focus_types.add(t_norm)
            for syn in syn_tokens:
                if syn in t_norm or t_norm in syn:
                    target_focus_types.add(t_norm)

        for n in graph._nodes.values():
            lbl_norm = self._normalize(n.label)
            if any(tok in lbl_norm or self._clean_stem(tok) in lbl_norm for tok in tokens if len(tok) >= 3):
                target_focus_labels.add(lbl_norm)

        scored_candidates: List[Tuple[float, int, int, int, int, int, GraphNode]] = []

        for node in graph._nodes.values():
            if self._is_garbage_node(node):
                continue

            val_norm = self._normalize(node.value)
            type_norm = self._normalize(node.type)
            lbl_norm = self._normalize(node.label)
            aliases_norm = [self._normalize(a) for a in getattr(node, "aliases", [])]
            schema_field = ""
            if getattr(node, "properties", None) and isinstance(node.properties, dict):
                schema_field = self._normalize(str(node.properties.get("schema_field_name", "")))

            score = 0.0
            matched_tokens = 0

            # Target focus boost: ensures the actual entity being asked for is prioritized
            if target_focus_types or target_focus_labels:
                if type_norm in target_focus_types or any(tfl in lbl_norm for tfl in target_focus_labels) or (schema_field and any(tfl in schema_field for tfl in target_focus_labels)):
                    score += 5.0
                    matched_tokens += 2

            # Substantial exact value match in question
            if len(val_norm) >= 3 and val_norm in norm_q:
                score += 2.0
            for a in aliases_norm:
                if len(a) >= 3 and a in norm_q:
                    score += 1.8

            # Substring and token matches
            for tok in tokens:
                if len(tok) >= 3 and tok in val_norm:
                    matched_tokens += 1
                    score += 1.5
                if tok in lbl_norm or (schema_field and tok in schema_field):
                    matched_tokens += 1
                    score += 1.2
                elif tok in type_norm:
                    score += 0.35
                for a in aliases_norm:
                    if tok in a:
                        score += 0.8

            # Stem matches on label & schema fields
            for st in stems:
                if len(st) >= 3:
                    if st in lbl_norm or (schema_field and st in schema_field):
                        score += 0.8
                    elif st in type_norm:
                        score += 0.2

            # Match against expanded synonyms
            for st in syn_tokens:
                if st in lbl_norm or (schema_field and st in schema_field):
                    score += 0.7
                elif st in type_norm:
                    score += 0.2

            if score > 0.0:
                has_digits = 1 if (node.type == "Amount" and re.search(r"\d", val_norm)) else 0
                page_coverage = len(node.source_pages or [])
                min_page = min(node.source_pages or [999])
                scored_candidates.append((
                    score,
                    matched_tokens,
                    has_digits,
                    1 if node.category == "EXPLICIT" else 0,
                    page_coverage,
                    -min_page,
                    node,
                ))

        # Sort: score DESC, matched_tokens DESC, valid digits DESC, EXPLICIT DESC, page coverage DESC, earlier pages DESC
        scored_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]), reverse=True)

        # Cluster seeds to prevent any single entity type from dominating all slots
        seen_ids = set()
        seeds: List[GraphNode] = []
        type_counts: Dict[str, int] = {}

        # Pass 1: Add target focus nodes first
        if target_focus_types or target_focus_labels:
            for cand in scored_candidates:
                node = cand[6]
                ntype = node.type.lower()
                nlbl = node.label.lower()
                if node.id not in seen_ids:
                    if ntype in target_focus_types or any(tfl in nlbl for tfl in target_focus_labels):
                        seen_ids.add(node.id)
                        seeds.append(node)
                        type_counts[ntype] = type_counts.get(ntype, 0) + 1
                        if len(seeds) >= max_seeds:
                            break

        # Pass 2: Add other high-scoring seeds with at most 2 per entity type
        for cand in scored_candidates:
            node = cand[6]
            ntype = node.type.lower()
            if node.id not in seen_ids:
                if type_counts.get(ntype, 0) < 2 or len(seeds) < 3:
                    seen_ids.add(node.id)
                    seeds.append(node)
                    type_counts[ntype] = type_counts.get(ntype, 0) + 1
                    if len(seeds) >= max_seeds:
                        break

        # Pass 3: Fill any remaining slots
        if len(seeds) < max_seeds:
            for cand in scored_candidates:
                node = cand[6]
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    seeds.append(node)
                    if len(seeds) >= max_seeds:
                        break

        logger.debug("graph.query.seed_found", count=len(seeds), seed_ids=[s.id for s in seeds])
        return seeds

    def find_vector_seed_nodes(
        self,
        graph: GraphMemory,
        question: str,
        top_k: int = DEFAULT_VECTOR_TOP_K,
        min_score: float = DEFAULT_VECTOR_MIN_SCORE,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Phase 4 ADDITIVE candidate source: semantic retrieval over evidence spans.

        Embeds the question and does cosine top-K search over the graph's evidence
        spans (built once, cached per graph). Returns the owning GraphNodes with
        their best cosine score. This finds paraphrased questions that keyword
        seed-finding misses (low lexical overlap, high semantic similarity).

        Degrades to [] if the embedding model/index is unavailable — the caller
        still has the keyword seeds, so retrieval never becomes a hard dependency.
        """
        try:
            index = get_index_for_graph(graph)
            if not index.available:
                return []
            hits = index.search(question, top_k=top_k, min_score=min_score)
            # Skip garbage nodes to match the keyword path's quality bar.
            out: List[Tuple[GraphNode, float]] = [
                (node, score) for node, score, _span in hits if not self._is_garbage_node(node)
            ]
            logger.debug(
                "graph.query.vector_seed_found",
                count=len(out),
                top_scores=[round(s, 3) for _n, s in out[:5]],
            )
            return out
        except Exception as exc:
            logger.warning("graph.query.vector_seed_error", error=str(exc))
            return []

    @staticmethod
    def _merge_seed_sets(
        keyword_seeds: List[GraphNode],
        vector_seeds: List[Tuple[GraphNode, float]],
        max_total: int,
    ) -> Tuple[List[GraphNode], Dict[str, Any]]:
        """
        Merge keyword and vector candidate sets, deduped by node id.

        Order policy: keyword seeds first (preserves the existing, well-tuned
        keyword ranking exactly), then vector-only seeds appended by descending
        cosine score. Capped at max_total so downstream BFS bounds are unchanged.
        Returns the merged seed list + telemetry about where each came from.

        DOCUMENTED BEHAVIOR (Phase 4, not a bug): adding vector seeds here can
        change which node the DETERMINISTIC fallback (_deterministic_fallback)
        ultimately selects, even though that method's own code is untouched.
        The fallback scores over the BFS-traversed node set, and a different
        seed set can yield a different traversed set. Verified on the Chander
        Kochhar paraphrase set: the fallback's chosen answer changed on 2/8
        queries, but net fallback correctness was UNCHANGED (2/8 correct before,
        2/8 after; zero correct->incorrect flips — both changed cases were
        already-incorrect either way). The safety net still functions; only its
        wrong-answer pick shifted on queries it was already failing.
        """
        merged: List[GraphNode] = []
        seen: Set[str] = set()

        for n in keyword_seeds:
            if n.id not in seen:
                seen.add(n.id)
                merged.append(n)

        vector_only_ids: List[str] = []
        for node, _score in sorted(vector_seeds, key=lambda x: x[1], reverse=True):
            if node.id not in seen:
                seen.add(node.id)
                merged.append(node)
                vector_only_ids.append(node.id)

        if max_total > 0:
            merged = merged[:max_total]

        telemetry = {
            "keyword_seed_count": len(keyword_seeds),
            "vector_seed_count": len(vector_seeds),
            "vector_only_added": [nid for nid in vector_only_ids if nid in {m.id for m in merged}],
            "merged_seed_count": len(merged),
        }
        return merged, telemetry

    def traverse_subgraph(
        self,
        graph: GraphMemory,
        seed_nodes: List[GraphNode],
    ) -> Tuple[List[GraphNode], List[GraphEdge], int]:
        """
        Bounded, cycle-safe Breadth-First Search (BFS) starting from seed nodes.
        Strictly limits depth (hops), nodes, and edges, preventing infinite loops.
        """
        if not seed_nodes:
            return [], [], 0

        visited_nodes: Set[str] = {s.id for s in seed_nodes}
        visited_edges: Set[str] = set()

        collected_nodes: List[GraphNode] = list(seed_nodes)
        collected_edges: List[GraphEdge] = []

        # Queue contains (node_id, current_hop)
        queue: deque[Tuple[str, int]] = deque([(s.id, 0) for s in seed_nodes])
        max_hop_reached = 0

        while queue and len(collected_nodes) < self.max_nodes:
            curr_id, curr_hop = queue.popleft()
            max_hop_reached = max(max_hop_reached, curr_hop)

            if curr_hop >= self.max_hops:
                continue

            # Bidirectional neighbor lookup
            neighbors = graph.read_neighbors(curr_id, direction="both")
            for edge, neighbor_node in neighbors:
                # Cycle prevention on edges
                if edge.id in visited_edges:
                    continue
                visited_edges.add(edge.id)

                if len(collected_edges) < self.max_edges:
                    collected_edges.append(edge)

                # Cycle prevention on nodes
                if neighbor_node.id not in visited_nodes:
                    visited_nodes.add(neighbor_node.id)
                    if len(collected_nodes) < self.max_nodes:
                        collected_nodes.append(neighbor_node)
                    queue.append((neighbor_node.id, curr_hop + 1))

        logger.debug(
            "graph.query.traversal_completed",
            nodes=len(collected_nodes),
            edges=len(collected_edges),
            max_hop=max_hop_reached,
        )
        return collected_nodes, collected_edges, max_hop_reached

    def format_evidence_context(
        self,
        graph: GraphMemory,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        max_evidence: int = 16,
        max_sources: int = 8,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved nodes, relationships, and provenances into a high-signal, compact context for the LLM.
        Ensures entity diversity so citations aren't crowded out by a single high-frequency entity.
        """
        lines = ["=== RELEVANT GRAPH ENTITIES ==="]
        for n in nodes[:15]:
            pages = f"Page {','.join(map(str, n.source_pages[:3]))}" if n.source_pages else "Page ?"
            if len(n.source_pages or []) > 3:
                pages += f",... ({len(n.source_pages)} pages total)"
            lines.append(f"- [{n.id}] {n.type} ({n.label}): \"{n.value}\" ({pages})")

        lines.append("\n=== RELATIONSHIPS ===")
        for e in edges[:15]:
            src = graph.get_node(e.source_node)
            tgt = graph.get_node(e.target_node)
            src_val = f"{src.value} ({src.type})" if src else e.source_node
            tgt_val = f"{tgt.value} ({tgt.type})" if tgt else e.target_node
            lines.append(f"- {src_val} --[{e.relationship}]--> {tgt_val} (Page {e.source_page})")

        lines.append("\n=== DOCUMENT EVIDENCE SPANS ===")
        sources: List[Dict[str, Any]] = []
        seen_ev: Set[Tuple[int, str]] = set()
        ev_count = 0

        # Pass 1: Select 1 top evidence span per node to guarantee entity diversity
        for n in nodes:
            if ev_count >= max_evidence:
                break
            sorted_ev = sorted(
                n.evidence or [],
                key=lambda ev: (-getattr(ev, "confidence", 0.5), getattr(ev, "page_number", 999)),
            )
            for ev in sorted_ev[:1]:
                text_clean = ev.text.strip()
                key = (ev.page_number, text_clean)
                if text_clean and key not in seen_ev:
                    seen_ev.add(key)
                    lines.append(f"- [Page {ev.page_number}]: \"{text_clean}\" (supports {n.label}: {n.value})")
                    ev_count += 1
                    if len(sources) < max_sources:
                        sources.append({
                            "page": ev.page_number,
                            "evidence": text_clean,
                            "node": n.value,
                            "type": n.type,
                        })

        # Pass 2: Select a second evidence span for key nodes up to limits
        for n in nodes:
            if ev_count >= max_evidence:
                break
            sorted_ev = sorted(
                n.evidence or [],
                key=lambda ev: (-getattr(ev, "confidence", 0.5), getattr(ev, "page_number", 999)),
            )
            for ev in sorted_ev[1:2]:
                text_clean = ev.text.strip()
                key = (ev.page_number, text_clean)
                if text_clean and key not in seen_ev:
                    seen_ev.add(key)
                    lines.append(f"- [Page {ev.page_number}]: \"{text_clean}\" (supports {n.label}: {n.value})")
                    ev_count += 1
                    if len(sources) < max_sources:
                        sources.append({
                            "page": ev.page_number,
                            "evidence": text_clean,
                            "node": n.value,
                            "type": n.type,
                        })

        # Edge evidence
        for e in edges:
            if ev_count >= max_evidence:
                break
            if e.evidence:
                text_clean = e.evidence.strip()
                key = (e.source_page, text_clean)
                if text_clean and key not in seen_ev:
                    seen_ev.add(key)
                    lines.append(f"- [Page {e.source_page}]: \"{text_clean}\" (supports relationship {e.relationship})")
                    ev_count += 1
                    if len(sources) < max_sources:
                        sources.append({
                            "page": e.source_page,
                            "evidence": text_clean,
                            "relationship": e.relationship,
                        })

        sources.sort(key=lambda s: s.get("page", 1))
        return "\n".join(lines), sources

    def query(
        self,
        graph: GraphMemory,
        question: str,
        llm: Optional[ExtractionLLMClient] = None,
        verify_grounding: bool = True,
    ) -> GraphQueryResult:
        """
        Execute a graph-backed query:
        1. Find seed entities (keyword) + vector evidence-span retrieval, merged.
        2. Bounded cycle-safe BFS traversal.
        3. Compact high-signal context extraction with capped citations.
        4. Grounded synthesis via LLM with strict role separation and fallback.
        5. (Phase 4 Part 2) Grounding verification: trace each answer claim back
           to specific evidence; flag ungrounded claims 'unverified'. Deterministic
           text-overlap primary + optional Bedrock entailment pass. Additive —
           never changes the answer text, only annotates it.
        """
        logger.info("graph.query.started", question=question)

        # Step 1a: Keyword seed-finding (existing, unchanged).
        keyword_seeds = self.find_seed_nodes(graph, question)

        # Step 1b: Vector seed-finding (Phase 4, ADDITIVE). Semantic retrieval
        # over evidence spans catches paraphrased questions the keyword path
        # misses. Degrades to [] if the embedding model/index is unavailable.
        vector_seeds = self.find_vector_seed_nodes(graph, question)

        # Step 1c: Merge keyword + vector candidates (dedup by node id). Keyword
        # ordering is preserved first; vector-only hits are appended by score.
        seeds, seed_telemetry = self._merge_seed_sets(
            keyword_seeds, vector_seeds, max_total=DEFAULT_MAX_SEEDS
        )
        if vector_seeds and seed_telemetry["vector_only_added"]:
            retrieval_mode = "hybrid"
        elif vector_seeds:
            retrieval_mode = "hybrid"
        else:
            retrieval_mode = "keyword_only"
        seed_telemetry["retrieval_mode"] = retrieval_mode

        # If no seeds found by EITHER source, check if question is asking for all entities of a type
        if not seeds:
            q_norm = self._normalize(question)
            for t, node_ids in getattr(graph, "_type_index", {}).items():
                if t in q_norm:
                    for nid in node_ids:
                        n = graph.get_node(nid)
                        if n and n not in seeds and not self._is_garbage_node(n):
                            seeds.append(n)
                            if len(seeds) >= 5:
                                break

        # If still no relevant nodes found in graph
        if not seeds:
            logger.info("graph.query.no_seeds_found", question=question)
            return GraphQueryResult(
                question=question,
                answer="The requested information could not be found in the document knowledge graph.",
                sources=[],
                retrieved_nodes=[],
                retrieved_edges=[],
                hops_traversed=0,
                mode="graph",
                seed_sources=seed_telemetry,
            )

        # Step 2: Bounded Traversal
        nodes, edges, max_hop = self.traverse_subgraph(graph, seeds)

        # Step 3: Format compact evidence context & sources
        context_text, sources = self.format_evidence_context(graph, nodes, edges)

        # Step 4: Synthesize answer
        answer = self._generate_answer(context_text, sources, nodes, question)
        logger.info("graph.query.answer_generated", answer_preview=answer[:100], sources_count=len(sources))

        # Step 5 (Phase 4 Part 2): verify grounding of the synthesized answer.
        # Deterministic overlap always runs; the optional Bedrock entailment pass
        # only upgrades heuristic-ungrounded claims and degrades gracefully.
        grounding: Dict[str, Any] = {}
        if verify_grounding and answer:
            try:
                grounding = check_grounding(
                    answer=answer,
                    evidence_items=sources,
                    use_llm=True,
                    llm_caller=self._bedrock_grounding_caller,
                )
                logger.info(
                    "graph.query.grounding_checked",
                    grounded=grounding.get("grounded_count"),
                    total=grounding.get("total_claims"),
                    unverified=len(grounding.get("unverified_claims", [])),
                )
            except Exception as exc:  # grounding must never break the answer path
                logger.warning("graph.query.grounding_error", error=str(exc))
                grounding = {}

        return GraphQueryResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_nodes=[n.to_dict() for n in nodes],
            retrieved_edges=[e.to_dict() for e in edges],
            hops_traversed=max_hop,
            mode="graph",
            seed_sources=seed_telemetry,
            grounding=grounding,
        )

    def _bedrock_grounding_caller(self, system: str, user: str) -> str:
        """Minimal Bedrock converse call for the optional grounding pass, reusing
        the same inference profile as answer synthesis. Returns '' on any failure
        so the grounding checker degrades to deterministic-overlap only.

        Inference model only — no other AWS services touched."""
        import os
        try:
            from app.config import settings as app_settings
        except ImportError:
            app_settings = None
        from src.config.settings import settings as a_settings

        bedrock_region = (
            getattr(app_settings, "bedrock_region", None)
            or getattr(a_settings, "bedrock_region", None)
            or os.getenv("IDP_BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or "ap-south-1"
        )
        bedrock_model_id = (
            getattr(app_settings, "bedrock_model_id", None)
            or getattr(a_settings, "bedrock_model_id", None)
            or os.getenv("IDP_BEDROCK_MODEL_ID")
            or os.getenv("BEDROCK_MODEL_ID")
            or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
        )
        try:
            import boto3
            profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
            if profile:
                b_client = boto3.Session(profile_name=profile, region_name=bedrock_region).client("bedrock-runtime", region_name=bedrock_region)
            else:
                b_client = boto3.client("bedrock-runtime", region_name=bedrock_region)
            kwargs = {
                "modelId": bedrock_model_id,
                "system": [{"text": system}],
                "messages": [{"role": "user", "content": [{"text": user}]}],
                "inferenceConfig": {"temperature": 0.0, "maxTokens": 600},
            }
            effort = getattr(app_settings, "bedrock_reasoning_effort", None) or getattr(a_settings, "bedrock_reasoning_effort", None) or "low"
            if effort and ("gpt-oss" in bedrock_model_id.lower() or "qdtz23c8eis1" in bedrock_model_id):
                kwargs["additionalModelRequestFields"] = {"reasoning_effort": effort}
            try:
                resp = b_client.converse(**kwargs)
            except Exception:
                kwargs.pop("additionalModelRequestFields", None)
                resp = b_client.converse(**kwargs)
            content = resp.get("output", {}).get("message", {}).get("content", [])
            return next((b["text"] for b in content if isinstance(b, dict) and "text" in b), "")
        except Exception as exc:
            logger.warning("graph.query.grounding_bedrock_unavailable", error=str(exc))
            return ""

    @staticmethod
    def _sanitize_model_output(content: str) -> str:
        """Strip <think> tags, conversational preamble, and leaked chain-of-thought scratchpad text."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip leading broken fragments or hanging quotes e.g. 'found". ' or '". '
        content = re.sub(r'^[^\w\s]*(?:found|stated|mentioned)?["\'”’\.]+\s*', '', content, flags=re.IGNORECASE).strip()
        # Strip asterisks / markdown bolding
        content = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", content)
        content = re.sub(r"[*_#`]+", "", content)
        content = re.sub(r"^(?:Assistant|Answer):\s*", "", content.strip(), flags=re.IGNORECASE)

        # Check for conversational reasoning preamble
        if re.search(r"^\s*(?:Based on the (?:prompt|constraint|evidence)|The user wants to know|Looking at the|From the relevant graph|To answer this question|I need to look|I can mention|1\.\s*Analyze)", content, re.IGNORECASE):
            # Check draft patterns
            m_draft = re.search(
                r'(?:combine into a smooth sentence|final polish|draft the response:?|final answer:?|answer:?)\s*[:"-]*\s*["\']?([^"\'\n]+)["\']?',
                content,
                re.IGNORECASE,
            )
            if m_draft and len(m_draft.group(1).strip()) > 5:
                cand = m_draft.group(1).strip()
                if not cand.endswith(":") and len(cand.split()) >= 3:
                    return cand

            # Try to extract explicit concluding statement
            m_concl = re.search(
                r"(?:(?:So|Therefore|In summary|Hence|In conclusion),?\s*([^\n]+))",
                content,
                re.IGNORECASE,
            )
            if m_concl:
                candidate = m_concl.group(1).strip()
                if len(candidate) > 10 and not candidate.endswith(":") and len(candidate.split()) >= 3:
                    return candidate

            # Extract from lines
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            meaningful_lines = []
            for line in lines:
                if not re.search(r"^\s*(?:\d+\.|\*|I will|I can|I should|Let's|The evidence|So I should|Looking at|From the|According to|Based on)", line, re.IGNORECASE):
                    if not line.endswith(":") and len(line) > 5:
                        meaningful_lines.append(line.strip())
            if meaningful_lines:
                if len(meaningful_lines) > 1 and re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", meaningful_lines[-1], re.IGNORECASE):
                    return f"{meaningful_lines[-2]} {meaningful_lines[-1]}"
                if not re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", meaningful_lines[-1], re.IGNORECASE):
                    return meaningful_lines[-1]

        # If output is too short, punctuation-only, ends in a colon, or merely a source citation without an answer, return empty so fallback kicks in
        clean_out = content.strip()
        clean_stripped = clean_out.rstrip(" :.-_#*")
        if (
            len(clean_out) <= 3
            or clean_stripped.endswith(":")
            or len(clean_out.split()) < 2
            or not re.search(r"\w", clean_out)
            or re.search(r"(?:prompt['s]*\s*constraint|prompt\s*constraint|system\s*prompt|internal\s*monologue|scratchpad)", clean_out, re.IGNORECASE)
            or re.match(r"^\(?(?:Source|Page):?\s*(?:Page\s*)?\d+\)?\.?$", clean_out, re.IGNORECASE)
            or re.match(r"^(?:\s*Source:\s*Page\s*\d+\s*)+$", clean_out, re.IGNORECASE)
            or any(bad in clean_out.lower() for bad in ["critical: do not", "internal monologue", "scratchpad", "system prompt", "knowledge graph evidence:", "based on the prompt", "e.g. 'source: page", "e.g. \"source: page", "found (e.g."])
            or clean_out.lower().startswith("found (")
        ):
            return ""
        return clean_out

    def _deterministic_fallback(
        self,
        question: str,
        nodes: List[GraphNode],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Universal, graph-grounded deterministic answering when LLM generation is unavailable."""
        q = question.lower()
        tokens = self._extract_query_tokens(question)
        stems = {self._clean_stem(t) for t in tokens}

        if not nodes:
            return "The requested information could not be found in the document graph."

        valid_nodes = [n for n in nodes if not self._is_garbage_node(n)]
        if not valid_nodes:
            return "The requested information could not be found in the document graph."

        # Dynamic scoring of nodes based on query token & stem overlap
        scored: List[Tuple[float, GraphNode]] = []
        for n in valid_nodes:
            val_norm = self._normalize(n.value)
            lbl_norm = self._normalize(n.label)
            type_norm = self._normalize(n.type)
            score = 0.0

            for tok in tokens:
                if len(tok) >= 3:
                    if tok in val_norm:
                        score += 2.0
                    if tok in lbl_norm:
                        score += 1.8
                    if tok in type_norm:
                        score += 0.8
            for st in stems:
                if len(st) >= 3:
                    if st in lbl_norm:
                        score += 1.2
                    if st in val_norm:
                        score += 1.0

            # Match against evidence text
            for ev in (n.evidence or []):
                ev_norm = self._normalize(getattr(ev, "text", ""))
                for tok in tokens:
                    if len(tok) >= 4 and tok in ev_norm:
                        score += 0.5

            if score > 0.0:
                scored.append((score, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [s[1] for s in scored] if scored else valid_nodes

        # 1. Multi-entity list queries (e.g. "which all doctors", "what are the items", "list all skills...")
        if any(w in q for w in ["which all", "all", "list", "what are the", "what are all"]):
            primary_type = top_nodes[0].type
            matching_list = [n for n in top_nodes if n.type == primary_type]
            seen_values = set()
            unique_list = []
            for n in matching_list:
                v_clean = n.value.strip().lower()
                if v_clean not in seen_values:
                    seen_values.add(v_clean)
                    page = n.source_pages[0] if n.source_pages else 1
                    lbl = n.label.replace("_", " ")
                    unique_list.append(f"{n.value} ({lbl}, Source: Page {page})")
            if unique_list:
                return f"The {primary_type.lower()}s recorded in the document are: {'; '.join(unique_list)}."

        # 2. Comparative / multi-part queries (e.g. payable from X and from Y, totals and subtotals)
        if len(top_nodes) >= 2 and any(w in q for w in ["and", "from", "between", "payable", "versus", "vs"]):
            items_formatted = []
            seen_v = set()
            for n in top_nodes[:3]:
                if n.value.strip().lower() not in seen_v:
                    seen_v.add(n.value.strip().lower())
                    page = n.source_pages[0] if n.source_pages else 1
                    items_formatted.append(f"{n.label.replace('_', ' ')} is {n.value} (Source: Page {page})")
            if items_formatted:
                return f"Based on the document: {', and '.join(items_formatted)}."

        # 3. Best single entity match
        best_node = top_nodes[0]
        page = best_node.source_pages[0] if best_node.source_pages else 1
        lbl_clean = best_node.label.replace("_", " ")
        return f"The {lbl_clean} is {best_node.value}. (Source: Page {page})"

    def _generate_answer(
        self,
        context_text: str,
        sources: List[Dict[str, Any]],
        nodes: List[GraphNode],
        question: str,
    ) -> str:
        """Call LLM API with proper system/user role separation, with fallback."""
        import os
        import httpx
        try:
            from app.config import settings as app_settings
        except ImportError:
            app_settings = None
        from src.config.settings import settings as a_settings

        api_key = (
            getattr(app_settings, "sarvam_api_key", None)
            or getattr(a_settings, "sarvam_api_key", None)
            or os.getenv("IDP_SARVAM_API_KEY")
            or os.getenv("SARVAM_API_KEY")
            or ""
        )
        base_url = (
            getattr(app_settings, "sarvam_base_url", None)
            or getattr(a_settings, "sarvam_base_url", None)
            or os.getenv("IDP_SARVAM_BASE_URL")
            or os.getenv("SARVAM_BASE_URL")
            or "https://api.sarvam.ai/v1"
        ).rstrip("/")
        model_name = (
            getattr(app_settings, "sarvam_model", None)
            or getattr(a_settings, "sarvam_model_name", None)
            or os.getenv("IDP_SARVAM_MODEL_NAME")
            or os.getenv("SARVAM_MODEL")
            or "sarvam-105b"
        )
        backend = (
            getattr(app_settings, "llm_provider", None)
            or getattr(a_settings, "extraction_backend", None)
            or os.getenv("IDP_EXTRACTION_BACKEND")
            or "bedrock"
        ).lower()
        bedrock_region = (
            getattr(app_settings, "bedrock_region", None)
            or getattr(a_settings, "bedrock_region", None)
            or os.getenv("IDP_BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or "ap-south-1"
        )
        bedrock_model_id = (
            getattr(app_settings, "bedrock_model_id", None)
            or getattr(a_settings, "bedrock_model_id", None)
            or os.getenv("IDP_BEDROCK_MODEL_ID")
            or os.getenv("BEDROCK_MODEL_ID")
            or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
        )

        system_prompt = (
            "You are a factual, concise Document QA assistant. "
            "Answer the user's question directly, accurately, and concisely in 1 or 2 plain sentences based strictly on the provided Document Knowledge Graph evidence. "
            "Always cite the exact source page number where the answer is found (e.g. 'Source: Page 4').\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. Grounding: Answer strictly based on the provided entities, properties, values, and relationships. Never guess or hallucinate.\n"
            "2. Entity Disambiguation: When multiple entities, amounts, parties (e.g. patient vs insurer, buyer vs vendor, employer vs candidate), or line items/deductions are asked, clearly distinguish each respective role and amount.\n"
            "3. Multi-Entity Listings: When asked to list items, persons, services, or figures (e.g. 'which all', 'what are the'), list all distinct matching entities found in the evidence.\n"
            "4. Conciseness: Do NOT output thinking, reasoning steps, internal monologue, numbered analysis lists, or scratchpads. Provide ONLY the final direct answer."
        )
        user_prompt = f"Document Knowledge Graph Evidence:\n{context_text}\n\nQuestion: {question}\nDirect Answer:"

        try:
            if backend == "bedrock" or (not api_key and bedrock_model_id):
                import boto3
                profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
                if profile:
                    b_client = boto3.Session(profile_name=profile, region_name=bedrock_region).client("bedrock-runtime", region_name=bedrock_region)
                else:
                    b_client = boto3.client("bedrock-runtime", region_name=bedrock_region)

                kwargs = {
                    "modelId": bedrock_model_id,
                    "system": [{"text": system_prompt}],
                    "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                    "inferenceConfig": {"temperature": 0.0, "maxTokens": 1500},
                }
                effort = getattr(app_settings, "bedrock_reasoning_effort", None) or getattr(a_settings, "bedrock_reasoning_effort", None) or "low"
                if effort and ("gpt-oss" in bedrock_model_id.lower() or "qdtz23c8eis1" in bedrock_model_id):
                    kwargs["additionalModelRequestFields"] = {"reasoning_effort": effort}

                try:
                    b_resp = b_client.converse(**kwargs)
                except Exception:
                    kwargs.pop("additionalModelRequestFields", None)
                    b_resp = b_client.converse(**kwargs)

                content = b_resp.get("output", {}).get("message", {}).get("content", [])
                text = next((block["text"] for block in content if isinstance(block, dict) and "text" in block), "")
                if text:
                    cleaned = self._sanitize_model_output(text)
                    if cleaned and len(cleaned) > 5:
                        return cleaned

            elif (backend == "sarvam" or api_key) and api_key:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1500,
                }
                headers = {
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                resp = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30.0)
                if resp.is_success:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = str(msg.get("content") or "").strip()
                        if content:
                            cleaned = self._sanitize_model_output(content)
                            is_conversational = bool(
                                re.search(
                                    r"(?:the user wants to know|i can mention|i should|i will|i think|if there are multiple|might be incomplete|found[\"\'”’\.]|looking at|from the relevant|to answer this question|let\'s)",
                                    cleaned,
                                    re.IGNORECASE,
                                )
                            )
                            if cleaned and len(cleaned) > 5 and not is_conversational:
                                return cleaned

                        # If content was empty or conversational, check reasoning_content
                        reasoning = str(msg.get("reasoning_content") or "").strip()
                        if reasoning:
                            m_draft = re.search(
                                r'(?:\*?Draft \d+:?\*?|final polish|Final Answer:?|In summary,?|The answer is:?)\s*([^\n]+)',
                                reasoning,
                                re.IGNORECASE,
                            )
                            if m_draft:
                                draft_cand = self._sanitize_model_output(m_draft.group(1).strip())
                                if draft_cand and len(draft_cand) > 5 and not draft_cand.endswith(":"):
                                    return draft_cand

                            # Also look for quoted final answer sentences in reasoning
                            quotes = re.findall(r'"([^"\n]{15,250})"', reasoning)
                            for q_text in reversed(quotes):
                                q_cleaned = self._sanitize_model_output(q_text)
                                if q_cleaned and len(q_cleaned) > 15 and not q_cleaned.endswith(":"):
                                    return q_cleaned
        except Exception as exc:
            logger.warning("graph.query_llm.error_falling_back_to_deterministic", error=str(exc))

        # Universal accurate deterministic fallback
        return self._deterministic_fallback(question, nodes, sources)
```

## File: src/ai/layer3_extraction/graph_agent/memory_manager.py
```python
"""
File: memory_manager.py
Purpose: Central Blackboard / Memory Manager for concurrent Layer 3 graph extraction.
Coordinates concurrent page workers, merges PageDeltas canonically,
remaps worker-local IDs to canonical IDs, rewrites edge endpoints,
and reconciles cross-page references post-merge.

Hardened with:
- False-merge protection (candidate generation, hard contradiction gates, heuristic scoring, ambiguity margins)
- Observation preservation and merge audit trails
- Explicit uncertainty states (ASSERTED, INFERRED, UNCERTAIN)
- Deterministic idempotency and checksum conflict detection
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
from src.ai.layer3_extraction.graph_agent.models import (
    CandidateEvaluation,
    Evidence,
    GraphEdge,
    GraphNode,
    MergeAuditRecord,
    PageDelta,
    PageDeltaValidationResult,
    PageProcessingState,
    ResolutionDecision,
    ResolutionStatus,
    TypeCompatibility,
    UnresolvedRef,
    validate_page_delta,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def getattr_settings(name: str, default):
    """Lazily read a settings value without a module-level import (avoids import cycles).
    Used by the deterministic cross-page linking pass (restore-linking task)."""
    try:
        from src.config.settings import settings as _s
        return getattr(_s, name, default)
    except Exception:
        return default


# Semantic keyword-to-anchor mapping covering medical, financial, legal, business domains
ANCHOR_KEYWORD_MAP: List[Tuple[re.Pattern, str]] = [
    # Healthcare
    (re.compile(r"(patient|subject|admittee)", re.I), "Patient"),
    (re.compile(r"(doctor|physician|surgeon|specialist|clinician)", re.I), "Doctor"),
    (re.compile(r"(hospital|clinic|facility|provider|institution)", re.I), "Hospital"),
    # Insurance & Claims
    (re.compile(r"(policy|coverage)", re.I), "Policy"),
    (re.compile(r"(claim|preauth)", re.I), "Claim"),
    # Financial / Banking / Commerce
    (re.compile(r"(account|bank|portfolio)", re.I), "Account"),
    (re.compile(r"(customer|client|buyer|purchaser)", re.I), "Customer"),
    (re.compile(r"(vendor|merchant|seller|supplier)", re.I), "Vendor"),
    (re.compile(r"(invoice|bill|receipt)", re.I), "Invoice"),
    (re.compile(r"(transaction|payment|disbursement|transfer)", re.I), "Transaction"),
    (re.compile(r"(order|purchase_order)", re.I), "Order"),
    # Legal / Contracts
    (re.compile(r"(plaintiff|claimant)", re.I), "Plaintiff"),
    (re.compile(r"(defendant|respondent)", re.I), "Defendant"),
    (re.compile(r"(case|litigation|lawsuit|matter)", re.I), "Case"),
    (re.compile(r"(contract|agreement|accord)", re.I), "Contract"),
    (re.compile(r"(court|tribunal|jurisdiction)", re.I), "Court"),
    # Corporate / Employment
    (re.compile(r"(employee|staff|personnel|worker)", re.I), "Employee"),
    (re.compile(r"(employer|organization|company|corporation|firm)", re.I), "Organization"),
]

# Type compatibility clusters
CLUSTER_PERSON: Set[str] = {
    "person", "doctor", "physician", "surgeon", "specialist", "clinician",
    "patient", "subject", "admittee", "employee", "staff", "worker",
    "customer", "client", "buyer", "purchaser", "plaintiff", "claimant",
    "defendant", "respondent", "user", "individual", "witness",
}

CLUSTER_ORGANIZATION: Set[str] = {
    "hospital", "clinic", "facility", "provider", "institution",
    "organization", "company", "corporation", "firm", "employer",
    "vendor", "merchant", "seller", "supplier", "court", "tribunal",
    "bank", "agency", "department",
}

CLUSTER_VALUE_CONCEPT: Set[str] = {
    "date", "time", "amount", "currency", "price", "fee", "cost",
    "identifier", "uhid", "ssn", "pan", "account_no", "policy_no",
    "admission", "record", "note", "diagnosis", "procedure",
}


def derive_anchor_types_from_schema(schema: type[BaseModel]) -> List[str]:
    """
    Derive anchor entity types dynamically from target schema fields using
    deterministic semantic heuristics across medical, financial, legal, etc. schemas.
    Does NOT hardcode healthcare-only concepts.
    """
    if not schema or not hasattr(schema, "model_fields"):
        return []

    derived: List[str] = []
    seen: Set[str] = set()

    for field_name in schema.model_fields.keys():
        clean_name = field_name.strip()
        matched = False
        for pattern, anchor_type in ANCHOR_KEYWORD_MAP:
            if pattern.search(clean_name):
                if anchor_type not in seen:
                    derived.append(anchor_type)
                    seen.add(anchor_type)
                matched = True
                break

        if not matched:
            m_name = re.match(r"^([a-zA-Z0-9]+)_(?:name|id|number|num|code)$", clean_name, re.I)
            if m_name:
                stem = m_name.group(1).capitalize()
                if len(stem) > 2 and stem not in seen:
                    derived.append(stem)
                    seen.add(stem)

    if not derived:
        for fname in list(schema.model_fields.keys())[:3]:
            cand = re.sub(r"[^a-zA-Z0-9]+", " ", fname).strip().title().replace(" ", "")
            if cand and cand not in seen:
                derived.append(cand)
                seen.add(cand)

    return derived


class GraphMemoryManager:
    """
    Central Blackboard / Memory Manager for Layer 3 extraction.

    Guarantees:
    - Candidate -> Verify -> Commit resolution pipeline.
    - LLM proposals (reused_node_id) never silently force canonical merges.
    - Hard contradiction gates block false merges even with high textual similarity.
    - Scores are explicit heuristic ranking scores, not calibrated probabilities.
    - Ambiguous candidates remain unresolved (status=UNCERTAIN).
    - Committed merges retain complete audit provenance and source observations.
    - Idempotent ingestion per document/page.
    """

    def __init__(
        self,
        anchor_types: Optional[List[str]] = None,
        anchor_max_k: int = 5,
        schema: Optional[type[BaseModel]] = None,
        match_threshold: float = 0.85,
        ambiguity_margin: float = 0.15,
        pipeline_version: str = "v1",
        algorithm_version: str = "entity_resolution_v2",
    ) -> None:
        self.graph = GraphMemory()
        self.anchor_max_k = max(1, anchor_max_k)
        self._schema = schema
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.pipeline_version = pipeline_version
        self.algorithm_version = algorithm_version

        if anchor_types:
            self.anchor_types = [a.strip() for a in anchor_types if a.strip()]
        elif schema is not None:
            self.anchor_types = derive_anchor_types_from_schema(schema)
        else:
            self.anchor_types = []

        self._unresolved: List[UnresolvedRef] = []
        self._lock = asyncio.Lock()
        # Scoped mapping: (page_number, local_id_or_key) -> canonical_id
        self._id_remap: Dict[Tuple[int, str], str] = {}
        # In-memory audit ledger of all canonical merges
        self._merge_ledger: List[MergeAuditRecord] = []
        # Idempotency tracking: page_number -> delta checksum
        self._ingested_deltas: Dict[int, str] = {}
        # Checksum conflict records: list of conflicts encountered
        self._checksum_conflicts: List[Dict[str, Any]] = []

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _gen_node_id(self, page_num: int, seq: int, prefix: str = "node") -> str:
        """Deterministic, content-free generated node id: {prefix}_{page}_{seq}
        (merge-determinism close-out, item 2). (page, seq) is unique across pages
        and carries NO document value in or behind the id. If the id already exists
        in the graph (collision with a reused raw_id), append a disambiguating counter
        and fail loudly in the log rather than silently overwrite."""
        base = f"{prefix}_{page_num}_{seq}"
        if base not in self.graph._nodes:
            return base
        n = 1
        while f"{base}_{n}" in self.graph._nodes:
            logger.warning("graph.node_id.collision", base=base, attempt=n)
            n += 1
        logger.warning("graph.node_id.collision_resolved", base=base, final=f"{base}_{n}")
        return f"{base}_{n}"

    def _gen_edge_id(self, page_num: int, seq: int, prefix: str = "edge") -> str:
        """Deterministic, content-free generated edge id: {prefix}_{page}_{seq}
        (merge-determinism close-out, item 2). Collision-checked against existing edges."""
        base = f"{prefix}_{page_num}_{seq}"
        if base not in self.graph._edges:
            return base
        n = 1
        while f"{base}_{n}" in self.graph._edges:
            logger.warning("graph.edge_id.collision", base=base, attempt=n)
            n += 1
        logger.warning("graph.edge_id.collision_resolved", base=base, final=f"{base}_{n}")
        return f"{base}_{n}"

    def clean_person_name(self, name: str) -> str:
        """Strip common honorifics and titles from person names."""
        norm = self._normalize(name)
        return re.sub(r"^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|surgeon:?|md|esq\.?)\s+", "", norm).strip()

    @property
    def merge_ledger(self) -> List[MergeAuditRecord]:
        return list(self._merge_ledger)

    def get_merge_ledger(self) -> List[MergeAuditRecord]:
        return list(self._merge_ledger)

    @property
    def checksum_conflicts(self) -> List[Dict[str, Any]]:
        return list(self._checksum_conflicts)

    def check_type_compatibility(self, type_a: str, type_b: str) -> TypeCompatibility:
        """
        Generic entity type compatibility check.
        Distinguishes COMPATIBLE, INCOMPATIBLE, and UNKNOWN without brittle single-string inequality.
        """
        a = self._normalize(type_a)
        b = self._normalize(type_b)

        if not a or not b or a == b or a == "entity" or b == "entity":
            return TypeCompatibility.COMPATIBLE

        # Cross-cluster incompatibilities (e.g. Person vs Hospital, Date vs Amount)
        in_person_a = a in CLUSTER_PERSON
        in_person_b = b in CLUSTER_PERSON
        in_org_a = a in CLUSTER_ORGANIZATION
        in_org_b = b in CLUSTER_ORGANIZATION
        in_val_a = a in CLUSTER_VALUE_CONCEPT
        in_val_b = b in CLUSTER_VALUE_CONCEPT

        if in_person_a and (in_org_b or in_val_b):
            return TypeCompatibility.INCOMPATIBLE
        if in_org_a and (in_person_b or in_val_b):
            return TypeCompatibility.INCOMPATIBLE
        if in_person_b and (in_org_a or in_val_a):
            return TypeCompatibility.INCOMPATIBLE
        if in_org_b and (in_person_a or in_val_a):
            return TypeCompatibility.INCOMPATIBLE

        # Intra-person explicit opposing roles
        doctor_roles = {"doctor", "physician", "surgeon", "specialist", "clinician"}
        patient_roles = {"patient", "subject", "admittee"}
        if (a in doctor_roles and b in patient_roles) or (a in patient_roles and b in doctor_roles):
            return TypeCompatibility.INCOMPATIBLE

        legal_plaintiff = {"plaintiff", "claimant"}
        legal_defendant = {"defendant", "respondent"}
        if (a in legal_plaintiff and b in legal_defendant) or (a in legal_defendant and b in legal_plaintiff):
            return TypeCompatibility.INCOMPATIBLE

        # Within same cluster
        if (in_person_a and in_person_b) or (in_org_a and in_org_b) or (in_val_a and in_val_b):
            return TypeCompatibility.COMPATIBLE

        return TypeCompatibility.UNKNOWN

    def check_hard_contradictions(
        self,
        incoming: GraphNode,
        candidate: GraphNode,
    ) -> Tuple[bool, List[str]]:
        """
        Deterministic hard contradiction gates evaluated BEFORE any candidate merge.
        Returns (has_contradiction, reasons).
        """
        reasons: List[str] = []

        # 1. Type compatibility gate
        compat = self.check_type_compatibility(incoming.type, candidate.type)
        if compat == TypeCompatibility.INCOMPATIBLE:
            reasons.append(
                f"INCOMPATIBLE_TYPES: incoming '{incoming.type}' is incompatible with candidate '{candidate.type}'"
            )

        # 2. Explicit conflicting identifiers gate (UHID, SSN, PAN, Account No, etc.).
        #    NOTE (audit #3): these property keys are NOT populated by the current
        #    Layer-3 agent (which writes only raw_id/schema_field_name/reused_node_id),
        #    so this gate is inert for the live pipeline. It is RETAINED (not removed)
        #    because it is a defensive, test-covered gate that works whenever a caller
        #    does populate the keys (see test_layer3_hardening fm_4/5/6). It is correct,
        #    just dormant for this pipeline's inputs.
        id_keys = (
            "uhid", "ssn", "pan", "account_no", "account_number",
            "policy_no", "policy_number", "case_number", "id_number",
            "identifier", "tax_id", "mrn", "ipd_no", "ref_no", "npi",
        )
        for key in id_keys:
            val_in = str(incoming.properties.get(key, "")).strip().lower()
            val_cand = str(candidate.properties.get(key, "")).strip().lower()
            if val_in and val_cand and val_in != val_cand:
                reasons.append(
                    f"CONFLICTING_IDENTIFIERS: '{key}' differs ('{val_in}' != '{val_cand}')"
                )

        # Identifier node value conflict (when entity itself is an identifier)
        incoming_is_id = (
            self._normalize(incoming.type) in ("identifier", "uhid", "ssn", "pan", "account", "policy")
            or "id" in self._normalize(incoming.label)
        )
        candidate_is_id = (
            self._normalize(candidate.type) in ("identifier", "uhid", "ssn", "pan", "account", "policy")
            or "id" in self._normalize(candidate.label)
        )
        if incoming_is_id and candidate_is_id:
            if self._normalize(incoming.value) != self._normalize(candidate.value):
                reasons.append(
                    f"CONFLICTING_IDENTIFIER_VALUES: '{incoming.value}' != '{candidate.value}'"
                )

        # 3. Strong identity attributes (DOB, Gender). Same note as above: dormant for
        #    the live pipeline (keys unpopulated) but retained as a correct defensive,
        #    test-covered gate.
        dob_in = (
            incoming.properties.get("dob")
            or incoming.properties.get("birth_date")
            or incoming.properties.get("date_of_birth")
        )
        dob_cand = (
            candidate.properties.get("dob")
            or candidate.properties.get("birth_date")
            or candidate.properties.get("date_of_birth")
        )
        if dob_in and dob_cand and str(dob_in).strip().lower() != str(dob_cand).strip().lower():
            reasons.append(f"CONFLICTING_DOB: '{dob_in}' != '{dob_cand}'")

        gender_in = incoming.properties.get("gender") or incoming.properties.get("sex")
        gender_cand = candidate.properties.get("gender") or candidate.properties.get("sex")
        if gender_in and gender_cand and str(gender_in).strip().lower() != str(gender_cand).strip().lower():
            reasons.append(f"CONFLICTING_GENDER: '{gender_in}' != '{gender_cand}'")

        # 4. Conflicting distinct person names (e.g. Arun Sharma vs Ramesh Sharma)
        val_in = self._normalize(incoming.value)
        val_cand = self._normalize(candidate.value)
        cand_aliases = [self._normalize(a) for a in candidate.aliases]
        in_aliases = [self._normalize(a) for a in incoming.aliases]

        # If incoming value is an explicit alias of candidate (or vice versa), it is not a contradiction
        is_alias_match = (val_in in cand_aliases) or (val_cand in in_aliases)

        incoming_person = self._normalize(incoming.type) in CLUSTER_PERSON
        candidate_person = self._normalize(candidate.type) in CLUSTER_PERSON
        if not is_alias_match and (incoming_person or candidate_person):
            clean_in = self.clean_person_name(incoming.value)
            clean_cand = self.clean_person_name(candidate.value)
            if clean_in and clean_cand:
                tokens_in = [t for t in clean_in.split() if len(t) > 0]
                tokens_cand = [t for t in clean_cand.split() if len(t) > 0]
                if len(tokens_in) >= 2 and len(tokens_cand) >= 2:
                    t_in_0 = tokens_in[0].rstrip(".")
                    t_cand_0 = tokens_cand[0].rstrip(".")
                    # An initial matching the first letter (e.g. "A." vs "Arun") is compatible
                    is_initial = (len(t_in_0) == 1 or len(t_cand_0) == 1) and (t_in_0[0] == t_cand_0[0])
                    if not is_initial and t_in_0 != t_cand_0:
                        reasons.append(
                            f"CONFLICTING_PERSON_NAMES: first name '{tokens_in[0]}' != '{tokens_cand[0]}' in '{clean_in}' vs '{clean_cand}'"
                        )

        return (len(reasons) > 0, reasons)

    # Node types with amount/date semantics for which a value match alone is NOT
    # sufficient to merge — their ROLE (label/tag) must also be compatible (audit #2).
    _AMOUNT_DATE_TYPES = frozenset({
        "amount", "money", "currency", "value", "quantity",
        "date", "datetime", "time", "period",
    })

    # Conservative label equivalence table (normalized). Only labels listed together
    # here are treated as the same role; everything else requires an EXACT normalized
    # label match. Unknown labels for amount/date types therefore do NOT merge unless
    # identical. Kept deliberately small and auditable.
    _LABEL_EQUIVALENCE = [
        {"requestedamount", "requested", "claimedamount", "claimed", "amountclaimed", "amountrequested"},
        {"sanctionedamount", "sanctioned", "approvedamount", "approved", "amountapproved", "amountsanctioned"},
        {"totalbillamount", "totalamount", "grossamount", "billamount", "totalhospitalbill"},
        {"admissiondate", "dateofadmission", "doa"},
        {"dischargedate", "dateofdischarge", "dod"},
    ]

    def _is_amount_date(self, node_type: str) -> bool:
        t = self._normalize(node_type)
        return any(k in t for k in self._AMOUNT_DATE_TYPES)

    def _labels_compatible(self, label_a: str, label_b: str) -> bool:
        """Conservative role compatibility: exact normalized match, or both labels in
        the same documented equivalence set. Unknown/empty-vs-set cases are NOT
        compatible (returns False) so they will not merge for amount/date types."""
        na = re.sub(r"[\s_\-]", "", self._normalize(label_a))
        nb = re.sub(r"[\s_\-]", "", self._normalize(label_b))
        if na and na == nb:
            return True
        for grp in self._LABEL_EQUIVALENCE:
            if na in grp and nb in grp:
                return True
        return False

    @staticmethod
    def _merge_schema_tags(target_node: GraphNode, incoming: GraphNode) -> None:
        """Audit #3: keep the UNION of schema_field_name tags on the merged node,
        sorted + deduplicated (deterministic), instead of dropping the incoming tag.
        Stores the union in properties['schema_field_names'] and keeps the scalar
        properties['schema_field_name'] = first sorted tag for back-compat with the
        resolver's existing exact-tag lookup."""
        tags = set()
        existing_union = target_node.properties.get("schema_field_names")
        if isinstance(existing_union, list):
            tags.update(t for t in existing_union if t)
        t_cur = str(target_node.properties.get("schema_field_name", "")).strip()
        if t_cur:
            tags.add(t_cur)
        t_in = str(incoming.properties.get("schema_field_name", "")).strip()
        if t_in:
            tags.add(t_in)
        if not tags:
            return
        union = sorted(tags)
        target_node.properties["schema_field_names"] = union
        target_node.properties["schema_field_name"] = union[0]

    def _merge_role_incompatible(self, incoming: GraphNode, candidate: GraphNode) -> Tuple[bool, str]:
        """Audit #2 gate: for amount/date-semantic nodes, block a value-based merge when
        labels are incompatible OR schema_field_name tags differ. Returns (block, reason).
        Only applies when the setting is on and at least one node is amount/date typed."""
        from src.config.settings import settings as _settings
        if not getattr(_settings, "graph_merge_label_gate", True):
            return (False, "")
        if not (self._is_amount_date(incoming.type) or self._is_amount_date(candidate.type)):
            return (False, "")
        # schema tag disagreement (both tagged, tags differ) blocks the merge
        tag_in = str(incoming.properties.get("schema_field_name", "")).strip()
        tag_cand = str(candidate.properties.get("schema_field_name", "")).strip()
        if tag_in and tag_cand and tag_in != tag_cand:
            return (True, f"TAG_MISMATCH:{tag_in}!={tag_cand}")
        # label incompatibility blocks the merge
        if not self._labels_compatible(incoming.label, candidate.label):
            return (True, "LABEL_INCOMPATIBLE")
        return (False, "")

    # ------------------------------------------------------------------ #
    # Deterministic post-merge cross-page linking (restore-linking task) #
    # ------------------------------------------------------------------ #

    # Normalized values that are too generic to key an identity link.
    _LINK_VALUE_STOPLIST = frozenset({
        "m", "f", "male", "female", "yes", "no", "true", "false", "na", "n/a",
        "none", "nil", "approved", "pending", "active", "inactive", "paid", "unpaid",
    })

    def _link_eligible_value(self, value: str) -> bool:
        """A node value is eligible to KEY a cross-page identity link only if it is not
        generic: at least min length, not in the stoplist, and (if pure digits) at least
        min length as well. Short codes / booleans / gender are excluded."""
        min_len = int(getattr_settings("graph_crosspage_link_min_value_len", 4))
        v = self._normalize(value)
        if not v or len(v) < min_len:
            return False
        if v in self._LINK_VALUE_STOPLIST:
            return False
        return True

    def _link_keys(self, node: GraphNode) -> set:
        """Normalized identity keys for a node: its normalized value plus normalized
        aliases, each paired with the normalized type. Amount/date types yield no keys
        (never linked). Generic values are excluded."""
        if self._is_amount_date(node.type):
            return set()
        ntype = self._normalize(node.type)
        keys = set()
        for raw in [node.value, *(node.aliases or [])]:
            if self._link_eligible_value(raw):
                keys.add((ntype, self._normalize(raw)))
        return keys

    def link_cross_page(self) -> int:
        """Add deterministic SAME_ENTITY_AS edges between same-type nodes on DIFFERENT
        pages that share a normalized identifier/name/alias. No model calls. Returns the
        number of links created. Deterministic: nodes and keys iterated in sorted order,
        fan-out capped, sequence-based edge ids. Never links amount/date nodes; never
        mutates/merges nodes. Idempotent (skips existing SAME_ENTITY_AS pairs).
        Counters recorded on self._link_stats for measurement."""
        max_fanout = int(getattr_settings("graph_crosspage_link_max_fanout", 3))
        created = 0
        skipped_amount_date = 0
        skipped_excluded_value = 0
        skipped_fanout = 0

        # Build key -> sorted list of node ids (deterministic).
        key_to_nodes: Dict[Tuple[str, str], List[str]] = {}
        for nid in sorted(self.graph._nodes.keys()):
            node = self.graph._nodes[nid]
            if self._is_amount_date(node.type):
                skipped_amount_date += 1
                continue
            node_keys = self._link_keys(node)
            if not node_keys:
                skipped_excluded_value += 1
                continue
            for k in node_keys:
                key_to_nodes.setdefault(k, [])
                if nid not in key_to_nodes[k]:
                    key_to_nodes[k].append(nid)

        fanout: Dict[str, int] = {}
        seq = 0
        for key in sorted(key_to_nodes.keys()):
            ids = sorted(key_to_nodes[key])
            if len(ids) < 2:
                continue
            # Only link nodes that actually sit on DIFFERENT pages.
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    na, nb = self.graph._nodes[a], self.graph._nodes[b]
                    pa, pb = set(na.source_pages or []), set(nb.source_pages or [])
                    if not (pa and pb and pa.isdisjoint(pb)):
                        continue  # same page (or page-set overlap) -> not a cross-page link
                    # fan-out cap (deterministic over sorted ids)
                    if fanout.get(a, 0) >= max_fanout or fanout.get(b, 0) >= max_fanout:
                        skipped_fanout += 1
                        continue
                    # idempotency: skip if a SAME_ENTITY_AS edge already exists either way
                    if self.graph.get_edges(source_node=a, target_node=b, relationship="SAME_ENTITY_AS") or \
                       self.graph.get_edges(source_node=b, target_node=a, relationship="SAME_ENTITY_AS"):
                        continue
                    edge_id = f"edge_link_{seq}"
                    seq += 1
                    link_edge = GraphEdge(
                        id=edge_id,
                        source_node=a,
                        target_node=b,
                        relationship="SAME_ENTITY_AS",
                        evidence="",  # derived; no document text
                        source_page=min(min(pa), min(pb)),
                        confidence=1.0,
                        status="DERIVED",
                        properties={
                            "derived": True,
                            "rule": "same_entity_normalized_value_or_alias_and_type",
                        },
                    )
                    try:
                        self.graph.create_edge(link_edge)
                        created += 1
                        fanout[a] = fanout.get(a, 0) + 1
                        fanout[b] = fanout.get(b, 0) + 1
                    except Exception as exc:
                        logger.debug("graph.crosspage_link.skip", error=str(exc))

        self._link_stats = {
            "created": created,
            "skipped_amount_date_nodes": skipped_amount_date,
            "skipped_excluded_value_nodes": skipped_excluded_value,
            "skipped_fanout": skipped_fanout,
        }
        logger.info(
            "graph.crosspage_link.done",
            links_created=created,
            skipped_amount_date_nodes=skipped_amount_date,
            skipped_excluded_value_nodes=skipped_excluded_value,
            skipped_fanout=skipped_fanout,
        )
        return created

    def score_candidate(
        self,
        incoming: GraphNode,
        candidate: GraphNode,
        proposed_reused_id: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        """
        Calculate a heuristic candidate ranking score [0.0 - 1.0].
        Explicitly a heuristic ranking score, NOT a calibrated probability.
        """
        val_in = self._normalize(incoming.value)
        val_cand = self._normalize(candidate.value)
        signals: List[str] = []
        score = 0.0

        # Exact normalized match
        if val_in == val_cand:
            score = 1.0
            signals.append("EXACT_VALUE")
        # Alias match
        elif (
            val_in in [self._normalize(a) for a in candidate.aliases]
            or any(self._normalize(a) == val_cand for a in incoming.aliases)
        ):
            score = 0.95
            signals.append("ALIAS_MATCH")
        # Clean person name match (honorifics stripped)
        elif self.clean_person_name(incoming.value) == self.clean_person_name(candidate.value) and len(self.clean_person_name(incoming.value)) > 3:
            score = 0.90
            signals.append("NORMALIZED_NAME_MATCH")
        # Substring / containment match
        elif (val_in in val_cand or val_cand in val_in) and min(len(val_in), len(val_cand)) > 3:
            score = 0.75
            signals.append("SUBSTRING_MATCH")
        else:
            # Token overlap / containment heuristic
            tok_in = set(val_in.split())
            tok_cand = set(val_cand.split())
            if tok_in and tok_cand:
                inter = tok_in & tok_cand
                if tok_in.issubset(tok_cand) or tok_cand.issubset(tok_in):
                    # e.g. "Robert Smith" in "Robert James Smith"
                    score = 0.85
                    signals.append("TOKEN_CONTAINMENT")
                elif inter:
                    union = tok_in | tok_cand
                    jaccard = len(inter) / len(union)
                    score = 0.50 + (0.35 * jaccard)
                    signals.append(f"TOKEN_OVERLAP_{round(jaccard, 2)}")

        # Signal boost: LLM suggested this node as candidate
        if proposed_reused_id and candidate.id == proposed_reused_id:
            score = min(1.0, score + 0.05)
            signals.append("LLM_PROPOSED_REUSE")

        # Signal boost: Exact type match
        if self._normalize(incoming.type) == self._normalize(candidate.type):
            score = min(1.0, score + 0.05)
            signals.append("EXACT_TYPE_MATCH")

        return (round(min(1.0, max(0.0, score)), 4), signals)

    def resolve_entity(
        self,
        incoming: GraphNode,
        page_num: int,
    ) -> ResolutionDecision:
        """
        Execute Candidate Generation -> Hard Contradiction Gates -> Scoring -> Margin Evaluation -> Decision.
        Does NOT mutate the canonical graph itself.
        """
        raw_val = (incoming.value or "").strip()
        if not raw_val:
            return ResolutionDecision(
                status=ResolutionStatus.NEW_ENTITY,
                rationale="Empty value, cannot resolve",
                algorithm_version=self.algorithm_version,
            )

        val_norm = self._normalize(raw_val)
        proposed_reused_id = incoming.properties.get("reused_node_id")

        # ---------------------------------------------------------
        # Step A: Candidate Generation
        # ---------------------------------------------------------
        candidate_nodes: Dict[str, GraphNode] = {}

        # 1. Proposed reused_node_id from LLM (examined as a candidate only!)
        if proposed_reused_id and proposed_reused_id in self.graph._nodes:
            cand = self.graph.get_node(proposed_reused_id)
            if cand:
                candidate_nodes[cand.id] = cand

        # 2. Inverted index alias lookup
        if val_norm in self.graph._alias_index:
            for cid in self.graph._alias_index[val_norm]:
                cand = self.graph.get_node(cid)
                if cand:
                    candidate_nodes[cand.id] = cand

        for a in incoming.aliases:
            a_norm = self._normalize(a)
            if a_norm in self.graph._alias_index:
                for cid in self.graph._alias_index[a_norm]:
                    cand = self.graph.get_node(cid)
                    if cand:
                        candidate_nodes[cand.id] = cand

        # 3. Existing reuse candidate logic
        reuse_cand = self.graph.find_reuse_candidate(
            node_type=incoming.type,
            value=incoming.value,
            label=incoming.label,
        )
        if reuse_cand and reuse_cand.id:
            candidate_nodes[reuse_cand.id] = reuse_cand

        # 4. Search nodes by query
        search_cands = self.graph.search_nodes(query=raw_val, limit=5)
        for cand in search_cands:
            candidate_nodes[cand.id] = cand

        # 5. Search by tokens if multi-word query (e.g. "Robert Smith" -> "Robert", "Smith")
        words = [w for w in val_norm.split() if len(w) > 3]
        for w in words:
            for cand in self.graph.search_nodes(query=w, limit=5):
                candidate_nodes[cand.id] = cand

        logger.debug(
            "entity_resolution.candidate_generated",
            incoming_id=incoming.id,
            incoming_value=raw_val,
            candidate_count=len(candidate_nodes),
            candidate_ids=list(candidate_nodes.keys()),
        )

        if not candidate_nodes:
            return ResolutionDecision(
                status=ResolutionStatus.NEW_ENTITY,
                rationale="No candidate nodes found in existing memory",
                algorithm_version=self.algorithm_version,
            )

        # ---------------------------------------------------------
        # Step B & C: Contradiction Gates & Heuristic Scoring
        # ---------------------------------------------------------
        evaluations: List[CandidateEvaluation] = []
        for cand_id, cand_node in candidate_nodes.items():
            has_contradiction, reasons = self.check_hard_contradictions(incoming, cand_node)
            # Audit #2 label/role-compatibility gate: for amount/date-semantic nodes,
            # an incompatible label or a differing schema tag is a hard block, applied
            # alongside the existing contradiction gates (does NOT touch scoring/tiebreak).
            role_block, role_reason = self._merge_role_incompatible(incoming, cand_node)
            if role_block:
                has_contradiction = True
                reasons = list(reasons) + [role_reason]
            score, signals = self.score_candidate(incoming, cand_node, proposed_reused_id)
            compat = self.check_type_compatibility(incoming.type, cand_node.type)

            evaluations.append(
                CandidateEvaluation(
                    candidate_id=cand_id,
                    candidate_node=cand_node,
                    candidate_score=score,
                    matching_signals=signals,
                    contradiction_detected=has_contradiction,
                    contradiction_reasons=reasons,
                    compatibility=compat.value,
                )
            )

        # Filter out candidates blocked by hard contradictions
        valid_evals = [e for e in evaluations if not e.contradiction_detected]
        contradictory_evals = [e for e in evaluations if e.contradiction_detected]

        for c_eval in contradictory_evals:
            logger.info(
                "entity_resolution.contradiction",
                incoming_id=incoming.id,
                incoming_value=incoming.value,
                candidate_id=c_eval.candidate_id,
                candidate_value=c_eval.candidate_node.value,
                candidate_score=c_eval.candidate_score,
                reasons=c_eval.contradiction_reasons,
            )

        if not valid_evals:
            # All candidates blocked by hard contradictions
            all_reasons = [r for c in contradictory_evals for r in c.contradiction_reasons]
            return ResolutionDecision(
                status=ResolutionStatus.REJECTED_CONTRADICTION,
                all_candidates=evaluations,
                contradiction_detected=True,
                contradiction_reasons=all_reasons,
                rationale="All candidate matches rejected by hard contradiction gates",
                algorithm_version=self.algorithm_version,
            )

        # ---------------------------------------------------------
        # Step D: Ambiguity Margin & Decision Evaluation
        # ---------------------------------------------------------
        valid_evals.sort(key=lambda x: x.candidate_score, reverse=True)
        top = valid_evals[0]
        second = valid_evals[1] if len(valid_evals) > 1 else None

        top_score = top.candidate_score
        second_score = second.candidate_score if second else 0.0
        decision_margin = round(top_score - second_score, 4)

        if top_score < self.match_threshold:
            if top_score >= 0.50:
                # Moderate score but below match threshold -> leave uncertain
                return ResolutionDecision(
                    status=ResolutionStatus.UNCERTAIN_MATCH,
                    winning_candidate=top,
                    all_candidates=evaluations,
                    candidate_score=top_score,
                    second_candidate_score=second_score,
                    decision_margin=decision_margin,
                    matching_signals=top.matching_signals,
                    resolution_method="SCORE_BELOW_THRESHOLD",
                    rationale=f"Top candidate score {top_score} below match threshold {self.match_threshold}",
                    algorithm_version=self.algorithm_version,
                )
            else:
                return ResolutionDecision(
                    status=ResolutionStatus.NEW_ENTITY,
                    winning_candidate=None,
                    all_candidates=evaluations,
                    candidate_score=top_score,
                    second_candidate_score=second_score,
                    decision_margin=decision_margin,
                    matching_signals=top.matching_signals,
                    resolution_method="LOW_SIMILARITY",
                    rationale=f"Top candidate score {top_score} indicates distinct entity",
                    algorithm_version=self.algorithm_version,
                )

        # Top score is >= match_threshold; check decision margin
        if second is not None and decision_margin < self.ambiguity_margin:
            # Ambiguous close candidates
            return ResolutionDecision(
                status=ResolutionStatus.UNCERTAIN_MATCH,
                winning_candidate=top,
                all_candidates=evaluations,
                candidate_score=top_score,
                second_candidate_score=second_score,
                decision_margin=decision_margin,
                matching_signals=top.matching_signals,
                resolution_method="INSUFFICIENT_MARGIN",
                rationale=f"Decision margin {decision_margin} below required ambiguity margin {self.ambiguity_margin}",
                algorithm_version=self.algorithm_version,
            )

        # Confirmed match
        method = top.matching_signals[0] if top.matching_signals else "CONFIRMED_HEURISTIC"
        return ResolutionDecision(
            status=ResolutionStatus.CONFIRMED_MATCH,
            canonical_id=top.candidate_id,
            winning_candidate=top,
            all_candidates=evaluations,
            candidate_score=top_score,
            second_candidate_score=second_score,
            decision_margin=decision_margin,
            matching_signals=top.matching_signals,
            resolution_method=method,
            rationale=f"Confirmed match to {top.candidate_id} with score {top_score} and margin {decision_margin}",
            algorithm_version=self.algorithm_version,
        )

    async def get_context_for_page(self, page_number: int) -> List[Dict[str, Any]]:
        """
        Return targeted anchor context only, rather than dumping the entire graph.
        Anchor ranking considers:
        1. Configured/derived anchor type match
        2. Node category (EXPLICIT > CONTEXTUAL > REFERENCE_SUPPORT)
        3. Node confidence
        4. Page coverage / proximity to page_number
        """
        async with self._lock:
            if not self.graph._nodes:
                return []

            norm_anchors = {self._normalize(a) for a in self.anchor_types}
            candidates: List[Tuple[float, GraphNode]] = []

            for node in self.graph._nodes.values():
                type_norm = self._normalize(node.type)
                label_norm = self._normalize(node.label)

                is_anchor_match = (
                    (type_norm in norm_anchors)
                    or (label_norm in norm_anchors)
                    or any(a in type_norm or a in label_norm for a in norm_anchors)
                ) if norm_anchors else True

                score = 0.0
                if is_anchor_match:
                    score += 10.0

                if node.category == "EXPLICIT":
                    score += 5.0
                elif node.category == "CONTEXTUAL":
                    score += 2.0

                score += node.confidence * 2.0

                if node.source_pages:
                    max_p = max(node.source_pages)
                    page_diff = abs(page_number - max_p)
                    score += max(0.0, 3.0 - (page_diff * 0.5))

                candidates.append((score, node))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top_nodes = [node for _, node in candidates[:self.anchor_max_k]]
            return [n.to_dict() for n in top_nodes]

    async def ingest_page_delta(
        self,
        delta: PageDelta,
        delta_checksum: Optional[str] = None,
    ) -> bool:
        """
        Canonically ingest a PageDelta into the central GraphMemory under lock.
        1. Checks idempotency: if page was already ingested with the same checksum, safely no-ops.
        2. Resolves entities via candidate -> contradiction check -> score -> margin -> decision pipeline.
        3. Fuses confirmed entities while preserving source pages, evidence, and audit logs.
        4. Maintains scoped (page_number, local_id) -> canonical_id mapping.
        5. Rewrites relationship endpoints to canonical IDs.
        6. Collects unresolved references for post-merge resolution.
        """
        async with self._lock:
            page_num = delta.page_number

            # Check idempotency
            if page_num in self._ingested_deltas:
                stored_checksum = self._ingested_deltas[page_num]
                if delta_checksum and stored_checksum and delta_checksum == stored_checksum:
                    logger.info(
                        "page.retry.idempotent_noop",
                        page_number=page_num,
                        checksum=delta_checksum,
                    )
                    return True
                elif delta_checksum and stored_checksum and delta_checksum != stored_checksum:
                    self._checksum_conflicts.append({
                        "page_number": page_num,
                        "stored_checksum": stored_checksum,
                        "incoming_checksum": delta_checksum,
                    })
                    logger.warning(
                        "page.retry.checksum_conflict",
                        page_number=page_num,
                        stored_checksum=stored_checksum,
                        incoming_checksum=delta_checksum,
                    )
                    return False

            # ---------------------------------------------------------
            # Step 1: Ingest and Canonicalize Entities (Nodes)
            # ---------------------------------------------------------
            # Per-delta sequence for deterministic, content-free generated ids
            # (merge-determinism close-out, item 2): a generated node id is
            # node_{page}_{seq}, where seq is the entity's position in this delta.
            # (page, seq) is unique across pages (page namespaces it) and carries no
            # document value in or behind the id.
            for _ent_seq, ent in enumerate(delta.entities):
                raw_id = ent.id
                ent_val = (ent.value or "").strip()
                if not ent_val:
                    continue

                # Run resolution pipeline
                decision = self.resolve_entity(ent, page_num=page_num)
                canonical_id: Optional[str] = None

                if decision.status == ResolutionStatus.CONFIRMED_MATCH and decision.canonical_id:
                    canonical_id = decision.canonical_id
                    # Fused merge: preserve source pages, evidence, aliases, observations
                    target_node = self.graph.get_node(canonical_id)
                    if target_node:
                        target_node.add_page(page_num)
                        for ev in ent.evidence:
                            target_node.add_evidence(ev.page_number, ev.text, ev.confidence)
                        for al in ent.aliases:
                            target_node.add_alias(al)
                        if ent.category == "EXPLICIT":
                            target_node.category = "EXPLICIT"

                        # Promote UNCERTAIN node to ASSERTED if incoming evidence is ASSERTED
                        if target_node.status == "UNCERTAIN" and ent.status == "ASSERTED":
                            target_node.status = "ASSERTED"
                            # If incoming has a longer/more complete value, update canonical value and register old as alias
                            if len(ent.value) > len(target_node.value):
                                old_val = target_node.value
                                self.graph.update_node(target_node.id, {"value": ent.value})
                                target_node.add_alias(old_val)

                        # Audit #3: keep the UNION of schema_field_name tags (sorted,
                        # deterministic) instead of silently dropping the incoming one.
                        self._merge_schema_tags(target_node, ent)

                        # Preserve raw observation
                        obs_list = target_node.properties.setdefault("observations", [])
                        obs_list.append({
                            "raw_id": raw_id,
                            "page_number": page_num,
                            "value": ent.value,
                            "label": ent.label,
                            "evidence": [e.to_dict() for e in ent.evidence],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                        # Update confidence metrics
                        target_node.resolution_confidence = decision.candidate_score
                        target_node.extraction_confidence = min(target_node.extraction_confidence, ent.extraction_confidence)
                        target_node.confidence = max(target_node.confidence, ent.confidence)

                        # Record audit
                        audit_rec = MergeAuditRecord(
                            source_node_id=raw_id,
                            source_page=page_num,
                            source_value=ent.value,
                            canonical_node_id=canonical_id,
                            canonical_value=target_node.value,
                            resolution_method=decision.resolution_method,
                            candidate_score=decision.candidate_score,
                            second_candidate_score=decision.second_candidate_score,
                            decision_margin=decision.decision_margin,
                            matching_signals=decision.matching_signals,
                            contradiction_result="PASSED_HARD_GATES",
                            contradiction_reasons=[],
                            source_pages=list(target_node.source_pages),
                            evidence=[e.to_dict() for e in target_node.evidence],
                            algorithm_version=self.algorithm_version,
                            pipeline_version=self.pipeline_version,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        self._merge_ledger.append(audit_rec)
                        target_node.properties.setdefault("merge_audit", []).append(audit_rec.to_dict())

                        logger.info(
                            "entity_resolution.confirmed",
                            source_id=raw_id,
                            source_value=ent.value,
                            canonical_id=canonical_id,
                            method=decision.resolution_method,
                            score=decision.candidate_score,
                            margin=decision.decision_margin,
                        )

                elif decision.status == ResolutionStatus.UNCERTAIN_MATCH:
                    # Ambiguous / uncertain match: do NOT merge into candidate
                    node_id = self._gen_node_id(page_num, _ent_seq, prefix="node_unc")
                    pages = list(ent.source_pages) if ent.source_pages else [page_num]
                    props = dict(ent.properties)
                    props["raw_id"] = raw_id
                    props["uncertain_candidate_id"] = (
                        decision.winning_candidate.candidate_id if decision.winning_candidate else None
                    )
                    props["candidate_score"] = decision.candidate_score
                    props["decision_margin"] = decision.decision_margin
                    props["rationale"] = decision.rationale

                    unc_node = GraphNode(
                        id=node_id,
                        type=ent.type,
                        label=ent.label,
                        value=ent_val,
                        properties=props,
                        aliases=list(ent.aliases),
                        evidence=list(ent.evidence),
                        source_pages=pages,
                        confidence=ent.confidence,
                        status="UNCERTAIN",
                        category=ent.category or "CONTEXTUAL",
                        extraction_confidence=ent.extraction_confidence,
                        resolution_confidence=decision.candidate_score,
                        evidence_strength=ent.evidence_strength,
                    )
                    self.graph.create_node(unc_node)
                    canonical_id = node_id

                    logger.info(
                        "entity_resolution.uncertain",
                        source_id=raw_id,
                        source_value=ent.value,
                        created_node_id=node_id,
                        competing_candidate=(
                            decision.winning_candidate.candidate_id if decision.winning_candidate else None
                        ),
                        margin=decision.decision_margin,
                    )

                else:
                    # NEW_ENTITY or REJECTED_CONTRADICTION
                    if raw_id and raw_id not in self.graph._nodes:
                        node_id = raw_id
                    else:
                        node_id = self._gen_node_id(page_num, _ent_seq, prefix="node")

                    pages = list(ent.source_pages) if ent.source_pages else [page_num]
                    props = dict(ent.properties)
                    props["raw_id"] = raw_id
                    if decision.status == ResolutionStatus.REJECTED_CONTRADICTION:
                        props["contradiction_reasons"] = decision.contradiction_reasons

                    new_node = GraphNode(
                        id=node_id,
                        type=ent.type,
                        label=ent.label,
                        value=ent_val,
                        properties=props,
                        aliases=list(ent.aliases),
                        evidence=list(ent.evidence),
                        source_pages=pages,
                        confidence=ent.confidence,
                        status=ent.status or "ASSERTED",
                        category=ent.category or "CONTEXTUAL",
                        extraction_confidence=ent.extraction_confidence,
                        resolution_confidence=None,  # NEW_ENTITY does not imply identity resolution against an existing entity
                        evidence_strength=ent.evidence_strength,
                    )
                    self.graph.create_node(new_node)
                    canonical_id = node_id

                    logger.info(
                        "entity_resolution.new_entity",
                        node_id=node_id,
                        type=ent.type,
                        value=ent_val,
                        status=new_node.status,
                        contradiction_detected=decision.contradiction_detected,
                    )

                # Register scoped mappings for this page worker
                self._id_remap[(page_num, raw_id)] = canonical_id
                self._id_remap[(page_num, self._normalize(raw_id))] = canonical_id
                self._id_remap[(page_num, self._normalize(ent_val))] = canonical_id
                if ent.label:
                    self._id_remap[(page_num, self._normalize(ent.label))] = canonical_id

                for a in ent.aliases:
                    if a:
                        self.graph.add_alias(canonical_id, a)
                        self._id_remap[(page_num, self._normalize(a))] = canonical_id

            # Incorporate worker's raw_id_map
            for raw_k, local_val in delta.raw_id_map.items():
                target_canonical = (
                    self._id_remap.get((page_num, local_val))
                    or self._id_remap.get((page_num, self._normalize(local_val)))
                )
                if target_canonical:
                    self._id_remap[(page_num, raw_k)] = target_canonical
                    self._id_remap[(page_num, self._normalize(raw_k))] = target_canonical

            # ---------------------------------------------------------
            # Step 2: Ingest and Rewrite Relationships (Edges)
            # ---------------------------------------------------------
            for _edge_seq, edge in enumerate(delta.relationships):
                raw_src = (edge.source_node or "").strip()
                raw_tgt = (edge.target_node or "").strip()
                if not raw_src or not raw_tgt:
                    continue

                canon_src = (
                    self._id_remap.get((page_num, raw_src))
                    or self._id_remap.get((page_num, self._normalize(raw_src)))
                )
                if not canon_src and raw_src in self.graph._nodes:
                    canon_src = raw_src

                canon_tgt = (
                    self._id_remap.get((page_num, raw_tgt))
                    or self._id_remap.get((page_num, self._normalize(raw_tgt)))
                )
                if not canon_tgt and raw_tgt in self.graph._nodes:
                    canon_tgt = raw_tgt

                if not canon_src or not canon_tgt:
                    logger.warning(
                        "graph.relationship.endpoint_unresolved",
                        page_number=page_num,
                        relationship=edge.relationship,
                        raw_src=raw_src,
                        raw_tgt=raw_tgt,
                        resolved_src=canon_src,
                        resolved_tgt=canon_tgt,
                    )
                    continue

                if canon_src not in self.graph._nodes or canon_tgt not in self.graph._nodes:
                    logger.warning(
                        "graph.relationship.endpoint_not_in_graph",
                        page_number=page_num,
                        relationship=edge.relationship,
                        raw_src=raw_src,
                        raw_tgt=raw_tgt,
                        resolved_src=canon_src,
                        resolved_tgt=canon_tgt,
                    )
                    continue

                if canon_src == canon_tgt:
                    logger.debug(
                        "graph.relationship.reflexive_skipped",
                        page_number=page_num,
                        node_id=canon_src,
                        relationship=edge.relationship,
                    )
                    continue

                # Deduplicate: avoid creating duplicate edges on page retries
                existing_edges = self.graph.get_edges(
                    source_node=canon_src,
                    target_node=canon_tgt,
                    relationship=edge.relationship,
                )
                duplicate = any(e.source_page == page_num for e in existing_edges)
                if not duplicate:
                    edge_id = self._gen_edge_id(page_num, _edge_seq, prefix="edge")
                    canon_edge = GraphEdge(
                        id=edge_id,
                        source_node=canon_src,
                        target_node=canon_tgt,
                        relationship=edge.relationship or "RELATED_TO",
                        evidence=edge.evidence or "",
                        source_page=page_num,
                        confidence=edge.confidence,
                        status=edge.status or "ASSERTED",
                        properties=dict(edge.properties),
                        extraction_confidence=edge.extraction_confidence,
                        relationship_confidence=edge.relationship_confidence,
                        evidence_strength=edge.evidence_strength,
                    )
                    try:
                        self.graph.create_edge(canon_edge)
                    except Exception as exc:
                        logger.debug("manager.edge_creation_skipped", error=str(exc))

            # ---------------------------------------------------------
            # Step 3: Collect Unresolved References for Post-Merge
            # ---------------------------------------------------------
            self._unresolved.extend(delta.unresolved_references)
            self._ingested_deltas[page_num] = delta_checksum or ""
            return True

    async def reconcile_cross_references(self) -> None:
        """
        Reconcile cross-page anaphoric references post-merge.
        Ranks candidates deterministically using target type, page proximity,
        confidence, and category, and enforces hard contradiction and ambiguity checks.
        Ambiguous cases are left unresolved.
        """
        async with self._lock:
            if not self._unresolved:
                return

            logger.info("graph.reconciliation.started", unresolved_count=len(self._unresolved))

            remaining_unresolved: List[UnresolvedRef] = []

            for _ref_seq, ref in enumerate(self._unresolved):
                phrase = (ref.phrase or "").strip()
                target_type = (ref.target_type or "").strip()
                page_num = ref.page_number

                if not phrase and not target_type:
                    remaining_unresolved.append(ref)
                    continue

                norm_tgt_type = self._normalize(target_type)
                candidates: List[Tuple[float, GraphNode]] = []

                for node in self.graph._nodes.values():
                    node_type_norm = self._normalize(node.type)
                    node_label_norm = self._normalize(node.label)

                    type_compat = self.check_type_compatibility(target_type, node.type)
                    if type_compat == TypeCompatibility.INCOMPATIBLE:
                        continue

                    type_match = False
                    if norm_tgt_type:
                        if norm_tgt_type in (node_type_norm, node_label_norm):
                            type_match = True
                        elif type_compat == TypeCompatibility.COMPATIBLE:
                            type_match = True
                    else:
                        type_match = True

                    if not type_match:
                        continue

                    score = 1.0
                    pages_before_or_at = [p for p in node.source_pages if p <= page_num]
                    if pages_before_or_at:
                        closest_page = max(pages_before_or_at)
                        page_distance = page_num - closest_page
                        score += 3.0 / (1.0 + page_distance)
                    else:
                        score -= 2.0

                    if node.category == "EXPLICIT":
                        score += 2.0
                    elif node.category == "CONTEXTUAL":
                        score += 1.0

                    score += node.confidence
                    candidates.append((score, node))

                if not candidates:
                    logger.info("graph.reconciliation.unresolved", phrase=phrase, reason="no_candidates")
                    remaining_unresolved.append(ref)
                    continue

                candidates.sort(key=lambda x: x[0], reverse=True)
                top_score, best_candidate = candidates[0]

                # Ambiguity check
                if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < self.ambiguity_margin:
                    logger.info(
                        "graph.reconciliation.unresolved",
                        phrase=phrase,
                        target_type=target_type,
                        reason="ambiguous_candidates",
                        cand1=best_candidate.id,
                        cand2=candidates[1][1].id,
                    )
                    remaining_unresolved.append(ref)
                    continue

                if top_score < 1.5:
                    logger.info(
                        "graph.reconciliation.unresolved",
                        phrase=phrase,
                        target_type=target_type,
                        reason="low_confidence",
                        score=top_score,
                    )
                    remaining_unresolved.append(ref)
                    continue

                canon_src = (
                    self._id_remap.get((page_num, ref.source_node_id))
                    or self._id_remap.get((page_num, self._normalize(ref.source_node_id)))
                    or ref.source_node_id
                )

                if canon_src and canon_src in self.graph._nodes and canon_src != best_candidate.id:
                    existing_edges = self.graph.get_edges(
                        source_node=canon_src,
                        target_node=best_candidate.id,
                        relationship="REFERS_TO",
                    )
                    duplicate = any(e.source_page == page_num for e in existing_edges)
                    if not duplicate:
                        ref_edge = GraphEdge(
                            id=self._gen_edge_id(page_num, _ref_seq, prefix="edge_ref"),
                            source_node=canon_src,
                            target_node=best_candidate.id,
                            relationship="REFERS_TO",
                            evidence=ref.rationale or phrase,
                            source_page=page_num,
                            confidence=round(min(1.0, top_score / 5.0), 3),
                            status="INFERRED",
                            relationship_confidence=round(min(1.0, top_score / 5.0), 3),
                            extraction_confidence=1.0,
                        )
                        self.graph.create_edge(ref_edge)
                elif not (canon_src and canon_src in self.graph._nodes):
                    remaining_unresolved.append(ref)
                    continue

                if phrase:
                    self.graph.add_alias(best_candidate.id, phrase)

                logger.info(
                    "graph.reconciliation.resolved",
                    phrase=phrase,
                    target_type=target_type,
                    resolved_node=best_candidate.id,
                    target_value=best_candidate.value,
                    score=top_score,
                )

            self._unresolved = remaining_unresolved

    @property
    def unresolved_count(self) -> int:
        return len(self._unresolved)
```

## File: src/ai/layer3_extraction/storage.py
```python
"""
File: storage.py
Purpose: PostgreSQL storage schema for IDP System.

  Tables:
    1. documents:
       Stores uploaded input documents (e.g. PDF binaries, size, hash, metadata).
    2. schemas:
       Stores confirmed target schemas (JSONB specification, fields, document type).
    3. document_markdowns:
       Stores converted Markdown outputs (.md) produced by Layer 1 & 2 conversion.
    4. extraction_runs:
       Stores structured field extraction results produced by Layer 3.

Owner: engineer-b@idp-pilot & engineer-a@idp-pilot
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    JSON,
    create_engine,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reserved sentinel owner values for system isolation
SENTINEL_LEGACY = "_legacy"
SENTINEL_ADMIN = "_admin"


class Base(DeclarativeBase):
    pass


def _normalize_pg_url(db_url: str) -> str:
    """
    Pin the PostgreSQL dialect to psycopg2, which is the driver installed in
    the image. SQLAlchemy 2.1 changed the default driver for a bare
    'postgresql://' URL to psycopg (v3); without an explicit '+psycopg2' the
    engine would try to import psycopg (v3), fail, and silently fall back to
    SQLite. Normalizing here keeps existing DATABASE_URL env values working.
    """
    if db_url.startswith("postgresql+"):
        return db_url  # driver already specified explicitly
    if db_url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + db_url[len("postgresql://"):]
    if db_url.startswith("postgres://"):
        return "postgresql+psycopg2://" + db_url[len("postgres://"):]
    return db_url


def _create_db_engine():
    db_url = getattr(settings, "database_url", "sqlite:///./idp_storage.db")
    try:
        if db_url.startswith("postgres"):
            db_url = _normalize_pg_url(db_url)
            eng = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            # Ping test connection
            with eng.connect() as conn:
                pass
            return eng
        return create_engine(db_url)
    except Exception as exc:
        logger.warning("storage.engine_init_fallback", error=str(exc))
        return create_engine("sqlite:///./idp_storage.db")


engine = _create_db_engine()
SessionLocal = sessionmaker(bind=engine)


def get_engine():
    return engine


def get_session():
    return SessionLocal()


# ==============================================================================
# Table 1: Uploaded Input Documents (PDFs)
# Shadow copy for input documents; non-authoritative when S3 is configured.
# Keyed by (owner, filename).
# ==============================================================================
class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("owner", "filename", name="uq_documents_owner_filename"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String, default="application/pdf")
    file_hash: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    file_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 2: Confirmed Target Schemas
# ==============================================================================
class SchemaRecord(Base):
    __tablename__ = "schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    document_type: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    owner: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    schema_name: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    sample_documents: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 3: Converted Markdown Outputs (.md)
# Keyed by (owner, doc_id).
# ==============================================================================
class MarkdownRecord(Base):
    __tablename__ = "document_markdowns"
    __table_args__ = (
        UniqueConstraint("owner", "doc_id", name="uq_document_markdowns_owner_doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    md_filename: Mapped[str] = mapped_column(String, nullable=False)
    markdown_content: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    schema_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    schema_ref_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pages_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 4: Structured Field Extraction Runs (Layer 3)
# Indexed by (owner, doc_id).
# ==============================================================================
class ExtractionRun(Base):
    __tablename__ = "extraction_runs"
    __table_args__ = (
        Index("ix_extraction_runs_owner_doc_id", "owner", "doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schema_name: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    page_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    llm_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 5: Pipeline Job Reports (Full JSON PDF Binaries)
# ==============================================================================
class JobPdfRecord(Base):
    __tablename__ = "job_pdfs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    pdf_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 6: Document Knowledge Graph Memory (Layer 3)
# ==============================================================================
class DocumentGraphRecord(Base):
    __tablename__ = "document_graphs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    job_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    owner: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    schema_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    strategy: Mapped[str] = mapped_column(String, default="graph_memory")
    graph_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 7: Durable Page Delta Checkpoints (Layer 3)
# ==============================================================================
class PageCheckpointRecord(Base):
    __tablename__ = "page_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_key: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String, default="_admin", nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pipeline_version: Mapped[str] = mapped_column(String, default="v1")
    extraction_version: Mapped[str] = mapped_column(String, default="v1")
    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    schema_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_delta_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    delta_checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    error_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 8: Users (Layer 3 Auth Persistence)
# ==============================================================================
class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


def init_db():
    """Create all 8 tables in PostgreSQL / SQLite if they do not exist."""
    global engine, SessionLocal
    try:
        Base.metadata.create_all(engine)
    except Exception as exc:
        logger.warning("storage.init_db_fallback", error=str(exc))
        engine = create_engine("sqlite:///./idp_storage.db")
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)


# ==============================================================================
# CRUD Helper Functions with Safe Error Handling
# ==============================================================================

def save_document(filename: str, file_bytes: bytes, owner: str, content_type: str = "application/pdf") -> Optional[int]:
    """
    Save or update an uploaded document (PDF) shadow copy in PostgreSQL / SQLite.
    Non-authoritative when S3 is configured. Keyed by (owner, filename).
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        session = SessionLocal()
        try:
            doc = session.query(DocumentRecord).filter_by(owner=owner, filename=filename).first()
            if doc:
                doc.file_size = len(file_bytes)
                doc.file_data = file_bytes
                doc.file_hash = file_hash
                doc.content_type = content_type
                doc.created_at = datetime.now(timezone.utc)
            else:
                doc = DocumentRecord(
                    owner=owner,
                    filename=filename,
                    file_size=len(file_bytes),
                    content_type=content_type,
                    file_hash=file_hash,
                    file_data=file_bytes,
                )
                session.add(doc)
            session.commit()
            session.refresh(doc)
            logger.info("storage.document_saved", owner=owner, file_name=filename, doc_id=doc.id, size=len(file_bytes))
            return doc.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_document_failed", owner=owner, file_name=filename, error=str(exc))
        return None


def save_schema_record(
    schema_id: str,
    document_type: str,
    schema_json: dict,
    session_id: Optional[str] = None,
    sample_documents: Optional[list] = None,
    confirmed_at: Optional[datetime] = None,
    owner: Optional[str] = None,
    schema_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Save or insert a target schema in PostgreSQL.

    Versioning: schemas are keyed by (owner, schema_name). Each new confirmation
    for the same owner+schema_name gets an auto-incremented version (v1, v2, ...).
    Returns a dict {id, version, schema_id} on success, or None on failure.
    """
    try:
        init_db()
        fields = (schema_json.get("schema") or {}).get("fields", []) or schema_json.get("fields", [])
        field_count = len(fields)
        # Fall back to document_type as the logical name if none supplied.
        eff_name = (schema_name or document_type or "schema").strip()
        session = SessionLocal()
        try:
            rec = session.query(SchemaRecord).filter_by(schema_id=schema_id).first()
            if rec:
                # Update-in-place (same schema_id): keep its version.
                rec.document_type = document_type
                rec.owner = owner if owner is not None else rec.owner
                rec.schema_name = eff_name
                rec.field_count = field_count
                rec.schema_json = schema_json
                rec.sample_documents = sample_documents or []
                rec.session_id = session_id
                if confirmed_at:
                    rec.confirmed_at = confirmed_at
                version = rec.version or 1
            else:
                # New schema_id: compute next version for this (owner, schema_name).
                max_v = (
                    session.query(SchemaRecord.version)
                    .filter(SchemaRecord.owner == owner, SchemaRecord.schema_name == eff_name)
                    .order_by(SchemaRecord.version.desc())
                    .first()
                )
                version = ((max_v[0] if max_v and max_v[0] else 0) + 1)
                # Reflect owner/name/version inside the stored json too (for readers).
                if isinstance(schema_json, dict):
                    schema_json = dict(schema_json)
                    schema_json.setdefault("owner", owner)
                    schema_json["schema_name"] = eff_name
                    schema_json["version"] = version
                rec = SchemaRecord(
                    schema_id=schema_id,
                    document_type=document_type,
                    owner=owner,
                    schema_name=eff_name,
                    version=version,
                    field_count=field_count,
                    schema_json=schema_json,
                    sample_documents=sample_documents or [],
                    session_id=session_id,
                    confirmed_at=confirmed_at or datetime.now(timezone.utc),
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.schema_saved", schema_id=schema_id, id=rec.id, owner=owner, schema_name=eff_name, version=rec.version, field_count=field_count)
            return {"id": rec.id, "version": rec.version, "schema_id": rec.schema_id}
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_schema_failed", schema_id=schema_id, error=str(exc))
        return None


def save_markdown_record(
    doc_id: str,
    md_filename: str,
    markdown_content: str,
    owner: str,
    schema_id: Optional[str] = None,
    schema_ref_json: Optional[dict] = None,
    page_count: int = 1,
    pages_json: Optional[list] = None,
) -> Optional[int]:
    """
    Save or update converted Markdown (.md) in PostgreSQL / SQLite.
    Keyed by (owner, doc_id).
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(MarkdownRecord).filter_by(owner=owner, doc_id=doc_id).first()
            if rec:
                rec.md_filename = md_filename
                rec.markdown_content = markdown_content
                rec.page_count = page_count
                rec.schema_id = schema_id
                rec.schema_ref_json = schema_ref_json
                rec.pages_json = pages_json
                rec.created_at = datetime.now(timezone.utc)
            else:
                rec = MarkdownRecord(
                    owner=owner,
                    doc_id=doc_id,
                    md_filename=md_filename,
                    markdown_content=markdown_content,
                    page_count=page_count,
                    schema_id=schema_id,
                    schema_ref_json=schema_ref_json,
                    pages_json=pages_json,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.markdown_saved", owner=owner, doc_id=doc_id, md_filename=md_filename, id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_markdown_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def get_markdown_record(doc_id: str, owner: str) -> Optional[dict]:
    """
    Retrieve stored markdown record by doc_id or md_filename, strictly scoped to owner.
    Candidate normalization is executed strictly WITHIN the owner scope.
    """
    if not doc_id or not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            from pathlib import Path as _Path
            stem = _Path(doc_id).stem
            candidates = list(dict.fromkeys([doc_id, stem, f"{stem}.pdf", f"{stem}.md"]))
            rec = (
                session.query(MarkdownRecord)
                .filter(
                    MarkdownRecord.owner == owner,
                    (MarkdownRecord.doc_id.in_(candidates) | MarkdownRecord.md_filename.in_(candidates)),
                )
                .order_by(MarkdownRecord.id.desc())
                .first()
            )
            if rec:
                return {
                    "id": rec.id,
                    "owner": rec.owner,
                    "doc_id": rec.doc_id,
                    "md_filename": rec.md_filename,
                    "markdown_content": rec.markdown_content,
                    "page_count": rec.page_count,
                    "schema_id": rec.schema_id,
                    "schema_ref_json": rec.schema_ref_json,
                    "pages_json": rec.pages_json,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                }
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_markdown_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def save_extraction_run(
    doc_id: str,
    page_count: int | None,
    schema_name: str,
    result_json: dict,
    owner: str,
    llm_provider: str | None = None,
    model_name: str | None = None,
    processing_time_seconds: float | None = None,
    page_outputs: list | None = None,
) -> Optional[int]:
    """
    Save one extraction run in PostgreSQL / SQLite, strictly associated with owner.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        page_details = None
        if page_outputs:
            page_details = {
                str(p.page_number): {
                    "engines_used":   list(getattr(p, "engines_used", []) or []),
                    "confidence":     getattr(p, "confidence", None),
                    "escalated":      getattr(p, "escalated", None),
                    "low_confidence": getattr(p, "low_confidence", None),
                    "capabilities":   list(getattr(p, "capabilities", []) or []),
                }
                for p in page_outputs
            }

        session = SessionLocal()
        try:
            run = ExtractionRun(
                owner=owner,
                doc_id=doc_id,
                page_count=page_count,
                schema_name=schema_name,
                result_json=result_json,
                page_details=page_details,
                llm_provider=llm_provider,
                model_name=model_name,
                processing_time_seconds=processing_time_seconds,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            logger.info("storage.extraction_saved", owner=owner, doc_id=doc_id, run_id=run.id)
            return run.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_extraction_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def save_job_pdf(job_id: str, pdf_bytes: bytes, filename: Optional[str] = None) -> Optional[int]:
    """
    Save or update a pipeline job PDF report in PostgreSQL.
    """
    if not filename:
        filename = f"{job_id}_report.pdf"
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(JobPdfRecord).filter_by(job_id=job_id).first()
            if rec:
                rec.filename = filename
                rec.pdf_data = pdf_bytes
                rec.created_at = datetime.now(timezone.utc)
            else:
                rec = JobPdfRecord(
                    job_id=job_id,
                    filename=filename,
                    pdf_data=pdf_bytes,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.job_pdf_saved", job_id=job_id, size=len(pdf_bytes), id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_job_pdf_failed", job_id=job_id, error=str(exc))
        return None


def get_job_pdf(job_id: str) -> Optional[tuple[bytes, str]]:
    """
    Retrieve stored job PDF binary and filename from PostgreSQL.
    Returns (pdf_bytes, filename) or None if not found.
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(JobPdfRecord).filter_by(job_id=job_id).first()
            if rec:
                return (rec.pdf_data, rec.filename)
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_job_pdf_failed", job_id=job_id, error=str(exc))
        return None


# ==============================================================================
# Document Graph Persistence (Layer 3)
# ==============================================================================

def save_document_graph(
    doc_id: str,
    graph_dict: dict,
    job_id: Optional[str] = None,
    owner: Optional[str] = None,
    schema_id: Optional[str] = None,
    strategy: str = "graph_memory",
) -> Optional[int]:
    """
    Save or update a document's GraphMemory representation in PostgreSQL / SQLite.
    Associated with doc_id, job_id, and owner for strict multi-tenant isolation.
    """
    try:
        init_db()
        nodes = graph_dict.get("nodes", [])
        edges = graph_dict.get("edges", [])
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord).filter_by(doc_id=doc_id)
            if job_id:
                query = query.filter_by(job_id=job_id)
            rec = query.first()

            if rec:
                rec.graph_json = graph_dict
                rec.node_count = len(nodes)
                rec.edge_count = len(edges)
                rec.strategy = strategy
                if owner:
                    rec.owner = owner
                if schema_id:
                    rec.schema_id = schema_id
                rec.updated_at = datetime.now(timezone.utc)
            else:
                rec = DocumentGraphRecord(
                    doc_id=doc_id,
                    job_id=job_id,
                    owner=owner,
                    schema_id=schema_id,
                    strategy=strategy,
                    graph_json=graph_dict,
                    node_count=len(nodes),
                    edge_count=len(edges),
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.document_graph_saved", doc_id=doc_id, job_id=job_id, nodes=len(nodes), edges=len(edges), id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_document_graph_failed", doc_id=doc_id, error=str(exc))
        return None


def get_document_graph(
    doc_id: Optional[str] = None,
    job_id: Optional[str] = None,
    owner: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve stored document graph dictionary.
    Enforces user isolation if owner is provided (non-admin).
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord)
            if doc_id:
                query = query.filter_by(doc_id=doc_id)
                if job_id:
                    query = query.filter_by(job_id=job_id)
            elif job_id:
                query = query.filter_by(job_id=job_id)
            else:
                return None

            if owner and owner != "admin":
                query = query.filter((DocumentGraphRecord.owner == owner) | (DocumentGraphRecord.owner == None))

            rec = query.order_by(DocumentGraphRecord.created_at.desc()).first()
            if rec and rec.graph_json:
                return rec.graph_json
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_document_graph_failed", doc_id=doc_id, job_id=job_id, error=str(exc))
        return None


def list_document_graphs(owner: Optional[str] = None) -> list:
    """
    List all available document graphs with summary metadata.
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord)
            if owner and owner != "admin":
                query = query.filter((DocumentGraphRecord.owner == owner) | (DocumentGraphRecord.owner == None))
            recs = query.order_by(DocumentGraphRecord.created_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "doc_id": r.doc_id,
                    "job_id": r.job_id,
                    "owner": r.owner,
                    "schema_id": r.schema_id,
                    "strategy": r.strategy,
                    "node_count": r.node_count,
                    "edge_count": r.edge_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recs
            ]
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_document_graphs_failed", error=str(exc))
        return []


# ==============================================================================
# Page Checkpoint Helpers (Layer 3 Durable Recovery)
# ==============================================================================

def compute_page_delta_checksum(delta_dict: Optional[dict]) -> str:
    """Compute deterministic SHA256 checksum of a PageDelta dictionary."""
    if not delta_dict:
        return ""
    serialized = json.dumps(delta_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_checkpoint_key(
    doc_id: str,
    page_number: int,
    owner: str,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
) -> str:
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    return f"{owner.strip()}:{doc_id}:p{page_number}:{pipeline_version}:{extraction_version}"


def save_page_checkpoint(
    doc_id: str,
    page_number: int,
    status: str,
    owner: str,
    page_delta_json: Optional[dict] = None,
    delta_checksum: Optional[str] = None,
    error_info: Optional[str] = None,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
    model_version: Optional[str] = None,
    prompt_version: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> Optional[int]:
    """
    Save or update a durable page checkpoint in PostgreSQL / SQLite.
    Protects against duplicate page processing and enables crash recovery.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        key = build_checkpoint_key(doc_id, page_number, owner=owner, pipeline_version=pipeline_version, extraction_version=extraction_version)
        if page_delta_json and not delta_checksum:
            delta_checksum = compute_page_delta_checksum(page_delta_json)

        session = SessionLocal()
        try:
            rec = session.query(PageCheckpointRecord).filter_by(checkpoint_key=key).first()
            if rec:
                rec.status = status
                rec.owner = owner
                if page_delta_json is not None:
                    rec.page_delta_json = page_delta_json
                if delta_checksum:
                    rec.delta_checksum = delta_checksum
                if error_info is not None:
                    rec.error_info = error_info
                rec.attempt_count += 1
                rec.updated_at = datetime.now(timezone.utc)
            else:
                rec = PageCheckpointRecord(
                    checkpoint_key=key,
                    owner=owner,
                    doc_id=doc_id,
                    page_number=page_number,
                    pipeline_version=pipeline_version,
                    extraction_version=extraction_version,
                    model_version=model_version,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    status=status,
                    page_delta_json=page_delta_json,
                    delta_checksum=delta_checksum,
                    attempt_count=1,
                    error_info=error_info,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info(
                "page.checkpoint.created",
                checkpoint_key=key,
                owner=owner,
                doc_id=doc_id,
                page_number=page_number,
                status=status,
                attempt=rec.attempt_count,
            )
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error(
            "page.checkpoint.save_failed",
            owner=owner,
            doc_id=doc_id,
            page_number=page_number,
            error=str(exc),
        )
        return None


def get_page_checkpoint(
    doc_id: str,
    page_number: int,
    owner: str,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
) -> Optional[dict]:
    """Retrieve durable page checkpoint dictionary."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        init_db()
        key = build_checkpoint_key(doc_id, page_number, owner=owner, pipeline_version=pipeline_version, extraction_version=extraction_version)
        session = SessionLocal()
        try:
            rec = session.query(PageCheckpointRecord).filter_by(checkpoint_key=key).first()
            if rec:
                return {
                    "id": rec.id,
                    "checkpoint_key": rec.checkpoint_key,
                    "owner": rec.owner,
                    "doc_id": rec.doc_id,
                    "page_number": rec.page_number,
                    "pipeline_version": rec.pipeline_version,
                    "extraction_version": rec.extraction_version,
                    "status": rec.status,
                    "page_delta_json": rec.page_delta_json,
                    "delta_checksum": rec.delta_checksum,
                    "attempt_count": rec.attempt_count,
                    "error_info": rec.error_info,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                    "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
                }
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_page_checkpoint_failed", owner=owner, doc_id=doc_id, page=page_number, error=str(exc))
        return None


def list_page_checkpoints(doc_id: str, owner: str, pipeline_version: Optional[str] = None) -> List[dict]:
    """List all checkpoints for a document in ascending page order."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return []
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(PageCheckpointRecord).filter_by(doc_id=doc_id, owner=owner)
            if pipeline_version:
                query = query.filter_by(pipeline_version=pipeline_version)
            recs = query.order_by(PageCheckpointRecord.page_number.asc()).all()
            return [
                {
                    "id": r.id,
                    "checkpoint_key": r.checkpoint_key,
                    "owner": r.owner,
                    "doc_id": r.doc_id,
                    "page_number": r.page_number,
                    "pipeline_version": r.pipeline_version,
                    "extraction_version": r.extraction_version,
                    "status": r.status,
                    "page_delta_json": r.page_delta_json,
                    "delta_checksum": r.delta_checksum,
                    "attempt_count": r.attempt_count,
                    "error_info": r.error_info,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in recs
            ]
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_page_checkpoints_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return []


def delete_page_checkpoints(doc_id: str, owner: str) -> int:
    """Delete all checkpoints for a given document."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return 0
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            deleted = session.query(PageCheckpointRecord).filter_by(doc_id=doc_id, owner=owner).delete()
            session.commit()
            return deleted
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.delete_page_checkpoints_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return 0


# ==============================================================================
# Schema Read Helpers (DB-backed reads for the web app)
# ==============================================================================

def _schema_record_to_dict(rec: "SchemaRecord") -> dict:
    """
    Normalize a SchemaRecord row into the same shape the web app previously
    read from schema_registry/<id>.json. The full original record is stored in
    schema_json, so return that as the authoritative payload, augmented with
    top-level summary fields the endpoints rely on.
    """
    payload = dict(rec.schema_json) if isinstance(rec.schema_json, dict) else {}
    # Ensure the summary fields are always present and consistent with the row.
    payload.setdefault("schema_id", rec.schema_id)
    payload.setdefault("document_type", rec.document_type)
    if rec.confirmed_at and not payload.get("confirmed_at"):
        payload["confirmed_at"] = rec.confirmed_at.isoformat()
    return payload


def get_schema_record(schema_id: str, owner: Optional[str] = None) -> Optional[dict]:
    """
    Retrieve a single confirmed schema record from PostgreSQL by schema_id.
    Returns the full stored schema_json dict (same shape as the old on-disk
    schema_registry/<id>.json), or None if not found.

    If ``owner`` is provided, the returned record must belong to that owner
    (i.e. rec.owner == owner).  Records with a NULL owner are shared/admin
    schemas and are visible to all callers regardless of owner.
    Pass owner=None (default) to skip owner-scoping (admin-only code paths).
    """
    if not schema_id:
        return None
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(SchemaRecord).filter_by(schema_id=schema_id).first()
            if not rec:
                # Fall back to disk schema_registry/
                from pathlib import Path
                reg_file = Path("schema_registry") / f"{schema_id}.json"
                if reg_file.is_file():
                    try:
                        data = json.loads(reg_file.read_text(encoding="utf-8"))
                        # Owner check on disk fallback
                        if owner is not None:
                            rec_owner = data.get("owner")
                            if rec_owner is not None and rec_owner != owner:
                                return None
                        return data
                    except Exception:
                        pass
                return None
            # Owner check on DB record
            if owner is not None and rec.owner is not None and rec.owner != owner:
                return None
            return _schema_record_to_dict(rec)
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_schema_record_failed", schema_id=schema_id, error=str(exc))
        return None



def schema_exists(schema_id: str) -> bool:
    """Return True if a schema with this schema_id exists in the DB."""
    if not schema_id:
        return False
    try:
        init_db()
        session = SessionLocal()
        try:
            return session.query(SchemaRecord).filter_by(schema_id=schema_id).first() is not None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.schema_exists_failed", schema_id=schema_id, error=str(exc))
        return False


def list_schema_records(owner: Optional[str] = None) -> List[dict]:
    """
    List confirmed schemas from PostgreSQL as summary dicts, newest first.

    Ordering: by confirmed_at desc (falling back to created_at desc).
    Owner filtering: if `owner` is provided, returns schemas whose stored
    record has that owner OR has no owner set (legacy/ownerless records remain
    visible, mirroring the previous disk behavior). Pass owner=None for admin
    (returns everything).
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            q = session.query(SchemaRecord)
            if owner is not None:
                # Strict per-user isolation: only schemas owned by this user.
                # Falls back to the owner embedded in schema_json for legacy rows.
                q = q.filter(SchemaRecord.owner == owner)
            recs = q.order_by(
                SchemaRecord.confirmed_at.desc().nullslast(), SchemaRecord.created_at.desc()
            ).all()
            results: List[dict] = []
            for rec in recs:
                data = rec.schema_json if isinstance(rec.schema_json, dict) else {}
                rec_owner = rec.owner or data.get("owner")
                # Legacy rows may have no owner column yet; enforce json owner too.
                if owner is not None and rec.owner is None and data.get("owner") not in (None, owner):
                    continue
                inner = data.get("schema") if isinstance(data.get("schema"), dict) else {}
                doc_type = inner.get("document_type") or data.get("document_type") or rec.document_type
                field_count = rec.field_count
                if not field_count:
                    field_count = len(inner.get("fields", []) or data.get("fields", []) or [])
                results.append({
                    "schema_id": rec.schema_id,
                    "document_type": doc_type,
                    "schema_name": rec.schema_name or data.get("schema_name") or doc_type,
                    "version": rec.version or data.get("version") or 1,
                    "confirmed_at": data.get("confirmed_at") or (rec.confirmed_at.isoformat() if rec.confirmed_at else None),
                    "field_count": field_count,
                    "sample_documents": rec.sample_documents or data.get("sample_documents", []) or [],
                    "owner": rec_owner,
                })
            return results
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_schema_records_failed", error=str(exc))
        return []


def get_latest_owned_schema_id(owner: str) -> Optional[str]:
    """
    Return the schema_id of the most recently confirmed schema owned by `owner`
    (used by the zero-parameter user pipeline trigger). Ownerless schemas are
    not considered "owned" here, matching the previous disk semantics which
    filtered strictly on owner == username.
    """
    if not owner:
        return None
    try:
        init_db()
        session = SessionLocal()
        try:
            recs = (
                session.query(SchemaRecord)
                .order_by(SchemaRecord.confirmed_at.desc().nullslast(), SchemaRecord.created_at.desc())
                .all()
            )
            for rec in recs:
                data = rec.schema_json if isinstance(rec.schema_json, dict) else {}
                if (rec.owner or data.get("owner")) == owner:
                    return rec.schema_id

            # Fall back to disk schema_registry/
            from pathlib import Path
            reg_dir = Path("schema_registry")
            if reg_dir.exists():
                candidates = []
                for f in reg_dir.glob("*.json"):
                    try:
                        sdata = json.loads(f.read_text(encoding="utf-8"))
                        if sdata.get("owner") == owner:
                            conf = sdata.get("confirmed_at") or ""
                            candidates.append((conf, sdata.get("schema_id") or f.stem))
                    except Exception:
                        pass
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]

            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_latest_owned_schema_failed", owner=owner, error=str(exc))
        return None


# ==============================================================================
# Extraction Run Read Helpers (DB-backed reads for the Query Bot JSON fallback)
# ==============================================================================

def get_latest_extracted_data(doc_id: str, owner: str) -> Optional[dict]:
    """
    Retrieve the most recent extraction result for a document from PostgreSQL / SQLite.
    Candidate matching is strictly evaluated WITHIN the specified owner scope.
    """
    if not doc_id or not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        from pathlib import Path as _Path
        stem = _Path(doc_id).stem
        candidates = list(dict.fromkeys([doc_id, stem, f"{stem}.pdf"]))
        init_db()
        session = SessionLocal()
        try:
            rec = (
                session.query(ExtractionRun)
                .filter(
                    ExtractionRun.owner == owner,
                    ExtractionRun.doc_id.in_(candidates),
                )
                .order_by(ExtractionRun.created_at.desc())
                .first()
            )
            if not rec or not rec.result_json:
                return None
            result = rec.result_json
            if isinstance(result, dict):
                # Prefer the nested extracted_data if present, else return as-is.
                return result.get("extracted_data") or result
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_latest_extracted_data_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def list_markdown_records(owner: str) -> List[dict]:
    """
    List converted-markdown output records from PostgreSQL / SQLite as summary dicts,
    newest first. Strictly filtered by owner.
    Admins can pass '_all' or '*' to inspect across all owners.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        return []
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(MarkdownRecord)
            if owner not in ("*", "_all"):
                query = query.filter(MarkdownRecord.owner == owner)
            recs = query.order_by(MarkdownRecord.created_at.desc()).all()
            out: List[dict] = []
            seen = set()
            for r in recs:
                # De-duplicate by (owner, md_filename)
                key = (r.owner, r.md_filename or f"{r.doc_id}.md")
                if key in seen:
                    continue
                seen.add(key)
                ref = r.schema_ref_json if isinstance(r.schema_ref_json, dict) else {}
                out.append({
                    "name": r.md_filename or f"{r.doc_id}.md",
                    "doc_id": r.doc_id,
                    "owner": r.owner,
                    "size": len(r.markdown_content or ""),
                    "modified": r.created_at.isoformat() if r.created_at else None,
                    "page_count": r.page_count,
                    "schema_id": r.schema_id or ref.get("schema_id"),
                    "source_pdf": ref.get("source_pdf") or (
                        f"{r.doc_id}" if str(r.doc_id).lower().endswith(".pdf") else f"{r.doc_id}.pdf"
                    ),
                    "schema_ref": ref or None,
                })
            return out
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_markdown_records_failed", owner=owner, error=str(exc))
        return []


def _owned_doc_ids(owner: str) -> set:
    """
    Return the set of doc_id variants owned by `owner`, derived from
    DocumentGraphRecord ownership (the authoritative per-user marker).
    """
    owned = set()
    if not owner:
        return owned
    try:
        init_db()
        session = SessionLocal()
        try:
            from pathlib import Path as _Path
            recs = session.query(DocumentGraphRecord).filter(DocumentGraphRecord.owner == owner).all()
            for r in recs:
                d = r.doc_id or ""
                stem = _Path(d).stem
                owned.update({d, stem, f"{stem}.pdf"})
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.owned_doc_ids_failed", owner=owner, error=str(exc))
    return owned


def list_markdown_records_for_owner(owner: str) -> List[dict]:
    """
    List processed-output (markdown) summaries that belong to `owner`.
    Directly delegates to list_markdown_records(owner=owner) enforcing DB isolation.
    """
    return list_markdown_records(owner=owner)
```

## File: schema_chatbot_v2/app/api/routes.py
```python
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.activity_log import log_activity
from app.core.auth import get_current_user
from app.core.conversation_manager import MAX_DOCUMENT_SAMPLES, MIN_DOCUMENT_SAMPLES, ConversationManager, TurnResult
from app.core.schema_state import _normalize_field_name, normalize_type, SUPPORTED_TYPES, SchemaState
from app.core.validator import validate_schema
from app.llm.factory import get_llm_adapter
from app.models.api_models import ChatRequest, ChatResponse, UpdateSchemaRequest
from app.output.schema_renderer import render_json, render_pdf
from app.storage.session_store import Session, get_session_store
from app.storage.user_store import Role, User

SCHEMA_REGISTRY_DIR = Path(__file__).resolve().parents[3] / "schema_registry"
if not SCHEMA_REGISTRY_DIR.exists():
    # Fallback if running from a different root
    alt = Path(__file__).resolve().parents[2] / "schema_registry"
    if alt.exists():
        SCHEMA_REGISTRY_DIR = alt

logger = logging.getLogger(__name__)
router = APIRouter()


class CustomFieldIn(BaseModel):
    name: str
    type: str = "string"
    required: bool = True
    description: Optional[str] = ""
    item_type: Optional[str] = None
    pattern: Optional[str] = None
    currency: Optional[str] = None


class CreateCustomSchemaRequest(BaseModel):
    document_type: str
    schema_name: Optional[str] = None
    fields: List[CustomFieldIn] = Field(default_factory=list)


def get_conversation_manager() -> ConversationManager:
    return ConversationManager(llm=get_llm_adapter(), store=get_session_store())


def _to_response(result: TurnResult) -> ChatResponse:
    return ChatResponse(
        session_id=result.session_id,
        message=result.message,
        state=result.state,
        schema=result.schema,
        completed=result.completed,
        schema_id=result.schema_id,
        errors=result.errors,
    )


def _check_session_access(session: Session, user: User) -> None:
    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)
    if session.owner and session.owner != user.username and not is_admin:
        raise HTTPException(
            status_code=403,
            detail="Access denied: session belongs to another user.",
        )


@router.post("/chat", response_model=ChatResponse, response_model_by_alias=True)
async def chat(
    req: ChatRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    from starlette.concurrency import run_in_threadpool

    if not req.session_id:
        result = await run_in_threadpool(manager.start_session)
        session = manager.store.get(result.session_id)
        if session:
            session.owner = user.username
            manager.store.save(session)
        return _to_response(result)

    session = manager.store.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    _check_session_access(session, user)
    if not session.owner:
        session.owner = user.username
        manager.store.save(session)

    if not req.message:
        raise HTTPException(status_code=400, detail="message is required when session_id is provided")

    try:
        result = await run_in_threadpool(manager.handle_message, req.session_id, req.message)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(result)


@router.post("/schema/infer", response_model=ChatResponse, response_model_by_alias=True)
async def infer_schema(
    files: List[UploadFile] = File(..., description=f"{MIN_DOCUMENT_SAMPLES}-{MAX_DOCUMENT_SAMPLES} sample PDFs of the same document type"),
    session_id: str | None = Form(
        default=None,
        description="Optional - feed samples into an existing session (e.g. one already mid-chat) instead of starting a new one",
    ),
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    from app.api.pipeline_routes import check_user_extract_rate_limit, validate_uploaded_pdf
    check_user_extract_rate_limit(user.username)

    if session_id is not None:
        existing_sess = manager.store.get(session_id)
        if existing_sess is None:
            raise HTTPException(status_code=404, detail="session not found")
        _check_session_access(existing_sess, user)
        if not existing_sess.owner:
            existing_sess.owner = user.username
            manager.store.save(existing_sess)

    if not (MIN_DOCUMENT_SAMPLES <= len(files) <= MAX_DOCUMENT_SAMPLES):
        raise HTTPException(
            status_code=400,
            detail=f"upload between {MIN_DOCUMENT_SAMPLES} and {MAX_DOCUMENT_SAMPLES} sample documents (got {len(files)})",
        )

    samples = []
    for f in files:
        safe_name, content = await validate_uploaded_pdf(f)
        samples.append(content)

        # Store input sample document in PostgreSQL
        try:
            from src.ai.layer3_extraction.storage import save_document
            save_document(filename=safe_name, file_bytes=content, owner=user.username, content_type=f.content_type or "application/pdf")
        except Exception:
            pass

    from starlette.concurrency import run_in_threadpool

    try:
        result = await run_in_threadpool(manager.start_from_documents, samples, session_id=session_id)
        if session_id is None:
            session = manager.store.get(result.session_id)
            if session:
                session.owner = user.username
                manager.store.save(session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_response(result)


@router.get("/session/{session_id}", response_model=ChatResponse, response_model_by_alias=True)
def get_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    session = manager.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    _check_session_access(session, user)
    return ChatResponse(
        session_id=session.session_id,
        message="",
        state=session.state.value,
        schema=session.schema_state.to_json_schema(),
        completed=session.completed,
        schema_id=session.schema_id,
    )


@router.post("/session/{session_id}/reset", response_model=ChatResponse, response_model_by_alias=True)
def reset_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    session = manager.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    _check_session_access(session, user)
    manager.store.delete(session_id)
    result = manager.start_session()
    new_sess = manager.store.get(result.session_id)
    if new_sess:
        new_sess.owner = user.username
        manager.store.save(new_sess)
    return _to_response(result)


@router.post("/session/{session_id}/schema", response_model=ChatResponse, response_model_by_alias=True)
def update_schema(
    session_id: str,
    req: UpdateSchemaRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    session = manager.store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    _check_session_access(session, user)
    try:
        result = manager.update_schema_manually(session_id, req.document_type, req.fields)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _to_response(result)


@router.post(
    "/schema/custom",
    summary="Create and persist a custom schema directly from the interactive builder",
)
def create_custom_schema(
    req: CreateCustomSchemaRequest,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    doc_type = (req.document_type or "").strip()
    if not doc_type:
        raise HTTPException(status_code=400, detail="document_type cannot be empty or whitespace-only")

    if not req.fields:
        raise HTTPException(status_code=400, detail="schema must contain at least one field")

    seen_normalized: set[str] = set()
    schema_state = SchemaState()
    schema_state.set_document_type(doc_type)

    for idx, f in enumerate(req.fields):
        raw_name = (f.name or "").strip()
        if not raw_name:
            raise HTTPException(status_code=400, detail=f"field at index {idx} has an empty or blank name")

        norm_name = _normalize_field_name(raw_name)
        if not norm_name:
            raise HTTPException(status_code=400, detail=f"field name '{f.name}' normalizes to empty string")

        if norm_name in seen_normalized:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate field name detected: '{f.name}' (normalized: '{norm_name}')",
            )
        seen_normalized.add(norm_name)

        raw_type = (f.type or "string").strip()
        item_type = f.item_type
        if "[" in raw_type and "]" in raw_type:
            item_type = raw_type[raw_type.find("[") + 1 : raw_type.find("]")].strip()
            raw_type = "array"

        norm_type, _ = normalize_type(raw_type)
        if not norm_type or norm_type not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported type '{f.type}' for field '{f.name}'. Supported types: {sorted(SUPPORTED_TYPES)}",
            )

        if norm_type == "array" and not item_type:
            item_type = "string"

        schema_state.add_field(
            norm_name,
            type=norm_type,
            required=bool(f.required),
            description=(f.description or "").strip(),
            item_type=item_type,
            pattern=f.pattern,
            currency=f.currency,
        )

    # Authoritative validation
    errors = validate_schema(schema_state)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Generate unique schema ID (collision check is DB-backed).
    from src.ai.layer3_extraction.storage import schema_exists
    schema_id = f"schema_{uuid.uuid4().hex[:12]}"
    while schema_exists(schema_id):
        schema_id = f"schema_{uuid.uuid4().hex[:12]}"

    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    owner_name = user.username if user else "admin"
    schema_name = (req.schema_name or "").strip() or schema_state.document_type or "schema"
    record = {
        "schema_id": schema_id,
        "document_type": schema_state.document_type,
        "schema_name": schema_name,
        "confirmed_at": now_iso,
        "session_id": None,
        "turn_count_at_confirm": 1,
        "schema": schema_state.to_json_schema(),
        "sample_documents": [],
        "owner": owner_name,
    }

    # Persist to PostgreSQL schemas table (authoritative store; no disk write).
    # Versioning is keyed by (owner, schema_name) inside save_schema_record.
    schema_version = 1
    try:
        from src.ai.layer3_extraction.storage import save_schema_record
        saved = save_schema_record(
            schema_id=schema_id,
            document_type=schema_state.document_type or "document",
            schema_json=record,
            session_id=None,
            sample_documents=[],
            owner=owner_name,
            schema_name=schema_name,
        )
        if saved and isinstance(saved, dict):
            schema_version = saved.get("version", 1)
            record["version"] = schema_version
    except Exception as exc:
        logger.warning("storage.schema_save_failed for schema_id=%s: %s", schema_id, exc)

    # Persist to schema_registry on disk for registry file compatibility
    try:
        from pathlib import Path
        reg_dir = Path("schema_registry")
        reg_dir.mkdir(parents=True, exist_ok=True)
        (reg_dir / f"{schema_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Audit log
    try:
        log_activity(
            user.username if user else "admin",
            "schema_confirmed",
            {"schema_id": schema_id, "document_type": schema_state.document_type},
        )
    except Exception:
        pass

    return {
        "status": "success",
        "schema_id": schema_id,
        "document_type": schema_state.document_type,
        "schema_name": schema_name,
        "version": schema_version,
        "confirmed_at": now_iso,
        "schema": schema_state.to_json_schema(),
        "record": record,
    }


# ---------------------------------------------------------------------------
# Schema download endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/schema/my",
    summary="List schemas owned by the current user",
    response_model=list,
)
def list_my_schemas(
    user: User = Depends(get_current_user),
) -> list:
    """
    Returns a list of confirmed schema summaries owned by the authenticated user.
    ADMIN sees all schemas. USER sees only their own.

    Each item contains: schema_id, document_type, confirmed_at, field_count.
    """
    from app.storage.user_store import Role
    from src.ai.layer3_extraction.storage import list_schema_records

    owner_filter = None if user.role == Role.ADMIN else user.username
    return [
        {
            "schema_id":     r["schema_id"],
            "document_type": r["document_type"],
            "schema_name":   r.get("schema_name"),
            "version":       r.get("version"),
            "confirmed_at":  r["confirmed_at"],
            "field_count":   r["field_count"],
            "owner":         r.get("owner"),
        }
        for r in list_schema_records(owner=owner_filter)
    ]

def _load_schema_record(schema_id: str, requesting_user: User) -> dict:
    """
    Load a confirmed schema record from the PostgreSQL schemas table (DB-backed).

    Access rules:
      - ADMIN: can download any schema.
      - USER:  can only download schemas they own (record.owner == username).
                If the record has no owner set (legacy records), access is
                granted to avoid breaking existing data.

    Raises 404 if the schema does not exist, 403 if the user does not own it.
    """
    from app.storage.user_store import Role
    from src.ai.layer3_extraction.storage import get_schema_record

    record = get_schema_record(schema_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"schema '{schema_id}' not found")

    # Ownership check for non-admin users
    if requesting_user.role != Role.ADMIN:
        owner = record.get("owner")
        if owner and owner != requesting_user.username:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to download this schema.",
            )

    return record


@router.get(
    "/schema/{schema_id}/download/json",
    summary="Download confirmed schema as JSON",
    response_class=Response,
    responses={
        200: {
            "content": {"application/json": {}},
            "description": "JSON file containing the full confirmed schema record",
        },
        404: {"description": "Schema not found"},
    },
)
@router.get(
    "/schema/{schema_id}/json",
    summary="Download confirmed schema as JSON (alias)",
    response_class=Response,
    include_in_schema=False,
)
def download_schema_json(
    schema_id: str,
    user: User = Depends(get_current_user),
) -> Response:
    """
    Download the confirmed extraction schema as a formatted JSON file.

    The file contains the complete schema record — document_type, all fields
    with their types/constraints, schema_id, confirmed_at timestamp, and
    session metadata. Suitable for direct import into downstream pipeline
    configuration or manual inspection.
    """
    record = _load_schema_record(schema_id, requesting_user=user)
    doc_type = (record.get("document_type") or "schema").replace(" ", "_")
    filename = f"{doc_type}_{schema_id}.json"

    payload = render_json(record)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/schema/{schema_id}/download/pdf",
    summary="Download confirmed schema as PDF",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF report of the confirmed schema with a formatted fields table",
        },
        404: {"description": "Schema not found"},
        500: {"description": "PDF generation failed"},
    },
)
@router.get(
    "/schema/{schema_id}/pdf",
    summary="Download confirmed schema as PDF (alias)",
    response_class=Response,
    include_in_schema=False,
)
def download_schema_pdf(
    schema_id: str,
    user: User = Depends(get_current_user),
) -> Response:
    """
    Download the confirmed extraction schema as a human-readable PDF.

    The PDF contains a metadata header (document_type, schema_id,
    confirmed_at) and a formatted table of all fields — name, type, required
    flag, and any additional constraints (item_type, currency, pattern,
    description). Suitable for review, sign-off, or sharing with non-technical
    stakeholders.
    """
    record = _load_schema_record(schema_id, requesting_user=user)
    doc_type = (record.get("document_type") or "schema").replace(" ", "_")
    filename = f"{doc_type}_{schema_id}.pdf"

    try:
        payload = render_pdf(record)
    except Exception as exc:
        logger.exception("PDF generation failed for schema_id=%s", schema_id)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
```

## File: src/ai/layer3_extraction/extractor.py
```python
"""
File: extractor.py
Purpose: Layer 3 extraction — multi-page field scanning and hardened Graph Memory extraction.

Supports three explicit Graph Memory concurrency modes:
- Mode A (Ordered Context Mode): Pages processed sequentially; Page N observes canonical
  memory produced by pages 1..N-1.
- Mode B (Concurrent Delta Mode): Pages extracted concurrently up to concurrency_limit;
  workers receive dispatch-time context snapshots; cross-page consistency recovered
  through serialized canonical merge and post-merge reconciliation.
- Mode C (Epoch / Batch Mode): Pages partitioned into epochs; concurrent extraction within
  an epoch, followed by canonical merge so the next epoch sees fresh canonical context.

Hardened with:
- PageDelta validation before persistence and canonical mutation (malformed deltas never reach graph).
- Durable page checkpoints (PostgreSQL / SQLite) with deterministic idempotency.
- Crash-safe recovery: skip completed pages and replay deltas idempotently on worker restart.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel

from src.adapters.llm.extraction_base import ExtractionLLMClient
from src.ai.layer3_extraction.scratchpad import Scratchpad
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def merge_page_deltas_sorted(manager: Any, deltas: List[Any]) -> None:
    """Merge a batch of PageDeltas into the manager's canonical graph in SORTED
    page-number order, regardless of the order they were produced/arrived in
    (merge-determinism task). Idempotent re-ingestion is handled by the manager's
    checksum logic. This is the single contract the concurrent extractor and the
    determinism tests share: merge order is page order, never completion order."""
    from src.ai.layer3_extraction.storage import compute_page_delta_checksum
    for delta in sorted(deltas, key=lambda d: d.page_number):
        checksum = compute_page_delta_checksum(delta.to_dict())
        await manager.ingest_page_delta(delta, delta_checksum=checksum)
    # Deterministic post-merge cross-page linking (restore-linking task). Runs here so
    # the shared merge contract (used by the eval replay AND the determinism tests)
    # gets the same linking the production pipeline applies. Gated + no model calls.
    from src.config.settings import settings as _settings
    if getattr(_settings, "graph_crosspage_linking", True) and hasattr(manager, "link_cross_page"):
        manager.link_cross_page()


def extract_by_page_scan(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    graph_out: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Scan every page against the cached target schema fields.
    Collects findings for ALL fields page-by-page into the Scratchpad,
    then provides the Scratchpad evidence to the LLM to produce the final
    hydrated Pydantic model.
    """
    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    schema_fields: List[Dict[str, str]] = [
        {"name": k, "description": v.description or k}
        for k, v in schema.model_fields.items()
    ]
    field_names = [f["name"] for f in schema_fields]
    scratchpad = Scratchpad(schema_field_names=field_names)

    logger.info("layer3.page_scan.start", total_pages=total_pages, schema_fields=field_names)

    for page_data in pages:
        page_md = page_data["markdown"]
        page_number = page_data["page_number"]

        if not page_md.strip():
            logger.debug("layer3.page_scan.skip_empty", page_number=page_number)
            continue

        matches = llm.check_page_for_fields(
            page_md,
            schema_fields,
            page_number=page_number,
            total_pages=total_pages,
        )
        scratchpad.update(page_number, matches)

    scratchpad_values = scratchpad.to_values_dict()
    if not scratchpad_values:
        evidence_text = "\n\n".join([p["markdown"] for p in pages if p.get("markdown")])
    else:
        evidence_text = scratchpad.format_evidence_for_llm()

    try:
        result = llm.extract(evidence_text, schema)
    except Exception as exc:
        logger.warning("layer3.llm_final_extraction.fallback_to_scratchpad", error=str(exc))
        try:
            default_instance = schema()
        except Exception:
            default_instance = None
        default_values: Dict[str, Any] = default_instance.model_dump() if default_instance is not None else {}
        result = schema.model_validate({**default_values, **scratchpad_values})

    res_dict = result.model_dump()
    merged = dict(res_dict)
    for k, v in scratchpad_values.items():
        if k in merged and (merged[k] in ("", None)) and v not in ("", None):
            merged[k] = v
    if merged != res_dict:
        result = schema.model_validate(merged)

    if graph_out is not None:
        graph_out["strategy"] = "page_scan"
        graph_out["stats"] = {"total_nodes": 0, "total_edges": 0, "strategy": "page_scan"}
        graph_out["scratchpad"] = scratchpad.snapshot()
        graph_out["provenance"] = scratchpad.provenance()
    return result


async def extract_with_graph_memory_ordered(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    owner: str,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE A: ORDERED CONTEXT MODE.
    Processes pages in sequence: Page N can observe canonical memory produced by pages 1..N-1.
    LLM inference executes outside the canonical merge lock.
    Enforces the hardened lifecycle: Observe -> PageDelta -> Validate -> Checkpoint -> Merge -> Reconcile -> Resolve.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        get_page_checkpoint,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)

    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    logger.info(
        "layer3.graph_memory_ordered.start",
        total_pages=total_pages,
        doc_id=doc_id,
        anchor_types=manager.anchor_types,
    )

    # MEASUREMENT ONLY (audit finding #1): stamp owner/doc on Converse metric records.
    try:
        from src.adapters.llm import llm_metrics as _llm_metrics
        _llm_metrics.set_run_context(owner=owner, doc_id=doc_id)
    except Exception:
        pass

    pages_succeeded: List[int] = []
    pages_failed: List[int] = []

    for page_data in pages:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")
        page_blocks = page_data.get("blocks", [])

        if not str(page_md or "").strip():
            pages_succeeded.append(page_num)
            continue

        # Check durable checkpoint for crash recovery
        replayed_delta: Optional[PageDelta] = None
        replayed_checksum: Optional[str] = None
        if doc_id:
            cp = get_page_checkpoint(doc_id, page_num, owner=owner, pipeline_version=pipeline_version)
            if cp and cp.get("status") in ("DELTA_PERSISTED", "COMPLETED", "MERGED") and cp.get("page_delta_json"):
                try:
                    replayed_delta = PageDelta.from_dict(cp["page_delta_json"])
                    replayed_checksum = cp.get("delta_checksum")
                    logger.info(
                        "page.checkpoint.recovered",
                        doc_id=doc_id,
                        page_number=page_num,
                        checksum=replayed_checksum,
                    )
                except Exception as exc:
                    logger.warning("page.checkpoint.recovery_failed", page=page_num, error=str(exc))

        if replayed_delta is not None:
            # Replay already verified delta idempotently
            await manager.ingest_page_delta(replayed_delta, delta_checksum=replayed_checksum)
            pages_succeeded.append(page_num)
            continue

        # 1. Fetch targeted canonical context (Page N sees memory from pages 1..N-1)
        context = await manager.get_context_for_page(page_num)

        if doc_id:
            save_page_checkpoint(doc_id, page_num, status="PROCESSING", owner=owner, pipeline_version=pipeline_version)

        # 2. Build isolated PageDelta (LLM runs outside lock)
        try:
            delta = await agent.build_page_delta_async(
                page_number=page_num,
                markdown=page_md,
                total_pages=total_pages,
                context=context,
                blocks=page_blocks,
            )
        except Exception as exc:
            logger.error("layer3.ordered.worker_failed", page_number=page_num, error=str(exc))
            pages_failed.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", owner=owner, error_info=str(exc), pipeline_version=pipeline_version)
            continue

        # 3. Validate PageDelta
        val_res = validate_page_delta(delta)
        if not val_res.is_valid:
            err_msg = f"Validation failed: {', '.join(val_res.errors)}"
            logger.error("page.validation_failed", page_number=page_num, errors=val_res.errors)
            pages_failed.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", owner=owner, error_info=err_msg, pipeline_version=pipeline_version)
            continue

        delta_dict = delta.to_dict()
        checksum = compute_page_delta_checksum(delta_dict)

        # 4. Durable checkpoint BEFORE canonical mutation
        if doc_id:
            save_page_checkpoint(
                doc_id,
                page_num,
                status="DELTA_PERSISTED",
                owner=owner,
                page_delta_json=delta_dict,
                delta_checksum=checksum,
                pipeline_version=pipeline_version,
            )

        # 5. Canonical ingestion under manager lock
        await manager.ingest_page_delta(delta, delta_checksum=checksum)

        # 6. Mark page completed
        if doc_id:
            save_page_checkpoint(doc_id, page_num, status="COMPLETED", owner=owner, pipeline_version=pipeline_version)
        pages_succeeded.append(page_num)

    # 7. Reconcile cross-page references
    await manager.reconcile_cross_references()

    # 8. Schema resolution
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), owner=owner, strategy="graph_memory_ordered")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_ordered"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        graph_out["unresolved_reference_count"] = manager.unresolved_count
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

    return result


def extract_with_graph_memory(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    owner: str,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Sequential / Ordered Graph-Memory Extraction Strategy.
    Wraps Mode A (Ordered Context Mode) synchronously for backward compatibility.
    ``owner`` is REQUIRED (no default) and forwarded to Mode A (ordered). Background
    / admin paths with no authenticated tenant must pass ``owner="_admin"`` explicitly
    at the call site — never rely on an implicit default (tenant-isolation hardening).
    """
    return _run_coroutine_sync(
        extract_with_graph_memory_ordered(
            pages_md=pages_md,
            schema=schema,
            llm=llm,
            graph_out=graph_out,
            doc_id=doc_id,
            owner=owner,
            **kwargs,
        )
    )


async def extract_with_graph_memory_concurrent(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    owner: str,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE B: CONCURRENT DELTA MODE.
    Processes pages concurrently up to concurrency_limit with isolated PageDelta generation.
    SEMANTIC GUARANTEE:
    Concurrent workers receive a dispatch-time snapshot context; they do NOT receive dynamically
    changing canonical memory during parallel inference (concurrency != fresh shared memory).
    Cross-page references and consistency are recovered through serialized canonical merge
    under manager lock and post-merge reconciliation.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        list_page_checkpoints,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)
    concurrency_limit = max(1, getattr(settings, "graph_concurrency_limit", 8))

    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    logger.info(
        "layer3.graph_memory_concurrent.start",
        total_pages=total_pages,
        concurrency_limit=concurrency_limit,
        doc_id=doc_id,
        anchor_types=manager.anchor_types,
    )

    # MEASUREMENT ONLY (audit finding #1): stamp owner/doc on Converse metric records.
    try:
        from src.adapters.llm import llm_metrics as _llm_metrics
        _llm_metrics.set_run_context(owner=owner, doc_id=doc_id)
    except Exception:
        pass

    # Deltas awaiting the serial sorted merge. Keyed by page_number so the merge
    # order is page order, NOT worker-completion/recovery order (merge-determinism
    # task). Both recovered (checkpoint-replayed) and freshly-built deltas go here;
    # a single sorted merge pass below guarantees a resumed run == a fresh run.
    built_deltas: Dict[int, "PageDelta"] = {}

    # Checkpoint recovery: collect recovered deltas (do NOT merge yet).
    recovered_pages: Set[int] = set()
    if doc_id:
        checkpoints = list_page_checkpoints(doc_id, owner=owner, pipeline_version=pipeline_version)
        for cp in checkpoints:
            if cp.get("status") in ("DELTA_PERSISTED", "COMPLETED", "MERGED") and cp.get("page_delta_json"):
                try:
                    replayed_delta = PageDelta.from_dict(cp["page_delta_json"])
                    built_deltas[cp["page_number"]] = replayed_delta
                    recovered_pages.add(cp["page_number"])
                    logger.info(
                        "page.checkpoint.recovered",
                        doc_id=doc_id,
                        page_number=cp["page_number"],
                        checksum=cp.get("delta_checksum"),
                    )
                except Exception as exc:
                    logger.warning("page.checkpoint.replay_failed", page=cp.get("page_number"), error=str(exc))

    semaphore = asyncio.Semaphore(concurrency_limit)
    pages_succeeded: List[int] = list(recovered_pages)
    pages_failed: List[int] = []
    pages_degraded: List[int] = []   # audit #1 (2b/2c): truncated/invalid after retry/re-ask
    pages_partial: List[int] = []    # audit #1 (2c): needed the salvage path
    failed_errors: Dict[int, str] = {}

    async def _build_page(page_data: Dict[str, Any]) -> None:
        page_num = page_data.get("page_number", 1)
        page_md = page_data.get("markdown", "")
        page_blocks = page_data.get("blocks", [])

        if page_num in recovered_pages:
            return  # Already recovered from checkpoint

        if not str(page_md or "").strip():
            pages_succeeded.append(page_num)
            return

        async with semaphore:
            # 1. Fetch targeted dispatch-time context (snapshot at dispatch)
            context = await manager.get_context_for_page(page_num)

            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="PROCESSING", owner=owner, pipeline_version=pipeline_version)

            # 2. Build isolated PageDelta (LLM runs concurrently outside manager lock)
            try:
                delta = await agent.build_page_delta_async(
                    page_number=page_num,
                    markdown=page_md,
                    total_pages=total_pages,
                    context=context,
                    blocks=page_blocks,
                    executor=llm_executor,
                )
            except Exception as exc:
                logger.error("layer3.concurrent.worker_failed", page_number=page_num, error=str(exc))
                pages_failed.append(page_num)
                failed_errors[page_num] = str(exc)
                if doc_id:
                    save_page_checkpoint(doc_id, page_num, status="FAILED", owner=owner, error_info=str(exc), pipeline_version=pipeline_version)
                return

        # 3. Validate PageDelta
        val_res = validate_page_delta(delta)
        if not val_res.is_valid:
            err_msg = f"Validation failed: {', '.join(val_res.errors)}"
            logger.error("page.validation_failed", page_number=page_num, errors=val_res.errors)
            pages_failed.append(page_num)
            failed_errors[page_num] = err_msg
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="FAILED", owner=owner, error_info=err_msg, pipeline_version=pipeline_version)
            return

        delta_dict = delta.to_dict()
        checksum = compute_page_delta_checksum(delta_dict)

        # 4. Durable checkpoint BEFORE canonical merge
        if doc_id:
            save_page_checkpoint(
                doc_id,
                page_num,
                status="DELTA_PERSISTED",
                owner=owner,
                page_delta_json=delta_dict,
                delta_checksum=checksum,
                pipeline_version=pipeline_version,
            )

        # Hand the built delta to the serial merge phase (do NOT merge here — merging
        # in worker-completion order is the nondeterminism this task fixes).
        built_deltas[page_num] = delta

    # Build phase: concurrent LLM work, no canonical mutation.
    tasks = [_build_page(p) for p in pages if p.get("page_number") not in recovered_pages]
    if tasks:
        # Dedicated pool sized from the same concurrency_limit as the semaphore, so each
        # admitted page gets a thread. Shut down (wait=True) when the build phase ends.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(concurrency_limit, 32),
            thread_name_prefix="graph-llm",
        ) as llm_executor:
            await asyncio.gather(*tasks)

    # Merge phase: SERIAL, in sorted page-number order (deterministic) over BOTH
    # recovered and freshly-built deltas. This single sorted pass is what makes a
    # resumed run produce the same graph as a fresh run. Each page's quality drives
    # checkpointing exactly as before (degraded/partial not COMPLETED).
    for page_num in sorted(built_deltas.keys()):
        delta = built_deltas[page_num]
        checksum = compute_page_delta_checksum(delta.to_dict())

        # 5. Ingest delta into central GraphMemory (canonical merge under lock).
        #    Recovered pages re-ingest idempotently (checksum match -> no-op).
        await manager.ingest_page_delta(delta, delta_checksum=checksum)

        # Recovered pages were already counted/checkpointed on their original run;
        # merge them (above) but don't re-checkpoint or double-count here.
        if page_num in recovered_pages:
            continue

        # 6. Mark page completed — BUT only if extraction was clean. A degraded
        #    page (truncated after retry / invalid JSON after re-ask) or a partial
        #    page (needed the salvage path) must NOT be checkpointed COMPLETED, so a
        #    resumed run re-processes it (audit #1, 2b/2c).
        page_quality = getattr(delta, "quality", "clean") or "clean"
        if page_quality == "degraded":
            pages_degraded.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="DEGRADED", owner=owner,
                                     error_info="truncated_or_invalid_after_retry",
                                     pipeline_version=pipeline_version)
        elif page_quality == "partial":
            pages_partial.append(page_num)
            pages_succeeded.append(page_num)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="PARTIAL", owner=owner,
                                     pipeline_version=pipeline_version)
        else:
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="COMPLETED", owner=owner, pipeline_version=pipeline_version)
            pages_succeeded.append(page_num)

    pages_succeeded.sort()
    pages_failed.sort()

    if pages_failed:
        logger.warning(
            "layer3.graph_memory_concurrent.partial_failures",
            failed_pages=pages_failed,
            succeeded_pages=pages_succeeded,
            errors=failed_errors,
        )

    # 7. Reconcile cross-page anaphoric references post-merge.
    #    NOTE: in concurrent mode this is effectively dead — workers build against an
    #    empty graph so the model emits no references to reconcile (restore-linking
    #    Step 0a). Retained unchanged; cross-page linking is restored by step 7b.
    await manager.reconcile_cross_references()

    # 7b. Deterministic post-merge cross-page linking (restore-linking task): add
    #     SAME_ENTITY_AS edges between same-entity nodes on different pages. No model
    #     calls; deterministic; gated by settings.graph_crosspage_linking (default on).
    if getattr(settings, "graph_crosspage_linking", True):
        manager.link_cross_page()

    # 8. Synthesize final target schema strictly from the complete Graph Memory
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), owner=owner, strategy="graph_memory_concurrent")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

    pages_degraded.sort()
    pages_partial.sort()

    logger.info(
        "layer3.graph_memory_concurrent.done",
        graph_stats=manager.graph.stats(),
        pages_total=total_pages,
        pages_succeeded=len(pages_succeeded),
        pages_failed=len(pages_failed),
        pages_degraded=len(pages_degraded),
        pages_partial=len(pages_partial),
    )

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_concurrent"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        # audit #1 (2b/2c): surface non-clean pages in the page results.
        graph_out["pages_degraded"] = len(pages_degraded)
        graph_out["degraded_pages"] = pages_degraded
        graph_out["pages_partial"] = len(pages_partial)
        graph_out["partial_pages"] = pages_partial
        graph_out["concurrency_limit"] = concurrency_limit
        graph_out["unresolved_reference_count"] = manager.unresolved_count
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

    return result


async def extract_with_graph_memory_batched(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    owner: str,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    epoch_size: int = 4,
    pipeline_version: str = "v1",
) -> BaseModel:
    """
    MODE C: EPOCH / BATCH MODE (Optional Strategy).
    Partitions pages into sequential epochs of size epoch_size.
    Within each epoch, workers execute concurrently; between epochs, PageDeltas are merged
    so the next epoch observes fresh canonical memory produced by all previous epochs.
    """
    from src.ai.layer3_extraction.graph_agent.agent import GraphExtractionAgent
    from src.ai.layer3_extraction.graph_agent.memory_manager import GraphMemoryManager
    from src.ai.layer3_extraction.graph_agent.models import PageDelta, validate_page_delta
    from src.ai.layer3_extraction.graph_agent.resolver import resolve_schema_from_graph
    from src.ai.layer3_extraction.storage import (
        compute_page_delta_checksum,
        save_document_graph,
        save_page_checkpoint,
    )
    from src.config.settings import settings

    pages: List[Dict] = []
    for i, p in enumerate(pages_md):
        if isinstance(p, dict):
            pages.append(p)
        else:
            pages.append({"markdown": p, "page_number": i + 1})

    total_pages = len(pages)
    anchor_types = getattr(settings, "graph_anchor_types", None) or []
    anchor_max_k = getattr(settings, "graph_anchor_max_k", 5)
    concurrency_limit = max(1, getattr(settings, "graph_concurrency_limit", 8))

    manager = GraphMemoryManager(
        anchor_types=anchor_types,
        anchor_max_k=anchor_max_k,
        schema=schema,
        pipeline_version=pipeline_version,
    )
    agent = GraphExtractionAgent(schema=schema, graph=manager.graph, llm=llm)

    semaphore = asyncio.Semaphore(concurrency_limit)
    pages_succeeded: List[int] = []
    pages_failed: List[int] = []

    # MEASUREMENT ONLY (audit finding #1): stamp owner/doc on Converse metric records.
    try:
        from src.adapters.llm import llm_metrics as _llm_metrics
        _llm_metrics.set_run_context(owner=owner, doc_id=doc_id)
    except Exception:
        pass

    # Partition into epochs
    epochs = [pages[i : i + epoch_size] for i in range(0, len(pages), epoch_size)]

    for epoch_idx, epoch_pages in enumerate(epochs):
        logger.info(
            "layer3.epoch.start",
            epoch_index=epoch_idx + 1,
            total_epochs=len(epochs),
            page_count=len(epoch_pages),
        )

        epoch_deltas: List[Tuple[PageDelta, str]] = []

        async def _process_epoch_page(page_data: Dict[str, Any]) -> None:
            page_num = page_data.get("page_number", 1)
            page_md = page_data.get("markdown", "")
            page_blocks = page_data.get("blocks", [])

            if not str(page_md or "").strip():
                pages_succeeded.append(page_num)
                return

            async with semaphore:
                context = await manager.get_context_for_page(page_num)
                try:
                    delta = await agent.build_page_delta_async(
                        page_number=page_num,
                        markdown=page_md,
                        total_pages=total_pages,
                        context=context,
                        blocks=page_blocks,
                    )
                except Exception as exc:
                    logger.error("layer3.epoch.worker_failed", page=page_num, error=str(exc))
                    pages_failed.append(page_num)
                    return

            val_res = validate_page_delta(delta)
            if not val_res.is_valid:
                logger.error("page.validation_failed", page=page_num, errors=val_res.errors)
                pages_failed.append(page_num)
                return

            delta_dict = delta.to_dict()
            chk = compute_page_delta_checksum(delta_dict)
            if doc_id:
                save_page_checkpoint(doc_id, page_num, status="DELTA_PERSISTED", owner=owner, page_delta_json=delta_dict, delta_checksum=chk)
            epoch_deltas.append((delta, chk))

        tasks = [_process_epoch_page(p) for p in epoch_pages]
        if tasks:
            await asyncio.gather(*tasks)

        # Merge epoch deltas into canonical memory before next epoch begins
        for d, chk in epoch_deltas:
            await manager.ingest_page_delta(d, delta_checksum=chk)
            if doc_id:
                save_page_checkpoint(doc_id, d.page_number, status="COMPLETED", owner=owner)
            pages_succeeded.append(d.page_number)

    await manager.reconcile_cross_references()
    result = resolve_schema_from_graph(graph=manager.graph, schema=schema, llm=llm)

    if doc_id:
        try:
            save_document_graph(doc_id=doc_id, graph_dict=manager.graph.snapshot(), owner=owner, strategy="graph_memory_batched")
        except Exception as exc:
            logger.warning("layer3.save_graph_failed", error=str(exc))

    if graph_out is not None:
        graph_out["strategy"] = "graph_memory_batched"
        graph_out["stats"] = manager.graph.stats()
        graph_out["snapshot"] = manager.graph.snapshot()
        graph_out["pages_total"] = total_pages
        graph_out["pages_succeeded"] = len(pages_succeeded)
        graph_out["pages_failed"] = len(pages_failed)
        graph_out["failed_pages"] = pages_failed
        graph_out["unresolved_reference_count"] = manager.unresolved_count
        graph_out["merge_ledger"] = [r.to_dict() for r in manager.get_merge_ledger()]

    return result


def _run_coroutine_sync(coro):
    """
    Safely execute an asyncio coroutine from synchronous code,
    preventing RuntimeError when an event loop is already running in the current thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def extract_document(
    pages_md: Union[List[str], List[Dict]],
    schema: type[BaseModel],
    llm: ExtractionLLMClient,
    owner: str,
    strategy: Optional[str] = None,
    graph_out: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseModel:
    """
    Unified entry point for Layer 3 extraction with strategy switch.
    Supported strategies:
    - "page_scan": Plain sequential page scanning with Scratchpad.
    - "graph_memory" / "graph_memory_ordered": Mode A (Ordered Context Mode).
    - "graph_memory_concurrent": Mode B (Concurrent Delta Mode).
    - "graph_memory_batched" / "graph_memory_epoch": Mode C (Epoch / Batch Mode).

    ``owner`` is REQUIRED (no default) and must be the authenticated tenant
    username (or "_admin" for background / admin paths, passed EXPLICITLY at the
    call site). It is forwarded to every graph-memory mode so checkpoints and
    extraction runs are keyed by tenant. Removing the former "_admin" default
    prevents a caller from silently writing another tenant's data as admin
    (tenant-isolation hardening).
    """
    from src.config.settings import settings

    selected_strategy = strategy or getattr(settings, "layer3_strategy", "graph_memory")

    if selected_strategy in ("graph_memory_concurrent",):
        return _run_coroutine_sync(
            extract_with_graph_memory_concurrent(
                pages_md=pages_md,
                schema=schema,
                llm=llm,
                graph_out=graph_out,
                doc_id=doc_id,
                owner=owner,
                **kwargs,
            )
        )
    elif selected_strategy in ("graph_memory_batched", "graph_memory_epoch"):
        return _run_coroutine_sync(
            extract_with_graph_memory_batched(
                pages_md=pages_md,
                schema=schema,
                llm=llm,
                graph_out=graph_out,
                doc_id=doc_id,
                owner=owner,
                **kwargs,
            )
        )
    elif selected_strategy in ("graph_memory", "graph_memory_ordered"):
        return extract_with_graph_memory(
            pages_md=pages_md,
            schema=schema,
            llm=llm,
            graph_out=graph_out,
            doc_id=doc_id,
            owner=owner,
            **kwargs,
        )
    return extract_by_page_scan(pages_md, schema, llm, graph_out=graph_out, **kwargs)
```

## File: schema_chatbot_v2/app/api/pipeline_routes.py
```python
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from PIL import Image

from app.core.activity_log import log_activity
from app.core.auth import get_current_user, require_admin
from app.storage.user_store import User

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "dataset_output"
SCHEMA_REGISTRY = ROOT / "schema_registry"

for d in (DATASET_DIR, OUTPUT_DIR, SCHEMA_REGISTRY):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, PageOutput, VLMAnalysis
from src.ai.layer1_routing.pipeline import process_document
from src.ai.layer3_extraction.ir_store import save_page_irs, delete_page_irs
from src.config.settings import settings as a_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
PIPELINE_AVAILABLE = True
router = APIRouter()


class JobControl:
    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.pause_event.set()  # set = running; cleared = paused
        self.kill_event = threading.Event()  # set = kill requested


_pipeline_jobs: Dict[str, Dict[str, Any]] = {}
_job_controls: Dict[str, JobControl] = {}

MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(20 * 1024 * 1024)))  # 20MB
RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_EXTRACT_MAX_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_EXTRACT_WINDOW_SECONDS", "60"))
_extract_rate_limits: Dict[str, List[float]] = {}
_rate_limit_lock = threading.Lock()


def check_user_extract_rate_limit(username: str) -> None:
    """
    Enforces a simple sliding window per-user rate limit on document extraction.
    """
    now = time.time()
    with _rate_limit_lock:
        history = _extract_rate_limits.setdefault(username, [])
        valid_history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(valid_history) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: maximum {RATE_LIMIT_MAX_REQUESTS} extraction requests per {RATE_LIMIT_WINDOW_SECONDS}s allowed."
            )
        valid_history.append(now)
        _extract_rate_limits[username] = valid_history


def reset_extract_rate_limits() -> None:
    """
    Resets in-memory extraction rate limits (for testing and isolation).
    """
    with _rate_limit_lock:
        _extract_rate_limits.clear()


async def validate_uploaded_pdf(file: UploadFile) -> Tuple[str, bytes]:
    """
    Validates that the upload is a non-empty .pdf without path traversal,
    and does not exceed MAX_UPLOAD_SIZE_BYTES.
    Returns (safe_filename, content_bytes).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")
    p = Path(file.filename)
    if p.name != file.filename or ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected.")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_BYTES} bytes."
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return p.name, content


RENDER_DPI = 150
MAT = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)


class MockLLMClient(LLMClient):
    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        raise RuntimeError(
            "MockLLMClient.classify_page called. Pages need a real VLM for scanned/mixed content."
        )

    def analyze_page(self, image_bytes: bytes, page_profile_hint: dict) -> VLMAnalysis:
        raise RuntimeError(
            "MockLLMClient.analyze_page called."
        )

    def transcribe_handwriting(self, image_bytes: bytes) -> Tuple[str, float]:
        raise RuntimeError(
            "MockLLMClient.transcribe_handwriting called."
        )


def _render_page(page: Any) -> Tuple[np.ndarray, bytes]:
    pix = page.get_pixmap(matrix=MAT, colorspace=pymupdf.csRGB)
    png = pix.tobytes("png")
    arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return arr, png


def _build_pages_for_pdf(pdf_path: Path, pdf_bytes: Optional[bytes] = None) -> List[Dict[str, Any]]:
    # Open from in-memory bytes (S3-sourced) when provided, else from a local path.
    if pdf_bytes is not None:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        arr, png = _render_page(page)
        pages.append({
            "page": page,
            "page_number": i + 1,
            "context": {
                "pdf_path": str(pdf_path),
                "image_array": arr,
                "image_bytes": png,
                "render_dpi": RENDER_DPI,
            },
            "image_bytes": png,
        })
    return pages


# ========================= Documents =========================


@router.get("/documents")
def list_documents(
    _: User = Depends(require_admin),
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    from src.ai.layer3_extraction import s3_documents as s3
    from src.ai.layer3_extraction.storage import list_markdown_records

    # Processed outputs are DB-backed (converted markdown records).
    output_records = list_markdown_records(owner=owner or "_admin")
    outputs = [
        {
            "name": o["name"],
            "size": o["size"],
            "modified": o["modified"],
            "schema_ref": o.get("schema_ref") or ({"schema_id": o["schema_id"]} if o.get("schema_id") else None),
        }
        for o in output_records
    ]
    # Stems that have a processed output, to compute has_output per document.
    processed_stems = set()
    for o in output_records:
        src = o.get("source_pdf") or ""
        processed_stems.add(Path(src).stem if src else Path(o["name"]).stem)

    def _has_output(name: str) -> bool:
        return Path(name).stem in processed_stems

    # Source PDFs are stored in S3 (shared documents prefix). Fall back to the
    # local dataset/ folder only when S3 is not configured (dev convenience).
    if s3.is_configured():
        docs = [
            {
                "name": d["name"],
                "size": d["size"],
                "modified": d["modified"],
                "has_output": _has_output(d["name"]),
                "source": "s3",
            }
            for d in sorted(s3.list_shared_documents(), key=lambda x: x["name"].lower())
        ]
    else:
        pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
        docs = [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "has_output": _has_output(p.name),
                "source": "disk",
            }
            for p in pdfs
        ]

    return {"documents": docs, "outputs": outputs}


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    from src.ai.layer3_extraction import s3_documents as s3
    saved: List[str] = []
    for f in files:
        safe_name, content = await validate_uploaded_pdf(f)
        content_type = f.content_type or "application/pdf"

        # Store the source PDF in S3 (shared documents prefix); fall back to disk.
        if s3.is_configured():
            s3.upload_shared_document(safe_name, content, content_type=content_type)
        else:
            (DATASET_DIR / safe_name).write_bytes(content)
        saved.append(safe_name)

        # Persist uploaded document into PostgreSQL documents table (shadow copy).
        try:
            from src.ai.layer3_extraction.storage import save_document
            doc_owner = "_admin" if getattr(user, "role", None) == Role.ADMIN else user.username
            save_document(filename=safe_name, file_bytes=content, owner=doc_owner, content_type=content_type)
        except Exception:
            pass
    return {"saved": saved, "count": len(saved)}


def _validate_doc_filename(filename: str) -> str:
    """
    Validates a document filename is a safe single-component .pdf name
    (no path traversal). Returns the sanitized name.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    p = Path(filename)
    if p.name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected.")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf documents are supported.")
    return p.name


@router.delete("/documents/{filename}")
def delete_document(filename: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    Delete a shared document PDF from S3 (admin only).
    """
    from src.ai.layer3_extraction import s3_documents as s3
    safe_name = _validate_doc_filename(filename)

    if s3.is_configured():
        if not s3.shared_document_exists(safe_name):
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        if not s3.delete_shared_document(safe_name):
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}' from storage.")
    else:
        target = DATASET_DIR / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")
        try:
            target.unlink()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete '{safe_name}': {exc}")

    logger.info("documents.deleted", doc_name=safe_name, by=_.username)
    return {"deleted": safe_name}


@router.get("/documents/output/{name}")
def download_output(name: str, owner: Optional[str] = None, _: User = Depends(require_admin)) -> Response:
    """
    Download the extracted-JSON output for a processed document (admin).
    `name` may be the output filename (e.g. foo.md / foo.extracted.json) or the
    source pdf/stem. Returns the extracted_data as a JSON attachment.
    """
    from src.ai.layer3_extraction.storage import get_latest_extracted_data
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid output name.")
    stem = Path(name).stem.replace(".extracted", "")
    eff_owner = owner or _lookup_doc_owner(name, stem) or "_admin"
    data = get_latest_extracted_data(stem, owner=eff_owner)
    if data is None and eff_owner != "_admin":
        data = get_latest_extracted_data(stem, owner="_admin")
    if data is None:
        raise HTTPException(status_code=404, detail=f"No extracted output found for '{name}'.")
    payload = json.dumps(data, indent=2, default=str)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{stem}.extracted.json"'},
    )


# ========================= Schemas =========================


@router.get("/schemas")
def list_schemas(_: User = Depends(require_admin)) -> Dict[str, Any]:
    # DB-backed: schemas are read from the PostgreSQL `schemas` table, not disk.
    from src.ai.layer3_extraction.storage import list_schema_records
    records = list_schema_records(owner=None)  # admin sees all
    schemas: List[Dict[str, Any]] = [
        {
            "file": f"{r['schema_id']}.json",
            "schema_id": r["schema_id"],
            "document_type": r["document_type"],
            "confirmed_at": r["confirmed_at"],
            "field_count": r["field_count"],
            "sample_documents": r.get("sample_documents", []),
        }
        for r in records
    ]
    return {"source": "database", "schemas": schemas}


@router.get("/schemas/{schema_id}")
def get_schema(schema_id: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    from src.ai.layer3_extraction.storage import get_schema_record
    record = get_schema_record(schema_id)
    if record is None:
        raise HTTPException(status_code=404, detail="schema not found")
    return record


# ========================= Pipeline =========================


@router.get("/pipeline/status")
def pipeline_status(_: User = Depends(require_admin)) -> Dict[str, Any]:
    routing_mode = getattr(a_settings, "routing_mode", None) if PIPELINE_AVAILABLE else None
    layer3_strategy = getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")
    return {
        "available": PIPELINE_AVAILABLE,
        "routing_mode": routing_mode,
        "layer3_strategy": layer3_strategy,
        "jobs": {
            k: {
                "job_id": k,
                "schema_id": v.get("schema_id"),
                "schema_file": v.get("schema_file"),
                "strategy": v.get("strategy", layer3_strategy),
                "status": v["status"],
                "created_at": v["created_at"],
                "total": len(v.get("targets", [])),
                "completed": len(v.get("successes", [])) + len(v.get("failures", [])),
            }
            for k, v in _pipeline_jobs.items()
        },
    }


@router.get("/pipeline/jobs/{job_id}")
def get_pipeline_job(job_id: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return _pipeline_jobs[job_id]


@router.post("/pipeline/jobs/{job_id}/pause")
def pause_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] not in ("running", "queued"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause job with status '{job['status']}'. Job must be 'running' or 'queued'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.clear()
    job["status"] = "paused"
    logger.info("pipeline.job_paused", job_id=job_id)
    log_activity(admin.username, "job_paused", {"job_id": job_id})
    return {"job_id": job_id, "status": "paused", "message": f"Job {job_id} paused."}


@router.post("/pipeline/jobs/{job_id}/resume")
def resume_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status '{job['status']}'. Job must be 'paused'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.pause_event.set()
    job["status"] = "running"
    logger.info("pipeline.job_resumed", job_id=job_id)
    log_activity(admin.username, "job_resumed", {"job_id": job_id})
    return {"job_id": job_id, "status": "running", "message": f"Job {job_id} resumed."}


@router.post("/pipeline/jobs/{job_id}/kill")
def kill_pipeline_job(job_id: str, admin: User = Depends(require_admin)) -> Dict[str, Any]:
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="job not found")
    job = _pipeline_jobs[job_id]
    if job["status"] in ("completed", "killed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot kill job with status '{job['status']}'."
        )
    ctrl = _job_controls.get(job_id)
    if ctrl:
        ctrl.kill_event.set()
        ctrl.pause_event.set()  # Unblock thread if it was paused
    job["status"] = "killed"
    if not job.get("finished_at"):
        job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    logger.info("pipeline.job_kill_requested", job_id=job_id)
    log_activity(admin.username, "job_killed", {"job_id": job_id})
    return {"job_id": job_id, "status": "killed", "message": f"Job {job_id} killed."}


@router.get("/pipeline/jobs/{job_id}/pdf", response_class=Response)
def download_job_pdf_report(job_id: str, user: User = Depends(get_current_user)) -> Response:
    """
    Download the generated PDF report for a job directly from PostgreSQL.
    If not yet stored in PostgreSQL, generate it now, save it, and stream it.
    """
    if job_id in _pipeline_jobs:
        job = _pipeline_jobs[job_id]
        if user.role != "admin" and job.get("owner") != user.username:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this job")

    from src.ai.layer3_extraction.storage import get_job_pdf, save_job_pdf
    from src.utils.job_pdf_report import generate_job_pdf

    res = get_job_pdf(job_id)
    if res:
        pdf_bytes, filename = res
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # If not stored yet in PostgreSQL, check if the job is in memory
    if job_id not in _pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job report not found in database or memory")

    job = _pipeline_jobs[job_id]
    try:
        pdf_bytes = generate_job_pdf(job)
        filename = f"{job_id}_report.pdf"
        save_job_pdf(job_id, pdf_bytes, filename)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("pipeline.job_pdf_generation_failed", job_id=job_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {exc}")


@router.post("/pipeline/run")
async def run_pipeline(
    schema_id: str = Form(...),
    documents: Optional[str] = Form(default=None),
    strategy: Optional[str] = Form(default=None),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Extraction pipeline unavailable (import failed)")

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    # DB-backed: load the confirmed schema from PostgreSQL, not the disk registry.
    from src.ai.layer3_extraction.storage import get_schema_record
    schema_record = get_schema_record(schema_id)
    if schema_record is None:
        raise HTTPException(status_code=404, detail=f"schema {schema_id} not found")
    # Display-only stand-in path; the worker only reads schema_path.name, never the file.
    schema_path = Path(f"{schema_id}.json")

    from src.ai.layer3_extraction import s3_documents as s3

    requested_names: Optional[List[str]] = None
    if documents:
        try:
            parsed = json.loads(documents)
            if isinstance(parsed, list):
                requested_names = [str(n) for n in parsed]
        except Exception:
            requested_names = None

    if s3.is_configured():
        # Resolve target PDFs from S3 (shared documents prefix) into local temp files.
        available = {d["name"]: d["key"] for d in s3.list_shared_documents()}
        names = requested_names if requested_names else sorted(available.keys())
        names = [n for n in names if n in available]
        if not names:
            raise HTTPException(status_code=400, detail="No documents to process")
        tmp_dir = Path(tempfile.mkdtemp(prefix="idp_docs_"))
        targets = []
        for n in names:
            data = s3.download_bytes_by_key(available[n])
            if not data:
                continue
            tp = tmp_dir / n
            tp.write_bytes(data)
            targets.append(tp)
        if not targets:
            raise HTTPException(status_code=400, detail="No documents to process")
    else:
        all_pdfs = sorted([p for p in DATASET_DIR.glob("*.pdf") if p.is_file()])
        if requested_names:
            selected = [DATASET_DIR / n for n in requested_names if (DATASET_DIR / n).exists()]
            targets = selected if selected else all_pdfs
        else:
            targets = all_pdfs
        if not targets:
            raise HTTPException(status_code=400, detail="No documents to process")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_id": schema_id,
        "schema_file": schema_path.name,
        "strategy": selected_strategy,
        "targets": [p.name for p in targets],
        "successes": [],
        "failures": [],
        "started_at": None,
        "finished_at": None,
        "wall_time_s": None,
        "owner": user.username,
        "current_document": None,
    }
    _pipeline_jobs[job_id] = job
    _job_controls[job_id] = JobControl()

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, targets, schema_path, schema_record, selected_strategy, user.username),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "targets": len(targets), "strategy": selected_strategy}


def _parse_pages_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Extract individual pages from an already-generated pipeline markdown file."""
    marker = re.compile(r"<!--\s*PAGE\s+(\d+)[^>]*-->", re.I)
    matches = list(marker.finditer(md_text))
    if not matches:
        return [{"markdown": md_text.strip(), "page_number": 1}]
    pages = []
    for i, m in enumerate(matches):
        pg_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end]
        close_match = re.search(rf"<!--\s*/PAGE\s+{pg_num}\s*-->", content, re.I)
        if close_match:
            content = content[:close_match.start()]
        else:
            content = re.sub(r"<!--\s*/PAGE\s+\d+\s*-->", "", content)
        pages.append({"markdown": content.strip(), "page_number": pg_num})
    return pages


def _run_pipeline_job(
    job_id: str,
    targets: List[Path],
    schema_path: Path,
    schema_record: Dict[str, Any],
    strategy: Optional[str] = None,
    owner: Optional[str] = None,
) -> None:
    job = _pipeline_jobs[job_id]
    ctrl = _job_controls.get(job_id)
    if not ctrl:
        ctrl = JobControl()
        _job_controls[job_id] = ctrl

    selected_strategy = strategy or job.get("strategy") or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")
    job["strategy"] = selected_strategy
    job_owner = owner or job.get("owner") or "_admin"
    job["owner"] = job_owner
    job["status"] = "running"
    job["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        from src.adapters.llm.factory import get_llm_client
        llm_client = get_llm_client()
    except Exception:
        llm_client = MockLLMClient()
    schema_id = schema_record["schema_id"]

    t_start = time.monotonic()
    for pdf in targets:
        job["current_document"] = pdf.name
        # Check before starting next PDF: pause wait loop and kill check
        while not ctrl.kill_event.is_set():
            if ctrl.pause_event.is_set():
                break
            time.sleep(0.2)

        if ctrl.kill_event.is_set():
            logger.info("pipeline.job_killed", job_id=job_id, pdf=pdf.name)
            job["status"] = "killed"
            job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            job["wall_time_s"] = round(time.monotonic() - t_start, 2)
            return

        owner = job.get("owner")
        job_out_dir = resolve_owner_output_dir(owner) if owner else OUTPUT_DIR
        job_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Discover existing converted markdown (disk or DB) to avoid destructive overwrite
        existing_md_text = None
        md_path = job_out_dir / f"{pdf.stem}.md"
        cand_files = [
            md_path,
            OUTPUT_DIR / f"{pdf.stem}.md",
            DATASET_DIR / f"{pdf.stem}.md",
        ]
        for cand in cand_files:
            if cand.exists() and cand.is_file():
                try:
                    c_txt = cand.read_text(encoding="utf-8")
                    if len(c_txt.strip()) > 100:
                        existing_md_text = c_txt
                        break
                except Exception:
                    pass

        if not existing_md_text:
            try:
                from src.ai.layer3_extraction.storage import get_markdown_record
                rec = get_markdown_record(pdf.name, owner=job_owner) or get_markdown_record(f"{pdf.stem}.md", owner=job_owner)
                if rec and rec.get("markdown_content") and len(rec["markdown_content"].strip()) > 100:
                    existing_md_text = rec["markdown_content"]
            except Exception:
                pass

        t0 = time.monotonic()
        try:
            results = None
            try:
                pages = _build_pages_for_pdf(pdf)
                results = process_document(
                    pages=pages,
                    llm_client=llm_client,
                    document_name=pdf.name,
                    document_id=f"{job_id}-{pdf.stem}",
                    write_output=True,
                    output_dir=str(job_out_dir),
                    overwrite=True,
                    ir_owner=job_owner,
                )
            except Exception as proc_exc:
                logger.warning("pipeline.process_document_failed", pdf=pdf.name, error=str(proc_exc))
                if not existing_md_text:
                    raise

            # Persist DocumentIR per owner (doc_id == full filename, matching the
            # Query Bot / graph store doc_id). Never fails the job; records a flag.
            ir_persisted = False
            try:
                from src.ai.schemas.document_ir import PageIR as _PageIR
                _page_irs = [
                    _PageIR(
                        page_number=o.page_number,
                        blocks=list(getattr(o, "blocks", []) or []),
                        source_engine=(o.engines_used[0] if getattr(o, "engines_used", None) else ""),
                    )
                    for o, _m in (results or [])
                ]
                if _page_irs:
                    save_page_irs(owner=job_owner, doc_id=pdf.name, pages=_page_irs)
                    ir_persisted = True
            except Exception as _ir_exc:
                logger.warning("pipeline.job.ir_persist_failed", pdf=pdf.name, error=str(_ir_exc))

            outputs = [r[0] for r in results] if results else []
            total_chars = sum(len(str(getattr(o, "markdown", "") or "").strip()) for o in outputs)
            avg_conf = sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0

            pages_for_layer3 = [{"markdown": r[0].markdown, "page_number": r[0].page_number} for r in results] if results else []

            # If local conversion yielded 0 chars (e.g. paddleocr missing in web container),
            # restore pre-existing valid markdown so Layer 3 receives full document text
            if total_chars == 0 and existing_md_text:
                logger.warning(
                    "pipeline.conversion_produced_0_chars_restoring_existing_md",
                    document=pdf.name,
                    existing_chars=len(existing_md_text),
                )
                md_path.write_text(existing_md_text, encoding="utf-8")
                parsed_pages = _parse_pages_from_markdown(existing_md_text)
                if parsed_pages and any(p.get("markdown") for p in parsed_pages):
                    pages_for_layer3 = parsed_pages
                    total_chars = sum(len(str((p.get("markdown") if isinstance(p, dict) else "") or "").strip()) for p in parsed_pages)
                    conf_match = re.search(r"Avg conf\s*:\s*([\d\.]+)", existing_md_text)
                    if conf_match:
                        try:
                            avg_conf = float(conf_match.group(1))
                        except Exception:
                            avg_conf = 0.95
                    else:
                        avg_conf = 0.95

            ref_path = job_out_dir / f"{pdf.stem}.schema_ref.json"
            ref_dict = {
                "source_pdf": pdf.name,
                "output_md": md_path.name,
                "schema_id": schema_id,
                "schema_registry_file": schema_path.name,
                "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                "pages": [
                    {
                        "page_number": p.get("page_number", idx + 1) if isinstance(p, dict) else idx + 1,
                        "engines_used": getattr(results[idx][0], "engines_used", ["paddleocr_printed"]) if (results and idx < len(results)) else ["paddleocr_printed"],
                        "capabilities": getattr(results[idx][0], "capabilities", ["has_printed_scan"]) if (results and idx < len(results)) else ["has_printed_scan"],
                        "confidence": avg_conf if total_chars > 0 and (not results or not any(str(getattr(r[0], "markdown", "") or "").strip() for r in results)) else (results[idx][0].confidence if (results and idx < len(results)) else avg_conf),
                        "escalated": getattr(results[idx][0], "escalated", False) if (results and idx < len(results)) else False,
                        "low_confidence": False if total_chars > 0 else (results[idx][0].low_confidence if (results and idx < len(results)) else True),
                        "chars": (
                            p.get("chars")
                            if isinstance(p, dict) and isinstance(p.get("chars"), int)
                            else (
                                p.get("markdown")
                                if isinstance(p, dict) and isinstance(p.get("markdown"), int)
                                else len(str((p.get("markdown") if isinstance(p, dict) else "") or "").strip())
                            )
                        ),
                    }
                    for idx, p in enumerate(pages_for_layer3)
                ],
            }
            if job.get("owner"):
                ref_dict["owner"] = job["owner"]
            ref_path.write_text(
                json.dumps(ref_dict, indent=2) + "\n",
                encoding="utf-8",
            )

            # Persist converted Markdown (.md) into PostgreSQL document_markdowns table
            try:
                from src.ai.layer3_extraction.storage import save_markdown_record
                md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                save_markdown_record(
                    doc_id=pdf.name,
                    md_filename=md_path.name,
                    markdown_content=md_text,
                    owner=job_owner,
                    schema_id=schema_id,
                    schema_ref_json=ref_dict,
                    page_count=len(pages_for_layer3),
                    pages_json=pages_for_layer3,
                )
            except Exception:
                pass

            elapsed = time.monotonic() - t0

            # ================= Automatically Continue to Layer 3 Extraction =================
            extracted_data = None
            db_id = None
            db_error = None
            extract_elapsed = 0.0
            graph_meta: Dict[str, Any] = {}

            try:
                from src.adapters.llm.extraction_factory import get_extraction_client
                from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
                from src.ai.layer3_extraction.schema_validation import extract_with_retry
                from src.ai.layer3_extraction.storage import init_db, save_extraction_run, save_document_graph
                from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

                raw_fields = (schema_record.get("schema") or {}).get("fields", [])
                fields_in = []
                for f in raw_fields:
                    fname = f.get("name") if isinstance(f, dict) else str(f)
                    fdesc = f.get("description", "") if isinstance(f, dict) else ""
                    if fname:
                        fields_in.append(SchemaFieldIn(name=fname, description=fdesc))

                if fields_in:
                    dynamic_schema = build_dynamic_schema(fields_in)
                    t_ext0 = time.monotonic()
                    extraction_llm = get_extraction_client()

                    extracted_result = extract_with_retry(
                        lambda: extract_document(
                            pages_for_layer3,
                            dynamic_schema,
                            extraction_llm,
                            strategy=selected_strategy,
                            graph_out=graph_meta,
                            owner=job_owner,
                        ),
                        dynamic_schema,
                    )
                    extract_elapsed = round(time.monotonic() - t_ext0, 2)
                    extracted_data = extracted_result.model_dump()

                    # Write dataset_output/<owner>/<stem>.extracted.json
                    json_out_path = job_out_dir / f"{pdf.stem}.extracted.json"
                    json_payload = {
                        "source_pdf": pdf.name,
                        "schema_used": schema_id,
                        "strategy_used": selected_strategy,
                        "document_type": schema_record.get("document_type") or (schema_record.get("schema") or {}).get("document_type"),
                        "processing_time_seconds": extract_elapsed,
                        "extracted_data": extracted_data,
                    }
                    if graph_meta:
                        json_payload["graph_memory"] = graph_meta
                    if job.get("owner"):
                        json_payload["owner"] = job["owner"]

                    json_out_path.write_text(
                        json.dumps(json_payload, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    # Save file checkpoint: dataset_output/<owner>/<stem>.graph.json (Requirement 7)
                    if graph_meta and graph_meta.get("snapshot"):
                        try:
                            graph_out_path = job_out_dir / f"{pdf.stem}.graph.json"
                            graph_out_path.write_text(
                                json.dumps(graph_meta["snapshot"], indent=2) + "\n",
                                encoding="utf-8",
                            )
                        except Exception as gf_exc:
                            logger.warning("pipeline.extract.graph_file_save_failed", pdf=pdf.name, error=str(gf_exc))

                    # Persist to PostgreSQL (Requirement 6)
                    try:
                        init_db()
                        db_id = save_extraction_run(
                            doc_id=pdf.name,
                            page_count=len(pages_for_layer3),
                            schema_name=schema_id,
                            result_json=extracted_data,
                            owner=job_owner,
                            llm_provider=a_settings.extraction_backend,
                            model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                            processing_time_seconds=extract_elapsed,
                            page_outputs=outputs if total_chars > 0 and results and any(str(getattr(r[0], "markdown", "") or "").strip() for r in results) else None,
                        )
                        if graph_meta and graph_meta.get("snapshot"):
                            save_document_graph(
                                doc_id=pdf.name,
                                graph_dict=graph_meta["snapshot"],
                                job_id=job_id,
                                owner=job.get("owner"),
                                schema_id=schema_id,
                                strategy=selected_strategy,
                            )
                            logger.info("pipeline.extract.graph_persisted", pdf=pdf.name, job_id=job_id)
                    except Exception as db_exc:
                        db_error = str(db_exc)
                        logger.warning("pipeline.extract.db_save_failed", pdf=pdf.name, error=str(db_exc))
            except Exception as ext_exc:
                logger.error("pipeline.layer3_extraction_failed", pdf=pdf.name, error=str(ext_exc))

            job["successes"].append({
                "pdf": pdf.name,
                "md": md_path.name,
                "schema_ref": ref_path.name,
                "avg_conf": round(avg_conf, 3),
                "chars": total_chars,
                "pages": len(pages_for_layer3),
                "elapsed_s": round(elapsed, 2),
                "strategy": selected_strategy,
                "extracted_json": f"{pdf.stem}.extracted.json" if extracted_data else None,
                "extracted_data": extracted_data,
                "graph_stats": graph_meta.get("stats") if graph_meta else None,
                "graph_nodes": len(graph_meta.get("snapshot", {}).get("nodes", [])) if graph_meta else 0,
                "graph_edges": len(graph_meta.get("snapshot", {}).get("edges", [])) if graph_meta else 0,
                "graph_memory": graph_meta if graph_meta else None,
                "extract_elapsed_s": extract_elapsed,
                "db_run_id": db_id,
                "db_error": db_error,
                "ir_persisted": ir_persisted,
            })
        except Exception as exc:
            logger.error("pipeline.doc_failed", pdf=pdf.name, error=str(exc), error_type=type(exc).__name__)
            job["failures"].append({
                "pdf": pdf.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(time.monotonic() - t0, 2),
            })

        # Check after PDF completed if kill was requested during doc processing
        if ctrl.kill_event.is_set():
            logger.info("pipeline.job_killed_after_doc", job_id=job_id, pdf=pdf.name)
            job["status"] = "killed"
            job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            job["wall_time_s"] = round(time.monotonic() - t_start, 2)
            return

    if ctrl.kill_event.is_set():
        job["status"] = "killed"
    else:
        job["status"] = "completed"
    job["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    job["wall_time_s"] = round(time.monotonic() - t_start, 2)

    # Automatically generate full JSON PDF report and persist into PostgreSQL
    try:
        from src.ai.layer3_extraction.storage import save_job_pdf
        from src.utils.job_pdf_report import generate_job_pdf
        pdf_bytes = generate_job_pdf(job)
        save_job_pdf(job_id=job_id, pdf_bytes=pdf_bytes, filename=f"{job_id}_report.pdf")
        logger.info("pipeline.job_pdf_auto_saved", job_id=job_id, size=len(pdf_bytes))
    except Exception as pdf_exc:
        logger.warning("pipeline.job_pdf_auto_save_failed", job_id=job_id, error=str(pdf_exc))


# ========================= Layer 3 Extraction =========================


# _parse_pages_from_markdown is defined above _run_pipeline_job



_SCHEMA_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-]+$")


def resolve_owner_output_dir(owner: str) -> Path:
    """
    Safely resolves OUTPUT_DIR / owner and guarantees it stays within OUTPUT_DIR.
    Normalizes owner to lowercase and rejects consecutive dots or traversal attempts.
    """
    if not owner or not isinstance(owner, str):
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    owner_clean = owner.strip().lower()
    if not owner_clean or ".." in owner_clean or "/" in owner_clean or "\\" in owner_clean:
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    owner_dir = (OUTPUT_DIR / owner_clean).resolve()
    if not owner_dir.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected in owner directory.")
    return owner_dir


def validate_md_filename(md_name: str, owner: Optional[str] = None) -> Path:
    """
    Validates that md_name is a strict single-component filename ending in .md
    and resolving strictly within OUTPUT_DIR or an owner subfolder.
    """
    if not md_name or not isinstance(md_name, str):
        raise HTTPException(status_code=400, detail="Invalid md_name.")
    p = Path(md_name)
    if p.suffix.lower() != ".md" or p.name != md_name or ".." in md_name or "/" in md_name or "\\" in md_name:
        raise HTTPException(status_code=400, detail="Invalid md_name: must be a simple filename ending with .md")

    # 1. If owner is provided, search exclusively in owner's directory first
    if owner:
        owner_dir = resolve_owner_output_dir(owner)
        owner_file = (owner_dir / p).resolve()
        if not owner_file.is_relative_to(OUTPUT_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Path traversal detected.")
        if owner_file.is_file():
            return owner_file

    # 2. Check root OUTPUT_DIR
    resolved = (OUTPUT_DIR / p).resolve()
    if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected.")
    if resolved.is_file():
        return resolved

    # 3. If not in root and no owner specified, search owner subdirectories
    cands = [cand.resolve() for cand in OUTPUT_DIR.glob(f"*/{p.name}") if cand.is_file() and cand.resolve().is_relative_to(OUTPUT_DIR.resolve())]
    if len(cands) == 1:
        return cands[0]
    elif len(cands) > 1:
        owners = sorted(list(set(c.parent.name for c in cands)))
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous md_name '{md_name}' found in multiple owner directories: {owners}. Specify owner or document path explicitly."
        )

    return resolved


def validate_doc_id(doc_id: str) -> str:
    """
    Validates that doc_id is a strict single-component document identifier or filename
    and resolves strictly within OUTPUT_DIR without directory escape.
    """
    if not doc_id or not isinstance(doc_id, str):
        raise HTTPException(status_code=400, detail="Invalid doc_id.")
    p = Path(doc_id)
    if p.name != doc_id or ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Path traversal detected in doc_id.")
    resolved = (OUTPUT_DIR / p).resolve()
    if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Path traversal detected in doc_id.")
    return doc_id


def load_strict_schema_record(schema_id: str, owner: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Validates schema_id against strict regex ^[A-Za-z0-9_\-]+$ and loads the
    confirmed schema record from the PostgreSQL `schemas` table (DB-backed).

    Tenant isolation: when ``owner`` is provided (always, for user-facing
    endpoints), the resolved schema must belong to that owner. This is enforced
    in BOTH the DB path (via get_schema_record(..., owner=owner)) AND this
    function's own schema_registry/ DISK FALLBACK below. A schema whose stored
    ``owner`` is set and differs from the caller is treated as not-found (404),
    so one tenant can never resolve another tenant's schema by guessing its
    schema_id. NULL-owner records are shared/admin and remain visible to all.
    Pass owner=None only on admin-gated code paths.
    """
    if not schema_id or not _SCHEMA_ID_REGEX.fullmatch(schema_id):
        raise HTTPException(status_code=400, detail=f"Invalid schema_id format: '{schema_id}'")
    from src.ai.layer3_extraction.storage import get_schema_record
    record = get_schema_record(schema_id, owner=owner)
    if record is None:
        reg_file = SCHEMA_REGISTRY / f"{schema_id}.json"
        if reg_file.is_file():
            try:
                data = json.loads(reg_file.read_text(encoding="utf-8"))
                # Owner-scope the disk fallback: deny if the file is owned by a
                # different tenant. NULL/absent owner == shared, visible to all.
                if owner is not None:
                    rec_owner = data.get("owner")
                    if rec_owner is not None and rec_owner != owner:
                        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found.")
                return data
            except HTTPException:
                raise
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found.")
    return record


resolve_strict_schema_file = load_strict_schema_record


@router.get("/pipeline/outputs")
def list_pipeline_outputs(_: User = Depends(require_admin)) -> Dict[str, Any]:
    """
    List all documents that have already finished Layer 1 & 2 processing,
    along with their bound target schema and whether Layer 3 extracted JSON exists.
    """
    outputs = sorted([p for p in OUTPUT_DIR.glob("*.md") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for md_path in outputs:
        ref_path = md_path.with_suffix(".schema_ref.json")
        ext_path = md_path.with_name(f"{md_path.stem}.extracted.json")
        ref_data: Dict[str, Any] = {}
        if ref_path.exists():
            try:
                ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
            except Exception:
                ref_data = {}

        schema_id = ref_data.get("schema_id")
        doc_type = ref_data.get("document_type")
        page_count = len(ref_data.get("pages", [])) if ref_data.get("pages") else None

        # Check schema (DB-backed) for field count if available
        field_count = None
        if schema_id:
            try:
                from src.ai.layer3_extraction.storage import get_schema_record
                s_data = get_schema_record(schema_id)
                if s_data:
                    field_count = len((s_data.get("schema") or {}).get("fields", []))
                    if not doc_type:
                        doc_type = s_data.get("document_type")
            except Exception:
                pass

        has_extracted = ext_path.exists()
        extracted_data = None
        if has_extracted:
            try:
                extracted_data = json.loads(ext_path.read_text(encoding="utf-8")).get("extracted_data")
            except Exception:
                pass

        items.append({
            "md_name": md_path.name,
            "stem": md_path.stem,
            "source_pdf": ref_data.get("source_pdf", f"{md_path.stem}.pdf"),
            "schema_id": schema_id,
            "document_type": doc_type or "Document",
            "field_count": field_count,
            "page_count": page_count,
            "modified": datetime.fromtimestamp(md_path.stat().st_mtime).isoformat(),
            "has_extracted": has_extracted,
            "extracted_data": extracted_data,
        })
    return {"outputs": items}


@router.post("/pipeline/extract/from-output")
async def extract_from_existing_output(
    md_name: str = Form(...),
    strategy: Optional[str] = Form(default=None),
    owner: Optional[str] = Form(default=None),
    user: User = Depends(require_admin),
) -> Dict[str, Any]:
    check_user_extract_rate_limit(user.username)
    """
    Run Layer 3 extraction directly on an ALREADY-PROCESSED pipeline document.
    Automatically retrieves the converted Markdown and the schema bound to it during Run Pipeline.
    Saves the extracted structured JSON to dataset_output/<stem>.extracted.json and stores in PostgreSQL.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)
    if not is_admin:
        if not owner or owner != user.username:
            raise HTTPException(status_code=403, detail="Access denied: non-admin users may only extract from their own output partition.")

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    md_path = validate_md_filename(md_name, owner=owner)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{md_name}' not found in dataset_output.")

    ref_path = md_path.with_suffix(".schema_ref.json")
    if not ref_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No schema reference found for '{md_name}'. Make sure the pipeline was run with a target schema.",
        )

    ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
    known_owner = owner or ref_data.get("owner") or _lookup_doc_owner(ref_data.get("source_pdf", md_path.stem), md_path.stem, username=owner)
    schema_id = ref_data.get("schema_id")
    if not schema_id:
        raise HTTPException(status_code=400, detail=f"No schema_id recorded in {ref_path.name}")

    # Owner-scoped: resolve the schema as the document's owner (non-admin callers
    # are already constrained above to owner == their own username).
    schema_json = load_strict_schema_record(schema_id, owner=known_owner)
    raw_fields = (schema_json.get("schema") or {}).get("fields", [])
    fields_in: List[SchemaFieldIn] = []
    for f in raw_fields:
        name = f.get("name") if isinstance(f, dict) else str(f)
        desc = f.get("description", "") if isinstance(f, dict) else ""
        if name:
            fields_in.append(SchemaFieldIn(name=name, description=desc))

    if not fields_in:
        raise HTTPException(status_code=400, detail=f"Schema '{schema_id}' contains no field definitions.")

    dynamic_schema = build_dynamic_schema(fields_in)
    md_content = md_path.read_text(encoding="utf-8")
    if len(md_content.strip()) < 100:
        cand = OUTPUT_DIR / md_path.name
        if cand.exists() and cand != md_path:
            try:
                c_txt = cand.read_text(encoding="utf-8")
                if len(c_txt.strip()) > 100:
                    md_content = c_txt
                    md_path.write_text(md_content, encoding="utf-8")
            except Exception:
                pass
        if len(md_content.strip()) < 100:
            try:
                from src.ai.layer3_extraction.storage import get_markdown_record
                rec = get_markdown_record(md_path.stem, owner=user.username) or get_markdown_record(f"{md_path.stem}.pdf", owner=user.username)
                if rec and rec.get("markdown_content") and len(rec["markdown_content"].strip()) > 100:
                    md_content = rec["markdown_content"]
                    md_path.write_text(md_content, encoding="utf-8")
            except Exception:
                pass
    pages = _parse_pages_from_markdown(md_content)

    def _execute():
        t_start = time.time()
        extraction_llm = get_extraction_client()
        graph_meta: Dict[str, Any] = {}
        extracted_result = extract_with_retry(
            lambda: extract_document(pages, dynamic_schema, extraction_llm, strategy=selected_strategy, graph_out=graph_meta, owner=known_owner or "_admin"),
            dynamic_schema,
        )
        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        # Save to output dir (same dir as source md)
        stem = md_path.stem
        out_dir = md_path.parent
        json_out_path = out_dir / f"{stem}.extracted.json"
        out_payload = {
            "source_doc": md_name,
            "source_pdf": ref_data.get("source_pdf", f"{stem}.pdf"),
            "schema_used": schema_id,
            "strategy_used": selected_strategy,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
        }
        if graph_meta:
            out_payload["graph_memory"] = graph_meta
        if known_owner:
            out_payload["owner"] = known_owner
        json_out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

        if known_owner and not ref_data.get("owner"):
            ref_data["owner"] = known_owner
            try:
                ref_path.write_text(json.dumps(ref_data, indent=2) + "\n", encoding="utf-8")
            except Exception as e_ref:
                logger.warning("pipeline.extract_from_output.schema_ref_update_failed", error=str(e_ref))

        # Save to PostgreSQL
        db_id = None
        db_error = None
        try:
            init_db()
            page_outputs = []
            for p_info in ref_data.get("pages", []):
                page_outputs.append(PageOutput(
                    page_number=p_info.get("page_number", 1),
                    markdown="",
                    engines_used=p_info.get("engines_used", ["pipeline_output"]),
                    confidence=p_info.get("confidence", 0.95),
                    capabilities=p_info.get("capabilities", []),
                    escalated=p_info.get("escalated", False),
                    escalation_attempts=0,
                    low_confidence=p_info.get("low_confidence", False),
                ))

            eff_run_owner = known_owner or user.username
            db_id = save_extraction_run(
                doc_id=ref_data.get("source_pdf", md_name),
                page_count=len(pages),
                schema_name=schema_id,
                result_json=result_dict,
                owner=eff_run_owner,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                processing_time_seconds=elapsed,
                page_outputs=page_outputs if page_outputs else None,
            )
        except Exception as exc:
            db_error = str(exc)
            logger.warning("pipeline.extract_from_output.db_save_failed", error=str(exc))

        return {
            "success": True,
            "doc_id": ref_data.get("source_pdf", md_name),
            "schema": schema_id,
            "document_type": ref_data.get("document_type", "Document"),
            "strategy": selected_strategy,
            "data": result_dict,
            "graph_stats": graph_meta.get("stats") if graph_meta else None,
            "graph_memory": graph_meta if graph_meta else None,
            "meta": {
                "pages_processed": len(pages),
                "processing_time_seconds": elapsed,
                "strategy": selected_strategy,
                "source_md": md_name,
                "output_json": f"{stem}.extracted.json",
                "llm_provider": a_settings.extraction_backend,
                "saved_to_db": db_id is not None,
                "db_run_id": db_id,
                "db_error": db_error,
            }
        }

    try:
        return await run_in_threadpool(_execute)
    except Exception as exc:
        logger.exception("pipeline.extract_from_output.failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


@router.post("/pipeline/extract")
async def extract_uploaded_document(
    file: UploadFile = File(...),
    schema_id: Optional[str] = Form(default=None),
    raw_schema: Optional[str] = Form(default=None),
    strategy: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    End-to-End Extraction (single PDF upload):
      1. Receives PDF file + target schema.
      2. Runs Layer 1 (inspection & routing) + Layer 2 (conversion & OCR).
      3. Saves converted markdown to dataset_output/<stem>.md.
      4. Runs Layer 3 (Sarvam/Ollama page-by-page field extraction with scratchpad & early stopping, or Graph Memory).
      5. Saves output JSON to dataset_output/<stem>.extracted.json.
      6. Persists extraction run into PostgreSQL database.
      7. Returns structured JSON result and metadata.
    """
    from starlette.concurrency import run_in_threadpool
    from src.adapters.llm.extraction_factory import get_extraction_client
    from src.adapters.llm.factory import get_llm_client
    from src.ai.layer3_extraction.extractor import extract_by_page_scan, extract_document
    from src.ai.layer3_extraction.page_loader import load_pages_with_confidence
    from src.ai.layer3_extraction.schema_validation import extract_with_retry
    from src.ai.layer3_extraction.storage import init_db, save_extraction_run
    from src.api.dynamic_schema import SchemaFieldIn, build_dynamic_schema

    selected_strategy = strategy or getattr(a_settings, "layer3_strategy", "graph_memory_concurrent")

    # Rate limiting
    check_user_extract_rate_limit(user.username)

    # Allowed extension, filename, and size validation
    filename, pdf_bytes = await validate_uploaded_pdf(file)

    # 1. Resolve Target Schema
    fields_in: List[SchemaFieldIn] = []
    schema_name = "custom_schema"
    if schema_id:
        # Owner-scoped: a user may only resolve their own (or a shared) schema.
        schema_data = load_strict_schema_record(schema_id, owner=user.username)
        schema_name = schema_data.get("schema_id", schema_id)
        raw_fields = (schema_data.get("schema") or {}).get("fields", [])
        for f in raw_fields:
            name = f.get("name") if isinstance(f, dict) else str(f)
            desc = f.get("description", "") if isinstance(f, dict) else ""
            if name:
                fields_in.append(SchemaFieldIn(name=name, description=desc))
    elif raw_schema:
        try:
            parsed = json.loads(raw_schema)
            if isinstance(parsed, dict) and "fields" in parsed:
                parsed = parsed["fields"]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        fields_in.append(SchemaFieldIn(name=item.get("name", ""), description=item.get("description", "")))
                    elif isinstance(item, str):
                        fields_in.append(SchemaFieldIn(name=item, description=""))
            schema_name = ",".join(f.name for f in fields_in)[:100]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid raw_schema format: {exc}")

    if not fields_in:
        raise HTTPException(status_code=400, detail="No target schema provided. Please select a schema or specify fields.")

    dynamic_schema = build_dynamic_schema(fields_in)

    # Store the uploaded source PDF in S3 (shared documents prefix) for persistence.
    from src.ai.layer3_extraction import s3_documents as s3
    if s3.is_configured():
        s3.upload_shared_document(filename, pdf_bytes, content_type=file.content_type or "application/pdf")

    # Persist uploaded document into PostgreSQL documents table (shadow copy)
    try:
        from src.ai.layer3_extraction.storage import save_document
        save_document(filename=filename, file_bytes=pdf_bytes, owner=user.username, content_type=file.content_type or "application/pdf")
    except Exception:
        pass

    def _execute():
        t_start = time.time()

        # Owner-scoped output dir: the same filename for two owners must never
        # collide on disk. Reuse the existing per-owner path helper + traversal
        # guard (resolve_owner_output_dir). This is a normal (non-admin) route,
        # so the owner is always the authenticated user.
        owner_out_dir = resolve_owner_output_dir(user.username)
        owner_out_dir.mkdir(parents=True, exist_ok=True)

        # Render directly from the in-memory upload bytes (no local disk write).
        pages = _build_pages_for_pdf(Path(filename), pdf_bytes=pdf_bytes)
        try:
            llm_client = get_llm_client()
        except Exception:
            llm_client = MockLLMClient()

        results = process_document(
            pages=pages,
            llm_client=llm_client,
            document_name=filename,
            document_id=f"extract-{uuid.uuid4().hex[:6]}",
            write_output=True,
            output_dir=str(owner_out_dir),
            overwrite=True,
            ir_owner=user.username,
        )

        # Persist DocumentIR per owner. doc_id == the FULL filename, matching the
        # doc_id the Query Bot / graph store use for the same document (save_document,
        # save_document_graph, and the query-bot all key on `filename`). This never
        # fails the request; a metadata-only warning is logged and ir_persisted is
        # surfaced in the response meta.
        ir_persisted = False
        try:
            from src.ai.schemas.document_ir import PageIR as _PageIR
            _page_irs = [
                _PageIR(
                    page_number=o.page_number,
                    blocks=list(getattr(o, "blocks", []) or []),
                    source_engine=(o.engines_used[0] if getattr(o, "engines_used", None) else ""),
                )
                for o, _m in results
            ]
            save_page_irs(owner=user.username, doc_id=filename, pages=_page_irs)
            ir_persisted = True
        except Exception as _ir_exc:
            logger.warning("pipeline.extract.ir_persist_failed", document=filename, error=str(_ir_exc))

        stem = Path(filename).stem
        md_file = owner_out_dir / f"{stem}.md"

        # Persist converted Markdown (.md) into PostgreSQL document_markdowns table
        try:
            from src.ai.layer3_extraction.storage import save_markdown_record
            md_text = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
            save_markdown_record(
                doc_id=filename,
                md_filename=md_file.name,
                markdown_content=md_text,
                owner=user.username,
                schema_id=schema_id or schema_name,
                page_count=len(results),
                pages_json=[{"page_number": r[0].page_number, "markdown": r[0].markdown} for r in results],
            )
        except Exception:
            pass

        page_outputs = [r[0] for r in results]
        pages_md = load_pages_with_confidence(page_outputs)

        extraction_llm = get_extraction_client()
        graph_meta: Dict[str, Any] = {}
        extracted_result = extract_with_retry(
            lambda: extract_document(pages_md, dynamic_schema, extraction_llm, strategy=selected_strategy, graph_out=graph_meta, owner=user.username),
            dynamic_schema,
        )

        elapsed = round(time.time() - t_start, 2)
        result_dict = extracted_result.model_dump()

        json_out_path = owner_out_dir / f"{stem}.extracted.json"
        out_payload = {
            "source_pdf": filename,
            "schema_used": schema_name,
            "strategy_used": selected_strategy,
            "processing_time_seconds": elapsed,
            "extracted_data": result_dict,
            "owner": user.username,
        }
        if graph_meta:
            out_payload["graph_memory"] = graph_meta
            try:
                from src.ai.layer3_extraction.storage import save_document_graph
                save_document_graph(
                    doc_id=filename,
                    job_id=None,
                    schema_id=schema_name,
                    strategy=selected_strategy,
                    graph_dict=graph_meta,
                    owner=user.username,
                )
            except Exception as g_err:
                logger.warning("pipeline.extract.save_graph_failed", error=str(g_err))
        json_out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

        db_id = None
        db_error = None
        try:
            init_db()
            db_id = save_extraction_run(
                doc_id=filename,
                page_count=len(page_outputs),
                schema_name=schema_name,
                result_json=result_dict,
                owner=user.username,
                llm_provider=a_settings.extraction_backend,
                model_name=a_settings.bedrock_model_id if a_settings.extraction_backend == "bedrock" else (a_settings.sarvam_model_name if a_settings.extraction_backend == "sarvam" else a_settings.extraction_model_name),
                processing_time_seconds=elapsed,
                page_outputs=page_outputs,
            )
        except Exception as exc:
            db_error = str(exc)
            logger.warning("pipeline.extract.db_save_failed", error=str(exc))

        return {
            "success": True,
            "doc_id": filename,
            "schema": schema_name,
            "strategy": selected_strategy,
            "data": result_dict,
            "graph_stats": graph_meta.get("stats") if graph_meta else None,
            "graph_memory": graph_meta if graph_meta else None,
            "meta": {
                "pages_processed": len(page_outputs),
                "processing_time_seconds": elapsed,
                "strategy": selected_strategy,
                "output_md": f"{stem}.md",
                "output_json": f"{stem}.extracted.json",
                "llm_provider": a_settings.extraction_backend,
                "saved_to_db": db_id is not None,
                "db_run_id": db_id,
                "db_error": db_error,
                "ir_persisted": ir_persisted,
            }
        }

    try:
        return await run_in_threadpool(_execute)
    except Exception as exc:
        logger.exception("pipeline.extract.failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


# ========================= Query Bot (JSON Q&A) =========================

# ========================= Query Bot (Graph Memory & JSON Fallback) =========================

from pydantic import BaseModel, Field
from app.core.auth import get_current_user
from app.storage.user_store import Role


def _lookup_doc_owner(doc_id: str, stem: str, username: Optional[str] = None, job_id: Optional[str] = None) -> Optional[str]:
    """
    Looks up the recorded owner of a document from PostgreSQL DB or disk metadata.
    If username is provided, checks that user's partition only (DB owner==username or OUTPUT_DIR/username).
    """
    from src.ai.layer3_extraction.storage import init_db, SessionLocal, DocumentGraphRecord
    try:
        init_db()
        sess = SessionLocal()
        try:
            candidates = [doc_id, f"{stem}.pdf", stem]
            q = sess.query(DocumentGraphRecord).filter(
                DocumentGraphRecord.doc_id.in_(candidates)
            )
            if job_id:
                q = q.filter_by(job_id=job_id)
            if username:
                user_rec = q.filter_by(owner=username).order_by(DocumentGraphRecord.created_at.desc()).first()
                if user_rec and user_rec.owner:
                    return user_rec.owner
            else:
                rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()
                if rec and rec.owner:
                    return rec.owner
        finally:
            sess.close()
    except Exception:
        pass

    if username:
        try:
            search_dirs = [resolve_owner_output_dir(username), OUTPUT_DIR]
        except HTTPException:
            search_dirs = [OUTPUT_DIR]
    else:
        search_dirs = [OUTPUT_DIR] + [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.resolve().is_relative_to(OUTPUT_DIR.resolve())]

    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for fname in (f"{stem}.extracted.json", f"{doc_id}.extracted.json", f"{stem}.schema_ref.json"):
            p = sdir / fname
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        owner = data.get("owner")
                        if owner:
                            if username:
                                if owner == username:
                                    return owner
                            else:
                                return owner
                except Exception:
                    pass
    return None


def _check_doc_access(owner: Optional[str], user: User, is_admin: bool) -> None:
    """
    Enforces that ownerless records are readable by admins only, and user records
    are readable only by the owning user or admins.
    """
    if is_admin:
        return
    if not owner:
        raise HTTPException(
            status_code=403,
            detail="Access denied: ownerless document readable by admin only.",
        )
    if owner != user.username:
        raise HTTPException(
            status_code=403,
            detail="Access denied: document belongs to another user.",
        )


class QueryBotRequest(BaseModel):
    extracted_data: Optional[Any] = Field(None, description="Full extracted JSON object or record dictionary")
    question: str = Field(..., description="User's natural language question about the extracted data")
    doc_id: Optional[str] = Field(None, description="Optional document name for reference")
    job_id: Optional[str] = Field(None, description="Optional job ID for reference")
    owner: Optional[str] = Field(None, description="Optional document owner for reference (required for admins when doc_id is ambiguous)")



@router.get("/api/query-bot/documents")
def get_query_bot_documents(
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all documents available for Query Bot inspection.
    Gathers documents from:
      1. PostgreSQL DocumentGraphRecord (with tenant isolation for non-admin users)
      2. Disk dataset_output/ (*.extracted.json, *.graph.json, *.schema_ref.json)
      3. Active in-memory pipeline jobs (_pipeline_jobs)
    """
    from src.ai.layer3_extraction.storage import (
        init_db,
        SessionLocal,
        DocumentGraphRecord,
    )

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)

    docs: Dict[str, Dict[str, Any]] = {}

    # Source 1: PostgreSQL Database
    try:
        init_db()
        session = SessionLocal()
        try:
            q = session.query(DocumentGraphRecord)
            if not is_admin:
                # Regular user: see only own documents (no ownerless docs)
                q = q.filter(DocumentGraphRecord.owner == user.username)
            for rec in q.order_by(DocumentGraphRecord.created_at.desc()).all():
                stem = Path(rec.doc_id).stem
                doc_key = f"{stem}.pdf" if not rec.doc_id.lower().endswith(".pdf") else rec.doc_id
                docs[doc_key] = {
                    "doc_id": doc_key,
                    "stem": stem,
                    "label": doc_key,
                    "has_graph": True,
                    "has_extracted": True,
                    "strategy": "graph_memory",
                    "source": "database",
                    "job_id": rec.job_id,
                    "owner": rec.owner,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                }
        finally:
            session.close()
    except Exception as db_err:
        logger.warning("query_bot.list_docs_db_error", error=str(db_err))

    # Source 2 (in-memory active pipeline jobs)
    # NOTE: The former disk source (dataset_output/*.graph.json / *.extracted.json)
    # has been removed. Documents/graphs/extractions are now DB-backed only.
    try:
        for job_id, job in _pipeline_jobs.items():
            if job.get("status") != "completed":
                continue
            job_owner = job.get("owner")
            if not is_admin:
                if not job_owner or job_owner != user.username:
                    continue
            for s in job.get("successes", []):
                doc_id = s.get("pdf") or s.get("doc_id") or (
                    s.get("extracted_json", "").replace(".extracted.json", ".pdf")
                    if s.get("extracted_json") else None
                )
                if doc_id:
                    stem = Path(doc_id).stem
                    strat = s.get("strategy") or job.get("strategy") or "graph_memory_concurrent"
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "doc_id": doc_id,
                            "stem": stem,
                            "label": doc_id,
                            "has_graph": (strat.startswith("graph_memory")),
                            "has_extracted": True,
                            "strategy": strat,
                            "source": "active_job",
                            "job_id": job_id,
                            "owner": job_owner,
                            "created_at": job.get("created_at"),
                        }
                    else:
                        docs[doc_id]["job_id"] = job_id
                        if strat:
                            docs[doc_id]["strategy"] = strat
    except Exception as job_err:
        logger.warning("query_bot.list_docs_job_error", error=str(job_err))

    sorted_docs = sorted(docs.values(), key=lambda d: d.get("created_at") or "", reverse=True)
    return {"documents": sorted_docs, "count": len(sorted_docs)}


@router.post("/api/query-bot/ask")
async def ask_query_bot(
    req: QueryBotRequest,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Query Bot:
    1. If Document Graph Memory is available (from PostgreSQL DB, file checkpoint dataset_output/<stem>.graph.json,
       or request payload's graph_memory), queries the graph via GraphQueryService with bounded multi-hop traversal
       and verbatim source citations (mode: "graph").
    2. Strictly enforces user/document isolation (User A cannot query User B's document graph).
    3. If no graph is found, falls back cleanly to the legacy extracted JSON LLM assistant (mode: "json_fallback").
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from src.ai.layer3_extraction.storage import (
        init_db,
        SessionLocal,
        DocumentGraphRecord,
    )
    try:
        from src.ai.layer3_extraction.graph_agent.graph_memory import GraphMemory
        from src.ai.layer3_extraction.graph_agent.query_service import GraphQueryService
    except ImportError as _g_imp_err:
        logger.warning("query_bot.graph_import_unavailable", error=str(_g_imp_err))
        GraphMemory = None
        GraphQueryService = None

    role_val = getattr(user.role, "value", user.role)
    is_admin = (role_val == "admin" or user.role == Role.ADMIN)

    if not is_admin and req.owner and req.owner != user.username:
        raise HTTPException(
            status_code=403,
            detail="Access denied: non-admin users cannot specify a different owner.",
        )

    # For admin: require explicit owner or job_id if doc_id is ambiguous across multiple owners
    if is_admin and req.doc_id and not req.owner and not req.job_id:
        stem = Path(req.doc_id).stem
        candidates = [req.doc_id, f"{stem}.pdf", stem]
        ambiguous_owners = set()
        try:
            init_db()
            session = SessionLocal()
            try:
                db_owners = session.query(DocumentGraphRecord.owner).filter(
                    DocumentGraphRecord.doc_id.in_(candidates),
                    DocumentGraphRecord.owner.isnot(None),
                ).distinct().all()
                for (o,) in db_owners:
                    if o:
                        ambiguous_owners.add(o)
            finally:
                session.close()
        except Exception:
            pass

        try:
            if OUTPUT_DIR.is_dir():
                for cdir in OUTPUT_DIR.iterdir():
                    if cdir.is_dir():
                        for fname in (f"{stem}.extracted.json", f"{req.doc_id}.extracted.json", f"{stem}.graph.json", f"{stem}.schema_ref.json"):
                            if (cdir / fname).is_file():
                                ambiguous_owners.add(cdir.name)
                                break
        except Exception:
            pass

        if len(ambiguous_owners) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Ambiguous doc_id '{req.doc_id}' exists for multiple owners ({', '.join(sorted(ambiguous_owners))}). Specify 'owner' or 'job_id' in the request.",
            )

    graph_dict = None

    # Validate doc_id up-front to prevent path traversal
    if req.doc_id:
        req.doc_id = validate_doc_id(req.doc_id)
        stem = Path(req.doc_id).stem
        if not is_admin:
            # Check if requesting user owns a copy in their partition/DB
            user_owner = _lookup_doc_owner(req.doc_id, stem, username=user.username)
            if not user_owner:
                # User does not own a copy; check if the document belongs to another user
                doc_owner = _lookup_doc_owner(req.doc_id, stem)
                if doc_owner and doc_owner != user.username:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied: document owned by '{doc_owner}'.",
                    )

    # Step 1: Check document graph in DB with strict user isolation
    if req.doc_id or req.job_id:
        try:
            init_db()
            session = SessionLocal()
            try:
                q = session.query(DocumentGraphRecord)
                if req.doc_id:
                    stem = Path(req.doc_id).stem
                    doc_candidates = [req.doc_id, f"{stem}.pdf", stem]
                    q = q.filter(DocumentGraphRecord.doc_id.in_(doc_candidates))
                    if req.job_id:
                        sub_q = q.filter_by(job_id=req.job_id)
                        if sub_q.first():
                            q = sub_q
                elif req.job_id:
                    q = q.filter_by(job_id=req.job_id)
                if not is_admin:
                    user_q = q.filter_by(owner=user.username)
                    db_rec = user_q.order_by(DocumentGraphRecord.created_at.desc()).first()
                    if not db_rec:
                        any_rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()
                        if any_rec:
                            _check_doc_access(any_rec.owner, user, is_admin)
                else:
                    if req.owner:
                        q = q.filter_by(owner=req.owner)
                    db_rec = q.order_by(DocumentGraphRecord.created_at.desc()).first()

                if db_rec:
                    # Enforce isolation: ownerless is admin-only, user docs are owner-only
                    _check_doc_access(db_rec.owner, user, is_admin)
                    graph_dict = db_rec.graph_json
            finally:
                session.close()
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("query_bot.db_lookup_error", error=str(exc))

    # Step 2 (removed): the former on-disk graph checkpoint fallback
    # (dataset_output/<stem>.graph.json) has been dropped. Graph memory is now
    # read exclusively from PostgreSQL (Step 1) or the request payload (Step 3).

    # Step 3: Check extracted_data payload for embedded graph
    if not graph_dict and req.extracted_data and isinstance(req.extracted_data, dict):
        if req.doc_id:
            stem = Path(req.doc_id).stem
            payload_owner = _lookup_doc_owner(req.doc_id, stem, username=(req.owner if is_admin else user.username), job_id=req.job_id)
            _check_doc_access(payload_owner, user, is_admin)
        if "graph_memory" in req.extracted_data and isinstance(req.extracted_data["graph_memory"], dict):
            graph_dict = req.extracted_data["graph_memory"].get("snapshot") or req.extracted_data["graph_memory"]
        elif "nodes" in req.extracted_data and "edges" in req.extracted_data:
            graph_dict = req.extracted_data

    # Path A: Graph Memory answering
    if GraphMemory and GraphQueryService and graph_dict and isinstance(graph_dict, dict) and "nodes" in graph_dict:
        try:
            graph_mem = GraphMemory.from_dict(graph_dict)
            service = GraphQueryService()
            result = service.query(graph_mem, req.question)
            return {
                "success": True,
                "question": req.question,
                "answer": result.answer,
                "sources": result.sources[:5] if result.sources else [],
                "doc_id": req.doc_id,
                "mode": "graph",
                "hops_traversed": result.hops_traversed,
                "retrieved_nodes_count": len(result.retrieved_nodes),
                "retrieved_edges_count": len(result.retrieved_edges),
            }
        except Exception as g_err:
            logger.error("query_bot.graph_query_failed", error=str(g_err))
            # Fall back to json_fallback if extracted_data is available

    # Path B: JSON Fallback (DB-backed) — read the latest extraction result for
    # this document from PostgreSQL (extraction_runs) instead of disk.
    if not req.extracted_data and req.doc_id:
        stem = Path(req.doc_id).stem
        eff_owner = req.owner if (is_admin and req.owner) else user.username
        # Enforce the same ownership checks as before using DB/sidecar ownership lookup.
        file_owner = _lookup_doc_owner(
            req.doc_id, stem,
            username=eff_owner,
            job_id=req.job_id,
        )
        _check_doc_access(file_owner, user, is_admin)
        try:
            from src.ai.layer3_extraction.storage import get_latest_extracted_data
            target_owner = req.owner if (is_admin and req.owner) else (file_owner if (is_admin and file_owner) else user.username)
            db_extracted = get_latest_extracted_data(req.doc_id, owner=target_owner)
            if db_extracted:
                req.extracted_data = db_extracted
            elif target_owner:
                owner_dir = resolve_owner_output_dir(target_owner)
                owner_ext_file = owner_dir / f"{stem}.extracted.json"
                if owner_ext_file.is_file():
                    ext_json = json.loads(owner_ext_file.read_text(encoding="utf-8"))
                    req.extracted_data = ext_json.get("extracted_data") or ext_json
        except Exception as ex_err:
            logger.warning("query_bot.db_extracted_lookup_error", error=str(ex_err))

    if not req.extracted_data:
        raise HTTPException(
            status_code=400,
            detail="No document graph found and no extracted JSON data provided to answer the question.",
        )

    import os
    import httpx
    from starlette.concurrency import run_in_threadpool
    from app.config import settings as app_settings

    def _query_llm() -> str:
        data_json_str = (
            req.extracted_data
            if isinstance(req.extracted_data, str)
            else json.dumps(req.extracted_data, indent=2)
        )

        prompt = (
            "You are a friendly document assistant. The user has uploaded a document and the following data was extracted from it:\n\n"
            f"{data_json_str}\n\n"
            "Answer the user's question naturally and conversationally, like a helpful human assistant. "
            "Use the data above to answer. Do not show JSON, field names, asterisks, or technical formatting. "
            "Just give a clear, plain English answer. If the information is not in the data, say so politely.\n\n"
            f"User: {req.question}\nAssistant:"
        )

        api_key = (
            getattr(app_settings, "sarvam_api_key", None)
            or getattr(a_settings, "sarvam_api_key", None)
            or os.getenv("IDP_SARVAM_API_KEY")
            or os.getenv("SARVAM_API_KEY")
            or ""
        )
        base_url = (
            getattr(app_settings, "sarvam_base_url", None)
            or getattr(a_settings, "sarvam_base_url", None)
            or os.getenv("IDP_SARVAM_BASE_URL")
            or os.getenv("SARVAM_BASE_URL")
            or "https://api.sarvam.ai/v1"
        ).rstrip("/")
        model_name = (
            getattr(app_settings, "sarvam_model", None)
            or getattr(a_settings, "sarvam_model_name", None)
            or os.getenv("IDP_SARVAM_MODEL_NAME")
            or os.getenv("SARVAM_MODEL")
            or "sarvam-105b"
        )
        backend = (
            getattr(app_settings, "llm_provider", None)
            or getattr(a_settings, "extraction_backend", None)
            or os.getenv("IDP_EXTRACTION_BACKEND")
            or os.getenv("LLM_PROVIDER")
            or "bedrock"
        ).lower()

        llm_error = None
        try:
            if backend == "bedrock":
                bedrock_model_id = (
                    getattr(app_settings, "bedrock_model_id", None)
                    or getattr(a_settings, "bedrock_model_id", None)
                    or os.getenv("IDP_BEDROCK_MODEL_ID")
                    or os.getenv("BEDROCK_MODEL_ID")
                    or "arn:aws:bedrock:ap-south-1:106611079163:application-inference-profile/qdtz23c8eis1"
                )
                bedrock_region = (
                    getattr(app_settings, "bedrock_region", None)
                    or getattr(a_settings, "bedrock_region", None)
                    or os.getenv("IDP_BEDROCK_REGION")
                    or os.getenv("BEDROCK_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "ap-south-1"
                )
                import boto3
                profile = os.getenv("AWS_PROFILE") or os.getenv("IDP_AWS_PROFILE")
                if profile:
                    b_client = boto3.Session(profile_name=profile, region_name=bedrock_region).client("bedrock-runtime", region_name=bedrock_region)
                else:
                    b_client = boto3.client("bedrock-runtime", region_name=bedrock_region)

                kwargs = {
                    "modelId": bedrock_model_id,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"temperature": 0.2, "maxTokens": 600},
                }
                effort = getattr(app_settings, "bedrock_reasoning_effort", None) or getattr(a_settings, "bedrock_reasoning_effort", None) or "low"
                if effort and ("gpt-oss" in bedrock_model_id.lower() or "qdtz23c8eis1" in bedrock_model_id):
                    kwargs["additionalModelRequestFields"] = {"reasoning_effort": effort}

                try:
                    b_resp = b_client.converse(**kwargs)
                except Exception:
                    kwargs.pop("additionalModelRequestFields", None)
                    b_resp = b_client.converse(**kwargs)

                content = b_resp.get("output", {}).get("message", {}).get("content", [])
                text = next((block["text"] for block in content if isinstance(block, dict) and "text" in block), "")
                if text.strip():
                    return text.strip()

            elif (backend == "sarvam" or api_key) and api_key:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                    "reasoning_effort": None,
                }
                headers = {
                    "api-subscription-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                resp = httpx.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                if resp.is_success:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content") or msg.get("reasoning_content") or ""
                        # Strip internal thinking tags if present
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                        # Strip markdown bold/italic markers
                        content = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", content)
                        # Strip "Assistant:" prefix if model echoed it
                        content = re.sub(r"^Assistant:\s*", "", content.strip())
                        if content.strip():
                            return content.strip()
                logger.warning(
                    "query_bot.sarvam_error",
                    status_code=resp.status_code,
                    body=resp.text[:200],
                )
                if resp.status_code in (401, 402, 403):
                    return f"Sarvam API Auth Error ({resp.status_code}): {resp.text}"

            elif backend == "ollama":
                # Fallback to Ollama (only if explicitly requested)
                ollama_url = getattr(a_settings, "ollama_base_url", "http://localhost:11434/v1").rstrip("/v1")
                ollama_model = getattr(a_settings, "extraction_model_name", "llama3.1")
                resp = httpx.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
                return f"LLM error: HTTP {resp.status_code}"
            else:
                pass
        except Exception as e:
            llm_error = str(e)
            logger.warning("query_bot.llm_call_failed", error=str(e))

        # Deterministic extraction fallback from extracted_data if LLM was unavailable
        try:
            extracted_dict = req.extracted_data if isinstance(req.extracted_data, dict) else {}
            norm_q = req.question.lower()
            for k, v in extracted_dict.items():
                if k.replace("_", " ") in norm_q or norm_q in str(v).lower():
                    return f"{k.replace('_', ' ').title()}: {v}"
        except Exception:
            pass

        return f"Error communicating with LLM: {llm_error}" if llm_error else "Unable to answer: no response from primary LLM provider."

    try:
        answer = await run_in_threadpool(_query_llm)
        return {
            "success": True,
            "question": req.question,
            "answer": answer,
            "sources": [],
            "doc_id": req.doc_id,
            "mode": "json_fallback",
        }
    except Exception as exc:
        logger.exception("query_bot.error")
        raise HTTPException(status_code=500, detail=f"Query bot failed: {exc}")
```
