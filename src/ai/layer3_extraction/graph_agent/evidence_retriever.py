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
