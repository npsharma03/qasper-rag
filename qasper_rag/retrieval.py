"""Paper-scoped retrieval: BM25, dense, reciprocal rank fusion, cross-encoder.

This reproduces the notebook's pipeline for one question at a time:

1. BM25 (``rank_bm25.BM25Okapi``) fitted on the selected paper's chunks only.
2. Dense: dot product of L2-normalized MiniLM vectors (= cosine similarity).
3. Hybrid: reciprocal rank fusion of the two full rankings (k = 60).
4. Rerank: cross-encoder over the hybrid top ``candidate_k`` (default 20).

All models read the decoded ``text`` column, as in the notebook, so results are
comparable with the saved validation numbers. Use ``display_text`` to show
text to people.

Retrieval never leaves the selected paper. That matches how the validation
numbers were measured and is much easier than searching the whole corpus.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import config
from .artifacts import ArtifactStore

EncodeFn = Callable[[Sequence[str]], np.ndarray]
RerankFn = Callable[[Sequence[tuple[str, str]]], np.ndarray]

RANK_COLUMNS = {
    "bm25": "bm25_rank",
    "dense": "dense_rank",
    "hybrid": "hybrid_rank",
    "final": "final_rank",
}


def bm25_tokens(text) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def reciprocal_rank_fusion(*rankings: np.ndarray, rrf_k: int = config.RRF_K) -> np.ndarray:
    """Fuse rankings (arrays of item indices, best first) into one ordering."""
    n = len(rankings[0])
    fused = np.zeros(n, dtype=np.float64)
    for ranking in rankings:
        fused[ranking] += 1.0 / (rrf_k + np.arange(1, n + 1))
    return np.argsort(-fused, kind="stable")


def ranks_from_ordering(ordering: np.ndarray) -> np.ndarray:
    """1-based rank of each item, given an ordering of item indices (best first)."""
    ranks = np.empty(len(ordering), dtype=np.int64)
    ranks[ordering] = np.arange(1, len(ordering) + 1)
    return ranks


@dataclass
class Retrieval:
    """Everything one query produced inside one paper.

    ``table`` has one row per chunk in the paper, in document order. The
    ``*_rank`` columns are 1-based positions over the whole paper. ``final_rank``
    is the cross-encoder order for the candidates (when reranked) followed by
    the remaining chunks in hybrid order; with ``reranked=False`` it equals
    ``hybrid_rank``. ``ce_score`` is NaN for chunks that were not reranked.
    """
    paper_id: str
    query: str
    table: pd.DataFrame
    reranked: bool
    candidate_k: int
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def final_label(self) -> str:
        return "Hybrid + cross-encoder" if self.reranked else "Hybrid RRF"

    def top(self, method: str = "final", k: int | None = None) -> pd.DataFrame:
        ordered = self.table.sort_values(RANK_COLUMNS[method], kind="stable")
        return ordered if k is None else ordered.head(k)

    def ordered_uids(self, method: str = "final", k: int | None = None) -> list[str]:
        return self.top(method, k)["chunk_uid"].tolist()


class PaperRetriever:
    """Runs the retrieval stack inside one paper at a time."""

    def __init__(self, store: ArtifactStore, encode_fn: EncodeFn | None = None,
                 rerank_fn: RerankFn | None = None, rrf_k: int = config.RRF_K):
        self.store = store
        self._encode = encode_fn
        self._rerank = rerank_fn
        self.rrf_k = rrf_k
        self._bm25_cache: dict[str, BM25Okapi] = {}

    def _bm25_for(self, paper_id: str) -> BM25Okapi:
        index = self._bm25_cache.get(paper_id)
        if index is None:
            positions = self.store.paper_positions[paper_id]
            texts = self.store.chunks["text"].to_numpy()[positions]
            index = BM25Okapi([bm25_tokens(t) for t in texts])
            self._bm25_cache[paper_id] = index
        return index

    def retrieve(self, paper_id: str, query: str, *, query_vec: np.ndarray | None = None,
                 candidate_k: int = config.CANDIDATE_K, rerank: bool = True) -> Retrieval:
        if paper_id not in self.store.paper_positions:
            raise KeyError(f"Unknown paper_id {paper_id!r}")
        positions = self.store.paper_positions[paper_id]
        chunks = self.store.chunks.iloc[positions]
        n = len(positions)
        timings: dict[str, float] = {}

        start = time.perf_counter()
        bm25_scores = np.asarray(self._bm25_for(paper_id).get_scores(bm25_tokens(query)), dtype=np.float64)
        bm25_ranking = np.argsort(-bm25_scores, kind="stable")
        timings["bm25"] = time.perf_counter() - start

        start = time.perf_counter()
        if query_vec is None:
            if self._encode is None:
                raise RuntimeError("No encoder configured; pass encode_fn or query_vec.")
            query_vec = np.asarray(self._encode([query]), dtype=np.float32)[0]
            timings["encode"] = time.perf_counter() - start
            start = time.perf_counter()
        dense_scores = self.store.vectors[positions] @ np.asarray(query_vec, dtype=np.float32)
        dense_ranking = np.argsort(-dense_scores, kind="stable")
        timings["dense"] = time.perf_counter() - start

        start = time.perf_counter()
        hybrid_ranking = reciprocal_rank_fusion(bm25_ranking, dense_ranking, rrf_k=self.rrf_k)
        timings["fusion"] = time.perf_counter() - start

        k = min(candidate_k, n)
        candidates = hybrid_ranking[:k]
        ce_scores = np.full(n, np.nan)
        if rerank:
            if self._rerank is None:
                raise RuntimeError("No reranker configured; pass rerank_fn or rerank=False.")
            start = time.perf_counter()
            texts = chunks["text"].to_numpy()
            pairs = [(query, texts[i]) for i in candidates]
            scores = np.asarray(self._rerank(pairs), dtype=np.float64).reshape(-1)
            if len(scores) != k:
                raise ValueError(f"Reranker returned {len(scores)} scores for {k} pairs")
            ce_scores[candidates] = scores
            order = np.argsort(-scores, kind="stable")
            final_ranking = np.concatenate([candidates[order], hybrid_ranking[k:]])
            timings["rerank"] = time.perf_counter() - start
        else:
            final_ranking = hybrid_ranking

        table = pd.DataFrame({
            "position": positions,
            "chunk_uid": chunks["chunk_uid"].to_numpy(),
            "section": chunks["section"].to_numpy(),
            "paragraph_id": chunks["paragraph_id"].to_numpy(),
            "chunk_id": chunks["chunk_id"].to_numpy(),
            "text": chunks["text"].to_numpy(),
            "display_text": chunks["display_text"].to_numpy(),
            "bm25_score": bm25_scores, "bm25_rank": ranks_from_ordering(bm25_ranking),
            "dense_score": dense_scores, "dense_rank": ranks_from_ordering(dense_ranking),
            "hybrid_rank": ranks_from_ordering(hybrid_ranking),
            "ce_score": ce_scores, "final_rank": ranks_from_ordering(final_ranking),
        })
        return Retrieval(paper_id=paper_id, query=query, table=table, reranked=bool(rerank),
                         candidate_k=k, timings=timings)
