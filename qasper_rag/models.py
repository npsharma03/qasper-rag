"""Model loading. Heavy imports are deferred so the rest of the package (and
its tests) work without torch installed."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(name: str = config.EMBEDDING_MODEL, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device or pick_device())


def load_reranker(name: str = config.RERANKER_MODEL, device: str | None = None,
                  max_length: int = config.RERANK_MAX_LENGTH):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=device or pick_device(), max_length=max_length)


def encode_fn(model, batch_size: int = config.ENCODE_BATCH_SIZE, show_progress_bar: bool = False):
    """Callable: list[str] -> float32 (n, d) L2-normalized embeddings."""
    def encode(texts: Sequence[str]) -> np.ndarray:
        vectors = model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress_bar,
        )
        return np.asarray(vectors, dtype=np.float32)
    return encode


def rerank_fn(model, batch_size: int = config.RERANK_BATCH_SIZE):
    """Callable: list[(query, text)] -> 1-D array of relevance scores."""
    def rerank(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores = model.predict(
            list(pairs), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(scores).reshape(-1)
    return rerank
