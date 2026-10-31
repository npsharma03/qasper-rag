"""Shared settings for the QASPER RAG pipeline.

Values mirror ``rag_updated.ipynb``. If you change a retrieval setting here,
rebuild the artifacts (``scripts/build_artifacts.py``) and re-run
``scripts/regression_check.py``. Paths are read from the environment at call
time so tests and deployments can redirect them.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Folder holding {train,validation,test}_qasper.parquet."""
    return Path(os.environ.get("QASPER_DATA_DIR", REPO_ROOT / "data"))


def app_artifacts_dir() -> Path:
    """Folder the Streamlit app reads: <root>/<split>/{chunks,questions}.parquet, vectors.npy."""
    return Path(os.environ.get("QASPER_APP_ARTIFACTS", REPO_ROOT / "artifacts" / "app"))


SPLITS = ("train", "validation", "test")

# Chunking. The tokenizer matches the embedding model so chunk windows are
# measured in the same tokens the encoder sees.
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 256
MAX_TOKENS = 250
OVERLAP = 40

# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RRF_K = 60
CANDIDATE_K = 20
ENCODE_BATCH_SIZE = 32
RERANK_BATCH_SIZE = 16
RERANK_MAX_LENGTH = 512

# Generation (the notebook's generation cells have not been run yet)
GENERATOR_MODEL = "gpt-4.1-mini-2025-04-14"
GENERATION_MAX_OUTPUT_TOKENS = 500
DEFAULT_CONTEXT_K = 5

# Chunk counts recorded in the notebook; used to sanity-check the preprocessing port.
EXPECTED_CHUNKS = {"train": 49_569, "validation": 13_946, "test": 20_799}
