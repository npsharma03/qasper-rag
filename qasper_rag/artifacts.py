"""On-disk artifacts the app reads, and the loader for them.

Layout, one folder per split::

    <root>/<split>/chunks.parquet      one row per chunk (no token_ids)
    <root>/<split>/questions.parquet   evidence labels + reference answers
    <root>/<split>/vectors.npy         float32, L2-normalized, row i = chunks row i
    <root>/<split>/manifest.json       settings the artifacts were built with

``vectors.npy`` is positional: row i belongs to chunk row i. The loader checks
the row count and the manifest's embedding model, and ``build_artifacts.py``
spot-checks alignment by re-encoding a sample of chunks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SCHEMA_VERSION = 1

APP_CHUNK_COLUMNS = [
    "chunk_uid", "split", "paper_id", "title", "section_index", "section",
    "paragraph_id", "paragraph_uid", "chunk_id", "token_count",
    "token_start", "token_end", "char_start", "char_end", "text", "display_text",
]


def write_split(out_dir: Path, chunks: pd.DataFrame, questions: pd.DataFrame,
                vectors: np.ndarray, manifest: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunks[APP_CHUNK_COLUMNS].reset_index(drop=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"vectors shape {vectors.shape} does not match {len(chunks)} chunks")
    norms = np.linalg.norm(vectors[: min(len(vectors), 1000)], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("vectors are not L2-normalized (dot product must equal cosine similarity)")
    chunks.to_parquet(out_dir / "chunks.parquet", index=False)
    questions.to_parquet(out_dir / "questions.parquet", index=False)
    np.save(out_dir / "vectors.npy", vectors)
    manifest = {**manifest, "schema_version": SCHEMA_VERSION,
                "n_chunks": int(len(chunks)), "vector_dim": int(vectors.shape[1])}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def available_splits(root: Path) -> list[str]:
    root = Path(root)
    return [s for s in config.SPLITS if (root / s / "manifest.json").exists()]


@dataclass
class ArtifactStore:
    split: str
    chunks: pd.DataFrame
    vectors: np.ndarray
    questions: pd.DataFrame
    manifest: dict
    paper_positions: dict[str, np.ndarray] = field(repr=False)
    papers: pd.DataFrame = field(repr=False)

    @classmethod
    def load(cls, root: Path, split: str) -> "ArtifactStore":
        folder = Path(root) / split
        if not (folder / "manifest.json").exists():
            raise FileNotFoundError(
                f"No artifacts for split '{split}' in {folder}. "
                f"Run: python scripts/build_artifacts.py --splits {split}"
            )
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Artifact schema {manifest.get('schema_version')} != {SCHEMA_VERSION}; rebuild.")
        if manifest.get("embedding_model") != config.EMBEDDING_MODEL:
            raise ValueError(
                f"Artifacts were embedded with {manifest.get('embedding_model')!r} but "
                f"config.EMBEDDING_MODEL is {config.EMBEDDING_MODEL!r}."
            )
        chunks = pd.read_parquet(folder / "chunks.parquet").reset_index(drop=True)
        questions = pd.read_parquet(folder / "questions.parquet").reset_index(drop=True)
        vectors = np.load(folder / "vectors.npy")
        if vectors.dtype != np.float32 or vectors.shape[0] != len(chunks):
            raise ValueError(f"vectors {vectors.shape} {vectors.dtype} do not match {len(chunks)} chunks")
        positions = {
            paper_id: np.asarray(idx, dtype=np.int64)
            for paper_id, idx in chunks.groupby("paper_id", sort=False).indices.items()
        }
        papers = (
            chunks.groupby("paper_id", sort=False)
            .agg(title=("title", "first"), n_chunks=("chunk_uid", "size"))
            .reset_index()
        )
        counts = questions.groupby("paper_id").size()
        papers["n_questions"] = papers["paper_id"].map(counts).fillna(0).astype(int)
        return cls(split=split, chunks=chunks, vectors=vectors, questions=questions,
                   manifest=manifest, paper_positions=positions, papers=papers)
