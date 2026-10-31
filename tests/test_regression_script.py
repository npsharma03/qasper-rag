"""Exercise scripts/regression_check.py on synthetic 'notebook' outputs.

The expected CSVs here are produced by the package itself, so this checks the
script's mechanics (label parity, cached query vectors, per-question hit
comparison, failure on a mismatch), not the notebook equivalence. The real
equivalence check is running the script against your notebook's artifacts.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qasper_rag.artifacts import ArtifactStore
from qasper_rag.evaluate import hit_flags
from qasper_rag.retrieval import PaperRetriever

from conftest import fake_embed

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "regression_check.py"
LABELS = {"bm25": "BM25", "dense": "MiniLM", "hybrid": "Hybrid RRF"}


@pytest.fixture()
def notebook_dir(tmp_path, artifacts_root):
    store = ArtifactStore.load(artifacts_root, "validation")
    nb = tmp_path / "nb"
    nb.mkdir()
    store.questions.to_parquet(nb / "validation_questions.parquet", index=False)
    eligible = store.questions[store.questions["eligible_for_evaluation"]].reset_index(drop=True)
    qv = fake_embed(eligible["question"].tolist())
    np.savez_compressed(nb / "validation_minilm_abc.npz", chunk_vectors=store.vectors, query_vectors=qv)
    retriever = PaperRetriever(store)
    rows = []
    for i, q in enumerate(eligible.itertuples()):
        res = retriever.retrieve(q.paper_id, q.question, query_vec=qv[i], rerank=False)
        for method, label in LABELS.items():
            flags = hit_flags(res.ordered_uids(method), list(q.gold_chunk_uids), (1, 5, 10))
            rows.append({"question_id": q.question_id, "paper_id": q.paper_id, "method": label, **flags})
    pd.DataFrame(rows).to_csv(nb / "validation_retrieval_per_question.csv", index=False)
    return nb


def run(nb, artifacts_root):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--notebook-dir", str(nb), "--artifacts", str(artifacts_root)],
        capture_output=True, text=True,
    )


def test_passes_when_outputs_match(notebook_dir, artifacts_root):
    out = run(notebook_dir, artifacts_root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout and "0 gold-chunk mismatches" in out.stdout


def test_fails_when_a_hit_flag_differs(notebook_dir, artifacts_root):
    csv = notebook_dir / "validation_retrieval_per_question.csv"
    df = pd.read_csv(csv)
    df.loc[0, "hit@1"] = 1 - df.loc[0, "hit@1"]
    df.to_csv(csv, index=False)
    out = run(notebook_dir, artifacts_root)
    assert out.returncode == 1 and "FAIL" in out.stdout


def test_fails_when_gold_labels_differ(notebook_dir, artifacts_root):
    path = notebook_dir / "validation_questions.parquet"
    q = pd.read_parquet(path)
    q["gold_chunk_uids"] = [list(v) for v in q["gold_chunk_uids"]]
    q.at[0, "gold_chunk_uids"] = ["not-a-real-chunk"]
    q.to_parquet(path, index=False)
    out = run(notebook_dir, artifacts_root)
    assert out.returncode == 1 and "gold-chunk mismatches" in out.stdout
