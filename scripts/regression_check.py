"""Check that the package reproduces the notebook's validation results.

Compares against files the notebook already saved in
``artifacts/preprocessing/<run_id>/``:

* ``validation_questions.parquet``                gold chunk labels
* ``validation_minilm_<hash>.npz``                cached question vectors
* ``validation_retrieval_per_question.csv``       BM25 / MiniLM / Hybrid RRF hits
* ``validation_reranker_per_question.csv``        Hybrid and Hybrid + cross-encoder hits

Checks
------
1. Same questions and the same gold chunk labels (tests the evidence-mapping port).
2. BM25, MiniLM and Hybrid RRF Hit@1/5/10 per question, for every eligible
   question. Uses the cached question vectors, so no model is loaded and the
   run takes seconds. Any mismatch is an error.
3. Optional (``--rerank-n N``): Hybrid + cross-encoder Hit@1/5/10/20 on N
   sampled questions with the real reranker. Different hardware can flip
   near-ties, so up to 2% of mismatching questions is reported as a warning.

    python scripts/regression_check.py --notebook-dir artifacts/preprocessing/<run_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qasper_rag import config, models  # noqa: E402
from qasper_rag.artifacts import ArtifactStore  # noqa: E402
from qasper_rag.evaluate import hit_flags  # noqa: E402
from qasper_rag.retrieval import PaperRetriever  # noqa: E402

KS = (1, 5, 10)
METHODS = {"bm25": "BM25", "dense": "MiniLM", "hybrid": "Hybrid RRF"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook-dir", type=Path, required=True,
                   help="The notebook's artifacts/preprocessing/<run_id> folder")
    p.add_argument("--artifacts", type=Path, default=None, help="App artifacts root (default: ./artifacts/app)")
    p.add_argument("--rerank-n", type=int, default=0, help="Questions to check with the real cross-encoder")
    p.add_argument("--device", default=None)
    return p.parse_args()


def check_labels(store: ArtifactStore, nb_questions: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    mine = store.questions.set_index("question_id")
    theirs = nb_questions.set_index("question_id")
    problems = 0
    if set(mine.index) != set(theirs.index):
        print(f"FAIL question sets differ: {len(set(mine.index) ^ set(theirs.index))} ids only on one side")
        problems += 1
    shared = mine.index.intersection(theirs.index)
    gold_diff = [q for q in shared
                 if sorted(mine.at[q, "gold_chunk_uids"]) != sorted(theirs.at[q, "gold_chunk_uids"])]
    elig_diff = [q for q in shared
                 if bool(mine.at[q, "eligible_for_evaluation"]) != bool(theirs.at[q, "eligible_for_evaluation"])]
    print(f"labels: {len(shared):,} shared questions, {len(gold_diff)} gold-chunk mismatches, "
          f"{len(elig_diff)} eligibility mismatches")
    problems += bool(gold_diff) + bool(elig_diff)
    return problems, theirs


def main() -> int:
    args = parse_args()
    root = args.artifacts or config.app_artifacts_dir()
    nb = args.notebook_dir
    store = ArtifactStore.load(root, "validation")

    nb_questions = pd.read_parquet(nb / "validation_questions.parquet")
    problems, _ = check_labels(store, nb_questions)

    cache = sorted(nb.glob("validation_minilm_*.npz"), key=lambda p: p.stat().st_mtime)
    if not cache:
        print("FAIL no validation_minilm_*.npz in the notebook folder")
        return 1
    with np.load(cache[-1], allow_pickle=False) as npz:
        query_vectors = npz["query_vectors"]
    eligible = nb_questions[nb_questions["eligible_for_evaluation"]].reset_index(drop=True)
    if len(eligible) != len(query_vectors):
        print(f"FAIL {len(eligible)} eligible questions but {len(query_vectors)} cached query vectors")
        return 1
    print(f"using {cache[-1].name}: {len(eligible):,} eligible questions")

    expected = pd.read_csv(nb / "validation_retrieval_per_question.csv").set_index(["question_id", "method"])
    retriever = PaperRetriever(store)
    gold_by_q = store.questions.set_index("question_id")["gold_chunk_uids"]

    mismatches = {m: 0 for m in METHODS}
    sums = {m: np.zeros(len(KS)) for m in METHODS}
    for i, row in enumerate(eligible.itertuples(index=False)):
        result = retriever.retrieve(row.paper_id, row.question, query_vec=query_vectors[i], rerank=False)
        gold = list(gold_by_q[row.question_id])
        for method, label in METHODS.items():
            got = hit_flags(result.ordered_uids(method), gold, KS)
            want = expected.loc[(row.question_id, label)]
            vals = [got[f"hit@{k}"] for k in KS]
            sums[method] += vals
            if any(got[f"hit@{k}"] != int(want[f"hit@{k}"]) for k in KS):
                mismatches[method] += 1
    print("\nretrieval parity (no reranker):")
    for method, label in METHODS.items():
        means = sums[method] / len(eligible)
        print(f"  {label:<11} " + "  ".join(f"hit@{k} {m:.3f}" for k, m in zip(KS, means))
              + f"   per-question mismatches: {mismatches[method]}")
        problems += mismatches[method] > 0

    if args.rerank_n > 0:
        print(f"\nreranker parity on {args.rerank_n} sampled questions (loads the cross-encoder) ...")
        expected_rr = pd.read_csv(nb / "validation_reranker_per_question.csv").set_index(["question_id", "method"])
        retriever = PaperRetriever(store, rerank_fn=models.rerank_fn(models.load_reranker(device=args.device)))
        rng = np.random.default_rng(0)
        picks = rng.choice(len(eligible), size=min(args.rerank_n, len(eligible)), replace=False)
        bad = 0
        for i in picks:
            row = eligible.iloc[int(i)]
            result = retriever.retrieve(row["paper_id"], row["question"], query_vec=query_vectors[int(i)], rerank=True)
            got = hit_flags(result.ordered_uids("final"), list(gold_by_q[row["question_id"]]), (1, 5, 10, 20))
            want = expected_rr.loc[(row["question_id"], "Hybrid + cross-encoder")]
            if any(got[f"hit@{k}"] != int(want[f"hit@{k}"]) for k in (1, 5, 10, 20)):
                bad += 1
        rate = bad / len(picks)
        print(f"  mismatching questions: {bad}/{len(picks)} ({rate:.1%})")
        if rate > 0.02:
            problems += 1

    print("\nPASS" if problems == 0 else f"\nFAIL ({problems} problem group(s))")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
