# QASPER RAG

A work-in-progress retrieval-augmented question-answering project using scientific papers from QASPER.

## Pipeline

1. Load the train, validation, and test splits.
2. Extract paragraphs and preserve their source metadata.
3. Create chunks of up to 250 tokens with 40-token overlap.
4. Prepare question and evidence labels.
5. Compare BM25, MiniLM embedding retrieval, and hybrid reciprocal rank fusion.
6. Rerank hybrid’s top 20 candidates with a pretrained cross-encoder.

## Validation results

Evaluated on 888 questions with mapped evidence. Retrieval is restricted to each question’s paper.

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|---:|
| BM25 | 21.3% | 58.1% | 75.3% | — |
| MiniLM | 22.5% | 61.1% | 78.4% | — |
| Hybrid RRF | 26.2% | 65.2% | 81.1% | 93.8% |
| Hybrid + cross-encoder | **35.5%** | **72.9%** | **86.4%** | **93.8%** |

Hit@k measures whether at least one of the top k retrieved chunks belongs to a mapped gold-evidence paragraph.

Cross-encoder reranking improves Hit@1 by 9.3 percentage points and Hit@10 by 5.3 points over hybrid retrieval. Hit@20 remains unchanged because reranking preserves the candidate set.

## Models

- **BM25:** keyword retrieval using corpus term statistics.
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Hybrid fusion:** reciprocal rank fusion with rank constant 60.

Both neural models use pretrained weights without fine-tuning.

## Files

- `rag_updated.ipynb`: current preprocessing and retrieval experiments.
- `rag_draft.ipynb`: earlier implementation.
- `data/`: local QASPER Parquet files.
- `artifacts/`: generated preprocessing outputs, caches, and evaluation results.

## Local setup

Install the notebook dependencies:

```bash
pip install jupyter pandas numpy pyarrow transformers sentence-transformers rank_bm25 tqdm
```

Place the dataset files in `data/`:

```text
train_qasper.parquet
validation_qasper.parquet
test_qasper.parquet
```

Open `rag_updated.ipynb`, check the data-path configuration, and run the preprocessing and retrieval cells in order.

## Limitations and next steps

These scores measure paragraph-level evidence retrieval, not answer correctness or complete evidence coverage. Questions without mapped evidence are excluded, and full-corpus retrieval has not been evaluated.

Answer-generation testing is pending. The next experiment compares answers generated from the top 5 versus top 10 reranked chunks, followed by a minimal interface showing answers, citations, and retrieved context.
