# QASPER RAG

A local research-paper question-answering project using QASPER.
Currently implements document preprocessing, embedding generation,
retrieval, and retrieval evaluation. LLM answer generation is planned.

## Pipeline

1. Extract paragraphs while preserving paper and section metadata.
2. Split long paragraphs into chunks of up to 250 tokens with 40-token overlap.
3. Generate normalized embeddings using Sentence Transformers.
4. Retrieve passages using FAISS, BM25, or reciprocal rank fusion.
5. Evaluate retrieval against QASPER evidence annotations.

## Initial validation results

MiniLM retrieval evaluated within each question's paper:

| Method | Hit@1 | Hit@5 | Hit@10 |
|--------|-------|-------|--------|
| BM25 | 0.213 | 0.581 | 0.753 |
| Embeddings | 0.225 | 0.611 | 0.784 |

Evaluated 888 questions; skipped 117 without mapped evidence.
A hit means a retrieved chunk belongs to an annotated evidence
paragraph. This metric does not measure answer accuracy.
Evidence from multiple annotators is pooled.

Hybrid retrieval and BGE embeddings are being explored.

## Files

- rag_work.ipynb: implementation and experiments
- data/: downloaded QASPER splits, excluded from Git
- artifacts/: saved chunks, embeddings, indexes, and results, excluded from Git

## Setup

Use Python 3.11 and install:

    python -m pip install pandas pyarrow datasets sentence-transformers faiss-cpu rank-bm25 tqdm

Place the downloaded Parquet splits in data/, update the notebook
paths to match their filenames, and run the notebook in order.
Save generated embeddings before closing the kernel.

## Dataset

https://huggingface.co/datasets/allenai/qasper
