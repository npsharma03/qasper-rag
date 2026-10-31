# QASPER RAG

A work-in-progress retrieval-augmented question-answering project using scientific papers from QASPER.

## Pipeline

1. Load the train, validation, and test splits.
2. Extract paragraphs and preserve their source metadata.
3. Create chunks of up to 250 tokens with 40-token overlap.
4. Prepare question and evidence labels.
5. Compare BM25, MiniLM embedding retrieval, and hybrid reciprocal rank fusion.
6. Rerank hybrid’s top 20 candidates with a pretrained cross-encoder.
7. Browse retrieval stage by stage, and optionally generate cited answers, in a Streamlit app.

## Validation results

Evaluated on 888 questions with mapped evidence. Retrieval is restricted to each question’s paper.

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|---:|
| BM25 | 21.3% | 58.1% | 75.3% | — |
| MiniLM | 22.5% | 61.1% | 78.4% | — |
| Hybrid RRF | 26.2% | 65.2% | 81.1% | 93.8% |
| Hybrid + cross-encoder | **35.5%** | **72.9%** | **86.4%** | **93.8%** |

Hit@k measures whether at least one of the top k retrieved chunks belongs to a mapped gold-evidence paragraph.

Cross-encoder reranking improves Hit@1 by 9.2 percentage points and Hit@10 by 5.3 points over hybrid retrieval. Hit@20 remains unchanged because reranking preserves the candidate set.

## Models

- **BM25:** keyword retrieval (`rank_bm25`), fitted separately on each paper's chunks (about 50 per paper), so term statistics are per-paper, not corpus-wide.
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Hybrid fusion:** reciprocal rank fusion with rank constant 60.

Both neural models use pretrained weights without fine-tuning.

## Files

- `rag_updated.ipynb`: current preprocessing and retrieval experiments (unchanged; it remains the experiment log).
- `rag_draft.ipynb`: earlier implementation.
- `qasper_rag/`: the notebook's logic as an importable package (`preprocess`, `retrieval`, `generate`, `artifacts`, `models`, `evaluate`, `config`).
- `scripts/build_artifacts.py`: builds the chunk, question and vector files the app reads.
- `scripts/regression_check.py`: checks the package against the notebook's saved validation outputs.
- `app.py`: Streamlit app.
- `tests/`: unit and app tests (synthetic data, no model downloads).
- `data/`: local QASPER Parquet files (not committed).
- `artifacts/`: generated outputs, caches and app files (not committed).

## Local setup

Install the dependencies (add `jupyter` to run the notebook):

```bash
pip install -r requirements.txt
```

Place the dataset files in `data/`:

```text
train_qasper.parquet
validation_qasper.parquet
test_qasper.parquet
```

Open `rag_updated.ipynb`, check the data-path configuration, and run the preprocessing and retrieval cells in order.

## Streamlit app

The app searches inside one selected paper, the same scope as the validation results above. It shows each retrieved chunk with its rank at every stage (BM25, dense, hybrid, reranked), highlights gold evidence for QASPER questions, and compares the methods side by side. Answer generation is optional.

```bash
pip install -r requirements.txt

# 1. Build app artifacts from the notebook's cached embeddings (no re-encoding).
python scripts/build_artifacts.py --splits validation \
    --vectors-from artifacts/preprocessing/<run_id>/validation_minilm_<hash>.npz

# 2. Check the package reproduces the notebook's validation numbers.
python scripts/regression_check.py --notebook-dir artifacts/preprocessing/<run_id>

# 3. Run the app.
streamlit run app.py
```

Without `--vectors-from`, `build_artifacts.py` encodes every chunk with MiniLM (about 18 minutes for validation on CPU). To enable answers, set `OPENAI_API_KEY` in your environment or in `.streamlit/secrets.toml` (git-ignored). The generation experiment (top 5 versus top 10 chunks) has not been run, so the app's default of 5 is a placeholder.

Chunks keep two text fields. `text` is the decoded token IDs (lowercase, wordpiece spacing) that BM25, the embeddings and the reranker used. `display_text` is the original paragraph text for the same window and is what the app and the generator show.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## Limitations and next steps

**What the scores measure.** Hit@k is paragraph-level evidence retrieval, not answer correctness or complete evidence coverage. Retrieval is paper-scoped (about 50 chunks per paper), which is much easier than searching the whole corpus, so a random top-10 already hits roughly one question in five. Full-corpus retrieval has not been evaluated, and no random or positional baseline is reported yet.

**Which questions are evaluated.** Only the 888 of 1,005 validation questions with at least one evidence entry matched to a paragraph are scored. Questions answered by unanswerable annotations, table or figure captions (`FLOAT SELECTED: Table …`), or bare section headings are excluded, so the hardest cases are likely under-represented. A question counts as a hit if any annotator's evidence is retrieved.

**Reproducibility.** `scripts/regression_check.py` reproduced the notebook's validation labels and the per-question BM25, MiniLM and Hybrid RRF Hit@1/5/10 from the package. Cross-encoder parity is checked separately with `--rerank-n`. All reported numbers come from the validation split; the test split has not been evaluated.

**Next steps.**
- Answer generation is implemented but untested. The planned experiment compares answers from the top 5 versus top 10 reranked chunks, with manual review of correctness and whether citations support the claims.
- Add confidence intervals (resampling by paper), random and positional baselines, and an automatic answer metric.
- Fit BM25 term statistics on the whole split instead of per paper, and compare.
- Run the test split once, after the configuration is frozen.
