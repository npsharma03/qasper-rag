"""Build the artifacts the Streamlit app reads.

Examples
--------
Reuse the embeddings you already computed in the notebook (no re-encoding)::

    python scripts/build_artifacts.py --splits validation \\
        --vectors-from artifacts/preprocessing/<run_id>/validation_minilm_<hash>.npz

Encode from scratch (about 18 minutes for validation on a laptop CPU)::

    python scripts/build_artifacts.py --splits validation test

When ``--vectors-from`` is used, a random sample of chunks is re-encoded and
compared with the cached rows. That catches a row-order mismatch between this
script's chunking and the notebook's.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qasper_rag import config, models  # noqa: E402
from qasper_rag.artifacts import write_split  # noqa: E402
from qasper_rag.preprocess import load_papers, preprocess_split, verify_display_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["validation"], choices=config.SPLITS)
    p.add_argument("--data-dir", type=Path, default=None, help="Folder with *_qasper.parquet (default: ./data)")
    p.add_argument("--out", type=Path, default=None, help="Output root (default: ./artifacts/app)")
    p.add_argument("--vectors-from", type=Path, default=None,
                   help="Notebook cache (.npz with chunk_vectors). Single split only.")
    p.add_argument("--verify-sample", type=int, default=64,
                   help="Chunks to re-encode when checking cached vectors (0 disables).")
    p.add_argument("--max-papers", type=int, default=None, help="Build a small trial set.")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or config.data_dir()
    out_root = args.out or config.app_artifacts_dir()
    if args.vectors_from and len(args.splits) != 1:
        raise SystemExit("--vectors-from works with exactly one split.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    encoder = None  # loaded lazily

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            print(f"Loading encoder {config.EMBEDDING_MODEL} ...")
            encoder = models.load_encoder(device=args.device)
        return encoder

    for split in args.splits:
        print(f"\n== {split} ==")
        papers = load_papers(data_dir, split, args.max_papers)
        print(f"{len(papers):,} papers")
        processed = preprocess_split(papers, split, tokenizer)
        chunks, questions = processed["chunks"], processed["questions"]
        print(f"{len(processed['paragraphs']):,} paragraphs, {len(chunks):,} chunks, "
              f"{len(questions):,} questions ({int(questions['eligible_for_evaluation'].sum()):,} eligible)")

        expected = config.EXPECTED_CHUNKS.get(split)
        if args.max_papers is None and expected is not None and len(chunks) != expected:
            print(f"WARNING: expected {expected:,} chunks (notebook) but built {len(chunks):,}. "
                  "The preprocessing port differs from the notebook; do not reuse its cached vectors.")

        check = verify_display_text(chunks, tokenizer)
        print(f"display_text check: {check['bad']}/{check['checked']} multi-part chunks drift by more than "
              f"4 tokens; {check['fallback_rows']} chunks fell back to decoded text")

        if args.vectors_from:
            with np.load(args.vectors_from, allow_pickle=False) as cache:
                vectors = cache["chunk_vectors"].astype(np.float32)
            if vectors.shape[0] != len(chunks):
                raise SystemExit(f"Cached vectors have {vectors.shape[0]:,} rows but there are "
                                 f"{len(chunks):,} chunks. Wrong cache file or different chunking.")
            if args.verify_sample > 0:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(chunks), size=min(args.verify_sample, len(chunks)), replace=False)
                fresh = models.encode_fn(get_encoder())(chunks["text"].to_numpy()[idx].tolist())
                cosine = np.sum(fresh * vectors[idx], axis=1)
                print(f"vector alignment: min cosine {cosine.min():.5f} over {len(idx)} sampled chunks")
                if cosine.min() < 0.999:
                    raise SystemExit("Cached vectors do not line up with these chunks (row order or text differs).")
            source = str(args.vectors_from)
        else:
            print("Encoding chunks (this is the slow step) ...")
            vectors = models.encode_fn(get_encoder(), show_progress_bar=True)(chunks["text"].tolist())
            source = "encoded by build_artifacts.py"

        parquet = data_dir / f"{split}_qasper.parquet"
        manifest = {
            "split": split,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": config.EMBEDDING_MODEL,
            "tokenizer": config.TOKENIZER_NAME,
            "max_tokens": config.MAX_TOKENS, "overlap": config.OVERLAP,
            "n_papers": int(len(papers)), "n_questions": int(len(questions)),
            "n_eligible_questions": int(questions["eligible_for_evaluation"].sum()),
            "max_papers": args.max_papers,
            "vectors_source": source,
            "source_parquet_bytes": parquet.stat().st_size,
        }
        write_split(out_root / split, chunks, questions, vectors, manifest)
        print(f"Wrote {out_root / split}")


if __name__ == "__main__":
    main()
