"""Preprocessing: QASPER Parquet -> paragraphs, chunks, questions.

A port of the preprocessing cells in ``rag_updated.ipynb``. Chunk windows,
``chunk_uid`` values, row order and evidence mapping are meant to be identical
to the notebook so cached embeddings and reranker scores stay valid. Two
additions:

* Chunks carry ``display_text``: the original, cased paragraph text for the
  chunk's token window. ``text`` is still the decoded token IDs (lowercased,
  wordpiece spacing) because BM25, the embeddings and the reranker scores were
  all computed from it. Use ``display_text`` for anything a person reads and
  for the generator context.
* Questions carry ``reference_answers`` (same extraction logic as the
  notebook's answer-generation cell).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PARAGRAPH_COLUMNS = [
    "split", "paper_id", "title", "section_index", "section",
    "paragraph_id", "paragraph_uid", "text",
]
CHUNK_COLUMNS = PARAGRAPH_COLUMNS + [
    "original_text", "token_ids", "token_count", "chunk_id", "chunk_uid",
    "token_start", "token_end", "char_start", "char_end", "display_text",
]
QUESTION_COLUMNS = [
    "split", "paper_id", "question_id", "question", "has_answerable_annotation",
    "gold_evidence", "gold_paragraph_uids", "gold_chunk_uids",
    "mapped_evidence_count", "unmapped_evidence_count",
    "eligible_for_evaluation", "reference_answers",
]


def normalize_text(text) -> str:
    return " ".join(str(text).split())


def load_papers(data_dir: Path, split: str, max_papers: int | None = None) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}_qasper.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Set QASPER_DATA_DIR or pass --data-dir.")
    df = pd.read_parquet(path)
    required = {"id", "title", "full_text", "qas"}
    if not required.issubset(df.columns):
        raise ValueError(f"{split}: missing columns {required - set(df.columns)}")
    if max_papers is not None:
        if max_papers < 1:
            raise ValueError("max_papers must be positive or None")
        df = df.head(max_papers).copy()
    if df["id"].duplicated().any():
        raise ValueError(f"{split}: duplicate paper IDs")
    return df.reset_index(drop=True)


def transform_papers(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per non-empty paragraph, with a stable ``paragraph_uid``."""
    records = []
    for paper in df.to_dict("records"):
        sections = paper["full_text"]["section_name"]
        paragraphs = paper["full_text"]["paragraphs"]
        if len(sections) != len(paragraphs):
            raise ValueError(f"Section/paragraph mismatch in paper {paper['id']}")
        for section_index, (section, section_paragraphs) in enumerate(zip(sections, paragraphs)):
            for paragraph_id, text in enumerate(section_paragraphs):
                text = str(text).strip()
                if text:
                    records.append({
                        "split": split, "paper_id": str(paper["id"]), "title": paper["title"],
                        "section_index": section_index, "section": section,
                        "paragraph_id": paragraph_id,
                        "paragraph_uid": json.dumps([split, str(paper["id"]), section_index, paragraph_id]),
                        "text": text,
                    })
    return pd.DataFrame(records, columns=PARAGRAPH_COLUMNS)


def check_chunk_fits(tokenizer, max_tokens: int, overlap: int,
                     max_sequence_length: int = config.MAX_SEQUENCE_LENGTH) -> tuple[int, int]:
    """Validate chunk settings against the encoder limit. Returns (sequence_limit, n_special)."""
    if not 0 <= overlap < max_tokens:
        raise ValueError("Require 0 <= overlap < max_tokens")
    limits = [max_sequence_length]
    limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(limit, int) and 0 < limit < 1_000_000:
        limits.append(limit)
    sequence_limit = min(limits)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if max_tokens + special_tokens > sequence_limit:
        raise ValueError(f"Chunk plus special tokens exceeds sequence limit {sequence_limit}")
    return sequence_limit, special_tokens


def tokenize_paragraphs(df: pd.DataFrame, tokenizer, batch_size: int = 512) -> pd.DataFrame:
    """Add ``token_ids``, ``offsets`` (character spans) and ``token_count``."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required (character offsets are used for display_text).")
    result = df.copy()
    texts = result["text"].tolist()
    token_ids, offsets = [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], add_special_tokens=False,
            truncation=False, padding=False,
            return_attention_mask=False, return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        token_ids.extend(encoded["input_ids"])
        offsets.extend(encoded["offset_mapping"])
    result["token_ids"] = token_ids
    result["offsets"] = offsets
    result["token_count"] = result["token_ids"].map(len)
    return result


def _char_span(text: str, offsets, start: int, end: int) -> tuple[int, int] | None:
    """Character span for tokens [start, end), widened to whitespace so display
    text never starts or ends mid-word. Returns None if offsets are unusable."""
    char_start, char_end = int(offsets[start][0]), int(offsets[end - 1][1])
    if not 0 <= char_start < char_end <= len(text):
        return None
    while char_start > 0 and not text[char_start - 1].isspace():
        char_start -= 1
    while char_end < len(text) and not text[char_end].isspace():
        char_end += 1
    return char_start, char_end


def chunk_paragraphs(df: pd.DataFrame, tokenizer, max_tokens: int = config.MAX_TOKENS,
                     overlap: int = config.OVERLAP) -> pd.DataFrame:
    """Sliding token windows inside each paragraph (never across paragraphs)."""
    check_chunk_fits(tokenizer, max_tokens, overlap)
    records = []
    for row in df.to_dict("records"):
        token_ids, offsets, paragraph = row["token_ids"], row["offsets"], row["text"]
        for chunk_id, start in enumerate(range(0, len(token_ids), max_tokens - overlap)):
            end = min(start + max_tokens, len(token_ids))
            chunk_tokens = token_ids[start:end]
            decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            if start == 0 and end == len(token_ids):
                char_start, char_end, display = 0, len(paragraph), paragraph
            else:
                span = _char_span(paragraph, offsets, start, end)
                if span is None:
                    char_start, char_end, display = -1, -1, decoded
                else:
                    char_start, char_end = span
                    display = paragraph[char_start:char_end].strip()
            record = dict(row)
            record.update({
                "original_text": paragraph,
                "text": decoded,
                "token_ids": chunk_tokens, "token_count": len(chunk_tokens),
                "chunk_id": chunk_id,
                "chunk_uid": json.dumps([row["split"], row["paper_id"], row["section_index"],
                                         row["paragraph_id"], chunk_id]),
                "token_start": start, "token_end": end,
                "char_start": char_start, "char_end": char_end,
                "display_text": display,
            })
            records.append(record)
            if end == len(token_ids):
                break
    return pd.DataFrame(records, columns=CHUNK_COLUMNS)


def reference_answers_for(annotations) -> list[str]:
    """Reference answers from one question's annotations (unanswerable ones skipped)."""
    answers = []
    for annotation in annotations["answer"]:
        if annotation.get("unanswerable", False):
            continue
        free_form = annotation.get("free_form_answer", "")
        spans = annotation.get("extractive_spans", [])
        yes_no = annotation.get("yes_no")
        if free_form and str(free_form).strip():
            text = str(free_form).strip()
        elif spans is not None and len(spans):
            text = "; ".join(str(span) for span in spans)
        elif yes_no is not None:
            text = "Yes" if bool(yes_no) else "No"
        else:
            text = ""
        if text:
            answers.append(text)
    return sorted(set(answers))


def prepare_questions(papers: pd.DataFrame, paragraphs: pd.DataFrame,
                      chunks: pd.DataFrame, split: str) -> pd.DataFrame:
    """Map QASPER evidence text to gold paragraphs and gold chunks.

    Matching is exact on whitespace-normalized text within the paper. Evidence
    that matches no paragraph (figure/table captions, bare section headings,
    text mismatches) is counted in ``unmapped_evidence_count``. A question is
    ``eligible_for_evaluation`` when at least one evidence entry mapped.
    """
    lookups: dict[str, dict[str, set]] = {}
    for paper_id, text, uid in zip(paragraphs["paper_id"], paragraphs["text"], paragraphs["paragraph_uid"]):
        lookups.setdefault(paper_id, {}).setdefault(normalize_text(text), set()).add(uid)
    children: dict[str, list[str]] = {}
    for parent, chunk_uid in zip(chunks["paragraph_uid"], chunks["chunk_uid"]):
        children.setdefault(parent, []).append(chunk_uid)

    records = []
    for paper in papers.to_dict("records"):
        paper_id = str(paper["id"])
        qas = paper["qas"]
        if len(qas["question"]) != len(qas["answers"]):
            raise ValueError(f"Question/answer mismatch in {paper_id}")
        lookup = lookups.get(paper_id, {})
        for i, (question, annotations) in enumerate(zip(qas["question"], qas["answers"])):
            answerable = False
            evidence: set[str] = set()
            for answer in annotations["answer"]:
                if answer.get("unanswerable", False):
                    continue
                answerable = True
                evidence.update(str(e) for e in answer.get("evidence", []))
            gold_paragraphs: set[str] = set()
            mapped = 0
            for text in evidence:
                matches = lookup.get(normalize_text(text), set())
                if matches:
                    mapped += 1
                    gold_paragraphs.update(matches)
            gold_chunks = sorted({uid for parent in gold_paragraphs for uid in children.get(parent, [])})
            records.append({
                "split": split, "paper_id": paper_id,
                "question_id": json.dumps([split, paper_id, i]), "question": question,
                "has_answerable_annotation": answerable, "gold_evidence": sorted(evidence),
                "gold_paragraph_uids": sorted(gold_paragraphs), "gold_chunk_uids": gold_chunks,
                "mapped_evidence_count": mapped, "unmapped_evidence_count": len(evidence) - mapped,
                "eligible_for_evaluation": bool(gold_chunks),
                "reference_answers": reference_answers_for(annotations),
            })
    return pd.DataFrame(records, columns=QUESTION_COLUMNS)


def preprocess_split(papers: pd.DataFrame, split: str, tokenizer,
                     max_tokens: int = config.MAX_TOKENS,
                     overlap: int = config.OVERLAP) -> dict[str, pd.DataFrame]:
    """Run the full preprocessing for one split."""
    paragraphs = tokenize_paragraphs(transform_papers(papers, split), tokenizer)
    chunks = chunk_paragraphs(paragraphs, tokenizer, max_tokens, overlap)
    questions = prepare_questions(papers, paragraphs, chunks, split)
    assert paragraphs["paragraph_uid"].is_unique
    assert chunks["chunk_uid"].is_unique
    assert chunks["token_count"].between(1, max_tokens).all()
    assert chunks["token_count"].eq(chunks["token_end"] - chunks["token_start"]).all()
    return {"paragraphs": paragraphs, "chunks": chunks, "questions": questions}


def verify_display_text(chunks: pd.DataFrame, tokenizer, sample: int = 500,
                        tolerance: int = 4, seed: int = 0) -> dict[str, float]:
    """Spot-check display_text against the token window it came from.

    Only multi-part paragraphs need this (single-chunk paragraphs use the
    paragraph text verbatim). display_text is widened to whitespace, so its
    token count can exceed ``token_count`` by a few tokens; larger drift means
    the character offsets are wrong.
    """
    multi = chunks[chunks.groupby("paragraph_uid")["chunk_id"].transform("count") > 1]
    if multi.empty:
        return {"checked": 0, "bad": 0, "bad_fraction": 0.0, "fallback_rows": 0}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(multi), size=min(sample, len(multi)), replace=False)
    picked = multi.iloc[idx]
    encoded = tokenizer(picked["display_text"].tolist(), add_special_tokens=False, truncation=False)
    bad = sum(abs(len(ids) - count) > tolerance for ids, count in zip(encoded["input_ids"], picked["token_count"]))
    return {
        "checked": int(len(picked)), "bad": int(bad), "bad_fraction": bad / len(picked),
        "fallback_rows": int((chunks["char_start"] < 0).sum()),
    }
