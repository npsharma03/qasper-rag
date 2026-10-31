"""Shared fixtures: a toy WordPiece tokenizer and a tiny synthetic QASPER split.

No network, torch or Hugging Face downloads are needed. The toy tokenizer is
lowercasing and wordpiece-based like MiniLM's, so decode() behaves the same way
(lowercase text, spaces around punctuation).
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from qasper_rag import config
from qasper_rag.artifacts import write_split
from qasper_rag.preprocess import preprocess_split

WORDS = ("retrieval dense sparse encoder transformer attention dataset benchmark accuracy recall precision "
         "training validation embedding ranking reranker corpus question answer evidence paragraph section "
         "model score baseline metric corpus language neural network").split()


LEADS = [
    ["alpha", "beta", "gamma", "delta", "omega"],
    ["zeta", "theta", "kappa", "sigma", "lambda"],
    ["iota", "upsilon", "epsilon", "eta", "rho"],
]


def build_toy_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials)
    lead_words = " ".join(w for group in LEADS for w in group)
    tok.train_from_iterator([" ".join(WORDS)] * 20 + [lead_words] * 20
                            + ["Don't stop, it's (really) a $5 test."] * 5, trainer)
    tok.decoder = decoders.WordPiece()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")


def make_paragraph(rng, n_words: int, lead: str) -> str:
    body = " ".join(rng.choice(WORDS, size=n_words))
    return f"{lead.capitalize()} {body}."


def make_papers() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    papers = []
    for p in range(3):
        sections = ["Introduction", "Method", "Results"]
        paragraphs = [
            [make_paragraph(rng, 18, LEADS[p][0]), make_paragraph(rng, 70, LEADS[p][1])],   # second one is long
            [make_paragraph(rng, 25, LEADS[p][2])],
            [make_paragraph(rng, 22, LEADS[p][3]), make_paragraph(rng, 15, LEADS[p][4])],
        ]
        short = paragraphs[0][0]
        long_ = paragraphs[0][1]
        answers = [
            # evidence is a short paragraph (text is stored with odd whitespace to test normalization)
            {"answer": [{"unanswerable": False, "evidence": [short.replace(" ", "  ")],
                         "free_form_answer": "Because of alpha.", "extractive_spans": [], "yes_no": None}]},
            # evidence is the long paragraph, which becomes several chunks
            {"answer": [{"unanswerable": False, "evidence": [long_],
                         "free_form_answer": "", "extractive_spans": ["beta span"], "yes_no": None}]},
            # table caption and a heading: neither maps to a paragraph
            {"answer": [{"unanswerable": False, "evidence": ["FLOAT SELECTED: Table 1: Results.", "Method"],
                         "free_form_answer": "", "extractive_spans": [], "yes_no": True}]},
            # unanswerable
            {"answer": [{"unanswerable": True, "evidence": [], "free_form_answer": "",
                         "extractive_spans": [], "yes_no": None}]},
        ]
        questions = ["What is alpha?", "Which beta is used?", "Is there a table?", "Who funded it?"]
        papers.append({
            "id": f"2001.0000{p}", "title": f"Toy paper {p}",
            "full_text": {"section_name": sections, "paragraphs": paragraphs},
            "qas": {"question": questions, "answers": answers},
        })
    return pd.DataFrame(papers)


def fake_embed(texts, dim: int = 64) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", str(text).lower()):
            out[i, zlib.crc32(word.encode()) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


def fake_rerank(pairs) -> np.ndarray:
    scores = []
    for q, t in pairs:
        qs, ts = set(re.findall(r"\w+", q.lower())), re.findall(r"\w+", t.lower())
        scores.append(sum(w in qs for w in ts) + 1e-6 * len(ts))
    return np.asarray(scores, dtype=np.float64)


@pytest.fixture(scope="session")
def toy_tokenizer():
    return build_toy_tokenizer()


@pytest.fixture(scope="session")
def toy_papers():
    return make_papers()


@pytest.fixture(scope="session")
def processed(toy_papers, toy_tokenizer):
    return preprocess_split(toy_papers, "validation", toy_tokenizer, max_tokens=24, overlap=6)


@pytest.fixture()
def artifacts_root(tmp_path, processed):
    chunks, questions = processed["chunks"], processed["questions"]
    vectors = fake_embed(chunks["text"].tolist())
    write_split(tmp_path / "validation", chunks, questions, vectors,
                {"split": "validation", "embedding_model": config.EMBEDDING_MODEL})
    return tmp_path
