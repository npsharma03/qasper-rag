import json

import numpy as np

from qasper_rag.preprocess import (check_chunk_fits, normalize_text, preprocess_split,
                                   reference_answers_for, verify_display_text)


def test_chunk_windows_are_consistent(processed):
    chunks = processed["chunks"]
    assert chunks["chunk_uid"].is_unique
    assert (chunks["token_count"] == chunks["token_end"] - chunks["token_start"]).all()
    assert chunks["token_count"].between(1, 24).all()
    # Overlap of 6 means windows advance by 18 tokens.
    multi = chunks[chunks.groupby("paragraph_uid")["chunk_id"].transform("count") > 1]
    assert len(multi) > 0
    for _, group in multi.groupby("paragraph_uid"):
        starts = group["token_start"].tolist()
        assert starts == [18 * i for i in range(len(starts))]
        assert group["token_end"].iloc[-1] == group["token_end"].max()


def test_chunks_never_cross_paragraphs(processed):
    chunks = processed["chunks"]
    uid = chunks["chunk_uid"].map(json.loads)
    assert (uid.map(lambda u: u[3]) == chunks["paragraph_id"]).all()
    assert (uid.map(lambda u: u[2]) == chunks["section_index"]).all()


def test_text_is_decoded_but_display_text_keeps_original(processed):
    chunks = processed["chunks"]
    single = chunks[chunks.groupby("paragraph_uid")["chunk_id"].transform("count") == 1]
    # single-chunk paragraphs: display text is the paragraph verbatim (cased, punctuation untouched)
    assert (single["display_text"] == single["original_text"]).all()
    assert single["display_text"].str.contains("[A-Z]").all()
    # decoded text is lowercase
    assert (chunks["text"] == chunks["text"].str.lower()).all()


def test_display_text_for_multi_part_chunks_is_an_original_substring(processed, toy_tokenizer):
    chunks = processed["chunks"]
    multi = chunks[chunks.groupby("paragraph_uid")["chunk_id"].transform("count") > 1]
    for row in multi.itertuples():
        assert row.char_start >= 0
        assert row.display_text == row.original_text[row.char_start:row.char_end].strip()
        # never starts or ends mid-word
        assert row.char_start == 0 or row.original_text[row.char_start - 1].isspace()
        assert row.char_end == len(row.original_text) or row.original_text[row.char_end].isspace()
    check = verify_display_text(chunks, toy_tokenizer)
    assert check["checked"] > 0 and check["bad"] == 0 and check["fallback_rows"] == 0


def test_evidence_mapping(processed):
    chunks, questions = processed["chunks"], processed["questions"]
    q = questions[questions["paper_id"] == "2001.00000"].reset_index(drop=True)
    assert len(q) == 4

    # Q0: whitespace-normalized exact match to a single-chunk paragraph.
    assert q.at[0, "mapped_evidence_count"] == 1 and q.at[0, "eligible_for_evaluation"]
    assert len(q.at[0, "gold_chunk_uids"]) == 1

    # Q1: evidence is the long paragraph -> every chunk of it is gold.
    parent = q.at[1, "gold_paragraph_uids"][0]
    expected = sorted(chunks.loc[chunks["paragraph_uid"] == parent, "chunk_uid"])
    assert list(q.at[1, "gold_chunk_uids"]) == expected and len(expected) > 1

    # Q2: table caption and bare heading do not map -> not eligible.
    assert q.at[2, "mapped_evidence_count"] == 0 and q.at[2, "unmapped_evidence_count"] == 2
    assert not q.at[2, "eligible_for_evaluation"]

    # Q3: unanswerable.
    assert not q.at[3, "has_answerable_annotation"] and not q.at[3, "eligible_for_evaluation"]


def test_gold_chunks_stay_inside_their_paper(processed):
    chunks, questions = processed["chunks"], processed["questions"]
    paper_of = dict(zip(chunks["chunk_uid"], chunks["paper_id"]))
    for row in questions.itertuples():
        assert all(paper_of[uid] == row.paper_id for uid in row.gold_chunk_uids)


def test_reference_answers():
    ann = {"answer": [
        {"unanswerable": False, "free_form_answer": " Free form ", "extractive_spans": ["a"], "yes_no": None},
        {"unanswerable": False, "free_form_answer": "", "extractive_spans": ["x", "y"], "yes_no": None},
        {"unanswerable": False, "free_form_answer": "", "extractive_spans": [], "yes_no": False},
        {"unanswerable": True, "free_form_answer": "ignored", "extractive_spans": [], "yes_no": None},
    ]}
    assert reference_answers_for(ann) == ["Free form", "No", "x; y"]


def test_settings_validation(toy_tokenizer):
    check_chunk_fits(toy_tokenizer, 250, 40)
    for bad in [(250, 250), (300, 40)]:
        try:
            check_chunk_fits(toy_tokenizer, *bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_normalize_text():
    assert normalize_text(" a \n b\t c ") == "a b c"


def test_preprocessing_is_deterministic(toy_papers, toy_tokenizer, processed):
    again = preprocess_split(toy_papers, "validation", toy_tokenizer, max_tokens=24, overlap=6)
    assert again["chunks"]["chunk_uid"].tolist() == processed["chunks"]["chunk_uid"].tolist()
    assert again["chunks"]["text"].tolist() == processed["chunks"]["text"].tolist()
