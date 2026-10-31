import json

import pandas as pd

from qasper_rag.generate import (ABSTAIN, LLMResult, build_prompt, build_sources, generate_answer,
                                 is_abstention, parse_citations)


def chunks_df():
    return pd.DataFrame({
        "chunk_uid": ["u1", "u2", "u3"], "section": ["Intro", "Method", "Results"],
        "text": ["lowercase one", "lowercase two", "lowercase three"],
        "display_text": ["Original One", "Original Two", "Original Three"],
    })


def test_sources_use_display_text_and_number_from_one():
    sources = build_sources(chunks_df())
    assert [s["citation"] for s in sources] == [1, 2, 3]
    assert sources[0]["text"] == "Original One"


def test_prompt_contains_only_question_and_context():
    prompt = json.loads(build_prompt("Q?", build_sources(chunks_df())))
    assert set(prompt) == {"question", "context"}
    assert set(prompt["context"][0]) == {"citation", "chunk_uid", "section", "text"}


def test_abstention_matching_is_tolerant():
    for text in ["INSUFFICIENT_EVIDENCE", "insufficient_evidence.", "`INSUFFICIENT_EVIDENCE`\n"]:
        assert is_abstention(text)
    assert not is_abstention("The answer is insufficient evidence for X [1].")
    assert ABSTAIN == "INSUFFICIENT_EVIDENCE"


def test_parse_citations():
    assert parse_citations("Yes [1][3], also [3] and [9].", 3) == ([1, 3], [9])
    assert parse_citations("no citations", 3) == ([], [])


def test_generate_answer_and_disk_cache(tmp_path):
    calls = []

    def llm(instructions, prompt):
        calls.append(prompt)
        return LLMResult("It uses Original Two [2], see also [7].", 120, 15)

    first = generate_answer("Q?", chunks_df(), llm, model_id="fake", cache_dir=tmp_path)
    assert not first.cached and first.cited == [2] and first.invalid_citations == [7]
    assert first.input_tokens == 120 and not first.abstained
    second = generate_answer("Q?", chunks_df(), llm, model_id="fake", cache_dir=tmp_path)
    assert second.cached and len(calls) == 1 and second.text == first.text
    # different model id or context -> different cache entry
    generate_answer("Q?", chunks_df().head(2), llm, model_id="fake", cache_dir=tmp_path)
    assert len(calls) == 2


def test_abstained_answer():
    answer = generate_answer("Q?", chunks_df(), lambda i, p: LLMResult("INSUFFICIENT_EVIDENCE."))
    assert answer.abstained and answer.cited == []
