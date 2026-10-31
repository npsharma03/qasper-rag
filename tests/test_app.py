"""Drive the Streamlit app headlessly with fake models and the synthetic artifacts."""
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qasper_rag import models
from qasper_rag.generate import LLMResult

from conftest import fake_embed, fake_rerank

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeEncoder:
    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False):
        return fake_embed(texts)


class FakeReranker:
    def predict(self, pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True):
        return fake_rerank(pairs)


@pytest.fixture()
def env(artifacts_root, monkeypatch):
    monkeypatch.setenv("QASPER_APP_ARTIFACTS", str(artifacts_root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(models, "load_encoder", lambda *a, **k: FakeEncoder())
    monkeypatch.setattr(models, "load_reranker", lambda *a, **k: FakeReranker())
    st.cache_resource.clear()
    yield artifacts_root
    st.cache_resource.clear()


def html_blocks(at):
    return [m.value for m in at.markdown if m.value.startswith("<div class=\"qr-card")]


def test_missing_artifacts_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("QASPER_APP_ARTIFACTS", str(tmp_path / "empty"))
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("No artifacts found" in e.value for e in at.error)


def test_pick_question_and_search(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.selectbox(key="split").value == "validation"
    assert any("Choose a paper" in i.value for i in at.info)

    pid = at.selectbox(key="paper_id").value
    at.button(key="search").click().run()
    assert not at.exception, at.exception
    cards = html_blocks(at)
    assert len(cards) == 5                       # default "chunks to show"
    assert all("Reranked" in c for c in cards)  # rerank on by default
    # original casing shown, not the lowercased decoded text
    assert any(any(ch.isupper() for ch in c) for c in cards)
    assert at.session_state["last"]["paper_id"] == pid


def test_gold_flag_and_reference_shown_for_dataset_question(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.slider(key="show_k").set_value(20)
    at.button(key="search").click().run()
    assert not at.exception, at.exception
    cards = html_blocks(at)
    assert any("Gold evidence" in c for c in cards)   # all chunks shown, so gold must appear
    assert len(at.expander) >= 1


def test_free_form_question_without_rerank(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Write your own").run()
    at.text_input(key="custom_question").set_value("which dense retrieval model").run()
    at.toggle(key="rerank").set_value(False).run()
    at.button(key="search").click().run()
    assert not at.exception, at.exception
    cards = html_blocks(at)
    assert cards and not any("Reranked" in c for c in cards)
    assert not any("Gold evidence" in c for c in cards)


def test_empty_free_form_question_warns(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Write your own").run()
    at.button(key="search").click().run()
    assert any("Enter a question" in w.value for w in at.warning)


def test_answer_tab_needs_key_then_generates(env, monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="search").click().run()
    assert any("OpenAI key" in i.value for i in at.info)
    assert not [b for b in at.button if b.key == "generate"]

    from qasper_rag import generate

    def fake_llm(instructions, prompt):
        return LLMResult("It reports [1] and [2], plus a bad cite [9].", 321, 17)

    fake_llm.model_id = "fake"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(generate, "openai_llm", lambda *a, **k: fake_llm)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="search").click().run()
    at.button(key="generate").click().run()
    assert not at.exception, at.exception
    assert any("not provided" in e.value for e in at.error)          # [9] flagged
    cards = html_blocks(at)
    assert sum("Cited in the answer" in c for c in cards) == 2       # sources 1 and 2
    assert any("321 input tokens" in c.value for c in at.caption)
