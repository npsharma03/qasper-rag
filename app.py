"""Streamlit app: ask a question about one QASPER paper.

Run:  streamlit run app.py
Needs artifacts from:  python scripts/build_artifacts.py

Retrieval stays inside the selected paper, the same setup as the validation
results in the README. Answer generation is optional and needs OPENAI_API_KEY.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from qasper_rag import config, models
from qasper_rag.artifacts import ArtifactStore, available_splits
from qasper_rag.evaluate import first_gold_rank
from qasper_rag.generate import generate_answer, openai_llm
from qasper_rag.retrieval import PaperRetriever, RANK_COLUMNS

st.set_page_config(page_title="QASPER paper QA", layout="wide")

STYLE = """
<style>
.qr-card { border: 1px solid rgba(128,128,128,.35); border-left-width: 4px; border-radius: 6px;
           padding: .7rem .9rem; margin: 0 0 .75rem 0; }
.qr-gold { border-left-color: #c9971c; }
.qr-cited { border-left-color: #2b5f9e; }
.qr-head { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline; margin-bottom: .35rem; }
.qr-rank { font-weight: 700; font-variant-numeric: tabular-nums; min-width: 1.6rem; }
.qr-sec { font-weight: 600; }
.qr-loc { opacity: .65; font-size: .85rem; }
.qr-flag { font-size: .8rem; font-weight: 600; color: #8a6500; }
.qr-trace { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .45rem; }
.qr-chip { border: 1px solid rgba(128,128,128,.4); border-radius: 4px; padding: 0 .4rem;
           font-size: .8rem; font-variant-numeric: tabular-nums; opacity: .8; }
.qr-chip b { font-weight: 700; }
.qr-chip-final { opacity: 1; border-color: #2b5f9e; }
.qr-text { max-width: 72ch; line-height: 1.55; font-size: .95rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the sentence encoder")
def _encode_fn():
    return models.encode_fn(models.load_encoder())


@st.cache_resource(show_spinner="Loading the cross-encoder")
def _rerank_fn():
    return models.rerank_fn(models.load_reranker())


@st.cache_resource(show_spinner="Loading chunks and vectors")
def _store(root: str, split: str) -> ArtifactStore:
    return ArtifactStore.load(Path(root), split)


@st.cache_resource
def _retriever(root: str, split: str) -> PaperRetriever:
    # Models load on first use, so browsing the reference questions stays instant.
    return PaperRetriever(
        _store(root, split),
        encode_fn=lambda texts: _encode_fn()(texts),
        rerank_fn=lambda pairs: _rerank_fn()(pairs),
    )


def _api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets file
        return None


# ---------------------------------------------------------------- rendering helpers
def _plain(text: str) -> str:
    return " ".join(str(text).split())


def _escape_dollars(text: str) -> str:
    return text.replace("$", r"\$")


def _card(head: str, body: str, *, trace: str = "", css: str = "") -> str:
    # Single line, no blank lines or indentation: markdown would end the HTML block otherwise.
    return (f'<div class="qr-card {css}"><div class="qr-head">{head}</div>{trace}'
            f'<div class="qr-text">{html.escape(_plain(body))}</div></div>')


def _location(row) -> str:
    part = f", part {int(row.chunk_id) + 1}" if int(row.chunk_id) > 0 else ""
    return f'<span class="qr-loc">paragraph {int(row.paragraph_id) + 1}{part}</span>'


def _trace(row, reranked: bool) -> str:
    chips = [("BM25", row.bm25_rank), ("Dense", row.dense_rank), ("Hybrid", row.hybrid_rank)]
    if reranked:
        chips.append(("Reranked", row.final_rank))
    out = []
    for i, (label, rank) in enumerate(chips):
        final = i == len(chips) - 1
        out.append(f'<span class="qr-chip{" qr-chip-final" if final else ""}">{label} <b>{int(rank)}</b></span>')
    if reranked and pd.notna(row.ce_score):
        out.append(f'<span class="qr-chip">score <b>{row.ce_score:.2f}</b></span>')
    return f'<div class="qr-trace">{"".join(out)}</div>'


def chunk_card(row, rank: int, *, gold: bool, reranked: bool) -> str:
    head = (f'<span class="qr-rank">{rank}</span><span class="qr-sec">{html.escape(str(row.section))}</span>'
            f'{_location(row)}' + ('<span class="qr-flag">Gold evidence</span>' if gold else ""))
    return _card(head, row.display_text, trace=_trace(row, reranked), css="qr-gold" if gold else "")


def source_card(source: dict, cited: bool) -> str:
    head = (f'<span class="qr-rank">[{source["citation"]}]</span>'
            f'<span class="qr-sec">{html.escape(str(source["section"]))}</span>'
            + ('<span class="qr-flag" style="color:#2b5f9e">Cited in the answer</span>' if cited else ""))
    return _card(head, source["text"], css="qr-cited" if cited else "")


# ---------------------------------------------------------------- sidebar
root = str(config.app_artifacts_dir())
splits = available_splits(Path(root))

st.title("QASPER paper QA")
st.caption("Pick a paper and ask a question. Retrieval runs inside that paper only, "
           "the same setup as the validation results in the README.")

if not splits:
    st.error(f"No artifacts found in {root}.")
    st.code("python scripts/build_artifacts.py --splits validation", language="bash")
    st.stop()

with st.sidebar:
    st.header("Settings")
    split = st.selectbox("Split", splits, key="split",
                         help="Tune on validation. Keep test for the final run.")
    rerank = st.toggle("Rerank with the cross-encoder", value=True, key="rerank")
    candidate_k = st.slider("Candidates to rerank", 5, 50, config.CANDIDATE_K, key="candidate_k",
                            disabled=not rerank)
    show_k = st.slider("Chunks to show", 1, 20, 5, key="show_k")
    context_k = st.slider("Chunks sent to the generator", 1, 20, config.DEFAULT_CONTEXT_K, key="context_k",
                          help="Top 5 versus top 10 has not been tested yet, so 5 is a placeholder.")

store = _store(root, split)
retriever = _retriever(root, split)

# ---------------------------------------------------------------- question input
papers = store.papers.sort_values("title").reset_index(drop=True)
titles = dict(zip(papers["paper_id"], papers["title"]))
paper_id = st.selectbox("Paper", papers["paper_id"].tolist(), key="paper_id",
                        format_func=lambda pid: f"{titles[pid]} ({pid})")

paper_questions = store.questions[store.questions["paper_id"] == paper_id].reset_index(drop=True)
modes = ["Pick a QASPER question", "Write your own"] if len(paper_questions) else ["Write your own"]
mode = st.radio("Question", modes, horizontal=True, key="mode")

question_row = None
if mode == modes[0] and len(paper_questions):
    labels = {
        r.question_id: r.question + ("" if r.eligible_for_evaluation else "  (no mapped evidence)")
        for r in paper_questions.itertuples()
    }
    chosen = st.selectbox("QASPER question", list(labels), format_func=labels.get, key=f"pick_{paper_id}")
    question_row = paper_questions[paper_questions["question_id"] == chosen].iloc[0]
    question = str(question_row["question"])
else:
    question = st.text_input("Your question", key="custom_question",
                             placeholder="What dataset do they evaluate on?")

if st.button("Search", type="primary", key="search"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving"):
            result = retriever.retrieve(paper_id, question.strip(), candidate_k=candidate_k, rerank=rerank)
        st.session_state["last"] = {
            "split": split, "paper_id": paper_id, "question": question.strip(), "result": result,
            "gold": set(question_row["gold_chunk_uids"]) if question_row is not None else None,
            "references": list(question_row["reference_answers"]) if question_row is not None else None,
            "gold_evidence": list(question_row["gold_evidence"]) if question_row is not None else None,
            "answer": None,
        }

last = st.session_state.get("last")
if not last or last["split"] != split:
    st.info("Choose a paper and a question, then search.")
    st.stop()

result, gold = last["result"], last["gold"]
st.subheader(last["question"])
st.caption(f"Searched {len(result.table)} chunks in: {titles.get(last['paper_id'], last['paper_id'])}")

tab_chunks, tab_answer, tab_compare = st.tabs(["Retrieved chunks", "Answer", "Compare methods"])

# ---------------------------------------------------------------- retrieved chunks
with tab_chunks:
    timing = ", ".join(f"{name} {secs * 1000:.0f} ms" for name, secs in result.timings.items())
    st.caption(f"Ranking: {result.final_label}. Timings: {timing}.")
    for rank, row in enumerate(result.top("final", show_k).itertuples(), start=1):
        st.markdown(chunk_card(row, rank, gold=bool(gold) and row.chunk_uid in gold, reranked=result.reranked),
                    unsafe_allow_html=True)
    if gold is not None:
        with st.expander("Reference answer and gold evidence"):
            if not gold:
                st.write("No evidence paragraph in this paper matched, so this question is excluded from "
                         "the validation metrics.")
            st.markdown("**Reference answers**")
            for text in last["references"] or ["(none: annotators marked it unanswerable)"]:
                st.markdown("- " + _escape_dollars(text))
            st.markdown("**Gold evidence**")
            for text in last["gold_evidence"] or ["(none)"]:
                st.markdown("- " + _escape_dollars(_plain(text)))

# ---------------------------------------------------------------- answer
with tab_answer:
    key = _api_key()
    if not key:
        st.info("Answers need an OpenAI key. Set OPENAI_API_KEY in your environment or in "
                ".streamlit/secrets.toml. Retrieval works without it.")
    else:
        st.caption(f"The generator sees the top {context_k} chunks ({config.GENERATOR_MODEL}). "
                   "It must cite chunks as [n] and abstain when the context is not enough.")
        if st.button("Generate answer", key="generate"):
            with st.spinner("Generating"):
                context = result.top("final", context_k)
                last["answer"] = generate_answer(
                    last["question"], context, openai_llm(api_key=key),
                    model_id=config.GENERATOR_MODEL,
                    cache_dir=config.app_artifacts_dir() / "answer_cache",
                )
                last["answer_context_k"] = context_k
    answer = last.get("answer")
    if answer is not None:
        if answer.abstained:
            st.warning("The model judged the retrieved chunks insufficient to answer.")
        else:
            st.markdown(_escape_dollars(answer.text))
        if answer.invalid_citations:
            st.error(f"The answer cites sources that were not provided: {answer.invalid_citations}.")
        if gold is not None:
            context_uids = {s["chunk_uid"] for s in answer.sources}
            st.caption("Gold evidence in the context: " + ("yes" if context_uids & gold else "no"))
        usage = f"{answer.input_tokens} input tokens, {answer.output_tokens} output tokens, {answer.seconds:.1f} s"
        st.caption(usage + (" (cached)" if answer.cached else ""))
        st.caption("A valid [n] only means the number exists. Check that the chunk supports the claim.")
        for source in answer.sources:
            st.markdown(source_card(source, source["citation"] in answer.cited), unsafe_allow_html=True)

# ---------------------------------------------------------------- compare methods
with tab_compare:
    methods = [("bm25", "BM25"), ("dense", "MiniLM"), ("hybrid", "Hybrid RRF")]
    if result.reranked:
        methods.append(("final", "Hybrid + cross-encoder"))
    depth = min(5, len(result.table))
    tops = {label: result.top(m, depth) for m, label in methods}
    rows = []
    for i in range(depth):
        record = {"Rank": i + 1}
        for label, top in tops.items():
            row = top.iloc[i]
            record[label] = (f"{row['section']}, paragraph {int(row['paragraph_id']) + 1}"
                             + (" [gold]" if gold and row["chunk_uid"] in gold else ""))
        rows.append(record)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if gold:
        first = []
        for m, label in methods:
            rank = first_gold_rank(result.table, gold, RANK_COLUMNS[m])
            first.append({"Method": label, "Rank of first gold chunk": rank if rank is not None else "none"})
        st.markdown("**Where the first gold chunk lands**")
        st.dataframe(pd.DataFrame(first), hide_index=True)
    elif gold is not None:
        st.caption("No gold chunk is available for this question.")
    else:
        st.caption("Gold evidence is only known for QASPER questions.")
    st.caption(f"Ranks are positions among all {len(result.table)} chunks in this paper.")
