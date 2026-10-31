import numpy as np
import pytest

from qasper_rag.artifacts import ArtifactStore, available_splits
from qasper_rag.evaluate import first_gold_rank, hit_flags
from qasper_rag.retrieval import PaperRetriever, reciprocal_rank_fusion, ranks_from_ordering

from conftest import fake_embed, fake_rerank


@pytest.fixture()
def retriever(artifacts_root):
    store = ArtifactStore.load(artifacts_root, "validation")
    return PaperRetriever(store, encode_fn=fake_embed, rerank_fn=fake_rerank)


def test_store_loads_and_validates(artifacts_root):
    assert available_splits(artifacts_root) == ["validation"]
    store = ArtifactStore.load(artifacts_root, "validation")
    assert store.vectors.shape == (len(store.chunks), 64)
    assert set(store.papers["paper_id"]) == {"2001.00000", "2001.00001", "2001.00002"}
    assert store.papers["n_questions"].tolist() == [4, 4, 4]


def test_store_rejects_wrong_embedding_model(artifacts_root):
    import json
    manifest = artifacts_root / "validation" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["embedding_model"] = "something/else"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="embedded with"):
        ArtifactStore.load(artifacts_root, "validation")


def test_rrf_matches_hand_computation():
    a = np.array([2, 0, 1, 3])   # item 2 best in ranking a
    b = np.array([0, 2, 3, 1])
    fused = reciprocal_rank_fusion(a, b, rrf_k=60)
    score = {i: 1 / (60 + list(a).index(i) + 1) + 1 / (60 + list(b).index(i) + 1) for i in range(4)}
    assert fused.tolist() == sorted(range(4), key=lambda i: -score[i])
    assert ranks_from_ordering(np.array([2, 0, 1])).tolist() == [2, 3, 1]


def test_retrieval_stays_inside_paper_and_ranks_are_permutations(retriever):
    res = retriever.retrieve("2001.00001", "which beta model is used", rerank=True)
    t = res.table
    assert (retriever.store.chunks.set_index("chunk_uid").loc[t["chunk_uid"], "paper_id"] == "2001.00001").all()
    n = len(t)
    for col in ["bm25_rank", "dense_rank", "hybrid_rank", "final_rank"]:
        assert sorted(t[col]) == list(range(1, n + 1))


def test_reranking_orders_candidates_by_ce_score(retriever):
    res = retriever.retrieve("2001.00001", "which beta model is used", candidate_k=5, rerank=True)
    t = res.table
    cands = t[t["ce_score"].notna()]
    assert len(cands) == 5
    # candidates are exactly the hybrid top 5 ...
    assert set(cands["chunk_uid"]) == set(res.ordered_uids("hybrid", 5))
    # ... and occupy final ranks 1-5 ordered by descending score
    top5 = res.top("final", 5)
    assert set(top5["chunk_uid"]) == set(cands["chunk_uid"])
    assert top5["ce_score"].is_monotonic_decreasing
    # everything after the candidates keeps hybrid order
    rest = res.top("final").iloc[5:]
    assert rest["hybrid_rank"].is_monotonic_increasing


def test_without_rerank_final_equals_hybrid(retriever):
    res = retriever.retrieve("2001.00000", "alpha retrieval", rerank=False)
    assert (res.table["final_rank"] == res.table["hybrid_rank"]).all()
    assert res.table["ce_score"].isna().all() and not res.reranked


def test_candidate_k_larger_than_paper(retriever):
    res = retriever.retrieve("2001.00000", "retrieval", candidate_k=500, rerank=True)
    assert res.candidate_k == len(res.table) and res.table["ce_score"].notna().all()


def test_precomputed_query_vector_is_used(artifacts_root):
    store = ArtifactStore.load(artifacts_root, "validation")
    r = PaperRetriever(store)  # no encoder at all
    vec = fake_embed(["alpha encoder"])[0]
    res = r.retrieve("2001.00000", "alpha encoder", query_vec=vec, rerank=False)
    assert "encode" not in res.timings
    with pytest.raises(RuntimeError, match="encoder"):
        r.retrieve("2001.00000", "alpha encoder", rerank=False)
    with pytest.raises(RuntimeError, match="reranker"):
        r.retrieve("2001.00000", "alpha encoder", query_vec=vec, rerank=True)


def test_unknown_paper(retriever):
    with pytest.raises(KeyError):
        retriever.retrieve("nope", "q")


def test_finds_gold_chunk_for_a_lexical_question(retriever):
    store = retriever.store
    q = store.questions[(store.questions["paper_id"] == "2001.00000")].iloc[0]
    # query built from the gold paragraph's own distinctive word
    res = retriever.retrieve("2001.00000", "alpha", rerank=True)
    gold = set(q["gold_chunk_uids"])
    assert hit_flags(res.ordered_uids("final"), gold, (1, 5))["hit@1"] == 1
    assert first_gold_rank(res.table, gold, "final_rank") == 1


def test_hit_flags_and_first_gold_rank():
    assert hit_flags(["a", "b", "c"], {"c"}, (1, 2, 3)) == {"hit@1": 0, "hit@2": 0, "hit@3": 1}
    assert hit_flags(["a"], set(), (1,)) == {"hit@1": 0}
