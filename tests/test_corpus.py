"""语料加载与确定性检索测试。"""
from __future__ import annotations

import re

from src.corpus import bigrams, normalize


def test_docs_loaded_and_well_formed(corpus):
    assert len(corpus.docs) == 9
    ids = set(corpus.docs)
    assert ids == {f"D0{i}" for i in range(1, 10)}
    for doc in corpus.docs.values():
        assert doc.title and doc.version_date
        assert doc.sections, f"{doc.doc_id} 没有小节"
        for sec in doc.sections:
            assert sec.get("heading") and sec.get("text")


def test_conflict_query_returns_both_docs(corpus):
    res = corpus.search("报销期限是多久")
    ids = [r["doc_id"] for r in res["results"]]
    assert "D01" in ids and "D02" in ids, f"冲突场景需要 D01/D02 同时命中，实际 {ids}"


def test_single_source_query_returns_only_d05(corpus):
    res = corpus.search("年假有几天")
    ids = [r["doc_id"] for r in res["results"]]
    assert ids == ["D05"], f"年假问题应只命中 D05，实际 {ids}"


def test_paraphrase_query_misses_threshold(corpus):
    res = corpus.search("出差住酒店一晚最多能报多少钱")
    assert res["results"] == []
    assert res["best_score"] < corpus.config["threshold"]


def test_rewritten_query_hits_d06_first(corpus):
    res = corpus.search("差旅住宿标准 每晚限额 报销")
    assert res["results"], "改写后的查询应命中"
    assert res["results"][0]["doc_id"] == "D06"


def test_out_of_corpus_query_empty(corpus):
    res = corpus.search("公司有健身房吗")
    assert res["results"] == []


def test_snippet_is_verbatim_substring(corpus):
    res = corpus.search("报销期限是多久")
    for r in res["results"]:
        doc = corpus.get(r["doc_id"])
        haystack = doc.title + "\n" + "\n".join(
            f"{s['heading']}\n{s['text']}" for s in doc.sections
        )
        assert r["snippet"] in haystack, f"snippet 必须是原文逐字片段: {r['snippet']!r}"


def test_top_k_clamped_and_empty_query(corpus):
    res = corpus.search("年假", top_k=999)
    assert len(res["results"]) <= corpus.config["top_k_max"]
    res = corpus.search("？")
    assert res["results"] == [] and res.get("empty_reason")


def test_bigrams_ignore_punctuation():
    assert bigrams("报销期限是多久？") == bigrams("报销期限是多久")
    assert "报销" in bigrams("报销？期限")


def test_normalize_keeps_punctuation():
    # 引用校验依赖标点保留，防止跨句拼接
    assert "。" in normalize("满一年后。")


def test_corpus_alt_replaces_content(corpus_alt):
    assert "D02" not in corpus_alt.docs
    text = " ".join(s["text"] for s in corpus_alt.get("D01").sections)
    assert "45 天" in text


def test_corpus_hash_stable(corpus):
    assert re.fullmatch(r"[0-9a-f]{64}", corpus.corpus_hash())
    assert corpus.corpus_hash() == corpus.corpus_hash()
