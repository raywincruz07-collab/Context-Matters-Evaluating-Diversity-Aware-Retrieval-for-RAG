from copy import deepcopy
from dataclasses import replace
import os
import pickle
import sys

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrievers.bm25_retriever as bm25_module
from retrievers.bm25_config import BM25_CONFIG, BM25Config
from retrievers.bm25_retriever import (
    BM25_CACHE_SCHEMA_VERSION,
    BM25Retriever,
)


@pytest.fixture(autouse=True)
def isolated_index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25_module, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(bm25_module, "_rank_bm25_version", lambda: "0.2.2")
    return tmp_path


def corpus():
    return [
        {"doc_id": 1, "text": "alpha beta beta", "metadata": "unchanged"},
        {"doc_id": "two", "text": "alpha gamma"},
        {"doc_id": 3, "text": "delta gamma gamma"},
    ]


def test_same_corpus_reuses_validated_cache_with_identical_results(monkeypatch):
    first = BM25Retriever()
    first.index(corpus())
    expected = first.retrieve("alpha beta", top_k=3)

    second = BM25Retriever()
    monkeypatch.setattr(
        second,
        "_build_index",
        lambda _: pytest.fail("validated cache should have been reused"),
    )
    second.index(corpus())

    assert second.runtime_index_fingerprint == first.runtime_index_fingerprint
    assert second.cache_path == first.cache_path
    assert [document["doc_id"] for document, _ in second.retrieve("alpha beta", 3)] == [
        document["doc_id"] for document, _ in expected
    ]
    assert [score for _, score in second.retrieve("alpha beta", 3)] == pytest.approx(
        [score for _, score in expected]
    )


@pytest.mark.parametrize(
    "changed_corpus",
    [
        lambda value: [value[0] | {"text": "alpha beta changed"}, *value[1:]],
        lambda value: [value[1], value[0], value[2]],
        lambda value: [value[0] | {"doc_id": 99}, *value[1:]],
        lambda value: [*value, {"doc_id": 4, "text": "new document"}],
    ],
)
def test_corpus_changes_produce_different_runtime_fingerprint(changed_corpus):
    original = BM25Retriever()
    original.index(corpus())
    changed = BM25Retriever()
    changed.index(changed_corpus(corpus()))
    assert changed.runtime_index_fingerprint != original.runtime_index_fingerprint
    assert changed.cache_path != original.cache_path


@pytest.mark.parametrize(
    "changed_config",
    [
        replace(BM25_CONFIG, k1=1.6),
        replace(BM25_CONFIG, b=0.6),
        replace(BM25_CONFIG, epsilon=0.3),
        replace(BM25_CONFIG, query_preprocessing="different preprocessing"),
        replace(BM25_CONFIG, ranking_semantics="different ranking semantics"),
    ],
)
def test_bm25_configuration_change_produces_different_fingerprint(changed_config):
    original = BM25Retriever()
    changed = BM25Retriever(changed_config)
    original.index(corpus())
    changed.index(corpus())
    assert changed.runtime_index_fingerprint != original.runtime_index_fingerprint


def test_runtime_uses_authoritative_default_config_object():
    retriever = BM25Retriever()
    assert retriever.runtime_config is BM25_CONFIG
    assert isinstance(retriever.runtime_config, BM25Config)


def test_library_version_change_produces_different_fingerprint(monkeypatch):
    original = BM25Retriever()
    original.index(corpus())
    monkeypatch.setattr(bm25_module, "_rank_bm25_version", lambda: "future-version")
    changed = BM25Retriever()
    changed.index(corpus())
    assert changed.runtime_index_fingerprint != original.runtime_index_fingerprint


def test_legacy_cache_is_ignored_and_preserved(isolated_index_dir):
    legacy_path = isolated_index_dir / "bm25_index.pkl"
    legacy_bytes = pickle.dumps(("unverified-index", "unverified-tokens"))
    legacy_path.write_bytes(legacy_bytes)

    retriever = BM25Retriever()
    retriever.index(corpus())

    assert legacy_path.read_bytes() == legacy_bytes
    assert retriever.cache_path != str(legacy_path)
    assert os.path.exists(retriever.cache_path)


def _tamper_cache(retriever, change):
    with open(retriever.cache_path, "rb") as file:
        payload = pickle.load(file)
    change(payload)
    with open(retriever.cache_path, "wb") as file:
        pickle.dump(payload, file)


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update(runtime_fingerprint="0" * 64),
        lambda payload: payload.update(cache_schema_version="legacy-schema"),
        lambda payload: payload.update(corpus_document_count=999),
        lambda payload: payload.update(tokenized_corpus=payload["tokenized_corpus"][:-1]),
    ],
)
def test_invalid_cache_metadata_or_shape_is_rebuilt(monkeypatch, change):
    original = BM25Retriever()
    original.index(corpus())
    _tamper_cache(original, change)

    fresh = BM25Retriever()
    calls = 0
    real_build = fresh._build_index

    def tracked_build(value):
        nonlocal calls
        calls += 1
        real_build(value)

    monkeypatch.setattr(fresh, "_build_index", tracked_build)
    fresh.index(corpus())
    assert calls == 1
    with open(fresh.cache_path, "rb") as file:
        repaired = pickle.load(file)
    assert repaired["cache_schema_version"] == BM25_CACHE_SCHEMA_VERSION
    assert repaired["runtime_fingerprint"] == fresh.runtime_index_fingerprint
    assert repaired["corpus_document_count"] == len(corpus())
    assert len(repaired["tokenized_corpus"]) == len(corpus())


def test_same_length_wrong_tokenized_corpus_content_is_rebuilt(monkeypatch):
    original = BM25Retriever()
    original.index(corpus())

    def replace_tokens(payload):
        payload["tokenized_corpus"][0] = ["wrong", "same-length", "content"]

    _tamper_cache(original, replace_tokens)
    fresh = BM25Retriever()
    calls = 0
    real_build = fresh._build_index

    def tracked_build(value):
        nonlocal calls
        calls += 1
        real_build(value)

    monkeypatch.setattr(fresh, "_build_index", tracked_build)
    fresh.index(corpus())

    expected = [document["text"].lower().split() for document in corpus()]
    assert calls == 1
    with open(fresh.cache_path, "rb") as file:
        repaired = pickle.load(file)
    assert repaired["tokenized_corpus"] == expected


def test_malformed_pickle_is_rebuilt(monkeypatch):
    original = BM25Retriever()
    original.index(corpus())
    with open(original.cache_path, "wb") as file:
        file.write(b"not a pickle")

    fresh = BM25Retriever()
    calls = 0
    real_build = fresh._build_index

    def tracked_build(value):
        nonlocal calls
        calls += 1
        real_build(value)

    monkeypatch.setattr(fresh, "_build_index", tracked_build)
    fresh.index(corpus())
    assert calls == 1
    assert isinstance(fresh.bm25, BM25Okapi)
    with open(fresh.cache_path, "rb") as file:
        assert pickle.load(file)["cache_schema_version"] == BM25_CACHE_SCHEMA_VERSION


def test_cache_write_is_final_and_leaves_no_temporary_file(isolated_index_dir):
    retriever = BM25Retriever()
    retriever.index(corpus())
    assert os.path.exists(retriever.cache_path)
    assert not list(isolated_index_dir.glob(".bm25_*.tmp"))


def test_ranking_and_scores_match_explicit_rank_bm25_reference():
    documents = corpus()
    retriever = BM25Retriever()
    retriever.index(documents)

    tokenized_corpus = [document["text"].lower().split() for document in documents]
    reference = BM25Okapi(
        tokenized_corpus,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
    )
    tokenized_query = "alpha beta".lower().split()
    scores = reference.get_scores(tokenized_query)
    top_indices = scores.argsort()[-3:][::-1]
    expected = [
        (documents[index], float(scores[index]))
        for index in top_indices
        if scores[index] > 0
    ]
    actual = retriever.retrieve("alpha beta", top_k=3)

    assert [document["doc_id"] for document, _ in actual] == [
        document["doc_id"] for document, _ in expected
    ]
    assert [score for _, score in actual] == pytest.approx(
        [score for _, score in expected], rel=0.0, abs=0.0
    )


def test_nonpositive_scores_remain_filtered():
    retriever = BM25Retriever()
    retriever.index(corpus())
    assert retriever.retrieve("term-absent-from-every-document", top_k=3) == []


def test_index_does_not_mutate_input_corpus():
    documents = corpus()
    before = deepcopy(documents)
    BM25Retriever().index(documents)
    assert documents == before


@pytest.mark.parametrize(
    "documents, error",
    [
        ([], ValueError),
        ([{"text": "missing id"}], ValueError),
        ([{"doc_id": True, "text": "text"}], TypeError),
        ([{"doc_id": 1.5, "text": "text"}], TypeError),
        ([{"doc_id": " ", "text": "text"}], ValueError),
        ([{"doc_id": 1, "text": "a"}, {"doc_id": 1, "text": "b"}], ValueError),
    ],
)
def test_invalid_or_duplicate_document_ids_are_rejected(documents, error):
    with pytest.raises(error):
        BM25Retriever().index(documents)


@pytest.mark.parametrize(
    "documents, error",
    [
        ([{"doc_id": 1}], ValueError),
        ([{"doc_id": 1, "text": None}], TypeError),
        ([{"doc_id": 1, "text": 7}], TypeError),
    ],
)
def test_invalid_document_text_is_rejected(documents, error):
    with pytest.raises(error):
        BM25Retriever().index(documents)
