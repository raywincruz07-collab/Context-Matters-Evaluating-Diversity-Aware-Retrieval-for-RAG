from dataclasses import replace
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CandidateArtifact,
    CorpusProvenance,
    CorpusRecord,
    DatasetProvenance,
    RawCandidateResult,
    build_bm25_retriever_provenance,
    compute_corpus_manifest_sha256,
    compute_document_id_map_sha256,
    compute_index_fingerprint_sha256,
    document_content_sha256,
    produce_bm25_candidate_artifact,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
GIT_A = "1" * 40


def records():
    return (
        CorpusRecord("doc-a", "source-a", "Alpha text", 0),
        CorpusRecord(2, 200, "Beta text ", 1),
        CorpusRecord("doc-c", None, " Gamma text", 2),
    )


def dataset():
    return DatasetProvenance(
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic fixture",
        config=None,
        revision="dataset-revision",
        split="test",
        sample_manifest_id="sample-manifest",
        sample_manifest_sha256=SHA_A,
        sampling_seed=None,
        sampling_algorithm=None,
    )


def corpus(corpus_records=None, **changes):
    corpus_records = records() if corpus_records is None else corpus_records
    values = dict(
        corpus_id="synthetic-corpus",
        source="synthetic fixture",
        revision="corpus-revision",
        document_count=len(corpus_records),
        manifest_sha256=compute_corpus_manifest_sha256(corpus_records),
        document_id_map_sha256=compute_document_id_map_sha256(corpus_records),
        preprocessing_version="exact-text.v1",
    )
    values.update(changes)
    return CorpusProvenance(**values)


def bm25(corpus_records=None, *, library_version="0.2.2"):
    corpus_records = records() if corpus_records is None else corpus_records
    fingerprint = compute_index_fingerprint_sha256(
        corpus_manifest_sha256=compute_corpus_manifest_sha256(corpus_records),
        document_id_map_sha256=compute_document_id_map_sha256(corpus_records),
        library_version=library_version,
    )
    return build_bm25_retriever_provenance(
        library_version=library_version,
        index_fingerprint_sha256=fingerprint,
    )


def produce(*, corpus_records=None, raw_results=None, **changes):
    corpus_records = records() if corpus_records is None else corpus_records
    raw_results = (
        (RawCandidateResult(2, 0.25), RawCandidateResult("doc-a", 7.5))
        if raw_results is None
        else raw_results
    )
    values = dict(
        dataset_provenance=dataset(),
        corpus_provenance=corpus(corpus_records),
        sample_id="q-1",
        query_text="Exact Query?",
        retriever_provenance=bm25(corpus_records),
        requested_top_n=3,
        raw_results=raw_results,
        corpus_records=corpus_records,
        producing_git_commit=GIT_A,
        worktree_clean=True,
        environment_fingerprint_sha256=SHA_B,
    )
    values.update(changes)
    return produce_bm25_candidate_artifact(**values)


def test_valid_results_produce_candidate_artifact_without_sorting():
    value = produce()
    assert isinstance(value, CandidateArtifact)
    assert [candidate.document_id for candidate in value.candidates] == [2, "doc-a"]
    assert [candidate.native_score for candidate in value.candidates] == [0.25, 7.5]
    assert [candidate.rank for candidate in value.candidates] == [1, 2]
    assert [candidate.corpus_position for candidate in value.candidates] == [1, 0]


def test_candidate_entries_use_authoritative_corpus_fields():
    value = produce(raw_results=(RawCandidateResult(2, 4.0),))
    candidate = value.candidates[0]
    assert candidate.document_id == 2
    assert candidate.source_document_id == 200
    assert candidate.corpus_position == 1
    assert candidate.document_content_sha256 == document_content_sha256("Beta text ")
    with pytest.raises(TypeError):
        RawCandidateResult(document_id=2, native_score=4.0, text="stale text")


def test_document_checksum_is_exact_utf8_text():
    expected = hashlib.sha256("text".encode("utf-8")).hexdigest()
    assert document_content_sha256("text") == expected
    assert document_content_sha256("text ") != expected
    assert document_content_sha256(" text") != expected


def test_unknown_and_duplicate_raw_document_ids_are_rejected():
    with pytest.raises(ValueError, match="absent from corpus"):
        produce(raw_results=(RawCandidateResult("unknown", 1.0),))
    with pytest.raises(ValueError, match="must be unique"):
        produce(
            raw_results=(RawCandidateResult(2, 2.0), RawCandidateResult(2, 1.0))
        )


def test_raw_result_count_and_container_validation():
    with pytest.raises(ValueError, match="at least one"):
        produce(raw_results=())
    with pytest.raises(ValueError, match="cannot exceed"):
        produce(
            requested_top_n=1,
            raw_results=(RawCandidateResult(2, 2.0), RawCandidateResult("doc-a", 1.0)),
        )
    with pytest.raises(TypeError, match="ordered sequence"):
        produce(raw_results="not-results")


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, True, np.bool_(True)])
def test_raw_candidate_rejects_invalid_scores(score):
    with pytest.raises((TypeError, ValueError)):
        RawCandidateResult("doc-a", score)


def test_corpus_rejects_duplicate_ids_and_positions():
    duplicate_id = (
        CorpusRecord("doc", None, "a", 0),
        CorpusRecord("doc", None, "b", 1),
    )
    duplicate_position = (
        CorpusRecord("a", None, "a", 0),
        CorpusRecord("b", None, "b", 0),
    )
    with pytest.raises(ValueError, match="document_id values must be unique"):
        compute_corpus_manifest_sha256(duplicate_id)
    with pytest.raises(ValueError, match="corpus_position values must be unique"):
        compute_corpus_manifest_sha256(duplicate_position)


def test_corpus_rejects_position_gap_and_tuple_order_mismatch():
    gap = (
        CorpusRecord("a", None, "a", 0),
        CorpusRecord("b", None, "b", 2),
    )
    reversed_order = (
        CorpusRecord("b", None, "b", 1),
        CorpusRecord("a", None, "a", 0),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        compute_document_id_map_sha256(gap)
    with pytest.raises(ValueError, match="tuple order"):
        compute_document_id_map_sha256(reversed_order)


def test_producer_validates_corpus_provenance_against_records():
    with pytest.raises(ValueError, match="document_count"):
        produce(corpus_provenance=corpus(document_count=4))
    with pytest.raises(ValueError, match="manifest_sha256"):
        produce(corpus_provenance=corpus(manifest_sha256=SHA_A))
    with pytest.raises(ValueError, match="document_id_map_sha256"):
        produce(corpus_provenance=corpus(document_id_map_sha256=SHA_A))


def test_manifest_hashes_are_deterministic_and_sensitive():
    original = records()
    assert compute_corpus_manifest_sha256(original) == compute_corpus_manifest_sha256(original)
    assert compute_document_id_map_sha256(original) == compute_document_id_map_sha256(original)

    reordered = (
        replace(original[1], corpus_position=0),
        replace(original[0], corpus_position=1),
        original[2],
    )
    changed_text = (replace(original[0], text="Alpha text "), *original[1:])
    changed_source = (replace(original[0], source_document_id="other"), *original[1:])
    changed_id = (replace(original[0], document_id="other"), *original[1:])

    assert compute_corpus_manifest_sha256(reordered) != compute_corpus_manifest_sha256(original)
    assert compute_document_id_map_sha256(reordered) != compute_document_id_map_sha256(original)
    assert compute_corpus_manifest_sha256(changed_text) != compute_corpus_manifest_sha256(original)
    assert compute_document_id_map_sha256(changed_text) == compute_document_id_map_sha256(original)
    assert compute_corpus_manifest_sha256(changed_source) != compute_corpus_manifest_sha256(original)
    assert compute_document_id_map_sha256(changed_source) == compute_document_id_map_sha256(original)
    assert compute_corpus_manifest_sha256(changed_id) != compute_corpus_manifest_sha256(original)
    assert compute_document_id_map_sha256(changed_id) != compute_document_id_map_sha256(original)


def test_manifest_preserves_integer_and_string_id_types():
    integer = (CorpusRecord(1, None, "text", 0),)
    string = (CorpusRecord("1", None, "text", 0),)
    assert compute_corpus_manifest_sha256(integer) != compute_corpus_manifest_sha256(string)
    assert compute_document_id_map_sha256(integer) != compute_document_id_map_sha256(string)


def test_bm25_provenance_describes_current_repository_behavior():
    value = bm25()
    assert value.retriever_name == "bm25"
    assert value.implementation == "retrievers.bm25_retriever.BM25Retriever"
    assert value.library_name == "rank_bm25"
    assert value.model_id is None and value.tokenizer_id is None
    assert value.query_preprocessing == "lowercase then split on whitespace"
    assert value.normalization == "none"
    assert "constructor defaults" in value.index_config


@pytest.mark.parametrize(
    "change",
    [
        {"library_version": "new-version"},
        {"query_preprocessing": "different query preprocessing"},
        {"index_config": "different BM25 configuration"},
    ],
)
def test_index_fingerprint_changes_with_bm25_configuration(change):
    corpus_records = records()
    base = dict(
        corpus_manifest_sha256=compute_corpus_manifest_sha256(corpus_records),
        document_id_map_sha256=compute_document_id_map_sha256(corpus_records),
        library_version="0.2.2",
    )
    original = compute_index_fingerprint_sha256(**base)
    assert compute_index_fingerprint_sha256(**(base | change)) != original


def test_index_fingerprint_changes_with_corpus_binding():
    corpus_records = records()
    base = dict(
        corpus_manifest_sha256=compute_corpus_manifest_sha256(corpus_records),
        document_id_map_sha256=compute_document_id_map_sha256(corpus_records),
        library_version="0.2.2",
    )
    original = compute_index_fingerprint_sha256(**base)
    assert compute_index_fingerprint_sha256(**(base | {"corpus_manifest_sha256": SHA_A})) != original
    assert compute_index_fingerprint_sha256(**(base | {"document_id_map_sha256": SHA_A})) != original


def test_producer_rejects_bm25_fingerprint_bound_to_different_corpus():
    original_records = records()
    changed_records = (
        replace(original_records[0], text="different corpus text"),
        *original_records[1:],
    )
    with pytest.raises(ValueError, match="index fingerprint"):
        produce(
            corpus_records=changed_records,
            corpus_provenance=corpus(changed_records),
            retriever_provenance=bm25(original_records),
        )


def test_producer_rejects_bm25_configuration_changed_without_new_fingerprint():
    stale = bm25()
    changed_configuration = replace(
        stale,
        query_preprocessing="different query preprocessing",
    )
    with pytest.raises(ValueError, match="index fingerprint"):
        produce(retriever_provenance=changed_configuration)


@pytest.mark.parametrize(
    "invalid_provenance",
    [
        lambda value: replace(value, retriever_name="contriever"),
        lambda value: replace(value, library_name="other-library"),
        lambda value: replace(
            value,
            model_id="some-model",
            model_revision="some-revision",
        ),
        lambda value: replace(
            value,
            tokenizer_id="some-tokenizer",
            tokenizer_revision="some-revision",
        ),
    ],
)
def test_producer_rejects_non_bm25_provenance_family(invalid_provenance):
    with pytest.raises(ValueError, match="BM25|bm25"):
        produce(retriever_provenance=invalid_provenance(bm25()))


def test_production_environment_does_not_change_scientific_artifact_id():
    original = produce()
    reproduced = produce(
        producing_git_commit="2" * 40,
        worktree_clean=False,
        environment_fingerprint_sha256="c" * 64,
    )
    assert reproduced.artifact_id == original.artifact_id
    assert reproduced.production_fingerprint_sha256 != original.production_fingerprint_sha256


def test_artifact_has_no_downstream_identity_fields():
    fields = produce().__dataclass_fields__
    assert "diversification_condition" not in fields
    assert "model_id" not in fields
    assert "context_mode" not in fields
