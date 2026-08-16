from dataclasses import replace
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    CandidateArtifact,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    RawCandidateResult,
    SampleManifest,
    SampleManifestEntry,
    build_dpr_cache_identity,
    build_dpr_retriever_provenance,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    produce_dpr_candidate_artifact,
    query_text_sha256,
)
from retrievers.dpr_config import DPR_CONFIG


QUERY = "Exact Query?"
GIT_SHA = "1" * 40
ENV_SHA = "e" * 64
INDEX_SHA = "f" * 64


def sample_manifest():
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-query-source",
        config="fixture-config",
        revision="fixture-query-revision",
        split="fixture-split",
        sampling_algorithm="fixture-source-order.v1",
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=(
            SampleManifestEntry(0, 0, 101, query_text_sha256(QUERY)),
        ),
    )


def corpus_records():
    return (
        CorpusRecord(1, "source-int", None, "stored int", "retrieval int", 0),
        CorpusRecord("1", "source-str", None, "stored str", " retrieval str ", 1),
        CorpusRecord(3, 303, "Title", "stored third", "retrieval third", 2),
    )


def corpus_manifest(records=None, samples=None, **changes):
    records = corpus_records() if records is None else records
    samples = sample_manifest() if samples is None else samples
    values = dict(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus-source",
        config=None,
        revision="fixture-corpus-revision",
        split="fixture-split",
        construction_algorithm="fixture-prepared-content.v1",
        input_sample_manifest_id=samples.manifest_id,
        input_sample_manifest_sha256=samples.sha256,
        dependencies=(),
        rng_family=None,
        sampling_seed=None,
        rng_state_semantics=None,
        requested_negatives_per_query=None,
        negative_sampling_scope=None,
        negative_exclusion_scope=None,
        negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=tuple(
            CorpusManifestEntry(
                position=record.corpus_position,
                doc_id=record.document_id,
                source_document_id=record.source_document_id,
                title_sha256=(
                    None
                    if record.title is None
                    else document_content_sha256(record.title)
                ),
                text_sha256=document_content_sha256(record.text),
                retrieval_content_sha256=document_content_sha256(
                    record.retrieval_content
                ),
            )
            for record in records
        ),
    )
    values.update(changes)
    return CorpusManifest(**values)


def inputs():
    samples = sample_manifest()
    dataset = dataset_provenance_from_sample_manifest(samples)
    records = corpus_records()
    manifest = corpus_manifest(records, samples)
    corpus = corpus_provenance_from_corpus_manifest(
        corpus_manifest=manifest,
        corpus_records=records,
        dataset_provenance=dataset,
    )
    identity = build_dpr_cache_identity(
        corpus_manifest=manifest,
        dpr_config=DPR_CONFIG,
    )
    retriever = build_dpr_retriever_provenance(
        cache_identity=identity,
        index_artifact_sha256=INDEX_SHA,
        transformers_version="fixture-transformers-version",
    )
    return samples, dataset, records, manifest, corpus, identity, retriever


def produce(*, raw_results=None, **changes):
    samples, dataset, records, manifest, corpus, identity, retriever = inputs()
    values = dict(
        sample_manifest=samples,
        dataset_provenance=dataset,
        corpus_manifest=manifest,
        corpus_provenance=corpus,
        cache_identity=identity,
        sample_id=0,
        query_text=QUERY,
        retriever_provenance=retriever,
        requested_top_n=5,
        raw_results=(
            RawCandidateResult("1", -2.5),
            RawCandidateResult(1, 0.0),
            RawCandidateResult(3, 7.25),
        ) if raw_results is None else raw_results,
        corpus_records=records,
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
    )
    values.update(changes)
    return produce_dpr_candidate_artifact(**values)


def test_valid_dpr_results_preserve_order_scores_ids_and_corpus_fields():
    value = produce()
    assert isinstance(value, CandidateArtifact)
    assert value.retriever == inputs()[-1]
    assert value.requested_top_n == 5
    assert [candidate.rank for candidate in value.candidates] == [1, 2, 3]
    assert [candidate.document_id for candidate in value.candidates] == ["1", 1, 3]
    assert [candidate.native_score for candidate in value.candidates] == [-2.5, 0.0, 7.25]
    assert [candidate.corpus_position for candidate in value.candidates] == [1, 0, 2]
    assert [candidate.source_document_id for candidate in value.candidates] == [
        "source-str", "source-int", 303
    ]
    assert value.candidates[0].document_content_sha256 == document_content_sha256(
        " retrieval str "
    )
    assert isinstance(value.candidates[0].document_id, str)
    assert isinstance(value.candidates[1].document_id, int)


def test_short_result_pool_is_allowed_and_identity_is_deterministic():
    result = (RawCandidateResult(1, -1.0),)
    first = produce(raw_results=result)
    second = produce(raw_results=result)
    assert len(first.candidates) == 1 < first.requested_top_n
    assert first.artifact_id == second.artifact_id


def test_dpr_cache_config_and_corpus_bindings_are_enforced():
    samples, dataset, records, manifest, corpus, identity, retriever = inputs()
    changed_config = replace(DPR_CONFIG, context_max_length=128)
    with pytest.raises(ValueError, match="DPRConfig does not match"):
        produce(dpr_config=changed_config)

    changed_manifest = replace(manifest, revision="different-corpus-revision")
    changed_identity = build_dpr_cache_identity(
        corpus_manifest=changed_manifest,
        dpr_config=DPR_CONFIG,
    )
    changed_retriever = build_dpr_retriever_provenance(
        cache_identity=changed_identity,
        index_artifact_sha256=INDEX_SHA,
        transformers_version="fixture-transformers-version",
    )
    with pytest.raises(ValueError, match="index_fingerprint_sha256"):
        produce(cache_identity=changed_identity)
    with pytest.raises(ValueError, match="CorpusManifest SHA"):
        produce(cache_identity=changed_identity, retriever_provenance=changed_retriever)


def test_mismatched_manifest_and_concrete_records_are_rejected():
    _, _, records, manifest, _, _, _ = inputs()
    with pytest.raises(ValueError, match="validated CorpusManifest"):
        produce(corpus_manifest=replace(manifest, revision="different-revision"))
    changed_records = (
        replace(records[0], retrieval_content="changed retrieval content"),
        *records[1:],
    )
    with pytest.raises(ValueError, match="retrieval_content"):
        produce(corpus_records=changed_records)


def test_sample_query_and_dataset_provenance_are_exactly_validated():
    with pytest.raises(KeyError, match="absent from the manifest"):
        produce(sample_id="0")
    with pytest.raises(ValueError, match="query_text does not match"):
        produce(query_text=f"{QUERY} ")
    _, dataset, *_ = inputs()
    with pytest.raises(ValueError, match="supplied SampleManifest"):
        produce(dataset_provenance=replace(dataset, revision="wrong-revision"))


def test_raw_result_container_elements_count_and_ids_are_validated():
    with pytest.raises(TypeError, match="ordered sequence"):
        produce(raw_results={RawCandidateResult(1, 1.0)})
    with pytest.raises(ValueError, match="at least one"):
        produce(raw_results=())
    with pytest.raises(TypeError, match="RawCandidateResult"):
        produce(raw_results=[("1", 1.0)])
    with pytest.raises(ValueError, match="cannot exceed"):
        produce(
            requested_top_n=1,
            raw_results=(RawCandidateResult(1, 1.0), RawCandidateResult("1", 0.0)),
        )
    with pytest.raises(ValueError, match="must be unique"):
        produce(raw_results=(RawCandidateResult(1, 1.0), RawCandidateResult(1, 0.0)))
    with pytest.raises(ValueError, match="absent from corpus"):
        produce(raw_results=(RawCandidateResult("missing", 1.0),))
