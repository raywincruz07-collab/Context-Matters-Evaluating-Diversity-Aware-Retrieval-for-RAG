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
    build_contriever_cache_identity,
    build_contriever_retriever_provenance,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    produce_contriever_candidate_artifact,
    query_text_sha256,
)
from retrievers.contriever_config import CONTRIEVER_CONFIG


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
        entries=(SampleManifestEntry(0, 0, 101, query_text_sha256(QUERY)),),
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
    identity = build_contriever_cache_identity(
        corpus_manifest=manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    retriever = build_contriever_retriever_provenance(
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
            RawCandidateResult("1", -0.5),
            RawCandidateResult(3, 10.0),
            RawCandidateResult(1, 0.0),
        )
        if raw_results is None
        else raw_results,
        corpus_records=records,
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
    )
    values.update(changes)
    return produce_contriever_candidate_artifact(**values)


def test_valid_results_preserve_every_artifact_projection_and_native_order():
    value = produce()
    samples, dataset, _, _, corpus, _, retriever = inputs()
    assert isinstance(value, CandidateArtifact)
    assert value.sample_id == 0
    assert value.query_text == QUERY
    assert value.dataset == dataset
    assert value.dataset.sample_manifest_id == samples.manifest_id
    assert value.corpus == corpus
    assert value.retriever == retriever
    assert value.requested_top_n == 5
    assert [candidate.rank for candidate in value.candidates] == [1, 2, 3]
    assert [candidate.document_id for candidate in value.candidates] == ["1", 3, 1]
    assert [candidate.native_score for candidate in value.candidates] == [
        -0.5,
        10.0,
        0.0,
    ]
    assert [candidate.source_document_id for candidate in value.candidates] == [
        "source-str",
        303,
        "source-int",
    ]
    assert [candidate.corpus_position for candidate in value.candidates] == [1, 2, 0]
    assert value.candidates[0].document_content_sha256 == document_content_sha256(
        " retrieval str "
    )
    assert value.candidates[0].document_content_sha256 != document_content_sha256(
        "stored str"
    )
    assert value.producing_git_commit == GIT_SHA
    assert value.worktree_clean is True
    assert value.environment_fingerprint_sha256 == ENV_SHA


def test_identical_inputs_are_deterministic_and_short_pool_is_allowed():
    results = (RawCandidateResult(1, -1.0),)
    first = produce(raw_results=results)
    second = produce(raw_results=results)
    assert first == second
    assert first.artifact_id == second.artifact_id
    assert len(first.candidates) == 1 < first.requested_top_n


@pytest.mark.parametrize(
    "field,value,error,message",
    [
        ("sample_manifest", {}, TypeError, "SampleManifest"),
        ("dataset_provenance", {}, TypeError, "DatasetProvenance"),
        ("corpus_provenance", {}, TypeError, "CorpusProvenance"),
        ("retriever_provenance", {}, TypeError, "RetrieverProvenance"),
        ("cache_identity", {}, TypeError, "ContrieverCacheIdentity"),
    ],
)
def test_wrong_provenance_input_types_rejected(field, value, error, message):
    with pytest.raises(error, match=message):
        produce(**{field: value})


def test_sample_query_and_dataset_provenance_are_exactly_validated():
    with pytest.raises(KeyError, match="absent from the manifest"):
        produce(sample_id="0")
    with pytest.raises(ValueError, match="query_text does not match"):
        produce(query_text=f"{QUERY} ")
    _, dataset, *_ = inputs()
    with pytest.raises(ValueError, match="supplied SampleManifest"):
        produce(dataset_provenance=replace(dataset, revision="wrong-revision"))


def test_corpus_records_and_provenance_are_exactly_validated():
    _, _, records, manifest, corpus, _, _ = inputs()
    with pytest.raises(TypeError, match="immutable tuple"):
        produce(corpus_records=list(records))
    changed_records = (
        replace(records[0], retrieval_content="changed retrieval content"),
        *records[1:],
    )
    with pytest.raises(ValueError, match="retrieval_content"):
        produce(corpus_records=changed_records)
    with pytest.raises(ValueError, match="validated CorpusManifest"):
        produce(corpus_manifest=replace(manifest, revision="different-revision"))
    with pytest.raises(ValueError, match="validated CorpusManifest"):
        produce(
            corpus_provenance=replace(corpus, document_id_map_sha256="0" * 64)
        )


def test_cache_config_and_corpus_bindings_are_enforced():
    _, _, _, manifest, _, _, _ = inputs()
    changed_config = replace(CONTRIEVER_CONFIG, document_max_length=256)
    with pytest.raises(ValueError, match="ContrieverConfig does not match"):
        produce(contriever_config=changed_config)

    changed_manifest = replace(manifest, revision="different-corpus-revision")
    changed_identity = build_contriever_cache_identity(
        corpus_manifest=changed_manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    changed_retriever = build_contriever_retriever_provenance(
        cache_identity=changed_identity,
        index_artifact_sha256=INDEX_SHA,
        transformers_version="fixture-transformers-version",
    )
    with pytest.raises(ValueError, match="index_fingerprint_sha256"):
        produce(cache_identity=changed_identity)
    with pytest.raises(ValueError, match="CorpusManifest SHA"):
        produce(
            cache_identity=changed_identity,
            retriever_provenance=changed_retriever,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("retriever_name", "dpr"),
        ("model_id", "wrong-model"),
        ("model_revision", "0" * 40),
        ("normalization", "l2"),
        ("score_semantics", "cosine"),
        ("index_type", "faiss.IndexFlatL2"),
    ],
)
def test_wrong_contriever_retriever_provenance_rejected(field, value):
    retriever = inputs()[-1]
    with pytest.raises(ValueError, match=field):
        produce(retriever_provenance=replace(retriever, **{field: value}))


def test_missing_or_malformed_physical_faiss_sha_rejected():
    retriever = inputs()[-1]
    with pytest.raises(ValueError, match="physical FAISS artifact SHA"):
        produce(retriever_provenance=replace(retriever, index_artifact_sha256=None))
    malformed = replace(retriever)
    object.__setattr__(malformed, "index_artifact_sha256", "bad")
    with pytest.raises(ValueError, match="index_artifact_sha256"):
        produce(retriever_provenance=malformed)


@pytest.mark.parametrize("raw_results", ["results", b"results"])
def test_text_like_raw_result_containers_rejected(raw_results):
    with pytest.raises(TypeError, match="ordered sequence"):
        produce(raw_results=raw_results)


def test_raw_result_elements_count_uniqueness_and_membership_validated():
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
        produce(
            raw_results=(RawCandidateResult(1, 1.0), RawCandidateResult(1, 0.0))
        )
    with pytest.raises(ValueError, match="absent from corpus"):
        produce(raw_results=(RawCandidateResult("missing", 1.0),))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_raw_candidate_contract_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        RawCandidateResult(1, score)
