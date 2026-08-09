from dataclasses import FrozenInstanceError, replace
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts.contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEmbeddingArtifactRef,
    CandidateEntry,
    CorpusProvenance,
    DatasetProvenance,
    RetrieverProvenance,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
GIT_A = "1" * 40


def entry(rank=1, document_id="doc-1", **changes):
    values = dict(
        rank=rank,
        document_id=document_id,
        source_document_id=f"source-{rank}",
        corpus_position=rank - 1,
        native_score=10.0 - rank,
        document_content_sha256=SHA_A,
    )
    values.update(changes)
    return CandidateEntry(**values)


def dataset(**changes):
    values = dict(
        dataset_id=DatasetId.PUBMEDQA,
        source="datasets/pubmedqa",
        config="pqa_labeled",
        revision="dataset-revision",
        split="train",
        sample_manifest_id="manifest-v1",
        sample_manifest_sha256=SHA_A,
        sampling_seed=42,
        sampling_algorithm="numpy.default_rng.choice.v1",
    )
    values.update(changes)
    return DatasetProvenance(**values)


def corpus(**changes):
    values = dict(
        corpus_id="pubmedqa-contexts",
        source="derived PubMedQA contexts",
        revision="corpus-revision",
        document_count=100,
        manifest_sha256=SHA_A,
        document_id_map_sha256=SHA_B,
        preprocessing_version="identity.v1",
    )
    values.update(changes)
    return CorpusProvenance(**values)


def retriever(**changes):
    values = dict(
        retriever_name="contriever",
        implementation="retrievers.contriever.ContrieverRetriever",
        library_name="transformers",
        library_version="5.14.1",
        model_id="facebook/contriever",
        model_revision="model-revision",
        tokenizer_id="facebook/contriever",
        tokenizer_revision="tokenizer-revision",
        query_preprocessing="raw text v1",
        document_preprocessing="raw text v1",
        normalization="L2",
        score_semantics="inner product of normalized embeddings",
        index_type="numpy exact matrix",
        index_config="float32; stable descending score order",
        index_fingerprint_sha256=SHA_A,
        index_artifact_sha256=None,
    )
    values.update(changes)
    return RetrieverProvenance(**values)


def artifact(**changes):
    values = dict(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        dataset=dataset(),
        corpus=corpus(),
        sample_id="q-1",
        query_text="  Exact query text?  ",
        retriever=retriever(),
        requested_top_n=2,
        candidates=(entry(), entry(2, "doc-2")),
        producing_git_commit=GIT_A,
        worktree_clean=True,
        environment_fingerprint_sha256=SHA_B,
    )
    values.update(changes)
    return CandidateArtifact(**values)


@pytest.mark.parametrize("document_id", ["doc", 7, np.int64(8)])
def test_candidate_entry_accepts_stable_string_and_integer_ids(document_id):
    assert entry(document_id=document_id).document_id == document_id


@pytest.mark.parametrize("document_id", [True, np.bool_(True), "", "  ", 1.5, object()])
def test_candidate_entry_rejects_invalid_document_ids(document_id):
    with pytest.raises((TypeError, ValueError)):
        entry(document_id=document_id)


@pytest.mark.parametrize("rank", [0, True, np.bool_(False), 1.0])
def test_candidate_entry_rejects_invalid_rank(rank):
    with pytest.raises((TypeError, ValueError)):
        entry(rank=rank)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, True, np.bool_(True)])
def test_candidate_entry_rejects_invalid_score(score):
    with pytest.raises((TypeError, ValueError)):
        entry(native_score=score)


def test_candidate_entry_rejects_malformed_checksum():
    with pytest.raises(ValueError):
        entry(document_content_sha256="A" * 64)


def test_dataset_provenance_validation():
    with pytest.raises(ValueError):
        dataset(revision="")
    with pytest.raises(ValueError):
        dataset(sample_manifest_sha256="bad")
    with pytest.raises(TypeError):
        dataset(sampling_seed=True)
    with pytest.raises((TypeError, ValueError)):
        dataset(sampling_seed=1, sampling_algorithm=None)
    assert dataset(sampling_seed=None, sampling_algorithm=None).sampling_seed is None


def test_corpus_provenance_validation():
    with pytest.raises(ValueError):
        corpus(document_count=0)
    with pytest.raises(ValueError):
        corpus(manifest_sha256="A" * 64)
    with pytest.raises(ValueError):
        corpus(document_id_map_sha256="bad")


def test_retriever_model_and_tokenizer_revision_pairs():
    with pytest.raises(ValueError):
        retriever(model_revision=None)
    with pytest.raises(ValueError):
        retriever(model_id=None)
    with pytest.raises(ValueError):
        retriever(tokenizer_revision=None)
    with pytest.raises(ValueError):
        retriever(tokenizer_id=None)
    bm25 = retriever(
        retriever_name="bm25",
        model_id=None,
        model_revision=None,
        tokenizer_id=None,
        tokenizer_revision=None,
    )
    assert bm25.model_id is None


def test_candidate_artifact_is_frozen_and_requires_tuple():
    value = artifact()
    with pytest.raises(FrozenInstanceError):
        value.sample_id = "changed"
    with pytest.raises(TypeError):
        artifact(candidates=list(value.candidates))


def test_candidate_artifact_candidate_invariants():
    with pytest.raises(ValueError):
        artifact(candidates=())
    with pytest.raises(ValueError):
        artifact(candidates=(entry(rank=2),))
    with pytest.raises(ValueError):
        artifact(candidates=(entry(), entry(2, "doc-1")))
    with pytest.raises(ValueError):
        artifact(candidates=(entry(), entry(2, "doc-2", corpus_position=0)))
    with pytest.raises(ValueError):
        artifact(requested_top_n=1)
    assert len(artifact(requested_top_n=20).candidates) == 2


def test_candidate_artifact_preserves_exact_query_and_records_dirty_worktree():
    value = artifact(worktree_clean=False)
    assert value.query_text == "  Exact query text?  "
    assert value.worktree_clean is False
    with pytest.raises(ValueError):
        artifact(producing_git_commit="1" * 39)


def test_canonical_serialization_is_deterministic_and_type_stable():
    value = artifact(sample_id=9, candidates=(entry(document_id=3),))
    assert value.artifact_id == value.artifact_id
    assert artifact().artifact_id == artifact().artifact_id
    payload = json.loads(value.canonical_json())
    assert payload["sample_id"] == 9
    assert payload["candidates"][0]["document_id"] == 3
    assert payload["candidates"][0]["native_score_hex"] == float(9.0).hex()
    assert re.fullmatch(r"candidate:sha256:[0-9a-f]{64}", value.artifact_id)


@pytest.mark.parametrize(
    "changed",
    [
        lambda a: replace(a, candidates=(replace(a.candidates[1], rank=1), replace(a.candidates[0], rank=2))),
        lambda a: replace(a, candidates=(replace(a.candidates[0], document_id="other"), a.candidates[1])),
        lambda a: replace(a, candidates=(replace(a.candidates[0], native_score=8.5), a.candidates[1])),
        lambda a: replace(a, candidates=(replace(a.candidates[0], document_content_sha256=SHA_B), a.candidates[1])),
        lambda a: replace(a, requested_top_n=3),
        lambda a: replace(a, query_text="different"),
        lambda a: replace(a, sample_id="q-2"),
        lambda a: replace(a, dataset=replace(a.dataset, revision="new-revision")),
        lambda a: replace(a, corpus=replace(a.corpus, manifest_sha256="c" * 64)),
        lambda a: replace(a, corpus=replace(a.corpus, document_id_map_sha256="c" * 64)),
        lambda a: replace(a, retriever=replace(a.retriever, model_revision="new-model-revision")),
        lambda a: replace(a, retriever=replace(a.retriever, query_preprocessing="new preprocessing")),
        lambda a: replace(a, retriever=replace(a.retriever, index_fingerprint_sha256="c" * 64)),
        lambda a: replace(a, producing_git_commit="2" * 40),
        lambda a: replace(a, environment_fingerprint_sha256="c" * 64),
    ],
)
def test_artifact_id_changes_with_scientific_payload(changed):
    original = artifact()
    assert changed(original).artifact_id != original.artifact_id


def test_embedding_reference_validation_and_exact_binding():
    candidate = artifact()
    ref = CandidateEmbeddingArtifactRef(
        representation_name="contriever diversification embeddings",
        model_id="facebook/contriever",
        model_revision="revision",
        tokenizer_revision="revision",
        pooling="mean pooling",
        truncation="max_length=512",
        normalization="L2",
        embedding_dimension=768,
        dtype="float32",
        artifact_sha256=SHA_A,
        candidate_artifact_id=candidate.artifact_id,
    )
    assert ref.candidate_artifact_id == candidate.artifact_id
    with pytest.raises(ValueError):
        replace(ref, embedding_dimension=0)
    with pytest.raises(ValueError):
        replace(ref, artifact_sha256="bad")
    with pytest.raises(ValueError):
        replace(ref, candidate_artifact_id=candidate.sha256)
    with pytest.raises(ValueError):
        replace(ref, model_revision="")


def test_top_twenty_raw_artifact_has_no_downstream_experiment_identity():
    candidates = tuple(
        entry(rank=i, document_id=f"doc-{i}", corpus_position=i - 1)
        for i in range(1, 21)
    )
    value = artifact(requested_top_n=20, candidates=candidates)
    fields = value.__dataclass_fields__
    assert len(value.candidates) == 20
    assert "diversification_condition" not in fields
    assert "model_id" not in fields
    assert "context_mode" not in fields
