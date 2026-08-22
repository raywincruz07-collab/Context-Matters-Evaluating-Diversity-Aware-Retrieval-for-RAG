import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION, SAMPLE_MANIFEST_SCHEMA_VERSION,
    CandidateArtifactConflictError, CorpusManifest, CorpusManifestEntry,
    CorpusRecord, RawCandidateResult, SampleManifest, SampleManifestEntry,
    build_colbert_cache_identity, corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest, document_content_sha256,
    query_text_sha256, read_candidate_artifact,
)
from retrievers.colbert_config import COLBERT_CONFIG
from scripts.build_corpus_manifests import (
    ValidatedPubMedQAQuery, ValidatedPubMedQARuntimeCorpus,
)
from scripts.run_pubmedqa_colbert_candidates import (
    CANDIDATE_POOL_SIZE, DEFAULT_OUTPUT_DIR, run_pubmedqa_colbert_candidates,
)


GIT_SHA = "1" * 40
ENV_SHA = "2" * 64
INDEX_SHA = "3" * 64


def runtime():
    texts = ("Query zero?", "Query one?")
    samples = SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA, source="fixture", config="pqa_labeled",
        revision="revision", split="train", sampling_algorithm="fixture.v1",
        sampling_seed=None, requested_sample_size=None, selection_dependencies=(),
        entries=tuple(SampleManifestEntry(
            position=i, sample_id=i, source_sample_id=f"pub-{i}",
            query_text_sha256=query_text_sha256(text),
        ) for i, text in enumerate(texts)),
    )
    records = tuple(CorpusRecord(
        document_id=f"doc-{i}", source_document_id=f"source-{i}", title=None,
        text=f"text {i}", retrieval_content=f"exact retrieval {i}", corpus_position=i,
    ) for i in range(20))
    corpus_manifest = CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA, source="fixture corpus", config=None,
        revision="revision", split="train", construction_algorithm="fixture.v1",
        input_sample_manifest_id=samples.manifest_id,
        input_sample_manifest_sha256=samples.sha256, dependencies=(), rng_family=None,
        sampling_seed=None, rng_state_semantics=None,
        requested_negatives_per_query=None, negative_sampling_scope=None,
        negative_exclusion_scope=None, negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=tuple(CorpusManifestEntry(
            i, record.document_id, record.source_document_id, None,
            document_content_sha256(record.text),
            document_content_sha256(record.retrieval_content),
        ) for i, record in enumerate(records)),
    )
    dataset = dataset_provenance_from_sample_manifest(samples)
    corpus = corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest, corpus_records=records,
        dataset_provenance=dataset,
    )
    return ValidatedPubMedQARuntimeCorpus(
        sample_manifest=samples, corpus_manifest=corpus_manifest,
        corpus_records=records, dataset_provenance=dataset,
        corpus_provenance=corpus,
        ordered_queries=tuple(ValidatedPubMedQAQuery(i, i, text)
                              for i, text in enumerate(texts)),
    )


class FakeRetriever:
    config = COLBERT_CONFIG
    cache_identity = None
    index_artifact_sha256 = None
    def __init__(self): self.index_calls = 0; self.retrieve_calls = []
    def index_from_corpus_records(self, *, corpus_manifest, corpus_records):
        self.index_calls += 1
        self.records = corpus_records
        self.cache_identity = build_colbert_cache_identity(
            corpus_manifest=corpus_manifest,
            checkpoint_snapshot_manifest_sha256="c" * 64,
        )
        self.index_artifact_sha256 = INDEX_SHA
    def retrieve(self, query, *, top_k):
        self.retrieve_calls.append((query, top_k))
        return tuple(RawCandidateResult(record.document_id, 20.0-rank)
                     for rank, record in enumerate(self.records))


def run(path, retriever=None, **changes):
    values = dict(
        runtime=runtime(), retriever=retriever or FakeRetriever(), output_dir=path,
        producing_git_commit=GIT_SHA, worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
    )
    values.update(changes)
    return run_pubmedqa_colbert_candidates(**values)


def test_runner_uses_canonical_path_indexes_once_and_limits_manifest_order(tmp_path):
    assert CANDIDATE_POOL_SIZE == 20
    assert DEFAULT_OUTPUT_DIR.name == "colbertv2"
    retriever = FakeRetriever()
    summary = run(tmp_path, retriever, max_samples=1)
    assert retriever.index_calls == 1
    assert retriever.retrieve_calls == [("Query zero?", 20)]
    assert summary.completed_sample_ids == (0,)
    artifact = read_candidate_artifact(tmp_path / "sample_0000.json")
    assert len(artifact.candidates) == 20
    assert [item.document_id for item in artifact.candidates] == [f"doc-{i}" for i in range(20)]
    assert artifact.retriever.model_revision == COLBERT_CONFIG.checkpoint_revision
    assert "checkpoint_snapshot_manifest_sha256" in artifact.retriever.index_config


def test_runner_resumes_exact_artifact_and_rejects_conflict(tmp_path):
    first = run(tmp_path, max_samples=1)
    assert first.completed_samples == 1
    second = run(tmp_path, max_samples=1)
    assert second.skipped_existing_samples == 1
    with pytest.raises(CandidateArtifactConflictError):
        run(tmp_path, max_samples=1, environment_fingerprint_sha256="9" * 64)


def test_producer_failure_is_reported_without_partial_artifact(tmp_path):
    retriever = FakeRetriever()
    def invalid(query, *, top_k):
        results = list(FakeRetriever.retrieve(retriever, query, top_k=top_k))
        results[1] = RawCandidateResult(results[1].document_id, 99.0)
        return tuple(results)
    retriever.retrieve = invalid
    summary = run(tmp_path, retriever, max_samples=1)
    assert summary.failed_samples == 1
    assert summary.failures[0].error_type == "ValueError"
    assert not (tmp_path / "sample_0000.json").exists()
