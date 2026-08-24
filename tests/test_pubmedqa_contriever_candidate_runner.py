from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    CandidateArtifactConflictError,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    SampleManifest,
    SampleManifestEntry,
    build_contriever_cache_identity,
    build_contriever_retriever_provenance,
    candidate_artifact_payload,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    produce_contriever_candidate_artifact,
    query_text_sha256,
    read_candidate_artifact,
)
from retrieval_artifacts.candidate_production import (
    build_retrieval_planned_record,
    candidate_production_plan_scientific_payload,
    plan_candidate_production,
    running_candidate_record,
)
from retrievers.contriever_config import CONTRIEVER_CONFIG
from scripts.build_corpus_manifests import (
    ValidatedPubMedQAQuery,
    ValidatedPubMedQARuntimeCorpus,
)
import scripts.run_pubmedqa_contriever_candidates as runner_module
from scripts.run_pubmedqa_contriever_candidates import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_OUTPUT_DIR,
    ContrieverCandidatePoolSizeError,
    LocalContrieverIndexIdentity,
    _environment_fingerprint,
    build_runtime_contriever_provenance,
    run_governed_pubmedqa_contriever_candidates,
    run_pubmedqa_contriever_candidates,
    validate_local_contriever_index_identity,
)
from run_registry import (
    EVIDENCE_AUTHORITY_SCHEMA_VERSION,
    REGISTRY_HEADER,
    RunRecordValidationError,
    RunRegistryConflictError,
    append_run_record,
    canonical_json,
    read_registry,
)


GIT_SHA = "1" * 40
ENV_SHA = "2" * 64
INDEX_SHA = "3" * 64
EMBEDDING_SHA = "4" * 64
TRANSFORMERS_VERSION = "fixture-transformers-version"
RUNTIME_SHA = "5" * 64


def sample_manifest():
    queries = ((5, "Exact Query A?"), (2, " Exact Query B? "))
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-query-source",
        config="fixture-config",
        revision="fixture-query-revision",
        split="fixture-split",
        sampling_algorithm="fixture-order.v1",
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=tuple(
            SampleManifestEntry(
                position=position,
                sample_id=sample_id,
                source_sample_id=100 + position,
                query_text_sha256=query_text_sha256(text),
            )
            for position, (sample_id, text) in enumerate(queries)
        ),
    )


def records():
    return tuple(
        CorpusRecord(
            document_id=position if position % 2 == 0 else f"doc-{position}",
            source_document_id=f"source-{position}",
            title=None,
            text=f"stored text {position}",
            retrieval_content=f" retrieval content {position} ",
            corpus_position=position,
        )
        for position in range(CANDIDATE_POOL_SIZE)
    )


def manifest(samples, corpus_records):
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus-source",
        config="fixture-corpus",
        revision="fixture-corpus-revision",
        split="fixture-corpus-split",
        construction_algorithm="fixture-corpus.v1",
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
                title_sha256=None,
                text_sha256=document_content_sha256(record.text),
                retrieval_content_sha256=document_content_sha256(
                    record.retrieval_content
                ),
            )
            for record in corpus_records
        ),
    )


def runtime():
    samples = sample_manifest()
    corpus_records = records()
    corpus_manifest = manifest(samples, corpus_records)
    dataset = dataset_provenance_from_sample_manifest(samples)
    corpus = corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset,
    )
    texts = ("Exact Query A?", " Exact Query B? ")
    return ValidatedPubMedQARuntimeCorpus(
        sample_manifest=samples,
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset,
        corpus_provenance=corpus,
        ordered_queries=tuple(
            ValidatedPubMedQAQuery(entry.position, entry.sample_id, text)
            for entry, text in zip(samples.entries, texts, strict=True)
        ),
    )


class FakeContriever:
    def __init__(
        self,
        *,
        config=CONTRIEVER_CONFIG,
        result_count=CANDIDATE_POOL_SIZE,
        fail_query=None,
        fail_index=False,
    ):
        self.config = config
        self.result_count = result_count
        self.fail_query = fail_query
        self.fail_index = fail_index
        self.index_calls = []
        self.retrieve_calls = []
        self.is_indexed = False
        self.cache_identity = None
        self.index_artifact_sha256 = None
        self.embedding_artifact_sha256 = None
        self.indexed_records = None

    def index_from_corpus_records(self, *, corpus_manifest, corpus_records):
        self.index_calls.append((corpus_manifest, corpus_records))
        if self.fail_index:
            raise RuntimeError("synthetic index failure")
        self.indexed_records = corpus_records
        self.cache_identity = build_contriever_cache_identity(
            corpus_manifest=corpus_manifest,
            contriever_config=self.config,
        )
        self.index_artifact_sha256 = INDEX_SHA
        self.embedding_artifact_sha256 = EMBEDDING_SHA
        self.is_indexed = True

    def retrieve(self, query, top_k):
        self.retrieve_calls.append((query, top_k))
        if query == self.fail_query:
            raise RuntimeError("synthetic retrieval failure")
        ordered = list(reversed(self.indexed_records))
        results = []
        for position in range(self.result_count):
            record = ordered[position % len(ordered)]
            score = -2.5 if position == 0 else (0.0 if position == 1 else 30.0 - position)
            results.append(({"doc_id": record.document_id}, score))
        return results


def run(output_dir, retriever=None, **changes):
    values = dict(
        runtime=runtime(),
        retriever=FakeContriever() if retriever is None else retriever,
        output_dir=output_dir,
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
        transformers_version=TRANSFORMERS_VERSION,
    )
    values.update(changes)
    return run_pubmedqa_contriever_candidates(**values)


def _governed_fixture(tmp_path, *, preexisting=False):
    active_runtime = runtime()
    output_dir = tmp_path / "candidates"
    if preexisting:
        run(output_dir, max_samples=1)
    cache_identity = build_contriever_cache_identity(
        corpus_manifest=active_runtime.corpus_manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    provenance = build_contriever_retriever_provenance(
        cache_identity=cache_identity,
        index_artifact_sha256=INDEX_SHA,
        transformers_version=TRANSFORMERS_VERSION,
    )
    local_index = LocalContrieverIndexIdentity(
        cache_fingerprint_sha256=cache_identity.fingerprint_sha256,
        embedding_path=tmp_path / "fixture.npy",
        embedding_artifact_sha256=EMBEDDING_SHA,
        index_path=tmp_path / "fixture.faiss",
        index_artifact_sha256=INDEX_SHA,
        metadata_path=tmp_path / "fixture.json",
        retriever_provenance=provenance,
        cache_environment={"device": "cpu"},
    )
    plan = plan_candidate_production(
        sample_manifest=active_runtime.sample_manifest,
        ordered_queries=active_runtime.ordered_queries,
        candidate_directory=output_dir,
        dataset_provenance=active_runtime.dataset_provenance,
        corpus_provenance=active_runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=active_runtime.corpus_records,
        candidate_pool=CANDIDATE_POOL_SIZE,
    )
    plan_payload = candidate_production_plan_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=active_runtime.sample_manifest.manifest_id,
        corpus_manifest_id=active_runtime.corpus_manifest.corpus_manifest_id,
        retriever="contriever",
        retriever_config_sha256=hashlib.sha256(
            CONTRIEVER_CONFIG.scientific_json().encode("utf-8")
        ).hexdigest(),
        index_artifact_id=f"index:sha256:{cache_identity.fingerprint_sha256}",
        candidate_pool=CANDIDATE_POOL_SIZE,
        top_k=5,
        candidate_directory="candidates",
        scheduled_sample_ids=plan.scheduled_sample_ids,
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
                "authorities": [
                    {
                        "dataset": "pubmedqa",
                        "evidence_role": (
                            "HISTORICAL_OBSERVED_CONTROL_REPLICATION"
                        ),
                        "sample_manifest_id": (
                            active_runtime.sample_manifest.manifest_id
                        ),
                        "sample_manifest_path": "fixture/sample.json",
                        "authority_protocols": ["fixture/protocol.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = lambda path, sha256, artifact_id: {
        "path": path,
        "sha256": sha256,
        "artifact_id": artifact_id,
    }
    planned = build_retrieval_planned_record(
        created_at="2026-08-24T10:00:00Z",
        plan_payload=plan_payload,
        git={
            "commit": GIT_SHA,
            "branch": "sprint3",
            "worktree_clean": True,
            "worktree_diff_sha256": None,
        },
        data={
            "dataset": "pubmedqa",
            "split": "fixture-split",
            "source": "fixture-source",
            "revision": "fixture-revision",
            "sample_manifest": artifact(
                "fixture/sample.json",
                "6" * 64,
                active_runtime.sample_manifest.manifest_id,
            ),
            "corpus_manifest": artifact(
                "fixture/corpus.json",
                "7" * 64,
                active_runtime.corpus_manifest.corpus_manifest_id,
            ),
        },
        retrieval_index=artifact(
            "fixture/index.faiss",
            INDEX_SHA,
            f"index:sha256:{cache_identity.fingerprint_sha256}",
        ),
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-a",
        output_directory="candidates",
        evidence_authority_path=authority_path,
    )
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(
        canonical_json(REGISTRY_HEADER) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(
        runtime=active_runtime,
        output_dir=output_dir,
        local_index=local_index,
        plan=plan,
        plan_payload=plan_payload,
        planned=planned,
        authority_path=authority_path,
        registry_path=registry_path,
        run_artifact_root=tmp_path / "runs",
        candidate_set_path=tmp_path / "candidate-set.json",
    )


def _append_first_running_attempt(fixture):
    append_run_record(
        fixture.registry_path,
        fixture.planned,
        evidence_authority_path=fixture.authority_path,
    )
    running = running_candidate_record(
        fixture.planned,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=1,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-a",
        evidence_authority_path=fixture.authority_path,
    )
    append_run_record(
        fixture.registry_path,
        running,
        evidence_authority_path=fixture.authority_path,
    )
    return running


def test_constants_paths_index_once_and_exact_producer_inputs(tmp_path):
    assert CANDIDATE_POOL_SIZE == 20
    assert DEFAULT_OUTPUT_DIR == (
        Path(__file__).resolve().parents[1]
        / "artifacts/candidates/pubmedqa/contriever"
    )
    retriever = FakeContriever()
    calls = []

    def producer_spy(**kwargs):
        calls.append(kwargs)
        return produce_contriever_candidate_artifact(**kwargs)

    summary = run(tmp_path, retriever, max_samples=1, producer=producer_spy)
    expected = runtime()
    assert len(retriever.index_calls) == 1
    assert retriever.index_calls[0] == (
        expected.corpus_manifest,
        expected.corpus_records,
    )
    assert retriever.retrieve_calls == [("Exact Query A?", 20)]
    assert calls[0]["sample_manifest"] == expected.sample_manifest
    assert calls[0]["dataset_provenance"] == expected.dataset_provenance
    assert calls[0]["corpus_manifest"] == expected.corpus_manifest
    assert calls[0]["corpus_provenance"] == expected.corpus_provenance
    assert calls[0]["corpus_records"] == expected.corpus_records
    assert calls[0]["cache_identity"] == retriever.cache_identity
    assert calls[0]["requested_top_n"] == 20
    assert [result.document_id for result in calls[0]["raw_results"]] == [
        record.document_id for record in reversed(expected.corpus_records)
    ]
    assert [result.native_score for result in calls[0]["raw_results"]][:2] == [
        -2.5,
        0.0,
    ]
    artifact = read_candidate_artifact(tmp_path / "sample_0000.json")
    assert len(artifact.candidates) == 20
    assert [candidate.native_score for candidate in artifact.candidates][:2] == [
        -2.5,
        0.0,
    ]
    assert summary.completed_sample_ids == (5,)
    assert summary.index_artifact_sha256 == INDEX_SHA
    assert summary.embedding_artifact_sha256 == EMBEDDING_SHA


def test_all_queries_frozen_order_and_max_samples_validation(tmp_path):
    retriever = FakeContriever()
    summary = run(tmp_path, retriever)
    assert summary.completed_sample_ids == (5, 2)
    assert retriever.retrieve_calls == [
        ("Exact Query A?", 20),
        (" Exact Query B? ", 20),
    ]
    assert len(retriever.index_calls) == 1
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            run(tmp_path / str(invalid), max_samples=invalid)
    for invalid in (True, False, 1.5, "1"):
        with pytest.raises(TypeError, match="non-boolean integer"):
            run(tmp_path / str(invalid), max_samples=invalid)


def test_config_mismatch_rejected_before_indexing(tmp_path):
    changed = FakeContriever(
        config=replace(CONTRIEVER_CONFIG, document_max_length=256)
    )
    with pytest.raises(ValueError, match="frozen CONTRIEVER_CONFIG"):
        run(tmp_path, changed)
    assert changed.index_calls == []


@pytest.mark.parametrize(
    "field,value,error,message",
    [
        ("is_indexed", False, RuntimeError, "must be indexed"),
        ("cache_identity", None, RuntimeError, "missing cache_identity"),
        ("embedding_artifact_sha256", None, ValueError, "embedding_artifact_sha256"),
        ("embedding_artifact_sha256", "bad", ValueError, "embedding_artifact_sha256"),
        ("index_artifact_sha256", None, ValueError, "index_artifact_sha256"),
        ("index_artifact_sha256", "bad", ValueError, "index_artifact_sha256"),
    ],
)
def test_runtime_provenance_requires_complete_index_state(field, value, error, message):
    active_runtime = runtime()
    retriever = FakeContriever()
    retriever.index_from_corpus_records(
        corpus_manifest=active_runtime.corpus_manifest,
        corpus_records=active_runtime.corpus_records,
    )
    setattr(retriever, field, value)
    with pytest.raises(error, match=message):
        build_runtime_contriever_provenance(
            active_runtime,
            retriever,
            transformers_version=TRANSFORMERS_VERSION,
        )


def test_runtime_provenance_is_canonical_and_binding_validator_runs(monkeypatch):
    active_runtime = runtime()
    retriever = FakeContriever()
    retriever.index_from_corpus_records(
        corpus_manifest=active_runtime.corpus_manifest,
        corpus_records=active_runtime.corpus_records,
    )
    provenance = build_runtime_contriever_provenance(
        active_runtime,
        retriever,
        transformers_version=TRANSFORMERS_VERSION,
    )
    assert provenance.retriever_name == "contriever"
    assert provenance.index_fingerprint_sha256 == (
        retriever.cache_identity.fingerprint_sha256
    )
    assert provenance.index_artifact_sha256 == INDEX_SHA

    monkeypatch.setattr(
        runner_module,
        "validate_contriever_index_binding",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("binding invoked")),
    )
    with pytest.raises(ValueError, match="binding invoked"):
        build_runtime_contriever_provenance(
            active_runtime,
            retriever,
            transformers_version=TRANSFORMERS_VERSION,
        )


def test_local_index_identity_rederived_from_metadata_without_model(monkeypatch, tmp_path):
    active_runtime = runtime()
    identity = build_contriever_cache_identity(
        corpus_manifest=active_runtime.corpus_manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    embeddings = tmp_path / "embeddings"
    indexes = tmp_path / "indexes"
    embeddings.mkdir()
    indexes.mkdir()
    embedding_path = embeddings / identity.embedding_cache_filename
    index_path = indexes / identity.faiss_cache_filename
    embedding_path.write_bytes(b"fixture embeddings")
    index_path.write_bytes(b"fixture index")
    embedding_sha = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    metadata = {
        "cache_fingerprint_sha256": identity.fingerprint_sha256,
        "cache_identity_schema_version": identity.schema_version,
        "cache_schema_version": runner_module.CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION,
        "corpus_manifest_id": active_runtime.corpus_manifest.corpus_manifest_id,
        "corpus_manifest_sha256": active_runtime.corpus_manifest.sha256,
        "contriever_scientific_json": CONTRIEVER_CONFIG.scientific_json(),
        "scientific_payload": identity.scientific_payload(),
        "embedding": {
            "document_count": active_runtime.corpus_manifest.document_count,
            "dtype": CONTRIEVER_CONFIG.embedding_dtype,
            "embedding_dimension": CONTRIEVER_CONFIG.embedding_dimension,
            "filename": embedding_path.name,
            "sha256": embedding_sha,
            "shape": [
                active_runtime.corpus_manifest.document_count,
                CONTRIEVER_CONFIG.embedding_dimension,
            ],
        },
        "faiss": {
            "dimension": CONTRIEVER_CONFIG.embedding_dimension,
            "filename": index_path.name,
            "index_type": CONTRIEVER_CONFIG.index_type,
            "ntotal": active_runtime.corpus_manifest.document_count,
            "sha256": index_sha,
        },
        "environment": {"device": "cpu"},
    }
    metadata_path = indexes / f"contriever_cache_{identity.fingerprint_sha256}.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(runner_module, "EMBEDDINGS_DIR", str(embeddings))
    monkeypatch.setattr(runner_module, "INDEX_DIR", str(indexes))

    resolved = validate_local_contriever_index_identity(
        active_runtime, transformers_version=TRANSFORMERS_VERSION
    )
    assert resolved.cache_fingerprint_sha256 == identity.fingerprint_sha256
    assert resolved.index_artifact_sha256 == index_sha
    assert resolved.retriever_provenance.model_id == "facebook/contriever"
    assert resolved.retriever_provenance.model_revision == (
        "2bd46a25019aeea091fd42d1f0fd4801675cf699"
    )

    index_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="physical SHA-256 mismatch"):
        validate_local_contriever_index_identity(
            active_runtime, transformers_version=TRANSFORMERS_VERSION
        )


@pytest.mark.parametrize("count", [19, 21])
@pytest.mark.parametrize("dry_run", [False, True])
def test_noncanonical_retrieved_pool_fails_without_write(tmp_path, count, dry_run):
    summary = run(
        tmp_path,
        FakeContriever(result_count=count),
        max_samples=1,
        dry_run=dry_run,
    )
    assert summary.completed_sample_ids == ()
    assert summary.failed_samples == 1
    assert summary.failures[0].error_type == ContrieverCandidatePoolSizeError.__name__
    assert summary.failures[0].message == (
        f"expected candidate count = 20; actual candidate count = {count}"
    )
    assert not (tmp_path / "sample_0000.json").exists()


def test_noncanonical_producer_output_fails_second_boundary(tmp_path):
    def short_producer(**kwargs):
        artifact = produce_contriever_candidate_artifact(**kwargs)
        return replace(artifact, candidates=artifact.candidates[:-1])

    summary = run(tmp_path, max_samples=1, producer=short_producer)
    assert summary.failed_samples == 1
    assert summary.failures[0].error_type == ContrieverCandidatePoolSizeError.__name__
    assert "producer returned" in summary.failures[0].message
    assert not (tmp_path / "sample_0000.json").exists()


def test_dry_run_indexes_retrieves_validates_but_writes_no_candidate(tmp_path):
    retriever = FakeContriever()
    calls = []

    def producer_spy(**kwargs):
        calls.append(kwargs)
        return produce_contriever_candidate_artifact(**kwargs)

    summary = run(
        tmp_path,
        retriever,
        max_samples=1,
        dry_run=True,
        producer=producer_spy,
    )
    assert summary.completed_samples == 1
    assert len(retriever.index_calls) == len(retriever.retrieve_calls) == len(calls) == 1
    assert not (tmp_path / "sample_0000.json").exists()


def test_normal_failure_records_and_continues_but_conflict_propagates(tmp_path):
    summary = run(tmp_path, FakeContriever(fail_query="Exact Query A?"))
    assert summary.failed_samples == 1
    assert summary.failures[0].sample_id == 5
    assert summary.failures[0].error_type == "RuntimeError"
    assert summary.completed_sample_ids == (2,)
    assert not (tmp_path / "sample_0000.json").exists()

    def conflicting_producer(**kwargs):
        raise CandidateArtifactConflictError("synthetic conflict")

    with pytest.raises(CandidateArtifactConflictError, match="synthetic conflict"):
        run(tmp_path / "conflict", max_samples=1, producer=conflicting_producer)


def test_index_failure_aborts_run_before_retrieval_or_artifact_write(tmp_path):
    retriever = FakeContriever(fail_index=True)

    with pytest.raises(RuntimeError, match="synthetic index failure"):
        run(tmp_path, retriever)

    assert len(retriever.index_calls) == 1
    assert retriever.retrieve_calls == []
    assert list(tmp_path.glob("sample_*.json")) == []


def _write_payload(path, artifact):
    path.write_text(json.dumps(candidate_artifact_payload(artifact)), encoding="utf-8")


def test_matching_existing_artifact_skips_safely(tmp_path):
    run(tmp_path, max_samples=1)
    retriever = FakeContriever()
    summary = run(tmp_path, retriever, max_samples=1)
    assert summary.skipped_sample_ids == (5,)
    assert summary.skipped_existing_samples == 1
    assert retriever.retrieve_calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: replace(artifact, query_text="wrong query"),
        lambda artifact: replace(
            artifact,
            dataset=replace(artifact.dataset, revision="wrong"),
        ),
        lambda artifact: replace(
            artifact,
            corpus=replace(artifact.corpus, revision="wrong"),
        ),
        lambda artifact: replace(
            artifact,
            retriever=replace(artifact.retriever, library_version="wrong"),
        ),
        lambda artifact: replace(artifact, requested_top_n=21),
        lambda artifact: replace(artifact, candidates=artifact.candidates[:-1]),
    ],
)
def test_conflicting_existing_artifact_is_never_overwritten(tmp_path, mutation):
    run(tmp_path, max_samples=1)
    path = tmp_path / "sample_0000.json"
    _write_payload(path, mutation(read_candidate_artifact(path)))
    before = path.read_bytes()
    with pytest.raises(CandidateArtifactConflictError, match="conflicts with run"):
        run(tmp_path, max_samples=1)
    assert path.read_bytes() == before


def test_existing_scientific_artifact_is_reusable_across_production_provenance(tmp_path):
    run(tmp_path, max_samples=1)
    path = tmp_path / "sample_0000.json"
    existing = read_candidate_artifact(path)
    changed = replace(
        existing,
        producing_git_commit="9" * 40,
        worktree_clean=False,
        environment_fingerprint_sha256="9" * 64,
    )
    _write_payload(path, changed)
    before = path.read_bytes()
    summary = run(tmp_path, max_samples=1)
    assert summary.skipped_sample_ids == (5,)
    assert path.read_bytes() == before


def test_summary_counts_ids_and_physical_hashes_are_exact(tmp_path):
    summary = run(tmp_path, FakeContriever(fail_query=" Exact Query B? "))
    assert summary.manifest_sample_count == 2
    assert summary.selected_sample_count == 2
    assert summary.completed_samples == 1
    assert summary.skipped_existing_samples == 0
    assert summary.failed_samples == 1
    assert summary.candidate_pool_size == 20
    assert summary.completed_sample_ids == (5,)
    assert summary.skipped_sample_ids == ()
    assert summary.failures[0].sample_id == 2
    assert summary.embedding_artifact_sha256 == EMBEDDING_SHA
    assert summary.index_artifact_sha256 == INDEX_SHA


def test_environment_fingerprint_is_deterministic_and_version_sensitive():
    values = dict(
        transformers_version="transformers-a",
        torch_version="torch-a",
        numpy_version="numpy-a",
        faiss_version=None,
        device="cpu",
    )
    assert _environment_fingerprint(**values) == _environment_fingerprint(**values)
    assert _environment_fingerprint(**values) != _environment_fingerprint(
        **(values | {"transformers_version": "transformers-b"})
    )


def test_governed_attempt_one_records_current_executor_fingerprints(
    monkeypatch, tmp_path
):
    fixture = _governed_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "REPOSITORY_ROOT", tmp_path)

    summary = run_governed_pubmedqa_contriever_candidates(
        runtime=fixture.runtime,
        retriever=FakeContriever(),
        local_index=fixture.local_index,
        output_dir=fixture.output_dir,
        transformers_version=TRANSFORMERS_VERSION,
        environment_fingerprint_sha256=ENV_SHA,
        runtime_fingerprint_sha256=RUNTIME_SHA,
        plan=fixture.plan,
        plan_payload=fixture.plan_payload,
        planned_record=fixture.planned,
        registry_path=fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
        run_artifact_root=fixture.run_artifact_root,
        candidate_set_path=fixture.candidate_set_path,
        hardware_summary="machine-a",
        timestamp=lambda: "2026-08-24T10:00:02Z",
    )

    records = read_registry(
        fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
    )
    running = records[1]
    assert [record["execution"]["status"] for record in records] == [
        "PLANNED",
        "RUNNING",
        "COMPLETE",
    ]
    assert running["execution"]["environment_sha256"] == ENV_SHA
    assert running["execution"]["runtime_sha256"] == RUNTIME_SHA
    assert summary.run_id == fixture.planned["run_id"]


def test_governed_same_environment_resume_allows_hardware_change_and_binds_output(
    monkeypatch, tmp_path
):
    fixture = _governed_fixture(tmp_path)
    first_running = _append_first_running_attempt(fixture)
    monkeypatch.setattr(runner_module, "REPOSITORY_ROOT", tmp_path)

    summary = run_governed_pubmedqa_contriever_candidates(
        runtime=fixture.runtime,
        retriever=FakeContriever(),
        local_index=fixture.local_index,
        output_dir=fixture.output_dir,
        transformers_version=TRANSFORMERS_VERSION,
        environment_fingerprint_sha256=ENV_SHA,
        runtime_fingerprint_sha256=RUNTIME_SHA,
        plan=fixture.plan,
        plan_payload=fixture.plan_payload,
        planned_record=first_running,
        registry_path=fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
        run_artifact_root=fixture.run_artifact_root,
        candidate_set_path=fixture.candidate_set_path,
        hardware_summary="machine-b",
        resume_reason="infrastructure interruption",
        timestamp=lambda: "2026-08-24T10:00:02Z",
    )

    records = read_registry(
        fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
    )
    resumed_running = records[2]
    assert [record["execution"]["status"] for record in records] == [
        "PLANNED",
        "RUNNING",
        "RUNNING",
        "COMPLETE",
    ]
    assert {record["run_id"] for record in records} == {fixture.planned["run_id"]}
    assert resumed_running["execution"]["attempt_count"] == 2
    assert resumed_running["execution"]["hardware_summary"] == "machine-b"
    assert resumed_running["execution"]["environment_sha256"] == ENV_SHA
    assert resumed_running["execution"]["runtime_sha256"] == RUNTIME_SHA
    artifact = read_candidate_artifact(fixture.output_dir / "sample_0000.json")
    assert artifact.environment_fingerprint_sha256 == (
        resumed_running["execution"]["environment_sha256"]
    )
    assert artifact.production_fingerprint_sha256 != replace(
        artifact,
        environment_fingerprint_sha256="9" * 64,
    ).production_fingerprint_sha256
    assert summary.run_id == first_running["run_id"]
    assert summary.attempt_count == 2


def test_governed_runner_rejects_retrieval_attempt_four_before_retrieval(
    monkeypatch, tmp_path
):
    fixture = _governed_fixture(tmp_path)
    first = _append_first_running_attempt(fixture)
    second = running_candidate_record(
        first,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=2,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-b",
        prior_failure_reason="attempt 1 infrastructure interruption",
        evidence_authority_path=fixture.authority_path,
    )
    append_run_record(
        fixture.registry_path,
        second,
        evidence_authority_path=fixture.authority_path,
    )
    third = running_candidate_record(
        second,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=3,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-c",
        prior_failure_reason="attempt 2 infrastructure interruption",
        evidence_authority_path=fixture.authority_path,
    )
    append_run_record(
        fixture.registry_path,
        third,
        evidence_authority_path=fixture.authority_path,
    )
    retrieval_calls = []

    def forbidden_retrieval(**kwargs):
        retrieval_calls.append(kwargs)
        raise AssertionError("retrieval must not start after the attempt ceiling")

    monkeypatch.setattr(
        runner_module,
        "run_pubmedqa_contriever_candidates",
        forbidden_retrieval,
    )
    monkeypatch.setattr(runner_module, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(
        RunRecordValidationError,
        match="RETRIEVAL permits at most 3 total attempts",
    ):
        run_governed_pubmedqa_contriever_candidates(
            runtime=fixture.runtime,
            retriever=object(),
            local_index=fixture.local_index,
            output_dir=fixture.output_dir,
            transformers_version=TRANSFORMERS_VERSION,
            environment_fingerprint_sha256=ENV_SHA,
            runtime_fingerprint_sha256=RUNTIME_SHA,
            plan=fixture.plan,
            plan_payload=fixture.plan_payload,
            planned_record=third,
            registry_path=fixture.registry_path,
            evidence_authority_path=fixture.authority_path,
            run_artifact_root=fixture.run_artifact_root,
            candidate_set_path=fixture.candidate_set_path,
            hardware_summary="machine-d",
            resume_reason="attempt 3 infrastructure interruption",
            timestamp=lambda: "2026-08-24T10:00:02Z",
        )

    assert retrieval_calls == []
    records = read_registry(
        fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
    )
    assert [record["execution"]["attempt_count"] for record in records] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    "current_environment,current_runtime,expected_field",
    [
        ("8" * 64, RUNTIME_SHA, "environment_sha256"),
        (ENV_SHA, "9" * 64, "runtime_sha256"),
    ],
)
def test_incompatible_resume_rejected_before_retrieval_and_preserves_candidates(
    monkeypatch,
    tmp_path,
    current_environment,
    current_runtime,
    expected_field,
):
    fixture = _governed_fixture(tmp_path, preexisting=True)
    first_running = _append_first_running_attempt(fixture)
    existing_path = fixture.output_dir / "sample_0000.json"
    existing_bytes = existing_path.read_bytes()
    retrieval_calls = []

    def forbidden_retrieval(**kwargs):
        retrieval_calls.append(kwargs)
        raise AssertionError("retrieval must not start before resume validation")

    monkeypatch.setattr(
        runner_module,
        "run_pubmedqa_contriever_candidates",
        forbidden_retrieval,
    )
    monkeypatch.setattr(runner_module, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(
        RunRegistryConflictError,
        match=rf"identical execution\.{expected_field}",
    ):
        run_governed_pubmedqa_contriever_candidates(
            runtime=fixture.runtime,
            retriever=object(),
            local_index=fixture.local_index,
            output_dir=fixture.output_dir,
            transformers_version=TRANSFORMERS_VERSION,
            environment_fingerprint_sha256=current_environment,
            runtime_fingerprint_sha256=current_runtime,
            plan=fixture.plan,
            plan_payload=fixture.plan_payload,
            planned_record=first_running,
            registry_path=fixture.registry_path,
            evidence_authority_path=fixture.authority_path,
            run_artifact_root=fixture.run_artifact_root,
            candidate_set_path=fixture.candidate_set_path,
            hardware_summary="machine-b",
            resume_reason="infrastructure interruption",
            timestamp=lambda: "2026-08-24T10:00:02Z",
        )

    records = read_registry(
        fixture.registry_path,
        evidence_authority_path=fixture.authority_path,
    )
    assert [record["execution"]["status"] for record in records] == [
        "PLANNED",
        "RUNNING",
    ]
    assert retrieval_calls == []
    assert not (fixture.output_dir / "sample_0001.json").exists()
    assert existing_path.read_bytes() == existing_bytes


def test_import_and_help_are_side_effect_free(tmp_path):
    root = Path(__file__).resolve().parents[1]
    watched = (
        root / "data/embeddings",
        root / "data/indices",
        root / "artifacts/candidates/pubmedqa/contriever",
    )
    before = {path: tuple(path.glob("*")) if path.exists() else () for path in watched}
    commands = (
        (sys.executable, "-c", "import scripts.run_pubmedqa_contriever_candidates"),
        (sys.executable, "-m", "scripts.run_pubmedqa_contriever_candidates", "--help"),
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "HF_HOME": str(tmp_path / "hf")},
        )
        assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "Contriever cache/index setup" in help_text
    assert "cache artifacts may still be created or updated" in help_text
    after = {path: tuple(path.glob("*")) if path.exists() else () for path in watched}
    assert after == before
    assert not (tmp_path / "hf").exists()


def test_main_preflight_does_not_construct_retriever_or_write_registry(
    monkeypatch, capsys, tmp_path
):
    import retrievers.contriever_retriever as contriever_module

    registry = tmp_path / "registry.jsonl"
    monkeypatch.setattr(
        runner_module,
        "parse_args",
        lambda: SimpleNamespace(
            cache_dir=None,
            output_dir=Path("unused"),
            max_samples=None,
            sample_ids=None,
            dry_run=False,
            preflight_only=True,
            registry_path=registry,
            evidence_authority_path=tmp_path / "authority.json",
            run_artifact_root=tmp_path / "runs",
            candidate_set_path=tmp_path / "set.json",
            resume_run_id=None,
            resume_reason=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: ()),
    )
    monkeypatch.setattr(
        runner_module,
        "load_validated_pubmedqa_runtime_corpus",
        lambda rows: object(),
    )
    monkeypatch.setattr(
        runner_module.importlib_metadata,
        "version",
        lambda distribution: "fixture-version",
    )
    monkeypatch.setattr(runner_module, "_git_provenance", lambda: (GIT_SHA, True))
    local_index = SimpleNamespace(cache_fingerprint_sha256="8" * 64)
    monkeypatch.setattr(
        runner_module,
        "validate_local_contriever_index_identity",
        lambda *args, **kwargs: local_index,
    )
    plan = SimpleNamespace(
        inspection=SimpleNamespace(valid_entries=(object(), object(), object())),
        scheduled_sample_ids=(3, 4),
    )
    monkeypatch.setattr(
        runner_module,
        "prepare_governed_contriever_run",
        lambda **kwargs: (
            plan,
            {"scheduled_sample_ids": [3, 4]},
            {"run_id": "run-fixture-" + "a" * 24},
        ),
    )
    monkeypatch.setattr(
        contriever_module,
        "ContrieverRetriever",
        lambda config: (_ for _ in ()).throw(AssertionError("must not construct")),
    )
    assert runner_module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["scheduled_sample_ids"] == [3, 4]
    assert output["registry_written"] is False
    assert not registry.exists()
