from dataclasses import replace
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
    build_dpr_cache_identity,
    candidate_artifact_payload,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    produce_dpr_candidate_artifact,
    query_text_sha256,
    read_candidate_artifact,
)
from retrievers.dpr_config import DPR_CONFIG
from scripts.build_corpus_manifests import (
    ValidatedPubMedQAQuery,
    ValidatedPubMedQARuntimeCorpus,
)
import scripts.run_pubmedqa_dpr_candidates as runner_module
from scripts.run_pubmedqa_dpr_candidates import (
    CANDIDATE_POOL_SIZE,
    DPRCandidatePoolSizeError,
    _environment_fingerprint,
    run_pubmedqa_dpr_candidates,
)


GIT_SHA = "1" * 40
ENV_SHA = "2" * 64
INDEX_SHA = "3" * 64
EMBEDDING_SHA = "4" * 64
TRANSFORMERS_VERSION = "fixture-transformers-version"


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
    query_texts = ("Exact Query A?", " Exact Query B? ")
    return ValidatedPubMedQARuntimeCorpus(
        sample_manifest=samples,
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset,
        corpus_provenance=corpus,
        ordered_queries=tuple(
            ValidatedPubMedQAQuery(entry.position, entry.sample_id, text)
            for entry, text in zip(samples.entries, query_texts, strict=True)
        ),
    )


class FakeDPR:
    def __init__(
        self,
        *,
        config=DPR_CONFIG,
        result_count=CANDIDATE_POOL_SIZE,
        fail_query=None,
        fail_index=False,
        complete_state=True,
    ):
        self.config = config
        self.result_count = result_count
        self.fail_query = fail_query
        self.fail_index = fail_index
        self.complete_state = complete_state
        self.index_calls = []
        self.retrieve_calls = []
        self.generic_index_calls = 0
        self.is_indexed = False
        self.cache_identity = None
        self.index_artifact_sha256 = None
        self.embedding_artifact_sha256 = None
        self.indexed_records = None

    def index(self, documents):
        self.generic_index_calls += 1
        raise AssertionError("generic DPR index() must not be used")

    def index_from_corpus_records(self, *, corpus_manifest, corpus_records):
        self.index_calls.append((corpus_manifest, corpus_records))
        if self.fail_index:
            raise RuntimeError("synthetic index failure")
        self.indexed_records = corpus_records
        self.cache_identity = build_dpr_cache_identity(
            corpus_manifest=corpus_manifest,
            dpr_config=self.config,
        )
        self.index_artifact_sha256 = INDEX_SHA
        self.embedding_artifact_sha256 = EMBEDDING_SHA
        self.is_indexed = True
        if not self.complete_state:
            self.embedding_artifact_sha256 = None

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
        retriever=FakeDPR() if retriever is None else retriever,
        output_dir=output_dir,
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
        transformers_version=TRANSFORMERS_VERSION,
    )
    values.update(changes)
    return run_pubmedqa_dpr_candidates(**values)


def test_index_once_full_corpus_manifest_order_and_producer_inputs(tmp_path):
    retriever = FakeDPR()
    calls = []

    def producer_spy(**kwargs):
        calls.append(kwargs)
        return produce_dpr_candidate_artifact(**kwargs)

    summary = run(tmp_path, retriever, max_samples=1, producer=producer_spy)
    expected_runtime = runtime()
    assert summary.selected_sample_count == 1
    assert summary.completed_sample_ids == (5,)
    assert len(retriever.index_calls) == 1
    assert retriever.generic_index_calls == 0
    assert retriever.index_calls[0][0] == expected_runtime.corpus_manifest
    assert retriever.index_calls[0][1] == expected_runtime.corpus_records
    assert len(retriever.index_calls[0][1]) == CANDIDATE_POOL_SIZE
    assert retriever.retrieve_calls == [("Exact Query A?", 20)]
    assert calls[0]["cache_identity"] == retriever.cache_identity
    assert calls[0]["retriever_provenance"].index_artifact_sha256 == INDEX_SHA
    assert calls[0]["requested_top_n"] == 20
    assert [result.document_id for result in calls[0]["raw_results"]] == [
        record.document_id for record in reversed(expected_runtime.corpus_records)
    ]
    assert [result.native_score for result in calls[0]["raw_results"]][:2] == [
        -2.5, 0.0
    ]
    artifact = read_candidate_artifact(tmp_path / "sample_0000.json")
    assert len(artifact.candidates) == 20
    assert [candidate.native_score for candidate in artifact.candidates][:2] == [
        -2.5, 0.0
    ]
    assert summary.index_artifact_sha256 == INDEX_SHA
    assert summary.embedding_artifact_sha256 == EMBEDDING_SHA


def test_all_queries_follow_frozen_order_and_index_is_not_repeated(tmp_path):
    retriever = FakeDPR()
    summary = run(tmp_path, retriever)
    assert summary.completed_sample_ids == (5, 2)
    assert retriever.retrieve_calls == [
        ("Exact Query A?", 20),
        (" Exact Query B? ", 20),
    ]
    assert len(retriever.index_calls) == 1


@pytest.mark.parametrize("count", [19, 21])
@pytest.mark.parametrize("dry_run", [False, True])
def test_noncanonical_pool_fails_without_padding_truncation_or_write(
    tmp_path, count, dry_run
):
    summary = run(
        tmp_path,
        FakeDPR(result_count=count),
        max_samples=1,
        dry_run=dry_run,
    )
    assert summary.completed_sample_ids == ()
    assert summary.failed_samples == 1
    assert summary.failures[0].error_type == DPRCandidatePoolSizeError.__name__
    assert summary.failures[0].message == (
        f"expected candidate count = 20; actual candidate count = {count}"
    )
    assert not (tmp_path / "sample_0000.json").exists()


def test_dry_run_indexes_retrieves_and_validates_without_candidate_write(tmp_path):
    retriever = FakeDPR()
    calls = []

    def producer_spy(**kwargs):
        calls.append(kwargs)
        return produce_dpr_candidate_artifact(**kwargs)

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


def test_changed_config_and_incomplete_or_failed_index_state_propagate(tmp_path):
    changed = FakeDPR(config=replace(DPR_CONFIG, context_max_length=128))
    with pytest.raises(ValueError, match="frozen DPR_CONFIG"):
        run(tmp_path, changed)
    assert changed.index_calls == []

    incomplete = FakeDPR(complete_state=False)
    with pytest.raises(ValueError, match="embedding_artifact_sha256"):
        run(tmp_path, incomplete)
    assert incomplete.retrieve_calls == []

    failed = FakeDPR(fail_index=True)
    with pytest.raises(RuntimeError, match="synthetic index failure"):
        run(tmp_path, failed)
    assert failed.retrieve_calls == []


def test_matching_resume_skips_and_short_or_conflicting_artifact_is_rejected(tmp_path):
    run(tmp_path, max_samples=1)
    retriever = FakeDPR()
    summary = run(tmp_path, retriever, max_samples=1)
    assert summary.skipped_sample_ids == (5,)
    assert retriever.retrieve_calls == []

    path = tmp_path / "sample_0000.json"
    stored = read_candidate_artifact(path)
    shortened = replace(stored, candidates=stored.candidates[:-1])
    path.write_text(
        json.dumps(candidate_artifact_payload(shortened)),
        encoding="utf-8",
    )
    with pytest.raises(CandidateArtifactConflictError, match="conflicts with run"):
        run(tmp_path, max_samples=1)

    conflicting = replace(shortened, query_text="different exact query")
    path.write_text(
        json.dumps(candidate_artifact_payload(conflicting)),
        encoding="utf-8",
    )
    with pytest.raises(CandidateArtifactConflictError):
        run(tmp_path, max_samples=1)


def test_ordinary_retrieval_failure_is_recorded_and_conflict_from_producer_propagates(
    tmp_path,
):
    summary = run(tmp_path, FakeDPR(fail_query="Exact Query A?"))
    assert summary.failed_samples == 1
    assert summary.failures[0].sample_id == 5
    assert summary.failures[0].error_type == "RuntimeError"
    assert summary.completed_sample_ids == (2,)
    assert not (tmp_path / "sample_0000.json").exists()

    def conflicting_producer(**kwargs):
        raise CandidateArtifactConflictError("synthetic conflict")

    with pytest.raises(CandidateArtifactConflictError, match="synthetic conflict"):
        run(tmp_path / "conflict", max_samples=1, producer=conflicting_producer)


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


def test_import_is_side_effect_free_and_creates_no_real_runtime_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    watched = (
        root / "data/embeddings",
        root / "data/indices",
        root / "artifacts/candidates/pubmedqa/dpr",
    )
    before = {path: tuple(path.glob("*")) if path.exists() else () for path in watched}
    commands = (
        (sys.executable, "-c", "import scripts.run_pubmedqa_dpr_candidates"),
        (sys.executable, "-m", "scripts.run_pubmedqa_dpr_candidates", "--help"),
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
    assert "DPR cache/index setup" in help_text
    assert "cache artifacts may still be created or updated" in help_text
    after = {path: tuple(path.glob("*")) if path.exists() else () for path in watched}
    assert after == before
    assert not (tmp_path / "hf").exists()


@pytest.mark.parametrize("failed_samples,expected", [(0, 0), (1, 1)])
def test_main_exit_code_reflects_query_failures(
    monkeypatch, capsys, failed_samples, expected
):
    import retrievers.dpr_original_retriever as dpr_module

    monkeypatch.setattr(
        runner_module,
        "parse_args",
        lambda: SimpleNamespace(
            cache_dir=None,
            output_dir=Path("unused"),
            max_samples=1,
            dry_run=True,
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
    monkeypatch.setattr(
        dpr_module,
        "OriginalDPRRetriever",
        lambda config: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        runner_module,
        "run_pubmedqa_dpr_candidates",
        lambda **kwargs: SimpleNamespace(failed_samples=failed_samples),
    )
    assert runner_module.main() == expected
    capsys.readouterr()
