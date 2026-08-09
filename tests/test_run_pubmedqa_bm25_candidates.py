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
from retrievers.bm25_config import BM25_CONFIG
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    CandidateArtifactConflictError,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    SampleManifest,
    SampleManifestEntry,
    candidate_artifact_payload,
    corpus_provenance_from_corpus_manifest,
    dataset_provenance_from_sample_manifest,
    document_content_sha256,
    produce_bm25_candidate_artifact,
    query_text_sha256,
    read_candidate_artifact,
)
from scripts.build_corpus_manifests import (
    ValidatedPubMedQAQuery,
    ValidatedPubMedQARuntimeCorpus,
)
from scripts.run_pubmedqa_bm25_candidates import (
    CANDIDATE_POOL_SIZE,
    CandidatePoolSizeError,
    main,
    run_pubmedqa_bm25_candidates,
)
import scripts.run_pubmedqa_bm25_candidates as runner_module


GIT_SHA = "1" * 40
ENV_SHA = "2" * 64


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
    leading = (
        CorpusRecord(10, "source-10", None, "stored wrong", "retrieval alpha", 0),
        CorpusRecord(3, "source-3", None, "stored beta", "retrieval beta", 1),
        CorpusRecord("x", "source-x", None, "stored gamma", "retrieval gamma", 2),
    )
    return leading + tuple(
        CorpusRecord(
            document_id=position + 100,
            source_document_id=f"source-{position + 100}",
            title=None,
            text=f"stored filler {position}",
            retrieval_content=f"retrieval filler {position}",
            corpus_position=position,
        )
        for position in range(3, CANDIDATE_POOL_SIZE)
    )


def manifest(sample, corpus_records):
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus-source",
        config="fixture-corpus",
        revision="fixture-corpus-revision",
        split="fixture-corpus-split",
        construction_algorithm="fixture-corpus.v1",
        input_sample_manifest_id=sample.manifest_id,
        input_sample_manifest_sha256=sample.sha256,
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
    sample = sample_manifest()
    corpus_records = records()
    corpus_manifest = manifest(sample, corpus_records)
    dataset_provenance = dataset_provenance_from_sample_manifest(sample)
    corpus_provenance = corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset_provenance,
    )
    return ValidatedPubMedQARuntimeCorpus(
        sample_manifest=sample,
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset_provenance,
        corpus_provenance=corpus_provenance,
        ordered_queries=(
            ValidatedPubMedQAQuery(0, 5, "Exact Query A?"),
            ValidatedPubMedQAQuery(1, 2, " Exact Query B? "),
        ),
    )


class FakeBM25:
    runtime_config = BM25_CONFIG

    def __init__(self, *, fail_query=None, result_count=CANDIDATE_POOL_SIZE):
        self.index_calls = 0
        self.retrieve_calls = []
        self.indexed_documents = None
        self.fail_query = fail_query
        self.result_count = result_count

    def index(self, documents):
        self.index_calls += 1
        self.indexed_documents = documents

    def retrieve(self, query, top_k=5):
        self.retrieve_calls.append((query, top_k))
        if query == self.fail_query:
            return []
        ordered = [
            (self.indexed_documents[2], 0.125),
            (self.indexed_documents[0], 7.75),
        ] + [
            (document, float(CANDIDATE_POOL_SIZE - position))
            for position, document in enumerate(
                [self.indexed_documents[1], *self.indexed_documents[3:]], start=2
            )
        ]
        return ordered[: self.result_count]


def run(tmp_path, retriever=None, **changes):
    values = dict(
        runtime=runtime(),
        retriever=FakeBM25() if retriever is None else retriever,
        output_dir=tmp_path,
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
        library_version="0.2.2",
    )
    values.update(changes)
    return run_pubmedqa_bm25_candidates(**values)


def test_manifest_order_pool_size_index_once_exact_queries_and_raw_order(tmp_path):
    retriever = FakeBM25()
    calls = []

    def producer_spy(**kwargs):
        calls.append(kwargs)
        return produce_bm25_candidate_artifact(**kwargs)

    summary = run(tmp_path, retriever, producer=producer_spy)
    assert summary.completed_sample_ids == (5, 2)
    assert summary.candidate_pool_size == CANDIDATE_POOL_SIZE == 20
    assert retriever.index_calls == 1
    assert retriever.retrieve_calls == [
        ("Exact Query A?", 20),
        (" Exact Query B? ", 20),
    ]
    assert [call["sample_id"] for call in calls] == [5, 2]
    assert [call["query_text"] for call in calls] == [
        "Exact Query A?", " Exact Query B? "
    ]
    assert [result.document_id for result in calls[0]["raw_results"]][:2] == ["x", 10]
    assert [result.native_score for result in calls[0]["raw_results"]][:2] == [
        0.125, 7.75
    ]
    assert len(calls[0]["raw_results"]) == 20
    assert retriever.indexed_documents[0] == {"doc_id": 10, "text": "retrieval alpha"}
    assert "stored wrong" not in str(retriever.indexed_documents)


def test_same_science_is_deterministic_and_round_trips(tmp_path):
    first = run(tmp_path)
    artifacts = tuple(read_candidate_artifact(path) for path in sorted(tmp_path.iterdir()))
    assert first.completed_samples == 2
    assert artifacts[0].artifact_id != artifacts[1].artifact_id
    ids = tuple(artifact.artifact_id for artifact in artifacts)
    second_dir = tmp_path / "second"
    run(second_dir)
    reproduced = tuple(
        read_candidate_artifact(path) for path in sorted(second_dir.iterdir())
    )
    assert tuple(artifact.artifact_id for artifact in reproduced) == ids


def test_resume_skips_matching_artifacts_without_per_query_retrieval(tmp_path):
    run(tmp_path)
    retriever = FakeBM25()
    summary = run(tmp_path, retriever)
    assert summary.completed_samples == 0
    assert summary.skipped_existing_samples == 2
    assert summary.skipped_sample_ids == (5, 2)
    assert retriever.index_calls == 1
    assert retriever.retrieve_calls == []


def test_conflicting_existing_artifact_is_rejected(tmp_path):
    run(tmp_path, max_samples=1)
    path = tmp_path / "sample_0000.json"
    stored = read_candidate_artifact(path)
    conflicting = replace(stored, query_text="different")
    path.write_text(
        json.dumps(candidate_artifact_payload(conflicting)), encoding="utf-8"
    )
    with pytest.raises(CandidateArtifactConflictError, match="conflicts with run"):
        run(tmp_path, max_samples=1)


def test_failure_is_explicit_and_not_a_fake_artifact(tmp_path):
    retriever = FakeBM25(fail_query="Exact Query A?")
    summary = run(tmp_path, retriever)
    assert summary.failed_samples == 1
    assert summary.failures[0].sample_id == 5
    assert summary.failures[0].error_type == "CandidatePoolSizeError"
    assert summary.failures[0].message == (
        "expected candidate count = 20; actual candidate count = 0"
    )
    assert not (tmp_path / "sample_0000.json").exists()
    assert (tmp_path / "sample_0001.json").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_short_candidate_pool_is_failed_and_never_completed_or_written(
    tmp_path, dry_run
):
    summary = run(
        tmp_path,
        FakeBM25(result_count=19),
        max_samples=1,
        dry_run=dry_run,
    )
    assert summary.completed_samples == 0
    assert summary.completed_sample_ids == ()
    assert summary.failed_samples == 1
    assert summary.failures[0].sample_id == 5
    assert summary.failures[0].error_type == CandidatePoolSizeError.__name__
    assert summary.failures[0].message == (
        "expected candidate count = 20; actual candidate count = 19"
    )
    assert not (tmp_path / "sample_0000.json").exists()


def test_exactly_twenty_candidates_complete(tmp_path):
    summary = run(tmp_path, FakeBM25(result_count=20), max_samples=1)
    assert summary.completed_sample_ids == (5,)
    assert summary.failed_samples == 0
    assert len(read_candidate_artifact(tmp_path / "sample_0000.json").candidates) == 20


def test_max_samples_is_execution_only_and_dry_run_writes_nothing(tmp_path):
    original_manifest_id = runtime().sample_manifest.manifest_id
    summary = run(tmp_path, max_samples=1, dry_run=True)
    assert summary.manifest_sample_count == 2
    assert summary.selected_sample_count == 1
    assert summary.completed_sample_ids == (5,)
    assert summary.sample_manifest_id == original_manifest_id
    assert list(tmp_path.iterdir()) == []


def test_import_and_help_have_no_runtime_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    commands = (
        (sys.executable, "-c", "import scripts.run_pubmedqa_bm25_candidates"),
        (sys.executable, "-m", "scripts.run_pubmedqa_bm25_candidates", "--help"),
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
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_samples, expected", [(0, 0), (1, 1)])
def test_main_exit_code_reflects_recorded_failures(
    monkeypatch, capsys, failed_samples, expected
):
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
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: ())
    )
    monkeypatch.setattr(
        runner_module, "load_validated_pubmedqa_runtime_corpus", lambda rows: object()
    )
    monkeypatch.setattr(runner_module.importlib_metadata, "version", lambda name: "0.2.2")
    monkeypatch.setattr(runner_module, "_git_provenance", lambda: (GIT_SHA, True))
    monkeypatch.setattr(runner_module, "BM25Retriever", lambda config: object())
    monkeypatch.setattr(
        runner_module,
        "run_pubmedqa_bm25_candidates",
        lambda **kwargs: SimpleNamespace(failed_samples=failed_samples),
    )
    assert main() == expected
    capsys.readouterr()


def test_help_explains_dry_run_still_indexes_and_retrieves():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        (sys.executable, "-m", "scripts.run_pubmedqa_bm25_candidates", "--help"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "dataset loading" in help_text
    assert "BM25 indexing and retrieval" in help_text
    assert "do not write CandidateArtifact files" in help_text
