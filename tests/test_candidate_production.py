from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    SAMPLE_MANIFEST_SCHEMA_VERSION,
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
    write_candidate_artifact,
)
from retrieval_artifacts.candidate_production import (
    CandidateProductionValidationError,
    build_retrieval_planned_record,
    candidate_production_output_inventory,
    candidate_production_plan_scientific_payload,
    inspect_candidate_directory,
    materialize_candidate_set_inventory,
    plan_candidate_production,
    running_candidate_record,
    select_manifest_queries,
    terminal_candidate_record,
    write_candidate_production_output,
)
from retrievers.contriever_config import CONTRIEVER_CONFIG
from run_registry import (
    EVIDENCE_AUTHORITY_SCHEMA_VERSION,
    REGISTRY_HEADER,
    append_run_record,
    candidate_set_scientific_payload,
    canonical_json,
    read_registry,
)
from scripts.build_corpus_manifests import (
    ValidatedPubMedQAQuery,
    ValidatedPubMedQARuntimeCorpus,
)


GIT_SHA = "1" * 40
ENV_SHA = "2" * 64
INDEX_SHA = "3" * 64
CONFIG_SHA = hashlib.sha256(
    CONTRIEVER_CONFIG.scientific_json().encode("utf-8")
).hexdigest()
RUNTIME_SHA = "7" * 64


def _runtime(sample_count=3):
    sample_ids = tuple(range(sample_count))
    questions = tuple(f"question {sample_id}" for sample_id in sample_ids)
    samples = SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="fixture-source",
        config="fixture-config",
        revision="fixture-revision",
        split="train",
        sampling_algorithm="fixture-order.v1",
        sampling_seed=None,
        requested_sample_size=sample_count,
        selection_dependencies=(),
        entries=tuple(
            SampleManifestEntry(
                position=position,
                sample_id=sample_id,
                source_sample_id=sample_id,
                query_text_sha256=query_text_sha256(question),
            )
            for position, (sample_id, question) in enumerate(
                zip(sample_ids, questions, strict=True)
            )
        ),
    )
    records = tuple(
        CorpusRecord(
            document_id=position,
            source_document_id=f"source-{position}",
            title=None,
            text=f"text {position}",
            retrieval_content=f"text {position}",
            corpus_position=position,
        )
        for position in range(20)
    )
    corpus_manifest = CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="fixture-source",
        config="fixture-config",
        revision="fixture-revision",
        split="train",
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
            for record in records
        ),
    )
    dataset = dataset_provenance_from_sample_manifest(samples)
    corpus = corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest,
        corpus_records=records,
        dataset_provenance=dataset,
    )
    runtime = ValidatedPubMedQARuntimeCorpus(
        sample_manifest=samples,
        corpus_manifest=corpus_manifest,
        corpus_records=records,
        dataset_provenance=dataset,
        corpus_provenance=corpus,
        ordered_queries=tuple(
            ValidatedPubMedQAQuery(position, sample_id, question)
            for position, (sample_id, question) in enumerate(
                zip(sample_ids, questions, strict=True)
            )
        ),
    )
    identity = build_contriever_cache_identity(
        corpus_manifest=corpus_manifest,
        contriever_config=CONTRIEVER_CONFIG,
    )
    provenance = build_contriever_retriever_provenance(
        cache_identity=identity,
        index_artifact_sha256=INDEX_SHA,
        transformers_version="fixture-transformers",
    )
    return runtime, identity, provenance


def _write_candidate(runtime, identity, provenance, directory, position, *, git=GIT_SHA):
    query = runtime.ordered_queries[position]
    artifact = produce_contriever_candidate_artifact(
        sample_manifest=runtime.sample_manifest,
        dataset_provenance=runtime.dataset_provenance,
        corpus_manifest=runtime.corpus_manifest,
        corpus_provenance=runtime.corpus_provenance,
        cache_identity=identity,
        sample_id=query.sample_id,
        query_text=query.query_text,
        retriever_provenance=provenance,
        requested_top_n=20,
        raw_results=tuple(
            RawCandidateResult(document_id=position, native_score=20.0 - position)
            for position in range(20)
        ),
        corpus_records=runtime.corpus_records,
        producing_git_commit=git,
        worktree_clean=True,
        environment_fingerprint_sha256=ENV_SHA,
    )
    path = directory / f"sample_{position:04d}.json"
    write_candidate_artifact(artifact, path)
    return path, artifact


def _plan(runtime, provenance, directory, **changes):
    values = dict(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=directory,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
    )
    values.update(changes)
    return plan_candidate_production(**values)


def test_complete_and_partial_directories_produce_exact_missing_plan(tmp_path):
    runtime, identity, provenance = _runtime()
    for position in range(3):
        _write_candidate(runtime, identity, provenance, tmp_path, position)
    complete = _plan(runtime, provenance, tmp_path)
    assert complete.scheduled_sample_ids == ()
    assert complete.skipped_valid_sample_ids == (0, 1, 2)

    (tmp_path / "sample_0001.json").unlink()
    partial = _plan(runtime, provenance, tmp_path)
    assert partial.scheduled_sample_ids == (1,)
    assert partial.skipped_valid_sample_ids == (0, 2)


def test_explicit_ids_are_validated_and_normalized_to_manifest_order(tmp_path):
    runtime, _, provenance = _runtime()
    assert tuple(
        query.sample_id
        for query in select_manifest_queries(
            runtime.ordered_queries, requested_sample_ids=(2, 0)
        )
    ) == (0, 2)
    plan = _plan(
        runtime, provenance, tmp_path, requested_sample_ids=(2, 0)
    )
    assert plan.requested_sample_ids == (0, 2)
    assert plan.scheduled_sample_ids == (0, 2)
    with pytest.raises(ValueError, match="duplicates"):
        _plan(runtime, provenance, tmp_path, requested_sample_ids=(0, 0))
    with pytest.raises(ValueError, match="absent"):
        _plan(runtime, provenance, tmp_path, requested_sample_ids=(99,))


def test_valid_existing_is_skipped_across_production_commits_and_never_rewritten(tmp_path):
    runtime, identity, provenance = _runtime()
    path, _ = _write_candidate(
        runtime, identity, provenance, tmp_path, 0, git="9" * 40
    )
    before = path.read_bytes()
    plan = _plan(runtime, provenance, tmp_path, requested_sample_ids=(0, 1))
    assert plan.skipped_valid_sample_ids == (0,)
    assert plan.scheduled_sample_ids == (1,)
    assert path.read_bytes() == before


def test_invalid_existing_and_extra_artifacts_fail_closed(tmp_path):
    runtime, identity, provenance = _runtime()
    path, artifact = _write_candidate(runtime, identity, provenance, tmp_path, 0)
    changed = replace(artifact, query_text="wrong")
    path.unlink()
    write_candidate_artifact(changed, path)
    before = path.read_bytes()
    with pytest.raises(CandidateProductionValidationError, match="invalid_sample_ids"):
        _plan(runtime, provenance, tmp_path)
    assert path.read_bytes() == before

    path.unlink()
    (tmp_path / "sample_9999.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CandidateProductionValidationError, match="extra_artifacts"):
        _plan(runtime, provenance, tmp_path)


def test_audited_shape_three_of_one_thousand_plans_exactly_997(tmp_path):
    runtime, identity, provenance = _runtime(sample_count=1000)
    for position in range(3):
        _write_candidate(runtime, identity, provenance, tmp_path, position)
    plan = _plan(runtime, provenance, tmp_path)
    assert plan.skipped_valid_sample_ids == (0, 1, 2)
    assert plan.scheduled_sample_ids == tuple(range(3, 1000))
    assert len(plan.scheduled_sample_ids) == 997


def _authority(path, manifest_id):
    path.write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
                "authorities": [
                    {
                        "dataset": "pubmedqa",
                        "evidence_role": "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
                        "sample_manifest_id": manifest_id,
                        "sample_manifest_path": "fixture/sample.json",
                        "authority_protocols": ["fixture/protocol.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _planned_record(tmp_path, runtime, authority_path, scheduled=(1, 2)):
    plan_payload = candidate_production_plan_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        corpus_manifest_id=runtime.corpus_manifest.corpus_manifest_id,
        retriever="contriever",
        retriever_config_sha256=CONFIG_SHA,
        index_artifact_id=f"index:sha256:{'4' * 64}",
        candidate_pool=20,
        top_k=5,
        candidate_directory="artifacts/candidates/pubmedqa/contriever",
        scheduled_sample_ids=scheduled,
    )
    ref = lambda path, sha, artifact_id: {
        "path": path,
        "sha256": sha,
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
            "split": "train",
            "source": "fixture-source",
            "revision": "fixture-revision",
            "sample_manifest": ref(
                "fixture/sample.json",
                "5" * 64,
                runtime.sample_manifest.manifest_id,
            ),
            "corpus_manifest": ref(
                "fixture/corpus.json",
                "6" * 64,
                runtime.corpus_manifest.corpus_manifest_id,
            ),
        },
        retrieval_index=ref(
            "fixture/index.faiss",
            INDEX_SHA,
            f"index:sha256:{'4' * 64}",
        ),
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="fixture-cpu",
        output_directory="artifacts/candidates/pubmedqa/contriever",
        evidence_authority_path=authority_path,
    )
    return plan_payload, planned


def test_planned_registry_record_and_retry_keep_same_run_id(tmp_path):
    runtime, _, _ = _runtime()
    authority_path = tmp_path / "authority.json"
    _authority(authority_path, runtime.sample_manifest.manifest_id)
    _, planned = _planned_record(tmp_path, runtime, authority_path)
    registry = tmp_path / "registry.jsonl"
    registry.write_text(canonical_json(REGISTRY_HEADER) + "\n", encoding="utf-8")
    append_run_record(registry, planned, evidence_authority_path=authority_path)
    running1 = running_candidate_record(
        planned,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=1,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-a",
        evidence_authority_path=authority_path,
    )
    append_run_record(registry, running1, evidence_authority_path=authority_path)
    running2 = running_candidate_record(
        running1,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=2,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="machine-b",
        prior_failure_reason="infrastructure interruption",
        evidence_authority_path=authority_path,
    )
    append_run_record(registry, running2, evidence_authority_path=authority_path)
    records = read_registry(registry, evidence_authority_path=authority_path)
    assert [record["execution"]["status"] for record in records] == [
        "PLANNED",
        "RUNNING",
        "RUNNING",
    ]
    assert {record["run_id"] for record in records} == {planned["run_id"]}


def test_resume_revalidates_and_skips_partial_valid_output(tmp_path):
    runtime, identity, provenance = _runtime()
    initial = _plan(runtime, provenance, tmp_path, requested_sample_ids=(1, 2))
    assert initial.scheduled_sample_ids == (1, 2)
    path, _ = _write_candidate(runtime, identity, provenance, tmp_path, 1)
    before = path.read_bytes()
    resumed = _plan(runtime, provenance, tmp_path, requested_sample_ids=(1, 2))
    assert resumed.skipped_valid_sample_ids == (1,)
    assert resumed.scheduled_sample_ids == (2,)
    assert path.read_bytes() == before


def test_output_inventory_never_relabels_preexisting_as_new(tmp_path):
    runtime, identity, provenance = _runtime()
    _, _ = _write_candidate(runtime, identity, provenance, tmp_path, 0)
    initial = _plan(runtime, provenance, tmp_path)
    _write_candidate(runtime, identity, provenance, tmp_path, 1)
    current = inspect_candidate_directory(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=tmp_path,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
    )
    inventory = candidate_production_output_inventory(
        run_id="run-sprint1-pubmedqa-s1-na-contriever-none-na-" + "a" * 24,
        producer_identity="fixture-producer",
        retriever="contriever",
        repository_root=tmp_path,
        scheduled_sample_ids=initial.scheduled_sample_ids,
        initial_preexisting_entries=initial.inspection.valid_entries,
        current_inspection=current,
    )
    assert inventory["reused_preexisting_valid_count"] == 1
    assert [entry["sample_id"] for entry in inventory["output_artifacts"]] == [1]
    assert [
        entry["sample_id"]
        for entry in inventory["preserved_preexisting_artifacts"]
    ] == [0]


def test_candidate_set_fails_missing_and_invalid_then_succeeds_order_independently(tmp_path):
    runtime, identity, provenance = _runtime()
    kwargs = dict(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=tmp_path,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        retriever="contriever",
    )
    with pytest.raises(CandidateProductionValidationError, match="incomplete"):
        materialize_candidate_set_inventory(**kwargs)

    for position in range(3):
        _write_candidate(runtime, identity, provenance, tmp_path, position)
    valid = materialize_candidate_set_inventory(**kwargs)
    reversed_payload = candidate_set_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=runtime.sample_manifest.manifest_id,
        retriever="contriever",
        expected_query_count=3,
        entries=list(reversed(valid["scientific_payload"]["entries"])),
    )
    assert valid["scientific_payload"] == reversed_payload

    path = tmp_path / "sample_0001.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["scientific_sha256"] = "0" * 64
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(CandidateProductionValidationError, match="invalid_sample_ids"):
        materialize_candidate_set_inventory(**kwargs)


def test_terminal_complete_requires_resolved_scope_and_failed_retains_partial_inventory(tmp_path):
    runtime, identity, provenance = _runtime()
    authority_path = tmp_path / "authority.json"
    _authority(authority_path, runtime.sample_manifest.manifest_id)
    _, planned = _planned_record(tmp_path, runtime, authority_path)
    running = running_candidate_record(
        planned,
        started_at="2026-08-24T10:00:01Z",
        attempt_count=1,
        environment_sha256=ENV_SHA,
        runtime_sha256=RUNTIME_SHA,
        hardware_summary="fixture-cpu",
        evidence_authority_path=authority_path,
    )
    _write_candidate(runtime, identity, provenance, tmp_path, 1)
    current = inspect_candidate_directory(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=tmp_path,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
    )
    payload = candidate_production_output_inventory(
        run_id=planned["run_id"],
        producer_identity="fixture",
        retriever="contriever",
        repository_root=tmp_path,
        scheduled_sample_ids=(1, 2),
        initial_preexisting_entries=(),
        current_inspection=current,
        failures=({"sample_id": 2, "error_type": "RuntimeError", "message": "x"},),
    )
    output_path = tmp_path / "output.json"
    write_candidate_production_output(output_path, payload)
    failed = terminal_candidate_record(
        running,
        completed_at="2026-08-24T10:00:02Z",
        output_inventory_path=output_path,
        repository_root=tmp_path,
        successful_count=1,
        failed_count=1,
        status_counts=payload["status_counts"],
        candidate_set_path=None,
        failure_reason="fixture failure",
        evidence_authority_path=authority_path,
    )
    assert failed["execution"]["status"] == "FAILED"
    assert failed["output"]["partial_output_retained"] is True
    assert len(failed["output"]["artifacts"]) == 1
    with pytest.raises(ValueError, match="candidate_set_path"):
        terminal_candidate_record(
            running,
            completed_at="2026-08-24T10:00:02Z",
            output_inventory_path=output_path,
            repository_root=tmp_path,
            successful_count=2,
            failed_count=0,
            status_counts={"VALID_OUTPUT": 2},
            candidate_set_path=None,
            failure_reason=None,
            evidence_authority_path=authority_path,
        )

    _write_candidate(runtime, identity, provenance, tmp_path, 0)
    _write_candidate(runtime, identity, provenance, tmp_path, 2)
    complete_inspection = inspect_candidate_directory(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=tmp_path,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
    )
    complete_payload = candidate_production_output_inventory(
        run_id=planned["run_id"],
        producer_identity="fixture",
        retriever="contriever",
        repository_root=tmp_path,
        scheduled_sample_ids=(1, 2),
        initial_preexisting_entries=(),
        current_inspection=complete_inspection,
    )
    complete_output_path = tmp_path / "complete-output.json"
    write_candidate_production_output(complete_output_path, complete_payload)
    candidate_set_path = tmp_path / "candidate-set.json"
    materialize_candidate_set_inventory(
        sample_manifest=runtime.sample_manifest,
        ordered_queries=runtime.ordered_queries,
        candidate_directory=tmp_path,
        dataset_provenance=runtime.dataset_provenance,
        corpus_provenance=runtime.corpus_provenance,
        retriever_provenance=provenance,
        corpus_records=runtime.corpus_records,
        candidate_pool=20,
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        retriever="contriever",
        output_path=candidate_set_path,
    )
    complete = terminal_candidate_record(
        running,
        completed_at="2026-08-24T10:00:03Z",
        output_inventory_path=complete_output_path,
        repository_root=tmp_path,
        successful_count=2,
        failed_count=0,
        status_counts=complete_payload["status_counts"],
        candidate_set_path=candidate_set_path,
        failure_reason=None,
        evidence_authority_path=authority_path,
    )
    assert complete["execution"]["status"] == "COMPLETE"
    assert complete["output"]["successful_row_count"] == 2
    assert complete["output"]["artifacts"][1]["row_count"] == 3
