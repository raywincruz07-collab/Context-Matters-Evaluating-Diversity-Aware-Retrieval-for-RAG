from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import run_registry as run_registry_module
from run_registry import (
    CANDIDATE_SET_SCHEMA_VERSION,
    EVIDENCE_AUTHORITY_SCHEMA_VERSION,
    REGISTRY_HEADER,
    RunRecordValidationError,
    RunRegistryConflictError,
    append_run_record,
    candidate_set_artifact_payload,
    candidate_set_artifact_ref,
    candidate_set_scientific_payload,
    canonical_json,
    file_sha256,
    finalize_planned_record,
    output_artifact,
    output_inventory_sha256,
    read_registry,
    stable_json_sha256,
    validate_run_record,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
GIT_SHA = "1" * 40
PUBMED_SAMPLE_SHA = "4704a5250f52144a38b7f4734f887fe11f5bad18198c285fb048b21d414387ff"
PUBMED_SAMPLE_ID = f"sample-manifest:sha256:{PUBMED_SAMPLE_SHA}"


def _artifact(path: str, sha256: str, artifact_id: str | None = None):
    return {"path": path, "sha256": sha256, "artifact_id": artifact_id}


def _diversification(method="none", parameters=None, seed=None):
    parameters = {} if parameters is None else parameters
    return {
        "method": method,
        "parameters": parameters,
        "config_sha256": stable_json_sha256(parameters),
        "seed": seed,
    }


def _planned_record(
    *,
    sprint="sprint1",
    origin="PROSPECTIVE_BACKFILL",
    evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
    context_mode="with_context",
    retriever="bm25",
    diversification=None,
    llm="llama-3.3-70b",
    branch="sprint3",
    hardware="fixture-cpu",
    sample_manifest_id=PUBMED_SAMPLE_ID,
    evidence_authority_path=None,
):
    has_context = context_mode == "with_context"
    record = {
        "schema_version": "sprint3.run-registry-record.v1",
        "created_at": "2026-08-24T10:00:00Z",
        "sprint": sprint,
        "stage": 1,
        "run_type": "GENERATION",
        "evidence_role": evidence_role,
        "origin": origin,
        "protocol_config_bundle_sha256": SHA_A,
        "git": {
            "commit": GIT_SHA,
            "branch": branch,
            "worktree_clean": True,
            "worktree_diff_sha256": None,
        },
        "data": {
            "dataset": "pubmedqa",
            "split": "train",
            "source": "qiaojin/PubMedQA",
            "revision": "fixture-revision",
            "sample_manifest": _artifact(
                "artifacts/sample_manifests/pubmedqa.json",
                SHA_B,
                sample_manifest_id,
            ),
            "corpus_manifest": (
                _artifact(
                    "artifacts/corpus_manifests/pubmedqa.json",
                    SHA_C,
                    f"corpus-manifest:sha256:{SHA_C}",
                )
                if has_context
                else None
            ),
        },
        "retrieval": (
            {
                "retriever": retriever,
                "config_sha256": SHA_D,
                "index": (
                    None
                    if retriever == "bm25"
                    else _artifact(
                        "artifacts/indexes/pubmedqa.idx",
                        SHA_A,
                        f"index:sha256:{SHA_A}",
                    )
                ),
                "candidate_set": _artifact(
                    "artifacts/candidates/pubmedqa_set.json",
                    SHA_B,
                    f"candidate-set:sha256:{SHA_B}",
                ),
                "selected_context": _artifact(
                    "artifacts/contexts/pubmedqa_top5.json",
                    SHA_C,
                    f"selected-context:sha256:{SHA_C}",
                ),
                "candidate_pool": 20,
                "top_k": 5,
            }
            if has_context
            else None
        ),
        "diversification": (
            (_diversification() if diversification is None else diversification)
            if has_context
            else None
        ),
        "context_mode": context_mode,
        "generation": {
            "llm_logical_id": llm,
            "provider": "fixture-provider",
            "physical_model_id": f"fixture/{llm}",
            "model_revision": "fixture-model-revision",
            "model_revision_kind": "IMMUTABLE_REVISION",
            "prompt_sha256": SHA_C,
            "decoding_sha256": SHA_D,
        },
        "evaluation": None,
        "execution": {
            "environment_sha256": SHA_A,
            "runtime_sha256": SHA_B,
            "hardware_summary": hardware,
            "started_at": None,
            "completed_at": None,
            "status": "PLANNED",
            "attempt_count": 0,
            "failure_reason": None,
            "parent_run_id": None,
            "resume_of": None,
        },
        "output": {
            "expected_row_count": 2,
            "completed_row_count": 0,
            "successful_row_count": 0,
            "failed_row_count": 0,
            "output_directory": "results/sprint3/raw/fixture-run",
            "partial_output_retained": False,
            "artifacts": [],
            "output_inventory_sha256": None,
            "raw_artifact_sha256": None,
        },
    }
    kwargs = (
        {}
        if evidence_authority_path is None
        else {"evidence_authority_path": evidence_authority_path}
    )
    return finalize_planned_record(record, **kwargs)


def _running(
    record,
    *,
    attempt=1,
    resume=False,
    hardware=None,
    prior_failure_reason=None,
):
    updated = deepcopy(record)
    updated["execution"].update(
        {
            "started_at": "2026-08-24T10:00:01Z",
            "status": "RUNNING",
            "attempt_count": attempt,
            "resume_of": record["run_id"] if resume else None,
            "failure_reason": prior_failure_reason,
        }
    )
    if hardware is not None:
        updated["execution"]["hardware_summary"] = hardware
    return validate_run_record(updated)


def _failed(record, reason="fixture infrastructure failure", *, attempt=3):
    updated = deepcopy(record)
    updated["execution"].update(
        {
            "started_at": "2026-08-24T10:00:01Z",
            "completed_at": "2026-08-24T10:00:02Z",
            "status": "FAILED",
            "attempt_count": attempt,
            "failure_reason": reason,
            "resume_of": record["run_id"] if attempt > 1 else None,
        }
    )
    return validate_run_record(updated)


def _complete(record, output_path: Path, repository_root: Path, *, attempt=1):
    updated = deepcopy(record)
    artifact = output_artifact(
        output_path,
        repository_root=repository_root,
        row_count=2,
        status_counts={"OK": 1, "ERROR": 1},
        artifact_id="fixture-output-v1",
    )
    updated["execution"].update(
        {
            "started_at": "2026-08-24T10:00:01Z",
            "completed_at": "2026-08-24T10:00:02Z",
            "status": "COMPLETE",
            "attempt_count": attempt,
            "resume_of": record["run_id"] if attempt > 1 else None,
        }
    )
    updated["output"].update(
        {
            "completed_row_count": 2,
            "successful_row_count": 1,
            "failed_row_count": 1,
            "partial_output_retained": False,
            "artifacts": [artifact],
            "output_inventory_sha256": output_inventory_sha256([artifact]),
            "raw_artifact_sha256": artifact["sha256"],
        }
    )
    return validate_run_record(updated)


def _retrieval_planned_record():
    record = deepcopy(_planned_record())
    record.pop("run_id")
    record.update(
        {
            "run_type": "RETRIEVAL",
            "context_mode": None,
            "diversification": None,
            "generation": None,
        }
    )
    record["retrieval"]["candidate_set"] = None
    record["retrieval"]["selected_context"] = None
    return finalize_planned_record(record)


def _write_authority(path: Path, entries):
    payload = {
        "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
        "authorities": [
            {
                "dataset": dataset,
                "evidence_role": role,
                "sample_manifest_id": manifest_id,
                "sample_manifest_path": f"artifacts/sample_manifests/{dataset}-{role}.json",
                "authority_protocols": [
                    "docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md"
                ],
            }
            for dataset, role, manifest_id in entries
        ],
    }
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path


def _refinalize(record, *, evidence_authority_path=None):
    value = deepcopy(record)
    value.pop("run_id", None)
    kwargs = (
        {}
        if evidence_authority_path is None
        else {"evidence_authority_path": evidence_authority_path}
    )
    return finalize_planned_record(value, **kwargs)


def _role_record(
    base,
    *,
    dataset,
    role,
    stage,
    manifest_id,
    authority_path,
    origin="CURRENT_PROTOCOL",
):
    raw = deepcopy(base)
    raw.pop("run_id", None)
    raw.update(
        {
            "sprint": "sprint3",
            "stage": stage,
            "evidence_role": role,
            "origin": origin,
        }
    )
    raw["data"].update(
        {
            "dataset": dataset,
            "split": role.lower(),
            "source": f"fixture/{dataset}",
            "revision": "immutable-fixture-revision",
            "sample_manifest": _artifact(
                f"artifacts/sample_manifests/{dataset}-{role}.json",
                SHA_A,
                manifest_id,
            ),
            "corpus_manifest": None,
        }
    )
    return finalize_planned_record(raw, evidence_authority_path=authority_path)


def test_valid_prospective_backfill_record_passes():
    record = _planned_record()
    assert record["run_id"].startswith(
        "run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-"
    )
    assert validate_run_record(record) == record


def test_missing_required_identity_fails():
    record = _planned_record()
    del record["data"]["sample_manifest"]
    with pytest.raises(RunRecordValidationError, match="data keys mismatch"):
        validate_run_record(record)


def test_duplicate_identical_registration_is_safe(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = _planned_record()
    assert append_run_record(registry, record) is True
    before = registry.read_bytes()
    assert append_run_record(registry, record) is False
    assert registry.read_bytes() == before
    assert read_registry(registry) == (record,)


def test_duplicate_run_id_with_conflicting_scientific_record_fails(tmp_path, monkeypatch):
    registry = tmp_path / "registry.jsonl"
    record = _planned_record()
    append_run_record(registry, record)
    conflicting_without_id = deepcopy(record)
    conflicting_without_id.pop("run_id")
    conflicting_without_id["generation"]["llm_logical_id"] = "gemma4-26b"
    monkeypatch.setattr(run_registry_module, "derive_run_id", lambda _: record["run_id"])
    conflicting = finalize_planned_record(conflicting_without_id)
    with pytest.raises(RunRegistryConflictError, match="conflicting scientific"):
        append_run_record(registry, conflicting)


def test_append_does_not_overwrite_prior_records(tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _planned_record(llm="llama-3.3-70b")
    second = _planned_record(llm="gemma4-26b")
    append_run_record(registry, first)
    first_bytes = registry.read_bytes()
    append_run_record(registry, second)
    assert registry.read_bytes().startswith(first_bytes)
    assert read_registry(registry) == (first, second)


def test_failed_run_is_retained_with_reason(tmp_path):
    registry = tmp_path / "registry.jsonl"
    planned = _planned_record()
    failed = _failed(planned)
    append_run_record(registry, planned)
    append_run_record(registry, _running(planned, attempt=1))
    append_run_record(
        registry,
        _running(
            planned,
            attempt=2,
            resume=True,
            prior_failure_reason="attempt 1 infrastructure failure",
        ),
    )
    append_run_record(
        registry,
        _running(
            planned,
            attempt=3,
            resume=True,
            prior_failure_reason="attempt 2 infrastructure failure",
        ),
    )
    append_run_record(registry, failed)
    records = read_registry(registry)
    assert [item["execution"]["status"] for item in records] == [
        "PLANNED",
        "RUNNING",
        "RUNNING",
        "RUNNING",
        "FAILED",
    ]
    assert records[-1]["execution"]["failure_reason"] == "fixture infrastructure failure"


def test_file_and_config_hashes_are_stable(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"exact fixture bytes\n")
    first_file_hash = file_sha256(path)
    first_config_hash = stable_json_sha256({"b": [2, 3], "a": 1})
    assert file_sha256(path) == first_file_hash
    assert stable_json_sha256({"a": 1, "b": [2, 3]}) == first_config_hash


def test_output_inventory_records_path_hash_rows_and_statuses(tmp_path):
    output_path = tmp_path / "results/sprint3/raw/output.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("fixture output\n", encoding="utf-8")
    artifact = output_artifact(
        output_path,
        repository_root=tmp_path,
        row_count=2,
        status_counts={"OK": 1, "ERROR": 1},
    )
    assert artifact["path"] == "results/sprint3/raw/output.jsonl"
    assert artifact["sha256"] == file_sha256(output_path)
    assert artifact["row_count"] == 2
    assert artifact["status_counts"] == {"OK": 1, "ERROR": 1}
    assert output_inventory_sha256([artifact]) == output_inventory_sha256([artifact])


def test_without_context_identity_is_shared_not_retriever_specific():
    first = _planned_record(context_mode="without_context", retriever="bm25")
    second = _planned_record(context_mode="without_context", retriever="colbertv2")
    assert first["retrieval"] is None
    assert first["diversification"] is None
    assert first["run_id"] == second["run_id"]
    invalid = deepcopy(first)
    invalid["retrieval"] = _planned_record()["retrieval"]
    with pytest.raises(RunRecordValidationError, match="WITHOUT_CONTEXT"):
        validate_run_record(invalid)


def test_branch_and_hardware_are_execution_provenance_not_scientific_identity():
    first = _planned_record(
        context_mode="without_context", branch="sprint3", hardware="cpu-a"
    )
    second = _planned_record(
        context_mode="without_context", branch="renamed-branch", hardware="gpu-b"
    )
    assert first["git"]["commit"] == second["git"]["commit"]
    assert first["run_id"] == second["run_id"]


def test_environment_and_runtime_are_provenance_but_resume_compatibility_is_exact(tmp_path):
    first = _planned_record(context_mode="without_context")
    changed = deepcopy(first)
    changed["execution"]["environment_sha256"] = SHA_D
    assert _refinalize(changed)["run_id"] == first["run_id"]

    registry = tmp_path / "registry.jsonl"
    append_run_record(registry, first)
    append_run_record(registry, _running(first))
    retry = _running(
        changed,
        attempt=2,
        resume=True,
        prior_failure_reason="transient infrastructure failure",
    )
    with pytest.raises(RunRegistryConflictError, match="environment_sha256"):
        append_run_record(registry, retry)


def test_without_context_rejects_corpus_and_cannot_be_forked_by_retrieval_provenance():
    record = _planned_record(context_mode="without_context")
    invalid = deepcopy(record)
    invalid["data"]["corpus_manifest"] = _artifact(
        "artifacts/corpus_manifests/pubmedqa.json",
        SHA_C,
        f"corpus-manifest:sha256:{SHA_C}",
    )
    with pytest.raises(RunRecordValidationError, match="WITHOUT_CONTEXT"):
        validate_run_record(invalid)


def test_with_context_generation_requires_explicit_baseline_and_selected_lineage():
    record = _planned_record()
    missing_baseline = deepcopy(record)
    missing_baseline["diversification"] = None
    with pytest.raises(RunRecordValidationError, match="canonical diversification"):
        validate_run_record(missing_baseline)
    missing_selected = deepcopy(record)
    missing_selected["retrieval"]["selected_context"] = None
    with pytest.raises(RunRecordValidationError, match="selected-context"):
        validate_run_record(missing_selected)
    assert record["diversification"]["method"] == "none"


def test_index_and_selected_context_require_content_addressed_ids():
    neural = _planned_record(retriever="dpr")
    assert neural["retrieval"]["index"]["artifact_id"] == f"index:sha256:{SHA_A}"
    assert neural["retrieval"]["selected_context"]["artifact_id"] == (
        f"selected-context:sha256:{SHA_C}"
    )

    malformed_index = deepcopy(neural)
    malformed_index["retrieval"]["index"]["artifact_id"] = "dpr-index-latest"
    with pytest.raises(RunRecordValidationError, match="index.artifact_id"):
        validate_run_record(malformed_index)

    malformed_context = deepcopy(neural)
    malformed_context["retrieval"]["selected_context"]["artifact_id"] = "top-five"
    with pytest.raises(RunRecordValidationError, match="selected_context.artifact_id"):
        validate_run_record(malformed_context)

    assert _planned_record(retriever="bm25")["retrieval"]["index"] is None


def test_candidate_set_inventory_is_stable_and_binds_exact_mapping(tmp_path):
    scientific = candidate_set_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=PUBMED_SAMPLE_ID,
        retriever="bm25",
        expected_query_count=2,
        entries=(
            {"sample_id": "q1", "candidate_artifact_id": f"candidate:sha256:{SHA_A}"},
            {"sample_id": "q2", "candidate_artifact_id": f"candidate:sha256:{SHA_B}"},
        ),
    )
    assert scientific["schema_version"] == CANDIDATE_SET_SCHEMA_VERSION
    compact = candidate_set_artifact_payload(scientific, provenance={"note": "one"})
    alternate = candidate_set_artifact_payload(scientific, provenance={"note": "two"})
    assert compact["candidate_set_id"] == alternate["candidate_set_id"]
    reversed_mapping = candidate_set_artifact_payload(
        candidate_set_scientific_payload(
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_manifest_id=PUBMED_SAMPLE_ID,
            retriever="bm25",
            expected_query_count=2,
            entries=tuple(reversed(scientific["entries"])),
        )
    )
    assert compact["candidate_set_id"] == reversed_mapping["candidate_set_id"]
    changed_mapping = deepcopy(scientific)
    changed_mapping["entries"][0]["candidate_artifact_id"] = (
        f"candidate:sha256:{SHA_D}"
    )
    assert (
        compact["candidate_set_id"]
        != candidate_set_artifact_payload(changed_mapping)["candidate_set_id"]
    )

    first_path = tmp_path / "artifacts/candidates/first.json"
    second_path = tmp_path / "artifacts/candidates/second.json"
    first_path.parent.mkdir(parents=True)
    first_path.write_text(canonical_json(compact) + "\n", encoding="utf-8")
    second_path.write_text(
        json.dumps(alternate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first_ref = candidate_set_artifact_ref(first_path, repository_root=tmp_path)
    second_ref = candidate_set_artifact_ref(second_path, repository_root=tmp_path)
    assert first_ref["sha256"] != second_ref["sha256"]
    assert first_ref["artifact_id"] == second_ref["artifact_id"]

    first_run = _planned_record()
    first_run["retrieval"]["candidate_set"] = first_ref
    second_run = deepcopy(first_run)
    second_run["retrieval"]["candidate_set"] = second_ref
    assert _refinalize(first_run)["run_id"] == _refinalize(second_run)["run_id"]


def test_candidate_set_inventory_rejects_incomplete_or_duplicate_mapping():
    with pytest.raises(RunRecordValidationError, match="expected_query_count"):
        candidate_set_scientific_payload(
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_manifest_id=PUBMED_SAMPLE_ID,
            retriever="bm25",
            expected_query_count=2,
            entries=(
                {"sample_id": "q1", "candidate_artifact_id": f"candidate:sha256:{SHA_A}"},
            ),
        )
    with pytest.raises(RunRecordValidationError, match="sample IDs must be unique"):
        candidate_set_scientific_payload(
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_manifest_id=PUBMED_SAMPLE_ID,
            retriever="bm25",
            expected_query_count=2,
            entries=(
                {"sample_id": "q1", "candidate_artifact_id": f"candidate:sha256:{SHA_A}"},
                {"sample_id": "q1", "candidate_artifact_id": f"candidate:sha256:{SHA_B}"},
            ),
        )
    with pytest.raises(RunRecordValidationError, match="sample IDs must be unique"):
        candidate_set_scientific_payload(
            dataset="pubmedqa",
            evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
            sample_manifest_id=PUBMED_SAMPLE_ID,
            retriever="bm25",
            expected_query_count=2,
            entries=(
                {"sample_id": 1, "candidate_artifact_id": f"candidate:sha256:{SHA_A}"},
                {
                    "sample_id": np.int64(1),
                    "candidate_artifact_id": f"candidate:sha256:{SHA_B}",
                },
            ),
        )


def test_candidate_set_uses_existing_string_vs_integer_id_distinction():
    payload = candidate_set_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=PUBMED_SAMPLE_ID,
        retriever="bm25",
        expected_query_count=2,
        entries=(
            {"sample_id": 1, "candidate_artifact_id": f"candidate:sha256:{SHA_A}"},
            {"sample_id": "1", "candidate_artifact_id": f"candidate:sha256:{SHA_B}"},
        ),
    )
    assert {type(entry["sample_id"]) for entry in payload["entries"]} == {int, str}


def test_dataset_role_governance_rejects_pubmedqa_selection_and_protected():
    for role, stage in (("SELECTION", 3), ("PROJECT_PROTECTED_FINAL", 5)):
        record = deepcopy(_planned_record(context_mode="without_context"))
        record["stage"] = stage
        record["evidence_role"] = role
        record["origin"] = "CURRENT_PROTOCOL"
        record["run_id"] = "run-placeholder-" + "0" * 24
        with pytest.raises(RunRecordValidationError, match="PubMedQA"):
            validate_run_record(record)


def test_evidence_authority_binds_dataset_role_to_exact_manifest(tmp_path):
    development_id = f"sample-manifest:sha256:{SHA_A}"
    selection_id = f"sample-manifest:sha256:{SHA_B}"
    protected_id = f"sample-manifest:sha256:{SHA_C}"
    authority = _write_authority(
        tmp_path / "authority.json",
        [
            ("hotpotqa", "DEVELOPMENT", development_id),
            ("hotpotqa", "SELECTION", selection_id),
            ("hotpotqa", "PROJECT_PROTECTED_FINAL", protected_id),
        ],
    )
    base = _planned_record(context_mode="without_context")
    selected = _role_record(
        base,
        dataset="hotpotqa",
        role="SELECTION",
        stage=3,
        manifest_id=selection_id,
        authority_path=authority,
    )
    assert selected["data"]["sample_manifest"]["artifact_id"] == selection_id

    wrong = deepcopy(selected)
    wrong["data"]["sample_manifest"]["artifact_id"] = protected_id
    wrong["run_id"] = "run-placeholder-" + "0" * 24
    with pytest.raises(RunRecordValidationError, match="does not match"):
        validate_run_record(wrong, evidence_authority_path=authority)



def test_hotpotqa_official_test_full_is_authorized_for_final_and_analysis(
    tmp_path,
):
    manifest_id = f"sample-manifest:sha256:{SHA_D}"
    authority = _write_authority(
        tmp_path / "authority.json",
        [("hotpotqa", "OFFICIAL_TEST_FULL", manifest_id)],
    )

    base = _planned_record(context_mode="without_context")

    final = _role_record(
        base,
        dataset="hotpotqa",
        role="OFFICIAL_TEST_FULL",
        stage=5,
        manifest_id=manifest_id,
        authority_path=authority,
    )

    assert final["stage"] == 5
    assert final["evidence_role"] == "OFFICIAL_TEST_FULL"

    analysis = deepcopy(final)
    analysis.pop("run_id")
    analysis.update(
        {
            "stage": 6,
            "run_type": "ANALYSIS",
            "context_mode": None,
            "generation": None,
        }
    )

    analyzed = finalize_planned_record(
        analysis,
        evidence_authority_path=authority,
    )

    assert analyzed["stage"] == 6
    assert analyzed["evidence_role"] == "OFFICIAL_TEST_FULL"

def test_unmaterialized_role_fails_closed_without_fabricated_manifest(tmp_path):
    authority = _write_authority(tmp_path / "authority.json", [])
    base = _planned_record(context_mode="without_context")
    raw = deepcopy(base)
    raw.pop("run_id")
    raw.update(
        {
            "sprint": "sprint3",
            "stage": 3,
            "evidence_role": "SELECTION",
            "origin": "CURRENT_PROTOCOL",
        }
    )
    raw["data"].update(
        {
            "dataset": "asqa",
            "split": "selection",
            "source": "din0s/asqa",
            "revision": "immutable-fixture-revision",
            "sample_manifest": _artifact(
                "artifacts/sample_manifests/asqa-selection.json",
                SHA_A,
                f"sample-manifest:sha256:{SHA_A}",
            ),
        }
    )
    with pytest.raises(RunRecordValidationError, match="no authorized"):
        finalize_planned_record(raw, evidence_authority_path=authority)


def test_stage_six_analysis_retains_consumed_protected_role(tmp_path):
    protected_id = f"sample-manifest:sha256:{SHA_C}"
    authority = _write_authority(
        tmp_path / "authority.json",
        [("asqa", "PROJECT_PROTECTED_FINAL", protected_id)],
    )
    base = _planned_record(context_mode="without_context")
    raw = deepcopy(base)
    raw.pop("run_id")
    raw.update(
        {
            "sprint": "sprint3",
            "stage": 6,
            "run_type": "ANALYSIS",
            "evidence_role": "PROJECT_PROTECTED_FINAL",
            "origin": "CURRENT_PROTOCOL",
            "context_mode": None,
            "generation": None,
        }
    )
    raw["data"].update(
        {
            "dataset": "asqa",
            "split": "dev",
            "source": "din0s/asqa",
            "revision": "immutable-fixture-revision",
            "sample_manifest": _artifact(
                "artifacts/sample_manifests/asqa-protected.json",
                SHA_A,
                protected_id,
            ),
            "corpus_manifest": None,
        }
    )
    analyzed = finalize_planned_record(raw, evidence_authority_path=authority)
    assert analyzed["stage"] == 6
    assert analyzed["evidence_role"] == "PROJECT_PROTECTED_FINAL"

    disguised_generation = deepcopy(analyzed)
    disguised_generation["generation"] = base["generation"]
    disguised_generation["context_mode"] = "without_context"
    disguised_generation["run_id"] = "run-placeholder-" + "0" * 24
    with pytest.raises(RunRecordValidationError, match="ANALYSIS.*forbids"):
        validate_run_record(
            disguised_generation, evidence_authority_path=authority
        )


def test_analysis_and_resource_pilot_cannot_masquerade_as_generation():
    generated = _planned_record()
    analysis_with_generation = deepcopy(generated)
    analysis_with_generation["run_type"] = "ANALYSIS"
    with pytest.raises(RunRecordValidationError, match="ANALYSIS.*forbids"):
        validate_run_record(analysis_with_generation)

    resource_with_generation = deepcopy(generated)
    resource_with_generation["run_type"] = "RESOURCE_PILOT"
    with pytest.raises(RunRecordValidationError, match="RESOURCE_PILOT forbids"):
        validate_run_record(resource_with_generation)


def test_analysis_retrieval_lineage_requires_explicit_none_baseline():
    generated = _planned_record()
    analysis = deepcopy(generated)
    analysis.pop("run_id")
    analysis.update(
        {"run_type": "ANALYSIS", "generation": None, "context_mode": None}
    )
    canonical = finalize_planned_record(analysis)
    assert canonical["diversification"]["method"] == "none"

    ambiguous = deepcopy(canonical)
    ambiguous["diversification"] = None
    with pytest.raises(RunRecordValidationError, match="method=none"):
        validate_run_record(ambiguous)


def test_resource_pilot_allows_first_stage_measurement_without_fake_treatment():
    generated = _planned_record()
    pilot = deepcopy(generated)
    pilot.pop("run_id")
    pilot.update(
        {
            "run_type": "RESOURCE_PILOT",
            "generation": None,
            "context_mode": None,
            "diversification": None,
        }
    )
    pilot["retrieval"]["candidate_set"] = None
    pilot["retrieval"]["selected_context"] = None
    assert finalize_planned_record(pilot)["run_type"] == "RESOURCE_PILOT"

    downstream = deepcopy(pilot)
    downstream["retrieval"]["candidate_set"] = generated["retrieval"]["candidate_set"]
    with pytest.raises(RunRecordValidationError, match="canonical diversification"):
        finalize_planned_record(downstream)


def test_canonical_candidate_pool_and_top_k_are_enforced():
    record = _planned_record()
    record["retrieval"]["candidate_pool"] = 10
    record["run_id"] = "run-placeholder-" + "0" * 24
    with pytest.raises(RunRecordValidationError, match="candidate_pool=20"):
        validate_run_record(record)


def test_stochastic_dpp_requires_explicit_seed():
    diversification = _diversification("dpp_sample", {"mode": "sample"}, None)
    with pytest.raises(RunRecordValidationError, match="requires an explicit seed"):
        _planned_record(diversification=diversification)
    seeded = _planned_record(
        diversification=_diversification("dpp_sample", {"mode": "sample"}, 1)
    )
    assert seeded["diversification"]["seed"] == 1


@pytest.mark.parametrize(
    ("method", "parameters", "seed"),
    (
        ("none", {}, None),
        ("mmr", {"lambda": 0.5}, None),
        ("agglo", {"k": 3}, None),
        ("dpp_map", {"mode": "map"}, None),
        ("kmeans", {"k": 3}, 42),
    ),
)
def test_deterministic_methods_use_only_frozen_seed_semantics(method, parameters, seed):
    record = _planned_record(
        diversification=_diversification(method, parameters, seed)
    )
    assert record["diversification"]["seed"] == seed
    if method != "kmeans":
        invalid = deepcopy(record)
        invalid["diversification"]["seed"] = 42
        with pytest.raises(RunRecordValidationError, match="must not carry"):
            validate_run_record(invalid)


def test_registry_v1_rejects_fabricated_historical_lifecycle_origin():
    with pytest.raises(RunRecordValidationError, match="origin is not canonical"):
        _planned_record(origin="HISTORICAL")
    assert _planned_record()["origin"] == "PROSPECTIVE_BACKFILL"


@pytest.mark.parametrize("sprint", ("sprint1", "sprint2", "sprint3"))
def test_only_canonical_sprint_values_are_allowed(sprint):
    assert _planned_record(sprint=sprint)["sprint"] == sprint


def test_arbitrary_sprint_and_sprint_retry_fork_are_rejected():
    for invalid_sprint in ("sprint1-retry", "sprint3-new", "final2"):
        with pytest.raises(RunRecordValidationError, match="sprint must be exactly"):
            _planned_record(sprint=invalid_sprint)

    planned = _planned_record()
    fake_retry = deepcopy(planned)
    fake_retry["sprint"] = "sprint1-retry"
    with pytest.raises(RunRecordValidationError, match="sprint must be exactly"):
        validate_run_record(fake_retry)


def test_infrastructure_retries_use_successive_running_attempts_same_run_id(tmp_path):
    registry = tmp_path / "registry.jsonl"
    planned = _planned_record(context_mode="without_context")
    first = _running(planned, attempt=1)
    second = _running(
        planned,
        attempt=2,
        resume=True,
        hardware="replacement-gpu",
        prior_failure_reason="attempt 1 transport interruption",
    )
    third = _running(
        planned,
        attempt=3,
        resume=True,
        prior_failure_reason="attempt 2 provider interruption",
    )
    exhausted = _failed(planned, attempt=3)
    exhausted["execution"]["resume_of"] = planned["run_id"]
    exhausted = validate_run_record(exhausted)

    for snapshot in (planned, first, second, third, exhausted):
        assert snapshot["run_id"] == planned["run_id"]
        append_run_record(registry, snapshot)
    assert [
        item["execution"]["attempt_count"] for item in read_registry(registry)
    ] == [0, 1, 2, 3, 3]
    assert [item["execution"]["status"] for item in read_registry(registry)] == [
        "PLANNED",
        "RUNNING",
        "RUNNING",
        "RUNNING",
        "FAILED",
    ]
    assert second["execution"]["failure_reason"] == "attempt 1 transport interruption"


def test_attempt_count_monotonicity_cap_and_resume_lineage_are_enforced(tmp_path):
    planned = _planned_record(context_mode="without_context")
    too_many = deepcopy(_running(planned))
    too_many["execution"]["attempt_count"] = 4
    with pytest.raises(RunRecordValidationError, match="at most 3"):
        validate_run_record(too_many)

    registry = tmp_path / "registry.jsonl"
    append_run_record(registry, planned)
    append_run_record(registry, _running(planned, attempt=1))
    missing_lineage = _running(
        planned,
        attempt=2,
        resume=False,
        prior_failure_reason="attempt 1 transport interruption",
    )
    with pytest.raises(RunRegistryConflictError, match="resume_of=run_id"):
        append_run_record(registry, missing_lineage)


def test_retrieval_attempts_one_through_three_are_valid_and_four_is_rejected():
    planned = _retrieval_planned_record()
    first = _running(planned, attempt=1)
    second = _running(
        planned,
        attempt=2,
        resume=True,
        prior_failure_reason="attempt 1 infrastructure interruption",
    )
    third = _running(
        planned,
        attempt=3,
        resume=True,
        prior_failure_reason="attempt 2 infrastructure interruption",
    )

    assert {planned["run_id"], first["run_id"], second["run_id"], third["run_id"]} == {
        planned["run_id"]
    }
    assert [
        first["execution"]["attempt_count"],
        second["execution"]["attempt_count"],
        third["execution"]["attempt_count"],
    ] == [1, 2, 3]

    with pytest.raises(RunRecordValidationError, match="RETRIEVAL permits at most 3"):
        _running(
            planned,
            attempt=4,
            resume=True,
            prior_failure_reason="attempt 3 infrastructure interruption",
        )


def test_retrieval_terminal_snapshots_use_the_actual_preceding_attempt(tmp_path):
    planned = _retrieval_planned_record()
    first = _running(planned, attempt=1)

    failed_one_registry = tmp_path / "retrieval-failed-one.jsonl"
    append_run_record(failed_one_registry, planned)
    append_run_record(failed_one_registry, first)
    append_run_record(failed_one_registry, _failed(planned, attempt=1))
    failed_one = read_registry(failed_one_registry)[-1]
    assert failed_one["execution"]["status"] == "FAILED"
    assert failed_one["execution"]["attempt_count"] == 1

    completed_two_registry = tmp_path / "retrieval-completed-two.jsonl"
    output_path = tmp_path / "retrieval-attempt-two-output.jsonl"
    output_path.write_text("fixture output\n", encoding="utf-8")
    second = _running(
        planned,
        attempt=2,
        resume=True,
        prior_failure_reason="attempt 1 infrastructure interruption",
    )
    complete_two = _complete(planned, output_path, tmp_path, attempt=2)
    append_run_record(completed_two_registry, planned)
    append_run_record(completed_two_registry, first)
    append_run_record(completed_two_registry, second)
    append_run_record(completed_two_registry, complete_two)
    completed_two = read_registry(completed_two_registry)[-1]
    assert completed_two["execution"]["status"] == "COMPLETE"
    assert completed_two["execution"]["attempt_count"] == 2

    jumped_registry = tmp_path / "retrieval-jumped-failure.jsonl"
    append_run_record(jumped_registry, planned)
    append_run_record(jumped_registry, first)
    with pytest.raises(RunRegistryConflictError, match="immediately preceding"):
        append_run_record(jumped_registry, _failed(planned, attempt=3))


def test_terminal_failure_uses_actual_immediately_preceding_attempt(tmp_path):
    planned = _planned_record(context_mode="without_context")

    failed_one_registry = tmp_path / "failed-one.jsonl"
    append_run_record(failed_one_registry, planned)
    append_run_record(failed_one_registry, _running(planned, attempt=1))
    append_run_record(failed_one_registry, _failed(planned, attempt=1))

    failed_two_registry = tmp_path / "failed-two.jsonl"
    append_run_record(failed_two_registry, planned)
    append_run_record(failed_two_registry, _running(planned, attempt=1))
    second = _running(
        planned,
        attempt=2,
        resume=True,
        prior_failure_reason="attempt 1 retryable infrastructure failure",
    )
    append_run_record(failed_two_registry, second)
    append_run_record(failed_two_registry, _failed(planned, attempt=2))

    jumped_registry = tmp_path / "jumped.jsonl"
    append_run_record(jumped_registry, planned)
    append_run_record(jumped_registry, _running(planned, attempt=1))
    with pytest.raises(RunRegistryConflictError, match="immediately preceding"):
        append_run_record(jumped_registry, _failed(planned, attempt=3))

    failed_three_registry = tmp_path / "failed-three.jsonl"
    append_run_record(failed_three_registry, planned)
    append_run_record(failed_three_registry, _running(planned, attempt=1))
    append_run_record(failed_three_registry, second)
    third = _running(
        planned,
        attempt=3,
        resume=True,
        prior_failure_reason="attempt 2 retryable infrastructure failure",
    )
    append_run_record(failed_three_registry, third)
    append_run_record(failed_three_registry, _failed(planned, attempt=3))

    assert read_registry(failed_one_registry)[-1]["execution"]["attempt_count"] == 1
    assert read_registry(failed_two_registry)[-1]["execution"]["attempt_count"] == 2
    assert read_registry(failed_three_registry)[-1]["execution"]["attempt_count"] == 3


def test_terminal_and_shortcut_lifecycle_transitions_are_prohibited(tmp_path):
    output_path = tmp_path / "results/sprint3/raw/output.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("fixture output\n", encoding="utf-8")
    planned = _planned_record()
    running = _running(planned)
    complete = _complete(planned, output_path, tmp_path)
    failed = _failed(planned)

    direct = tmp_path / "direct.jsonl"
    append_run_record(direct, planned)
    with pytest.raises(RunRegistryConflictError, match="PLANNED->COMPLETE"):
        append_run_record(direct, complete)
    with pytest.raises(RunRegistryConflictError, match="PLANNED->FAILED"):
        append_run_record(direct, failed)

    completed_registry = tmp_path / "complete.jsonl"
    for snapshot in (planned, running, complete):
        append_run_record(completed_registry, snapshot)
    with pytest.raises(RunRegistryConflictError, match="COMPLETE->FAILED"):
        append_run_record(completed_registry, failed)

    failed_registry = tmp_path / "failed.jsonl"
    append_run_record(failed_registry, planned)
    append_run_record(failed_registry, running)
    append_run_record(
        failed_registry,
        _running(
            planned,
            attempt=2,
            resume=True,
            prior_failure_reason="attempt 1 infrastructure failure",
        ),
    )
    append_run_record(
        failed_registry,
        _running(
            planned,
            attempt=3,
            resume=True,
            prior_failure_reason="attempt 2 infrastructure failure",
        ),
    )
    append_run_record(failed_registry, failed)
    with pytest.raises(RunRegistryConflictError, match="FAILED->COMPLETE"):
        append_run_record(failed_registry, complete)


def test_created_at_change_is_safe_planned_reregistration_not_new_science(tmp_path):
    registry = tmp_path / "registry.jsonl"
    planned = _planned_record()
    append_run_record(registry, planned)
    changed = deepcopy(planned)
    changed["created_at"] = "2026-08-24T11:00:00Z"
    assert append_run_record(registry, changed) is False
    assert len(read_registry(registry)) == 1


def test_run_type_payload_matrix_rejects_decorative_or_incompatible_payloads():
    generation = _planned_record()
    invalid_generation = deepcopy(generation)
    invalid_generation["evaluation"] = {
        "bundle_version": "v1",
        "bundle_sha256": SHA_A,
        "metric_registry_sha256": SHA_B,
    }
    with pytest.raises(RunRecordValidationError, match="forbids evaluation"):
        validate_run_record(invalid_generation)

    retrieval = deepcopy(generation)
    retrieval.pop("run_id")
    retrieval.update(
        {"run_type": "RETRIEVAL", "context_mode": None, "generation": None}
    )
    retrieval["diversification"] = None
    retrieval["retrieval"]["candidate_set"] = None
    retrieval["retrieval"]["selected_context"] = None
    retrieval = finalize_planned_record(retrieval)
    assert retrieval["run_type"] == "RETRIEVAL"

    missing_retrieval = deepcopy(retrieval)
    missing_retrieval["retrieval"] = None
    with pytest.raises(RunRecordValidationError, match="RETRIEVAL requires"):
        validate_run_record(missing_retrieval)

    diversification = deepcopy(retrieval)
    diversification.pop("run_id")
    diversification["run_type"] = "DIVERSIFICATION"
    diversification["diversification"] = _diversification()
    diversification["retrieval"]["candidate_set"] = generation["retrieval"]["candidate_set"]
    diversification = finalize_planned_record(diversification)
    assert diversification["run_type"] == "DIVERSIFICATION"

    evaluation = deepcopy(generation)
    evaluation["run_type"] = "EVALUATION"
    evaluation["evaluation"] = None
    with pytest.raises(RunRecordValidationError, match="EVALUATION requires"):
        validate_run_record(evaluation)

    valid_evaluation = deepcopy(generation)
    valid_evaluation.pop("run_id")
    valid_evaluation.update(
        {
            "run_type": "EVALUATION",
            "context_mode": None,
            "generation": None,
            "retrieval": None,
            "diversification": None,
            "evaluation": {
                "bundle_version": "fixture-v1",
                "bundle_sha256": SHA_A,
                "metric_registry_sha256": SHA_B,
            },
        }
    )
    valid_evaluation["data"]["corpus_manifest"] = None
    assert finalize_planned_record(valid_evaluation)["run_type"] == "EVALUATION"


def test_model_physical_identity_requires_explicit_revision_state():
    record = _planned_record(context_mode="without_context")
    unavailable = deepcopy(record)
    unavailable["generation"].update(
        {"model_revision": None, "model_revision_kind": "NOT_PROVIDED_BY_PROVIDER"}
    )
    assert _refinalize(unavailable)["generation"]["model_revision"] is None
    invalid = deepcopy(unavailable)
    invalid["generation"]["model_revision_kind"] = "IMMUTABLE_REVISION"
    with pytest.raises(RunRecordValidationError, match="requires an immutable"):
        validate_run_record(invalid)


def test_failed_partial_output_is_retained_and_absence_is_explicit(tmp_path):
    output_path = tmp_path / "results/sprint3/raw/partial.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("retained row\n", encoding="utf-8")
    planned = _planned_record()
    artifact = output_artifact(
        output_path,
        repository_root=tmp_path,
        row_count=1,
        status_counts={"OK": 1},
        artifact_id="partial-output-v1",
    )
    partial = deepcopy(_failed(planned))
    partial["output"].update(
        {
            "completed_row_count": 1,
            "successful_row_count": 1,
            "failed_row_count": 0,
            "partial_output_retained": True,
            "artifacts": [artifact],
            "output_inventory_sha256": output_inventory_sha256([artifact]),
            "raw_artifact_sha256": artifact["sha256"],
        }
    )
    partial = validate_run_record(partial)
    assert partial["output"]["partial_output_retained"] is True
    assert _failed(planned)["output"]["partial_output_retained"] is False


def test_complete_lifecycle_and_output_are_append_only(tmp_path):
    registry = tmp_path / "registry.jsonl"
    output_path = tmp_path / "results/sprint3/raw/output.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("fixture output\n", encoding="utf-8")
    planned = _planned_record()
    running = _running(planned)
    complete = _complete(planned, output_path, tmp_path)
    append_run_record(registry, planned)
    append_run_record(registry, running)
    append_run_record(registry, complete)
    assert [item["execution"]["status"] for item in read_registry(registry)] == [
        "PLANNED",
        "RUNNING",
        "COMPLETE",
    ]
    with pytest.raises(RunRegistryConflictError, match="invalid lifecycle"):
        append_run_record(registry, running)


def test_failed_append_or_validation_does_not_corrupt_registry(tmp_path):
    registry = tmp_path / "registry.jsonl"
    record = _planned_record()
    append_run_record(registry, record)
    before = registry.read_bytes()

    invalid = deepcopy(record)
    invalid["output"]["completed_row_count"] = 1
    with pytest.raises(RunRecordValidationError):
        append_run_record(registry, invalid)
    assert registry.read_bytes() == before

    conflicting = deepcopy(record)
    conflicting["execution"]["environment_sha256"] = SHA_D
    running = _running(record)
    append_run_record(registry, running)
    before_running_conflict = registry.read_bytes()
    conflicting_running = _running(
        conflicting,
        attempt=2,
        resume=True,
        prior_failure_reason="attempt 1 transport interruption",
    )
    with pytest.raises(RunRegistryConflictError, match="environment_sha256"):
        append_run_record(registry, conflicting_running)
    assert registry.read_bytes() == before_running_conflict
    assert canonical_json(REGISTRY_HEADER).encode() in before
