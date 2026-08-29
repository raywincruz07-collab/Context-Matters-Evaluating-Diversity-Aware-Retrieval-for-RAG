from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import generation.runner as runner_module
from generation._io import stable_json_sha256
from generation.artifacts import (
    build_generation_artifact,
    generation_artifact_path,
    write_generation_artifact,
)
from generation.maki import (
    CanonicalMakiAdapter,
    MakiConfig,
    MakiInfrastructureError,
    PRIMARY_LLM_LOGICAL_IDS,
)
from generation.prompts import render_pubmedqa_prompt
from generation.repeatability import (
    RepeatabilityGateError,
    require_passing_repeatability_gate,
    run_repeatability_gate,
    write_repeatability_prompt_manifest,
)
from generation.runner import (
    GenerationBlock,
    _request_for_query,
    _running_record,
    build_generation_planned_record,
    canonical_pubmedqa_generation_matrix,
    execute_generation_block,
    expected_matrix_row_count,
)
from run_registry import (
    EVIDENCE_AUTHORITY_SCHEMA_VERSION,
    append_run_record,
    read_registry,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
GIT_SHA = "1" * 40
SAMPLE_ID = f"sample-manifest:sha256:{SHA_A}"


@dataclass(frozen=True)
class Query:
    position: int
    sample_id: int
    query_text: str


@dataclass(frozen=True)
class Runtime:
    ordered_queries: tuple[Query, ...]


def _authority(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
                "authorities": [
                    {
                        "dataset": "pubmedqa",
                        "evidence_role": "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
                        "sample_manifest_id": SAMPLE_ID,
                        "sample_manifest_path": "artifacts/sample.json",
                        "authority_protocols": ["docs/protocol.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _config(logical_id: str) -> MakiConfig:
    return MakiConfig(
        base_url="https://maki.example/v1",
        logical_model_id=logical_id,
        physical_model_id=f"physical/{logical_id}",
        model_revision=None,
        model_revision_kind="NOT_PROVIDED_BY_PROVIDER",
        direct_mode_status="NOT_SUPPORTED_BY_PROVIDER",
        direct_mode_control={},
        timeout_seconds=5,
    )


def _bindings():
    return {logical_id: _config(logical_id) for logical_id in PRIMARY_LLM_LOGICAL_IDS}


def _planned(adapter, authority, *, output="outputs"):
    return build_generation_planned_record(
        created_at="2026-08-24T00:00:00Z",
        block=GenerationBlock("without_context", adapter.config.logical_model_id),
        adapter=adapter,
        git={
            "commit": GIT_SHA,
            "branch": "sprint3",
            "worktree_clean": True,
            "worktree_diff_sha256": None,
        },
        sample_manifest_ref={
            "path": "artifacts/sample.json",
            "sha256": SHA_B,
            "artifact_id": SAMPLE_ID,
        },
        corpus_manifest_ref=None,
        retrieval=None,
        environment_sha256=stable_json_sha256({"python": "fixture"}),
        runtime_sha256=stable_json_sha256({"adapter": "fixture"}),
        hardware_summary="fixture",
        output_directory=output,
        evidence_authority_path=authority,
    )


def _repeatability_manifest(path: Path):
    entries = []
    for index in range(20):
        prompt = render_pubmedqa_prompt(question=f"Development question {index}?")
        entries.append(
            {
                "prompt_id": f"development-{index:02d}",
                "dataset": "hotpotqa",
                "evidence_role": "DEVELOPMENT",
                "sample_id": index,
                "prompt": prompt.provenance_payload(),
            }
        )
    write_repeatability_prompt_manifest(path, entries)


def _passing_gate(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, MakiConfig]]:
    monkeypatch.setenv("MAKI_API_KEY", "fixture-secret")
    prompt_manifest = tmp_path / "prompts.json"
    _repeatability_manifest(prompt_manifest)
    bindings = _bindings()

    def transport(**kwargs):
        question = kwargs["payload"]["messages"][1]["content"].splitlines()[1]
        return {
            "id": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"Decision: yes\nExplanation: {question}",
                    },
                }
            ],
        }

    adapters = {
        logical_id: CanonicalMakiAdapter(
            config,
            transport=transport,
            sleep=lambda _: None,
            clock=lambda: "2026-08-24T00:00:00Z",
        )
        for logical_id, config in bindings.items()
    }
    gate_path = tmp_path / "gate.json"
    gate = run_repeatability_gate(
        prompt_manifest_path=prompt_manifest,
        adapters=adapters,
        output_path=gate_path,
        created_at="2026-08-24T00:00:00Z",
    )
    assert gate["scientific_payload"]["all_primary_models_passed"] is True
    assert sum(
        len(prompt["calls"])
        for model in gate["scientific_payload"]["models"]
        for prompt in model["prompt_results"]
    ) == 180
    return gate_path, bindings


def test_matrix_is_three_shared_plus_twelve_context_blocks():
    blocks = canonical_pubmedqa_generation_matrix()
    assert len(blocks) == 15
    assert len([item for item in blocks if item.context_mode == "without_context"]) == 3
    assert len([item for item in blocks if item.context_mode == "with_context"]) == 12
    assert all(item.retriever is None for item in blocks[:3])
    assert expected_matrix_row_count() == 15000
    with pytest.raises(ValueError, match="retriever-independent"):
        GenerationBlock("without_context", "llama-3.3-70b", "bm25")


def test_repeatability_gate_binds_all_physical_runtime_identities(tmp_path, monkeypatch):
    gate_path, bindings = _passing_gate(tmp_path, monkeypatch)
    identities = {key: value.runtime_identity() for key, value in bindings.items()}
    require_passing_repeatability_gate(gate_path, model_runtime_identities=identities)
    changed = {key: dict(value) for key, value in identities.items()}
    changed["llama-3.3-70b"]["physical_model_id"] = "different"
    with pytest.raises(RepeatabilityGateError, match="differs"):
        require_passing_repeatability_gate(gate_path, model_runtime_identities=changed)

    failed = json.loads(gate_path.read_text(encoding="utf-8"))
    first_model = failed["scientific_payload"]["models"][0]
    for prompt_result in first_model["prompt_results"][:2]:
        prompt_result["calls"][2]["raw_content"] = "different"
        prompt_result["all_three_stripped_identical"] = False
    first_model["identical_prompt_count"] = 18
    first_model["passed"] = False
    failed["scientific_payload"]["all_primary_models_passed"] = False
    digest = stable_json_sha256(failed["scientific_payload"])
    failed["scientific_sha256"] = digest
    failed["gate_id"] = f"generation-repeatability-gate:sha256:{digest}"
    failed_path = tmp_path / "failed_gate.json"
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(RepeatabilityGateError, match="did not pass"):
        require_passing_repeatability_gate(
            failed_path, model_runtime_identities=identities
        )


def test_missing_only_resume_reuses_valid_row_and_completes_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "PUBMEDQA_EXPECTED_ROWS", 2)
    monkeypatch.setenv("MAKI_API_KEY", "fixture-secret")
    gate_path, bindings = _passing_gate(tmp_path / "gate-assets", monkeypatch)
    calls = 0

    def transport(**_):
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Decision: yes\nExplanation: Exact."},
                }
            ]
        }

    adapter = CanonicalMakiAdapter(
        bindings["llama-3.3-70b"],
        transport=transport,
        sleep=lambda _: None,
        clock=lambda: "2026-08-24T00:00:00Z",
    )
    authority = tmp_path / "authority.json"
    _authority(authority)
    planned = _planned(adapter, authority)
    registry = tmp_path / "registry.jsonl"
    append_run_record(registry, planned, evidence_authority_path=authority)
    running1 = _running_record(
        planned,
        attempt_count=1,
        started_at="2026-08-24T00:00:00Z",
        prior_failure_reason=None,
        evidence_authority_path=authority,
    )
    append_run_record(registry, running1, evidence_authority_path=authority)

    runtime = Runtime((Query(0, 10, "Q0?"), Query(1, 11, "Q1?")))
    environment = {"python": "fixture"}
    runtime_provenance = {"adapter": "fixture"}
    request, _ = _request_for_query(
        planned=planned,
        adapter=adapter,
        query=runtime.ordered_queries[0],
        selected_context=None,
    )
    existing = build_generation_artifact(
        request=request,
        status="OK",
        raw_content="Decision: no\nExplanation: Existing.",
        finish_reason="stop",
        provider_metadata={},
        parsed_output={"decision": "no"},
        attempts=[
            {
                "attempt": 1,
                "started_at": "2026-08-24T00:00:00Z",
                "completed_at": "2026-08-24T00:00:00Z",
                "outcome": "SUCCESS",
                "error_type": None,
                "error_message": None,
            }
        ],
        environment=environment,
        runtime=runtime_provenance,
        hardware_summary="fixture",
        created_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:00:00Z",
    )
    output_dir = tmp_path / "outputs"
    write_generation_artifact(existing, generation_artifact_path(output_dir, 0))
    terminal = execute_generation_block(
        planned_record=planned,
        runtime=runtime,
        adapter=adapter,
        repeatability_gate_path=gate_path,
        all_model_runtime_identities={
            key: value.runtime_identity() for key, value in bindings.items()
        },
        output_directory=output_dir,
        output_inventory_path=output_dir / "inventory.json",
        environment=environment,
        runtime_provenance=runtime_provenance,
        hardware_summary="fixture",
        registry_path=registry,
        evidence_authority_path=authority,
        repository_root=tmp_path,
        clock=lambda: "2026-08-24T00:00:01Z",
    )
    assert calls == 1
    assert terminal["execution"]["status"] == "COMPLETE"
    assert terminal["execution"]["attempt_count"] == 2
    assert terminal["output"]["completed_row_count"] == 2
    records = read_registry(registry, evidence_authority_path=authority)
    assert [item["execution"]["status"] for item in records] == [
        "PLANNED", "RUNNING", "RUNNING", "COMPLETE"
    ]


def test_exhausted_per_row_transport_becomes_terminal_error_not_failed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "PUBMEDQA_EXPECTED_ROWS", 1)
    monkeypatch.setenv("MAKI_API_KEY", "fixture-secret")
    gate_path, bindings = _passing_gate(tmp_path / "gate-assets", monkeypatch)
    calls = 0

    def transport(**_):
        nonlocal calls
        calls += 1
        raise MakiInfrastructureError("temporary provider failure")

    adapter = CanonicalMakiAdapter(
        bindings["llama-3.3-70b"],
        transport=transport,
        sleep=lambda _: None,
        clock=lambda: "2026-08-24T00:00:00Z",
    )
    authority = tmp_path / "authority.json"
    _authority(authority)
    planned = _planned(adapter, authority)
    output = tmp_path / "outputs"
    terminal = execute_generation_block(
        planned_record=planned,
        runtime=Runtime((Query(0, 1, "Q?"),)),
        adapter=adapter,
        repeatability_gate_path=gate_path,
        all_model_runtime_identities={
            key: value.runtime_identity() for key, value in bindings.items()
        },
        output_directory=output,
        output_inventory_path=output / "inventory.json",
        environment={"python": "fixture"},
        runtime_provenance={"adapter": "fixture"},
        hardware_summary="fixture",
        registry_path=tmp_path / "registry.jsonl",
        evidence_authority_path=authority,
        repository_root=tmp_path,
        clock=lambda: "2026-08-24T00:00:01Z",
    )
    assert calls == 3
    assert terminal["execution"]["status"] == "COMPLETE"
    assert terminal["output"]["failed_row_count"] == 1
    row = json.loads(generation_artifact_path(output, 0).read_text(encoding="utf-8"))
    assert row["observation"]["status"] == "ERROR"
    assert len(row["execution_provenance"]["attempts"]) == 3
