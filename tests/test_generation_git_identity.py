from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generation.cli_support import (  # noqa: E402
    canonical_generation_git_identity,
    git_registry_identity,
    pubmedqa_generation_replacement_output_directory,
    validate_pubmedqa_generation_replacement,
)
from scripts import run_pubmedqa_generation as generation_cli  # noqa: E402
from run_registry import (  # noqa: E402
    EVIDENCE_AUTHORITY_SCHEMA_VERSION,
    REGISTRY_HEADER,
    append_run_record,
    canonical_json,
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
REPLACEMENT_GIT_COMMIT = "d" * 40
CANONICAL_REGISTRY = Path("artifacts/run_registry/run_registry_v1.jsonl")
CANONICAL_AUTHORITY = Path(
    "artifacts/run_registry/evidence_manifest_authority_v1.json"
)
CANONICAL_OUTPUT_ROOT = Path("results/sprint3/raw/pubmedqa/generation")
DEFAULT_OUTPUT_DIRECTORY = (
    CANONICAL_OUTPUT_ROOT / "without_context" / "llama-3.3-70b"
)
WITH_CONTEXT_BM25_OUTPUT_DIRECTORY = (
    CANONICAL_OUTPUT_ROOT / "with_context" / "bm25" / "llama-3.3-70b"
)


@dataclass(frozen=True)
class IsolatedRepository:
    root: Path
    commit: str

    @property
    def registry(self) -> Path:
        return self.root / CANONICAL_REGISTRY

    @property
    def authority(self) -> Path:
        return self.root / CANONICAL_AUTHORITY


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> IsolatedRepository:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-b", "sprint3")
    _git(root, "config", "user.name", "Fixture User")
    _git(root, "config", "user.email", "fixture@example.test")

    registry = root / CANONICAL_REGISTRY
    registry.parent.mkdir(parents=True)
    registry.write_text(canonical_json(REGISTRY_HEADER) + "\n", encoding="utf-8")

    authority = root / CANONICAL_AUTHORITY
    authority.write_text(
        canonical_json(
            {
                "schema_version": EVIDENCE_AUTHORITY_SCHEMA_VERSION,
                "authorities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "src").mkdir()
    (root / "src/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs/frozen.json").write_text("{}\n", encoding="utf-8")

    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture baseline")
    commit = _git(root, "rev-parse", "HEAD")
    return IsolatedRepository(root=root, commit=commit)


def _planned_record(
    repository: IsolatedRepository,
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    logical_model_id: str = "llama-3.3-70b",
    context_mode: str = "without_context",
    retriever: str = "bm25",
    git_commit: str | None = None,
    parent_run_id: str | None = None,
) -> dict:
    has_context = context_mode == "with_context"
    return finalize_planned_record(
        {
            "schema_version": "sprint3.run-registry-record.v1",
            "created_at": "2026-08-30T00:00:00Z",
            "sprint": "sprint3",
            "stage": 1,
            "run_type": "GENERATION",
            "evidence_role": "SYNTHETIC",
            "origin": "CURRENT_PROTOCOL",
            "protocol_config_bundle_sha256": SHA_A,
            "git": {
                "commit": repository.commit if git_commit is None else git_commit,
                "branch": "sprint3",
                "worktree_clean": True,
                "worktree_diff_sha256": None,
            },
            "data": {
                "dataset": "pubmedqa",
                "split": "fixture",
                "source": "fixture/pubmedqa",
                "revision": "immutable-fixture-revision",
                "sample_manifest": {
                    "path": "artifacts/sample_manifests/pubmedqa.json",
                    "sha256": SHA_B,
                    "artifact_id": f"sample-manifest:sha256:{SHA_B}",
                },
                "corpus_manifest": (
                    {
                        "path": "artifacts/corpus_manifests/pubmedqa.json",
                        "sha256": SHA_C,
                        "artifact_id": f"corpus-manifest:sha256:{SHA_C}",
                    }
                    if has_context
                    else None
                ),
            },
            "retrieval": (
                {
                    "retriever": retriever,
                    "config_sha256": SHA_A,
                    "index": (
                        None
                        if retriever == "bm25"
                        else {
                            "path": "artifacts/indexes/pubmedqa.idx",
                            "sha256": SHA_A,
                            "artifact_id": f"index:sha256:{SHA_A}",
                        }
                    ),
                    "candidate_set": {
                        "path": "artifacts/candidates/pubmedqa_set.json",
                        "sha256": SHA_B,
                        "artifact_id": f"candidate-set:sha256:{SHA_B}",
                    },
                    "selected_context": {
                        "path": "artifacts/contexts/pubmedqa_top5.json",
                        "sha256": SHA_C,
                        "artifact_id": f"selected-context:sha256:{SHA_C}",
                    },
                    "candidate_pool": 20,
                    "top_k": 5,
                }
                if has_context
                else None
            ),
            "diversification": (
                {
                    "method": "none",
                    "parameters": {},
                    "config_sha256": stable_json_sha256({}),
                    "seed": None,
                }
                if has_context
                else None
            ),
            "context_mode": context_mode,
            "generation": {
                "llm_logical_id": logical_model_id,
                "provider": "fixture-provider",
                "physical_model_id": f"fixture/{logical_model_id}",
                "model_revision": None,
                "model_revision_kind": "NOT_PROVIDED_BY_PROVIDER",
                "prompt_sha256": SHA_A,
                "decoding_sha256": SHA_B,
            },
            "evaluation": None,
            "execution": {
                "environment_sha256": SHA_A,
                "runtime_sha256": SHA_B,
                "hardware_summary": "fixture-cpu",
                "started_at": None,
                "completed_at": None,
                "status": "PLANNED",
                "attempt_count": 0,
                "failure_reason": None,
                "parent_run_id": parent_run_id,
                "resume_of": None,
            },
            "output": {
                "expected_row_count": 1000,
                "completed_row_count": 0,
                "successful_row_count": 0,
                "failed_row_count": 0,
                "output_directory": output_directory.as_posix(),
                "partial_output_retained": False,
                "artifacts": [],
                "output_inventory_sha256": None,
                "raw_artifact_sha256": None,
            },
        },
        evidence_authority_path=repository.authority,
    )


def _append_planned(
    repository: IsolatedRepository,
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    logical_model_id: str = "llama-3.3-70b",
    context_mode: str = "without_context",
    retriever: str = "bm25",
    git_commit: str | None = None,
    parent_run_id: str | None = None,
) -> dict:
    planned = _planned_record(
        repository,
        output_directory=output_directory,
        logical_model_id=logical_model_id,
        context_mode=context_mode,
        retriever=retriever,
        git_commit=git_commit,
        parent_run_id=parent_run_id,
    )
    assert append_run_record(
        repository.registry,
        planned,
        evidence_authority_path=repository.authority,
    )
    return planned


def _running(repository: IsolatedRepository, planned: dict) -> dict:
    running = deepcopy(planned)
    running["execution"].update(
        {
            "started_at": "2026-08-30T00:00:01Z",
            "status": "RUNNING",
            "attempt_count": 1,
        }
    )
    running = validate_run_record(
        running, evidence_authority_path=repository.authority
    )
    assert append_run_record(
        repository.registry,
        running,
        evidence_authority_path=repository.authority,
    )
    return running


def _complete_with_error_rows(
    repository: IsolatedRepository,
    planned: dict,
    *,
    failed_row_count: int = 660,
) -> dict:
    running = _running(repository, planned)
    inventory_path = _write_output(
        repository,
        Path(planned["output"]["output_directory"])
        / "generation_output_inventory_v1.json",
    )
    artifact = output_artifact(
        inventory_path,
        repository_root=repository.root,
        row_count=1000,
        status_counts={"OK": 1000 - failed_row_count, "ERROR": failed_row_count},
    )
    complete = deepcopy(running)
    complete["execution"].update(
        {
            "completed_at": "2026-08-30T00:00:02Z",
            "status": "COMPLETE",
        }
    )
    complete["output"].update(
        {
            "completed_row_count": 1000,
            "successful_row_count": 1000 - failed_row_count,
            "failed_row_count": failed_row_count,
            "artifacts": [artifact],
            "output_inventory_sha256": output_inventory_sha256([artifact]),
            "raw_artifact_sha256": artifact["sha256"],
        }
    )
    complete = validate_run_record(
        complete,
        evidence_authority_path=repository.authority,
    )
    assert append_run_record(
        repository.registry,
        complete,
        evidence_authority_path=repository.authority,
    )
    return complete


def _eligible_replacement_parent(repository: IsolatedRepository) -> dict:
    planned = _append_planned(repository)
    return _complete_with_error_rows(repository, planned)


def _append_replacement(
    repository: IsolatedRepository,
    parent: dict,
    *,
    output_directory: Path | None = None,
    logical_model_id: str = "llama-3.3-70b",
) -> dict:
    canonical_output = pubmedqa_generation_replacement_output_directory(
        parent_run_id=parent["run_id"],
        replacement_git_commit=REPLACEMENT_GIT_COMMIT,
    )
    return _append_planned(
        repository,
        output_directory=canonical_output if output_directory is None else output_directory,
        logical_model_id=logical_model_id,
        git_commit=REPLACEMENT_GIT_COMMIT,
        parent_run_id=parent["run_id"],
    )


def _write_output(
    repository: IsolatedRepository,
    relative_path: Path,
    *,
    content: str = "{}\n",
) -> Path:
    path = repository.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _identity(repository: IsolatedRepository) -> dict:
    return canonical_generation_git_identity(repository_root=repository.root)


def _assert_clean(identity: dict, repository: IsolatedRepository) -> None:
    assert identity == {
        "commit": repository.commit,
        "branch": "sprint3",
        "worktree_clean": True,
        "worktree_diff_sha256": None,
    }


def _assert_dirty(identity: dict, repository: IsolatedRepository) -> None:
    assert identity["commit"] == repository.commit
    assert identity["branch"] == "sprint3"
    assert identity["worktree_clean"] is False
    assert len(identity["worktree_diff_sha256"]) == 64


def test_strict_git_identity_still_rejects_governed_runtime_dirtiness(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")

    _assert_dirty(git_registry_identity(repository.root), repository)


def test_canonical_generation_identity_accepts_exactly_clean_worktree(
    repository: IsolatedRepository,
) -> None:
    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_accepts_valid_append_only_registry(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)

    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_accepts_registered_samples_and_inventory(
    repository: IsolatedRepository,
) -> None:
    planned = _append_planned(repository)
    _running(repository, planned)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0999.json")
    _write_output(
        repository,
        DEFAULT_OUTPUT_DIRECTORY / "generation_output_inventory_v1.json",
    )

    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_accepts_exact_without_context_directory(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository, output_directory=DEFAULT_OUTPUT_DIRECTORY)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")

    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_accepts_exact_with_context_directory(
    repository: IsolatedRepository,
) -> None:
    _append_planned(
        repository,
        output_directory=WITH_CONTEXT_BM25_OUTPUT_DIRECTORY,
        context_mode="with_context",
        retriever="bm25",
    )
    _write_output(
        repository,
        WITH_CONTEXT_BM25_OUTPUT_DIRECTORY / "sample_0000.json",
    )

    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_accepts_registered_replacement_directory(
    repository: IsolatedRepository,
) -> None:
    parent = _eligible_replacement_parent(repository)
    replacement = _append_replacement(repository, parent)
    replacement_output = Path(replacement["output"]["output_directory"])
    _write_output(repository, replacement_output / "sample_0000.json")

    assert replacement["run_id"] != parent["run_id"]
    assert replacement["execution"]["parent_run_id"] == parent["run_id"]
    assert replacement_output != Path(parent["output"]["output_directory"])
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )
    assert validate_pubmedqa_generation_replacement(records[:-1], replacement) == parent
    _assert_clean(_identity(repository), repository)


def test_replacement_lineage_rejects_parent_commit_reuse(
    repository: IsolatedRepository,
) -> None:
    parent = _eligible_replacement_parent(repository)
    output = pubmedqa_generation_replacement_output_directory(
        parent_run_id=parent["run_id"],
        replacement_git_commit=parent["git"]["commit"],
    )
    replacement = _planned_record(
        repository,
        output_directory=output,
        git_commit=parent["git"]["commit"],
        parent_run_id=parent["run_id"],
    )
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )

    with pytest.raises(ValueError, match="later amendment Git commit"):
        validate_pubmedqa_generation_replacement(records, replacement)


def test_canonical_generation_identity_rejects_arbitrary_replacement_directory(
    repository: IsolatedRepository,
) -> None:
    parent = _eligible_replacement_parent(repository)
    arbitrary = CANONICAL_OUTPUT_ROOT / "replacements" / parent["run_id"] / "arbitrary"
    _append_replacement(repository, parent, output_directory=arbitrary)
    _write_output(repository, arbitrary / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_replacement_lineage_rejects_missing_parent(
    repository: IsolatedRepository,
) -> None:
    missing_parent = "run-missing-parent-" + "1" * 24
    output = pubmedqa_generation_replacement_output_directory(
        parent_run_id=missing_parent,
        replacement_git_commit=REPLACEMENT_GIT_COMMIT,
    )
    replacement = _append_planned(
        repository,
        output_directory=output,
        git_commit=REPLACEMENT_GIT_COMMIT,
        parent_run_id=missing_parent,
    )
    _write_output(repository, output / "sample_0000.json")
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )

    with pytest.raises(ValueError, match="absent"):
        validate_pubmedqa_generation_replacement(records, replacement)
    _assert_dirty(_identity(repository), repository)


def test_replacement_lineage_rejects_nonterminal_parent(
    repository: IsolatedRepository,
) -> None:
    parent = _append_planned(repository)
    output = pubmedqa_generation_replacement_output_directory(
        parent_run_id=parent["run_id"],
        replacement_git_commit=REPLACEMENT_GIT_COMMIT,
    )
    replacement = _planned_record(
        repository,
        output_directory=output,
        git_commit=REPLACEMENT_GIT_COMMIT,
        parent_run_id=parent["run_id"],
    )
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )

    with pytest.raises(ValueError, match="terminal"):
        validate_pubmedqa_generation_replacement(records, replacement)


def test_replacement_lineage_rejects_parent_without_error_rows(
    repository: IsolatedRepository,
) -> None:
    planned = _append_planned(repository)
    parent = _complete_with_error_rows(repository, planned, failed_row_count=0)
    output = pubmedqa_generation_replacement_output_directory(
        parent_run_id=parent["run_id"],
        replacement_git_commit=REPLACEMENT_GIT_COMMIT,
    )
    replacement = _planned_record(
        repository,
        output_directory=output,
        git_commit=REPLACEMENT_GIT_COMMIT,
        parent_run_id=parent["run_id"],
    )
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )

    with pytest.raises(ValueError, match="failed/error rows"):
        validate_pubmedqa_generation_replacement(records, replacement)


def test_replacement_lineage_rejects_mismatched_scientific_block(
    repository: IsolatedRepository,
) -> None:
    parent = _eligible_replacement_parent(repository)
    replacement = _append_replacement(
        repository,
        parent,
        logical_model_id="gemma4-26b",
    )
    output = Path(replacement["output"]["output_directory"])
    _write_output(repository, output / "sample_0000.json")
    records = read_registry(
        repository.registry,
        evidence_authority_path=repository.authority,
    )

    with pytest.raises(ValueError, match="scientific identity differs"):
        validate_pubmedqa_generation_replacement(records[:-1], replacement)
    _assert_dirty(_identity(repository), repository)


def test_registered_replacement_does_not_exempt_unrelated_source_edit(
    repository: IsolatedRepository,
) -> None:
    parent = _eligible_replacement_parent(repository)
    replacement = _append_replacement(repository, parent)
    output = Path(replacement["output"]["output_directory"])
    _write_output(repository, output / "sample_0000.json")
    (repository.root / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_registered_arbitrary_descendant(
    repository: IsolatedRepository,
) -> None:
    arbitrary = CANONICAL_OUTPUT_ROOT / "arbitrary-block"
    _append_planned(repository, output_directory=arbitrary)
    _write_output(repository, arbitrary / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_registered_extra_nested_directory(
    repository: IsolatedRepository,
) -> None:
    nested = DEFAULT_OUTPUT_DIRECTORY / "nested"
    _append_planned(repository, output_directory=nested)
    _write_output(repository, nested / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_wrong_logical_id_directory(
    repository: IsolatedRepository,
) -> None:
    wrong_logical_id = CANONICAL_OUTPUT_ROOT / "without_context" / "gemma4-26b"
    _append_planned(repository, output_directory=wrong_logical_id)
    _write_output(repository, wrong_logical_id / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_unsupported_logical_id(
    repository: IsolatedRepository,
) -> None:
    unsupported = CANONICAL_OUTPUT_ROOT / "without_context" / "fixture-llm"
    _append_planned(
        repository,
        output_directory=unsupported,
        logical_model_id="fixture-llm",
    )
    _write_output(repository, unsupported / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_wrong_retriever_directory(
    repository: IsolatedRepository,
) -> None:
    wrong_retriever = (
        CANONICAL_OUTPUT_ROOT / "with_context" / "dpr" / "llama-3.3-70b"
    )
    _append_planned(
        repository,
        output_directory=wrong_retriever,
        context_mode="with_context",
        retriever="bm25",
    )
    _write_output(repository, wrong_retriever / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_unsupported_retriever(
    repository: IsolatedRepository,
) -> None:
    unsupported = (
        CANONICAL_OUTPUT_ROOT / "with_context" / "splade" / "llama-3.3-70b"
    )
    _append_planned(
        repository,
        output_directory=unsupported,
        context_mode="with_context",
        retriever="splade",
    )
    _write_output(repository, unsupported / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


@pytest.mark.parametrize("lifecycle", ["RUNNING", "FAILED"])
def test_canonical_generation_identity_accepts_registered_partial_sample_set(
    repository: IsolatedRepository,
    lifecycle: str,
) -> None:
    planned = _append_planned(repository)
    running = _running(repository, planned)
    first = _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")
    second = _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0042.json")
    if lifecycle == "FAILED":
        artifacts = [
            output_artifact(
                path,
                repository_root=repository.root,
                row_count=1,
                status_counts={"OK": 1},
            )
            for path in (first, second)
        ]
        failed = deepcopy(running)
        failed["execution"].update(
            {
                "completed_at": "2026-08-30T00:00:02Z",
                "status": "FAILED",
                "failure_reason": "fixture infrastructure failure",
            }
        )
        failed["output"].update(
            {
                "completed_row_count": 2,
                "successful_row_count": 2,
                "partial_output_retained": True,
                "artifacts": artifacts,
                "output_inventory_sha256": output_inventory_sha256(artifacts),
                "raw_artifact_sha256": SHA_A,
            }
        )
        failed = validate_run_record(
            failed, evidence_authority_path=repository.authority
        )
        assert append_run_record(
            repository.registry,
            failed,
            evidence_authority_path=repository.authority,
        )

    _assert_clean(_identity(repository), repository)


def test_canonical_generation_identity_rejects_registry_truncation(
    repository: IsolatedRepository,
) -> None:
    committed = repository.registry.read_bytes()
    repository.registry.write_bytes(committed[:-1])

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_rewritten_registry_prefix(
    repository: IsolatedRepository,
) -> None:
    committed = repository.registry.read_bytes()
    repository.registry.write_bytes(committed.replace(b"{", b"{ ", 1))

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_malformed_appended_registry_record(
    repository: IsolatedRepository,
) -> None:
    with repository.registry.open("ab") as handle:
        handle.write(b"{not-json}\n")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_staged_registry_modification(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    _git(repository.root, "add", CANONICAL_REGISTRY.as_posix())

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_tracked_source_edit(
    repository: IsolatedRepository,
) -> None:
    (repository.root / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_staged_config_change(
    repository: IsolatedRepository,
) -> None:
    config = repository.root / "configs/frozen.json"
    config.write_text('{"changed":true}\n', encoding="utf-8")
    _git(repository.root, "add", "configs/frozen.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_untracked_notebook(
    repository: IsolatedRepository,
) -> None:
    _write_output(repository, Path("analysis/untracked.ipynb"))

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_untracked_selected_context(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    _write_output(
        repository,
        Path("artifacts/selected_contexts/pubmedqa/bm25/sample_0000.json"),
    )

    _assert_dirty(_identity(repository), repository)


@pytest.mark.parametrize(
    "unexpected",
    [
        Path("notes.json"),
        Path("nested/sample_0000.json"),
        Path(".sample_0000.json.temporary"),
    ],
)
def test_canonical_generation_identity_rejects_unexpected_file_in_registered_directory(
    repository: IsolatedRepository,
    unexpected: Path,
) -> None:
    _append_planned(repository)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / unexpected)

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_sample_1000(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_1000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_near_prefix_generation_path(
    repository: IsolatedRepository,
) -> None:
    near_prefix = Path(
        "results/sprint3/raw/pubmedqa/generation-near/without_context/llama-3.3-70b"
    )
    _append_planned(repository, output_directory=near_prefix)
    _write_output(repository, near_prefix / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_unregistered_generation_directory(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    unregistered = CANONICAL_OUTPUT_ROOT / "without_context" / "gemma4-26b"
    _write_output(repository, unregistered / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_registered_directory_outside_root(
    repository: IsolatedRepository,
) -> None:
    outside = Path("results/sprint3/raw/pubmedqa/not-generation/fixture-block")
    _append_planned(repository, output_directory=outside)
    _write_output(repository, outside / "sample_0000.json")

    _assert_dirty(_identity(repository), repository)


def test_second_independent_preflight_accepts_complete_block_evidence_at_same_commit(
    repository: IsolatedRepository,
) -> None:
    planned = _append_planned(repository)
    running = _running(repository, planned)
    _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")
    inventory_path = _write_output(
        repository,
        DEFAULT_OUTPUT_DIRECTORY / "generation_output_inventory_v1.json",
    )
    inventory = output_artifact(
        inventory_path,
        repository_root=repository.root,
        row_count=1000,
        status_counts={"OK": 1000},
    )
    complete = deepcopy(running)
    complete["execution"].update(
        {
            "completed_at": "2026-08-30T00:00:02Z",
            "status": "COMPLETE",
        }
    )
    complete["output"].update(
        {
            "completed_row_count": 1000,
            "successful_row_count": 1000,
            "artifacts": [inventory],
            "output_inventory_sha256": output_inventory_sha256([inventory]),
            "raw_artifact_sha256": inventory["sha256"],
        }
    )
    complete = validate_run_record(
        complete, evidence_authority_path=repository.authority
    )
    assert append_run_record(
        repository.registry,
        complete,
        evidence_authority_path=repository.authority,
    )

    first_preflight = _identity(repository)
    second_preflight = _identity(repository)
    _assert_clean(first_preflight, repository)
    _assert_clean(second_preflight, repository)
    assert first_preflight["commit"] == second_preflight["commit"] == repository.commit


def test_canonical_generation_identity_rejects_invalid_lifecycle_appended_by_hand(
    repository: IsolatedRepository,
) -> None:
    planned = _append_planned(repository)
    inventory_path = _write_output(
        repository,
        DEFAULT_OUTPUT_DIRECTORY / "generation_output_inventory_v1.json",
    )
    inventory = output_artifact(
        inventory_path,
        repository_root=repository.root,
        row_count=1000,
        status_counts={"OK": 1000},
    )
    complete = deepcopy(planned)
    complete["execution"].update(
        {
            "started_at": "2026-08-30T00:00:01Z",
            "completed_at": "2026-08-30T00:00:02Z",
            "status": "COMPLETE",
            "attempt_count": 1,
        }
    )
    complete["output"].update(
        {
            "completed_row_count": 1000,
            "successful_row_count": 1000,
            "artifacts": [inventory],
            "output_inventory_sha256": output_inventory_sha256([inventory]),
            "raw_artifact_sha256": inventory["sha256"],
        }
    )
    complete = validate_run_record(
        complete, evidence_authority_path=repository.authority
    )
    with repository.registry.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(complete) + "\n")

    _assert_dirty(_identity(repository), repository)


def test_canonical_generation_identity_rejects_staged_generation_output(
    repository: IsolatedRepository,
) -> None:
    _append_planned(repository)
    output = _write_output(repository, DEFAULT_OUTPUT_DIRECTORY / "sample_0000.json")
    _git(repository.root, "add", output.relative_to(repository.root).as_posix())

    _assert_dirty(_identity(repository), repository)


@pytest.mark.parametrize(
    ("action", "registered_planned", "custom_path", "expected_identity"),
    [
        ("run", None, None, "canonical"),
        ("replace", None, None, "canonical"),
        ("resume", {}, None, "strict"),
        ("status", {}, None, "strict"),
        ("run", None, "registry", "strict"),
        ("run", None, "authority", "strict"),
    ],
)
def test_generation_cli_selects_scoped_identity_only_for_fresh_canonical_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    registered_planned: dict | None,
    custom_path: str | None,
    expected_identity: str,
) -> None:
    class IdentitySelected(Exception):
        pass

    args = SimpleNamespace(
        action=action,
        cache_dir=None,
        context_mode="without_context",
        evidence_authority=(
            tmp_path / "alternate-authority.json"
            if custom_path == "authority"
            else generation_cli.DEFAULT_EVIDENCE_AUTHORITY_PATH
        ),
        llm="llama-3.3-70b",
        model_bindings=Path("unused-bindings.json"),
        registry=(
            tmp_path / "alternate-registry.jsonl"
            if custom_path == "registry"
            else generation_cli.DEFAULT_REGISTRY_PATH
        ),
        retriever=None,
    )
    monkeypatch.setattr(
        generation_cli,
        "load_model_bindings",
        lambda _path: {"llama-3.3-70b": object()},
    )
    monkeypatch.setattr(
        generation_cli,
        "adapter_from_bindings",
        lambda _bindings, _llm: object(),
    )
    monkeypatch.setattr(
        generation_cli,
        "load_pubmedqa_runtime_local_only",
        lambda *, cache_dir: object(),
    )

    def select(identity: str):
        raise IdentitySelected(identity)

    monkeypatch.setattr(
        generation_cli,
        "canonical_generation_git_identity",
        lambda: select("canonical"),
    )
    monkeypatch.setattr(
        generation_cli,
        "git_registry_identity",
        lambda: select("strict"),
    )

    with pytest.raises(IdentitySelected, match=expected_identity):
        generation_cli._build_inputs(
            args,
            registered_planned=registered_planned,
        )
