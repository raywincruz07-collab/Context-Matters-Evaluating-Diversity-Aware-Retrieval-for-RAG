import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import generation.cli_support as cli_support  # noqa: E402
from generation.cli_support import neural_index_artifact_ref  # noqa: E402
from retrieval_artifacts.colbert_cache_identity import (  # noqa: E402
    fingerprint_colbert_index_directory,
)
from run_registry import RunRecordValidationError, artifact_ref  # noqa: E402
from scripts import run_pubmedqa_generation as generation_cli  # noqa: E402


SCIENTIFIC_SHA = "a" * 64
OTHER_SHA = "b" * 64
CANDIDATE_SET_ID = f"candidate-set:sha256:{'c' * 64}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("retriever", ("dpr", "contriever"))
def test_file_neural_index_reference_preserves_file_sha_semantics(
    tmp_path: Path, retriever: str
) -> None:
    repository = tmp_path / "repository"
    index = repository / "artifacts/indexes/index.faiss"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"fixture FAISS index")
    physical_sha256 = _file_sha256(index)

    reference = neural_index_artifact_ref(
        repository_root=repository,
        retriever=retriever,
        artifact_path=index,
        index_fingerprint_sha256=SCIENTIFIC_SHA,
        index_artifact_sha256=physical_sha256,
    )

    assert reference == {
        "path": "artifacts/indexes/index.faiss",
        "sha256": physical_sha256,
        "artifact_id": f"index:sha256:{SCIENTIFIC_SHA}",
    }


@pytest.mark.parametrize("retriever", ("dpr", "contriever"))
def test_file_neural_index_reference_rejects_physical_sha_mismatch(
    tmp_path: Path, retriever: str
) -> None:
    index = tmp_path / "repository/index.faiss"
    index.parent.mkdir()
    index.write_bytes(b"fixture FAISS index")

    with pytest.raises(ValueError, match="index_artifact_sha256"):
        neural_index_artifact_ref(
            repository_root=index.parent,
            retriever=retriever,
            artifact_path=index,
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


def test_colbert_index_reference_uses_whole_directory_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    index = repository / "artifacts/indexes/colbert/indexes/canonical"
    (index / "nested").mkdir(parents=True)
    (index / "metadata.json").write_text("{}\n", encoding="utf-8")
    (index / "nested/0.pt").write_bytes(b"PLAID fixture")
    physical_sha256 = fingerprint_colbert_index_directory(index)
    calls = []

    def fingerprint_spy(path: Path) -> str:
        calls.append(path)
        return physical_sha256

    monkeypatch.setattr(
        cli_support, "fingerprint_colbert_index_directory", fingerprint_spy
    )
    reference = neural_index_artifact_ref(
        repository_root=repository,
        retriever="colbertv2",
        artifact_path=index,
        index_fingerprint_sha256=SCIENTIFIC_SHA,
        index_artifact_sha256=physical_sha256,
    )

    assert calls == [index.resolve()]
    assert reference == {
        "path": "artifacts/indexes/colbert/indexes/canonical",
        "sha256": physical_sha256,
        "artifact_id": f"index:sha256:{SCIENTIFIC_SHA}",
    }


def test_colbert_index_reference_rejects_directory_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    index = tmp_path / "repository/index"
    index.mkdir(parents=True)
    (index / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="index_artifact_sha256"):
        neural_index_artifact_ref(
            repository_root=index.parent,
            retriever="colbertv2",
            artifact_path=index,
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


def test_colbert_index_reference_rejects_regular_file(tmp_path: Path) -> None:
    index = tmp_path / "repository/index"
    index.parent.mkdir()
    index.write_bytes(b"not a PLAID directory")

    with pytest.raises(ValueError, match="must be a directory"):
        neural_index_artifact_ref(
            repository_root=index.parent,
            retriever="colbertv2",
            artifact_path=index,
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


@pytest.mark.parametrize("retriever", ("dpr", "contriever"))
def test_file_neural_index_reference_rejects_directory(
    tmp_path: Path, retriever: str
) -> None:
    index = tmp_path / "repository/index"
    index.mkdir(parents=True)

    with pytest.raises(RunRecordValidationError, match="artifact file"):
        neural_index_artifact_ref(
            repository_root=index.parent,
            retriever=retriever,
            artifact_path=index,
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


@pytest.mark.parametrize("retriever", ("dpr", "colbertv2"))
def test_neural_index_reference_rejects_path_outside_repository(
    tmp_path: Path, retriever: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    if retriever == "colbertv2":
        outside.mkdir()
    else:
        outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside repository_root"):
        neural_index_artifact_ref(
            repository_root=repository,
            retriever=retriever,
            artifact_path=outside,
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("index_fingerprint_sha256", None),
        ("index_fingerprint_sha256", "invalid"),
        ("index_artifact_sha256", None),
        ("index_artifact_sha256", "invalid"),
    ),
)
def test_neural_index_reference_rejects_missing_or_invalid_declared_sha(
    tmp_path: Path, field: str, value: object
) -> None:
    index = tmp_path / "index.faiss"
    index.write_bytes(b"fixture")
    arguments = {
        "repository_root": tmp_path,
        "retriever": "dpr",
        "artifact_path": index,
        "index_fingerprint_sha256": SCIENTIFIC_SHA,
        "index_artifact_sha256": _file_sha256(index),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        neural_index_artifact_ref(**arguments)


def test_neural_index_reference_rejects_unsupported_retriever(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported neural retriever"):
        neural_index_artifact_ref(
            repository_root=tmp_path,
            retriever="bm25",
            artifact_path=tmp_path / "unused",
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        )


def test_generic_run_registry_artifact_reference_remains_file_only(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "index"
    directory.mkdir()

    with pytest.raises(RunRecordValidationError, match="artifact file"):
        artifact_ref(directory, repository_root=tmp_path)


def _generation_args(**changes: object) -> SimpleNamespace:
    values = {
        "action": "run",
        "cache_dir": None,
        "candidate_directory": None,
        "candidate_set": None,
        "context_mode": "without_context",
        "evidence_authority": Path("/tmp/fixture-authority.json"),
        "index_artifact": None,
        "llm": "llama-3.3-70b",
        "model_bindings": Path("unused-bindings.json"),
        "output_directory": generation_cli.REPOSITORY_ROOT / "results/fixture",
        "output_inventory": None,
        "registry": Path("/tmp/fixture-registry.jsonl"),
        "repeatability_gate": Path("unused-gate.json"),
        "retriever": None,
        "selected_context_directory": None,
        "selected_context_set": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _patch_build_inputs_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = SimpleNamespace(
        sample_manifest=SimpleNamespace(
            manifest_id=f"sample-manifest:sha256:{SCIENTIFIC_SHA}"
        ),
        corpus_manifest=SimpleNamespace(
            corpus_manifest_id=f"corpus-manifest:sha256:{OTHER_SHA}"
        ),
    )
    candidate = SimpleNamespace(
        retriever=SimpleNamespace(
            retriever_name="bm25",
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=None,
        ),
        scientific_payload=lambda: {"retriever": {"retriever_name": "bm25"}},
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
        lambda *, cache_dir: runtime,
    )
    monkeypatch.setattr(
        generation_cli,
        "git_registry_identity",
        lambda: {
            "commit": "1" * 40,
            "branch": "sprint3",
            "worktree_clean": True,
            "worktree_diff_sha256": None,
        },
    )
    monkeypatch.setattr(
        generation_cli,
        "artifact_ref",
        lambda path, *, repository_root, artifact_id=None: {
            "path": Path(path).name,
            "sha256": SCIENTIFIC_SHA,
            "artifact_id": artifact_id,
        },
    )
    monkeypatch.setattr(
        generation_cli,
        "candidate_set_artifact_ref",
        lambda path, *, repository_root: {
            "path": Path(path).name,
            "sha256": OTHER_SHA,
            "artifact_id": CANDIDATE_SET_ID,
        },
    )
    monkeypatch.setattr(
        generation_cli, "read_candidate_artifact", lambda _path: candidate
    )
    monkeypatch.setattr(
        generation_cli,
        "read_selected_context_set",
        lambda _path: {
            "selected_context_set_id": f"selected-context:sha256:{OTHER_SHA}",
            "scientific_payload": {"candidate_set_id": CANDIDATE_SET_ID},
        },
    )
    monkeypatch.setattr(
        generation_cli,
        "provenance_hashes",
        lambda _adapter: ({}, {}, SCIENTIFIC_SHA, OTHER_SHA),
    )
    monkeypatch.setattr(
        generation_cli,
        "build_generation_planned_record",
        lambda **arguments: arguments,
    )


def test_build_inputs_preserves_without_context_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build_inputs_dependencies(monkeypatch)

    planned = generation_cli._build_inputs(_generation_args())[4]

    assert planned["corpus_manifest_ref"] is None
    assert planned["retrieval"] is None


def test_build_inputs_preserves_bm25_null_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build_inputs_dependencies(monkeypatch)
    monkeypatch.setattr(
        generation_cli,
        "neural_index_artifact_ref",
        lambda **_arguments: pytest.fail("BM25 must not build a neural index ref"),
    )
    args = _generation_args(
        candidate_directory=Path("candidate-directory"),
        candidate_set=Path("candidate-set.json"),
        context_mode="with_context",
        retriever="bm25",
        selected_context_directory=Path("selected-context-directory"),
        selected_context_set=Path("selected-context-set.json"),
    )

    planned = generation_cli._build_inputs(args)[4]

    assert planned["retrieval"]["index"] is None


def test_build_inputs_uses_candidate_declared_colbert_index_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build_inputs_dependencies(monkeypatch)
    candidate = SimpleNamespace(
        retriever=SimpleNamespace(
            retriever_name="colbertv2",
            index_fingerprint_sha256=SCIENTIFIC_SHA,
            index_artifact_sha256=OTHER_SHA,
        ),
        scientific_payload=lambda: {
            "retriever": {"retriever_name": "colbertv2"}
        },
    )
    monkeypatch.setattr(
        generation_cli, "read_candidate_artifact", lambda _path: candidate
    )
    calls = []

    def index_reference(**arguments: object) -> dict[str, object]:
        calls.append(arguments)
        return {
            "path": "artifacts/indexes/colbert/canonical",
            "sha256": OTHER_SHA,
            "artifact_id": f"index:sha256:{SCIENTIFIC_SHA}",
        }

    monkeypatch.setattr(
        generation_cli, "neural_index_artifact_ref", index_reference
    )
    index = Path("artifacts/indexes/colbert/canonical")
    args = _generation_args(
        candidate_directory=Path("candidate-directory"),
        candidate_set=Path("candidate-set.json"),
        context_mode="with_context",
        index_artifact=index,
        retriever="colbertv2",
        selected_context_directory=Path("selected-context-directory"),
        selected_context_set=Path("selected-context-set.json"),
    )

    planned = generation_cli._build_inputs(args)[4]

    assert calls == [
        {
            "repository_root": generation_cli.REPOSITORY_ROOT,
            "retriever": "colbertv2",
            "artifact_path": index,
            "index_fingerprint_sha256": SCIENTIFIC_SHA,
            "index_artifact_sha256": OTHER_SHA,
        }
    ]
    assert planned["retrieval"]["index"] == {
        "path": "artifacts/indexes/colbert/canonical",
        "sha256": OTHER_SHA,
        "artifact_id": f"index:sha256:{SCIENTIFIC_SHA}",
    }


@pytest.mark.parametrize(
    ("context_mode", "retriever", "message"),
    (
        ("with_context", "bm25", "BM25 must not carry"),
        ("without_context", None, "WITHOUT_CONTEXT forbids"),
    ),
)
def test_build_inputs_preserves_non_neural_index_rejection(
    monkeypatch: pytest.MonkeyPatch,
    context_mode: str,
    retriever: str | None,
    message: str,
) -> None:
    _patch_build_inputs_dependencies(monkeypatch)
    changes = {
        "context_mode": context_mode,
        "retriever": retriever,
        "index_artifact": Path("forbidden-index"),
    }
    if context_mode == "with_context":
        changes.update(
            candidate_directory=Path("candidate-directory"),
            candidate_set=Path("candidate-set.json"),
            selected_context_directory=Path("selected-context-directory"),
            selected_context_set=Path("selected-context-set.json"),
        )

    with pytest.raises(ValueError, match=message):
        generation_cli._build_inputs(_generation_args(**changes))
