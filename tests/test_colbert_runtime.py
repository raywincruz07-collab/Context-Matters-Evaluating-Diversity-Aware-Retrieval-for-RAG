from dataclasses import FrozenInstanceError, asdict, replace
import inspect
from pathlib import Path
import subprocess
from types import SimpleNamespace
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrievers.colbert_runtime as runtime_module
from retrievers.colbert_checkpoint import ResolvedColBERTCheckpoint
from retrievers.colbert_config import COLBERT_CONFIG
from retrievers.colbert_runtime import (
    build_stanford_colbert_config,
    initialize_colbert_checkpoint,
    validate_effective_stanford_config,
)


REVISION = "0855eac81381e0323a846f1ed7d8452d4c648b50"
RUNTIME_VERSIONS = {
    "colbert-ai": "0.2.22",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
}


class FakeStanfordConfig:
    def __init__(self, **kwargs):
        self.constructor_kwargs = dict(kwargs)
        for name, value in kwargs.items():
            setattr(self, name, value)


def resolved(tmp_path: Path, *, checkpoint_id=None, revision=REVISION, exists=True):
    snapshot = tmp_path / "snapshots" / REVISION
    if exists:
        snapshot.mkdir(parents=True)
    return ResolvedColBERTCheckpoint(
        checkpoint_id=(
            COLBERT_CONFIG.checkpoint_id if checkpoint_id is None else checkpoint_id
        ),
        checkpoint_revision=revision,
        snapshot_path=snapshot,
    )


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(
        runtime_module.importlib_metadata,
        "version",
        lambda distribution: RUNTIME_VERSIONS[distribution],
    )
    monkeypatch.setattr(
        runtime_module,
        "_stanford_config_class",
        lambda: FakeStanfordConfig,
    )


def expected_values(snapshot: Path):
    return {
        "checkpoint": str(snapshot.resolve()),
        "dim": 128,
        "query_maxlen": 32,
        "doc_maxlen": 180,
        "similarity": "cosine",
        "interaction": "colbert",
        "mask_punctuation": True,
        "attend_to_mask_tokens": False,
        "index_bsize": 64,
        "nbits": 2,
        "kmeans_niters": 4,
        "nranks": 1,
        "ncells": 2,
        "centroid_score_threshold": 0.45,
        "ndocs": 1024,
    }


@pytest.mark.parametrize("distribution", RUNTIME_VERSIONS)
def test_wrong_runtime_distribution_version_is_rejected(
    tmp_path, monkeypatch, distribution
):
    versions = dict(RUNTIME_VERSIONS)
    versions[distribution] = "wrong-version"
    monkeypatch.setattr(
        runtime_module.importlib_metadata,
        "version",
        lambda name: versions[name],
    )

    with pytest.raises(
        RuntimeError,
        match=rf"{distribution} version mismatch: expected .* got wrong-version",
    ):
        build_stanford_colbert_config(resolved(tmp_path))


@pytest.mark.parametrize("missing_distribution", RUNTIME_VERSIONS)
def test_missing_runtime_distribution_is_rejected_clearly(
    tmp_path, monkeypatch, missing_distribution
):
    def version(distribution):
        if distribution == missing_distribution:
            raise runtime_module.importlib_metadata.PackageNotFoundError(distribution)
        return RUNTIME_VERSIONS[distribution]

    monkeypatch.setattr(runtime_module.importlib_metadata, "version", version)

    with pytest.raises(
        RuntimeError,
        match=rf"{missing_distribution}==.* is not installed",
    ):
        build_stanford_colbert_config(resolved(tmp_path))


def test_exact_runtime_distribution_names_are_checked(tmp_path, monkeypatch):
    calls = []

    def version(distribution):
        calls.append(distribution)
        return RUNTIME_VERSIONS[distribution]

    monkeypatch.setattr(runtime_module.importlib_metadata, "version", version)
    build_stanford_colbert_config(resolved(tmp_path))
    assert calls == ["colbert-ai", "transformers", "huggingface-hub"]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"checkpoint_id": "wrong/checkpoint"}, "checkpoint ID"),
        ({"revision": "0" * 40}, "checkpoint revision"),
        ({"exists": False}, "must be a directory"),
    ],
)
def test_resolved_checkpoint_binding_is_required(tmp_path, change, message):
    with pytest.raises((ValueError, NotADirectoryError), match=message):
        build_stanford_colbert_config(resolved(tmp_path, **change))


def test_every_frozen_runtime_field_is_explicitly_translated(tmp_path):
    checkpoint = resolved(tmp_path)
    stanford = build_stanford_colbert_config(checkpoint)

    assert stanford.constructor_kwargs == expected_values(checkpoint.snapshot_path)
    assert stanford.checkpoint == str(checkpoint.snapshot_path.resolve())
    assert stanford.interaction == "colbert"
    assert COLBERT_CONFIG.interaction == "ColBERT late interaction / MaxSim"
    assert "candidate_pool_size" not in stanford.constructor_kwargs
    assert "seed" not in stanford.constructor_kwargs
    assert "ranking" not in stanford.constructor_kwargs
    assert "post_filtering" not in stanford.constructor_kwargs
    assert "post_reranking" not in stanford.constructor_kwargs
    assert "tie_manipulation" not in stanford.constructor_kwargs


def test_scientific_config_is_not_mutated(tmp_path):
    before = asdict(COLBERT_CONFIG)
    build_stanford_colbert_config(resolved(tmp_path))
    assert asdict(COLBERT_CONFIG) == before


@pytest.mark.parametrize("field", expected_values(Path("/tmp/example")).keys())
def test_effective_config_validation_catches_every_field_mismatch(tmp_path, field):
    checkpoint = resolved(tmp_path)
    values = expected_values(checkpoint.snapshot_path)
    expected = values[field]
    if type(expected) is bool:
        values[field] = not expected
    elif type(expected) is int:
        values[field] = expected + 1
    elif type(expected) is float:
        values[field] = expected + 0.01
    else:
        values[field] = f"wrong-{expected}"

    with pytest.raises(ValueError, match=field):
        validate_effective_stanford_config(
            SimpleNamespace(**values),
            checkpoint,
        )


def test_effective_config_validation_is_strict_about_types(tmp_path):
    checkpoint = resolved(tmp_path)
    values = expected_values(checkpoint.snapshot_path)
    values["dim"] = True

    with pytest.raises(ValueError, match="dim"):
        validate_effective_stanford_config(SimpleNamespace(**values), checkpoint)


def test_initializer_uses_exact_path_explicit_config_and_validates_post_init(
    tmp_path, monkeypatch
):
    checkpoint_resolution = resolved(tmp_path)
    calls = []

    class FakeCheckpoint:
        def __init__(self, path, *, colbert_config, verbose):
            calls.append((path, colbert_config, verbose))
            self.colbert_config = colbert_config

    monkeypatch.setattr(
        runtime_module,
        "_stanford_checkpoint_class",
        lambda: FakeCheckpoint,
    )
    result = initialize_colbert_checkpoint(
        checkpoint_resolution,
        verbose=7,
    )

    assert calls == [
        (
            str(checkpoint_resolution.snapshot_path.resolve()),
            result.checkpoint.colbert_config,
            7,
        )
    ]
    assert result.effective_config.payload() == expected_values(
        checkpoint_resolution.snapshot_path
    )
    assert result.colbert_ai_version == "0.2.22"
    assert result.transformers_version == "4.57.6"
    assert result.huggingface_hub_version == "0.36.2"
    with pytest.raises(FrozenInstanceError):
        result.colbert_ai_version = "different"
    with pytest.raises(FrozenInstanceError):
        result.effective_config.dim = 64


def test_version_validation_precedes_checkpoint_initialization(tmp_path, monkeypatch):
    calls = []

    def version(distribution):
        if distribution == "transformers":
            return "5.14.1"
        return RUNTIME_VERSIONS[distribution]

    def forbidden_checkpoint_class():
        calls.append("checkpoint")
        raise AssertionError("Checkpoint initialization must not be reached")

    monkeypatch.setattr(runtime_module.importlib_metadata, "version", version)
    monkeypatch.setattr(
        runtime_module,
        "_stanford_checkpoint_class",
        forbidden_checkpoint_class,
    )

    with pytest.raises(RuntimeError, match="transformers version mismatch"):
        initialize_colbert_checkpoint(resolved(tmp_path))
    assert calls == []


@pytest.mark.parametrize("field", ["doc_maxlen", "nbits", "ncells"])
def test_initializer_rejects_backend_overwriting_explicit_values(
    tmp_path, monkeypatch, field
):
    class OverwritingCheckpoint:
        def __init__(self, path, *, colbert_config, verbose):
            setattr(colbert_config, field, getattr(colbert_config, field) + 1)
            self.colbert_config = colbert_config

    monkeypatch.setattr(
        runtime_module,
        "_stanford_checkpoint_class",
        lambda: OverwritingCheckpoint,
    )

    with pytest.raises(ValueError, match=field):
        initialize_colbert_checkpoint(resolved(tmp_path))


def test_runtime_source_has_no_ragatouille_or_hugging_face_download_dependency():
    source = inspect.getsource(runtime_module).lower()
    assert "ragatouille" not in source
    assert "snapshot_download" not in source
    assert "hf_hub_download" not in source


def test_runtime_import_is_lazy_for_stanford_colbert_and_torch():
    repository_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = str(repository_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    script = (
        "import sys; "
        "assert 'colbert' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "import retrievers.colbert_runtime; "
        "assert 'colbert' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
