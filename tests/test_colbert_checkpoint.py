from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrievers.colbert_checkpoint as checkpoint_module
from retrievers.colbert_checkpoint import resolve_colbert_checkpoint
from retrievers.colbert_config import COLBERT_CONFIG


REVISION = "0855eac81381e0323a846f1ed7d8452d4c648b50"


def snapshot_directory(tmp_path: Path, revision: str = REVISION) -> Path:
    path = tmp_path / "hub" / "models--colbert-ir--colbertv2.0" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def fail_if_downloaded(**kwargs):
    raise AssertionError("snapshot_download must not be called")


def test_wrong_config_type_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "snapshot_download", fail_if_downloaded)

    with pytest.raises(TypeError, match="config must be a ColBERTConfig"):
        resolve_colbert_checkpoint(
            object(),
            cache_dir=tmp_path / "cache",
            local_files_only=True,
        )


@pytest.mark.parametrize("cache_dir", ["", "   "])
def test_empty_cache_dir_fails_before_download(cache_dir, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "snapshot_download", fail_if_downloaded)

    with pytest.raises(ValueError, match="cache_dir"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=cache_dir,
            local_files_only=True,
        )


def test_non_bool_local_files_only_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "snapshot_download", fail_if_downloaded)

    with pytest.raises(TypeError, match="local_files_only must be a bool"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=tmp_path / "cache",
            local_files_only=1,
        )


@pytest.mark.parametrize("local_files_only", [True, False])
def test_exact_download_arguments_and_resolved_snapshot(
    tmp_path, monkeypatch, local_files_only
):
    snapshot = snapshot_directory(tmp_path)
    cache_dir = tmp_path / "explicit-cache"
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return snapshot

    monkeypatch.setattr(checkpoint_module, "snapshot_download", fake_snapshot_download)
    result = resolve_colbert_checkpoint(
        COLBERT_CONFIG,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    assert calls == [
        {
            "repo_id": "colbert-ir/colbertv2.0",
            "revision": REVISION,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
    ]
    assert result.checkpoint_id == COLBERT_CONFIG.checkpoint_id
    assert result.checkpoint_revision == REVISION
    assert result.snapshot_path == snapshot.resolve()
    assert result.snapshot_path.is_absolute()


def test_resolved_checkpoint_is_immutable(tmp_path, monkeypatch):
    snapshot = snapshot_directory(tmp_path)
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: snapshot,
    )
    result = resolve_colbert_checkpoint(
        COLBERT_CONFIG,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    with pytest.raises(FrozenInstanceError):
        result.snapshot_path = tmp_path


def test_relative_returned_path_is_resolved_to_absolute(tmp_path, monkeypatch):
    snapshot = snapshot_directory(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative = snapshot.relative_to(tmp_path)
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: relative,
    )

    result = resolve_colbert_checkpoint(
        COLBERT_CONFIG,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    assert result.snapshot_path == snapshot.resolve()
    assert result.snapshot_path.is_absolute()


def test_wrong_snapshot_revision_is_rejected(tmp_path, monkeypatch):
    wrong = snapshot_directory(tmp_path, revision="0" * 40)
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: wrong,
    )

    with pytest.raises(ValueError, match="snapshot revision does not match"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=tmp_path / "cache",
            local_files_only=True,
        )


def test_nonexistent_snapshot_path_is_rejected(tmp_path, monkeypatch):
    missing = tmp_path / "snapshots" / REVISION
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: missing,
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=tmp_path / "cache",
            local_files_only=True,
        )


def test_snapshot_file_instead_of_directory_is_rejected(tmp_path, monkeypatch):
    snapshot_file = tmp_path / REVISION
    snapshot_file.write_text("not a snapshot directory", encoding="utf-8")
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: snapshot_file,
    )

    with pytest.raises(NotADirectoryError, match="not a directory"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=tmp_path / "cache",
            local_files_only=True,
        )


def test_downloader_exception_propagates_without_fallback(tmp_path, monkeypatch):
    calls = []

    def fail_download(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic checkpoint resolution failure")

    monkeypatch.setattr(checkpoint_module, "snapshot_download", fail_download)

    with pytest.raises(RuntimeError, match="synthetic checkpoint resolution failure"):
        resolve_colbert_checkpoint(
            COLBERT_CONFIG,
            cache_dir=tmp_path / "cache",
            local_files_only=False,
        )

    assert len(calls) == 1
    assert calls[0]["revision"] == REVISION


def test_resolution_does_not_change_scientific_config_or_add_physical_path(
    tmp_path, monkeypatch
):
    snapshot = snapshot_directory(tmp_path)
    before = asdict(COLBERT_CONFIG)
    scientific_before = COLBERT_CONFIG.scientific_payload()
    monkeypatch.setattr(
        checkpoint_module,
        "snapshot_download",
        lambda **kwargs: snapshot,
    )

    result = resolve_colbert_checkpoint(
        COLBERT_CONFIG,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    assert asdict(COLBERT_CONFIG) == before
    assert COLBERT_CONFIG.scientific_payload() == scientific_before
    assert "snapshot_path" not in scientific_before
    assert result.snapshot_path == snapshot.resolve()
