"""Offline data loading and secret-free execution provenance for generation CLIs."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from generation._io import file_sha256, stable_json_sha256
from generation.maki import CanonicalMakiAdapter, MakiConfig, PRIMARY_LLM_LOGICAL_IDS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_pubmedqa_runtime_local_only(*, cache_dir: Path | None = None):
    """Load the pinned dataset from an existing cache with network disabled."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    from scripts.build_corpus_manifests import (
        PUBMEDQA_CONFIG,
        PUBMEDQA_REVISION,
        PUBMEDQA_SOURCE,
        PUBMEDQA_SPLIT,
        load_validated_pubmedqa_runtime_corpus,
    )

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if cache_dir is None else str(cache_dir),
        download_mode="reuse_dataset_if_exists",
    )
    return load_validated_pubmedqa_runtime_corpus(tuple(rows))


def git_registry_identity(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not bool(status)
    diff_sha = None
    if not clean:
        tracked = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(tracked)
        for raw_path in sorted(part for part in untracked.split(b"\0") if part):
            digest.update(raw_path)
            digest.update(b"\0")
            path = repository_root / raw_path.decode("utf-8")
            if path.is_file():
                digest.update(path.read_bytes())
            digest.update(b"\0")
        diff_sha = digest.hexdigest()
    return {
        "commit": commit,
        "branch": branch,
        "worktree_clean": clean,
        "worktree_diff_sha256": diff_sha,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "requests": _package_version("requests"),
        "generation_package_files": {
            path.name: file_sha256(path)
            for path in sorted((REPOSITORY_ROOT / "src/generation").glob("*.py"))
        },
    }


def load_model_bindings(path: Path) -> dict[str, MakiConfig]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, Mapping) or set(stored) != {"models"}:
        raise ValueError("model bindings must contain exactly a models object")
    models = stored["models"]
    if not isinstance(models, Mapping) or set(models) != set(PRIMARY_LLM_LOGICAL_IDS):
        raise ValueError("model bindings require exactly the three primary logical IDs")
    result: dict[str, MakiConfig] = {}
    for logical_id in PRIMARY_LLM_LOGICAL_IDS:
        value = models[logical_id]
        if not isinstance(value, Mapping):
            raise ValueError("each model binding must be an object")
        result[logical_id] = MakiConfig(
            base_url=value["base_url"],
            logical_model_id=logical_id,
            physical_model_id=value["physical_model_id"],
            model_revision=value.get("model_revision"),
            model_revision_kind=value["model_revision_kind"],
            direct_mode_status=value["direct_mode_status"],
            direct_mode_control=value["direct_mode_control"],
            timeout_seconds=value.get("timeout_seconds", 300.0),
        )
    return result


def adapter_from_bindings(
    bindings: Mapping[str, MakiConfig], logical_id: str
) -> CanonicalMakiAdapter:
    return CanonicalMakiAdapter(bindings[logical_id])


def runtime_provenance(adapter: CanonicalMakiAdapter) -> dict[str, Any]:
    return {
        "runtime_identity": adapter.config.runtime_identity(),
        "environment_variable_names": ["MAKI_API_KEY"],
        "secret_values_persisted": False,
    }


def provenance_hashes(adapter: CanonicalMakiAdapter) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    environment = environment_provenance()
    runtime = runtime_provenance(adapter)
    return environment, runtime, stable_json_sha256(environment), stable_json_sha256(runtime)
