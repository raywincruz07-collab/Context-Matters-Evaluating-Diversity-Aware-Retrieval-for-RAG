from dataclasses import FrozenInstanceError, asdict, replace
import inspect
import json
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
    CanonicalColBERTRetriever,
    apply_colbert_index_seed,
    build_stanford_colbert_config,
    initialize_colbert_checkpoint,
    validate_effective_stanford_config,
)
from retrievers.colbert_checkpoint_provenance import (
    ColBERTCheckpointPhysicalProvenance,
)
from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    document_content_sha256,
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
        "amp": True,
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
    assert stanford.amp is True
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


def _physical(checkpoint):
    return ColBERTCheckpointPhysicalProvenance(
        checkpoint_id=COLBERT_CONFIG.checkpoint_id,
        checkpoint_revision=REVISION,
        snapshot_path=checkpoint.snapshot_path.resolve(), files=(), file_count=0,
        total_bytes=0, snapshot_manifest_sha256="c" * 64, metadata=None,
    )


def _corpus():
    records = tuple(CorpusRecord(
        document_id=f"canonical-{position}", source_document_id=f"source-{position}",
        title=None, text=f"text-{position}", retrieval_content=f"exact {position}",
        corpus_position=position,
    ) for position in range(20))
    manifest = CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA, source="fixture", config=None,
        revision="revision", split="train", construction_algorithm="fixture.v1",
        input_sample_manifest_id=None, input_sample_manifest_sha256=None,
        dependencies=(), rng_family=None, sampling_seed=None,
        rng_state_semantics=None, requested_negatives_per_query=None,
        negative_sampling_scope=None, negative_exclusion_scope=None,
        negative_sampling_without_replacement=None, final_source_id_ordering=None,
        entries=tuple(CorpusManifestEntry(
            record.corpus_position, record.document_id, record.source_document_id,
            None, document_content_sha256(record.text),
            document_content_sha256(record.retrieval_content),
        ) for record in records),
    )
    return manifest, records


def test_direct_index_search_maps_pids_and_preserves_native_order(tmp_path, monkeypatch):
    checkpoint = resolved(tmp_path)
    index_root = tmp_path / "indexes"
    calls = []

    class Context:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class FakeRun:
        def context(self, config): calls.append(("run", config)); return Context()
    class FakeRunConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items(): setattr(self, key, value)
    class FakeIndexer:
        def __init__(self, **kwargs): calls.append(("indexer", kwargs))
        def index(self, *, name, collection, overwrite):
            calls.append(("index", name, tuple(collection), overwrite))
            directory = index_root / "sprint3_pubmedqa_colbertv2" / "indexes" / name
            marker = directory.parent / f".{name}.sprint3-complete.json"
            assert not marker.exists()
            directory.mkdir(parents=True)
            (directory / "metadata.json").write_text("{}", encoding="utf-8")
    class FakeSearcher:
        def __init__(self, **kwargs): calls.append(("searcher", kwargs))
        def search(self, query, k):
            return list(reversed(range(20))), list(range(1, 21)), [20.0-i for i in range(20)]

    monkeypatch.setattr(runtime_module, "_stanford_run_classes", lambda: (FakeRun, FakeRunConfig))
    monkeypatch.setattr(runtime_module, "_stanford_indexer_class", lambda: FakeIndexer)
    monkeypatch.setattr(runtime_module, "_stanford_searcher_class", lambda: FakeSearcher)
    monkeypatch.setattr(runtime_module, "require_colbert_runtime_prerequisites", lambda: None)
    seeded = []
    monkeypatch.setattr(runtime_module, "apply_colbert_index_seed", seeded.append)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=index_root,
    )
    manifest, records = _corpus()
    retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)
    results = retriever.retrieve("Exact query", top_k=20)
    assert seeded == [12345]
    assert calls[0][1].amp is True
    assert calls[2][2] == tuple(record.retrieval_content for record in records)
    assert calls[2][3] is False
    index_config = calls[1][1]["config"]
    assert not hasattr(index_config, "ncells")
    assert not hasattr(index_config, "centroid_score_threshold")
    assert not hasattr(index_config, "ndocs")
    search_config = calls[3][1]["config"]
    assert search_config.ncells == 2
    assert search_config.centroid_score_threshold == 0.45
    assert search_config.ndocs == 1024
    assert [result.document_id for result in results] == [
        f"canonical-{pid}" for pid in reversed(range(20))
    ]
    assert retriever.index_artifact_sha256 is not None
    assert retriever.completion_record_path.is_file()
    assert json.loads(retriever.completion_record_path.read_text()) == (
        retriever._completion_record_payload()
    )


def test_preexisting_index_without_completion_record_is_rejected(tmp_path):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    from retrieval_artifacts import build_colbert_cache_identity
    retriever.cache_identity = build_colbert_cache_identity(
        corpus_manifest=manifest, checkpoint_snapshot_manifest_sha256="c" * 64
    )
    retriever.index_directory.mkdir(parents=True)
    (retriever.index_directory / "file").write_bytes(b"index")
    with pytest.raises(RuntimeError, match=r"incomplete ColBERT index: .*cleanup/rebuild"):
        retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)


def _mock_existing_index_backend(monkeypatch):
    class Context:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Run:
        def context(self, config): return Context()
    class RunConfig:
        def __init__(self, **kwargs): pass
    class Searcher:
        def __init__(self, **kwargs): pass
    monkeypatch.setattr(runtime_module, "_stanford_run_classes", lambda: (Run, RunConfig))
    monkeypatch.setattr(runtime_module, "_stanford_searcher_class", lambda: Searcher)
    monkeypatch.setattr(runtime_module, "_stanford_indexer_class", lambda: pytest.fail("must not index"))
    monkeypatch.setattr(runtime_module, "apply_colbert_index_seed", lambda seed: pytest.fail("must not seed"))
    monkeypatch.setattr(runtime_module, "require_colbert_runtime_prerequisites", lambda: None)


def test_valid_existing_index_and_matching_completion_record_are_reused(tmp_path, monkeypatch):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    from retrieval_artifacts import build_colbert_cache_identity
    retriever.cache_identity = build_colbert_cache_identity(
        corpus_manifest=manifest, checkpoint_snapshot_manifest_sha256="c" * 64
    )
    retriever.index_directory.mkdir(parents=True)
    (retriever.index_directory / "file").write_bytes(b"index")
    retriever._write_completion_record()
    _mock_existing_index_backend(monkeypatch)
    retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)
    assert retriever.is_indexed


def test_missing_ninja_prevents_completed_index_searcher_initialization(
    tmp_path, monkeypatch
):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    from retrieval_artifacts import build_colbert_cache_identity
    retriever.cache_identity = build_colbert_cache_identity(
        corpus_manifest=manifest, checkpoint_snapshot_manifest_sha256="c" * 64
    )
    retriever.index_directory.mkdir(parents=True)
    (retriever.index_directory / "file").write_bytes(b"index")
    retriever._write_completion_record()
    monkeypatch.setattr(
        runtime_module.subprocess, "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ninja")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_stanford_searcher_class",
        lambda: pytest.fail("must not construct Searcher"),
    )
    monkeypatch.setattr(
        runtime_module,
        "_stanford_indexer_class",
        lambda: pytest.fail("must not construct Indexer"),
    )
    with pytest.raises(
        RuntimeError, match="ColBERT runtime prerequisite missing: ninja"
    ):
        retriever.index_from_corpus_records(
            corpus_manifest=manifest, corpus_records=records
        )
    assert retriever.is_indexed is False


def test_wrong_completion_record_identity_is_rejected(tmp_path):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    from retrieval_artifacts import build_colbert_cache_identity
    retriever.cache_identity = build_colbert_cache_identity(
        corpus_manifest=manifest, checkpoint_snapshot_manifest_sha256="c" * 64
    )
    retriever.index_directory.mkdir(parents=True)
    (retriever.index_directory / "file").write_bytes(b"index")
    payload = retriever._completion_record_payload()
    payload["index_fingerprint_sha256"] = "0" * 64
    retriever.completion_record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion record identity mismatch"):
        retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)


def test_missing_ninja_fails_before_indexer_is_invoked(tmp_path, monkeypatch):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    monkeypatch.setattr(
        runtime_module.subprocess, "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ninja")),
    )
    monkeypatch.setattr(runtime_module, "_stanford_indexer_class", lambda: pytest.fail("must not construct Indexer"))
    with pytest.raises(RuntimeError, match="ColBERT runtime prerequisite missing: ninja"):
        retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)
    assert not retriever.index_directory.exists()
    assert not retriever.completion_record_path.exists()


def test_failed_indexer_leaves_partial_files_without_completion_record(tmp_path, monkeypatch):
    checkpoint = resolved(tmp_path)
    retriever = CanonicalColBERTRetriever(
        resolved_checkpoint=checkpoint, checkpoint_provenance=_physical(checkpoint),
        index_root=tmp_path / "indexes",
    )
    manifest, records = _corpus()
    class Context:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Run:
        def context(self, config): return Context()
    class RunConfig:
        def __init__(self, **kwargs): pass
    class FailingIndexer:
        def __init__(self, **kwargs): pass
        def index(self, *, name, collection, overwrite):
            retriever.index_directory.mkdir(parents=True)
            (retriever.index_directory / "partial").write_bytes(b"forensics")
            assert not retriever.completion_record_path.exists()
            raise RuntimeError("synthetic Stanford failure")
    monkeypatch.setattr(runtime_module, "require_colbert_runtime_prerequisites", lambda: None)
    monkeypatch.setattr(runtime_module, "apply_colbert_index_seed", lambda seed: None)
    monkeypatch.setattr(runtime_module, "_stanford_run_classes", lambda: (Run, RunConfig))
    monkeypatch.setattr(runtime_module, "_stanford_indexer_class", lambda: FailingIndexer)
    with pytest.raises(RuntimeError, match="synthetic Stanford failure"):
        retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)
    assert (retriever.index_directory / "partial").read_bytes() == b"forensics"
    assert not retriever.completion_record_path.exists()
    with pytest.raises(RuntimeError, match="incomplete ColBERT index"):
        retriever.index_from_corpus_records(corpus_manifest=manifest, corpus_records=records)


def test_retriever_rejects_undersized_native_result_pool():
    retriever = object.__new__(CanonicalColBERTRetriever)
    retriever.config = COLBERT_CONFIG
    retriever.pid_to_document_id = tuple(f"doc-{i}" for i in range(20))
    retriever._searcher = SimpleNamespace(
        search=lambda query, k: (
            list(range(19)), list(range(1, 20)), [19.0-i for i in range(19)]
        )
    )
    with pytest.raises(ValueError, match="non-canonical candidate count"):
        retriever.retrieve("query", top_k=20)


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
