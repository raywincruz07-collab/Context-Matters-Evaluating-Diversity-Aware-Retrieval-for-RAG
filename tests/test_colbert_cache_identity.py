from dataclasses import replace
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    build_colbert_cache_identity,
    colbert_index_settings,
    colbert_search_settings,
    fingerprint_colbert_index_directory,
)
from retrievers.colbert_config import COLBERT_CONFIG


def manifest():
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA, source="fixture", config=None,
        revision="revision", split="train", construction_algorithm="fixture.v1",
        input_sample_manifest_id=None, input_sample_manifest_sha256=None,
        dependencies=(), rng_family=None, sampling_seed=None,
        rng_state_semantics=None, requested_negatives_per_query=None,
        negative_sampling_scope=None, negative_exclusion_scope=None,
        negative_sampling_without_replacement=None, final_source_id_ordering=None,
        entries=(CorpusManifestEntry(0, "doc", "source", None, "a"*64, "b"*64),),
    )


def identity(config=COLBERT_CONFIG, checkpoint_sha="c" * 64):
    return build_colbert_cache_identity(
        corpus_manifest=manifest(),
        checkpoint_snapshot_manifest_sha256=checkpoint_sha,
        colbert_config=config,
    )


def test_index_identity_binds_corpus_checkpoint_and_index_settings():
    base = identity()
    assert base.scientific_payload()["index_settings"] == colbert_index_settings()
    assert base.fingerprint_sha256 != identity(checkpoint_sha="d" * 64).fingerprint_sha256
    assert base.fingerprint_sha256 != identity(
        replace(COLBERT_CONFIG, nbits=4)
    ).fingerprint_sha256
    assert base.fingerprint_sha256 != identity(
        replace(COLBERT_CONFIG, amp=False)
    ).fingerprint_sha256
    assert colbert_index_settings()["amp"] is True


def test_search_only_settings_do_not_change_physical_index_identity():
    base = identity().fingerprint_sha256
    for changed in (
        replace(COLBERT_CONFIG, candidate_pool_size=21),
        replace(COLBERT_CONFIG, search_ncells=3),
        replace(COLBERT_CONFIG, search_centroid_score_threshold=0.5),
        replace(COLBERT_CONFIG, search_ndocs=2048),
        replace(COLBERT_CONFIG, ranking="different native ranking declaration"),
    ):
        assert identity(changed).fingerprint_sha256 == base
    assert colbert_search_settings()["candidate_pool_size"] == 20


def test_complete_directory_fingerprint_is_deterministic_and_content_sensitive(tmp_path):
    index = tmp_path / "index"
    (index / "nested").mkdir(parents=True)
    (index / "b.bin").write_bytes(b"b")
    (index / "nested/a.bin").write_bytes(b"a")
    first = fingerprint_colbert_index_directory(index)
    assert first == fingerprint_colbert_index_directory(index)
    (index / "nested/a.bin").write_bytes(b"changed")
    assert fingerprint_colbert_index_directory(index) != first
    (index / "extra.bin").write_bytes(b"extra")
    assert fingerprint_colbert_index_directory(index) != first
