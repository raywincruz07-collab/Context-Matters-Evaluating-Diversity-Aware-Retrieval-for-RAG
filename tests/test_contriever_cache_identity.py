from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CONTRIEVER_CACHE_IDENTITY_SCHEMA_VERSION,
    CORPUS_MANIFEST_SCHEMA_VERSION,
    ContrieverCacheIdentity,
    CorpusManifest,
    CorpusManifestEntry,
    build_contriever_cache_identity,
    contriever_embedding_cache_filename,
    contriever_faiss_cache_filename,
)
from retrievers.contriever_config import CONTRIEVER_CONFIG, ContrieverConfig
from scripts.build_corpus_manifests import load_frozen_pubmedqa_corpus_manifest


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def corpus_manifest() -> CorpusManifest:
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus",
        config=None,
        revision="fixture-revision",
        split="fixture-split",
        construction_algorithm="fixture-construction.v1",
        input_sample_manifest_id=None,
        input_sample_manifest_sha256=None,
        dependencies=(),
        rng_family=None,
        sampling_seed=None,
        rng_state_semantics=None,
        requested_negatives_per_query=None,
        negative_sampling_scope=None,
        negative_exclusion_scope=None,
        negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=(
            CorpusManifestEntry(0, 0, "source-0", None, SHA_A, SHA_B),
            CorpusManifestEntry(1, "one", "source-1", None, SHA_B, SHA_C),
        ),
    )


def test_identical_inputs_produce_identical_payload_json_and_fingerprint():
    first = build_contriever_cache_identity(corpus_manifest=corpus_manifest())
    second = build_contriever_cache_identity(corpus_manifest=corpus_manifest())
    assert first.scientific_payload() == second.scientific_payload()
    assert first.scientific_json() == second.scientific_json()
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert json.loads(first.scientific_json()) == first.scientific_payload()
    assert first.scientific_json() == json.dumps(
        first.scientific_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_payload_is_exact_complete_and_excludes_non_scientific_fields():
    identity = build_contriever_cache_identity(corpus_manifest=corpus_manifest())
    payload = identity.scientific_payload()
    assert payload == {
        "corpus": {
            "corpus_manifest_id": identity.corpus_manifest.corpus_manifest_id,
            "corpus_manifest_sha256": identity.corpus_manifest.sha256,
            "document_count": 2,
        },
        "contriever_config": CONTRIEVER_CONFIG.scientific_payload(),
        "schema_version": CONTRIEVER_CACHE_IDENTITY_SCHEMA_VERSION,
    }
    assert set(payload["contriever_config"]) == set(
        ContrieverConfig.__dataclass_fields__
    )
    forbidden = {
        "git_commit",
        "timestamp",
        "hostname",
        "machine",
        "cuda",
        "device",
        "cache_path",
        "physical_embedding_sha256",
        "physical_faiss_sha256",
        "environment_fingerprint",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["contriever_config"])


@pytest.mark.parametrize(
    "changed_config",
    [
        replace(CONTRIEVER_CONFIG, model_revision="0" * 40),
        replace(CONTRIEVER_CONFIG, tokenizer_revision="1" * 40),
        replace(CONTRIEVER_CONFIG, query_max_length=256),
        replace(CONTRIEVER_CONFIG, document_max_length=256),
        replace(CONTRIEVER_CONFIG, pooling="different pooling"),
        replace(CONTRIEVER_CONFIG, normalization="l2"),
        replace(CONTRIEVER_CONFIG, compute_dtype="float16"),
        replace(CONTRIEVER_CONFIG, autocast=True),
        replace(CONTRIEVER_CONFIG, query_batch_size=32),
        replace(CONTRIEVER_CONFIG, document_batch_size=32),
    ],
)
def test_scientific_config_changes_change_fingerprint(changed_config):
    original = build_contriever_cache_identity(corpus_manifest=corpus_manifest())
    changed = build_contriever_cache_identity(
        corpus_manifest=corpus_manifest(),
        contriever_config=changed_config,
    )
    assert changed.fingerprint_sha256 != original.fingerprint_sha256


def test_corpus_manifest_change_changes_fingerprint():
    original_manifest = corpus_manifest()
    changed_manifest = replace(original_manifest, revision="different-revision")
    assert build_contriever_cache_identity(
        corpus_manifest=changed_manifest
    ).fingerprint_sha256 != build_contriever_cache_identity(
        corpus_manifest=original_manifest
    ).fingerprint_sha256


def test_fingerprint_and_cache_filenames_are_deterministic():
    identity = build_contriever_cache_identity(corpus_manifest=corpus_manifest())
    fingerprint = identity.fingerprint_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert identity.embedding_cache_filename == (
        f"contriever_embeddings_{fingerprint}.npy"
    )
    assert identity.faiss_cache_filename == f"contriever_index_{fingerprint}.faiss"
    assert contriever_embedding_cache_filename(fingerprint) == (
        identity.embedding_cache_filename
    )
    assert contriever_faiss_cache_filename(fingerprint) == (
        identity.faiss_cache_filename
    )


@pytest.mark.parametrize("value", [None, "", "A" * 64, "g" * 64, "0" * 63])
def test_cache_filename_helpers_reject_malformed_fingerprints(value):
    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        contriever_embedding_cache_filename(value)
    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        contriever_faiss_cache_filename(value)


def test_schema_and_input_types_are_validated():
    manifest = corpus_manifest()
    with pytest.raises(ValueError, match="unsupported Contriever cache identity"):
        ContrieverCacheIdentity(
            "sprint3.contriever-cache-identity.v2",
            manifest,
            CONTRIEVER_CONFIG,
        )
    with pytest.raises(TypeError, match="corpus_manifest must be a CorpusManifest"):
        build_contriever_cache_identity(corpus_manifest="not-a-manifest")
    with pytest.raises(TypeError, match="contriever_config must be a ContrieverConfig"):
        build_contriever_cache_identity(
            corpus_manifest=manifest,
            contriever_config={},
        )


def test_manifest_sha_and_id_consistency_are_validated(monkeypatch):
    manifest = corpus_manifest()
    monkeypatch.setattr(
        CorpusManifest,
        "sha256",
        property(lambda self: "A" * 64),
    )
    with pytest.raises(ValueError, match="CorpusManifest SHA"):
        build_contriever_cache_identity(corpus_manifest=manifest)


def test_manifest_id_must_encode_same_sha(monkeypatch):
    manifest = corpus_manifest()
    monkeypatch.setattr(
        CorpusManifest,
        "corpus_manifest_id",
        property(lambda self: f"corpus-manifest:sha256:{'0' * 64}"),
    )
    with pytest.raises(ValueError, match="ID must match"):
        build_contriever_cache_identity(corpus_manifest=manifest)


def test_frozen_pubmedqa_manifest_has_stable_repeated_fingerprint():
    path = Path("artifacts/corpus_manifests/pubmedqa_corpus_manifest_v1.json")
    manifest = load_frozen_pubmedqa_corpus_manifest(path)
    first = build_contriever_cache_identity(corpus_manifest=manifest)
    second = build_contriever_cache_identity(corpus_manifest=manifest)
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert first.scientific_json() == second.scientific_json()
    assert first.scientific_payload()["corpus"]["document_count"] == 3358
