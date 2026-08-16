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
    CORPUS_MANIFEST_SCHEMA_VERSION,
    DPR_CACHE_IDENTITY_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    DPRCacheIdentity,
    build_dpr_cache_identity,
    dpr_embedding_cache_filename,
    dpr_faiss_cache_filename,
)
from retrievers.dpr_config import DPR_CONFIG
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
    first = build_dpr_cache_identity(corpus_manifest=corpus_manifest())
    second = build_dpr_cache_identity(corpus_manifest=corpus_manifest())
    assert first.scientific_payload() == second.scientific_payload()
    assert first.scientific_json() == second.scientific_json()
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert json.loads(first.scientific_json()) == first.scientific_payload()


@pytest.mark.parametrize(
    "changed_config",
    [
        replace(DPR_CONFIG, question_model_revision="0" * 40),
        replace(DPR_CONFIG, context_model_revision="1" * 40),
        replace(DPR_CONFIG, query_max_length=128),
        replace(DPR_CONFIG, representation="different-representation"),
    ],
)
def test_dpr_scientific_config_changes_fingerprint(changed_config):
    original = build_dpr_cache_identity(corpus_manifest=corpus_manifest())
    changed = build_dpr_cache_identity(
        corpus_manifest=corpus_manifest(), dpr_config=changed_config
    )
    assert changed.fingerprint_sha256 != original.fingerprint_sha256


def test_corpus_manifest_change_changes_fingerprint():
    original_manifest = corpus_manifest()
    changed_manifest = replace(original_manifest, revision="different-revision")
    original = build_dpr_cache_identity(corpus_manifest=original_manifest)
    changed = build_dpr_cache_identity(corpus_manifest=changed_manifest)
    assert changed.fingerprint_sha256 != original.fingerprint_sha256


def test_payload_is_explicit_and_excludes_non_scientific_inputs():
    identity = build_dpr_cache_identity(corpus_manifest=corpus_manifest())
    payload = identity.scientific_payload()
    assert payload == {
        "corpus": {
            "corpus_manifest_id": identity.corpus_manifest.corpus_manifest_id,
            "corpus_manifest_sha256": identity.corpus_manifest.sha256,
            "document_count": 2,
        },
        "dpr_config": json.loads(DPR_CONFIG.scientific_json()),
        "schema_version": DPR_CACHE_IDENTITY_SCHEMA_VERSION,
    }
    assert {
        "git_commit",
        "timestamp",
        "cache_path",
        "hostname",
        "machine",
        "device",
    }.isdisjoint(payload)


def test_fingerprint_and_cache_filenames_are_deterministic():
    identity = build_dpr_cache_identity(corpus_manifest=corpus_manifest())
    fingerprint = identity.fingerprint_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert identity.embedding_cache_filename == f"dpr_embeddings_{fingerprint}.npy"
    assert identity.faiss_cache_filename == f"dpr_index_{fingerprint}.faiss"
    assert dpr_embedding_cache_filename(fingerprint) == identity.embedding_cache_filename
    assert dpr_faiss_cache_filename(fingerprint) == identity.faiss_cache_filename


@pytest.mark.parametrize("value", ["", "A" * 64, "g" * 64, "0" * 63])
def test_cache_filename_rejects_malformed_fingerprint(value):
    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        dpr_embedding_cache_filename(value)


def test_malformed_identity_inputs_fail_loudly():
    manifest = corpus_manifest()
    with pytest.raises(ValueError, match="schema_version must be a non-empty string"):
        DPRCacheIdentity("", manifest, DPR_CONFIG)
    with pytest.raises(TypeError, match="corpus_manifest must be a CorpusManifest"):
        build_dpr_cache_identity(corpus_manifest="not-a-manifest")
    with pytest.raises(TypeError, match="dpr_config must be a DPRConfig"):
        build_dpr_cache_identity(corpus_manifest=manifest, dpr_config={})
    with pytest.raises(ValueError, match="unsupported DPR cache identity schema"):
        DPRCacheIdentity("sprint3.dpr-cache-identity.v2", manifest, DPR_CONFIG)


def test_frozen_pubmedqa_manifest_has_stable_repeated_fingerprint():
    path = Path("artifacts/corpus_manifests/pubmedqa_corpus_manifest_v1.json")
    manifest = load_frozen_pubmedqa_corpus_manifest(path)
    first = build_dpr_cache_identity(corpus_manifest=manifest)
    second = build_dpr_cache_identity(corpus_manifest=manifest)
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert first.scientific_json() == second.scientific_json()
    assert first.scientific_payload()["corpus"]["document_count"] == 3358
