from dataclasses import replace
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusProvenance,
    build_dpr_cache_identity,
    build_dpr_retriever_provenance,
    validate_dpr_index_binding,
)
from retrievers.dpr_config import DPR_CONFIG


SHA_A = "a" * 64
SHA_B = "b" * 64
FAISS_SHA = "f" * 64
TRANSFORMERS_VERSION = "fixture-transformers-5.14.1"


def corpus_manifest():
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus",
        config=None,
        revision="fixture-corpus-revision",
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
            CorpusManifestEntry(1, "one", "source-1", None, SHA_B, SHA_A),
        ),
    )


def cache_identity(manifest=None, config=DPR_CONFIG):
    return build_dpr_cache_identity(
        corpus_manifest=corpus_manifest() if manifest is None else manifest,
        dpr_config=config,
    )


def corpus_provenance(manifest=None):
    value = corpus_manifest() if manifest is None else manifest
    return CorpusProvenance(
        corpus_id=value.corpus_manifest_id,
        source=value.source,
        revision=value.revision,
        document_count=value.document_count,
        manifest_sha256=value.sha256,
        document_id_map_sha256="d" * 64,
        preprocessing_version=value.construction_algorithm,
    )


def provenance(identity=None, config=DPR_CONFIG):
    return build_dpr_retriever_provenance(
        cache_identity=cache_identity(config=config) if identity is None else identity,
        index_artifact_sha256=FAISS_SHA,
        transformers_version=TRANSFORMERS_VERSION,
        dpr_config=config,
    )


def test_builder_maps_exact_question_side_and_dpr_method():
    identity = cache_identity()
    value = provenance(identity)
    assert value.retriever_name == "dpr"
    assert value.implementation == (
        "retrievers.dpr_original_retriever.OriginalDPRRetriever"
    )
    assert value.library_name == "transformers"
    assert value.library_version == TRANSFORMERS_VERSION
    assert value.model_id == DPR_CONFIG.question_model_id
    assert value.model_revision == DPR_CONFIG.question_model_revision
    assert value.tokenizer_id == DPR_CONFIG.question_tokenizer_id
    assert value.tokenizer_revision == DPR_CONFIG.question_tokenizer_revision
    assert value.query_preprocessing == DPR_CONFIG.query_preprocessing
    assert value.document_preprocessing == DPR_CONFIG.document_preprocessing
    assert value.normalization == DPR_CONFIG.normalization
    assert value.score_semantics == DPR_CONFIG.score_semantics
    assert value.index_type == DPR_CONFIG.index_type
    assert value.index_fingerprint_sha256 == identity.fingerprint_sha256
    assert value.index_artifact_sha256 == FAISS_SHA


def test_index_config_preserves_complete_dual_encoder_identity():
    value = provenance()
    assert value.index_config == DPR_CONFIG.scientific_json()
    parsed = json.loads(value.index_config)
    assert parsed == DPR_CONFIG.scientific_payload()
    assert parsed["context_model_id"] == DPR_CONFIG.context_model_id
    assert parsed["context_model_revision"] == DPR_CONFIG.context_model_revision
    assert parsed["context_tokenizer_id"] == DPR_CONFIG.context_tokenizer_id
    assert (
        parsed["context_tokenizer_revision"]
        == DPR_CONFIG.context_tokenizer_revision
    )


def test_builder_and_validator_are_deterministic():
    identity = cache_identity()
    first = provenance(identity)
    second = provenance(identity)
    assert first == second
    validate_dpr_index_binding(
        corpus_provenance=corpus_provenance(),
        retriever_provenance=first,
        cache_identity=identity,
    )


def test_context_revision_changes_fingerprint_without_changing_top_level_model():
    changed_config = replace(DPR_CONFIG, context_model_revision="0" * 40)
    original = provenance()
    changed_identity = cache_identity(config=changed_config)
    changed = provenance(changed_identity, config=changed_config)
    assert changed.model_id == original.model_id == DPR_CONFIG.question_model_id
    assert changed.index_fingerprint_sha256 != original.index_fingerprint_sha256


def test_question_revision_and_corpus_manifest_change_fingerprint():
    changed_question = replace(DPR_CONFIG, question_model_revision="1" * 40)
    assert (
        cache_identity(config=changed_question).fingerprint_sha256
        != cache_identity().fingerprint_sha256
    )
    manifest = corpus_manifest()
    changed_manifest = replace(manifest, revision="different-corpus-revision")
    assert (
        cache_identity(changed_manifest).fingerprint_sha256
        != cache_identity(manifest).fingerprint_sha256
    )


@pytest.mark.parametrize("value", ["", "A" * 64, "g" * 64, "0" * 63])
def test_builder_rejects_malformed_physical_index_sha(value):
    with pytest.raises(ValueError, match="index_artifact_sha256"):
        build_dpr_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=value,
            transformers_version=TRANSFORMERS_VERSION,
        )


def test_builder_rejects_config_cache_identity_mismatch():
    changed = replace(DPR_CONFIG, context_max_length=128)
    with pytest.raises(ValueError, match="DPRConfig does not match"):
        build_dpr_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=FAISS_SHA,
            transformers_version=TRANSFORMERS_VERSION,
            dpr_config=changed,
        )


def test_validator_rejects_corpus_manifest_sha_or_count_mismatch():
    identity = cache_identity()
    value = provenance(identity)
    corpus = corpus_provenance()
    with pytest.raises(ValueError, match="CorpusManifest SHA"):
        validate_dpr_index_binding(
            corpus_provenance=replace(corpus, manifest_sha256="0" * 64),
            retriever_provenance=value,
            cache_identity=identity,
        )
    with pytest.raises(ValueError, match="document count"):
        validate_dpr_index_binding(
            corpus_provenance=replace(corpus, document_count=3),
            retriever_provenance=value,
            cache_identity=identity,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("retriever_name", "not-dpr"),
        ("implementation", "wrong.implementation"),
        ("library_name", "not-transformers"),
        ("model_id", "wrong-model"),
        ("tokenizer_revision", "0" * 40),
        ("query_preprocessing", "wrong preprocessing"),
        ("document_preprocessing", "wrong preprocessing"),
        ("normalization", "l2"),
        ("score_semantics", "cosine"),
        ("index_type", "faiss.IndexFlatL2"),
        ("index_fingerprint_sha256", "0" * 64),
    ],
)
def test_validator_rejects_wrong_declared_dpr_fields(field, value):
    identity = cache_identity()
    with pytest.raises(ValueError, match=field):
        validate_dpr_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(provenance(identity), **{field: value}),
            cache_identity=identity,
        )


def test_context_identity_cannot_disappear_from_index_config():
    identity = cache_identity()
    value = provenance(identity)
    parsed = json.loads(value.index_config)
    parsed.pop("context_model_revision")
    changed_json = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with pytest.raises(ValueError, match="context encoder identity"):
        validate_dpr_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(value, index_config=changed_json),
            cache_identity=identity,
        )


@pytest.mark.parametrize("index_config", ["null", "5", '"x"', "[]", "true"])
def test_index_config_must_decode_to_json_object(index_config):
    identity = cache_identity()
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        validate_dpr_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_config=index_config
            ),
            cache_identity=identity,
        )


def test_validator_requires_physical_faiss_sha():
    identity = cache_identity()
    with pytest.raises(ValueError, match="physical FAISS artifact SHA"):
        validate_dpr_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_artifact_sha256=None
            ),
            cache_identity=identity,
        )
