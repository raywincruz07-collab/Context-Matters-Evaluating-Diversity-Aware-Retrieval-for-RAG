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
    build_contriever_cache_identity,
    build_contriever_retriever_provenance,
    validate_contriever_index_binding,
)
from retrievers.contriever_config import CONTRIEVER_CONFIG


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


def cache_identity(manifest=None, config=CONTRIEVER_CONFIG):
    return build_contriever_cache_identity(
        corpus_manifest=corpus_manifest() if manifest is None else manifest,
        contriever_config=config,
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


def provenance(identity=None, config=CONTRIEVER_CONFIG):
    return build_contriever_retriever_provenance(
        cache_identity=(
            cache_identity(config=config) if identity is None else identity
        ),
        index_artifact_sha256=FAISS_SHA,
        transformers_version=TRANSFORMERS_VERSION,
        contriever_config=config,
    )


def canonical_json(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_builder_captures_exact_canonical_contriever_provenance():
    identity = cache_identity()
    value = provenance(identity)
    assert value.retriever_name == "contriever"
    assert value.implementation == (
        "retrievers.contriever_retriever.ContrieverRetriever"
    )
    assert value.library_name == "transformers"
    assert value.library_version == TRANSFORMERS_VERSION
    assert value.model_id == CONTRIEVER_CONFIG.model_id
    assert value.model_revision == CONTRIEVER_CONFIG.model_revision
    assert value.tokenizer_id == CONTRIEVER_CONFIG.tokenizer_id
    assert value.tokenizer_revision == CONTRIEVER_CONFIG.tokenizer_revision
    assert value.query_preprocessing == CONTRIEVER_CONFIG.query_preprocessing
    assert value.document_preprocessing == CONTRIEVER_CONFIG.document_preprocessing
    assert value.normalization == "none"
    assert value.score_semantics == CONTRIEVER_CONFIG.score_semantics
    assert value.index_type == "faiss.IndexFlatIP"
    assert value.index_config == CONTRIEVER_CONFIG.scientific_json()
    assert json.loads(value.index_config) == CONTRIEVER_CONFIG.scientific_payload()
    assert value.index_fingerprint_sha256 == identity.fingerprint_sha256
    assert value.index_artifact_sha256 == FAISS_SHA


def test_builder_and_validator_are_deterministic():
    identity = cache_identity()
    first = provenance(identity)
    second = provenance(identity)
    assert first == second
    validate_contriever_index_binding(
        corpus_provenance=corpus_provenance(),
        retriever_provenance=first,
        cache_identity=identity,
    )


def test_builder_rejects_wrong_types_and_empty_transformers_version():
    with pytest.raises(TypeError, match="ContrieverCacheIdentity"):
        build_contriever_retriever_provenance(
            cache_identity={},
            index_artifact_sha256=FAISS_SHA,
            transformers_version=TRANSFORMERS_VERSION,
        )
    with pytest.raises(TypeError, match="ContrieverConfig"):
        build_contriever_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=FAISS_SHA,
            transformers_version=TRANSFORMERS_VERSION,
            contriever_config={},
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_contriever_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=FAISS_SHA,
            transformers_version=" ",
        )


@pytest.mark.parametrize("value", ["", "A" * 64, "g" * 64, "0" * 63])
def test_builder_rejects_malformed_physical_faiss_sha(value):
    with pytest.raises(ValueError, match="index_artifact_sha256"):
        build_contriever_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=value,
            transformers_version=TRANSFORMERS_VERSION,
        )


def test_builder_and_validator_reject_config_cache_identity_mismatch():
    changed = replace(CONTRIEVER_CONFIG, document_max_length=256)
    with pytest.raises(ValueError, match="ContrieverConfig does not match"):
        build_contriever_retriever_provenance(
            cache_identity=cache_identity(),
            index_artifact_sha256=FAISS_SHA,
            transformers_version=TRANSFORMERS_VERSION,
            contriever_config=changed,
        )
    with pytest.raises(ValueError, match="ContrieverConfig does not match"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=provenance(),
            cache_identity=cache_identity(),
            contriever_config=changed,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("retriever_name", "not-contriever"),
        ("implementation", "wrong.implementation"),
        ("library_name", "not-transformers"),
        ("model_id", "wrong-model"),
        ("model_revision", "0" * 40),
        ("tokenizer_id", "wrong-tokenizer"),
        ("tokenizer_revision", "0" * 40),
        ("query_preprocessing", "wrong preprocessing"),
        ("document_preprocessing", "wrong preprocessing"),
        ("normalization", "l2"),
        ("score_semantics", "cosine"),
        ("index_type", "faiss.IndexFlatL2"),
        ("index_fingerprint_sha256", "0" * 64),
    ],
)
def test_validator_rejects_wrong_declared_fields(field, value):
    identity = cache_identity()
    with pytest.raises(ValueError, match=field):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(provenance(identity), **{field: value}),
            cache_identity=identity,
        )


@pytest.mark.parametrize(
    "index_config",
    ["not-json", "null", "5", '"x"', "[]", "true"],
)
def test_index_config_must_be_json_object(index_config):
    identity = cache_identity()
    with pytest.raises(ValueError, match="index_config"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_config=index_config
            ),
            cache_identity=identity,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("pooling"),
        lambda payload: payload.update(pooling="cls"),
        lambda payload: payload.update(normalization="l2"),
        lambda payload: payload.update(query_max_length=256),
        lambda payload: payload.update(document_max_length=256),
        lambda payload: payload.update(query_batch_size=32),
        lambda payload: payload.update(document_batch_size=32),
        lambda payload: payload.update(model_revision="0" * 40),
        lambda payload: payload.update(tokenizer_revision="0" * 40),
    ],
)
def test_validator_rejects_partial_or_altered_complete_config(mutation):
    identity = cache_identity()
    payload = CONTRIEVER_CONFIG.scientific_payload()
    mutation(payload)
    with pytest.raises(ValueError, match="complete ContrieverConfig"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_config=canonical_json(payload)
            ),
            cache_identity=identity,
        )


def test_validator_rejects_noncanonical_equivalent_config_json():
    identity = cache_identity()
    noncanonical = json.dumps(CONTRIEVER_CONFIG.scientific_payload(), indent=2)
    with pytest.raises(ValueError, match="canonical ContrieverConfig JSON"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_config=noncanonical
            ),
            cache_identity=identity,
        )


def test_validator_rejects_corpus_manifest_sha_or_count_mismatch():
    identity = cache_identity()
    value = provenance(identity)
    corpus = corpus_provenance()
    with pytest.raises(ValueError, match="CorpusManifest SHA"):
        validate_contriever_index_binding(
            corpus_provenance=replace(corpus, manifest_sha256="0" * 64),
            retriever_provenance=value,
            cache_identity=identity,
        )
    with pytest.raises(ValueError, match="document count"):
        validate_contriever_index_binding(
            corpus_provenance=replace(corpus, document_count=3),
            retriever_provenance=value,
            cache_identity=identity,
        )


def test_validator_requires_valid_physical_faiss_sha():
    identity = cache_identity()
    with pytest.raises(ValueError, match="physical FAISS artifact SHA"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=replace(
                provenance(identity), index_artifact_sha256=None
            ),
            cache_identity=identity,
        )
    malformed = provenance(identity)
    object.__setattr__(malformed, "index_artifact_sha256", "bad")
    with pytest.raises(ValueError, match="index_artifact_sha256"):
        validate_contriever_index_binding(
            corpus_provenance=corpus_provenance(),
            retriever_provenance=malformed,
            cache_identity=identity,
        )


def test_validator_rejects_wrong_argument_types():
    identity = cache_identity()
    value = provenance(identity)
    corpus = corpus_provenance()
    with pytest.raises(TypeError, match="corpus_provenance"):
        validate_contriever_index_binding(
            corpus_provenance={},
            retriever_provenance=value,
            cache_identity=identity,
        )
    with pytest.raises(TypeError, match="retriever_provenance"):
        validate_contriever_index_binding(
            corpus_provenance=corpus,
            retriever_provenance={},
            cache_identity=identity,
        )
    with pytest.raises(TypeError, match="cache_identity"):
        validate_contriever_index_binding(
            corpus_provenance=corpus,
            retriever_provenance=value,
            cache_identity={},
        )
    with pytest.raises(TypeError, match="contriever_config"):
        validate_contriever_index_binding(
            corpus_provenance=corpus,
            retriever_provenance=value,
            cache_identity=identity,
            contriever_config={},
        )
