from dataclasses import FrozenInstanceError, replace
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievers.colbert_config import COLBERT_CONFIG, ColBERTConfig


REVISION = "0855eac81381e0323a846f1ed7d8452d4c648b50"

SCIENTIFIC_MUTATIONS = {
    "retriever_name": "different-retriever",
    "backend": "different backend",
    "colbert_ai_version": "0.2.21",
    "checkpoint_id": "different/checkpoint",
    "checkpoint_revision": "0" * 40,
    "checkpoint_loading": "different immutable loading policy",
    "dim": 64,
    "query_maxlen": 64,
    "doc_maxlen": 256,
    "similarity": "dot product",
    "interaction": "different interaction",
    "mask_punctuation": False,
    "attend_to_mask_tokens": True,
    "query_preprocessing": "different query policy",
    "document_preprocessing": "different document policy",
    "index_engine": "different index engine",
    "nbits": 4,
    "kmeans_niters": 8,
    "index_bsize": 32,
    "nranks": 2,
    "amp": False,
    "seed": 54321,
    "candidate_pool_size": 10,
    "search_ncells": 4,
    "search_centroid_score_threshold": 0.5,
    "search_ndocs": 512,
    "ranking": "different ranking",
    "post_filtering": "positive only",
    "post_reranking": "enabled",
    "tie_manipulation": "corpus position",
}


def test_every_frozen_colbert_field():
    assert COLBERT_CONFIG.retriever_name == "colbertv2"
    assert COLBERT_CONFIG.backend == "direct Stanford ColBERT"
    assert COLBERT_CONFIG.colbert_ai_version == "0.2.22"
    assert COLBERT_CONFIG.checkpoint_id == "colbert-ir/colbertv2.0"
    assert COLBERT_CONFIG.checkpoint_revision == REVISION
    assert COLBERT_CONFIG.checkpoint_loading == (
        "resolve the pinned revision without fallback; validate and fingerprint "
        "the physical checkpoint snapshot before canonical indexing"
    )
    assert COLBERT_CONFIG.dim == 128
    assert COLBERT_CONFIG.query_maxlen == 32
    assert COLBERT_CONFIG.doc_maxlen == 180
    assert COLBERT_CONFIG.similarity == "cosine"
    assert COLBERT_CONFIG.interaction == "ColBERT late interaction / MaxSim"
    assert COLBERT_CONFIG.mask_punctuation is True
    assert COLBERT_CONFIG.attend_to_mask_tokens is False
    assert COLBERT_CONFIG.query_preprocessing == (
        "exact query string; no manual lowercasing or external text normalization; "
        "tokenizer-native behavior preserved"
    )
    assert COLBERT_CONFIG.document_preprocessing == (
        "exact validated CorpusRecord.retrieval_content; no manual lowercasing or "
        "external text normalization; tokenizer-native behavior preserved"
    )
    assert COLBERT_CONFIG.index_engine == "Stanford ColBERT PLAID"
    assert COLBERT_CONFIG.nbits == 2
    assert COLBERT_CONFIG.kmeans_niters == 4
    assert COLBERT_CONFIG.index_bsize == 64
    assert COLBERT_CONFIG.nranks == 1
    assert COLBERT_CONFIG.amp is True
    assert COLBERT_CONFIG.seed == 12345
    assert COLBERT_CONFIG.candidate_pool_size == 20
    assert COLBERT_CONFIG.search_ncells == 2
    assert COLBERT_CONFIG.search_centroid_score_threshold == 0.45
    assert COLBERT_CONFIG.search_ndocs == 1024
    assert COLBERT_CONFIG.ranking == "native ColBERT ranking, higher score is better"
    assert COLBERT_CONFIG.post_filtering == "none"
    assert COLBERT_CONFIG.post_reranking == "none"
    assert COLBERT_CONFIG.tie_manipulation == "none"


def test_config_is_frozen_and_payload_contains_exactly_all_fields():
    with pytest.raises(FrozenInstanceError):
        COLBERT_CONFIG.query_maxlen = 64
    assert set(COLBERT_CONFIG.scientific_payload()) == set(
        ColBERTConfig.__dataclass_fields__
    )
    assert set(SCIENTIFIC_MUTATIONS) == set(ColBERTConfig.__dataclass_fields__)


def test_scientific_json_is_canonical_stable_and_deterministic():
    first = COLBERT_CONFIG.scientific_json()
    second = COLBERT_CONFIG.scientific_json()
    assert first == second
    assert json.loads(first) == COLBERT_CONFIG.scientific_payload()
    assert first == json.dumps(
        COLBERT_CONFIG.scientific_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.mark.parametrize("field,value", SCIENTIFIC_MUTATIONS.items())
def test_every_scientific_field_change_alters_scientific_json(field, value):
    changed = replace(COLBERT_CONFIG, **{field: value})
    assert changed.scientific_json() != COLBERT_CONFIG.scientific_json()


@pytest.mark.parametrize(
    "value", ["main", "A" * 40, "a" * 39, "g" * 40]
)
def test_invalid_checkpoint_revision_rejected(value):
    with pytest.raises(ValueError, match="full lowercase 40-character"):
        replace(COLBERT_CONFIG, checkpoint_revision=value)


@pytest.mark.parametrize(
    "field",
    [
        "retriever_name",
        "backend",
        "colbert_ai_version",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_loading",
        "similarity",
        "interaction",
        "query_preprocessing",
        "document_preprocessing",
        "index_engine",
        "ranking",
        "post_filtering",
        "post_reranking",
        "tie_manipulation",
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_every_string_field_rejects_empty_or_whitespace(field, value):
    with pytest.raises(ValueError, match=field):
        replace(COLBERT_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim", 0),
        ("query_maxlen", True),
        ("doc_maxlen", -1),
        ("nbits", False),
        ("kmeans_niters", 0),
        ("index_bsize", -1),
        ("nranks", True),
        ("candidate_pool_size", 0),
        ("search_ncells", False),
        ("search_ndocs", -1),
    ],
)
def test_positive_integer_fields_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        replace(COLBERT_CONFIG, **{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_seed_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="non-negative non-boolean integer"):
        replace(COLBERT_CONFIG, seed=value)


@pytest.mark.parametrize("value", [True, "0.45", float("nan"), float("inf")])
def test_search_threshold_requires_finite_non_boolean_real(value):
    with pytest.raises(ValueError, match="finite non-boolean real"):
        replace(COLBERT_CONFIG, search_centroid_score_threshold=value)


def test_search_threshold_is_canonicalized_to_float_in_scientific_identity():
    config = replace(COLBERT_CONFIG, search_centroid_score_threshold=1)
    assert config.scientific_payload()["search_centroid_score_threshold"] == 1.0
    assert '"search_centroid_score_threshold":1.0' in config.scientific_json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("mask_punctuation", 1),
        ("mask_punctuation", "true"),
        ("attend_to_mask_tokens", 0),
        ("attend_to_mask_tokens", "false"),
        ("amp", 1),
        ("amp", "true"),
    ],
)
def test_boolean_fields_require_strict_bool_type(field, value):
    with pytest.raises(TypeError, match=field):
        replace(COLBERT_CONFIG, **{field: value})


def test_scientific_payload_excludes_runtime_and_environment_metadata():
    forbidden = {
        "cache_path",
        "cuda_device",
        "environment_fingerprint",
        "git_commit",
        "hostname",
        "local_checkpoint_path",
        "timestamp",
    }
    assert forbidden.isdisjoint(COLBERT_CONFIG.scientific_payload())
