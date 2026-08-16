from dataclasses import FrozenInstanceError, replace
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievers.contriever_config import CONTRIEVER_CONFIG, ContrieverConfig


REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"


def test_every_frozen_contriever_field():
    assert CONTRIEVER_CONFIG.model_id == "facebook/contriever"
    assert CONTRIEVER_CONFIG.model_revision == REVISION
    assert CONTRIEVER_CONFIG.tokenizer_id == "facebook/contriever"
    assert CONTRIEVER_CONFIG.tokenizer_revision == REVISION
    assert CONTRIEVER_CONFIG.model_loader == "AutoModel"
    assert CONTRIEVER_CONFIG.tokenizer_loader == "AutoTokenizer"
    assert CONTRIEVER_CONFIG.query_preprocessing == (
        "exact query string passed to tokenizer; no manual lowercasing or external "
        "text normalization; tokenizer-native behavior preserved"
    )
    assert CONTRIEVER_CONFIG.document_preprocessing == (
        "exact validated CorpusRecord.retrieval_content passed to tokenizer; no "
        "manual lowercasing or external text normalization; tokenizer-native "
        "behavior preserved"
    )
    assert CONTRIEVER_CONFIG.pooling == (
        "attention-mask-aware mean pooling of last_hidden_state"
    )
    assert CONTRIEVER_CONFIG.embedding_dimension == 768
    assert CONTRIEVER_CONFIG.query_max_length == 512
    assert CONTRIEVER_CONFIG.document_max_length == 512
    assert CONTRIEVER_CONFIG.truncation_enabled is True
    assert CONTRIEVER_CONFIG.truncation_semantics == (
        "single-sequence truncation delegated to tokenizer/Transformers with "
        "max_length=512"
    )
    assert CONTRIEVER_CONFIG.padding_semantics == (
        "dynamic padding to longest encoded input in each batch"
    )
    assert CONTRIEVER_CONFIG.normalization == "none"
    assert CONTRIEVER_CONFIG.compute_dtype == "float32"
    assert CONTRIEVER_CONFIG.autocast is False
    assert CONTRIEVER_CONFIG.embedding_dtype == "float32"
    assert CONTRIEVER_CONFIG.score_semantics == (
        "raw dot product / inner product of unnormalized embeddings"
    )
    assert CONTRIEVER_CONFIG.ranking_direction == "higher score is better"
    assert CONTRIEVER_CONFIG.index_type == "faiss.IndexFlatIP"
    assert CONTRIEVER_CONFIG.index_device == "cpu"
    assert CONTRIEVER_CONFIG.score_sign_filtering == "none"
    assert CONTRIEVER_CONFIG.query_batch_size == 64
    assert CONTRIEVER_CONFIG.document_batch_size == 64


def test_config_is_frozen_and_payload_contains_exactly_all_fields():
    with pytest.raises(FrozenInstanceError):
        CONTRIEVER_CONFIG.query_max_length = 256
    assert set(CONTRIEVER_CONFIG.scientific_payload()) == set(
        ContrieverConfig.__dataclass_fields__
    )


def test_scientific_json_is_canonical_stable_and_deterministic():
    first = CONTRIEVER_CONFIG.scientific_json()
    second = CONTRIEVER_CONFIG.scientific_json()
    assert first == second
    assert json.loads(first) == CONTRIEVER_CONFIG.scientific_payload()
    assert first == json.dumps(
        CONTRIEVER_CONFIG.scientific_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.mark.parametrize(
    "changed",
    [
        replace(CONTRIEVER_CONFIG, model_revision="0" * 40),
        replace(CONTRIEVER_CONFIG, tokenizer_revision="1" * 40),
        replace(CONTRIEVER_CONFIG, query_max_length=256),
        replace(CONTRIEVER_CONFIG, document_max_length=256),
        replace(CONTRIEVER_CONFIG, embedding_dimension=384),
        replace(CONTRIEVER_CONFIG, query_batch_size=32),
        replace(CONTRIEVER_CONFIG, document_batch_size=32),
        replace(CONTRIEVER_CONFIG, query_preprocessing="different exact policy"),
        replace(
            CONTRIEVER_CONFIG,
            document_preprocessing="different exact document policy",
        ),
        replace(CONTRIEVER_CONFIG, model_loader="DifferentModelLoader"),
        replace(CONTRIEVER_CONFIG, tokenizer_loader="DifferentTokenizerLoader"),
        replace(CONTRIEVER_CONFIG, pooling="different pooling"),
        replace(CONTRIEVER_CONFIG, normalization="l2"),
        replace(CONTRIEVER_CONFIG, compute_dtype="float16"),
        replace(CONTRIEVER_CONFIG, embedding_dtype="float64"),
        replace(CONTRIEVER_CONFIG, score_semantics="cosine similarity"),
        replace(CONTRIEVER_CONFIG, ranking_direction="lower score is better"),
        replace(CONTRIEVER_CONFIG, index_type="faiss.IndexFlatL2"),
        replace(CONTRIEVER_CONFIG, index_device="cuda"),
        replace(CONTRIEVER_CONFIG, score_sign_filtering="positive only"),
        replace(CONTRIEVER_CONFIG, truncation_enabled=False),
        replace(CONTRIEVER_CONFIG, autocast=True),
    ],
)
def test_scientifically_relevant_changes_alter_scientific_json(changed):
    assert changed.scientific_json() != CONTRIEVER_CONFIG.scientific_json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "main"),
        ("model_revision", "A" * 40),
        ("tokenizer_revision", "a" * 39),
        ("tokenizer_revision", "g" * 40),
    ],
)
def test_invalid_revision_rejected(field, value):
    with pytest.raises(ValueError, match="full lowercase 40-character"):
        replace(CONTRIEVER_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_dimension", 0),
        ("embedding_dimension", True),
        ("query_max_length", -1),
        ("query_max_length", False),
        ("document_max_length", 0),
        ("document_max_length", True),
        ("query_batch_size", 0),
        ("query_batch_size", False),
        ("document_batch_size", -1),
        ("document_batch_size", True),
    ],
)
def test_invalid_dimensions_lengths_and_batch_sizes_rejected(field, value):
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        replace(CONTRIEVER_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("truncation_enabled", 1),
        ("truncation_enabled", "true"),
        ("autocast", 0),
        ("autocast", "false"),
    ],
)
def test_boolean_fields_require_strict_bool_type(field, value):
    with pytest.raises(TypeError, match=field):
        replace(CONTRIEVER_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_loader",
        "tokenizer_loader",
        "query_preprocessing",
        "document_preprocessing",
        "pooling",
        "truncation_semantics",
        "padding_semantics",
        "normalization",
        "compute_dtype",
        "embedding_dtype",
        "score_semantics",
        "ranking_direction",
        "index_type",
        "index_device",
        "score_sign_filtering",
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_every_string_field_rejects_empty_or_whitespace(field, value):
    with pytest.raises(ValueError, match=field):
        replace(CONTRIEVER_CONFIG, **{field: value})
