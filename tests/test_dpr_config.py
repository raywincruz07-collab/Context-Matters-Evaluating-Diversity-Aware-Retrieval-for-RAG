from dataclasses import FrozenInstanceError, replace
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievers.dpr_config import DPR_CONFIG, DPRConfig


QUESTION_REVISION = "d04a52f6d2f96c60117a925e8c24c4043a75f265"
CONTEXT_REVISION = "bb21a3c2b1656d60c6a8e920283bc40dabddadb8"


def test_authoritative_model_and_tokenizer_revisions():
    assert (
        DPR_CONFIG.question_model_id
        == "facebook/dpr-question_encoder-single-nq-base"
    )
    assert DPR_CONFIG.question_model_revision == QUESTION_REVISION
    assert DPR_CONFIG.question_tokenizer_id == DPR_CONFIG.question_model_id
    assert DPR_CONFIG.question_tokenizer_revision == QUESTION_REVISION
    assert DPR_CONFIG.context_model_id == "facebook/dpr-ctx_encoder-single-nq-base"
    assert DPR_CONFIG.context_model_revision == CONTEXT_REVISION
    assert DPR_CONFIG.context_tokenizer_id == DPR_CONFIG.context_model_id
    assert DPR_CONFIG.context_tokenizer_revision == CONTEXT_REVISION


def test_authoritative_encoding_and_scoring_method():
    assert DPR_CONFIG.query_max_length == 256
    assert DPR_CONFIG.context_max_length == 256
    assert DPR_CONFIG.truncation_enabled is True
    assert DPR_CONFIG.paired_truncation_strategy == (
        "delegated to tokenizer/Transformers default"
    )
    assert DPR_CONFIG.padding_semantics == (
        "dynamic padding to longest encoded input in each batch"
    )
    assert DPR_CONFIG.context_title_policy == "empty title paired with passage text"
    assert DPR_CONFIG.representation == "pooler_output"
    assert DPR_CONFIG.embedding_dimension == 768
    assert DPR_CONFIG.embedding_dtype == "float32"
    assert DPR_CONFIG.normalization == "none"
    assert DPR_CONFIG.score_semantics == (
        "raw dot product / inner product of unnormalized embeddings"
    )
    assert DPR_CONFIG.index_type == "faiss.IndexFlatIP"
    assert DPR_CONFIG.score_sign_filtering == "none"
    assert DPR_CONFIG.context_batch_size == 16
    assert DPR_CONFIG.query_batch_size == 1


def test_scientific_json_is_canonical_and_deterministic():
    first = DPR_CONFIG.scientific_json()
    second = DPR_CONFIG.scientific_json()
    assert first == second
    assert json.loads(first) == DPR_CONFIG.scientific_payload()
    assert first == json.dumps(
        DPR_CONFIG.scientific_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.mark.parametrize(
    "changed",
    [
        replace(DPR_CONFIG, question_model_revision="0" * 40),
        replace(DPR_CONFIG, context_model_revision="1" * 40),
        replace(DPR_CONFIG, query_max_length=128),
        replace(DPR_CONFIG, context_max_length=128),
        replace(DPR_CONFIG, representation="different representation"),
        replace(DPR_CONFIG, normalization="l2"),
        replace(DPR_CONFIG, score_semantics="different scoring"),
        replace(DPR_CONFIG, index_type="different index"),
    ],
)
def test_scientific_method_changes_change_json(changed):
    assert changed.scientific_json() != DPR_CONFIG.scientific_json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("question_model_revision", "main"),
        ("question_tokenizer_revision", "A" * 40),
        ("context_model_revision", "a" * 39),
        ("context_tokenizer_revision", "g" * 40),
    ],
)
def test_invalid_revision_rejected(field, value):
    with pytest.raises(ValueError, match="full lowercase 40-character hexadecimal SHA"):
        replace(DPR_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_max_length", 0),
        ("context_max_length", True),
        ("embedding_dimension", -1),
        ("context_batch_size", 0),
        ("query_batch_size", False),
    ],
)
def test_positive_integer_invariants(field, value):
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        replace(DPR_CONFIG, **{field: value})


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        DPR_CONFIG.query_max_length = 128


def test_scientific_payload_excludes_production_metadata():
    payload = DPR_CONFIG.scientific_payload()
    forbidden = {
        "cache_path",
        "git_commit",
        "hostname",
        "cuda_device",
        "timestamp",
    }
    assert forbidden.isdisjoint(payload)
    assert set(payload) == set(DPRConfig.__dataclass_fields__)
