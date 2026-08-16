from dataclasses import replace
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrievers.dpr_original_retriever as dpr_module
from retrievers.dpr_config import DPR_CONFIG
from retrievers.dpr_original_retriever import OriginalDPRRetriever


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        batch_size = len(args[0])
        return {"input_ids": torch.ones((batch_size, 2), dtype=torch.long)}


class FakeEncoder:
    def __init__(self, values=None):
        self.values = values
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        if self.values is None:
            values = torch.arange(
                batch_size * DPR_CONFIG.embedding_dimension, dtype=torch.float32
            ).reshape(batch_size, DPR_CONFIG.embedding_dimension)
        else:
            values = self.values[:batch_size]
        return SimpleNamespace(pooler_output=values)


@pytest.fixture
def fake_model_loaders(monkeypatch):
    tokenizer_calls = []
    question_calls = []
    context_calls = []
    question_tokenizer = FakeTokenizer()
    context_tokenizer = FakeTokenizer()
    question_encoder = FakeEncoder()
    context_encoder = FakeEncoder()

    def load_tokenizer(model_id, **kwargs):
        tokenizer_calls.append((model_id, kwargs))
        if model_id == DPR_CONFIG.question_tokenizer_id:
            return question_tokenizer
        if model_id == DPR_CONFIG.context_tokenizer_id:
            return context_tokenizer
        raise AssertionError(f"unexpected tokenizer ID: {model_id}")

    def load_question(model_id, **kwargs):
        question_calls.append((model_id, kwargs))
        return question_encoder

    def load_context(model_id, **kwargs):
        context_calls.append((model_id, kwargs))
        return context_encoder

    monkeypatch.setattr(dpr_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        dpr_module.DPRQuestionEncoder, "from_pretrained", load_question
    )
    monkeypatch.setattr(dpr_module.DPRContextEncoder, "from_pretrained", load_context)

    return SimpleNamespace(
        tokenizer_calls=tokenizer_calls,
        question_calls=question_calls,
        context_calls=context_calls,
        question_tokenizer=question_tokenizer,
        context_tokenizer=context_tokenizer,
        question_encoder=question_encoder,
        context_encoder=context_encoder,
    )


def test_models_and_tokenizers_load_exact_pinned_ids_and_revisions(
    fake_model_loaders,
):
    retriever = OriginalDPRRetriever()
    retriever._load_models()

    assert fake_model_loaders.tokenizer_calls == [
        (
            DPR_CONFIG.question_tokenizer_id,
            {"revision": DPR_CONFIG.question_tokenizer_revision},
        ),
        (
            DPR_CONFIG.context_tokenizer_id,
            {"revision": DPR_CONFIG.context_tokenizer_revision},
        ),
    ]
    assert fake_model_loaders.question_calls == [
        (
            DPR_CONFIG.question_model_id,
            {"revision": DPR_CONFIG.question_model_revision},
        )
    ]
    assert fake_model_loaders.context_calls == [
        (
            DPR_CONFIG.context_model_id,
            {"revision": DPR_CONFIG.context_model_revision},
        )
    ]


def test_query_encoding_uses_exact_text_and_config(fake_model_loaders):
    config = replace(DPR_CONFIG, query_max_length=123, query_batch_size=2)
    retriever = OriginalDPRRetriever(config)
    embeddings = retriever._encode_questions(["Exact Query?", "Second", "Third"])

    calls = fake_model_loaders.question_tokenizer.calls
    assert [call[0][0] for call in calls] == [
        ["Exact Query?", "Second"],
        ["Third"],
    ]
    assert all(call[1]["padding"] is True for call in calls)
    assert all(call[1]["truncation"] is config.truncation_enabled for call in calls)
    assert all(call[1]["max_length"] == 123 for call in calls)
    assert embeddings.shape == (3, config.embedding_dimension)
    assert embeddings.dtype == np.dtype(config.embedding_dtype)


def test_context_encoding_uses_exact_retrieval_content_and_config(
    fake_model_loaders,
):
    config = replace(DPR_CONFIG, context_max_length=124, context_batch_size=2)
    retriever = OriginalDPRRetriever(config)
    documents = [
        {"retrieval_content": " Exact passage ", "text": "wrong stored text"},
        {"retrieval_content": "Second passage"},
        {"retrieval_content": "Third passage"},
    ]
    embeddings = retriever._encode_contexts(documents)

    calls = fake_model_loaders.context_tokenizer.calls
    assert [call[0] for call in calls] == [
        (["", ""], [" Exact passage ", "Second passage"]),
        ([""], ["Third passage"]),
    ]
    assert all(call[1]["padding"] is True for call in calls)
    assert all(call[1]["truncation"] is config.truncation_enabled for call in calls)
    assert all(call[1]["max_length"] == 124 for call in calls)
    assert embeddings.dtype == np.dtype(config.embedding_dtype)


@pytest.mark.parametrize(
    "document,error_type,message",
    [
        ({"text": "not authoritative"}, ValueError, "missing retrieval_content"),
        ({"retrieval_content": 123}, TypeError, "must be a string"),
    ],
)
def test_invalid_retrieval_content_fails_before_model_loading(
    monkeypatch, document, error_type, message
):
    retriever = OriginalDPRRetriever()
    monkeypatch.setattr(
        retriever,
        "_load_models",
        lambda: pytest.fail("invalid documents must fail before model loading"),
    )
    with pytest.raises(error_type, match=message):
        retriever._encode_contexts([document])


def test_pooler_output_is_used_without_normalization(fake_model_loaders):
    values = torch.tensor([[3.0] + [4.0] + [0.0] * 766], dtype=torch.float32)
    fake_model_loaders.question_encoder.values = values
    retriever = OriginalDPRRetriever()
    embedding = retriever._encode_questions(["query"])
    assert embedding[0, :2].tolist() == [3.0, 4.0]
    assert np.linalg.norm(embedding[0]) == pytest.approx(5.0)


def test_unsupported_representation_fails_loudly():
    with pytest.raises(ValueError, match="unsupported DPR representation"):
        OriginalDPRRetriever(replace(DPR_CONFIG, representation="mean_pooling"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalization", "l2"),
        ("index_type", "faiss.IndexFlatL2"),
        ("score_sign_filtering", "positive only"),
    ],
)
def test_unsupported_scoring_or_index_mode_fails_loudly(field, value):
    with pytest.raises(ValueError, match=f"unsupported DPR {field}"):
        OriginalDPRRetriever(replace(DPR_CONFIG, **{field: value}))


def test_generic_index_rejects_corpus_without_scientific_provenance():
    retriever = OriginalDPRRetriever()
    with pytest.raises(ValueError, match="index_from_corpus_records"):
        retriever.index([{"retrieval_content": "one"}])


def test_retrieve_preserves_faiss_order_and_nonpositive_scores(monkeypatch):
    retriever = OriginalDPRRetriever()
    retriever.corpus = [
        {"retrieval_content": "zero"},
        {"retrieval_content": "negative"},
        {"retrieval_content": "unused"},
    ]
    retriever.is_indexed = True
    monkeypatch.setattr(
        retriever,
        "_encode_questions",
        lambda questions: np.zeros((1, DPR_CONFIG.embedding_dimension), dtype=np.float32),
    )
    retriever.faiss_index = SimpleNamespace(
        search=lambda query, top_k: (
            np.array([[0.0, -2.5, -np.inf]], dtype=np.float32),
            np.array([[0, 1, -1]], dtype=np.int64),
        )
    )

    results = retriever.retrieve("query", top_k=3)
    assert [document["retrieval_content"] for document, _ in results] == [
        "zero",
        "negative",
    ]
    assert [score for _, score in results] == [0.0, -2.5]


def test_default_config_object_is_authoritative_runtime_source():
    retriever = OriginalDPRRetriever()
    assert retriever.config is DPR_CONFIG
    assert retriever.embedding_dim == DPR_CONFIG.embedding_dimension
