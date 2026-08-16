from dataclasses import replace
import os
import sys
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
import retrievers.contriever_retriever as contriever_module
from retrievers.contriever_config import CONTRIEVER_CONFIG
from retrievers.contriever_retriever import ContrieverRetriever
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    document_content_sha256,
)
from retrieval_artifacts.contriever_cache_identity import (
    build_contriever_cache_identity,
)


def _entry(record):
    return CorpusManifestEntry(
        position=record.corpus_position,
        doc_id=record.document_id,
        source_document_id=record.source_document_id,
        title_sha256=(
            None
            if record.title is None
            else document_content_sha256(record.title)
        ),
        text_sha256=document_content_sha256(record.text),
        retrieval_content_sha256=document_content_sha256(
            record.retrieval_content
        ),
    )


def _records():
    return (
        CorpusRecord(7, "source-7", None, "wrong stored text", " Exact A! ", 0),
        CorpusRecord("b", "source-b", "unused title", "other text", "Exact B?", 1),
    )


def _manifest(records=None):
    records = _records() if records is None else records
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic",
        config=None,
        revision="fixture-revision",
        split="train",
        construction_algorithm="fixture.v1",
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
        entries=tuple(_entry(record) for record in records),
    )


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        count = len(texts)
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
        }


class _Model:
    def __init__(self):
        self.to_calls = []
        self.float_calls = 0
        self.eval_calls = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def float(self):
        self.float_calls += 1
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, **inputs):
        count, tokens = inputs["input_ids"].shape
        hidden = torch.zeros(
            (count, tokens, CONTRIEVER_CONFIG.embedding_dimension),
            dtype=torch.float32,
        )
        hidden[:, :, 0] = 3.0
        hidden[:, :, 1] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def loaders(monkeypatch):
    monkeypatch.setattr(contriever_module.torch.cuda, "is_available", lambda: False)
    tokenizer = _Tokenizer()
    model = _Model()
    tokenizer_calls = []
    model_calls = []

    def load_tokenizer(identifier, **kwargs):
        tokenizer_calls.append((identifier, kwargs))
        return tokenizer

    def load_model(identifier, **kwargs):
        model_calls.append((identifier, kwargs))
        return model

    monkeypatch.setattr(
        contriever_module.AutoTokenizer, "from_pretrained", load_tokenizer
    )
    monkeypatch.setattr(contriever_module.AutoModel, "from_pretrained", load_model)
    return SimpleNamespace(
        tokenizer=tokenizer,
        model=model,
        tokenizer_calls=tokenizer_calls,
        model_calls=model_calls,
    )


def test_wrong_config_type_rejected():
    with pytest.raises(TypeError, match="ContrieverConfig"):
        ContrieverRetriever(config={})


@pytest.mark.parametrize(
    "field,value",
    [
        ("pooling", "cls"),
        ("normalization", "l2"),
        ("compute_dtype", "float16"),
        ("autocast", True),
        ("index_type", "faiss.IndexFlatL2"),
        ("index_device", "cuda"),
        ("model_loader", "SentenceTransformer"),
        ("tokenizer_loader", "OtherTokenizer"),
    ],
)
def test_unsupported_runtime_modes_fail_loudly(field, value):
    with pytest.raises(ValueError, match=f"unsupported Contriever {field}"):
        ContrieverRetriever(replace(CONTRIEVER_CONFIG, **{field: value}))


def test_runtime_accepts_separate_nondefault_query_and_document_lengths():
    config = replace(
        CONTRIEVER_CONFIG,
        query_max_length=123,
        document_max_length=321,
    )
    retriever = ContrieverRetriever(config)
    assert retriever.config.query_max_length == 123
    assert retriever.config.document_max_length == 321


def test_model_loading_is_pinned_lazy_idempotent_float32_and_eval(loaders):
    retriever = ContrieverRetriever()
    assert retriever.model is None and retriever.tokenizer is None
    retriever._load_model()
    retriever._load_model()
    assert loaders.tokenizer_calls == [
        (
            CONTRIEVER_CONFIG.tokenizer_id,
            {"revision": CONTRIEVER_CONFIG.tokenizer_revision},
        )
    ]
    assert loaders.model_calls == [
        (
            CONTRIEVER_CONFIG.model_id,
            {"revision": CONTRIEVER_CONFIG.model_revision},
        )
    ]
    assert loaders.model.to_calls == ["cpu"]
    assert loaders.model.float_calls == 1
    assert loaders.model.eval_calls == 1


def test_query_tokenization_preserves_exact_text_and_uses_config(loaders):
    config = replace(CONTRIEVER_CONFIG, query_batch_size=2, query_max_length=123)
    retriever = ContrieverRetriever(config)
    result = retriever._encode_queries([" Exact Query?! ", "Second", "Third"])
    assert [texts for texts, _ in loaders.tokenizer.calls] == [
        [" Exact Query?! ", "Second"],
        ["Third"],
    ]
    assert all(call[1]["padding"] is True for call in loaders.tokenizer.calls)
    assert all(call[1]["truncation"] is True for call in loaders.tokenizer.calls)
    assert all(call[1]["max_length"] == 123 for call in loaders.tokenizer.calls)
    assert all(call[1]["return_tensors"] == "pt" for call in loaders.tokenizer.calls)
    assert result.dtype == np.float32
    assert np.linalg.norm(result[0]) == pytest.approx(5.0)


def test_document_tokenization_uses_only_exact_retrieval_content(loaders):
    config = replace(CONTRIEVER_CONFIG, document_batch_size=1, document_max_length=321)
    retriever = ContrieverRetriever(config)
    documents = [
        {
            "doc_id": 7,
            "retrieval_content": " Exact passage! ",
            "text": "wrong stored text",
            "title": "unused",
        },
        {"doc_id": "b", "retrieval_content": "Second"},
    ]
    retriever._encode_documents(documents)
    assert [texts for texts, _ in loaders.tokenizer.calls] == [
        [" Exact passage! "],
        ["Second"],
    ]
    assert all(call[1]["max_length"] == 321 for call in loaders.tokenizer.calls)


@pytest.mark.parametrize(
    "document,error,message",
    [
        ({"text": "wrong"}, ValueError, "missing retrieval_content"),
        ({"retrieval_content": 3}, TypeError, "must be a string"),
    ],
)
def test_invalid_document_content_fails_before_loading(
    monkeypatch, document, error, message
):
    retriever = ContrieverRetriever()
    monkeypatch.setattr(
        retriever,
        "_load_model",
        lambda: pytest.fail("model loading must not occur"),
    )
    with pytest.raises(error, match=message):
        retriever._encode_documents([document])


def test_attention_mask_mean_pooling_ignores_padding_and_does_not_normalize():
    hidden = torch.tensor([[[3.0, 4.0], [5.0, 8.0], [999.0, 999.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = ContrieverRetriever._mean_pool(
        SimpleNamespace(last_hidden_state=hidden), mask
    )
    assert pooled.tolist() == [[4.0, 6.0]]
    assert torch.linalg.vector_norm(pooled).item() == pytest.approx(np.sqrt(52))


def test_zero_attention_mask_row_fails():
    output = SimpleNamespace(last_hidden_state=torch.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="zero-token row"):
        ContrieverRetriever._mean_pool(output, torch.zeros((1, 2), dtype=torch.long))


def test_missing_last_hidden_state_fails():
    with pytest.raises(ValueError, match="last_hidden_state"):
        ContrieverRetriever._mean_pool(SimpleNamespace(), torch.ones((1, 2)))


def test_encoding_does_not_invoke_autocast(loaders, monkeypatch):
    monkeypatch.setattr(
        contriever_module.torch,
        "autocast",
        lambda *args, **kwargs: pytest.fail("autocast must not be used"),
    )
    retriever = ContrieverRetriever()
    retriever._encode_queries(["query"])


@pytest.mark.parametrize(
    "embeddings,error,message",
    [
        (np.zeros((2, 767), dtype=np.float32), ValueError, "shape"),
        (np.zeros((2, 768), dtype=np.float64), ValueError, "dtype"),
        (
            np.full((2, 768), np.nan, dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.full((2, 768), np.inf, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_embedding_validation_rejects_wrong_shape_dtype_and_nonfinite(
    embeddings, error, message
):
    with pytest.raises(error, match=message):
        ContrieverRetriever()._validate_embeddings(embeddings, document_count=2)


def test_generic_index_requires_validated_corpus_path():
    with pytest.raises(ValueError, match="index_from_corpus_records"):
        ContrieverRetriever().index([{"retrieval_content": "text"}])


def test_manifest_drift_fails_before_model_encoding(monkeypatch):
    records = _records()
    changed = (replace(records[0], retrieval_content="changed"), records[1])
    retriever = ContrieverRetriever()
    monkeypatch.setattr(
        retriever,
        "_encode_documents",
        lambda documents: pytest.fail("encoding must not occur"),
    )
    with pytest.raises(ValueError, match="retrieval_content"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(records), corpus_records=changed
        )


def test_validated_index_builds_exact_flat_ip_and_cache_identity(monkeypatch):
    records = _records()
    manifest = _manifest(records)
    embeddings = np.zeros((2, 768), dtype=np.float32)
    embeddings[:, 0] = [1.0, -1.0]
    retriever = ContrieverRetriever()
    monkeypatch.setattr(retriever, "_encode_documents", lambda documents: embeddings)
    retriever.index_from_corpus_records(
        corpus_manifest=manifest, corpus_records=records
    )
    assert isinstance(retriever.faiss_index, faiss.IndexFlatIP)
    assert retriever.faiss_index.d == 768
    assert retriever.faiss_index.ntotal == 2
    assert retriever.corpus == [
        {"doc_id": 7, "retrieval_content": " Exact A! "},
        {"doc_id": "b", "retrieval_content": "Exact B?"},
    ]
    assert retriever.cache_identity == build_contriever_cache_identity(
        corpus_manifest=manifest, contriever_config=CONTRIEVER_CONFIG
    )
    assert retriever.is_indexed is True


class _SearchIndex:
    def search(self, query_embedding, top_k):
        assert query_embedding.dtype == np.float32
        assert top_k == 4
        return (
            np.array([[-2.5, 0.0, -999.0, 7.0]], dtype=np.float32),
            np.array([[2, 0, -1, 1]], dtype=np.int64),
        )


def test_retrieve_preserves_faiss_order_scores_and_only_omits_sentinel(monkeypatch):
    retriever = ContrieverRetriever()
    retriever.corpus = [
        {"doc_id": 0, "retrieval_content": "zero"},
        {"doc_id": 1, "retrieval_content": "positive"},
        {"doc_id": 2, "retrieval_content": "negative"},
    ]
    retriever.faiss_index = _SearchIndex()
    retriever.is_indexed = True
    monkeypatch.setattr(
        retriever,
        "_encode_queries",
        lambda queries: np.zeros((1, 768), dtype=np.float32),
    )
    results = retriever.retrieve(" Exact query ", top_k=4)
    assert [(document["doc_id"], score) for document, score in results] == [
        (2, -2.5),
        (0, 0.0),
        (1, 7.0),
    ]


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.5, "2"])
def test_invalid_top_k_rejected(top_k):
    retriever = ContrieverRetriever()
    retriever.is_indexed = True
    expected = TypeError if isinstance(top_k, bool) or not isinstance(top_k, int) else ValueError
    with pytest.raises(expected, match="top_k"):
        retriever.retrieve("query", top_k=top_k)


def test_retrieve_requires_index_and_string_query():
    with pytest.raises(RuntimeError, match="Index not built"):
        ContrieverRetriever().retrieve("query")
    retriever = ContrieverRetriever()
    retriever.is_indexed = True
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(123)


def test_unload_model_preserves_index_and_corpus(monkeypatch):
    retriever = ContrieverRetriever()
    retriever.model = object()
    retriever.tokenizer = object()
    retriever.faiss_index = object()
    retriever.corpus = [{"doc_id": 1, "retrieval_content": "x"}]
    retriever.is_indexed = True
    monkeypatch.setattr(contriever_module.torch.cuda, "is_available", lambda: False)
    retriever.unload_model()
    assert retriever.model is None and retriever.tokenizer is None
    assert retriever.faiss_index is not None
    assert retriever.corpus == [{"doc_id": 1, "retrieval_content": "x"}]
    assert retriever.is_indexed is True
