from dataclasses import replace
import json
import os
from pathlib import Path
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
from retrievers.contriever_retriever import (
    CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION,
)
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


@pytest.fixture(autouse=True)
def _isolated_cache_directories(tmp_path, monkeypatch):
    embeddings_dir = tmp_path / "embeddings"
    indices_dir = tmp_path / "indices"
    monkeypatch.setattr(
        contriever_module, "EMBEDDINGS_DIR", str(embeddings_dir)
    )
    monkeypatch.setattr(contriever_module, "INDEX_DIR", str(indices_dir))
    return embeddings_dir, indices_dir


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


def _synthetic_embeddings(records=None, dtype=np.float32):
    count = len(_records() if records is None else records)
    embeddings = np.zeros((count, CONTRIEVER_CONFIG.embedding_dimension), dtype=dtype)
    embeddings[0, 0] = 1.0
    if count > 1:
        embeddings[1, 0] = -1.0
    return embeddings


def _install_encoder(monkeypatch, retriever, embeddings, calls):
    def encode(documents):
        calls.append(tuple(document["retrieval_content"] for document in documents))
        return np.array(embeddings, copy=True)

    monkeypatch.setattr(retriever, "_encode_documents", encode)


def _build_cache(monkeypatch, *, config=CONTRIEVER_CONFIG, manifest=None, records=None):
    records = _records() if records is None else records
    manifest = _manifest(records) if manifest is None else manifest
    retriever = ContrieverRetriever(config)
    calls = []
    _install_encoder(monkeypatch, retriever, _synthetic_embeddings(records), calls)
    retriever.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    return retriever, calls, manifest, records


def _load_metadata(retriever):
    with open(retriever.cache_metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_metadata(retriever, metadata):
    with open(retriever.cache_metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        file.write("\n")


def test_cache_miss_writes_fingerprinted_complete_cache(monkeypatch):
    retriever, calls, manifest, _ = _build_cache(monkeypatch)
    fingerprint = retriever.cache_identity.fingerprint_sha256
    assert calls == [(" Exact A! ", "Exact B?")]
    assert Path(retriever.embedding_cache_path).name == (
        f"contriever_embeddings_{fingerprint}.npy"
    )
    assert Path(retriever.faiss_cache_path).name == (
        f"contriever_index_{fingerprint}.faiss"
    )
    assert Path(retriever.cache_metadata_path).name == (
        f"contriever_cache_{fingerprint}.json"
    )
    assert Path(retriever.embedding_cache_path).is_file()
    assert Path(retriever.faiss_cache_path).is_file()
    assert Path(retriever.cache_metadata_path).is_file()
    assert isinstance(retriever.faiss_index, faiss.IndexFlatIP)
    assert retriever.faiss_index.d == 768
    assert retriever.faiss_index.ntotal == manifest.document_count

    metadata = _load_metadata(retriever)
    assert metadata == {
        **retriever._expected_metadata_binding(
            manifest, cache_identity=retriever.cache_identity
        ),
        "embedding": {
            "document_count": 2,
            "dtype": "float32",
            "embedding_dimension": 768,
            "filename": Path(retriever.embedding_cache_path).name,
            "sha256": retriever.embedding_artifact_sha256,
            "shape": [2, 768],
        },
        "environment": retriever._environment_metadata(),
        "faiss": {
            "dimension": 768,
            "filename": Path(retriever.faiss_cache_path).name,
            "index_type": "faiss.IndexFlatIP",
            "ntotal": 2,
            "sha256": retriever.index_artifact_sha256,
        },
    }
    assert metadata["cache_schema_version"] == (
        CONTRIEVER_CACHE_METADATA_SCHEMA_VERSION
    )
    assert metadata["embedding"]["sha256"] == contriever_module._physical_sha256(
        retriever.embedding_cache_path
    )
    assert metadata["faiss"]["sha256"] == contriever_module._physical_sha256(
        retriever.faiss_cache_path
    )
    expected_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    assert Path(retriever.cache_metadata_path).read_text(encoding="utf-8") == (
        expected_json
    )


def test_valid_cache_reuse_bypasses_encoding_and_model_loading(monkeypatch):
    original, _, manifest, records = _build_cache(monkeypatch)
    cached = ContrieverRetriever()
    monkeypatch.setattr(
        cached,
        "_encode_documents",
        lambda documents: pytest.fail("valid cache must bypass encoding"),
    )
    monkeypatch.setattr(
        cached,
        "_load_model",
        lambda: pytest.fail("valid cache must bypass model loading"),
    )
    cached.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    assert cached.is_indexed is True
    assert cached.faiss_index.d == original.faiss_index.d
    assert cached.faiss_index.ntotal == original.faiss_index.ntotal
    assert cached.embedding_artifact_sha256 == original.embedding_artifact_sha256
    assert cached.index_artifact_sha256 == original.index_artifact_sha256


def test_cache_reuse_validates_records_before_cache_access(monkeypatch):
    _, _, manifest, records = _build_cache(monkeypatch)
    changed = (replace(records[0], retrieval_content="changed"), records[1])
    retriever = ContrieverRetriever()
    monkeypatch.setattr(
        retriever,
        "_load_validated_cache",
        lambda **kwargs: pytest.fail("cache access must follow record validation"),
    )
    monkeypatch.setattr(
        retriever,
        "_encode_documents",
        lambda documents: pytest.fail("encoding must not occur"),
    )
    with pytest.raises(ValueError, match="retrieval_content"):
        retriever.index_from_corpus_records(
            corpus_manifest=manifest,
            corpus_records=changed,
        )


def test_corpus_and_supported_config_changes_use_distinct_cache_paths(monkeypatch):
    original, _, manifest, records = _build_cache(monkeypatch)
    changed_manifest = replace(manifest, construction_algorithm="fixture.v2")
    changed_corpus, corpus_calls, _, _ = _build_cache(
        monkeypatch,
        manifest=changed_manifest,
        records=records,
    )
    changed_config = replace(
        CONTRIEVER_CONFIG,
        query_batch_size=32,
        document_batch_size=32,
    )
    changed_method, config_calls, _, _ = _build_cache(
        monkeypatch,
        config=changed_config,
        manifest=manifest,
        records=records,
    )
    assert corpus_calls and config_calls
    assert changed_corpus.cache_identity.fingerprint_sha256 != (
        original.cache_identity.fingerprint_sha256
    )
    assert changed_method.cache_identity.fingerprint_sha256 != (
        original.cache_identity.fingerprint_sha256
    )
    assert changed_corpus.embedding_cache_path != original.embedding_cache_path
    assert changed_method.embedding_cache_path != original.embedding_cache_path


@pytest.mark.parametrize(
    "damage",
    [
        "missing_metadata",
        "missing_embedding",
        "missing_faiss",
        "corrupt_embedding",
        "corrupt_faiss",
        "fingerprint",
        "manifest_sha",
        "scientific_json",
        "scientific_payload",
        "embedding_filename",
        "embedding_shape_metadata",
        "embedding_dtype_metadata",
        "faiss_index_type",
        "faiss_dimension_metadata",
        "faiss_ntotal_metadata",
        "missing_environment",
        "nondict_environment",
        "malformed_json",
        "nondict_metadata",
    ],
)
def test_incomplete_corrupt_or_tampered_cache_rebuilds(monkeypatch, damage):
    original, _, manifest, records = _build_cache(monkeypatch)
    metadata = _load_metadata(original)
    if damage == "missing_metadata":
        Path(original.cache_metadata_path).unlink()
    elif damage == "missing_embedding":
        Path(original.embedding_cache_path).unlink()
    elif damage == "missing_faiss":
        Path(original.faiss_cache_path).unlink()
    elif damage == "corrupt_embedding":
        Path(original.embedding_cache_path).write_bytes(b"corrupt npy")
    elif damage == "corrupt_faiss":
        Path(original.faiss_cache_path).write_bytes(b"corrupt faiss")
    elif damage == "fingerprint":
        metadata["cache_fingerprint_sha256"] = "0" * 64
        _write_metadata(original, metadata)
    elif damage == "manifest_sha":
        metadata["corpus_manifest_sha256"] = "0" * 64
        _write_metadata(original, metadata)
    elif damage == "scientific_json":
        metadata["contriever_scientific_json"] = "{}"
        _write_metadata(original, metadata)
    elif damage == "scientific_payload":
        metadata["scientific_payload"] = {}
        _write_metadata(original, metadata)
    elif damage == "embedding_filename":
        metadata["embedding"]["filename"] = "wrong.npy"
        _write_metadata(original, metadata)
    elif damage == "embedding_shape_metadata":
        metadata["embedding"]["shape"] = [1, 768]
        _write_metadata(original, metadata)
    elif damage == "embedding_dtype_metadata":
        metadata["embedding"]["dtype"] = "float64"
        _write_metadata(original, metadata)
    elif damage == "faiss_index_type":
        metadata["faiss"]["index_type"] = "faiss.IndexFlatL2"
        _write_metadata(original, metadata)
    elif damage == "faiss_dimension_metadata":
        metadata["faiss"]["dimension"] = 767
        _write_metadata(original, metadata)
    elif damage == "faiss_ntotal_metadata":
        metadata["faiss"]["ntotal"] = 1
        _write_metadata(original, metadata)
    elif damage == "missing_environment":
        metadata.pop("environment")
        _write_metadata(original, metadata)
    elif damage == "nondict_environment":
        metadata["environment"] = "local"
        _write_metadata(original, metadata)
    elif damage == "malformed_json":
        Path(original.cache_metadata_path).write_text("not json", encoding="utf-8")
    elif damage == "nondict_metadata":
        Path(original.cache_metadata_path).write_text("[]", encoding="utf-8")

    rebuilt = ContrieverRetriever()
    calls = []
    _install_encoder(monkeypatch, rebuilt, _synthetic_embeddings(), calls)
    rebuilt.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    assert calls == [(" Exact A! ", "Exact B?")]
    assert Path(rebuilt.embedding_cache_path).is_file()
    assert Path(rebuilt.faiss_cache_path).is_file()
    assert Path(rebuilt.cache_metadata_path).is_file()


def test_different_environment_metadata_does_not_invalidate_cache(monkeypatch):
    original, _, manifest, records = _build_cache(monkeypatch)
    metadata = _load_metadata(original)
    metadata["environment"] = {"device": "different-machine"}
    _write_metadata(original, metadata)
    cached = ContrieverRetriever()
    monkeypatch.setattr(
        cached,
        "_encode_documents",
        lambda documents: pytest.fail("descriptive environment must not invalidate"),
    )
    cached.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    assert cached.is_indexed is True


def test_cache_loading_disables_numpy_pickle(monkeypatch):
    original, _, manifest, records = _build_cache(monkeypatch)
    real_load = contriever_module.np.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(contriever_module.np, "load", checked_load)
    cached = ContrieverRetriever()
    cached.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    assert calls == [False]


@pytest.mark.parametrize(
    "damage",
    [
        "embedding_shape",
        "embedding_dtype",
        "embedding_nan",
        "embedding_inf",
        "faiss_type",
        "faiss_dimension",
        "faiss_ntotal",
    ],
)
def test_loaded_artifacts_are_semantically_validated(monkeypatch, damage):
    original, _, manifest, records = _build_cache(monkeypatch)
    metadata = _load_metadata(original)
    if damage.startswith("embedding_"):
        if damage == "embedding_shape":
            replacement = np.zeros((1, 768), dtype=np.float32)
        elif damage == "embedding_dtype":
            replacement = _synthetic_embeddings(dtype=np.float64)
        else:
            replacement = _synthetic_embeddings()
            replacement[0, 0] = np.nan if damage == "embedding_nan" else np.inf
        with open(original.embedding_cache_path, "wb") as file:
            np.save(file, replacement, allow_pickle=False)
        metadata["embedding"]["sha256"] = contriever_module._physical_sha256(
            original.embedding_cache_path
        )
    else:
        if damage == "faiss_type":
            replacement_index = faiss.IndexFlatL2(768)
            replacement_index.add(_synthetic_embeddings())
        elif damage == "faiss_dimension":
            replacement_index = faiss.IndexFlatIP(767)
            replacement_index.add(np.zeros((2, 767), dtype=np.float32))
        else:
            replacement_index = faiss.IndexFlatIP(768)
            replacement_index.add(np.zeros((1, 768), dtype=np.float32))
        faiss.write_index(replacement_index, original.faiss_cache_path)
        metadata["faiss"]["sha256"] = contriever_module._physical_sha256(
            original.faiss_cache_path
        )
    _write_metadata(original, metadata)

    rebuilt = ContrieverRetriever()
    calls = []
    _install_encoder(monkeypatch, rebuilt, _synthetic_embeddings(), calls)
    rebuilt.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    assert calls == [(" Exact A! ", "Exact B?")]


def test_nonfinite_new_embeddings_fail_before_cache_write(monkeypatch):
    embeddings = _synthetic_embeddings()
    embeddings[0, 0] = np.nan
    retriever = ContrieverRetriever()
    calls = []
    _install_encoder(monkeypatch, retriever, embeddings, calls)
    with pytest.raises(ValueError, match="finite"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(),
            corpus_records=_records(),
        )
    assert calls
    assert retriever.cache_metadata_path is None


def test_metadata_is_written_last_and_partial_pair_is_not_reused(monkeypatch):
    retriever = ContrieverRetriever()
    calls = []
    _install_encoder(monkeypatch, retriever, _synthetic_embeddings(), calls)
    monkeypatch.setattr(
        retriever,
        "_write_json_atomically",
        lambda path, metadata: (_ for _ in ()).throw(RuntimeError("metadata failure")),
    )
    with pytest.raises(RuntimeError, match="metadata failure"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(),
            corpus_records=_records(),
        )
    identity = build_contriever_cache_identity(corpus_manifest=_manifest())
    embedding_path = Path(contriever_module.EMBEDDINGS_DIR) / (
        identity.embedding_cache_filename
    )
    faiss_path = Path(contriever_module.INDEX_DIR) / identity.faiss_cache_filename
    metadata_path = Path(contriever_module.INDEX_DIR) / (
        f"contriever_cache_{identity.fingerprint_sha256}.json"
    )
    assert embedding_path.is_file()
    assert faiss_path.is_file()
    assert not metadata_path.exists()
    assert retriever.corpus is None
    assert retriever.cache_identity is None
    assert retriever.faiss_index is None
    assert retriever.is_indexed is False

    rebuilt = ContrieverRetriever()
    rebuilt_calls = []
    _install_encoder(monkeypatch, rebuilt, _synthetic_embeddings(), rebuilt_calls)
    rebuilt.index_from_corpus_records(
        corpus_manifest=_manifest(),
        corpus_records=_records(),
    )
    assert rebuilt_calls


def test_atomic_writes_leave_no_temporary_files(monkeypatch):
    retriever, _, _, _ = _build_cache(monkeypatch)
    assert not list(Path(retriever.embedding_cache_path).parent.glob(".*.tmp"))
    assert not list(Path(retriever.faiss_cache_path).parent.glob(".*.tmp"))


def test_atomic_numpy_failure_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "embeddings" / "target.npy"
    destination.parent.mkdir()
    monkeypatch.setattr(
        contriever_module.np,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    with pytest.raises(RuntimeError, match="write failed"):
        ContrieverRetriever._write_numpy_atomically(
            str(destination), _synthetic_embeddings()
        )
    assert not list(destination.parent.glob(".contriever_embeddings_*"))


def test_atomic_faiss_failure_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "indices" / "target.faiss"
    destination.parent.mkdir()
    monkeypatch.setattr(
        contriever_module.faiss,
        "write_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    with pytest.raises(RuntimeError, match="write failed"):
        ContrieverRetriever._write_faiss_atomically(
            str(destination), faiss.IndexFlatIP(768)
        )
    assert not list(destination.parent.glob(".contriever_index_*"))


def test_atomic_json_failure_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "indices" / "target.json"
    destination.parent.mkdir()
    monkeypatch.setattr(
        contriever_module.json,
        "dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    with pytest.raises(RuntimeError, match="write failed"):
        ContrieverRetriever._write_json_atomically(str(destination), {"valid": True})
    assert not list(destination.parent.glob(".contriever_cache_*"))


def test_cached_index_preserves_raw_inner_product_retrieval(monkeypatch):
    fresh, _, manifest, records = _build_cache(monkeypatch)
    query = np.zeros((1, 768), dtype=np.float32)
    query[0, 0] = 1.0
    monkeypatch.setattr(fresh, "_encode_queries", lambda queries: query)
    expected = fresh.retrieve("query", top_k=2)

    cached = ContrieverRetriever()
    monkeypatch.setattr(
        cached,
        "_encode_documents",
        lambda documents: pytest.fail("cache should be reused"),
    )
    cached.index_from_corpus_records(
        corpus_manifest=manifest,
        corpus_records=records,
    )
    monkeypatch.setattr(cached, "_encode_queries", lambda queries: query)
    actual = cached.retrieve("query", top_k=2)
    assert [(document["doc_id"], score) for document, score in actual] == [
        (7, 1.0),
        ("b", -1.0),
    ]
    assert actual == expected


def _records_b():
    return (
        CorpusRecord("b-0", "source-b-0", None, "stored B0", "B positive", 0),
        CorpusRecord("b-1", "source-b-1", None, "stored B1", "B zero", 1),
        CorpusRecord("b-2", "source-b-2", None, "stored B2", "B negative", 2),
    )


def _state_snapshot(retriever):
    return (
        retriever.corpus,
        retriever.cache_identity,
        retriever.embedding_cache_path,
        retriever.faiss_cache_path,
        retriever.cache_metadata_path,
        retriever.faiss_index,
        retriever.embedding_artifact_sha256,
        retriever.index_artifact_sha256,
        retriever.is_indexed,
    )


def _assert_a_retrieval_still_works(monkeypatch, retriever):
    query = np.zeros((1, 768), dtype=np.float32)
    query[0, 0] = 1.0
    monkeypatch.setattr(retriever, "_encode_queries", lambda queries: query)
    results = retriever.retrieve("query", top_k=1)
    assert results[0][0]["doc_id"] == 7
    assert results[0][1] == 1.0


def test_same_instance_successfully_commits_complete_b_state(monkeypatch):
    retriever, _, _, _ = _build_cache(monkeypatch)
    state_a = _state_snapshot(retriever)
    records_b = _records_b()
    manifest_b = _manifest(records_b)
    embeddings_b = _synthetic_embeddings(records_b)
    calls = []
    _install_encoder(monkeypatch, retriever, embeddings_b, calls)

    retriever.index_from_corpus_records(
        corpus_manifest=manifest_b,
        corpus_records=records_b,
    )
    assert calls == [("B positive", "B zero", "B negative")]
    assert retriever.corpus == [
        {"doc_id": "b-0", "retrieval_content": "B positive"},
        {"doc_id": "b-1", "retrieval_content": "B zero"},
        {"doc_id": "b-2", "retrieval_content": "B negative"},
    ]
    assert retriever.cache_identity == build_contriever_cache_identity(
        corpus_manifest=manifest_b
    )
    assert retriever.embedding_cache_path != state_a[2]
    assert retriever.faiss_cache_path != state_a[3]
    assert retriever.cache_metadata_path != state_a[4]
    assert retriever.faiss_index is not state_a[5]
    assert retriever.faiss_index.ntotal == 3
    assert retriever.embedding_artifact_sha256 != state_a[6]
    assert retriever.index_artifact_sha256 != state_a[7]
    assert retriever.is_indexed is True

    query = np.zeros((1, 768), dtype=np.float32)
    query[0, 0] = 1.0
    monkeypatch.setattr(retriever, "_encode_queries", lambda queries: query)
    assert retriever.retrieve("query", top_k=1)[0][0]["doc_id"] == "b-0"


def test_same_instance_encoding_failure_preserves_complete_a_state(monkeypatch):
    retriever, _, _, _ = _build_cache(monkeypatch)
    state_a = _state_snapshot(retriever)
    records_b = _records_b()
    monkeypatch.setattr(
        retriever,
        "_encode_documents",
        lambda documents: (_ for _ in ()).throw(RuntimeError("encoding failed")),
    )
    with pytest.raises(RuntimeError, match="encoding failed"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(records_b),
            corpus_records=records_b,
        )
    assert _state_snapshot(retriever) == state_a
    _assert_a_retrieval_still_works(monkeypatch, retriever)


def test_same_instance_artifact_write_failure_preserves_complete_a_state(
    monkeypatch,
):
    retriever, _, _, _ = _build_cache(monkeypatch)
    state_a = _state_snapshot(retriever)
    records_b = _records_b()
    _install_encoder(monkeypatch, retriever, _synthetic_embeddings(records_b), [])
    monkeypatch.setattr(
        retriever,
        "_write_cache_pair",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("artifact failed")),
    )
    with pytest.raises(RuntimeError, match="artifact failed"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(records_b),
            corpus_records=records_b,
        )
    assert _state_snapshot(retriever) == state_a
    _assert_a_retrieval_still_works(monkeypatch, retriever)


def test_same_instance_metadata_failure_preserves_a_then_retry_commits_b(
    monkeypatch,
):
    retriever, _, _, _ = _build_cache(monkeypatch)
    state_a = _state_snapshot(retriever)
    records_b = _records_b()
    manifest_b = _manifest(records_b)
    embeddings_b = _synthetic_embeddings(records_b)
    _install_encoder(monkeypatch, retriever, embeddings_b, [])
    original_writer = retriever._write_json_atomically
    monkeypatch.setattr(
        retriever,
        "_write_json_atomically",
        lambda path, metadata: (_ for _ in ()).throw(RuntimeError("metadata failed")),
    )
    with pytest.raises(RuntimeError, match="metadata failed"):
        retriever.index_from_corpus_records(
            corpus_manifest=manifest_b,
            corpus_records=records_b,
        )
    assert _state_snapshot(retriever) == state_a
    _assert_a_retrieval_still_works(monkeypatch, retriever)

    monkeypatch.setattr(retriever, "_write_json_atomically", original_writer)
    _install_encoder(monkeypatch, retriever, embeddings_b, [])
    retriever.index_from_corpus_records(
        corpus_manifest=manifest_b,
        corpus_records=records_b,
    )
    assert retriever.corpus[0]["doc_id"] == "b-0"
    assert retriever.cache_identity == build_contriever_cache_identity(
        corpus_manifest=manifest_b
    )
    assert retriever.faiss_index.ntotal == 3
    assert retriever.is_indexed is True


def test_first_index_failure_preserves_initial_unindexed_state(monkeypatch):
    retriever = ContrieverRetriever()
    initial = _state_snapshot(retriever)
    monkeypatch.setattr(
        retriever,
        "_encode_documents",
        lambda documents: (_ for _ in ()).throw(RuntimeError("encoding failed")),
    )
    with pytest.raises(RuntimeError, match="encoding failed"):
        retriever.index_from_corpus_records(
            corpus_manifest=_manifest(),
            corpus_records=_records(),
        )
    assert _state_snapshot(retriever) == initial
    assert retriever.corpus is None
    assert retriever.faiss_index is None
    assert retriever.cache_identity is None
    assert retriever.embedding_artifact_sha256 is None
    assert retriever.index_artifact_sha256 is None
    assert retriever.is_indexed is False
