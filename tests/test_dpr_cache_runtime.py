from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    document_content_sha256,
)
import retrievers.dpr_original_retriever as dpr_module
from retrievers.dpr_config import DPR_CONFIG
from retrievers.dpr_original_retriever import (
    DPR_CACHE_METADATA_SCHEMA_VERSION,
    OriginalDPRRetriever,
)


class FakeIndexFlatIP:
    def __init__(self, dimension):
        self.d = dimension
        self.ntotal = 0
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, embeddings):
        self.vectors = np.array(embeddings, copy=True)
        self.ntotal = len(embeddings)

    def search(self, queries, top_k):
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")
        result_scores = np.full((len(queries), top_k), -np.inf, dtype=np.float32)
        result_indices = np.full((len(queries), top_k), -1, dtype=np.int64)
        available = min(top_k, self.ntotal)
        for row in range(len(queries)):
            selected = order[row, :available]
            result_indices[row, :available] = selected
            result_scores[row, :available] = scores[row, selected]
        return result_scores, result_indices


def _write_fake_index(index, path):
    with open(path, "wb") as file:
        np.save(file, index.vectors, allow_pickle=False)


def _read_fake_index(path):
    with open(path, "rb") as file:
        vectors = np.load(file, allow_pickle=False)
    index = FakeIndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


@pytest.fixture(autouse=True)
def isolated_cache_and_fake_faiss(tmp_path, monkeypatch):
    embeddings_dir = tmp_path / "embeddings"
    indices_dir = tmp_path / "indices"
    monkeypatch.setattr(dpr_module, "EMBEDDINGS_DIR", str(embeddings_dir))
    monkeypatch.setattr(dpr_module, "INDEX_DIR", str(indices_dir))
    monkeypatch.setattr(dpr_module.faiss, "IndexFlatIP", FakeIndexFlatIP)
    monkeypatch.setattr(dpr_module.faiss, "write_index", _write_fake_index)
    monkeypatch.setattr(dpr_module.faiss, "read_index", _read_fake_index)
    return embeddings_dir, indices_dir


def corpus_records():
    return (
        CorpusRecord(0, "source-0", None, "stored zero", "positive", 0),
        CorpusRecord("one", "source-1", None, "stored one", "zero", 1),
        CorpusRecord(2, "source-2", None, "stored two", "negative", 2),
    )


def corpus_manifest(records=None):
    values = corpus_records() if records is None else records
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
        entries=tuple(
            CorpusManifestEntry(
                position=record.corpus_position,
                doc_id=record.document_id,
                source_document_id=record.source_document_id,
                title_sha256=None,
                text_sha256=document_content_sha256(record.text),
                retrieval_content_sha256=document_content_sha256(
                    record.retrieval_content
                ),
            )
            for record in values
        ),
    )


def synthetic_embeddings(dtype=np.float32):
    values = np.zeros((3, DPR_CONFIG.embedding_dimension), dtype=dtype)
    values[0, 0] = 1.0
    values[2, 0] = -1.0
    return values


def install_fake_encoder(monkeypatch, retriever, embeddings, calls):
    def encode(documents):
        calls.append(tuple(document["retrieval_content"] for document in documents))
        return np.array(embeddings, copy=True)

    monkeypatch.setattr(retriever, "_encode_contexts", encode)


def build_cache(monkeypatch, embeddings=None):
    retriever = OriginalDPRRetriever()
    calls = []
    install_fake_encoder(
        monkeypatch,
        retriever,
        synthetic_embeddings() if embeddings is None else embeddings,
        calls,
    )
    records = corpus_records()
    manifest = corpus_manifest(records)
    retriever.index_from_corpus_records(
        corpus_manifest=manifest, corpus_records=records
    )
    return retriever, calls, manifest, records


def load_metadata(retriever):
    with open(retriever.cache_metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_metadata(retriever, metadata):
    with open(retriever.cache_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, sort_keys=True, separators=(",", ":"))


def test_fingerprinted_paths_and_complete_metadata(monkeypatch):
    retriever, calls, manifest, _ = build_cache(monkeypatch)
    fingerprint = retriever.cache_identity.fingerprint_sha256
    assert calls == [("positive", "zero", "negative")]
    assert Path(retriever.embedding_cache_path).name == (
        f"dpr_embeddings_{fingerprint}.npy"
    )
    assert Path(retriever.faiss_cache_path).name == f"dpr_index_{fingerprint}.faiss"
    assert Path(retriever.cache_metadata_path).name == f"dpr_cache_{fingerprint}.json"
    metadata = load_metadata(retriever)
    assert metadata["cache_schema_version"] == DPR_CACHE_METADATA_SCHEMA_VERSION
    assert metadata["cache_fingerprint_sha256"] == fingerprint
    assert metadata["corpus_manifest_id"] == manifest.corpus_manifest_id
    assert metadata["corpus_manifest_sha256"] == manifest.sha256
    assert metadata["dpr_scientific_json"] == DPR_CONFIG.scientific_json()
    assert set(metadata["environment"]) == {
        "device",
        "faiss_version",
        "numpy_version",
        "python_version",
        "torch_version",
        "transformers_version",
    }


def test_legacy_global_cache_files_are_ignored_and_preserved(
    isolated_cache_and_fake_faiss, monkeypatch
):
    embeddings_dir, indices_dir = isolated_cache_and_fake_faiss
    embeddings_dir.mkdir(parents=True)
    indices_dir.mkdir(parents=True)
    legacy_embeddings = embeddings_dir / "dpr_embeddings.npy"
    legacy_index = indices_dir / "dpr_faiss.index"
    legacy_embeddings.write_bytes(b"untrusted embeddings")
    legacy_index.write_bytes(b"untrusted index")
    retriever, _, _, _ = build_cache(monkeypatch)
    assert legacy_embeddings.read_bytes() == b"untrusted embeddings"
    assert legacy_index.read_bytes() == b"untrusted index"
    assert Path(retriever.embedding_cache_path) != legacy_embeddings
    assert Path(retriever.faiss_cache_path) != legacy_index


def test_records_are_validated_before_cache_lookup(monkeypatch):
    retriever = OriginalDPRRetriever()
    monkeypatch.setattr(
        retriever,
        "_load_validated_cache",
        lambda **kwargs: pytest.fail("cache must not be considered before validation"),
    )
    records = corpus_records()
    changed = (replace(records[0], retrieval_content="different"), *records[1:])
    with pytest.raises(ValueError, match="retrieval_content"):
        retriever.index_from_corpus_records(
            corpus_manifest=corpus_manifest(records), corpus_records=changed
        )


def test_valid_cache_hit_bypasses_encoding(monkeypatch):
    original, _, manifest, records = build_cache(monkeypatch)
    fresh = OriginalDPRRetriever()
    monkeypatch.setattr(
        fresh,
        "_encode_contexts",
        lambda documents: pytest.fail("valid cache must bypass model encoding"),
    )
    fresh.index_from_corpus_records(
        corpus_manifest=manifest, corpus_records=records
    )
    assert fresh.embedding_artifact_sha256 == original.embedding_artifact_sha256
    assert fresh.index_artifact_sha256 == original.index_artifact_sha256


@pytest.mark.parametrize(
    "damage",
    [
        "missing_metadata",
        "missing_embeddings",
        "missing_index",
        "corrupt_embeddings",
        "corrupt_index",
        "fingerprint",
        "metadata_schema",
        "embedding_filename",
        "malformed_metadata",
        "embedding_shape",
        "embedding_dtype",
        "faiss_dimension",
        "faiss_ntotal",
    ],
)
def test_invalid_or_incomplete_cache_rebuilds_complete_pair(monkeypatch, damage):
    original, _, manifest, records = build_cache(monkeypatch)
    metadata = load_metadata(original)

    if damage == "missing_metadata":
        Path(original.cache_metadata_path).unlink()
    elif damage == "missing_embeddings":
        Path(original.embedding_cache_path).unlink()
    elif damage == "missing_index":
        Path(original.faiss_cache_path).unlink()
    elif damage == "corrupt_embeddings":
        Path(original.embedding_cache_path).write_bytes(b"corrupt npy")
    elif damage == "corrupt_index":
        Path(original.faiss_cache_path).write_bytes(b"corrupt faiss")
    elif damage == "fingerprint":
        metadata["cache_fingerprint_sha256"] = "0" * 64
        write_metadata(original, metadata)
    elif damage == "metadata_schema":
        metadata["cache_schema_version"] = "sprint3.dpr-cache-metadata.v2"
        write_metadata(original, metadata)
    elif damage == "embedding_filename":
        metadata["embedding"]["filename"] = "wrong.npy"
        write_metadata(original, metadata)
    elif damage == "malformed_metadata":
        Path(original.cache_metadata_path).write_text("not json", encoding="utf-8")
    elif damage in {"embedding_shape", "embedding_dtype"}:
        replacement = (
            np.zeros((2, DPR_CONFIG.embedding_dimension), dtype=np.float32)
            if damage == "embedding_shape"
            else synthetic_embeddings(dtype=np.float64)
        )
        with open(original.embedding_cache_path, "wb") as file:
            np.save(file, replacement, allow_pickle=False)
        metadata["embedding"]["sha256"] = dpr_module._physical_sha256(
            original.embedding_cache_path
        )
        write_metadata(original, metadata)
    elif damage in {"faiss_dimension", "faiss_ntotal"}:
        replacement = FakeIndexFlatIP(
            DPR_CONFIG.embedding_dimension + (1 if damage == "faiss_dimension" else 0)
        )
        replacement.add(
            np.zeros(
                (
                    2 if damage == "faiss_ntotal" else 3,
                    replacement.d,
                ),
                dtype=np.float32,
            )
        )
        _write_fake_index(replacement, original.faiss_cache_path)
        metadata["faiss"]["sha256"] = dpr_module._physical_sha256(
            original.faiss_cache_path
        )
        write_metadata(original, metadata)

    fresh = OriginalDPRRetriever()
    calls = []
    install_fake_encoder(monkeypatch, fresh, synthetic_embeddings(), calls)
    fresh.index_from_corpus_records(
        corpus_manifest=manifest, corpus_records=records
    )
    assert calls == [("positive", "zero", "negative")]
    assert Path(fresh.embedding_cache_path).is_file()
    assert Path(fresh.faiss_cache_path).is_file()
    assert Path(fresh.cache_metadata_path).is_file()


def test_nonfinite_new_embeddings_fail_without_metadata(monkeypatch):
    embeddings = synthetic_embeddings()
    embeddings[0, 0] = np.nan
    retriever = OriginalDPRRetriever()
    calls = []
    install_fake_encoder(monkeypatch, retriever, embeddings, calls)
    with pytest.raises(ValueError, match="finite"):
        retriever.index_from_corpus_records(
            corpus_manifest=corpus_manifest(), corpus_records=corpus_records()
        )
    assert calls
    assert not Path(retriever.cache_metadata_path).exists()


def test_metadata_is_written_last_and_incomplete_pair_is_not_trusted(monkeypatch):
    retriever = OriginalDPRRetriever()
    calls = []
    install_fake_encoder(monkeypatch, retriever, synthetic_embeddings(), calls)
    monkeypatch.setattr(
        retriever,
        "_write_json_atomically",
        lambda path, metadata: (_ for _ in ()).throw(RuntimeError("metadata failure")),
    )
    with pytest.raises(RuntimeError, match="metadata failure"):
        retriever.index_from_corpus_records(
            corpus_manifest=corpus_manifest(), corpus_records=corpus_records()
        )
    assert Path(retriever.embedding_cache_path).is_file()
    assert Path(retriever.faiss_cache_path).is_file()
    assert not Path(retriever.cache_metadata_path).exists()

    fresh = OriginalDPRRetriever()
    fresh_calls = []
    install_fake_encoder(monkeypatch, fresh, synthetic_embeddings(), fresh_calls)
    fresh.index_from_corpus_records(
        corpus_manifest=corpus_manifest(), corpus_records=corpus_records()
    )
    assert fresh_calls


def test_physical_checksums_match_finalized_bytes(monkeypatch):
    retriever, _, _, _ = build_cache(monkeypatch)
    metadata = load_metadata(retriever)
    assert metadata["embedding"]["sha256"] == dpr_module._physical_sha256(
        retriever.embedding_cache_path
    )
    assert metadata["faiss"]["sha256"] == dpr_module._physical_sha256(
        retriever.faiss_cache_path
    )
    assert retriever.embedding_artifact_sha256 == metadata["embedding"]["sha256"]
    assert retriever.index_artifact_sha256 == metadata["faiss"]["sha256"]


def test_cached_and_fresh_indexes_preserve_scores_order_and_nonpositive_values(
    monkeypatch,
):
    fresh, _, manifest, records = build_cache(monkeypatch)
    query = np.zeros((1, DPR_CONFIG.embedding_dimension), dtype=np.float32)
    query[0, 0] = 1.0
    monkeypatch.setattr(fresh, "_encode_questions", lambda questions: query)
    expected = fresh.retrieve("query", top_k=3)

    cached = OriginalDPRRetriever()
    monkeypatch.setattr(
        cached,
        "_encode_contexts",
        lambda documents: pytest.fail("cache should be reused"),
    )
    cached.index_from_corpus_records(
        corpus_manifest=manifest, corpus_records=records
    )
    monkeypatch.setattr(cached, "_encode_questions", lambda questions: query)
    actual = cached.retrieve("query", top_k=3)

    assert [document["doc_id"] for document, _ in actual] == [0, "one", 2]
    assert [score for _, score in actual] == [1.0, 0.0, -1.0]
    assert actual == expected
