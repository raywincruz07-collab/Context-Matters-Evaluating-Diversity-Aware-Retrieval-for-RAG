from dataclasses import replace
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrievers.bm25_retriever import BM25Retriever
import retrievers.bm25_retriever as bm25_module
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    SampleManifest,
    SampleManifestEntry,
    bm25_runtime_documents_from_corpus_records,
    document_content_sha256,
    query_text_sha256,
)
from scripts.build_corpus_manifests import (
    PUBMEDQA_CONFIG,
    PUBMEDQA_REVISION,
    PUBMEDQA_SOURCE,
    PUBMEDQA_SPLIT,
    build_pubmedqa_corpus_manifest,
    prepare_pubmedqa_runtime_corpus,
)


def pubmedqa_rows():
    return (
        {
            "pubid": 101,
            "question": "Exact fixture question?",
            "context": {"contexts": [" first ", "second"], "labels": ["A", "B"]},
        },
        {
            "pubid": 202,
            "question": "Other fixture question?",
            "context": {"contexts": ["third"], "labels": ["C"]},
        },
    )


def sample_manifest():
    rows = pubmedqa_rows()
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source=PUBMEDQA_SOURCE,
        config=PUBMEDQA_CONFIG,
        revision=PUBMEDQA_REVISION,
        split=PUBMEDQA_SPLIT,
        sampling_algorithm="fixture-source-order.v1",
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=tuple(
            SampleManifestEntry(
                position=position,
                sample_id=position,
                source_sample_id=row["pubid"],
                query_text_sha256=query_text_sha256(row["question"]),
            )
            for position, row in enumerate(rows)
        ),
    )


def pubmedqa_build():
    return build_pubmedqa_corpus_manifest(
        pubmedqa_rows(),
        sample_manifest(),
        expected_sample_count=2,
        expected_document_count=3,
    )


def entry(record):
    return CorpusManifestEntry(
        position=record.corpus_position,
        doc_id=record.document_id,
        source_document_id=record.source_document_id,
        title_sha256=(
            None if record.title is None else document_content_sha256(record.title)
        ),
        text_sha256=document_content_sha256(record.text),
        retrieval_content_sha256=document_content_sha256(record.retrieval_content),
    )


def generic_manifest(records):
    return CorpusManifest(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-corpus",
        config=None,
        revision="fixture-revision",
        split="fixture-split",
        construction_algorithm="fixture-prepared-content.v1",
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
        entries=tuple(entry(record) for record in records),
    )


def divergent_records():
    return (
        CorpusRecord(7, "source-7", None, "wrong stored text", "unique retrieval phrase", 0),
        CorpusRecord("b", "source-b", None, "filler alpha", "filler alpha", 1),
        CorpusRecord(9, "source-9", None, "filler beta", "filler beta", 2),
    )


def test_pubmedqa_runtime_preparation_reuses_builder_and_derives_provenance():
    sample = sample_manifest()
    build = pubmedqa_build()
    prepared = prepare_pubmedqa_runtime_corpus(
        pubmedqa_rows(),
        sample_manifest=sample,
        frozen_corpus_manifest=build.corpus_manifest,
        expected_sample_count=2,
        expected_document_count=3,
    )
    assert prepared.corpus_records == build.corpus_records
    assert prepared.corpus_manifest == build.corpus_manifest
    assert prepared.dataset_provenance.sample_manifest_id == sample.manifest_id
    assert prepared.corpus_provenance.manifest_sha256 == build.corpus_manifest.sha256
    assert tuple(query.sample_id for query in prepared.ordered_queries) == (0, 1)
    assert tuple(query.query_text for query in prepared.ordered_queries) == (
        "Exact fixture question?", "Other fixture question?"
    )


def test_pubmedqa_runtime_preparation_rejects_wrong_frozen_manifest():
    build = pubmedqa_build()
    wrong = replace(
        build.corpus_manifest,
        construction_algorithm="different-construction.v1",
    )
    with pytest.raises(ValueError, match="does not match frozen CorpusManifest"):
        prepare_pubmedqa_runtime_corpus(
            pubmedqa_rows(),
            sample_manifest=sample_manifest(),
            frozen_corpus_manifest=wrong,
            expected_sample_count=2,
            expected_document_count=3,
        )


def test_bm25_adapter_preserves_ids_order_and_exact_retrieval_content():
    records = divergent_records()
    documents = bm25_runtime_documents_from_corpus_records(
        corpus_manifest=generic_manifest(records), corpus_records=records
    )
    assert documents == [
        {"doc_id": 7, "text": "unique retrieval phrase"},
        {"doc_id": "b", "text": "filler alpha"},
        {"doc_id": 9, "text": "filler beta"},
    ]
    assert documents[0]["text"] != records[0].text


def test_bm25_indexes_retrieval_content_not_stored_text(tmp_path, monkeypatch):
    records = divergent_records()
    documents = bm25_runtime_documents_from_corpus_records(
        corpus_manifest=generic_manifest(records), corpus_records=records
    )
    monkeypatch.setattr(bm25_module, "INDEX_DIR", str(tmp_path))
    retriever = BM25Retriever()
    retriever.index(documents)
    results = retriever.retrieve("unique retrieval phrase", top_k=3)
    assert results and results[0][0]["doc_id"] == 7
    assert retriever.retrieve("wrong stored text", top_k=3) == []
    assert "unique" in retriever.tokenized_corpus[0]
    assert "wrong" not in retriever.tokenized_corpus[0]


@pytest.mark.parametrize(
    "changed_records, message",
    (
        (lambda values: (replace(values[0], text="changed"), *values[1:]), "text"),
        (
            lambda values: (
                replace(values[0], retrieval_content="changed"), *values[1:]
            ),
            "retrieval_content",
        ),
        (
            lambda values: (replace(values[0], document_id="7"), *values[1:]),
            "document_id",
        ),
        (
            lambda values: (values[1], values[0], values[2]),
            "tuple order",
        ),
    ),
)
def test_bm25_adapter_rejects_record_manifest_drift(changed_records, message):
    records = divergent_records()
    manifest = generic_manifest(records)
    with pytest.raises(ValueError, match=message):
        bm25_runtime_documents_from_corpus_records(
            corpus_manifest=manifest,
            corpus_records=changed_records(records),
        )
