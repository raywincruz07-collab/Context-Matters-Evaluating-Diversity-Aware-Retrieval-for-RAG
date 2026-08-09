from dataclasses import replace
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
import retrieval_artifacts
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    CorpusManifestEntry,
    CorpusRecord,
    DatasetProvenance,
    compute_corpus_records_integrity_sha256,
    compute_document_id_map_sha256,
    compute_index_fingerprint_sha256,
    corpus_provenance_from_corpus_manifest,
    document_content_sha256,
    validate_corpus_dataset_binding,
    validate_corpus_records_against_manifest,
)


SAMPLE_SHA = "a" * 64
SAMPLE_ID = f"sample-manifest:sha256:{SAMPLE_SHA}"


def records():
    return (
        CorpusRecord(0, "source-0", None, "text zero", "prepared zero", 0),
        CorpusRecord(
            "1", "source-1", "Title one", "text one", "prepared one", 1
        ),
    )


def dataset(**changes):
    values = dict(
        dataset_id=DatasetId.HOTPOTQA,
        source="synthetic-query-source",
        config="fixture-queries",
        revision="fixture-query-revision",
        split="fixture-query-split",
        sample_manifest_id=SAMPLE_ID,
        sample_manifest_sha256=SAMPLE_SHA,
        sampling_seed=42,
        sampling_algorithm="fixture-query-sampling.v1",
    )
    values.update(changes)
    return DatasetProvenance(**values)


def entry(record):
    return CorpusManifestEntry(
        position=int(record.corpus_position),
        doc_id=record.document_id,
        source_document_id=record.source_document_id,
        title_sha256=(
            None
            if record.title is None
            else document_content_sha256(record.title)
        ),
        text_sha256=document_content_sha256(record.text),
        retrieval_content_sha256=document_content_sha256(record.retrieval_content),
    )


def manifest(corpus_records=None, *, bound=True, **changes):
    corpus_records = records() if corpus_records is None else corpus_records
    values = dict(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.HOTPOTQA,
        source="synthetic-corpus-source",
        config="fixture-corpus",
        revision="fixture-corpus-revision",
        split="fixture-corpus-split",
        construction_algorithm="fixture-corpus-construction.v1",
        input_sample_manifest_id=SAMPLE_ID if bound else None,
        input_sample_manifest_sha256=SAMPLE_SHA if bound else None,
        dependencies=(),
        rng_family=None,
        sampling_seed=None,
        rng_state_semantics=None,
        requested_negatives_per_query=None,
        negative_sampling_scope=None,
        negative_exclusion_scope=None,
        negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=tuple(entry(record) for record in corpus_records),
    )
    values.update(changes)
    return CorpusManifest(**values)


def bridge(corpus_records=None, corpus_manifest=None, dataset_provenance=None):
    corpus_records = records() if corpus_records is None else corpus_records
    corpus_manifest = manifest(corpus_records) if corpus_manifest is None else corpus_manifest
    dataset_provenance = dataset() if dataset_provenance is None else dataset_provenance
    return corpus_provenance_from_corpus_manifest(
        corpus_manifest=corpus_manifest,
        corpus_records=corpus_records,
        dataset_provenance=dataset_provenance,
    )


def test_bridge_derives_scientific_identity_and_document_map():
    corpus_records = records()
    corpus_manifest = manifest(corpus_records)
    value = bridge(corpus_records, corpus_manifest)
    assert value.corpus_id == corpus_manifest.corpus_manifest_id
    assert value.manifest_sha256 == corpus_manifest.sha256
    assert value.source == corpus_manifest.source
    assert value.revision == corpus_manifest.revision
    assert value.document_count == corpus_manifest.document_count
    assert value.preprocessing_version == corpus_manifest.construction_algorithm
    assert value.document_id_map_sha256 == compute_document_id_map_sha256(
        corpus_records
    )


def test_concrete_integrity_is_deterministic_and_not_manifest_identity():
    corpus_records = records()
    corpus_manifest = manifest(corpus_records)
    first = compute_corpus_records_integrity_sha256(corpus_records)
    assert first == compute_corpus_records_integrity_sha256(corpus_records)
    assert first != corpus_manifest.sha256
    assert not hasattr(retrieval_artifacts, "compute_corpus_manifest_sha256")


def test_count_and_malformed_record_positions_are_rejected():
    with pytest.raises(ValueError, match="count"):
        validate_corpus_records_against_manifest(manifest(), records()[:1])
    bad_position = (
        replace(records()[0], corpus_position=1),
        records()[1],
    )
    with pytest.raises(ValueError, match="corpus_position"):
        validate_corpus_records_against_manifest(manifest(), bad_position)


def test_record_tuple_order_is_not_repaired():
    reversed_records = (
        replace(records()[1], corpus_position=1),
        replace(records()[0], corpus_position=0),
    )
    with pytest.raises(ValueError, match="tuple order"):
        validate_corpus_records_against_manifest(manifest(), reversed_records)


@pytest.mark.parametrize(
    "changed_records, message",
    [
        (lambda value: (replace(value[0], document_id=7), value[1]), "document_id"),
        (lambda value: (replace(value[0], document_id="0"), value[1]), "document_id"),
        (
            lambda value: (
                replace(value[0], source_document_id="different"),
                value[1],
            ),
            "source_document_id",
        ),
        (lambda value: (replace(value[0], title="unexpected"), value[1]), "title presence"),
        (lambda value: (replace(value[0], text="different"), value[1]), "text"),
        (
            lambda value: (
                replace(value[0], retrieval_content="different"),
                value[1],
            ),
            "retrieval_content",
        ),
    ],
)
def test_record_content_mismatches_are_rejected(changed_records, message):
    with pytest.raises(ValueError, match=message):
        validate_corpus_records_against_manifest(manifest(), changed_records(records()))


def test_present_title_hash_mismatch_is_rejected():
    changed = (records()[0], replace(records()[1], title="Different title"))
    with pytest.raises(ValueError, match="title does not match"):
        validate_corpus_records_against_manifest(manifest(), changed)


def test_corpus_record_requires_source_id_and_explicit_content_types():
    with pytest.raises(TypeError, match="source_document_id"):
        CorpusRecord(0, None, None, "text", "prepared", 0)
    with pytest.raises(TypeError, match="title"):
        CorpusRecord(0, "source", 7, "text", "prepared", 0)
    with pytest.raises(TypeError, match="text"):
        CorpusRecord(0, "source", None, None, "prepared", 0)
    with pytest.raises(TypeError, match="retrieval_content"):
        CorpusRecord(0, "source", None, "text", None, 0)


def test_dataset_and_sample_manifest_mismatches_are_rejected():
    corpus_manifest = manifest()
    with pytest.raises(ValueError, match="dataset_id"):
        validate_corpus_dataset_binding(
            corpus_manifest=corpus_manifest,
            dataset_provenance=dataset(dataset_id=DatasetId.PUBMEDQA),
        )
    with pytest.raises(ValueError, match="SampleManifest ID"):
        validate_corpus_dataset_binding(
            corpus_manifest=corpus_manifest,
            dataset_provenance=dataset(sample_manifest_id="different"),
        )
    with pytest.raises(ValueError, match="SampleManifest SHA"):
        validate_corpus_dataset_binding(
            corpus_manifest=corpus_manifest,
            dataset_provenance=dataset(sample_manifest_sha256="b" * 64),
        )


def test_bound_and_query_independent_corpora_are_accepted():
    assert bridge().manifest_sha256 == manifest().sha256
    unbound = manifest(bound=False)
    assert bridge(corpus_manifest=unbound).manifest_sha256 == unbound.sha256


def test_construction_only_change_changes_scientific_provenance_and_index():
    original_manifest = manifest()
    changed_manifest = replace(
        original_manifest,
        construction_algorithm="fixture-other-construction.v1",
    )
    original = bridge(corpus_manifest=original_manifest)
    changed = bridge(corpus_manifest=changed_manifest)
    assert changed.corpus_id != original.corpus_id
    assert changed.manifest_sha256 != original.manifest_sha256
    assert compute_corpus_records_integrity_sha256(
        records()
    ) == compute_corpus_records_integrity_sha256(records())
    original_index = compute_index_fingerprint_sha256(
        corpus_manifest_sha256=original.manifest_sha256,
        document_id_map_sha256=original.document_id_map_sha256,
        library_version="fixture-library-version",
    )
    changed_index = compute_index_fingerprint_sha256(
        corpus_manifest_sha256=changed.manifest_sha256,
        document_id_map_sha256=changed.document_id_map_sha256,
        library_version="fixture-library-version",
    )
    assert changed_index != original_index
