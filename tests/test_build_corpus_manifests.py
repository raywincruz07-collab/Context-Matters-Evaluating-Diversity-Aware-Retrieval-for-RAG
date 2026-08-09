import csv
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    dataset_provenance_from_sample_manifest,
    query_text_sha256,
)
from scripts.build_corpus_manifests import (
    CORPUS_MANIFEST_ARTIFACT_FORMAT,
    PUBMEDQA_CONFIG,
    PUBMEDQA_CONSTRUCTION_ALGORITHM,
    PUBMEDQA_OUTPUT_PATH,
    PUBMEDQA_REVISION,
    PUBMEDQA_SECTION_SOURCE_ID_PROTOCOL,
    PUBMEDQA_SOURCE,
    PUBMEDQA_SPLIT,
    build_pubmedqa_corpus_manifest,
    corpus_manifest_artifact_payload,
    pubmedqa_section_source_document_id,
    read_and_verify_corpus_manifest_artifact,
    verify_pubmedqa_historical_gold_assignments,
    write_corpus_manifest_artifact,
)
from retrieval_artifacts import corpus_provenance_from_corpus_manifest


def fixture_rows():
    return (
        {
            "pubid": 101,
            "question": "Exact question α?",
            "context": {
                "contexts": ["  First section.  ", "Second section.\n"],
                "labels": ["BACKGROUND", "METHODS"],
            },
            "long_answer": "must not enter retrieval",
            "final_decision": "yes",
        },
        {
            "pubid": 202,
            "question": "Question two?",
            "context": {
                "contexts": [" Third section "],
                "labels": ["RESULTS"],
            },
        },
    )


def fixture_sample_manifest(rows=None):
    values = fixture_rows() if rows is None else rows
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source=PUBMEDQA_SOURCE,
        config=PUBMEDQA_CONFIG,
        revision=PUBMEDQA_REVISION,
        split=PUBMEDQA_SPLIT,
        sampling_algorithm="fixture_full_split_source_order.v1",
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=tuple(
            SampleManifestEntry(
                position=i,
                sample_id=i,
                source_sample_id=row["pubid"],
                query_text_sha256=query_text_sha256(row["question"]),
            )
            for i, row in enumerate(values)
        ),
    )


def fixture_build(rows=None, manifest=None):
    return build_pubmedqa_corpus_manifest(
        fixture_rows() if rows is None else rows,
        fixture_sample_manifest() if manifest is None else manifest,
        expected_sample_count=2,
        expected_document_count=3,
    )


def write_historical_fixture(path: Path, gold_groups=((0, 1), (2,))):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("qa_id", "gold_doc_ids"))
        writer.writeheader()
        for position, ids in enumerate(gold_groups):
            writer.writerow({"qa_id": position, "gold_doc_ids": list(ids)})


def test_deterministic_exact_source_order_and_record_semantics():
    first = fixture_build()
    second = fixture_build()
    assert first == second
    assert first.corpus_manifest.corpus_manifest_id == second.corpus_manifest.corpus_manifest_id
    assert tuple(record.document_id for record in first.corpus_records) == (0, 1, 2)
    assert tuple(record.corpus_position for record in first.corpus_records) == (0, 1, 2)
    assert tuple(record.source_document_id for record in first.corpus_records) == (
        "pubmedqa:pubid:101:context:0",
        "pubmedqa:pubid:101:context:1",
        "pubmedqa:pubid:202:context:0",
    )
    assert len({record.source_document_id for record in first.corpus_records}) == 3
    assert all(record.title is None for record in first.corpus_records)
    assert tuple(record.text for record in first.corpus_records) == (
        "First section.", "Second section.", "Third section"
    )
    assert all(record.text == record.retrieval_content for record in first.corpus_records)
    assert "BACKGROUND" not in "".join(record.text for record in first.corpus_records)
    with pytest.raises(FrozenInstanceError):
        first.corpus_records[0].text = "changed"


def test_source_id_protocol_is_explicit_and_rejects_ambiguous_pubids():
    assert PUBMEDQA_SECTION_SOURCE_ID_PROTOCOL == "pubmedqa_pubid_context_ordinal.v1"
    assert pubmedqa_section_source_document_id("123", 4) == "pubmedqa:pubid:123:context:4"
    for invalid in ("0123", " 123", "abc", True, -1):
        with pytest.raises((TypeError, ValueError)):
            pubmedqa_section_source_document_id(invalid, 0)


def test_exact_utf8_hashes_and_manifest_binding_and_no_sampling():
    sample_manifest = fixture_sample_manifest()
    build = fixture_build(manifest=sample_manifest)
    entry = build.corpus_manifest.entries[0]
    expected_hash = hashlib.sha256("First section.".encode("utf-8")).hexdigest()
    assert entry.text_sha256 == expected_hash
    assert entry.retrieval_content_sha256 == expected_hash
    assert build.corpus_manifest.input_sample_manifest_id == sample_manifest.manifest_id
    assert build.corpus_manifest.input_sample_manifest_sha256 == sample_manifest.sha256
    assert build.corpus_manifest.construction_algorithm == PUBMEDQA_CONSTRUCTION_ALGORITHM
    assert build.corpus_manifest.dependencies == ()
    assert build.corpus_manifest.rng_family is None
    assert build.corpus_manifest.sampling_seed is None
    assert build.corpus_manifest.requested_negatives_per_query is None
    provenance = corpus_provenance_from_corpus_manifest(
        corpus_manifest=build.corpus_manifest,
        corpus_records=build.corpus_records,
        dataset_provenance=dataset_provenance_from_sample_manifest(sample_manifest),
    )
    assert provenance.corpus_id == build.corpus_manifest.corpus_manifest_id
    assert provenance.manifest_sha256 == build.corpus_manifest.sha256


def test_context_label_length_mismatch_and_empty_context_rejected():
    rows = list(fixture_rows())
    rows[0] = {**rows[0], "context": {"contexts": ["a", "b"], "labels": ["x"]}}
    with pytest.raises(ValueError, match="length mismatch"):
        fixture_build(tuple(rows))
    rows = list(fixture_rows())
    rows[0] = {**rows[0], "context": {"contexts": ["   ", "b"], "labels": ["x", "y"]}}
    with pytest.raises(ValueError, match="non-empty after strip"):
        fixture_build(tuple(rows))


def test_wrong_count_reorder_pubid_and_question_are_rejected():
    manifest = fixture_sample_manifest()
    with pytest.raises(ValueError, match="expected 2"):
        build_pubmedqa_corpus_manifest(
            fixture_rows()[:1], manifest,
            expected_sample_count=2, expected_document_count=2,
        )
    with pytest.raises(ValueError, match="pubid mismatch"):
        fixture_build(tuple(reversed(fixture_rows())), manifest)
    changed = list(fixture_rows())
    changed[0] = {**changed[0], "pubid": 999}
    with pytest.raises(ValueError, match="pubid mismatch"):
        fixture_build(tuple(changed), manifest)
    changed = list(fixture_rows())
    changed[0] = {**changed[0], "question": "changed"}
    with pytest.raises(ValueError, match="question mismatch"):
        fixture_build(tuple(changed), manifest)


def test_expected_document_count_mismatch_is_rejected():
    with pytest.raises(
        ValueError, match="expected 4 PubMedQA documents, found 3"
    ):
        build_pubmedqa_corpus_manifest(
            fixture_rows(),
            fixture_sample_manifest(),
            expected_sample_count=2,
            expected_document_count=4,
        )


def test_integral_family_pubid_matches_but_string_remains_distinct():
    rows = list(fixture_rows())
    rows[0] = {**rows[0], "pubid": np.int64(101)}
    build = fixture_build(tuple(rows), fixture_sample_manifest())
    assert build.corpus_records[0].source_document_id == (
        "pubmedqa:pubid:101:context:0"
    )

    rows[0] = {**rows[0], "pubid": "101"}
    with pytest.raises(ValueError, match="pubid mismatch"):
        fixture_build(tuple(rows), fixture_sample_manifest())


def test_duplicate_pubids_rejected():
    rows = list(fixture_rows())
    rows[1] = {**rows[1], "pubid": 101}
    # SampleManifest itself protects this source identity before corpus building.
    with pytest.raises(ValueError, match="source_sample_id"):
        fixture_sample_manifest(tuple(rows))


def test_historical_gold_assignment_gate(tmp_path):
    historical = tmp_path / "historical.csv"
    write_historical_fixture(historical)
    result = verify_pubmedqa_historical_gold_assignments(
        fixture_build(), historical, expected_sample_count=2
    )
    assert result.sample_count == 2
    assert result.document_count == 3
    assert result.gold_assignment_count == 3
    write_historical_fixture(historical, ((0,), (1, 2)))
    with pytest.raises(ValueError, match="grouping mismatch"):
        verify_pubmedqa_historical_gold_assignments(
            fixture_build(), historical, expected_sample_count=2
        )


def test_artifact_round_trip_and_tampering_detection(tmp_path):
    manifest = fixture_build().corpus_manifest
    path = tmp_path / "corpus.json"
    write_corpus_manifest_artifact(manifest, path)
    assert read_and_verify_corpus_manifest_artifact(path) == manifest
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["artifact_format"] == CORPUS_MANIFEST_ARTIFACT_FORMAT
    stored["scientific_payload"]["entries"][0]["text_sha256"] = "0" * 64
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match payload"):
        read_and_verify_corpus_manifest_artifact(path)


@pytest.mark.parametrize("field", ["corpus_manifest_id", "corpus_manifest_sha256"])
def test_artifact_wrong_declared_identity_rejected(tmp_path, field):
    manifest = fixture_build().corpus_manifest
    stored = corpus_manifest_artifact_payload(manifest)
    stored[field] = ("corpus-manifest:sha256:" if field.endswith("id") else "") + "0" * 64
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match payload"):
        read_and_verify_corpus_manifest_artifact(path)


def test_core_builder_does_not_write_reserved_real_artifact():
    existed_before = PUBMEDQA_OUTPUT_PATH.exists()
    fixture_build()
    assert PUBMEDQA_OUTPUT_PATH.exists() is existed_before
