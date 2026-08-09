from dataclasses import FrozenInstanceError, replace
import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    dataset_provenance_from_sample_manifest,
    query_text_sha256,
    verify_manifest_sample,
)


def entry(position=0, sample_id="fixture-q1", query="Fixture question one?", **changes):
    values = dict(
        position=position,
        sample_id=sample_id,
        source_sample_id=f"fixture-source-{position + 1}",
        query_text_sha256=query_text_sha256(query),
    )
    values.update(changes)
    return SampleManifestEntry(**values)


def manifest(**changes):
    values = dict(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-fixture-dataset",
        config="synthetic-fixture-config",
        revision="fixture-revision-abc",
        split="fixture-split",
        sampling_algorithm="fixture-full-population-order-v1",
        sampling_seed=None,
        requested_sample_size=None,
        entries=(
            entry(),
            entry(1, "fixture-q2", "Fixture question two?"),
        ),
    )
    values.update(changes)
    return SampleManifest(**values)


def test_valid_full_style_manifest():
    value = manifest()
    assert value.actual_sample_size == 2
    assert value.sampling_seed is None
    assert value.requested_sample_size is None


def test_valid_sampled_manifest_with_seed():
    value = manifest(
        sampling_algorithm="fixture-seeded-sample-v1",
        sampling_seed=42,
        requested_sample_size=2,
    )
    assert value.sampling_seed == 42
    assert value.requested_sample_size == value.actual_sample_size == 2


def test_manifest_and_entries_are_frozen_and_entries_require_tuple():
    value = manifest()
    with pytest.raises(FrozenInstanceError):
        value.split = "changed"
    with pytest.raises(FrozenInstanceError):
        value.entries[0].sample_id = "changed"
    with pytest.raises(TypeError, match="immutable tuple"):
        manifest(entries=list(value.entries))


@pytest.mark.parametrize(
    "entries",
    [
        (entry(position=1),),
        (entry(), entry(position=2, sample_id="fixture-q2")),
        (entry(1, "fixture-q2"), entry(0, "fixture-q1")),
    ],
)
def test_invalid_entry_positions_are_rejected(entries):
    with pytest.raises(ValueError, match="positions"):
        manifest(entries=entries)


def test_duplicate_sample_and_source_ids_are_rejected():
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        manifest(entries=(entry(), entry(1, "fixture-q1")))
    with pytest.raises(ValueError, match="source_sample_id values must be unique"):
        manifest(
            entries=(
                entry(source_sample_id="same-source"),
                entry(1, "fixture-q2", source_sample_id="same-source"),
            )
        )
    assert manifest(
        entries=(
            entry(source_sample_id=None),
            entry(1, "fixture-q2", source_sample_id=None),
        )
    )


@pytest.mark.parametrize(
    "invalid_id",
    [True, False, np.bool_(True), 1.5, object(), "", "   "],
)
def test_entry_rejects_invalid_sample_ids(invalid_id):
    with pytest.raises((TypeError, ValueError)):
        entry(sample_id=invalid_id)
    with pytest.raises((TypeError, ValueError)):
        entry(source_sample_id=invalid_id)


def test_query_checksum_uses_exact_utf8_text():
    expected = hashlib.sha256("question".encode("utf-8")).hexdigest()
    assert query_text_sha256("question") == expected
    assert query_text_sha256("question ") != expected
    assert query_text_sha256(" question") != expected


def test_manifest_identity_is_deterministic_and_well_formed():
    assert manifest().manifest_id == manifest().manifest_id
    assert manifest().scientific_json() == manifest().scientific_json()
    assert re.fullmatch(r"sample-manifest:sha256:[0-9a-f]{64}", manifest().manifest_id)


def test_entry_order_changes_manifest_identity():
    original = manifest()
    reordered = replace(
        original,
        entries=(
            replace(original.entries[1], position=0),
            replace(original.entries[0], position=1),
        ),
    )
    assert reordered.manifest_id != original.manifest_id


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(
            value,
            entries=(
                replace(value.entries[0], query_text_sha256=query_text_sha256("Changed?")),
                value.entries[1],
            ),
        ),
        lambda value: replace(value, revision="fixture-revision-def"),
        lambda value: replace(value, source="other-synthetic-source"),
        lambda value: replace(value, config="other-synthetic-config"),
        lambda value: replace(value, split="other-fixture-split"),
        lambda value: replace(value, sampling_algorithm="other-fixture-algorithm-v1"),
        lambda value: replace(value, sampling_seed=7),
        lambda value: replace(value, requested_sample_size=3),
    ],
)
def test_scientific_fields_change_manifest_identity(changed):
    original = manifest()
    assert changed(original).manifest_id != original.manifest_id


def test_integer_and_string_sample_ids_have_distinct_identity():
    integer = manifest(entries=(entry(sample_id=1),))
    string = manifest(entries=(entry(sample_id="1"),))
    assert integer.manifest_id != string.manifest_id
    assert integer.scientific_payload()["entries"][0]["sample_id"] == 1
    assert string.scientific_payload()["entries"][0]["sample_id"] == "1"


def test_manifest_field_validation():
    with pytest.raises(ValueError):
        manifest(schema_version="fixture-wrong-schema")
    with pytest.raises(TypeError):
        manifest(dataset_id="pubmedqa")
    for field_name in ("source", "revision", "split", "sampling_algorithm"):
        with pytest.raises(ValueError):
            manifest(**{field_name: " "})
    with pytest.raises(ValueError):
        manifest(config="")
    with pytest.raises(TypeError):
        manifest(sampling_seed=True)
    with pytest.raises(ValueError):
        manifest(sampling_seed=-1)
    with pytest.raises(TypeError):
        manifest(requested_sample_size=2.0)
    with pytest.raises(ValueError):
        manifest(requested_sample_size=0)
    with pytest.raises(ValueError):
        manifest(entries=())


def test_dataset_provenance_builder_maps_every_field_exactly():
    value = manifest(
        sampling_algorithm="fixture-seeded-sample-v1",
        sampling_seed=42,
        requested_sample_size=2,
    )
    provenance = dataset_provenance_from_sample_manifest(value)
    assert provenance.dataset_id is value.dataset_id
    assert provenance.source == value.source
    assert provenance.config == value.config
    assert provenance.revision == value.revision
    assert provenance.split == value.split
    assert provenance.sample_manifest_id == value.manifest_id
    assert provenance.sample_manifest_sha256 == value.sha256
    assert provenance.sampling_seed == value.sampling_seed
    assert provenance.sampling_algorithm == value.sampling_algorithm


def test_verify_manifest_sample_accepts_exact_query():
    value = manifest()
    found = verify_manifest_sample(
        value,
        sample_id="fixture-q2",
        query_text="Fixture question two?",
    )
    assert found is value.entries[1]


def test_verify_manifest_sample_rejects_unknown_id_and_changed_query():
    value = manifest()
    with pytest.raises(KeyError, match="absent"):
        verify_manifest_sample(value, sample_id="fixture-missing", query_text="Question?")
    with pytest.raises(ValueError, match="does not match"):
        verify_manifest_sample(
            value,
            sample_id="fixture-q1",
            query_text="Fixture question one? ",
        )


def test_manifest_has_no_downstream_experiment_fields():
    fields = SampleManifest.__dataclass_fields__
    for forbidden in (
        "retriever",
        "diversification_condition",
        "model_id",
        "context_mode",
        "generation",
        "metric_id",
        "timestamp",
        "environment",
        "git_commit",
    ):
        assert forbidden not in fields
