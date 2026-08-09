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
    SampleSelectionDependency,
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


def dependency(**changes):
    values = dict(
        role="fixture-eligibility",
        source="synthetic-fixture-selection-source",
        config="synthetic-fixture-config",
        revision="fixture-selection-revision-abc",
        split="fixture-selection-split",
    )
    values.update(changes)
    return SampleSelectionDependency(**values)


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
        selection_dependencies=(),
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


def test_valid_selection_dependency_is_frozen_and_empty_tuple_is_allowed():
    value = dependency()
    assert manifest().selection_dependencies == ()
    with pytest.raises(FrozenInstanceError):
        value.role = "changed"


def test_selection_dependency_validation():
    for field_name in ("role", "source", "revision", "split"):
        with pytest.raises(ValueError):
            dependency(**{field_name: " "})
    with pytest.raises(ValueError):
        dependency(config="")
    assert dependency(config=None).config is None


def test_selection_dependency_collection_validation():
    value = dependency()
    with pytest.raises(TypeError, match="immutable tuple"):
        manifest(selection_dependencies=[value])
    with pytest.raises(TypeError, match="SampleSelectionDependency"):
        manifest(selection_dependencies=("not-a-dependency",))
    with pytest.raises(ValueError, match="duplicates"):
        manifest(selection_dependencies=(value, value))


def test_selection_dependency_is_structured_in_scientific_payload():
    value = dependency()
    payload = manifest(selection_dependencies=(value,)).scientific_payload()
    assert payload["selection_dependencies"] == [
        {
            "role": value.role,
            "source": value.source,
            "config": value.config,
            "revision": value.revision,
            "split": value.split,
        }
    ]


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
    with_dependency = dict(selection_dependencies=(dependency(),))
    assert manifest(**with_dependency).manifest_id == manifest(**with_dependency).manifest_id


@pytest.mark.parametrize(
    "changed_dependency",
    [
        lambda value: replace(value, role="fixture-other-role"),
        lambda value: replace(value, source="synthetic-fixture-other-source"),
        lambda value: replace(value, config="synthetic-fixture-other-config"),
        lambda value: replace(value, revision="fixture-selection-revision-def"),
        lambda value: replace(value, split="fixture-selection-other-split"),
    ],
)
def test_dependency_presence_and_fields_change_manifest_identity(changed_dependency):
    original = manifest()
    value = dependency()
    with_dependency = manifest(selection_dependencies=(value,))
    changed = manifest(selection_dependencies=(changed_dependency(value),))
    assert with_dependency.manifest_id != original.manifest_id
    assert changed.manifest_id != with_dependency.manifest_id


def test_dependency_order_changes_manifest_identity():
    first = dependency(role="fixture-first")
    second = dependency(role="fixture-second")
    assert manifest(selection_dependencies=(first, second)).manifest_id != manifest(
        selection_dependencies=(second, first)
    ).manifest_id


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
        manifest(schema_version="sprint3.sample-manifest.v1")
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


def test_dataset_provenance_is_bound_to_selection_dependencies():
    original = manifest()
    changed = manifest(selection_dependencies=(dependency(),))
    original_provenance = dataset_provenance_from_sample_manifest(original)
    changed_provenance = dataset_provenance_from_sample_manifest(changed)
    assert changed_provenance.sample_manifest_id == changed.manifest_id
    assert changed_provenance.sample_manifest_sha256 == changed.sha256
    assert changed_provenance.sample_manifest_id != original_provenance.sample_manifest_id
    assert (
        changed_provenance.sample_manifest_sha256
        != original_provenance.sample_manifest_sha256
    )


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
    for contract in (SampleManifest, SampleSelectionDependency):
        fields = contract.__dataclass_fields__
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
