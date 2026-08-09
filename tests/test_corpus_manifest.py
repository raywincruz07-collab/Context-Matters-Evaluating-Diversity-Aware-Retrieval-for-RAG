from dataclasses import FrozenInstanceError, replace
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusConstructionDependency,
    CorpusManifest,
    CorpusManifestEntry,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SAMPLE_ID = f"sample-manifest:sha256:{SHA_A}"


def entry(position=0, doc_id=0, source_document_id="source-0", **changes):
    values = dict(
        position=position,
        doc_id=doc_id,
        source_document_id=source_document_id,
        title_sha256=None,
        text_sha256=SHA_B,
        retrieval_content_sha256=SHA_C,
    )
    values.update(changes)
    return CorpusManifestEntry(**values)


def dependency(**changes):
    values = dict(
        role="fixture-corpus-source",
        source="synthetic-corpus-dependency",
        config="fixture-config",
        revision="fixture-dependency-revision",
        split="fixture-split",
    )
    values.update(changes)
    return CorpusConstructionDependency(**values)


def manifest(**changes):
    values = dict(
        schema_version=CORPUS_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-primary-corpus",
        config="fixture-config",
        revision="fixture-corpus-revision",
        split="fixture-split",
        construction_algorithm="fixture-corpus-construction.v1",
        input_sample_manifest_id=SAMPLE_ID,
        input_sample_manifest_sha256=SHA_A,
        dependencies=(),
        rng_family=None,
        sampling_seed=None,
        rng_state_semantics=None,
        requested_negatives_per_query=None,
        negative_sampling_scope=None,
        negative_exclusion_scope=None,
        negative_sampling_without_replacement=None,
        final_source_id_ordering=None,
        entries=(
            entry(),
            entry(1, 1, "source-1", text_sha256="d" * 64),
        ),
    )
    values.update(changes)
    return CorpusManifest(**values)


def sampled_manifest(**changes):
    values = dict(
        dataset_id=DatasetId.HOTPOTQA,
        rng_family="python.random.Random",
        sampling_seed=42,
        rng_state_semantics="continue state after query sampling",
        requested_negatives_per_query=50,
        negative_sampling_scope="global",
        negative_exclusion_scope="all gold IDs across selected queries",
        negative_sampling_without_replacement=True,
        final_source_id_ordering="lexicographic ascending",
    )
    values.update(changes)
    return manifest(**values)


def test_entry_and_manifest_are_frozen_and_optional_fields_are_valid():
    value = manifest()
    assert value.document_count == 2
    assert value.entries[0].title_sha256 is None
    assert value.rng_family is None
    with pytest.raises(FrozenInstanceError):
        value.revision = "changed"
    with pytest.raises(FrozenInstanceError):
        value.entries[0].doc_id = "changed"


def test_dependency_is_frozen_and_validated():
    value = dependency(config=None)
    with pytest.raises(FrozenInstanceError):
        value.role = "changed"
    for name in ("role", "source", "revision", "split"):
        with pytest.raises(ValueError):
            dependency(**{name: " "})
    with pytest.raises(ValueError):
        dependency(config="")


def test_collection_types_and_duplicates_are_validated():
    dep = dependency()
    with pytest.raises(TypeError, match="immutable tuple"):
        manifest(dependencies=[dep])
    with pytest.raises(TypeError, match="CorpusConstructionDependency"):
        manifest(dependencies=("bad",))
    with pytest.raises(ValueError, match="duplicates"):
        manifest(dependencies=(dep, dep))
    with pytest.raises(TypeError, match="immutable tuple"):
        manifest(entries=list(manifest().entries))


@pytest.mark.parametrize("bad_id", [True, False, np.bool_(True), 1.5, "", " "])
def test_entry_rejects_invalid_ids(bad_id):
    with pytest.raises((TypeError, ValueError)):
        entry(doc_id=bad_id)
    with pytest.raises((TypeError, ValueError)):
        entry(source_document_id=bad_id)


def test_entry_hash_validation():
    for name in ("title_sha256", "text_sha256", "retrieval_content_sha256"):
        with pytest.raises(ValueError):
            entry(**{name: "A" * 64})
        with pytest.raises(ValueError):
            entry(**{name: "abc"})


def test_positions_and_document_ids_must_be_unique_and_contiguous():
    with pytest.raises(ValueError, match="positions"):
        manifest(entries=(entry(position=1),))
    with pytest.raises(ValueError, match="doc_id values"):
        manifest(entries=(entry(), entry(1, 0, "source-1")))
    with pytest.raises(ValueError, match="source_document_id values"):
        manifest(entries=(entry(), entry(1, 1, "source-0")))


def test_manifest_core_validation_and_sample_manifest_binding():
    with pytest.raises(ValueError):
        manifest(schema_version="sprint3.corpus-manifest.v0")
    with pytest.raises(TypeError):
        manifest(dataset_id="pubmedqa")
    for name in ("source", "revision", "split", "construction_algorithm"):
        with pytest.raises(ValueError):
            manifest(**{name: " "})
    with pytest.raises(ValueError):
        manifest(config="")
    assert manifest().input_sample_manifest_id == SAMPLE_ID
    assert manifest().input_sample_manifest_sha256 == SHA_A
    unbound = manifest(
        input_sample_manifest_id=None,
        input_sample_manifest_sha256=None,
    )
    assert unbound.scientific_payload()["input_sample_manifest_id"] is None
    assert unbound.scientific_payload()["input_sample_manifest_sha256"] is None
    with pytest.raises(ValueError, match="provided together"):
        manifest(input_sample_manifest_sha256=None)
    with pytest.raises(ValueError, match="provided together"):
        manifest(input_sample_manifest_id=None)
    with pytest.raises(ValueError, match="SampleManifest ID"):
        manifest(input_sample_manifest_id="not-an-id")
    with pytest.raises(ValueError, match="same manifest"):
        manifest(input_sample_manifest_sha256=SHA_B)


def test_bound_and_unbound_corpora_have_distinct_identity():
    bound = manifest()
    unbound = manifest(
        input_sample_manifest_id=None,
        input_sample_manifest_sha256=None,
    )
    assert bound.corpus_manifest_id != unbound.corpus_manifest_id


def test_rng_and_negative_sampling_combinations_are_strict():
    assert sampled_manifest().requested_negatives_per_query == 50
    with pytest.raises(ValueError, match="provided together"):
        manifest(rng_family="python.random.Random")
    with pytest.raises(TypeError):
        sampled_manifest(sampling_seed=True)
    with pytest.raises(ValueError):
        sampled_manifest(sampling_seed=-1)
    with pytest.raises(ValueError, match="all negative-sampling"):
        manifest(requested_negatives_per_query=50)
    with pytest.raises(ValueError):
        sampled_manifest(requested_negatives_per_query=-1)
    with pytest.raises(TypeError):
        sampled_manifest(negative_sampling_without_replacement=1)
    with pytest.raises(ValueError, match="requires explicit RNG"):
        manifest(
            requested_negatives_per_query=0,
            negative_sampling_scope="global",
            negative_exclusion_scope="all selected-query gold IDs",
            negative_sampling_without_replacement=True,
            final_source_id_ordering="lexicographic ascending",
        )
    with pytest.raises(
        ValueError, match="negative sampling requires a SampleManifest binding"
    ):
        sampled_manifest(
            input_sample_manifest_id=None,
            input_sample_manifest_sha256=None,
        )


def test_identity_and_json_are_deterministic_and_well_formed():
    first = sampled_manifest(dependencies=(dependency(),))
    second = sampled_manifest(dependencies=(dependency(),))
    assert first.scientific_payload() == second.scientific_payload()
    assert first.scientific_json() == second.scientific_json()
    assert json.loads(first.scientific_json()) == first.scientific_payload()
    assert first.sha256 == second.sha256
    assert first.corpus_manifest_id == second.corpus_manifest_id
    assert re.fullmatch(
        r"corpus-manifest:sha256:[0-9a-f]{64}", first.corpus_manifest_id
    )


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, revision="fixture-other-revision"),
        lambda value: replace(
            value,
            input_sample_manifest_id=f"sample-manifest:sha256:{SHA_B}",
            input_sample_manifest_sha256=SHA_B,
        ),
        lambda value: replace(value, construction_algorithm="fixture-other.v1"),
        lambda value: replace(value, rng_family="fixture.other.Random"),
        lambda value: replace(value, sampling_seed=43),
        lambda value: replace(value, rng_state_semantics="reset before negatives"),
        lambda value: replace(value, requested_negatives_per_query=49),
        lambda value: replace(value, negative_sampling_scope="per-query"),
        lambda value: replace(value, negative_exclusion_scope="current-query gold IDs"),
        lambda value: replace(value, negative_sampling_without_replacement=False),
        lambda value: replace(value, final_source_id_ordering="source encounter order"),
    ],
)
def test_construction_fields_change_identity(changed):
    original = sampled_manifest()
    assert changed(original).corpus_manifest_id != original.corpus_manifest_id


def test_dependency_order_is_nonsemantic_but_content_changes_identity():
    first = dependency(role="fixture-first")
    second = dependency(role="fixture-second", config=None)
    original = sampled_manifest(dependencies=(first, second))
    changed_revision = sampled_manifest(
        dependencies=(replace(first, revision="fixture-other-revision"), second)
    )
    reordered = sampled_manifest(dependencies=(second, first))
    assert changed_revision.corpus_manifest_id != original.corpus_manifest_id
    assert reordered.corpus_manifest_id == original.corpus_manifest_id
    assert reordered.scientific_json() == original.scientific_json()
    assert (
        sampled_manifest(dependencies=(first,)).corpus_manifest_id
        != original.corpus_manifest_id
    )


def test_entry_order_and_content_fields_change_identity():
    original = manifest()
    reversed_entries = tuple(
        replace(value, position=position)
        for position, value in enumerate(reversed(original.entries))
    )
    variants = (
        replace(original, entries=reversed_entries),
        replace(
            original,
            entries=(
                replace(original.entries[0], source_document_id="other-source"),
                original.entries[1],
            ),
        ),
        replace(
            original,
            entries=(
                replace(original.entries[0], text_sha256="e" * 64),
                original.entries[1],
            ),
        ),
        replace(
            original,
            entries=(
                replace(original.entries[0], retrieval_content_sha256="f" * 64),
                original.entries[1],
            ),
        ),
    )
    assert all(value.corpus_manifest_id != original.corpus_manifest_id for value in variants)


def test_identifier_types_are_preserved_in_scientific_payload():
    integer = manifest(entries=(entry(doc_id=1, source_document_id=2),))
    string = manifest(entries=(entry(doc_id="1", source_document_id="2"),))
    assert integer.scientific_payload()["entries"][0]["doc_id"] == 1
    assert string.scientific_payload()["entries"][0]["doc_id"] == "1"
    assert integer.corpus_manifest_id != string.corpus_manifest_id


def test_scientific_payload_excludes_production_and_runtime_metadata():
    payload = manifest().scientific_payload()
    forbidden = {
        "git_commit",
        "worktree_clean",
        "machine",
        "gpu",
        "cache_path",
        "timestamp",
        "environment",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(CorpusManifest.__dataclass_fields__)
