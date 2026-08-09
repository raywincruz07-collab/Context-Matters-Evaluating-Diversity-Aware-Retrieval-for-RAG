import csv
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from scripts.build_sample_manifests import (
    FULL_SPLIT_SOURCE_ORDER_ALGORITHM,
    build_full_split_manifest,
    build_verify_and_write,
    read_and_verify_manifest_artifact,
)


def fixture_rows():
    return (
        {"pubid": 202, "question": "Fixture question B?"},
        {"pubid": 101, "question": " Fixture question A? "},
    )


def build_fixture_manifest(rows=None):
    values = fixture_rows() if rows is None else rows
    return build_full_split_manifest(
        values,
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-fixture-source",
        config="synthetic-fixture-config",
        revision="fixture-revision-abc",
        split="fixture-split",
        expected_size=2,
    )


def write_historical_fixture(path: Path, questions=None):
    values = fixture_rows() if questions is None else questions
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("qa_id", "question"))
        writer.writeheader()
        for position, row in enumerate(values):
            writer.writerow({"qa_id": position, "question": row["question"]})


def test_fixture_manifest_is_deterministic_and_preserves_full_split_order():
    first = build_fixture_manifest()
    second = build_fixture_manifest()
    assert first == second
    assert first.manifest_id == second.manifest_id
    assert tuple(entry.position for entry in first.entries) == (0, 1)
    assert tuple(entry.sample_id for entry in first.entries) == (0, 1)
    assert tuple(entry.source_sample_id for entry in first.entries) == (202, 101)
    assert all(type(entry.source_sample_id) is int for entry in first.entries)
    assert first.entries[1].query_text_sha256 == hashlib.sha256(
        " Fixture question A? ".encode("utf-8")
    ).hexdigest()
    assert first.selection_dependencies == ()
    assert first.requested_sample_size is None
    assert first.sampling_seed is None
    assert first.sampling_algorithm == FULL_SPLIT_SOURCE_ORDER_ALGORITHM
    with pytest.raises(FrozenInstanceError):
        first.entries[0].sample_id = 99


def test_historical_mismatch_blocks_artifact_creation(tmp_path):
    historical = tmp_path / "historical.csv"
    output = tmp_path / "manifest.json"
    changed = list(fixture_rows())
    changed[1] = {"pubid": 101, "question": "Changed fixture question"}
    write_historical_fixture(historical, changed)
    with pytest.raises(ValueError, match="question mismatch"):
        build_verify_and_write(
            fixture_rows(),
            dataset_id=DatasetId.PUBMEDQA,
            source="synthetic-fixture-source",
            config="synthetic-fixture-config",
            revision="fixture-revision-abc",
            split="fixture-split",
            expected_size=2,
            historical_artifact=historical,
            output_path=output,
        )
    assert not output.exists()


def test_round_trip_preserves_manifest_identity(tmp_path):
    historical = tmp_path / "historical.csv"
    output = tmp_path / "manifest.json"
    write_historical_fixture(historical)
    written = build_verify_and_write(
        fixture_rows(),
        dataset_id=DatasetId.PUBMEDQA,
        source="synthetic-fixture-source",
        config="synthetic-fixture-config",
        revision="fixture-revision-abc",
        split="fixture-split",
        expected_size=2,
        historical_artifact=historical,
        output_path=output,
    )
    reconstructed = read_and_verify_manifest_artifact(output)
    assert reconstructed == written
    assert reconstructed.manifest_id == written.manifest_id
    assert reconstructed.sha256 == written.sha256
