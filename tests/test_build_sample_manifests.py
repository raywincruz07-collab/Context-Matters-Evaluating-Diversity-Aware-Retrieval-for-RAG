import csv
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from scripts.build_sample_manifests import (
    FULL_SPLIT_SOURCE_ORDER_ALGORITHM,
    HOTPOTQA_SAMPLING_ALGORITHM,
    HotpotSelection,
    build_hotpotqa_manifest,
    build_full_split_manifest,
    build_verify_and_write,
    build_verify_and_write_hotpotqa,
    physical_file_sha256,
    read_and_verify_manifest_artifact,
    select_hotpotqa_population,
)
from retrieval_artifacts import SampleSelectionDependency


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


def hotpot_query_rows():
    return tuple(
        {"_id": qid, "text": f"Exact fixture query {qid}?"}
        for qid in ("1", "10", "2", "3", "4")
    )


def hotpot_qrel_rows():
    return (
        {"query-id": 4, "score": 0},
        {"query-id": 3, "score": 2},
        {"query-id": 1, "score": 1},
        {"query-id": 2, "score": -1},
        {"query-id": 10, "score": 1},
        {"query-id": 3, "score": 1},
    )


def fixture_dependency(**changes):
    values = dict(
        role="fixture-eligibility",
        source="synthetic-fixture-qrels",
        config="synthetic-fixture-config",
        revision="fixture-qrels-revision-abc",
        split="fixture-test",
    )
    values.update(changes)
    return SampleSelectionDependency(**values)


def select_fixture_hotpot():
    return select_hotpotqa_population(
        hotpot_query_rows(),
        hotpot_qrel_rows(),
        requested_size=2,
        seed=42,
        expected_query_count=5,
    )


def test_hotpot_selection_exact_threshold_string_conversion_and_sorting():
    selection = select_fixture_hotpot()
    eligible = ["1", "10", "3"]
    expected = tuple(sorted(random.Random(42).sample(eligible, 2)))
    assert selection.selected_qids == expected
    assert selection.selected_qids == tuple(sorted(selection.selected_qids))
    assert selection.relevant_qrel_row_count == 4
    assert selection.eligible_qid_count == 3
    reordered = select_hotpotqa_population(
        tuple(reversed(hotpot_query_rows())),
        tuple(reversed(hotpot_qrel_rows())),
        requested_size=2,
        seed=42,
        expected_query_count=5,
    )
    assert reordered.selected_qids == selection.selected_qids


def test_hotpot_manifest_identity_entries_and_dependency():
    selection = select_fixture_hotpot()
    dependency = fixture_dependency()
    manifest = build_hotpotqa_manifest(
        selection,
        source="synthetic-fixture-queries",
        config="synthetic-fixture-config",
        revision="fixture-query-revision-abc",
        split="fixture-queries",
        qrels_dependency=dependency,
        requested_size=2,
        seed=42,
    )
    assert manifest.sampling_algorithm == HOTPOTQA_SAMPLING_ALGORITHM
    assert manifest.sampling_seed == 42
    assert manifest.requested_sample_size == 2
    assert manifest.selection_dependencies == (dependency,)
    assert all(type(entry.source_sample_id) is str for entry in manifest.entries)
    assert manifest.entries[0].query_text_sha256 == hashlib.sha256(
        selection.selected_query_texts[0].encode("utf-8")
    ).hexdigest()
    changed = build_hotpotqa_manifest(
        selection,
        source="synthetic-fixture-queries",
        config="synthetic-fixture-config",
        revision="fixture-query-revision-abc",
        split="fixture-queries",
        qrels_dependency=fixture_dependency(revision="fixture-qrels-revision-def"),
        requested_size=2,
        seed=42,
    )
    assert changed.manifest_id != manifest.manifest_id


@pytest.mark.parametrize("mismatch", ["id", "question"])
def test_hotpot_historical_mismatch_blocks_artifact_creation(tmp_path, mismatch):
    selection = select_fixture_hotpot()
    historical = tmp_path / "historical.csv"
    output = tmp_path / "manifest.json"
    with historical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("qa_id", "beir_query_id", "question"),
        )
        writer.writeheader()
        for position, (qid, text) in enumerate(
            zip(selection.selected_qids, selection.selected_query_texts, strict=True)
        ):
            writer.writerow(
                {
                    "qa_id": position,
                    "beir_query_id": "changed" if mismatch == "id" and position == 0 else qid,
                    "question": "changed" if mismatch == "question" and position == 0 else text,
                }
            )
    expected_error = (
        "historical query ID mismatch"
        if mismatch == "id"
        else "historical question mismatch"
    )
    with pytest.raises(ValueError, match=expected_error):
        build_verify_and_write_hotpotqa(
            hotpot_query_rows(),
            hotpot_qrel_rows(),
            source="synthetic-fixture-queries",
            config="synthetic-fixture-config",
            revision="fixture-query-revision-abc",
            split="fixture-queries",
            qrels_dependency=fixture_dependency(),
            requested_size=2,
            seed=42,
            expected_query_count=5,
            historical_artifact=historical,
            output_path=output,
            provenance_note="Synthetic fixture provenance note.",
        )
    assert not output.exists()


def test_hotpot_missing_and_duplicate_query_ids_fail():
    with pytest.raises(ValueError, match="absent"):
        select_hotpotqa_population(
            ({"_id": "present", "text": "Fixture text"},),
            ({"query-id": "missing", "score": 1},),
            requested_size=1,
            seed=42,
        )
    with pytest.raises(ValueError, match="duplicate"):
        select_hotpotqa_population(
            (
                {"_id": "same", "text": "Fixture one"},
                {"_id": "same", "text": "Fixture two"},
            ),
            ({"query-id": "same", "score": 1},),
            requested_size=1,
            seed=42,
        )


def test_hotpot_round_trip_preserves_identity(tmp_path):
    selection = select_fixture_hotpot()
    historical = tmp_path / "historical.csv"
    output = tmp_path / "manifest.json"
    with historical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("qa_id", "beir_query_id", "question"),
        )
        writer.writeheader()
        for position, (qid, text) in enumerate(
            zip(selection.selected_qids, selection.selected_query_texts, strict=True)
        ):
            writer.writerow(
                {"qa_id": position, "beir_query_id": qid, "question": text}
            )
    written, returned_selection = build_verify_and_write_hotpotqa(
        hotpot_query_rows(),
        hotpot_qrel_rows(),
        source="synthetic-fixture-queries",
        config="synthetic-fixture-config",
        revision="fixture-query-revision-abc",
        split="fixture-queries",
        qrels_dependency=fixture_dependency(),
        requested_size=2,
        seed=42,
        expected_query_count=5,
        historical_artifact=historical,
        output_path=output,
        provenance_note="Synthetic fixture provenance note.",
    )
    reconstructed = read_and_verify_manifest_artifact(
        output,
        expected_provenance_note="Synthetic fixture provenance note.",
    )
    assert returned_selection == selection
    assert reconstructed == written
    assert reconstructed.manifest_id == written.manifest_id
    assert reconstructed.sha256 == written.sha256
    assert physical_file_sha256(output) == hashlib.sha256(output.read_bytes()).hexdigest()


def test_committed_pubmedqa_artifact_physical_hash_is_unchanged():
    artifact = Path("artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json")
    assert physical_file_sha256(artifact) == (
        "3b04a4d92a32738da68eb9c8a3cfeb579e4559adc2402716d83cc815aeaf95da"
    )
