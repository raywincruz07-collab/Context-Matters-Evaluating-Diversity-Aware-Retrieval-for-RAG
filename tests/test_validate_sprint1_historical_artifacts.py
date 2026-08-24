import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_sprint1_historical_artifacts import (
    EXPECTED_SHA256_BY_FILENAME,
    HistoricalArtifactValidationError,
    METADATA_FILENAME,
    REQUIRED_COLUMNS,
    RESULT_FILENAMES_BY_RETRIEVER,
    physical_file_sha256,
    validate_historical_sprint1_artifacts,
)


def _fixture_row(retriever: str, qa_id: str) -> dict[str, object]:
    retrieved = [202] if retriever == "bm25" and qa_id == "60" else [1, 2, 3, 4, 5]
    return {
        "question_index": qa_id,
        "qa_id": qa_id,
        "retriever": retriever,
        "question": f"Historical fixture question {qa_id}?",
        "gold_doc_ids": [1],
        "retrieved_doc_ids": retrieved,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "llm_model": "ministral-3-14b",
        "temperature": 0.0,
        "max_tokens": 512,
        "prediction": f"Historical fixture prediction {qa_id}",
        "ground_truth": "Historical fixture ground truth",
        "exact_match": 0.0,
        "f1": 0.5,
        "rouge_l": 0.4,
        "row_status": "OK",
        "row_error": "",
    }


def _write_csv(path: Path, rows, *, fieldnames=REQUIRED_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_fixture(root: Path) -> Path:
    raw_dir = root / "results/sprint1/raw"
    raw_dir.mkdir(parents=True)
    metadata = {
        "retrievers": list(RESULT_FILENAMES_BY_RETRIEVER),
        "top_k": 5,
        "limit": None,
        "with_generation": True,
        "generator_config": {
            "llm_provider": "Historical fixture provider",
            "llm_model": "ministral-3-14b",
            "llm_host": "https://fixture.invalid/v1",
            "temperature": 0.0,
            "max_tokens": 512,
            "top_k": 5,
        },
    }
    (raw_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for retriever, filename in RESULT_FILENAMES_BY_RETRIEVER.items():
        _write_csv(
            raw_dir / filename,
            [_fixture_row(retriever, "60"), _fixture_row(retriever, "61")],
        )
    return raw_dir


def _fixture_hashes(raw_dir: Path) -> dict[str, str]:
    return {
        filename: (
            physical_file_sha256(raw_dir / filename)
            if (raw_dir / filename).is_file()
            else "0" * 64
        )
        for filename in EXPECTED_SHA256_BY_FILENAME
    }


def _validate_fixture(root: Path, raw_dir: Path):
    output = root / "artifacts/audit_inventories/fixture_inventory.json"
    payload = validate_historical_sprint1_artifacts(
        input_dir=raw_dir,
        output_path=output,
        repository_root=root,
        expected_row_count=2,
        expected_sha256_by_filename=_fixture_hashes(raw_dir),
        git_identity={"commit": "fixture-commit", "worktree_clean": True},
    )
    return output, payload


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_valid_historical_shaped_fixture_passes_and_writes_inventory(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    output, payload = _validate_fixture(tmp_path, raw_dir)
    assert output.is_file()
    assert payload["aggregate"] == {
        "result_file_count": 4,
        "total_rows": 8,
        "status_counts": {"OK": 8},
        "retrieved_document_cardinality_counts": {"1": 1, "5": 7},
        "known_anomaly_count": 1,
        "source_hashes_unchanged_during_validation": True,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_missing_file_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    (raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]).unlink()
    with pytest.raises(HistoricalArtifactValidationError, match="missing historical"):
        _validate_fixture(tmp_path, raw_dir)


def test_wrong_row_count_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    _write_csv(path, _read_rows(path)[:1])
    with pytest.raises(HistoricalArtifactValidationError, match="expected 2 rows"):
        _validate_fixture(tmp_path, raw_dir)


def test_duplicate_qa_id_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    rows = _read_rows(path)
    rows[1]["qa_id"] = rows[0]["qa_id"]
    _write_csv(path, rows)
    with pytest.raises(HistoricalArtifactValidationError, match="duplicate qa_id"):
        _validate_fixture(tmp_path, raw_dir)


def test_blank_prediction_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    rows = _read_rows(path)
    rows[0]["prediction"] = "   "
    _write_csv(path, rows)
    with pytest.raises(HistoricalArtifactValidationError, match="blank prediction"):
        _validate_fixture(tmp_path, raw_dir)


def test_unexpected_status_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    rows = _read_rows(path)
    rows[0]["row_status"] = "ERROR"
    _write_csv(path, rows)
    with pytest.raises(HistoricalArtifactValidationError, match="unexpected row_status"):
        _validate_fixture(tmp_path, raw_dir)


def test_missing_metric_column_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    rows = _read_rows(path)
    fieldnames = tuple(name for name in REQUIRED_COLUMNS if name != "rouge_l")
    _write_csv(path, rows, fieldnames=fieldnames)
    with pytest.raises(HistoricalArtifactValidationError, match="missing required columns"):
        _validate_fixture(tmp_path, raw_dir)


def test_unexpected_retrieved_cardinality_fails(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["dpr"]
    rows = _read_rows(path)
    rows[0]["retrieved_doc_ids"] = "[1, 2]"
    _write_csv(path, rows)
    with pytest.raises(
        HistoricalArtifactValidationError, match="unexpected retrieved-document cardinality"
    ):
        _validate_fixture(tmp_path, raw_dir)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (("qa_id", "62"), ("question_index", "61"), ("retrieved_doc_ids", "[999]")),
)
def test_bm25_one_document_anomaly_requires_exact_verified_identity(
    tmp_path, changed_field, changed_value
):
    raw_dir = _write_valid_fixture(tmp_path)
    path = raw_dir / RESULT_FILENAMES_BY_RETRIEVER["bm25"]
    rows = _read_rows(path)
    rows[0][changed_field] = changed_value
    _write_csv(path, rows)
    with pytest.raises(
        HistoricalArtifactValidationError, match="unexpected retrieved-document cardinality"
    ):
        _validate_fixture(tmp_path, raw_dir)


def test_validator_does_not_modify_source_inputs(tmp_path):
    raw_dir = _write_valid_fixture(tmp_path)
    before = {
        path.name: (path.read_bytes(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in raw_dir.iterdir()
    }
    _validate_fixture(tmp_path, raw_dir)
    after = {
        path.name: (path.read_bytes(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in raw_dir.iterdir()
    }
    assert after == before
