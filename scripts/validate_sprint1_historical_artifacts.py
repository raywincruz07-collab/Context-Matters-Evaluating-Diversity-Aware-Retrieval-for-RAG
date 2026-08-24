#!/usr/bin/env python3
"""Validate immutable historical Sprint-1 result artifacts without changing them."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RAW_DIR = REPOSITORY_ROOT / "results/sprint1/raw"
DEFAULT_INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "artifacts/audit_inventories/sprint1_historical_artifact_inventory_v1.json"
)

VALIDATOR_VERSION = "sprint1_historical_artifact_validator_v1"
ARTIFACT_FORMAT = "context-matters.sprint1-historical-artifact-audit.v1"
EXPECTED_ROW_COUNT = 1000
EXPECTED_GENERATOR = "ministral-3-14b"
EXPECTED_TEMPERATURE = 0.0
EXPECTED_TOP_K = 5
EXPECTED_MAX_TOKENS = 512
EXPECTED_STATUS = "OK"

METADATA_FILENAME = "EXPERIMENT_METADATA.json"
RESULT_FILENAMES_BY_RETRIEVER = {
    "bm25": "fullrag_bm25_top5.csv",
    "dpr": "fullrag_dpr_top5.csv",
    "contriever": "fullrag_contriever_top5.csv",
    "colbertv2": "fullrag_colbertv2_top5.csv",
}
EXPECTED_SHA256_BY_FILENAME = {
    METADATA_FILENAME: (
        "3f1a681384ff47615a4345ed6cf017beed85f7794acaba5778a315dc97d48002"
    ),
    "fullrag_bm25_top5.csv": (
        "2e007274d74eb1ec44b910adc76c7748d9b77bd5e32487e841df97a859d977d0"
    ),
    "fullrag_dpr_top5.csv": (
        "9258928b4691cc7eece6f7ca26d964977a70076ccf0b528bf798946125803518"
    ),
    "fullrag_contriever_top5.csv": (
        "b17ec564df77dee58ea1e5f2ef74d82cbe903de86fd83a8b9097f87674251c05"
    ),
    "fullrag_colbertv2_top5.csv": (
        "1dad45c4039f75e09257226c53239deccb99beec0cf933c2789c2a67fc380a41"
    ),
}

REQUIRED_COLUMNS = (
    "question_index",
    "qa_id",
    "retriever",
    "question",
    "gold_doc_ids",
    "retrieved_doc_ids",
    "recall_at_k",
    "mrr",
    "llm_model",
    "temperature",
    "max_tokens",
    "prediction",
    "ground_truth",
    "exact_match",
    "f1",
    "rouge_l",
    "row_status",
    "row_error",
)
METRIC_FIELDS = ("recall_at_k", "mrr", "exact_match", "f1", "rouge_l")

KNOWN_BM25_ANOMALY = {
    "retriever": "bm25",
    "qa_id": "60",
    "question_index": "60",
    "retrieved_doc_ids": (202,),
}


class HistoricalArtifactValidationError(ValueError):
    """Raised when an immutable historical artifact violates its audit contract."""


def physical_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_exact_number(value: object, expected: float | int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoricalArtifactValidationError(
            f"metadata {field} must be numeric, found {value!r}"
        )
    if float(value) != float(expected):
        raise HistoricalArtifactValidationError(
            f"metadata {field} must be {expected}, found {value!r}"
        )


def _parse_float(value: str, *, field: str, row_number: int, filename: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} has invalid {field}: {value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} has out-of-range {field}: {value!r}"
        )
    if field == "exact_match" and parsed not in (0.0, 1.0):
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} has non-binary exact_match: {value!r}"
        )
    return parsed


def _parse_retrieved_document_ids(
    value: str, *, row_number: int, filename: str
) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} has malformed retrieved_doc_ids"
        ) from exc
    if not isinstance(parsed, list):
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} retrieved_doc_ids must be a list"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise HistoricalArtifactValidationError(
            f"{filename} row {row_number} retrieved_doc_ids must contain integers"
        )
    return tuple(parsed)


def _is_known_bm25_anomaly(
    row: Mapping[str, str], retriever: str, retrieved_doc_ids: tuple[int, ...]
) -> bool:
    return (
        retriever == KNOWN_BM25_ANOMALY["retriever"]
        and row["qa_id"] == KNOWN_BM25_ANOMALY["qa_id"]
        and row["question_index"] == KNOWN_BM25_ANOMALY["question_index"]
        and retrieved_doc_ids == KNOWN_BM25_ANOMALY["retrieved_doc_ids"]
        and row["row_status"] == EXPECTED_STATUS
    )


def _validate_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalArtifactValidationError(
            f"cannot read valid historical metadata: {path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise HistoricalArtifactValidationError("historical metadata must be an object")
    if metadata.get("retrievers") != list(RESULT_FILENAMES_BY_RETRIEVER):
        raise HistoricalArtifactValidationError(
            "metadata retrievers must be exactly bm25, dpr, contriever, colbertv2"
        )
    if metadata.get("with_generation") is not True:
        raise HistoricalArtifactValidationError(
            "metadata with_generation must record true"
        )
    _require_exact_number(metadata.get("top_k"), EXPECTED_TOP_K, "top_k")
    generator = metadata.get("generator_config")
    if not isinstance(generator, dict):
        raise HistoricalArtifactValidationError(
            "metadata generator_config must be an object"
        )
    if generator.get("llm_model") != EXPECTED_GENERATOR:
        raise HistoricalArtifactValidationError(
            f"metadata llm_model must be {EXPECTED_GENERATOR!r}"
        )
    _require_exact_number(
        generator.get("temperature"), EXPECTED_TEMPERATURE, "generator_config.temperature"
    )
    _require_exact_number(
        generator.get("max_tokens"), EXPECTED_MAX_TOKENS, "generator_config.max_tokens"
    )
    _require_exact_number(
        generator.get("top_k"), EXPECTED_TOP_K, "generator_config.top_k"
    )
    return metadata


def _validate_result_file(
    path: Path,
    *,
    retriever: str,
    expected_row_count: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
            if missing_columns:
                raise HistoricalArtifactValidationError(
                    f"{path.name} is missing required columns: {', '.join(missing_columns)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise HistoricalArtifactValidationError(
            f"cannot read historical CSV {path}"
        ) from exc

    if len(rows) != expected_row_count:
        raise HistoricalArtifactValidationError(
            f"{path.name} expected {expected_row_count} rows, found {len(rows)}"
        )

    qa_ids: list[str] = []
    seen_qa_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    cardinality_counts: Counter[int] = Counter()
    metric_values: dict[str, list[float]] = {field: [] for field in METRIC_FIELDS}
    anomalies: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=2):
        qa_id = row["qa_id"]
        if not qa_id.strip():
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} has a blank qa_id"
            )
        if qa_id in seen_qa_ids:
            raise HistoricalArtifactValidationError(
                f"{path.name} has duplicate qa_id {qa_id!r}"
            )
        seen_qa_ids.add(qa_id)
        qa_ids.append(qa_id)

        if row["retriever"] != retriever:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} retriever must be {retriever!r}"
            )
        if row["llm_model"] != EXPECTED_GENERATOR:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} llm_model must be {EXPECTED_GENERATOR!r}"
            )
        try:
            temperature = float(row["temperature"])
            max_tokens = int(row["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} has invalid generation configuration"
            ) from exc
        if not math.isfinite(temperature) or temperature != EXPECTED_TEMPERATURE:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} temperature must be {EXPECTED_TEMPERATURE}"
            )
        if max_tokens != EXPECTED_MAX_TOKENS:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} max_tokens must be {EXPECTED_MAX_TOKENS}"
            )
        if not row["prediction"].strip():
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} has a blank prediction"
            )

        status = row["row_status"]
        status_counts[status] += 1
        if status != EXPECTED_STATUS:
            raise HistoricalArtifactValidationError(
                f"{path.name} row {row_number} has unexpected row_status {status!r}"
            )

        for field in METRIC_FIELDS:
            metric_values[field].append(
                _parse_float(
                    row[field], field=field, row_number=row_number, filename=path.name
                )
            )

        retrieved_doc_ids = _parse_retrieved_document_ids(
            row["retrieved_doc_ids"], row_number=row_number, filename=path.name
        )
        cardinality = len(retrieved_doc_ids)
        cardinality_counts[cardinality] += 1
        if cardinality == EXPECTED_TOP_K:
            continue
        if _is_known_bm25_anomaly(row, retriever, retrieved_doc_ids):
            anomalies.append(
                {
                    "code": "KNOWN_BM25_ONE_DOCUMENT_ROW",
                    "qa_id": row["qa_id"],
                    "question_index": row["question_index"],
                    "retrieved_doc_ids": list(retrieved_doc_ids),
                    "row_status": row["row_status"],
                    "disposition": "preserved_as_historical_observation",
                }
            )
            continue
        raise HistoricalArtifactValidationError(
            f"{path.name} row {row_number} has unexpected retrieved-document "
            f"cardinality {cardinality} for qa_id {qa_id!r}"
        )

    if len(seen_qa_ids) != expected_row_count:
        raise HistoricalArtifactValidationError(
            f"{path.name} expected {expected_row_count} unique QA IDs, "
            f"found {len(seen_qa_ids)}"
        )
    if retriever == "bm25" and len(anomalies) != 1:
        raise HistoricalArtifactValidationError(
            f"{path.name} must contain exactly the verified BM25 qa_id=60 anomaly"
        )
    if retriever != "bm25" and anomalies:
        raise HistoricalArtifactValidationError(
            f"{path.name} contains an anomaly outside the BM25 contract"
        )

    metric_inventory = {
        field: {
            "present": True,
            "valid_count": len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for field, values in metric_values.items()
    }
    inventory = {
        "source_path": None,
        "sha256": None,
        "retriever": retriever,
        "row_count": len(rows),
        "unique_qa_count": len(seen_qa_ids),
        "generator": EXPECTED_GENERATOR,
        "temperature": EXPECTED_TEMPERATURE,
        "max_tokens": EXPECTED_MAX_TOKENS,
        "status_counts": dict(sorted(status_counts.items())),
        "retrieved_document_cardinality_counts": {
            str(key): value for key, value in sorted(cardinality_counts.items())
        },
        "metric_field_validation": metric_inventory,
        "anomalies": anomalies,
    }
    return inventory, tuple(qa_ids)


def _git_identity(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HistoricalArtifactValidationError(
            "cannot resolve Git identity for audit inventory"
        ) from exc
    return {"commit": commit, "worktree_clean": not bool(status)}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomically(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def validate_historical_sprint1_artifacts(
    *,
    input_dir: Path,
    output_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    expected_sha256_by_filename: Mapping[str, str] = EXPECTED_SHA256_BY_FILENAME,
    git_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the five frozen historical inputs and write a separate inventory."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    repository_root = Path(repository_root)
    expected_filenames = (
        METADATA_FILENAME,
        *RESULT_FILENAMES_BY_RETRIEVER.values(),
    )
    if set(expected_sha256_by_filename) != set(expected_filenames):
        raise HistoricalArtifactValidationError(
            "expected SHA-256 contract must identify exactly the five audited inputs"
        )

    input_paths = {name: input_dir / name for name in expected_filenames}
    missing = [name for name, path in input_paths.items() if not path.is_file()]
    if missing:
        raise HistoricalArtifactValidationError(
            f"missing historical input files: {', '.join(missing)}"
        )

    resolved_output = output_path.resolve()
    resolved_input_dir = input_dir.resolve()
    if resolved_output == resolved_input_dir or resolved_input_dir in resolved_output.parents:
        raise HistoricalArtifactValidationError(
            "audit inventory must not be written inside the historical raw directory"
        )

    hashes_before = {
        name: physical_file_sha256(path) for name, path in input_paths.items()
    }
    for name, expected_hash in expected_sha256_by_filename.items():
        if hashes_before[name] != expected_hash:
            raise HistoricalArtifactValidationError(
                f"historical input SHA-256 mismatch for {name}: "
                f"expected {expected_hash}, found {hashes_before[name]}"
            )

    metadata = _validate_metadata(input_paths[METADATA_FILENAME])
    result_inventories: list[dict[str, Any]] = []
    reference_qa_ids: tuple[str, ...] | None = None
    aggregate_status_counts: Counter[str] = Counter()
    aggregate_cardinality_counts: Counter[int] = Counter()
    total_anomalies = 0

    for retriever, filename in RESULT_FILENAMES_BY_RETRIEVER.items():
        inventory, qa_ids = _validate_result_file(
            input_paths[filename],
            retriever=retriever,
            expected_row_count=expected_row_count,
        )
        if reference_qa_ids is None:
            reference_qa_ids = qa_ids
        elif qa_ids != reference_qa_ids:
            raise HistoricalArtifactValidationError(
                f"{filename} QA identity/order differs from the other retriever files"
            )
        inventory["source_path"] = _display_path(
            input_paths[filename], repository_root
        )
        inventory["sha256"] = hashes_before[filename]
        result_inventories.append(inventory)
        aggregate_status_counts.update(inventory["status_counts"])
        aggregate_cardinality_counts.update(
            {
                int(key): value
                for key, value in inventory[
                    "retrieved_document_cardinality_counts"
                ].items()
            }
        )
        total_anomalies += len(inventory["anomalies"])

    if reference_qa_ids is None:
        raise HistoricalArtifactValidationError("no historical result files validated")
    expected_cardinality_counts = Counter(
        {EXPECTED_TOP_K: expected_row_count * 4 - 1, 1: 1}
    )
    if aggregate_cardinality_counts != expected_cardinality_counts:
        raise HistoricalArtifactValidationError(
            "aggregate retrieved-document cardinalities do not match the "
            f"historical contract: {dict(aggregate_cardinality_counts)}"
        )
    if total_anomalies != 1:
        raise HistoricalArtifactValidationError(
            f"expected exactly one disclosed historical anomaly, found {total_anomalies}"
        )

    hashes_after = {
        name: physical_file_sha256(path) for name, path in input_paths.items()
    }
    if hashes_after != hashes_before:
        raise HistoricalArtifactValidationError(
            "a historical input changed while validation was in progress"
        )

    identity = dict(git_identity) if git_identity is not None else _git_identity(repository_root)
    payload = {
        "artifact_format": ARTIFACT_FORMAT,
        "validator_version": VALIDATOR_VERSION,
        "git_identity": identity,
        "historical_contract": {
            "dataset": "PubMedQA PQA-L1000",
            "evidence_role": "HISTORICAL_OBSERVED",
            "expected_result_file_count": 4,
            "expected_rows_per_result_file": expected_row_count,
            "expected_unique_qa_ids_per_result_file": expected_row_count,
            "generator": EXPECTED_GENERATOR,
            "temperature": EXPECTED_TEMPERATURE,
            "top_k": EXPECTED_TOP_K,
            "max_tokens": EXPECTED_MAX_TOKENS,
            "expected_status": EXPECTED_STATUS,
        },
        "metadata": {
            "source_path": _display_path(
                input_paths[METADATA_FILENAME], repository_root
            ),
            "sha256": hashes_before[METADATA_FILENAME],
            "retrievers": metadata["retrievers"],
            "with_generation": metadata["with_generation"],
            "generator_config": metadata["generator_config"],
        },
        "question_identity": {
            "exact_order_consistent_across_retrievers": True,
            "common_unique_qa_count": len(set(reference_qa_ids)),
            "ordered_qa_ids_sha256": _canonical_json_sha256(reference_qa_ids),
        },
        "result_files": result_inventories,
        "aggregate": {
            "result_file_count": len(result_inventories),
            "total_rows": sum(item["row_count"] for item in result_inventories),
            "status_counts": dict(sorted(aggregate_status_counts.items())),
            "retrieved_document_cardinality_counts": {
                str(key): value
                for key, value in sorted(aggregate_cardinality_counts.items())
            },
            "known_anomaly_count": total_anomalies,
            "source_hashes_unchanged_during_validation": True,
        },
    }
    _write_json_atomically(payload, output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate immutable historical Sprint-1 result artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HISTORICAL_RAW_DIR,
        help="Directory containing the five historical inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help="Separate machine-readable audit inventory path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inventory = validate_historical_sprint1_artifacts(
            input_dir=args.input_dir,
            output_path=args.output,
        )
    except HistoricalArtifactValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    aggregate = inventory["aggregate"]
    print(
        "VALIDATION PASSED: "
        f"{aggregate['result_file_count']} result files, "
        f"{aggregate['total_rows']} rows, "
        f"{aggregate['known_anomaly_count']} disclosed anomaly"
    )
    print(f"Inventory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
