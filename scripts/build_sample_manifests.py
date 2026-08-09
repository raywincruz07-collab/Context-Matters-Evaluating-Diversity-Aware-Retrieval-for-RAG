#!/usr/bin/env python3
"""Build deterministic Sprint 3 sample-manifest artifacts.

This milestone supports only the pinned PubMedQA population. The resulting
manifest freezes a revision for the clean Sprint 3 rerun; it does not identify
the exact upstream revision used historically in Sprint 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts.sample_manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    SampleSelectionDependency,
    query_text_sha256,
)


PUBMEDQA_SOURCE = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
PUBMEDQA_SPLIT = "train"
PUBMEDQA_EXPECTED_SIZE = 1000
FULL_SPLIT_SOURCE_ORDER_ALGORITHM = "full_split_source_order.v1"
ARTIFACT_FORMAT = "sprint3.sample-manifest-artifact.v1"
PROVENANCE_NOTE = (
    "This manifest freezes an immutable revision for the clean Sprint 3 rerun; "
    "it does not identify the exact upstream revision used historically in Sprint 1."
)
DEFAULT_HISTORICAL_ARTIFACT = (
    REPOSITORY_ROOT / "results/sprint1/raw/fullrag_bm25_top5.csv"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json"
)


def build_full_split_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: DatasetId,
    source: str,
    config: str | None,
    revision: str,
    split: str,
    expected_size: int,
) -> SampleManifest:
    """Build a full-split manifest while preserving supplied row order exactly."""
    if len(rows) != expected_size:
        raise ValueError(f"expected {expected_size} rows, found {len(rows)}")
    entries = tuple(
        SampleManifestEntry(
            position=position,
            sample_id=position,
            source_sample_id=row["pubid"],
            query_text_sha256=query_text_sha256(row["question"]),
        )
        for position, row in enumerate(rows)
    )
    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        source=source,
        config=config,
        revision=revision,
        split=split,
        sampling_algorithm=FULL_SPLIT_SOURCE_ORDER_ALGORITHM,
        sampling_seed=None,
        requested_sample_size=None,
        selection_dependencies=(),
        entries=entries,
    )


def verify_historical_questions(
    rows: Sequence[Mapping[str, Any]],
    historical_artifact: Path,
    *,
    expected_size: int,
) -> None:
    """Require exact historical IDs and question strings in source order."""
    with historical_artifact.open("r", encoding="utf-8", newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    if len(historical_rows) != expected_size:
        raise ValueError(
            f"historical artifact must contain {expected_size} rows, "
            f"found {len(historical_rows)}"
        )
    if len(rows) != expected_size:
        raise ValueError(f"pinned dataset must contain {expected_size} rows, found {len(rows)}")
    for position, (historical, pinned) in enumerate(
        zip(historical_rows, rows, strict=True)
    ):
        if historical.get("qa_id") != str(position):
            raise ValueError(f"historical qa_id mismatch at position {position}")
        if historical.get("question") != pinned["question"]:
            raise ValueError(f"historical question mismatch at position {position}")


def manifest_from_scientific_payload(payload: Mapping[str, Any]) -> SampleManifest:
    """Reconstruct a manifest from its deterministic scientific payload."""
    dependencies = tuple(
        SampleSelectionDependency(**dependency)
        for dependency in payload["selection_dependencies"]
    )
    entries = tuple(SampleManifestEntry(**entry) for entry in payload["entries"])
    return SampleManifest(
        schema_version=payload["schema_version"],
        dataset_id=DatasetId(payload["dataset_id"]),
        source=payload["source"],
        config=payload["config"],
        revision=payload["revision"],
        split=payload["split"],
        sampling_algorithm=payload["sampling_algorithm"],
        sampling_seed=payload["sampling_seed"],
        requested_sample_size=payload["requested_sample_size"],
        selection_dependencies=dependencies,
        entries=entries,
    )


def artifact_payload(manifest: SampleManifest) -> dict[str, Any]:
    return {
        "artifact_format": ARTIFACT_FORMAT,
        "manifest_id": manifest.manifest_id,
        "provenance_note": PROVENANCE_NOTE,
        "scientific_payload": manifest.scientific_payload(),
        "sha256": manifest.sha256,
    }


def write_manifest_artifact(manifest: SampleManifest, output_path: Path) -> None:
    """Write deterministic JSON atomically after all compatibility checks pass."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, output_path)


def read_and_verify_manifest_artifact(output_path: Path) -> SampleManifest:
    """Round-trip and verify the stored manifest identity and population."""
    with output_path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("unexpected sample-manifest artifact format")
    manifest = manifest_from_scientific_payload(stored["scientific_payload"])
    if stored.get("manifest_id") != manifest.manifest_id:
        raise ValueError("stored manifest_id does not match scientific payload")
    if stored.get("sha256") != manifest.sha256:
        raise ValueError("stored sha256 does not match scientific payload")
    return manifest


def build_verify_and_write(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: DatasetId,
    source: str,
    config: str | None,
    revision: str,
    split: str,
    expected_size: int,
    historical_artifact: Path,
    output_path: Path,
) -> SampleManifest:
    """Apply the historical gate before writing and verify the written artifact."""
    verify_historical_questions(
        rows,
        historical_artifact,
        expected_size=expected_size,
    )
    manifest = build_full_split_manifest(
        rows,
        dataset_id=dataset_id,
        source=source,
        config=config,
        revision=revision,
        split=split,
        expected_size=expected_size,
    )
    write_manifest_artifact(manifest, output_path)
    reconstructed = read_and_verify_manifest_artifact(output_path)
    if reconstructed != manifest:
        raise ValueError("round-trip manifest differs from constructed manifest")
    if reconstructed.actual_sample_size != expected_size:
        raise ValueError("round-trip manifest has an unexpected sample size")
    return reconstructed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--historical-artifact",
        type=Path,
        default=DEFAULT_HISTORICAL_ARTIFACT,
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    rows = load_dataset(
        PUBMEDQA_SOURCE,
        PUBMEDQA_CONFIG,
        split=PUBMEDQA_SPLIT,
        revision=PUBMEDQA_REVISION,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    manifest = build_verify_and_write(
        rows,
        dataset_id=DatasetId.PUBMEDQA,
        source=PUBMEDQA_SOURCE,
        config=PUBMEDQA_CONFIG,
        revision=PUBMEDQA_REVISION,
        split=PUBMEDQA_SPLIT,
        expected_size=PUBMEDQA_EXPECTED_SIZE,
        historical_artifact=args.historical_artifact,
        output_path=args.output,
    )
    source_ids = tuple(entry.source_sample_id for entry in manifest.entries)
    print(f"manifest_path={args.output}")
    print(f"manifest_id={manifest.manifest_id}")
    print(f"sha256={manifest.sha256}")
    print(f"actual_sample_size={manifest.actual_sample_size}")
    print(f"unique_source_sample_ids={len(set(source_ids))}")
    print(f"source_sample_id_types={sorted({type(value).__name__ for value in source_ids})}")
    print("historical_compatibility=1000/1000 exact ordered questions")
    print("claim_boundary=clean Sprint 3 frozen revision, not proven historical revision")


if __name__ == "__main__":
    main()
