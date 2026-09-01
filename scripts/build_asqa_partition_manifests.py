#!/usr/bin/env python3
"""Build deterministic ASQA DEVELOPMENT and SELECTION manifests."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metric_registry import DatasetId
from retrieval_artifacts.sample_manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    query_text_sha256,
)
from scripts.build_sample_manifests import (
    read_and_verify_manifest_artifact,
    write_manifest_artifact,
)


ASQA_SOURCE = "din0s/asqa"
ASQA_REVISION = "084060f16b46f3165318f760b2339208b19a0bde"
ASQA_TRAIN_FILE = "data/train-00000-of-00001-87b7d64f7913b544.parquet"
ASQA_DEV_FILE = "data/dev-00000-of-00001-58a9a40c6e69f07b.parquet"

ASQA_TRAIN_SHA256 = (
    "ce8b9c0563bfcc746afaea504701aa90"
    "da9f997ab1f45fe5dc7ba64fd6d7f619"
)

ASQA_DEV_SHA256 = (
    "9e017bc3a4409902ec994e87c1d0407c"
    "06d4b09342bae982031b6c3ea4daaf5b"
)

EXPECTED_TRAIN = 4353
EXPECTED_DEV = 948
DEVELOPMENT_SIZE = 3482
SELECTION_SIZE = 871

PARTITION_NAMESPACE = (
    "context-matters-rag|sprint3|ASQA|"
    "TRAIN_INTERNAL_PARTITION|20260823|"
)

DEVELOPMENT_OUTPUT = (
    ROOT / "artifacts/sample_manifests/asqa_development_manifest_v1.json"
)

SELECTION_OUTPUT = (
    ROOT / "artifacts/sample_manifests/asqa_selection_manifest_v1.json"
)

PROVENANCE_OUTPUT = (
    ROOT / "artifacts/sample_manifests/asqa_partition_provenance_v1.json"
)

BUILDER_VERSION = "asqa_partition_manifest_builder.v1"


def file_sha256(path: Path) -> str:
    h = sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def priority(sample_id: str) -> str:
    return sha256(
        (PARTITION_NAMESPACE + sample_id).encode("utf-8")
    ).hexdigest()


def ordered_id_sequence_sha256(ids: list[str]) -> str:
    """Hash an ordered canonical ASQA ID sequence deterministically."""
    encoded = json.dumps(
        ids,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_manifest(
    rows: list[dict],
    *,
    role: str,
) -> SampleManifest:

    entries = tuple(
        SampleManifestEntry(
            position=position,
            sample_id=row["sample_id"],
            source_sample_id=row["sample_id"],
            query_text_sha256=query_text_sha256(
                row["ambiguous_question"]
            ),
        )
        for position, row in enumerate(rows)
    )

    return SampleManifest(
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        dataset_id=DatasetId.ASQA,
        source=ASQA_SOURCE,
        config=None,
        revision=ASQA_REVISION,
        split="train",
        sampling_algorithm=(
            "asqa_train_internal_partition_sha256_20260823."
            f"{role.lower()}.v1"
        ),
        sampling_seed=None,
        requested_sample_size=len(entries),
        selection_dependencies=(),
        entries=entries,
    )


def main() -> None:

    path = Path(
        hf_hub_download(
            repo_id=ASQA_SOURCE,
            repo_type="dataset",
            revision=ASQA_REVISION,
            filename=ASQA_TRAIN_FILE,
            cache_dir="/workspace/hf-cache",
        )
    )

    actual_sha = file_sha256(path)

    if actual_sha != ASQA_TRAIN_SHA256:
        raise RuntimeError(
            f"ASQA train SHA256 mismatch: {actual_sha}"
        )

    table = pq.read_table(
        path,
        columns=["sample_id", "ambiguous_question"],
    )

    rows = table.to_pylist()

    dev_path = Path(
        hf_hub_download(
            repo_id=ASQA_SOURCE,
            repo_type="dataset",
            revision=ASQA_REVISION,
            filename=ASQA_DEV_FILE,
            cache_dir="/workspace/hf-cache",
        )
    )

    actual_dev_sha = file_sha256(dev_path)

    if actual_dev_sha != ASQA_DEV_SHA256:
        raise RuntimeError(
            f"ASQA dev SHA256 mismatch: {actual_dev_sha}"
        )

    dev_ids = pq.read_table(
        dev_path,
        columns=["sample_id"],
    )["sample_id"].to_pylist()

    if len(dev_ids) != EXPECTED_DEV:
        raise RuntimeError(
            f"Expected {EXPECTED_DEV} dev rows, found {len(dev_ids)}"
        )

    if len(set(dev_ids)) != EXPECTED_DEV:
        raise RuntimeError("ASQA dev sample_id values are not unique")

    for sid in dev_ids:
        if not isinstance(sid, str):
            raise TypeError("ASQA dev sample_id must be a string")
        if sid != str(int(sid)):
            raise ValueError(
                f"Non-canonical ASQA dev sample_id: {sid!r}"
            )

    if len(rows) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} train rows, found {len(rows)}"
        )

    seen: set[str] = set()

    for row in rows:
        sid = row["sample_id"]

        if not isinstance(sid, str):
            raise TypeError("ASQA sample_id must be a string")

        if sid != str(int(sid)):
            raise ValueError(
                f"Non-canonical ASQA sample_id: {sid!r}"
            )

        if sid in seen:
            raise ValueError(f"Duplicate ASQA sample_id: {sid}")

        seen.add(sid)

    # Membership depends ONLY on sample_id.
    ordered = sorted(
        rows,
        key=lambda row: (
            priority(row["sample_id"]),
            row["sample_id"].encode("utf-8"),
        ),
    )

    development_rows = ordered[:DEVELOPMENT_SIZE]
    selection_rows = ordered[DEVELOPMENT_SIZE:]

    if len(development_rows) != DEVELOPMENT_SIZE:
        raise RuntimeError("DEVELOPMENT size mismatch")

    if len(selection_rows) != SELECTION_SIZE:
        raise RuntimeError("SELECTION size mismatch")

    development_ids = {
        row["sample_id"] for row in development_rows
    }

    selection_ids = {
        row["sample_id"] for row in selection_rows
    }

    if development_ids & selection_ids:
        raise RuntimeError("Partition overlap detected")

    if len(development_ids | selection_ids) != EXPECTED_TRAIN:
        raise RuntimeError("Partition union mismatch")

    if (development_ids | selection_ids) & set(dev_ids):
        raise RuntimeError("Train/dev sample_id overlap detected")

    development_manifest = build_manifest(
        development_rows,
        role="DEVELOPMENT",
    )

    selection_manifest = build_manifest(
        selection_rows,
        role="SELECTION",
    )

    write_manifest_artifact(
        development_manifest,
        DEVELOPMENT_OUTPUT,
        provenance_note=(
            "ASQA DEVELOPMENT partition derived prospectively "
            "from exact source sample_id only."
        ),
    )

    write_manifest_artifact(
        selection_manifest,
        SELECTION_OUTPUT,
        provenance_note=(
            "ASQA SELECTION partition derived prospectively "
            "from exact source sample_id only."
        ),
    )

    dev_check = read_and_verify_manifest_artifact(
        DEVELOPMENT_OUTPUT
    )

    sel_check = read_and_verify_manifest_artifact(
        SELECTION_OUTPUT
    )

    if dev_check != development_manifest:
        raise RuntimeError("DEVELOPMENT manifest round-trip mismatch")

    if sel_check != selection_manifest:
        raise RuntimeError("SELECTION manifest round-trip mismatch")

    train_ids_source_order = [
        row["sample_id"] for row in rows
    ]
    development_ids_ordered = [
        row["sample_id"] for row in development_rows
    ]
    selection_ids_ordered = [
        row["sample_id"] for row in selection_rows
    ]

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    provenance_scientific_payload = {
        "builder_version": BUILDER_VERSION,
        "dataset": "ASQA",
        "source": ASQA_SOURCE,
        "revision": ASQA_REVISION,
        "train": {
            "file": ASQA_TRAIN_FILE,
            "physical_sha256": ASQA_TRAIN_SHA256,
            "count": EXPECTED_TRAIN,
            "ordered_id_sequence_sha256": ordered_id_sequence_sha256(
                train_ids_source_order
            ),
        },
        "protected_dev": {
            "file": ASQA_DEV_FILE,
            "physical_sha256": ASQA_DEV_SHA256,
            "count": EXPECTED_DEV,
            "ordered_id_sequence_sha256": ordered_id_sequence_sha256(
                dev_ids
            ),
            "content_inspected_for_partition": False,
        },
        "train_dev_disjoint": True,
        "partition": {
            "namespace": PARTITION_NAMESPACE,
            "hash_algorithm": "SHA-256",
            "encoding": "UTF-8",
            "canonical_sample_id": "exact_validated_source_decimal_string",
            "tie_rule": "ascending(lowercase_sha256, UTF-8_sample_id_bytes)",
            "forced_development_exposure_ids": [],
            "forced_development_exposure_sha256": ordered_id_sequence_sha256(
                []
            ),
            "development": {
                "count": DEVELOPMENT_SIZE,
                "ordered_id_sequence_sha256": ordered_id_sequence_sha256(
                    development_ids_ordered
                ),
                "manifest_id": development_manifest.manifest_id,
                "manifest_sha256": development_manifest.sha256,
            },
            "selection": {
                "count": SELECTION_SIZE,
                "ordered_id_sequence_sha256": ordered_id_sequence_sha256(
                    selection_ids_ordered
                ),
                "manifest_id": selection_manifest.manifest_id,
                "manifest_sha256": selection_manifest.sha256,
            },
        },
    }

    scientific_json = json.dumps(
        provenance_scientific_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    provenance_payload = {
        "artifact_format": "sprint3.asqa-partition-provenance.v1",
        "scientific_payload": provenance_scientific_payload,
        "scientific_sha256": sha256(
            scientific_json.encode("utf-8")
        ).hexdigest(),
        "metadata": {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "git_commit": git_commit,
        },
    }

    PROVENANCE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = PROVENANCE_OUTPUT.with_suffix(
        PROVENANCE_OUTPUT.suffix + ".tmp"
    )

    temporary_output.write_text(
        json.dumps(
            provenance_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    temporary_output.replace(PROVENANCE_OUTPUT)

    print("=== ASQA PARTITION MANIFEST BUILD ===")
    print("Source revision:", ASQA_REVISION)
    print("Train SHA256:", actual_sha)
    print("DEVELOPMENT:", development_manifest.actual_sample_size)
    print("SELECTION:", selection_manifest.actual_sample_size)
    print("Overlap:", len(development_ids & selection_ids))
    print()
    print("DEVELOPMENT manifest:", development_manifest.manifest_id)
    print("SELECTION manifest:", selection_manifest.manifest_id)
    print(
        "Partition provenance:",
        provenance_payload["scientific_sha256"],
    )
    print()
    print("ASQA partition manifest build: PASS")


if __name__ == "__main__":
    main()
