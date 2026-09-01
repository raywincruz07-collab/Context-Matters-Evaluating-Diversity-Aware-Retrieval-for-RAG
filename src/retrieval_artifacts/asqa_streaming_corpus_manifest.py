"""Streaming corpus manifests for the canonical ASQA DPR Wikipedia corpus.

This module is intentionally separate from :mod:`corpus_manifest`, whose
entry-oriented representation is appropriate for small corpora but would
materialize millions of Python objects for the DPR passage collection.

The ``logical_entry_stream_sha256`` is the body-only ordered corpus-content
identity.  The root ``scientific_sha256`` additionally binds exact compressed
source-archive provenance.  Shard layout and physical shard hashes remain
non-scientific storage metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, BinaryIO


ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION = (
    "sprint3.asqa-dpr-streaming-corpus-manifest.v1"
)
ASQA_DPR_CORPUS_MANIFEST_ARTIFACT_FORMAT = (
    "sprint3.asqa-dpr-streaming-corpus-manifest-artifact.v1"
)
ASQA_DPR_SHARD_FORMAT = "sprint3.asqa-dpr-body-sha256-shards.v1"
ASQA_DPR_LOGICAL_CORPUS_ID = "dpr-wikipedia-2018-12-20"
ASQA_DPR_CANONICAL_SURFACE = "exact DPR passage body"
ASQA_DPR_EXPECTED_PASSAGE_COUNT = 21_015_324
ASQA_DPR_EXPECTED_SOURCE_SHA256 = (
    "c39b020c855a2b5c25ffef3abe4a3b6f"
    "9b829ad7dbc14ec3d163d34d7c53ea8d"
)
ASQA_DPR_DEFAULT_SHARD_SIZE = 1_000_000

_DPR_TSV_HEADER = ["id", "text", "title"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_BYTES_RE = re.compile(rb"^[0-9a-f]{64}$")
_FILE_HASH_CHUNK_SIZE = 1024 * 1024
_SCIENTIFIC_PAYLOAD_KEYS = {
    "canonical_surface",
    "document_id_map_sha256",
    "first_passage_id",
    "last_passage_id",
    "logical_corpus_id",
    "logical_entry_stream_sha256",
    "passage_count",
    "schema_version",
    "source_compressed_sha256",
    "title_used_for_canonical_retrieval",
}
_ROOT_ARTIFACT_KEYS = {
    "artifact_format",
    "scientific_payload",
    "scientific_sha256",
    "storage",
}
_STORAGE_KEYS = {"format", "shard_size", "shards"}
_SHARD_KEYS = {
    "first_position",
    "last_position",
    "path",
    "physical_sha256",
    "row_count",
    "size_bytes",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_optional_passage_count(value: object) -> int | None:
    if value is None:
        return None
    return _require_positive_integer(value, "expected_passage_count")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_FILE_HASH_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class _HashingBinaryReader:
    """Minimal read proxy that hashes the exact compressed bytes gzip consumes."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        self._digest.update(block)
        return block

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _finalize_shard(
    handle: BinaryIO,
    path: Path,
    physical_sha256: Any,
    *,
    row_count: int,
    first_position: int,
    last_position: int,
    size_bytes: int,
) -> dict[str, Any]:
    try:
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    return {
        "filename": path.name,
        "first_position": first_position,
        "last_position": last_position,
        "physical_sha256": physical_sha256.hexdigest(),
        "row_count": row_count,
        "size_bytes": size_bytes,
    }


def _build_staged_shards(
    compressed_source: _HashingBinaryReader,
    staging_directory: Path,
    *,
    shard_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logical_entry_digest = hashlib.sha256()
    document_id_map_digest = hashlib.sha256()
    shard_metadata: list[dict[str, Any]] = []
    shard_handle: BinaryIO | None = None
    shard_path: Path | None = None
    shard_digest: Any = None
    shard_row_count = 0
    shard_first_position = 0
    shard_size_bytes = 0
    passage_count = 0
    first_passage_id: str | None = None
    last_passage_id: str | None = None

    try:
        with gzip.open(
            compressed_source,
            mode="rt",
            encoding="utf-8",
            newline="",
        ) as source_handle:
            reader = csv.reader(source_handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration:
                header = None
            except csv.Error as error:
                raise ValueError("malformed DPR TSV header") from error
            if header != _DPR_TSV_HEADER:
                raise ValueError(
                    f"DPR TSV header must be exactly {_DPR_TSV_HEADER!r}; "
                    f"found {header!r}"
                )

            try:
                for position, row in enumerate(reader):
                    if len(row) != 3:
                        raise ValueError(
                            "DPR TSV record at position "
                            f"{position} must contain exactly 3 fields; "
                            f"found {len(row)}"
                        )
                    passage_id, body, _title = row
                    expected_id = str(position + 1)
                    if passage_id != expected_id:
                        raise ValueError(
                            "DPR passage ID at position "
                            f"{position} must be {expected_id!r}; "
                            f"found {passage_id!r}"
                        )
                    if body == "":
                        raise ValueError(
                            f"DPR passage body at position {position} must be non-empty"
                        )

                    body_sha256 = hashlib.sha256(
                        body.encode("utf-8")
                    ).hexdigest()
                    logical_row = (
                        f"{position}\t{passage_id}\t{body_sha256}\n"
                    ).encode("utf-8")
                    id_map_row = f"{position}\t{passage_id}\n".encode("utf-8")

                    if shard_handle is None:
                        shard_index = len(shard_metadata)
                        shard_path = staging_directory / (
                            f"part-{shard_index:06d}.tsv"
                        )
                        shard_handle = shard_path.open("xb")
                        shard_digest = hashlib.sha256()
                        shard_row_count = 0
                        shard_first_position = position
                        shard_size_bytes = 0

                    shard_handle.write(logical_row)
                    shard_digest.update(logical_row)
                    shard_row_count += 1
                    shard_size_bytes += len(logical_row)
                    logical_entry_digest.update(logical_row)
                    document_id_map_digest.update(id_map_row)

                    if first_passage_id is None:
                        first_passage_id = passage_id
                    last_passage_id = passage_id
                    passage_count = position + 1

                    if shard_row_count == shard_size:
                        assert shard_path is not None
                        shard_metadata.append(
                            _finalize_shard(
                                shard_handle,
                                shard_path,
                                shard_digest,
                                row_count=shard_row_count,
                                first_position=shard_first_position,
                                last_position=position,
                                size_bytes=shard_size_bytes,
                            )
                        )
                        shard_handle = None
                        shard_path = None
                        shard_digest = None
            except csv.Error as error:
                raise ValueError(
                    f"malformed DPR TSV record near input line {reader.line_num}"
                ) from error
    except BaseException:
        if shard_handle is not None:
            shard_handle.close()
        raise

    if shard_handle is not None:
        assert shard_path is not None
        shard_metadata.append(
            _finalize_shard(
                shard_handle,
                shard_path,
                shard_digest,
                row_count=shard_row_count,
                first_position=shard_first_position,
                last_position=passage_count - 1,
                size_bytes=shard_size_bytes,
            )
        )

    if passage_count == 0:
        raise ValueError("DPR TSV must contain at least one passage")
    assert first_passage_id is not None
    assert last_passage_id is not None
    stream_summary = {
        "document_id_map_sha256": document_id_map_digest.hexdigest(),
        "first_passage_id": first_passage_id,
        "last_passage_id": last_passage_id,
        "logical_entry_stream_sha256": logical_entry_digest.hexdigest(),
        "passage_count": passage_count,
    }
    return stream_summary, shard_metadata


def _storage_directory_name(
    root_manifest_path: Path,
    *,
    shard_size: int,
    shard_metadata: list[dict[str, Any]],
) -> str:
    layout = {
        "format": ASQA_DPR_SHARD_FORMAT,
        "shard_size": shard_size,
        "shards": shard_metadata,
    }
    layout_sha256 = hashlib.sha256(_canonical_json_bytes(layout)).hexdigest()
    return f"{root_manifest_path.name}.shards-{layout_sha256}"


def _staged_shards_match_existing(
    final_directory: Path,
    shard_metadata: list[dict[str, Any]],
) -> bool:
    if not final_directory.is_dir() or final_directory.is_symlink():
        return False
    expected_names = {metadata["filename"] for metadata in shard_metadata}
    actual_names = {path.name for path in final_directory.iterdir()}
    if actual_names != expected_names:
        return False
    for metadata in shard_metadata:
        path = final_directory / metadata["filename"]
        if not path.is_file() or path.is_symlink():
            return False
        if path.stat().st_size != metadata["size_bytes"]:
            return False
        if _file_sha256(path) != metadata["physical_sha256"]:
            return False
    return True


def _publish_staged_shards(
    staging_directory: Path,
    final_directory: Path,
    shard_metadata: list[dict[str, Any]],
) -> None:
    if final_directory.exists():
        if _staged_shards_match_existing(final_directory, shard_metadata):
            return
        raise FileExistsError(
            f"existing ASQA shard directory differs: {final_directory}"
        )
    try:
        os.replace(staging_directory, final_directory)
    except OSError:
        if final_directory.exists() and _staged_shards_match_existing(
            final_directory, shard_metadata
        ):
            return
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = _canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
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
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_asqa_streaming_corpus_manifest(
    source_tsv_gz: str | os.PathLike[str],
    root_manifest_path: str | os.PathLike[str],
    *,
    shard_size: int = ASQA_DPR_DEFAULT_SHARD_SIZE,
    expected_passage_count: int | None = ASQA_DPR_EXPECTED_PASSAGE_COUNT,
    expected_source_sha256: str | None = ASQA_DPR_EXPECTED_SOURCE_SHA256,
) -> dict[str, Any]:
    """Build a streaming ASQA DPR corpus manifest and body-hash shards.

    ``root_manifest_path`` is required intentionally; importing or calling this
    module without an explicit destination never writes inside the repository.
    Pass ``expected_passage_count=None`` only when deliberately disabling the
    canonical-count guard.  Pass ``expected_source_sha256=None`` only when
    deliberately disabling the canonical archive guard for a synthetic
    development fixture.
    """
    shard_size = _require_positive_integer(shard_size, "shard_size")
    expected_passage_count = _require_optional_passage_count(
        expected_passage_count
    )
    if expected_source_sha256 is not None:
        expected_source_sha256 = _require_sha256(
            expected_source_sha256, "expected_source_sha256"
        )

    source_path = Path(source_tsv_gz)
    root_path = Path(root_manifest_path)
    if source_path.resolve() == root_path.resolve():
        raise ValueError("source_tsv_gz and root_manifest_path must differ")

    root_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as raw_source, tempfile.TemporaryDirectory(
        dir=root_path.parent,
        prefix=f".{root_path.name}.build-",
    ) as temporary_root:
        compressed_source = _HashingBinaryReader(raw_source)
        staging_directory = Path(temporary_root) / "shards"
        staging_directory.mkdir()
        stream_summary, shard_metadata = _build_staged_shards(
            compressed_source, staging_directory, shard_size=shard_size
        )
        source_compressed_sha256 = compressed_source.hexdigest()
        if (
            expected_source_sha256 is not None
            and source_compressed_sha256 != expected_source_sha256
        ):
            raise ValueError(
                "compressed DPR source SHA-256 mismatch: "
                f"expected {expected_source_sha256}, "
                f"found {source_compressed_sha256}"
            )
        passage_count = stream_summary["passage_count"]
        if (
            expected_passage_count is not None
            and passage_count != expected_passage_count
        ):
            raise ValueError(
                "DPR passage count mismatch: "
                f"expected {expected_passage_count}, found {passage_count}"
            )

        scientific_payload = {
            "canonical_surface": ASQA_DPR_CANONICAL_SURFACE,
            "document_id_map_sha256": stream_summary[
                "document_id_map_sha256"
            ],
            "first_passage_id": stream_summary["first_passage_id"],
            "last_passage_id": stream_summary["last_passage_id"],
            "logical_corpus_id": ASQA_DPR_LOGICAL_CORPUS_ID,
            "logical_entry_stream_sha256": stream_summary[
                "logical_entry_stream_sha256"
            ],
            "passage_count": passage_count,
            "schema_version": ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION,
            "source_compressed_sha256": source_compressed_sha256,
            "title_used_for_canonical_retrieval": False,
        }
        scientific_sha256 = hashlib.sha256(
            _canonical_json_bytes(scientific_payload)
        ).hexdigest()

        storage_directory_name = _storage_directory_name(
            root_path,
            shard_size=shard_size,
            shard_metadata=shard_metadata,
        )
        final_storage_directory = root_path.parent / storage_directory_name
        storage_shards = [
            {
                "first_position": metadata["first_position"],
                "last_position": metadata["last_position"],
                "path": PurePosixPath(
                    storage_directory_name, metadata["filename"]
                ).as_posix(),
                "physical_sha256": metadata["physical_sha256"],
                "row_count": metadata["row_count"],
                "size_bytes": metadata["size_bytes"],
            }
            for metadata in shard_metadata
        ]
        artifact = {
            "artifact_format": ASQA_DPR_CORPUS_MANIFEST_ARTIFACT_FORMAT,
            "scientific_payload": scientific_payload,
            "scientific_sha256": scientific_sha256,
            "storage": {
                "format": ASQA_DPR_SHARD_FORMAT,
                "shard_size": shard_size,
                "shards": storage_shards,
            },
        }

        _publish_staged_shards(
            staging_directory,
            final_storage_directory,
            shard_metadata,
        )
        _atomic_write_json(root_path, artifact)
    return artifact


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} fields mismatch; missing={missing}, unexpected={unexpected}"
        )


def _validate_scientific_payload(payload: Mapping[str, Any]) -> int:
    _require_exact_keys(
        payload, _SCIENTIFIC_PAYLOAD_KEYS, "scientific_payload"
    )
    expected_constants = {
        "canonical_surface": ASQA_DPR_CANONICAL_SURFACE,
        "logical_corpus_id": ASQA_DPR_LOGICAL_CORPUS_ID,
        "schema_version": ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION,
    }
    for name, expected in expected_constants.items():
        if payload.get(name) != expected:
            raise ValueError(
                f"scientific_payload {name} must equal {expected!r}"
            )
    if payload.get("title_used_for_canonical_retrieval") is not False:
        raise ValueError(
            "scientific_payload title_used_for_canonical_retrieval "
            "must be the boolean false"
        )
    passage_count = _require_positive_integer(
        payload.get("passage_count"), "scientific_payload passage_count"
    )
    if payload.get("first_passage_id") != "1":
        raise ValueError("scientific_payload first_passage_id must equal '1'")
    if payload.get("last_passage_id") != str(passage_count):
        raise ValueError(
            "scientific_payload last_passage_id does not match passage_count"
        )
    for name in (
        "document_id_map_sha256",
        "logical_entry_stream_sha256",
        "source_compressed_sha256",
    ):
        _require_sha256(payload.get(name), f"scientific_payload {name}")
    return passage_count


def _safe_shard_path(root_path: Path, stored_path: object) -> Path:
    if not isinstance(stored_path, str) or not stored_path:
        raise ValueError("storage shard path must be a non-empty string")
    if "\\" in stored_path:
        raise ValueError("storage shard path must use POSIX separators")
    relative = PurePosixPath(stored_path)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError("storage shard path must be a safe relative path")
    candidate = root_path.parent.joinpath(*relative.parts)
    resolved_parent = root_path.parent.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_parent)
    except ValueError as error:
        raise ValueError("storage shard path escapes manifest directory") from error
    return candidate


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate JSON object key: {name!r}")
            value[name] = item
        return value

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("ASQA corpus-manifest root must be a JSON object")
    return value


def verify_asqa_streaming_corpus_manifest(
    root_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Stream and verify a root manifest and all declared body-hash shards."""
    root_path = Path(root_manifest_path)
    artifact = _load_json_object(root_path)
    _require_exact_keys(artifact, _ROOT_ARTIFACT_KEYS, "root artifact")
    if (
        artifact.get("artifact_format")
        != ASQA_DPR_CORPUS_MANIFEST_ARTIFACT_FORMAT
    ):
        raise ValueError("unexpected ASQA corpus-manifest artifact format")

    payload = artifact.get("scientific_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("scientific_payload must be an object")
    passage_count = _validate_scientific_payload(payload)
    declared_scientific_sha256 = _require_sha256(
        artifact.get("scientific_sha256"), "scientific_sha256"
    )
    computed_scientific_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    if declared_scientific_sha256 != computed_scientific_sha256:
        raise ValueError("scientific_sha256 does not match scientific_payload")

    storage = artifact.get("storage")
    if not isinstance(storage, Mapping):
        raise ValueError("storage must be an object")
    _require_exact_keys(storage, _STORAGE_KEYS, "storage")
    if storage.get("format") != ASQA_DPR_SHARD_FORMAT:
        raise ValueError("unexpected ASQA shard storage format")
    shard_size = _require_positive_integer(
        storage.get("shard_size"), "storage shard_size"
    )
    shards = storage.get("shards")
    if not isinstance(shards, list):
        raise ValueError("storage shards must be a list")
    expected_shard_count = (passage_count + shard_size - 1) // shard_size
    if len(shards) != expected_shard_count:
        raise ValueError(
            "storage shard count does not match passage_count and shard_size"
        )

    logical_entry_digest = hashlib.sha256()
    document_id_map_digest = hashlib.sha256()
    seen_paths: set[Path] = set()
    global_position = 0
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError(f"storage shard {shard_index} must be an object")
        _require_exact_keys(shard, _SHARD_KEYS, f"storage shard {shard_index}")
        shard_path = _safe_shard_path(root_path, shard.get("path"))
        resolved_shard_path = shard_path.resolve()
        if resolved_shard_path in seen_paths:
            raise ValueError("storage shard paths must be unique")
        seen_paths.add(resolved_shard_path)

        remaining = passage_count - global_position
        expected_row_count = min(shard_size, remaining)
        declared_row_count = _require_positive_integer(
            shard.get("row_count"),
            f"storage shard {shard_index} row_count",
        )
        if declared_row_count != expected_row_count:
            raise ValueError(
                f"storage shard {shard_index} row_count is inconsistent"
            )
        expected_first_position = global_position
        expected_last_position = global_position + expected_row_count - 1
        for name, expected in (
            ("first_position", expected_first_position),
            ("last_position", expected_last_position),
        ):
            value = shard.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"storage shard {shard_index} {name} must be an integer"
                )
            if value != expected:
                raise ValueError(
                    f"storage shard {shard_index} {name} is inconsistent"
                )
        declared_size_bytes = _require_positive_integer(
            shard.get("size_bytes"),
            f"storage shard {shard_index} size_bytes",
        )
        declared_physical_sha256 = _require_sha256(
            shard.get("physical_sha256"),
            f"storage shard {shard_index} physical_sha256",
        )

        physical_digest = hashlib.sha256()
        actual_size_bytes = 0
        actual_row_count = 0
        with shard_path.open("rb") as handle:
            for line in handle:
                physical_digest.update(line)
                actual_size_bytes += len(line)
                if not line.endswith(b"\n"):
                    raise ValueError(
                        f"storage shard {shard_index} has a non-terminated row"
                    )
                fields = line[:-1].split(b"\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"storage shard {shard_index} row must have 3 fields"
                    )
                position_bytes, passage_id_bytes, body_sha256_bytes = fields
                expected_position_bytes = str(global_position).encode("ascii")
                expected_passage_id_bytes = str(global_position + 1).encode(
                    "ascii"
                )
                if position_bytes != expected_position_bytes:
                    raise ValueError(
                        "storage shard position is not contiguous in corpus order"
                    )
                if passage_id_bytes != expected_passage_id_bytes:
                    raise ValueError(
                        "storage shard passage ID is not the canonical sequential ID"
                    )
                if _SHA256_BYTES_RE.fullmatch(body_sha256_bytes) is None:
                    raise ValueError(
                        "storage shard body SHA-256 must be lowercase hexadecimal"
                    )
                logical_entry_digest.update(line)
                document_id_map_digest.update(
                    position_bytes + b"\t" + passage_id_bytes + b"\n"
                )
                global_position += 1
                actual_row_count += 1

        if actual_row_count != declared_row_count:
            raise ValueError(
                f"storage shard {shard_index} row_count does not match file"
            )
        if actual_size_bytes != declared_size_bytes:
            raise ValueError(
                f"storage shard {shard_index} size_bytes does not match file"
            )
        if physical_digest.hexdigest() != declared_physical_sha256:
            raise ValueError(
                f"storage shard {shard_index} physical SHA-256 mismatch"
            )

    if global_position != passage_count:
        raise ValueError("verified shard passage count does not match manifest")
    if (
        logical_entry_digest.hexdigest()
        != payload["logical_entry_stream_sha256"]
    ):
        raise ValueError(
            "logical entry stream SHA-256 does not match scientific_payload"
        )
    if (
        document_id_map_digest.hexdigest()
        != payload["document_id_map_sha256"]
    ):
        raise ValueError(
            "document ID map SHA-256 does not match scientific_payload"
        )
    return artifact


def verify_canonical_asqa_streaming_corpus_manifest(
    root_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify integrity and require the frozen canonical ASQA DPR corpus."""
    artifact = verify_asqa_streaming_corpus_manifest(root_manifest_path)
    payload = artifact["scientific_payload"]
    canonical_requirements = {
        "passage_count": ASQA_DPR_EXPECTED_PASSAGE_COUNT,
        "source_compressed_sha256": ASQA_DPR_EXPECTED_SOURCE_SHA256,
        "first_passage_id": "1",
        "last_passage_id": str(ASQA_DPR_EXPECTED_PASSAGE_COUNT),
        "logical_corpus_id": ASQA_DPR_LOGICAL_CORPUS_ID,
        "canonical_surface": ASQA_DPR_CANONICAL_SURFACE,
    }
    for name, expected in canonical_requirements.items():
        if payload[name] != expected:
            raise ValueError(
                "canonical ASQA DPR corpus mismatch: "
                f"{name} must equal {expected!r}; found {payload[name]!r}"
            )
    if payload["title_used_for_canonical_retrieval"] is not False:
        raise ValueError(
            "canonical ASQA DPR corpus mismatch: "
            "title_used_for_canonical_retrieval must be the boolean false"
        )
    return artifact


def read_and_verify_asqa_streaming_corpus_manifest(
    root_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Compatibility spelling for reading and fully verifying an artifact."""
    return verify_asqa_streaming_corpus_manifest(root_manifest_path)


__all__ = [
    "ASQA_DPR_CANONICAL_SURFACE",
    "ASQA_DPR_CORPUS_MANIFEST_ARTIFACT_FORMAT",
    "ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION",
    "ASQA_DPR_DEFAULT_SHARD_SIZE",
    "ASQA_DPR_EXPECTED_PASSAGE_COUNT",
    "ASQA_DPR_EXPECTED_SOURCE_SHA256",
    "ASQA_DPR_LOGICAL_CORPUS_ID",
    "ASQA_DPR_SHARD_FORMAT",
    "build_asqa_streaming_corpus_manifest",
    "read_and_verify_asqa_streaming_corpus_manifest",
    "verify_canonical_asqa_streaming_corpus_manifest",
    "verify_asqa_streaming_corpus_manifest",
]
