import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrieval_artifacts.asqa_streaming_corpus_manifest as asqa_manifest
from retrieval_artifacts.asqa_streaming_corpus_manifest import (
    ASQA_DPR_CANONICAL_SURFACE,
    ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION,
    ASQA_DPR_LOGICAL_CORPUS_ID,
    build_asqa_streaming_corpus_manifest,
    verify_canonical_asqa_streaming_corpus_manifest,
    verify_asqa_streaming_corpus_manifest,
)


PASSAGES = (
    ("1", "First passage.", "First title"),
    ("2", "Body with\ta quoted tab.", "Second title"),
    ("3", "Body with a quoted\nnewline.", "Third title"),
    ("4", "Unicode café α.", "Fourth title"),
    ("5", "Final passage.", "Fifth title"),
)


def write_dpr_fixture(
    path: Path,
    rows=PASSAGES,
    *,
    header=("id", "text", "title"),
) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle,
            mode="wb",
            filename="",
            mtime=0,
        ) as compressed_handle:
            with io.TextIOWrapper(
                compressed_handle,
                encoding="utf-8",
                newline="",
            ) as text_handle:
                writer = csv.writer(
                    text_handle,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writerow(header)
                writer.writerows(rows)


def build_fixture(tmp_path: Path, name: str, *, shard_size: int = 2):
    source = tmp_path / "psgs_w100.tsv.gz"
    if not source.exists():
        write_dpr_fixture(source)
    root = tmp_path / name
    artifact = build_asqa_streaming_corpus_manifest(
        source,
        root,
        shard_size=shard_size,
        expected_passage_count=5,
        expected_source_sha256=None,
    )
    return source, root, artifact


def shard_paths(root: Path, artifact: dict) -> list[Path]:
    return [root.parent / item["path"] for item in artifact["storage"]["shards"]]


def test_successful_five_passage_build_with_two_row_shards(tmp_path):
    source, root, artifact = build_fixture(tmp_path, "manifest.json")

    assert root.exists()
    assert verify_asqa_streaming_corpus_manifest(root) == artifact
    assert [
        shard["row_count"] for shard in artifact["storage"]["shards"]
    ] == [2, 2, 1]
    assert [
        (shard["first_position"], shard["last_position"])
        for shard in artifact["storage"]["shards"]
    ] == [(0, 1), (2, 3), (4, 4)]

    logical_rows = b"".join(path.read_bytes() for path in shard_paths(root, artifact))
    decoded_rows = [line.split("\t") for line in logical_rows.decode().splitlines()]
    expected_body_hashes = [
        hashlib.sha256(body.encode("utf-8")).hexdigest()
        for _passage_id, body, _title in PASSAGES
    ]
    assert [fields[0] for fields in decoded_rows] == ["0", "1", "2", "3", "4"]
    assert [fields[1] for fields in decoded_rows] == ["1", "2", "3", "4", "5"]
    assert [fields[2] for fields in decoded_rows] == expected_body_hashes
    assert all(body.encode("utf-8") not in logical_rows for _, body, _ in PASSAGES)
    assert all(title.encode("utf-8") not in logical_rows for _, _, title in PASSAGES)

    payload = artifact["scientific_payload"]
    expected_logical_rows = b"".join(
        f"{position}\t{passage_id}\t{body_hash}\n".encode("utf-8")
        for position, ((passage_id, _body, _title), body_hash) in enumerate(
            zip(PASSAGES, expected_body_hashes, strict=True)
        )
    )
    expected_id_map = b"".join(
        f"{position}\t{passage_id}\n".encode("utf-8")
        for position, (passage_id, _body, _title) in enumerate(PASSAGES)
    )
    assert payload == {
        "canonical_surface": ASQA_DPR_CANONICAL_SURFACE,
        "document_id_map_sha256": hashlib.sha256(expected_id_map).hexdigest(),
        "first_passage_id": "1",
        "last_passage_id": "5",
        "logical_corpus_id": ASQA_DPR_LOGICAL_CORPUS_ID,
        "logical_entry_stream_sha256": hashlib.sha256(
            expected_logical_rows
        ).hexdigest(),
        "passage_count": 5,
        "schema_version": ASQA_DPR_CORPUS_MANIFEST_SCHEMA_VERSION,
        "source_compressed_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "title_used_for_canonical_retrieval": False,
    }
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert artifact["scientific_sha256"] == hashlib.sha256(
        canonical_payload
    ).hexdigest()


def test_scientific_identity_is_independent_of_shard_size(tmp_path):
    _source, root_two, two = build_fixture(tmp_path, "two.json", shard_size=2)
    _source, root_three, three = build_fixture(tmp_path, "three.json", shard_size=3)

    assert two["scientific_payload"] == three["scientific_payload"]
    assert two["scientific_sha256"] == three["scientific_sha256"]
    assert two["storage"] != three["storage"]
    assert len(shard_paths(root_two, two)) == 3
    assert len(shard_paths(root_three, three)) == 2
    assert verify_asqa_streaming_corpus_manifest(root_two) == two
    assert verify_asqa_streaming_corpus_manifest(root_three) == three


def test_title_changes_preserve_body_identity_but_change_root_science(tmp_path):
    first_source = tmp_path / "first.tsv.gz"
    second_source = tmp_path / "second.tsv.gz"
    write_dpr_fixture(first_source)
    changed_titles = tuple(
        (passage_id, body, f"Changed title {position}")
        for position, (passage_id, body, _title) in enumerate(PASSAGES)
    )
    write_dpr_fixture(second_source, changed_titles)

    first = build_asqa_streaming_corpus_manifest(
        first_source,
        tmp_path / "first.json",
        shard_size=2,
        expected_passage_count=5,
        expected_source_sha256=None,
    )
    second = build_asqa_streaming_corpus_manifest(
        second_source,
        tmp_path / "second.json",
        shard_size=2,
        expected_passage_count=5,
        expected_source_sha256=None,
    )

    first_payload = first["scientific_payload"]
    second_payload = second["scientific_payload"]
    # The canonical retrieval-content identity is body-only and represented by
    # the logical entry stream. The root scientific manifest additionally binds
    # the exact compressed source archive, so archive/title changes intentionally
    # change the root scientific SHA even when body retrieval content is unchanged.
    assert (
        first_payload["logical_entry_stream_sha256"]
        == second_payload["logical_entry_stream_sha256"]
    )
    assert (
        first_payload["document_id_map_sha256"]
        == second_payload["document_id_map_sha256"]
    )
    assert (
        first_payload["source_compressed_sha256"]
        != second_payload["source_compressed_sha256"]
    )
    assert first["scientific_sha256"] != second["scientific_sha256"]


def test_wrong_header_is_rejected(tmp_path):
    source = tmp_path / "wrong-header.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source, header=("id", "title", "text"))

    with pytest.raises(ValueError, match="header must be exactly"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=5,
            expected_source_sha256=None,
        )
    assert not root.exists()


def test_record_without_exactly_three_columns_is_rejected(tmp_path):
    source = tmp_path / "four-columns.tsv.gz"
    root = tmp_path / "manifest.json"
    rows = (("1", "body", "title", "unexpected"),)
    write_dpr_fixture(source, rows)

    with pytest.raises(ValueError, match="exactly 3 fields"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=1,
            expected_source_sha256=None,
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "rows",
    (
        (("1", "body one", "title"), ("3", "body three", "title")),
        (("2", "body two", "title"), ("1", "body one", "title")),
    ),
)
def test_id_gap_or_reorder_is_rejected(tmp_path, rows):
    source = tmp_path / "bad-ids.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source, rows)

    with pytest.raises(ValueError, match="passage ID at position"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=2,
            expected_source_sha256=None,
        )
    assert not root.exists()


def test_empty_body_is_rejected(tmp_path):
    source = tmp_path / "empty-body.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source, (("1", "", "title"),))

    with pytest.raises(ValueError, match="body at position 0 must be non-empty"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=1,
            expected_source_sha256=None,
        )
    assert not root.exists()


def test_expected_count_mismatch_is_rejected(tmp_path):
    source = tmp_path / "count.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source)

    with pytest.raises(ValueError, match="expected 6, found 5"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=6,
            expected_source_sha256=None,
        )
    assert not root.exists()


def test_matching_expected_compressed_source_sha_succeeds(tmp_path):
    source = tmp_path / "matching-source-sha.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    artifact = build_asqa_streaming_corpus_manifest(
        source,
        root,
        shard_size=2,
        expected_passage_count=5,
        expected_source_sha256=source_sha256,
    )

    assert (
        artifact["scientific_payload"]["source_compressed_sha256"]
        == source_sha256
    )


def test_default_source_sha_guard_rejects_synthetic_fixture(tmp_path):
    source = tmp_path / "noncanonical-source.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source)

    with pytest.raises(ValueError, match="compressed DPR source SHA-256 mismatch"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=5,
        )
    assert not root.exists()


def test_expected_compressed_source_sha_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source-sha.tsv.gz"
    root = tmp_path / "manifest.json"
    write_dpr_fixture(source)

    with pytest.raises(ValueError, match="compressed DPR source SHA-256 mismatch"):
        build_asqa_streaming_corpus_manifest(
            source,
            root,
            shard_size=2,
            expected_passage_count=5,
            expected_source_sha256="0" * 64,
        )
    assert not root.exists()


def test_development_manifest_passes_generic_but_fails_canonical_verifier(
    tmp_path,
):
    _source, root, artifact = build_fixture(tmp_path, "manifest.json")

    assert verify_asqa_streaming_corpus_manifest(root) == artifact
    with pytest.raises(ValueError, match="canonical ASQA DPR corpus mismatch"):
        verify_canonical_asqa_streaming_corpus_manifest(root)


def test_canonical_verifier_success_with_monkeypatched_fixture_identity(
    tmp_path, monkeypatch
):
    source, root, artifact = build_fixture(tmp_path, "manifest.json")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(asqa_manifest, "ASQA_DPR_EXPECTED_PASSAGE_COUNT", 5)
    monkeypatch.setattr(
        asqa_manifest,
        "ASQA_DPR_EXPECTED_SOURCE_SHA256",
        source_sha256,
    )

    assert verify_canonical_asqa_streaming_corpus_manifest(root) == artifact


def test_tampered_shard_is_rejected_by_verifier(tmp_path):
    _source, root, artifact = build_fixture(tmp_path, "manifest.json")
    first_shard = shard_paths(root, artifact)[0]
    contents = bytearray(first_shard.read_bytes())
    first_row_last_hex = contents.index(b"\n") - 1
    contents[first_row_last_hex] = (
        ord("0") if contents[first_row_last_hex] != ord("0") else ord("1")
    )
    first_shard.write_bytes(contents)

    with pytest.raises(ValueError, match="physical SHA-256 mismatch"):
        verify_asqa_streaming_corpus_manifest(root)


def test_verifier_requires_literal_false_for_title_usage(tmp_path):
    _source, root, artifact = build_fixture(tmp_path, "manifest.json")
    artifact["scientific_payload"]["title_used_for_canonical_retrieval"] = 0
    canonical_payload = json.dumps(
        artifact["scientific_payload"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    artifact["scientific_sha256"] = hashlib.sha256(canonical_payload).hexdigest()
    root.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="must be the boolean false"):
        verify_asqa_streaming_corpus_manifest(root)
