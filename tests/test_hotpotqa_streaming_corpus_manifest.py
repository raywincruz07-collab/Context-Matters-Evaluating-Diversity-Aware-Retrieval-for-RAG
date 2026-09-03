import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrieval_artifacts.hotpotqa_streaming_corpus_manifest as hotpot_manifest
from retrieval_artifacts.hotpotqa_streaming_corpus_manifest import (
    HOTPOTQA_CONFIG,
    HOTPOTQA_EXPECTED_DOCUMENT_COUNT,
    HOTPOTQA_RETRIEVAL_SERIALIZATION,
    HOTPOTQA_REVISION,
    HOTPOTQA_SOURCE,
    HOTPOTQA_SPLIT,
    build_hotpotqa_streaming_corpus_manifest,
    canonical_retrieval_content,
    verify_hotpotqa_streaming_corpus_manifest,
)


ROWS = (
    {
        "_id": "doc-A",
        "title": "First title",
        "text": "First body.",
    },
    {
        "_id": "17",
        "title": "  Title with outer spaces  ",
        "text": "Body with\ta tab.",
    },
    {
        "_id": "doc-unicode",
        "title": "Unicode café α",
        "text": "Body with\nembedded newline.",
    },
    {
        "_id": "empty-text",
        "title": "Title only",
        "text": "",
    },
    {
        "_id": "empty-both",
        "title": "   ",
        "text": "",
    },
)


def build_fixture(tmp_path: Path, name: str, *, shard_size: int = 2):
    root = tmp_path / name
    artifact = build_hotpotqa_streaming_corpus_manifest(
        ROWS,
        root,
        shard_size=shard_size,
        expected_document_count=len(ROWS),
    )
    return root, artifact


def shard_paths(root: Path, artifact: dict) -> list[Path]:
    return [
        root.parent / item["path"]
        for item in artifact["storage"]["manifest_shards"]
    ]


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def test_retrieval_content_formula_is_exact():
    assert canonical_retrieval_content("Title", "Body") == "Title Body"
    assert canonical_retrieval_content("", " Body ") == "Body"
    assert canonical_retrieval_content(" Title ", "") == "Title"
    assert canonical_retrieval_content("   ", "") == ""
    assert (
        canonical_retrieval_content(" A  ", "  B ")
        == "A     B"
    )
    assert (
        canonical_retrieval_content("Title\nX", "Body\tY")
        == "Title\nX Body\tY"
    )


def test_successful_build_and_verify(tmp_path):
    root, artifact = build_fixture(tmp_path, "manifest.json")

    assert root.exists()
    assert verify_hotpotqa_streaming_corpus_manifest(root) == artifact

    payload = artifact["scientific_payload"]
    assert payload["source"] == HOTPOTQA_SOURCE
    assert payload["revision"] == HOTPOTQA_REVISION
    assert payload["config"] == HOTPOTQA_CONFIG
    assert payload["split"] == HOTPOTQA_SPLIT
    assert payload["retrieval_serialization"] == HOTPOTQA_RETRIEVAL_SERIALIZATION
    assert payload["document_count"] == len(ROWS)
    assert payload["first_source_document_id"] == "doc-A"
    assert payload["last_source_document_id"] == "empty-both"

    assert [
        shard["row_count"]
        for shard in artifact["storage"]["manifest_shards"]
    ] == [2, 2, 1]

    logical_lines = []
    for path in shard_paths(root, artifact):
        logical_lines.extend(path.read_text(encoding="utf-8").splitlines())

    assert len(logical_lines) == len(ROWS)

    for position, (line, source_row) in enumerate(
        zip(logical_lines, ROWS, strict=True)
    ):
        stored = json.loads(line)
        retrieval_content = canonical_retrieval_content(
            source_row["title"],
            source_row["text"],
        )
        assert stored == {
            "position": position,
            "source_document_id": source_row["_id"],
            "text_sha256": hashlib.sha256(
                source_row["text"].encode("utf-8")
            ).hexdigest(),
            "title_sha256": hashlib.sha256(
                source_row["title"].encode("utf-8")
            ).hexdigest(),
            "retrieval_content_sha256": hashlib.sha256(
                retrieval_content.encode("utf-8")
            ).hexdigest(),
        }

        if source_row["title"]:
            assert source_row["title"] not in line
        if source_row["text"]:
            assert source_row["text"] not in line

    expected_logical_stream = b"".join(
        canonical_json_bytes(json.loads(line)) + b"\n"
        for line in logical_lines
    )
    expected_id_map = b"".join(
        canonical_json_bytes(
            {
                "position": position,
                "source_document_id": row["_id"],
            }
        )
        + b"\n"
        for position, row in enumerate(ROWS)
    )

    assert payload["logical_entry_stream_sha256"] == hashlib.sha256(
        expected_logical_stream
    ).hexdigest()

    assert payload["document_id_map_sha256"] == hashlib.sha256(
        expected_id_map
    ).hexdigest()


def test_empty_retrieval_content_is_retained(tmp_path):
    rows = (
        {"_id": "x", "title": "   ", "text": ""},
    )
    root = tmp_path / "manifest.json"

    artifact = build_hotpotqa_streaming_corpus_manifest(
        rows,
        root,
        shard_size=1,
        expected_document_count=1,
    )

    assert artifact["scientific_payload"]["document_count"] == 1
    assert verify_hotpotqa_streaming_corpus_manifest(root) == artifact


def test_scientific_identity_is_independent_of_shard_size(tmp_path):
    root_two, two = build_fixture(tmp_path, "two.json", shard_size=2)
    root_three, three = build_fixture(tmp_path, "three.json", shard_size=3)

    assert two["scientific_payload"] == three["scientific_payload"]
    assert two["scientific_sha256"] == three["scientific_sha256"]
    assert two["storage"] != three["storage"]

    assert verify_hotpotqa_streaming_corpus_manifest(root_two) == two
    assert verify_hotpotqa_streaming_corpus_manifest(root_three) == three


def test_duplicate_source_document_id_is_rejected(tmp_path):
    rows = (
        {"_id": "same", "title": "A", "text": "one"},
        {"_id": "same", "title": "B", "text": "two"},
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_hotpotqa_streaming_corpus_manifest(
            rows,
            tmp_path / "manifest.json",
            shard_size=2,
            expected_document_count=2,
        )


@pytest.mark.parametrize("field", ("_id", "title", "text"))
def test_non_string_source_field_is_rejected(tmp_path, field):
    row = {"_id": "1", "title": "Title", "text": "Text"}
    row[field] = None

    with pytest.raises(TypeError, match=field):
        build_hotpotqa_streaming_corpus_manifest(
            (row,),
            tmp_path / "manifest.json",
            shard_size=1,
            expected_document_count=1,
        )


def test_expected_count_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected 6, found 5"):
        build_hotpotqa_streaming_corpus_manifest(
            ROWS,
            tmp_path / "manifest.json",
            shard_size=2,
            expected_document_count=6,
        )


def test_tampered_manifest_shard_is_rejected(tmp_path):
    root, artifact = build_fixture(tmp_path, "manifest.json")
    first_shard = shard_paths(root, artifact)[0]

    lines = first_shard.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["title_sha256"] = "0" * 64
    lines[0] = canonical_json_bytes(row).decode("utf-8")

    first_shard.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        verify_hotpotqa_streaming_corpus_manifest(root)


def test_default_expected_count_is_full_hotpotqa():
    assert HOTPOTQA_EXPECTED_DOCUMENT_COUNT == 5_233_329
