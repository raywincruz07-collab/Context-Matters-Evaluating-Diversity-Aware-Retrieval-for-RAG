from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


EXPECTED_HEADER = ["id", "text", "title"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    manifest_path = Path(args.manifest)
    output = Path(args.output)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    scientific = manifest["scientific_payload"]

    expected_count = scientific["passage_count"]
    expected_logical_sha = scientific["logical_entry_stream_sha256"]
    expected_id_map_sha = scientific["document_id_map_sha256"]

    if scientific["canonical_surface"] != "exact DPR passage body":
        raise ValueError("Manifest is not body-only")

    if scientific["title_used_for_canonical_retrieval"] is not False:
        raise ValueError("Manifest unexpectedly allows titles")

    output.parent.mkdir(parents=True, exist_ok=True)

    logical_digest = hashlib.sha256()
    id_map_digest = hashlib.sha256()
    output_digest = hashlib.sha256()

    count = 0

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

        try:
            with gzip.open(
                source,
                mode="rt",
                encoding="utf-8",
                newline="",
            ) as src:
                reader = csv.reader(
                    src,
                    delimiter="\t",
                    strict=True,
                )

                header = next(reader)

                if header != EXPECTED_HEADER:
                    raise ValueError(
                        f"Unexpected DPR header: {header!r}"
                    )

                for position, row in enumerate(reader):
                    if len(row) != 3:
                        raise ValueError(
                            f"Row {position} has {len(row)} fields"
                        )

                    passage_id, body, _title = row

                    expected_passage_id = str(position + 1)

                    if passage_id != expected_passage_id:
                        raise ValueError(
                            f"Passage-ID mismatch at position {position}: "
                            f"{passage_id!r} != {expected_passage_id!r}"
                        )

                    if body == "":
                        raise ValueError(
                            f"Empty body at position {position}"
                        )

                    # ColBERT's TSV loader is line-based and tab-separated.
                    # These characters would make the exact body ambiguous.
                    if "\t" in body or "\n" in body or "\r" in body:
                        raise ValueError(
                            f"Body at position {position} contains "
                            "tab/newline characters"
                        )

                    body_sha = hashlib.sha256(
                        body.encode("utf-8")
                    ).hexdigest()

                    logical_row = (
                        f"{position}\t"
                        f"{passage_id}\t"
                        f"{body_sha}\n"
                    ).encode("utf-8")

                    id_map_row = (
                        f"{position}\t{passage_id}\n"
                    ).encode("utf-8")

                    colbert_row = (
                        f"{position}\t{body}\n"
                    ).encode("utf-8")

                    logical_digest.update(logical_row)
                    id_map_digest.update(id_map_row)
                    output_digest.update(colbert_row)

                    temp.write(colbert_row)

                    count = position + 1

                    if count % 1_000_000 == 0:
                        print(
                            f"processed {count:,} passages",
                            flush=True,
                        )

            temp.flush()
            os.fsync(temp.fileno())

        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    if count != expected_count:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Passage count mismatch: "
            f"{count} != {expected_count}"
        )

    logical_sha = logical_digest.hexdigest()
    id_map_sha = id_map_digest.hexdigest()

    if logical_sha != expected_logical_sha:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            "Logical corpus SHA256 mismatch"
        )

    if id_map_sha != expected_id_map_sha:
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            "Document-ID-map SHA256 mismatch"
        )

    os.replace(temp_path, output)

    print()
    print("=== COLBERT COLLECTION ===")
    print("passage_count:", count)
    print("first ColBERT PID: 0")
    print("first DPR passage ID: 1")
    print("last ColBERT PID:", count - 1)
    print("last DPR passage ID:", count)
    print("logical_entry_stream_sha256:", logical_sha)
    print("document_id_map_sha256:", id_map_sha)
    print("colbert_collection_sha256:", output_digest.hexdigest())
    print("output_bytes:", output.stat().st_size)
    print()
    print("ASQA COLBERT COLLECTION BUILD: PASS")


if __name__ == "__main__":
    main()
