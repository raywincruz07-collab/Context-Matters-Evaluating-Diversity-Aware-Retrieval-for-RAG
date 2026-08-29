#!/usr/bin/env python3
"""Materialize the frozen 20-prompt generation repeatability manifest.

This pre-execution utility performs dataset reads and deterministic local
artifact construction only. It makes no model, Maki, or other generation API
calls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
for value in (REPOSITORY_ROOT, SRC_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from generation.prompts import render_pubmedqa_prompt
from generation.repeatability import write_repeatability_prompt_manifest


HOTPOTQA_QUERY_SOURCE = "BeIR/hotpotqa"
HOTPOTQA_QUERY_CONFIG = "queries"
HOTPOTQA_QUERY_REVISION = "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"
HOTPOTQA_QUERY_SPLIT = "queries"
HOTPOTQA_QRELS_SOURCE = "BeIR/hotpotqa-qrels"
HOTPOTQA_QRELS_CONFIG = "default"
HOTPOTQA_QRELS_REVISION = "b15429e9244c8ec966985d7778427c3b1543b314"
HOTPOTQA_QRELS_SPLIT = "train"
EXPECTED_DEVELOPMENT_QUERY_COUNT = 85_000
REPEATABILITY_PROMPT_COUNT = 20
SELECTION_NAMESPACE = "generation-repeatability-v1"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "artifacts/generation_repeatability/repeatability_prompts_v1.json"
)


@dataclass(frozen=True)
class SelectedDevelopmentQuery:
    query_id: str
    question: str


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _selection_key(query_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        f"{SELECTION_NAMESPACE}|{query_id}".encode("utf-8")
    ).digest()
    return digest, query_id


def select_development_queries(
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
) -> tuple[SelectedDevelopmentQuery, ...]:
    """Validate the BEIR-train population and select its first frozen 20 IDs."""
    query_text_by_id: dict[str, str] = {}
    for row in query_rows:
        if "_id" not in row or "text" not in row:
            raise ValueError("pinned query row must contain _id and text")
        query_id = _require_nonempty_string(row["_id"], "query _id")
        question = _require_nonempty_string(
            row["text"], f"question text for query ID {query_id!r}"
        )
        if query_id in query_text_by_id:
            raise ValueError(f"duplicate pinned query ID: {query_id!r}")
        query_text_by_id[query_id] = question

    development_ids: set[str] = set()
    for row in qrel_rows:
        if "query-id" not in row:
            raise ValueError("pinned train qrel row must contain query-id")
        development_ids.add(
            _require_nonempty_string(row["query-id"], "train qrel query-id")
        )

    if len(development_ids) != EXPECTED_DEVELOPMENT_QUERY_COUNT:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_DEVELOPMENT_QUERY_COUNT:,} unique DEVELOPMENT query IDs, "
            f"found {len(development_ids):,}"
        )

    missing_ids = development_ids.difference(query_text_by_id)
    if missing_ids:
        preview = sorted(missing_ids)[:5]
        raise ValueError(
            f"{len(missing_ids)} DEVELOPMENT query IDs are missing from the "
            f"pinned query source; first IDs: {preview!r}"
        )

    ordered_ids = sorted(development_ids, key=_selection_key)
    selected_ids = ordered_ids[:REPEATABILITY_PROMPT_COUNT]
    if len(selected_ids) != REPEATABILITY_PROMPT_COUNT:
        raise RuntimeError("failed to select exactly 20 repeatability query IDs")
    return tuple(
        SelectedDevelopmentQuery(
            query_id=query_id,
            question=query_text_by_id[query_id],
        )
        for query_id in selected_ids
    )


def build_repeatability_entries(
    selected: tuple[SelectedDevelopmentQuery, ...],
) -> list[dict[str, Any]]:
    if len(selected) != REPEATABILITY_PROMPT_COUNT:
        raise ValueError("repeatability entry construction requires exactly 20 queries")
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in selected:
        query_id = _require_nonempty_string(item.query_id, "selected query ID")
        if query_id in seen_ids:
            raise ValueError(f"duplicate selected query ID: {query_id!r}")
        seen_ids.add(query_id)
        question = _require_nonempty_string(
            item.question, f"selected question text for query ID {query_id!r}"
        )
        prompt = render_pubmedqa_prompt(question=question)
        if prompt.context_mode != "without_context":
            raise RuntimeError("repeatability prompt must be WITHOUT_CONTEXT")
        entries.append(
            {
                "prompt_id": query_id,
                "dataset": "hotpotqa",
                "evidence_role": "DEVELOPMENT",
                "sample_id": query_id,
                "prompt": prompt.provenance_payload(),
            }
        )
    return entries


def materialize_repeatability_prompt_manifest(
    *,
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    selected = select_development_queries(query_rows, qrel_rows)
    entries = build_repeatability_entries(selected)
    write_repeatability_prompt_manifest(Path(output_path), entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    cache_dir = None if args.cache_dir is None else str(args.cache_dir)
    query_rows = load_dataset(
        HOTPOTQA_QUERY_SOURCE,
        HOTPOTQA_QUERY_CONFIG,
        revision=HOTPOTQA_QUERY_REVISION,
        split=HOTPOTQA_QUERY_SPLIT,
        cache_dir=cache_dir,
    )
    qrel_rows = load_dataset(
        HOTPOTQA_QRELS_SOURCE,
        HOTPOTQA_QRELS_CONFIG,
        revision=HOTPOTQA_QRELS_REVISION,
        split=HOTPOTQA_QRELS_SPLIT,
        cache_dir=cache_dir,
    )
    materialize_repeatability_prompt_manifest(
        query_rows=query_rows,
        qrel_rows=qrel_rows,
        output_path=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
