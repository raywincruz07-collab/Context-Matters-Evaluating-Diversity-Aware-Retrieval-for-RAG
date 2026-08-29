import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generation.prompts import render_pubmedqa_prompt
from generation.repeatability import read_repeatability_prompt_manifest
from scripts.materialize_generation_repeatability_manifest import (
    EXPECTED_DEVELOPMENT_QUERY_COUNT,
    build_repeatability_entries,
    materialize_repeatability_prompt_manifest,
    select_development_queries,
)


def _development_rows():
    query_ids = [f"query-{index:05d}" for index in range(EXPECTED_DEVELOPMENT_QUERY_COUNT)]
    queries = [
        {"_id": query_id, "text": f"Exact question text {query_id}?"}
        for query_id in query_ids
    ]
    qrels = [{"query-id": query_id, "corpus-id": "document"} for query_id in query_ids]
    return queries, qrels


def _expected_ids(qrels):
    ids = {row["query-id"] for row in qrels}
    return sorted(
        ids,
        key=lambda query_id: (
            hashlib.sha256(
                f"generation-repeatability-v1|{query_id}".encode("utf-8")
            ).digest(),
            query_id,
        ),
    )[:20]


def test_selection_is_exact_and_independent_of_source_row_order():
    queries, qrels = _development_rows()
    expected = _expected_ids(qrels)

    forward = select_development_queries(queries, qrels)
    reversed_rows = select_development_queries(reversed(queries), reversed(qrels))

    assert [item.query_id for item in forward] == expected
    assert forward == reversed_rows
    assert len(forward) == 20
    assert all(item.question == f"Exact question text {item.query_id}?" for item in forward)


def test_materialized_manifest_uses_existing_canonical_schema(tmp_path):
    queries, qrels = _development_rows()
    output = tmp_path / "repeatability_prompts_v1.json"

    materialize_repeatability_prompt_manifest(
        query_rows=queries,
        qrel_rows=qrels,
        output_path=output,
    )
    manifest = read_repeatability_prompt_manifest(output)
    entries = manifest["scientific_payload"]["entries"]

    assert manifest["scientific_payload"]["prompt_count"] == 20
    assert [entry["position"] for entry in entries] == list(range(20))
    for entry in entries:
        query_id = entry["sample_id"]
        assert entry["prompt_id"] == query_id
        assert entry["dataset"] == "hotpotqa"
        assert entry["evidence_role"] == "DEVELOPMENT"
        assert entry["prompt"] == render_pubmedqa_prompt(
            question=f"Exact question text {query_id}?"
        ).provenance_payload()
        assert entry["prompt"]["context_mode"] == "without_context"

    materialize_repeatability_prompt_manifest(
        query_rows=reversed(queries),
        qrel_rows=reversed(qrels),
        output_path=output,
    )
    incompatible = json.loads(output.read_text(encoding="utf-8"))
    incompatible["scientific_payload"]["entries"][0]["sample_id"] = "changed"
    output.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing immutable artifact differs"):
        materialize_repeatability_prompt_manifest(
            query_rows=queries,
            qrel_rows=qrels,
            output_path=output,
        )


def test_population_validation_rejects_wrong_count_missing_ids_and_empty_text():
    queries, qrels = _development_rows()
    with pytest.raises(ValueError, match="85,000 unique DEVELOPMENT"):
        select_development_queries(queries, qrels[:-1])

    missing_queries = queries[:-1]
    with pytest.raises(ValueError, match="missing from the pinned query source"):
        select_development_queries(missing_queries, qrels)

    empty_queries = list(queries)
    empty_queries[0] = {"_id": empty_queries[0]["_id"], "text": "   "}
    with pytest.raises(ValueError, match="question text"):
        select_development_queries(empty_queries, qrels)


def test_entry_builder_rejects_non_twenty_selection():
    with pytest.raises(ValueError, match="exactly 20"):
        build_repeatability_entries(())
