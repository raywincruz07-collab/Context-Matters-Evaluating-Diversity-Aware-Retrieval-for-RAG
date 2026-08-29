from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.metric_registry import DatasetId
from generation._io import stable_json_sha256
from generation.artifacts import (
    GenerationArtifactConflictError,
    GenerationStatus,
    build_generation_artifact,
    generation_request_payload,
    read_generation_artifact,
    write_generation_artifact,
)
from generation.cli_support import load_model_bindings, runtime_provenance
from generation.maki import (
    CanonicalMakiAdapter,
    MakiConfig,
    MakiInfrastructureError,
)
from generation.prompts import (
    PROMPT_BUNDLE_SHA256,
    PUBMEDQA_WITH_CONTEXT_TEMPLATE,
    PUBMEDQA_WITHOUT_CONTEXT_TEMPLATE,
    SYSTEM_INSTRUCTION,
    render_context_block,
    render_pubmedqa_prompt,
)
from generation.pubmedqa import classify_pubmedqa_response
from generation.runner import _request_for_query
from generation.selected_context import (
    SelectedContextConflictError,
    build_relevance_selected_context,
    read_selected_context,
    write_selected_context,
)
from retrieval_artifacts import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEntry,
    CorpusProvenance,
    CorpusRecord,
    DatasetProvenance,
    RetrieverProvenance,
    document_content_sha256,
    write_candidate_artifact,
)
from run_registry import (
    candidate_set_artifact_payload,
    candidate_set_scientific_payload,
    write_candidate_set_artifact,
)
from generation.selected_context import materialize_relevance_selected_contexts


SHA_A = "a" * 64
SHA_B = "b" * 64
GIT_SHA = "1" * 40


def _bodies():
    return tuple(f"Canonical body {index}." for index in range(1, 6))


def _candidate_fixture():
    records = tuple(
        CorpusRecord(
            document_id=index,
            source_document_id=f"source-{index}",
            title=None,
            text=f"Canonical body {index + 1}.",
            retrieval_content=f"Canonical body {index + 1}.",
            corpus_position=index,
        )
        for index in range(20)
    )
    dataset = DatasetProvenance(
        dataset_id=DatasetId.PUBMEDQA,
        source="qiaojin/PubMedQA",
        config="pqa_labeled",
        revision="revision",
        split="train",
        sample_manifest_id=f"sample-manifest:sha256:{SHA_A}",
        sample_manifest_sha256=SHA_A,
        sampling_seed=None,
        sampling_algorithm=None,
    )
    corpus = CorpusProvenance(
        corpus_id=f"corpus-manifest:sha256:{SHA_B}",
        source="qiaojin/PubMedQA",
        revision="revision",
        document_count=20,
        manifest_sha256=SHA_B,
        document_id_map_sha256=SHA_A,
        preprocessing_version="fixture.v1",
    )
    retriever = RetrieverProvenance(
        retriever_name="bm25",
        implementation="fixture",
        library_name="fixture",
        library_version="1",
        model_id=None,
        model_revision=None,
        tokenizer_id=None,
        tokenizer_revision=None,
        query_preprocessing="exact",
        document_preprocessing="exact",
        normalization="none",
        score_semantics="higher",
        index_type="in-memory",
        index_config="fixture",
        index_fingerprint_sha256=SHA_A,
        index_artifact_sha256=None,
    )
    candidate = CandidateArtifact(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        dataset=dataset,
        corpus=corpus,
        sample_id=7,
        query_text="Exact question?",
        retriever=retriever,
        requested_top_n=20,
        candidates=tuple(
            CandidateEntry(
                rank=index + 1,
                document_id=index,
                source_document_id=f"source-{index}",
                corpus_position=index,
                native_score=float(20 - index),
                document_content_sha256=document_content_sha256(records[index].text),
            )
            for index in range(20)
        ),
        producing_git_commit=GIT_SHA,
        worktree_clean=True,
        environment_fingerprint_sha256=SHA_B,
    )
    scientific = candidate_set_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=dataset.sample_manifest_id,
        retriever="bm25",
        expected_query_count=1,
        entries=[{"sample_id": 7, "candidate_artifact_id": candidate.artifact_id}],
    )
    return records, candidate, candidate_set_artifact_payload(scientific)


def _maki_config(**overrides):
    values = {
        "base_url": "https://maki.example/v1",
        "logical_model_id": "llama-3.3-70b",
        "physical_model_id": "provider/physical-llama",
        "model_revision": None,
        "model_revision_kind": "NOT_PROVIDED_BY_PROVIDER",
        "direct_mode_status": "SUPPORTED_AND_ENABLED",
        "direct_mode_control": {"chat_template_kwargs": {"enable_thinking": False}},
        "timeout_seconds": 5,
    }
    values.update(overrides)
    return MakiConfig(**values)


def test_prompt_assets_render_exact_bytes_and_hash_deterministically():
    without = render_pubmedqa_prompt(question="Exact question?")
    with_context = render_pubmedqa_prompt(
        question="Exact question?", passage_bodies=_bodies()
    )
    assert without.system_message == SYSTEM_INSTRUCTION
    assert without.user_template == PUBMEDQA_WITHOUT_CONTEXT_TEMPLATE
    assert with_context.user_template == PUBMEDQA_WITH_CONTEXT_TEMPLATE
    assert without.user_message == (
        "Question:\nExact question?\n\nOutput format:\n"
        "Decision: <yes|no|maybe>\nExplanation: <1-3 concise factual sentences>"
    )
    assert with_context.context_block == render_context_block(_bodies())
    assert with_context.context_block.startswith("[Document 1]\nCanonical body 1.")
    assert "\n\n[Document 2]\n" in with_context.context_block
    assert with_context.provenance_payload() == render_pubmedqa_prompt(
        question="Exact question?", passage_bodies=_bodies()
    ).provenance_payload()
    assert len(PROMPT_BUNDLE_SHA256) == 64
    assert without.rendered_prompt_sha256 != with_context.rendered_prompt_sha256


def test_context_renderer_rejects_noncanonical_count_and_outer_whitespace():
    with pytest.raises(ValueError, match="exactly five"):
        render_context_block(("one",))
    with pytest.raises(ValueError, match="outer whitespace"):
        render_context_block((" body", "b", "c", "d", "e"))


def test_selected_context_is_exact_content_addressed_top_five(tmp_path):
    records, candidate, candidate_set = _candidate_fixture()
    selected = build_relevance_selected_context(
        candidate_artifact=candidate,
        candidate_set=candidate_set,
        corpus_records=records,
    )
    assert [item.rank for item in selected.passages] == [1, 2, 3, 4, 5]
    assert [item.document_id for item in selected.passages] == [0, 1, 2, 3, 4]
    assert selected.context_block == render_context_block(_bodies())
    path = tmp_path / "context.json"
    write_selected_context(selected, path)
    assert read_selected_context(path) == selected
    write_selected_context(selected, path)
    changed = deepcopy(selected.wrapper())
    changed["scientific_payload"]["passages"][0]["passage_body"] = "altered"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        read_selected_context(path)


def test_selected_context_order_and_body_change_identity():
    records, candidate, candidate_set = _candidate_fixture()
    original = build_relevance_selected_context(
        candidate_artifact=candidate,
        candidate_set=candidate_set,
        corpus_records=records,
    )
    entries = list(candidate.candidates)
    entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(ValueError, match="ranks"):
        CandidateArtifact(
            **{**candidate.__dict__, "candidates": tuple(entries)}
        )
    changed_records = list(records)
    changed_records[0] = CorpusRecord(
        **{**records[0].__dict__, "text": "Different body."}
    )
    with pytest.raises(ValueError, match="differs"):
        build_relevance_selected_context(
            candidate_artifact=candidate,
            candidate_set=candidate_set,
            corpus_records=tuple(changed_records),
        )
    assert original.artifact_id.startswith("selected-context:sha256:")


def test_selected_context_materializer_is_missing_only_and_builds_complete_set(tmp_path):
    records, base, _ = _candidate_fixture()
    candidates = (
        CandidateArtifact(**{**base.__dict__, "sample_id": 10, "query_text": "Q0?"}),
        CandidateArtifact(**{**base.__dict__, "sample_id": 11, "query_text": "Q1?"}),
    )
    candidate_dir = tmp_path / "candidates"
    for position, candidate in enumerate(candidates):
        write_candidate_artifact(candidate, candidate_dir / f"sample_{position:04d}.json")
    scientific = candidate_set_scientific_payload(
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_manifest_id=base.dataset.sample_manifest_id,
        retriever="bm25",
        expected_query_count=2,
        entries=[
            {"sample_id": item.sample_id, "candidate_artifact_id": item.artifact_id}
            for item in candidates
        ],
    )
    candidate_set_path = tmp_path / "candidate_set.json"
    write_candidate_set_artifact(candidate_set_path, scientific)
    runtime = SimpleNamespace(
        ordered_queries=(
            SimpleNamespace(position=0, sample_id=10, query_text="Q0?"),
            SimpleNamespace(position=1, sample_id=11, query_text="Q1?"),
        ),
        corpus_records=records,
    )
    kwargs = {
        "runtime": runtime,
        "candidate_directory": candidate_dir,
        "candidate_set_path": candidate_set_path,
        "output_directory": tmp_path / "contexts",
        "output_inventory_path": tmp_path / "context_set.json",
        "repository_root": tmp_path,
    }
    first = materialize_relevance_selected_contexts(**kwargs)
    mtimes = [
        (tmp_path / "contexts" / f"sample_{position:04d}.json").stat().st_mtime_ns
        for position in range(2)
    ]
    second = materialize_relevance_selected_contexts(**kwargs)
    assert first == second
    assert first["scientific_payload"]["expected_query_count"] == 2
    assert mtimes == [
        (tmp_path / "contexts" / f"sample_{position:04d}.json").stat().st_mtime_ns
        for position in range(2)
    ]


def test_with_context_generation_request_binds_exact_selected_context_identity():
    records, candidate, candidate_set = _candidate_fixture()
    selected = build_relevance_selected_context(
        candidate_artifact=candidate,
        candidate_set=candidate_set,
        corpus_records=records,
    )
    adapter = CanonicalMakiAdapter(_maki_config(), transport=lambda **_: {})
    planned = {
        "run_id": "run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-" + "a" * 24,
        "evidence_role": "HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        "retrieval": {
            "retriever": "bm25",
            "candidate_set": {"artifact_id": candidate_set["candidate_set_id"]},
        },
    }
    request, prompt = _request_for_query(
        planned=planned,
        adapter=adapter,
        query=SimpleNamespace(sample_id=7, query_text="Exact question?"),
        selected_context=selected.wrapper(),
    )
    assert request["selected_context_id"] == selected.artifact_id
    assert request["candidate_set_id"] == candidate_set["candidate_set_id"]
    assert request["prompt"]["context_block_sha256"] == selected.scientific_payload()["context_block_sha256"]
    assert prompt.context_block == selected.context_block
    changed = deepcopy(planned)
    changed["retrieval"]["candidate_set"]["artifact_id"] = f"candidate-set:sha256:{'f' * 64}"
    with pytest.raises(ValueError, match="candidate set differs"):
        _request_for_query(
            planned=changed,
            adapter=adapter,
            query=SimpleNamespace(sample_id=7, query_text="Exact question?"),
            selected_context=selected.wrapper(),
        )


@pytest.mark.parametrize(
    ("raw", "finish", "refusal", "expected"),
    [
        ("Decision: yes\nExplanation: Supported.", "stop", False, GenerationStatus.OK),
        ("I cannot answer this question.", "stop", False, GenerationStatus.REFUSAL),
        ("Malformed", "stop", False, GenerationStatus.PARSE_FAILURE),
        ("Decision: yes", "length", False, GenerationStatus.TRUNCATED),
        ("Malformed", "stop", True, GenerationStatus.REFUSAL),
        ("I cannot answer.", "length", True, GenerationStatus.TRUNCATED),
    ],
)
def test_pubmedqa_status_precedence(raw, finish, refusal, expected):
    status, _ = classify_pubmedqa_response(
        raw_content=raw, finish_reason=finish, provider_refusal=refusal
    )
    assert status is expected


def test_pubmedqa_error_requires_exhausted_transport():
    status, parsed = classify_pubmedqa_response(
        raw_content=None, finish_reason=None, transport_exhausted=True
    )
    assert status is GenerationStatus.ERROR
    assert parsed is None


def test_maki_requires_physical_id_and_has_no_qwen_default():
    with pytest.raises(ValueError, match="physical_model_id"):
        _maki_config(physical_model_id="")
    assert "qwen" not in _maki_config().physical_model_id.lower()


@pytest.mark.parametrize("seed", [True, False, 1.0, "20260823"])
def test_maki_seed_must_be_a_non_boolean_integer_or_null(seed):
    with pytest.raises(TypeError, match="integer or null"):
        _maki_config(seed=seed)


def test_maki_seed_is_separate_from_direct_mode_and_runtime_provenance():
    prompt = render_pubmedqa_prompt(question="Q?")
    seeded = CanonicalMakiAdapter(
        _maki_config(seed=20260823), transport=lambda **_: {}
    )
    seeded_payload = seeded.request_payload(prompt)
    assert seeded_payload["seed"] == 20260823
    assert list(seeded_payload).count("seed") == 1
    assert seeded.config.runtime_identity()["seed"] == 20260823
    assert runtime_provenance(seeded)["runtime_identity"]["seed"] == 20260823

    unseeded = CanonicalMakiAdapter(_maki_config(seed=None), transport=lambda **_: {})
    assert "seed" not in unseeded.request_payload(prompt)
    assert unseeded.config.runtime_identity()["seed"] is None

    with pytest.raises(ValueError, match="cannot override frozen keys"):
        _maki_config(seed=20260823, direct_mode_control={"seed": 7})


def test_v3_bindings_keep_ministral_direct_mode_unsupported_and_seeded():
    bindings = load_model_bindings(
        Path(__file__).resolve().parents[1]
        / "configs/sprint3/maki_model_bindings_v3.json"
    )
    assert {config.seed for config in bindings.values()} == {20260823}
    for logical_id in ("llama-3.3-70b", "gemma4-26b"):
        config = bindings[logical_id]
        assert config.direct_mode_status == "SUPPORTED_AND_ENABLED"
        assert dict(config.direct_mode_control) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    ministral = bindings["ministral-3-14b"]
    assert ministral.direct_mode_status == "NOT_SUPPORTED_BY_PROVIDER"
    assert dict(ministral.direct_mode_control) == {}
    payload = CanonicalMakiAdapter(
        ministral, transport=lambda **_: {}
    ).request_payload(render_pubmedqa_prompt(question="Q?"))
    assert payload["seed"] == 20260823
    assert "chat_template_kwargs" not in payload


def test_maki_exact_payload_and_content_is_never_retried(monkeypatch):
    monkeypatch.setenv("MAKI_API_KEY", "super-secret-value")
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Malformed"},
                }
            ],
        }

    adapter = CanonicalMakiAdapter(
        _maki_config(), transport=transport, sleep=lambda _: None, clock=lambda: "2026-08-24T00:00:00Z"
    )
    prompt = render_pubmedqa_prompt(question="Q?")
    result = adapter.complete(prompt)
    assert len(calls) == 1
    assert calls[0]["payload"]["messages"] == [dict(item) for item in prompt.messages]
    assert calls[0]["payload"]["temperature"] == 0
    assert calls[0]["payload"]["max_tokens"] == 256
    assert calls[0]["payload"]["n"] == 1
    assert calls[0]["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "seed" not in calls[0]["payload"]
    assert "super-secret-value" not in json.dumps(result.provider_metadata)


def test_maki_retries_only_infrastructure_at_most_three(monkeypatch):
    monkeypatch.setenv("MAKI_API_KEY", "secret")
    calls = 0

    def succeeds_on_third(**_):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise MakiInfrastructureError("temporary")
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Decision: no"}}]}

    adapter = CanonicalMakiAdapter(
        _maki_config(), transport=succeeds_on_third, sleep=lambda _: None, clock=lambda: "2026-08-24T00:00:00Z"
    )
    result = adapter.complete(render_pubmedqa_prompt(question="Q?"))
    assert calls == 3
    assert result.transport_exhausted is False
    assert [item["outcome"] for item in result.attempts] == [
        "INFRASTRUCTURE_ERROR", "INFRASTRUCTURE_ERROR", "SUCCESS"
    ]

    def always_fails(**_):
        raise MakiInfrastructureError("Bearer secret-token")

    exhausted = CanonicalMakiAdapter(
        _maki_config(), transport=always_fails, sleep=lambda _: None, clock=lambda: "2026-08-24T00:00:00Z"
    ).complete(render_pubmedqa_prompt(question="Q?"))
    assert exhausted.transport_exhausted is True
    assert len(exhausted.attempts) == 3
    assert "secret-token" not in json.dumps(exhausted.attempts)


def _request(prompt):
    return generation_request_payload(
        run_id="run-sprint1-pubmedqa-s1-without-context-shared-none-llama-" + "a" * 24,
        dataset="pubmedqa",
        evidence_role="HISTORICAL_OBSERVED_CONTROL_REPLICATION",
        sample_id=1,
        question_text_sha256="c" * 64,
        llm_logical_id="llama-3.3-70b",
        provider="Mannheim Maki",
        physical_model_id="provider/model",
        model_revision=None,
        model_revision_kind="NOT_PROVIDED_BY_PROVIDER",
        condition="WITHOUT_CONTEXT",
        retriever=None,
        candidate_set_id=None,
        selected_context_id=None,
        prompt=prompt.provenance_payload(),
        decoding={
            "decoding_version": "v1",
            "temperature": 0,
            "max_tokens": 256,
            "n": 1,
            "canonical_generation_replicas": 1,
            "direct_mode_status": "NOT_SUPPORTED_BY_PROVIDER",
            "direct_mode_control": {},
        },
    )


def test_generation_artifact_atomic_reuse_and_conflict(tmp_path):
    prompt = render_pubmedqa_prompt(question="Q?")
    attempt = [{
        "attempt": 1,
        "started_at": "2026-08-24T00:00:00Z",
        "completed_at": "2026-08-24T00:00:01Z",
        "outcome": "SUCCESS",
        "error_type": None,
        "error_message": None,
    }]
    artifact = build_generation_artifact(
        request=_request(prompt),
        status="OK",
        raw_content="Decision: yes",
        finish_reason="stop",
        provider_metadata={},
        parsed_output={"decision": "yes"},
        attempts=attempt,
        environment={"python": "fixture"},
        runtime={"adapter": "fixture"},
        hardware_summary="fixture",
        created_at="2026-08-24T00:00:00Z",
        completed_at="2026-08-24T00:00:01Z",
    )
    path = tmp_path / "row.json"
    write_generation_artifact(artifact, path)
    write_generation_artifact(artifact, path)
    assert read_generation_artifact(path) == artifact
    conflicting = build_generation_artifact(
        **{
            "request": _request(prompt),
            "status": "OK",
            "raw_content": "Decision: no",
            "finish_reason": "stop",
            "provider_metadata": {},
            "parsed_output": {"decision": "no"},
            "attempts": attempt,
            "environment": {"python": "fixture"},
            "runtime": {"adapter": "fixture"},
            "hardware_summary": "fixture",
            "created_at": "2026-08-24T00:00:00Z",
            "completed_at": "2026-08-24T00:00:01Z",
        }
    )
    with pytest.raises(GenerationArtifactConflictError):
        write_generation_artifact(conflicting, path)
