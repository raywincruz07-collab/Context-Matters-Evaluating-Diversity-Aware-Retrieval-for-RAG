"""CPU-only tests for structured Sprint 3 query-level evaluation."""

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.contracts import ComponentStatus, ContextMode, MetricStatus, ReasonCode
from evaluation.metric_registry import DatasetId
import evaluation.query_evaluator as query_evaluator
from evaluation.query_evaluator import (
    GenerationEvaluationInput,
    RetrievalEvaluationInput,
    evaluate_generation_metric,
    evaluate_generation_metrics,
    evaluate_retrieval_metric,
    evaluate_retrieval_metrics,
)


def _retrieval_input(**overrides):
    values = {
        "dataset_id": DatasetId.PUBMEDQA,
        "context_mode": ContextMode.WITH_CONTEXT,
        "retrieved_doc_ids": ["a", "x", "b", "y"],
        "gold_doc_ids": ["a", "b"],
        "selected_embeddings": np.eye(4),
        "evaluator_available": True,
        "protocol_frozen": True,
    }
    values.update(overrides)
    return RetrievalEvaluationInput(**values)


def _generation_input(**overrides):
    values = {
        "dataset_id": DatasetId.PUBMEDQA,
        "context_mode": ContextMode.WITH_CONTEXT,
        "generation_status": ComponentStatus.SUCCESS,
        "prediction": "answer",
        "reference_answer": "answer",
        "reference_available": True,
        "evaluator_available": True,
        "protocol_frozen": True,
    }
    values.update(overrides)
    return GenerationEvaluationInput(**values)


def _assert_result(result, status, reason=None, value=None):
    assert result.status is status
    assert result.reason is reason
    assert result.value == value


def test_asqa_retrieval_registry_decision_precedes_legacy_metric(monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("legacy metric must not be called")

    monkeypatch.setattr(query_evaluator, "legacy_recall_at_k", fail_if_called)
    monkeypatch.setattr(query_evaluator, "legacy_mrr", fail_if_called)
    inputs = _retrieval_input(dataset_id=DatasetId.ASQA)

    for metric_id in ("recall_at_k", "mrr_at_k"):
        _assert_result(
            evaluate_retrieval_metric(metric_id, inputs),
            MetricStatus.NOT_COMPUTED,
            ReasonCode.PROTOCOL_NOT_FROZEN,
        )


@pytest.mark.parametrize("metric_id", ["exact_match", "token_f1", "rouge_l"])
def test_hotpot_missing_reference_precedes_legacy_answer_metric(metric_id, monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("legacy metric must not be called")

    monkeypatch.setattr(query_evaluator, f"legacy_{metric_id}", fail_if_called)
    result = evaluate_generation_metric(
        metric_id,
        _generation_input(
            dataset_id=DatasetId.HOTPOTQA,
            reference_answer=None,
            reference_available=False,
        ),
    )
    _assert_result(
        result, MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
    )


def test_blocked_faithfulness_decisions_are_preserved():
    without_context = evaluate_generation_metric(
        "faithfulness_to_context",
        _generation_input(context_mode=ContextMode.WITHOUT_CONTEXT),
    )
    _assert_result(
        without_context,
        MetricStatus.NOT_APPLICABLE,
        ReasonCode.NOT_DEFINED_FOR_CONTEXT_MODE,
    )
    with_context = evaluate_generation_metric(
        "faithfulness_to_context", _generation_input()
    )
    _assert_result(
        with_context, MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
    )


def test_unimplemented_pubmedqa_decision_metric_is_not_configured():
    result = evaluate_generation_metric(
        "pubmedqa_decision_accuracy", _generation_input()
    )
    _assert_result(
        result, MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_NOT_CONFIGURED
    )


def test_unknown_metric_and_wrong_scope_raise():
    with pytest.raises(KeyError, match="unknown metric_id"):
        evaluate_retrieval_metric("unknown", _retrieval_input())
    with pytest.raises(ValueError, match="not a retrieval-artifact metric"):
        evaluate_retrieval_metric("exact_match", _retrieval_input())
    with pytest.raises(ValueError, match="not a generation metric"):
        evaluate_generation_metric("recall_at_k", _generation_input())


@pytest.mark.parametrize(
    ("retrieved", "gold", "expected"),
    [
        (["a", "x", "b", "y"], ["a", "b"], 1.0),
        (["x", "a", "y"], ["a", "b"], 0.5),
    ],
)
def test_recall_hand_checked(retrieved, gold, expected):
    result = evaluate_retrieval_metric(
        "recall_at_k",
        _retrieval_input(retrieved_doc_ids=retrieved, gold_doc_ids=gold),
    )
    _assert_result(result, MetricStatus.MEASURED, value=expected)


def test_mrr_hand_checked_and_retrieval_order_preserved():
    inputs = _retrieval_input(
        retrieved_doc_ids=["x", "y", "b", "a"], gold_doc_ids=["a", "b"]
    )
    assert inputs.retrieved_doc_ids == ("x", "y", "b", "a")
    result = evaluate_retrieval_metric("mrr_at_k", inputs)
    _assert_result(result, MetricStatus.MEASURED, value=pytest.approx(1 / 3))


def test_no_hit_is_a_genuine_measured_zero():
    result = evaluate_retrieval_metric(
        "mrr_at_k",
        _retrieval_input(retrieved_doc_ids=["x", "y"], gold_doc_ids=["a", "b"]),
    )
    _assert_result(result, MetricStatus.MEASURED, value=0.0)


@pytest.mark.parametrize("metric_id", ["recall_at_k", "mrr_at_k"])
def test_missing_qrels_are_not_computed(metric_id):
    result = evaluate_retrieval_metric(
        metric_id, _retrieval_input(gold_doc_ids=None)
    )
    _assert_result(
        result, MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
    )


@pytest.mark.parametrize("gold_doc_ids", [[], ["a", "a"]])
@pytest.mark.parametrize("metric_id", ["recall_at_k", "mrr_at_k"])
def test_malformed_qrels_fail_structurally(metric_id, gold_doc_ids):
    result = evaluate_retrieval_metric(
        metric_id, _retrieval_input(gold_doc_ids=gold_doc_ids)
    )
    _assert_result(result, MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)


@pytest.mark.parametrize(
    ("embeddings", "expected"),
    [
        (np.array([[1.0, 0.0], [1.0, 0.0]]), 0.0),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), 1.0),
        (np.array([[1.0, 0.0], [-1.0, 0.0]]), 2.0),
    ],
)
def test_retrieval_diversity_hand_checked(embeddings, expected):
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=["a", "b"], selected_embeddings=embeddings
        ),
    )
    _assert_result(result, MetricStatus.MEASURED, value=expected)


def test_missing_and_insufficient_embeddings_are_not_computed():
    missing = evaluate_retrieval_metric(
        "retrieval_diversity", _retrieval_input(selected_embeddings=None)
    )
    _assert_result(
        missing, MetricStatus.NOT_COMPUTED, ReasonCode.MISSING_EMBEDDING
    )
    insufficient = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=["a"],
            selected_embeddings=np.array([[1.0, 0.0]]),
        ),
    )
    _assert_result(
        insufficient, MetricStatus.NOT_COMPUTED, ReasonCode.INSUFFICIENT_ITEMS
    )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[np.nan, 0.0], [1.0, 0.0]]),
        np.array([[np.inf, 0.0], [1.0, 0.0]]),
        np.array([[1.0j, 0.0], [1.0, 0.0]]),
        np.array([[True, False], [False, True]]),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([1.0, 0.0]),
        np.empty((2, 0)),
    ],
)
def test_malformed_embeddings_fail_structurally(embeddings):
    retrieved_doc_ids = ["a", "b"]
    if np.asarray(embeddings).ndim == 2:
        retrieved_doc_ids = [f"doc-{i}" for i in range(len(embeddings))]
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=retrieved_doc_ids, selected_embeddings=embeddings
        ),
    )
    _assert_result(result, MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)


@pytest.mark.parametrize(
    ("retrieved_count", "embedding_rows"), [(5, 4), (4, 5)]
)
def test_embedding_row_count_must_match_complete_retrieval_set(
    retrieved_count, embedding_rows
):
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=list(range(retrieved_count)),
            selected_embeddings=np.eye(embedding_rows),
        ),
    )
    _assert_result(result, MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)


def test_matching_embedding_row_count_remains_measured():
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=[0, 1, 2, 3], selected_embeddings=np.eye(4)
        ),
    )
    assert result.status is MetricStatus.MEASURED


def test_unsigned_integer_embeddings_are_valid_real_numeric_input():
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            retrieved_doc_ids=[0, 1],
            selected_embeddings=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        ),
    )
    _assert_result(result, MetricStatus.MEASURED, value=1.0)


def test_evaluator_unavailable_precedes_embedding_validation(monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("legacy metric must not be called")

    monkeypatch.setattr(
        query_evaluator, "legacy_retrieval_diversity", fail_if_called
    )
    result = evaluate_retrieval_metric(
        "retrieval_diversity",
        _retrieval_input(
            selected_embeddings=np.array([np.nan]), evaluator_available=False
        ),
    )
    _assert_result(
        result, MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_UNAVAILABLE
    )


def test_generation_input_statuses_precede_input_validation():
    failed = evaluate_generation_metric(
        "exact_match",
        _generation_input(
            generation_status=ComponentStatus.FAILED, prediction=None
        ),
    )
    _assert_result(failed, MetricStatus.FAILED, ReasonCode.GENERATION_FAILED)
    not_run = evaluate_generation_metric(
        "exact_match",
        _generation_input(
            generation_status=ComponentStatus.NOT_RUN, prediction=None
        ),
    )
    _assert_result(
        not_run,
        MetricStatus.NOT_COMPUTED,
        ReasonCode.GENERATION_NOT_ATTEMPTED,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"prediction": None},
        {"reference_answer": None, "reference_available": True},
    ],
)
def test_generation_success_with_missing_required_text_fails(overrides):
    result = evaluate_generation_metric("exact_match", _generation_input(**overrides))
    _assert_result(result, MetricStatus.FAILED, ReasonCode.INVALID_METRIC_INPUT)


def test_exact_match_measured_one_and_zero():
    match = evaluate_generation_metric(
        "exact_match",
        _generation_input(prediction="The Answer!", reference_answer="answer"),
    )
    _assert_result(match, MetricStatus.MEASURED, value=1.0)
    mismatch = evaluate_generation_metric(
        "exact_match",
        _generation_input(prediction="yes", reference_answer="no"),
    )
    _assert_result(mismatch, MetricStatus.MEASURED, value=0.0)


def test_token_f1_partial_overlap():
    result = evaluate_generation_metric(
        "token_f1",
        _generation_input(prediction="red blue", reference_answer="red green"),
    )
    _assert_result(result, MetricStatus.MEASURED, value=0.5)


def test_custom_rouge_l_hand_checked():
    result = evaluate_generation_metric(
        "rouge_l",
        _generation_input(prediction="x y z", reference_answer="x z"),
    )
    _assert_result(result, MetricStatus.MEASURED, value=pytest.approx(0.8))


def test_retrieval_batch_isolates_metric_exception(monkeypatch):
    def raise_metric(*args):
        raise RuntimeError("test failure")

    monkeypatch.setattr(query_evaluator, "legacy_recall_at_k", raise_metric)
    results = evaluate_retrieval_metrics(_retrieval_input())
    _assert_result(
        results["recall_at_k"], MetricStatus.FAILED, ReasonCode.METRIC_EXCEPTION
    )
    assert results["mrr_at_k"].status is MetricStatus.MEASURED
    assert results["retrieval_diversity"].status is MetricStatus.MEASURED


def test_generation_batch_isolates_metric_exception(monkeypatch):
    def raise_metric(*args):
        raise RuntimeError("test failure")

    monkeypatch.setattr(query_evaluator, "legacy_token_f1", raise_metric)
    results = evaluate_generation_metrics(_generation_input())
    assert results["exact_match"].status is MetricStatus.MEASURED
    _assert_result(
        results["token_f1"], MetricStatus.FAILED, ReasonCode.METRIC_EXCEPTION
    )
    assert results["rouge_l"].status is MetricStatus.MEASURED


def test_inputs_are_frozen_and_document_id_sequences_are_copied():
    retrieved = ["a", "b"]
    inputs = _retrieval_input(retrieved_doc_ids=retrieved)
    retrieved.append("c")
    assert inputs.retrieved_doc_ids == ("a", "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.protocol_frozen = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        _generation_input().prediction = "changed"
