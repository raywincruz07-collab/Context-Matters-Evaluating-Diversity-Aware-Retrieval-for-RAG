"""CPU-only tests for the Sprint 3 metric applicability registry."""

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.contracts import ComponentStatus, ContextMode, MetricStatus, ReasonCode
from evaluation.metric_registry import (
    METRIC_REGISTRY,
    ApplicabilityContext,
    DatasetId,
    MetricDefinition,
    MetricRegistry,
    MetricScope,
)


def _context(**overrides):
    values = {
        "dataset_id": DatasetId.PUBMEDQA,
        "context_mode": ContextMode.WITH_CONTEXT,
        "generation_status": ComponentStatus.SUCCESS,
        "reference_available": True,
        "evaluator_available": True,
        "protocol_frozen": True,
    }
    values.update(overrides)
    return ApplicabilityContext(**values)


def _assert_unavailable(decision, status, reason):
    assert not decision.eligible
    assert decision.status is status
    assert decision.reason is reason


def test_unknown_metric_id_is_rejected():
    with pytest.raises(KeyError, match="unknown metric_id"):
        METRIC_REGISTRY.get("unknown")
    with pytest.raises(KeyError, match="unknown metric_id"):
        METRIC_REGISTRY.decide("unknown", _context())


def test_duplicate_metric_ids_are_rejected():
    definition = METRIC_REGISTRY.get("recall_at_k")
    with pytest.raises(ValueError, match="duplicate metric_id"):
        MetricRegistry([definition, definition])


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        MetricScope("unknown")
    definition = METRIC_REGISTRY.get("recall_at_k")
    with pytest.raises(TypeError):
        MetricDefinition(
            metric_id="new_metric",
            version="sprint3.new_metric.v1",
            scope="retrieval_artifact",
            allowed_datasets=definition.allowed_datasets,
            allowed_context_modes=definition.allowed_context_modes,
            requires_generation=False,
            requires_reference=False,
            requires_evaluator=True,
            protocol_frozen_datasets=definition.protocol_frozen_datasets,
        )


@pytest.mark.parametrize("metric_id", ["recall_at_k", "mrr_at_k"])
@pytest.mark.parametrize("dataset_id", [DatasetId.PUBMEDQA, DatasetId.HOTPOTQA])
def test_relevance_retrieval_metrics_are_eligible_for_frozen_datasets(
    metric_id, dataset_id
):
    assert METRIC_REGISTRY.decide(
        metric_id, _context(dataset_id=dataset_id)
    ).eligible


@pytest.mark.parametrize("metric_id", ["recall_at_k", "mrr_at_k"])
def test_relevance_retrieval_metrics_are_blocked_for_asqa(metric_id):
    decision = METRIC_REGISTRY.decide(
        metric_id, _context(dataset_id=DatasetId.ASQA)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
    )


@pytest.mark.parametrize(
    "metric_id", ["recall_at_k", "mrr_at_k", "retrieval_diversity"]
)
def test_retrieval_metrics_are_independent_of_context_mode(metric_id):
    with_context = METRIC_REGISTRY.decide(
        metric_id, _context(context_mode=ContextMode.WITH_CONTEXT)
    )
    without_context = METRIC_REGISTRY.decide(
        metric_id, _context(context_mode=ContextMode.WITHOUT_CONTEXT)
    )
    assert with_context == without_context
    assert with_context.eligible
    assert METRIC_REGISTRY.get(metric_id).scope is MetricScope.RETRIEVAL_ARTIFACT


@pytest.mark.parametrize(
    "context_mode", [ContextMode.WITH_CONTEXT, ContextMode.WITHOUT_CONTEXT]
)
def test_pubmedqa_decision_accuracy_is_eligible_in_both_context_modes(context_mode):
    assert METRIC_REGISTRY.decide(
        "pubmedqa_decision_accuracy", _context(context_mode=context_mode)
    ).eligible


@pytest.mark.parametrize("dataset_id", [DatasetId.HOTPOTQA, DatasetId.ASQA])
def test_pubmedqa_decision_accuracy_is_not_applicable_elsewhere(dataset_id):
    decision = METRIC_REGISTRY.decide(
        "pubmedqa_decision_accuracy", _context(dataset_id=dataset_id)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET
    )


@pytest.mark.parametrize("metric_id", ["exact_match", "token_f1", "rouge_l"])
def test_hotpot_answer_metrics_report_unavailable_reference(metric_id):
    decision = METRIC_REGISTRY.decide(
        metric_id,
        _context(dataset_id=DatasetId.HOTPOTQA, reference_available=False),
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
    )


def test_faithfulness_context_decisions_follow_scientific_priority():
    with_context = METRIC_REGISTRY.decide(
        "faithfulness_to_context", _context()
    )
    _assert_unavailable(
        with_context, MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
    )

    without_context = METRIC_REGISTRY.decide(
        "faithfulness_to_context",
        _context(
            context_mode=ContextMode.WITHOUT_CONTEXT,
            generation_status=ComponentStatus.FAILED,
            evaluator_available=False,
            protocol_frozen=False,
        ),
    )
    _assert_unavailable(
        without_context,
        MetricStatus.NOT_APPLICABLE,
        ReasonCode.NOT_DEFINED_FOR_CONTEXT_MODE,
    )


@pytest.mark.parametrize("dataset_id", [DatasetId.PUBMEDQA, DatasetId.HOTPOTQA])
def test_asqa_alpha_ndcg_is_not_applicable_to_other_datasets(dataset_id):
    decision = METRIC_REGISTRY.decide(
        "asqa_alpha_ndcg", _context(dataset_id=dataset_id)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET
    )


def test_asqa_alpha_ndcg_cannot_become_eligible():
    decision = METRIC_REGISTRY.decide(
        "asqa_alpha_ndcg",
        _context(dataset_id=DatasetId.ASQA, protocol_frozen=True),
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
    )


@pytest.mark.parametrize("dataset_id", [DatasetId.PUBMEDQA, DatasetId.HOTPOTQA])
def test_asqa_official_answer_is_not_applicable_to_other_datasets(dataset_id):
    decision = METRIC_REGISTRY.decide(
        "asqa_official_answer", _context(dataset_id=dataset_id)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET
    )


def test_asqa_official_answer_cannot_become_eligible():
    decision = METRIC_REGISTRY.decide(
        "asqa_official_answer",
        _context(dataset_id=DatasetId.ASQA, protocol_frozen=True),
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.PROTOCOL_NOT_FROZEN
    )


def test_failed_generation_fails_otherwise_eligible_generation_metric():
    decision = METRIC_REGISTRY.decide(
        "exact_match", _context(generation_status=ComponentStatus.FAILED)
    )
    _assert_unavailable(
        decision, MetricStatus.FAILED, ReasonCode.GENERATION_FAILED
    )


def test_generation_not_run_blocks_otherwise_eligible_generation_metric():
    decision = METRIC_REGISTRY.decide(
        "exact_match", _context(generation_status=ComponentStatus.NOT_RUN)
    )
    _assert_unavailable(
        decision,
        MetricStatus.NOT_COMPUTED,
        ReasonCode.GENERATION_NOT_ATTEMPTED,
    )


def test_unavailable_evaluator_blocks_metric_that_requires_it():
    decision = METRIC_REGISTRY.decide(
        "retrieval_diversity", _context(evaluator_available=False)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_UNAVAILABLE
    )


def test_unavailable_reference_blocks_metric_that_requires_it():
    decision = METRIC_REGISTRY.decide(
        "exact_match", _context(reference_available=False)
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_COMPUTED, ReasonCode.REFERENCE_UNAVAILABLE
    )


def test_dataset_inapplicability_wins_over_operational_failures():
    decision = METRIC_REGISTRY.decide(
        "pubmedqa_decision_accuracy",
        _context(
            dataset_id=DatasetId.HOTPOTQA,
            generation_status=ComponentStatus.FAILED,
            reference_available=False,
            evaluator_available=False,
            protocol_frozen=False,
        ),
    )
    _assert_unavailable(
        decision, MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET
    )


def test_retrieval_metric_ignores_generation_failure():
    decision = METRIC_REGISTRY.decide(
        "recall_at_k", _context(generation_status=ComponentStatus.FAILED)
    )
    assert decision.eligible


def test_registry_and_definitions_are_immutable():
    definition = METRIC_REGISTRY.get("recall_at_k")
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.metric_id = "changed"
    with pytest.raises(TypeError):
        METRIC_REGISTRY.definitions["new"] = definition
    with pytest.raises(dataclasses.FrozenInstanceError):
        METRIC_REGISTRY._definitions = {}
