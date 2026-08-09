"""CPU-only tests for Sprint 3 metric aggregation and context pairing."""

import dataclasses
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.aggregation import (
    DIFFERENCE_DIRECTION,
    MetricObservation,
    aggregate_metric_group,
    compare_context_modes,
)
from evaluation.contracts import ContextMode, MetricResult, MetricStatus, ReasonCode
from evaluation.metric_registry import DatasetId


def _result(status=MetricStatus.MEASURED, value=1.0):
    if status is MetricStatus.MEASURED:
        return MetricResult(value=value, status=status)
    reason = {
        MetricStatus.NOT_APPLICABLE: ReasonCode.NOT_DEFINED_FOR_DATASET,
        MetricStatus.NOT_COMPUTED: ReasonCode.REFERENCE_UNAVAILABLE,
        MetricStatus.FAILED: ReasonCode.METRIC_EXCEPTION,
    }[status]
    return MetricResult(value=None, status=status, reason=reason)


def _record(sample_id, value=1.0, status=MetricStatus.MEASURED, **overrides):
    values = {
        "sample_id": sample_id,
        "metric_id": "exact_match",
        "metric_result": _result(status, value),
        "dataset_id": DatasetId.PUBMEDQA,
        "dataset_split": "test",
        "retriever": "contriever",
        "diversification_condition": "none",
        "model_provider": "provider",
        "model_id": "model-a",
        "model_revision": "rev-1",
        "context_mode": ContextMode.WITH_CONTEXT,
        "generation_replica": 0,
        "retrieval_artifact_id": f"artifact-{sample_id}",
    }
    metric_id = overrides.get("metric_id", values["metric_id"])
    context_mode = overrides.get("context_mode", values["context_mode"])
    if metric_id in {
        "recall_at_k",
        "mrr_at_k",
        "retrieval_diversity",
        "asqa_alpha_ndcg",
    }:
        values["model_provider"] = None
        values["model_id"] = None
        values["model_revision"] = None
        values["context_mode"] = None
        values["generation_replica"] = None
    elif context_mode is ContextMode.WITHOUT_CONTEXT:
        values["retriever"] = None
        values["diversification_condition"] = None
        values["retrieval_artifact_id"] = None
    values.update(overrides)
    return MetricObservation(**values)


def test_all_measured_includes_zero_and_uses_sample_std():
    summary = aggregate_metric_group(
        [_record("q1", 1.0), _record("q2", 0.0), _record("q3", 0.5)],
        ["q1", "q2", "q3"],
    )
    assert summary.n_measured == 3
    assert summary.mean == 0.5
    assert summary.std == 0.5
    assert summary.measured_sample_ids == ("q1", "q2", "q3")


def test_mixed_statuses_are_all_counted_without_filtering():
    summary = aggregate_metric_group(
        [
            _record("q1", 0.0),
            _record("q2", status=MetricStatus.NOT_APPLICABLE),
            _record("q3", status=MetricStatus.NOT_COMPUTED),
            _record("q4", status=MetricStatus.FAILED),
        ],
        ["q1", "q2", "q3", "q4", "q5"],
    )
    assert summary.n_rows_present == 4
    assert summary.n_measured == 1
    assert summary.n_not_applicable == 1
    assert summary.n_not_computed == 1
    assert summary.n_failed == 1
    assert summary.n_missing_rows == 1
    assert summary.missing_sample_ids == ("q5",)
    assert summary.mean == 0.0
    assert summary.std is None
    assert summary.n_rows_present == (
        summary.n_measured
        + summary.n_not_applicable
        + summary.n_not_computed
        + summary.n_failed
    )
    assert summary.n_total_expected == summary.n_rows_present + summary.n_missing_rows


def test_no_measured_rows_have_missing_mean_and_std():
    summary = aggregate_metric_group(
        [
            _record("q1", status=MetricStatus.NOT_COMPUTED),
            _record("q2", status=MetricStatus.FAILED),
        ],
        ["q1", "q2"],
    )
    assert summary.n_measured == 0
    assert summary.mean is None
    assert summary.std is None


def test_one_measured_value_has_no_standard_deviation():
    summary = aggregate_metric_group([_record("q1", 0.25)], ["q1"])
    assert summary.mean == 0.25
    assert summary.std is None


def test_two_values_use_ddof_one_sample_standard_deviation():
    summary = aggregate_metric_group(
        [_record("q1", 0.0), _record("q2", 2.0)], ["q1", "q2"]
    )
    assert summary.std == pytest.approx(math.sqrt(2.0))


def test_output_id_order_follows_expected_manifest_not_record_order():
    summary = aggregate_metric_group(
        [_record("q3"), _record("q1")], ["q1", "q2", "q3", "q4"]
    )
    assert summary.measured_sample_ids == ("q1", "q3")
    assert summary.missing_sample_ids == ("q2", "q4")


def test_duplicate_observed_sample_is_rejected():
    with pytest.raises(ValueError, match="duplicate sample_id"):
        aggregate_metric_group([_record("q1"), _record("q1")], ["q1"])


def test_duplicate_expected_sample_is_rejected():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        aggregate_metric_group([_record("q1")], ["q1", "q1"])


def test_unknown_observed_sample_is_rejected():
    with pytest.raises(ValueError, match="absent from expected manifest"):
        aggregate_metric_group([_record("q2")], ["q1"])


def test_mixed_metric_ids_are_rejected():
    with pytest.raises(ValueError, match="one logical metric group"):
        aggregate_metric_group(
            [_record("q1"), _record("q2", metric_id="token_f1")],
            ["q1", "q2"],
        )


@pytest.mark.parametrize(
    ("field_name", "changed"),
    [
        ("dataset_id", DatasetId.HOTPOTQA),
        ("dataset_split", "dev"),
        ("retriever", "bm25"),
        ("diversification_condition", "mmr_0.5"),
        ("model_provider", "other-provider"),
        ("model_id", "model-b"),
        ("model_revision", "rev-2"),
        ("context_mode", ContextMode.WITHOUT_CONTEXT),
        ("generation_replica", 1),
    ],
)
def test_generation_group_rejects_mixed_dimensions(field_name, changed):
    with pytest.raises(ValueError, match="one logical metric group"):
        aggregate_metric_group(
            [_record("q1"), _record("q2", **{field_name: changed})],
            ["q1", "q2"],
        )


def test_retrieval_group_uses_clean_retrieval_only_identity():
    summary = aggregate_metric_group(
        [
            _record("q1", metric_id="recall_at_k"),
            _record("q2", metric_id="recall_at_k"),
        ],
        ["q1", "q2"],
    )
    assert summary.n_measured == 2
    assert summary.mean == 1.0
    assert summary.model_provider is None
    assert summary.model_id is None
    assert summary.model_revision is None
    assert summary.context_mode is None
    assert summary.generation_replica is None


def test_duplicated_retrieval_observation_is_rejected_not_deduplicated():
    first = _record("q1", metric_id="recall_at_k")
    copied = _record("q1", metric_id="recall_at_k")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        aggregate_metric_group([first, copied], ["q1"])


@pytest.mark.parametrize(
    ("field_name", "changed"),
    [("retriever", "bm25"), ("diversification_condition", "mmr_0.5")],
)
def test_retrieval_group_validates_retriever_and_condition(field_name, changed):
    with pytest.raises(ValueError, match="one logical metric group"):
        aggregate_metric_group(
            [
                _record("q1", metric_id="recall_at_k"),
                _record("q2", metric_id="recall_at_k", **{field_name: changed}),
            ],
            ["q1", "q2"],
        )


def test_hand_checked_paired_context_comparison():
    expected = ["q1", "q2", "q3", "q4"]
    with_records = [
        _record("q3", status=MetricStatus.FAILED),
        _record("q1", 1.0),
        _record("q2", 0.5),
    ]
    without_records = [
        _record("q4", 0.0, context_mode=ContextMode.WITHOUT_CONTEXT),
        _record("q2", 0.5, context_mode=ContextMode.WITHOUT_CONTEXT),
        _record("q3", 1.0, context_mode=ContextMode.WITHOUT_CONTEXT),
        _record("q1", 0.0, context_mode=ContextMode.WITHOUT_CONTEXT),
    ]
    summary = compare_context_modes(with_records, without_records, expected)
    assert summary.n_pair_expected == 4
    assert summary.n_with_context_present == 3
    assert summary.n_without_context_present == 4
    assert summary.n_pair_rows_present == 3
    assert summary.n_pair_complete == 2
    assert summary.n_pair_incomplete == 2
    assert summary.paired_sample_ids == ("q1", "q2")
    assert summary.with_context_mean_on_pairs == 0.75
    assert summary.without_context_mean_on_pairs == 0.25
    assert summary.mean_paired_difference == 0.5
    assert summary.missing_with_context_sample_ids == ("q4",)
    assert summary.missing_without_context_sample_ids == ()
    assert summary.incomplete_pair_sample_ids == ("q3", "q4")
    assert summary.difference_direction == DIFFERENCE_DIRECTION
    assert summary.retriever == "contriever"
    assert summary.diversification_condition == "none"


def test_zero_complete_pairs_have_no_effect_means():
    summary = compare_context_modes(
        [_record("q1", status=MetricStatus.FAILED)],
        [
            _record(
                "q1",
                status=MetricStatus.NOT_COMPUTED,
                context_mode=ContextMode.WITHOUT_CONTEXT,
            )
        ],
        ["q1"],
    )
    assert summary.n_pair_complete == 0
    assert summary.with_context_mean_on_pairs is None
    assert summary.without_context_mean_on_pairs is None
    assert summary.mean_paired_difference is None


def test_measured_zero_is_a_complete_pair():
    summary = compare_context_modes(
        [_record("q1", 0.0)],
        [_record("q1", 0.0, context_mode=ContextMode.WITHOUT_CONTEXT)],
        ["q1"],
    )
    assert summary.n_pair_complete == 1
    assert summary.mean_paired_difference == 0.0


def test_context_comparison_rejects_retrieval_metric():
    with pytest.raises(ValueError, match="only generation-scope"):
        compare_context_modes(
            [_record("q1", metric_id="recall_at_k")],
            [_record("q1", metric_id="recall_at_k")],
            ["q1"],
        )


def test_context_comparison_rejects_wrong_side_mode():
    with pytest.raises(ValueError, match="with_context_records"):
        compare_context_modes(
            [_record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)],
            [_record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)],
            ["q1"],
        )


@pytest.mark.parametrize(
    ("field_name", "changed"),
    [
        ("model_provider", "other"),
        ("model_id", "model-b"),
        ("model_revision", "rev-2"),
        ("dataset_id", DatasetId.HOTPOTQA),
        ("dataset_split", "dev"),
        ("metric_id", "token_f1"),
        ("generation_replica", 1),
    ],
)
def test_context_comparison_rejects_group_mismatch(field_name, changed):
    with pytest.raises(ValueError, match="one paired logical group"):
        compare_context_modes(
            [_record("q1")],
            [
                _record(
                    "q1",
                    context_mode=ContextMode.WITHOUT_CONTEXT,
                    **{field_name: changed},
                )
            ],
            ["q1"],
        )


@pytest.mark.parametrize(
    ("field_name", "changed"),
    [("retriever", "bm25"), ("diversification_condition", "mmr_0.5")],
)
def test_context_comparison_requires_one_with_context_condition(field_name, changed):
    with pytest.raises(ValueError, match="one retrieval condition"):
        compare_context_modes(
            [_record("q1"), _record("q2", **{field_name: changed})],
            [
                _record("q1", context_mode=ContextMode.WITHOUT_CONTEXT),
                _record("q2", context_mode=ContextMode.WITHOUT_CONTEXT),
            ],
            ["q1", "q2"],
        )


def test_context_comparison_uses_canonical_no_context_without_artifact():
    without = _record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)
    assert without.retrieval_artifact_id is None
    summary = compare_context_modes([_record("q1")], [without], ["q1"])
    assert summary.n_pair_complete == 1


@pytest.mark.parametrize("side", ["with", "without"])
def test_context_comparison_rejects_duplicate_sample_on_either_side(side):
    with_records = [_record("q1")]
    without_records = [
        _record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)
    ]
    if side == "with":
        with_records.append(_record("q1"))
    else:
        without_records.append(
            _record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)
        )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        compare_context_modes(with_records, without_records, ["q1"])


def test_context_comparison_rejects_unknown_sample():
    with pytest.raises(ValueError, match="absent from expected manifest"):
        compare_context_modes(
            [_record("q2")],
            [_record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)],
            ["q1"],
        )


def test_paired_output_order_follows_expected_manifest():
    summary = compare_context_modes(
        [_record("q3"), _record("q1")],
        [
            _record("q3", context_mode=ContextMode.WITHOUT_CONTEXT),
            _record("q1", context_mode=ContextMode.WITHOUT_CONTEXT),
        ],
        ["q1", "q2", "q3"],
    )
    assert summary.paired_sample_ids == ("q1", "q3")
    assert summary.incomplete_pair_sample_ids == ("q2",)
    assert summary.missing_with_context_sample_ids == ("q2",)
    assert summary.missing_without_context_sample_ids == ("q2",)


def test_summary_types_are_frozen():
    metric_summary = aggregate_metric_group([_record("q1")], ["q1"])
    paired_summary = compare_context_modes(
        [_record("q1")],
        [_record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)],
        ["q1"],
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        metric_summary.mean = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        paired_summary.n_pair_complete = 0


def test_canonical_no_context_group_aggregates_without_retrieval_metadata():
    records = [
        _record("q1", 0.2, context_mode=ContextMode.WITHOUT_CONTEXT),
        _record("q2", 0.4, context_mode=ContextMode.WITHOUT_CONTEXT),
    ]
    summary = aggregate_metric_group(records, ["q1", "q2"])
    assert summary.retriever is None
    assert summary.diversification_condition is None
    assert summary.context_mode is ContextMode.WITHOUT_CONTEXT
    assert summary.mean == pytest.approx(0.3)


def test_no_context_observation_rejects_retrieval_metadata():
    with pytest.raises(ValueError, match="retrieval metadata=None"):
        _record(
            "q1",
            context_mode=ContextMode.WITHOUT_CONTEXT,
            retriever="bm25",
        )


@pytest.mark.parametrize(
    "missing_field", ["retriever", "diversification_condition", "retrieval_artifact_id"]
)
def test_with_context_generation_requires_retrieval_metadata(missing_field):
    with pytest.raises(ValueError, match="retrieval metadata must be non-empty"):
        _record("q1", **{missing_field: None})


@pytest.mark.parametrize(
    "missing_field", ["retriever", "diversification_condition", "retrieval_artifact_id"]
)
def test_retrieval_observation_requires_retrieval_metadata(missing_field):
    with pytest.raises(ValueError, match="retrieval metadata must be non-empty"):
        _record("q1", metric_id="recall_at_k", **{missing_field: None})


def test_retrieval_observation_accepts_only_retrieval_identity():
    observation = _record("q1", metric_id="recall_at_k")
    assert observation.model_provider is None
    assert observation.model_id is None
    assert observation.model_revision is None
    assert observation.context_mode is None
    assert observation.generation_replica is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_provider", "provider"),
        ("model_id", "model-a"),
        ("model_revision", "rev-1"),
        ("context_mode", ContextMode.WITH_CONTEXT),
        ("generation_replica", 0),
    ],
)
def test_retrieval_observation_rejects_generation_identity(field_name, value):
    with pytest.raises(ValueError, match="generation metadata=None"):
        _record("q1", metric_id="recall_at_k", **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "expected_error"),
    [
        ("model_provider", ValueError),
        ("model_id", ValueError),
        ("context_mode", TypeError),
        ("generation_replica", ValueError),
    ],
)
def test_generation_observation_requires_generation_identity(
    field_name, expected_error
):
    with pytest.raises(expected_error):
        _record("q1", **{field_name: None})


def test_same_canonical_no_context_records_are_reused_across_conditions():
    no_context = [
        _record("q1", 0.2, context_mode=ContextMode.WITHOUT_CONTEXT),
        _record("q2", 0.4, context_mode=ContextMode.WITHOUT_CONTEXT),
    ]
    baseline = [_record("q1", 0.5), _record("q2", 0.6)]
    diversified = [
        _record("q1", 0.7, diversification_condition="mmr_0.5"),
        _record("q2", 0.8, diversification_condition="mmr_0.5"),
    ]

    baseline_summary = compare_context_modes(baseline, no_context, ["q1", "q2"])
    diversified_summary = compare_context_modes(
        diversified, no_context, ["q1", "q2"]
    )

    assert baseline_summary.diversification_condition == "none"
    assert diversified_summary.diversification_condition == "mmr_0.5"
    assert all(record.retriever is None for record in no_context)
    assert all(record.retrieval_artifact_id is None for record in no_context)


def test_generation_context_metric_cannot_be_context_compared():
    with pytest.raises(ValueError, match="only generation-scope"):
        compare_context_modes(
            [
                _record(
                    "q1",
                    metric_id="faithfulness_to_context",
                    status=MetricStatus.NOT_COMPUTED,
                )
            ],
            [
                _record(
                    "q1",
                    metric_id="faithfulness_to_context",
                    status=MetricStatus.NOT_APPLICABLE,
                    context_mode=ContextMode.WITHOUT_CONTEXT,
                )
            ],
            ["q1"],
        )


def test_summaries_are_self_describing():
    generation = aggregate_metric_group([_record("q1")], ["q1"])
    assert generation.metric_scope.value == "generation"
    assert generation.dataset_id is DatasetId.PUBMEDQA
    assert generation.retriever == "contriever"
    assert generation.model_id == "model-a"
    assert generation.context_mode is ContextMode.WITH_CONTEXT

    paired = compare_context_modes(
        [_record("q1")],
        [_record("q1", context_mode=ContextMode.WITHOUT_CONTEXT)],
        ["q1"],
    )
    assert paired.metric_scope.value == "generation"
    assert paired.dataset_id is DatasetId.PUBMEDQA
    assert paired.retriever == "contriever"
    assert paired.diversification_condition == "none"
    assert paired.model_provider == "provider"
