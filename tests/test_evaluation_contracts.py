"""CPU-only tests for the Sprint 3 evaluation data contract."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.contracts import (
    SCHEMA_VERSION,
    ComponentStatus,
    ContextMode,
    MetricResult,
    MetricStatus,
    ReasonCode,
    RowIdentity,
    metric_result_to_csv,
    metric_result_to_json,
)


def _identity(**overrides):
    values = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-001",
        "dataset_id": "dataset",
        "dataset_split": "test",
        "sample_id": "sample-001",
        "retrieval_artifact_id": "retrieval-001",
        "model_provider": "provider",
        "model_id": "model-a",
        "model_revision": "revision-1",
        "context_mode": ContextMode.WITH_CONTEXT,
        "generation_replica": 0,
    }
    values.update(overrides)
    return RowIdentity(**values)


@pytest.mark.parametrize("value", [0.0, 1.25, -2.5, 3, np.float32(0.5)])
def test_measured_finite_real_values_are_valid(value):
    result = MetricResult(value=value, status=MetricStatus.MEASURED)
    assert result.value == value


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, True, False, np.bool_(True), "0.5"],
)
def test_measured_invalid_values_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        MetricResult(value=value, status=MetricStatus.MEASURED)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (MetricStatus.NOT_APPLICABLE, ReasonCode.NOT_DEFINED_FOR_DATASET),
        (MetricStatus.NOT_COMPUTED, ReasonCode.EVALUATOR_NOT_CONFIGURED),
        (MetricStatus.FAILED, ReasonCode.METRIC_EXCEPTION),
    ],
)
def test_non_measured_statuses_require_none_value(status, reason):
    result = MetricResult(value=None, status=status, reason=reason)
    assert result.value is None


@pytest.mark.parametrize(
    "status",
    [MetricStatus.NOT_APPLICABLE, MetricStatus.NOT_COMPUTED, MetricStatus.FAILED],
)
def test_non_measured_statuses_reject_numeric_zero(status):
    with pytest.raises(ValueError):
        MetricResult(value=0.0, status=status, reason=ReasonCode.METRIC_EXCEPTION)


@pytest.mark.parametrize(
    "status",
    [MetricStatus.NOT_APPLICABLE, MetricStatus.NOT_COMPUTED, MetricStatus.FAILED],
)
def test_non_measured_statuses_require_controlled_reason(status):
    with pytest.raises(ValueError):
        MetricResult(value=None, status=status)
    with pytest.raises(ValueError):
        MetricResult(value=None, status=status, reason="metric_exception")


def test_measured_result_does_not_accept_reason():
    with pytest.raises(ValueError):
        MetricResult(
            value=0.0,
            status=MetricStatus.MEASURED,
            reason=ReasonCode.INVALID_METRIC_INPUT,
        )


def test_unknown_metric_status_is_rejected():
    with pytest.raises(TypeError):
        MetricResult(value=0.0, status="measured")
    with pytest.raises(ValueError):
        MetricStatus("unknown")


def test_unknown_component_status_is_rejected():
    with pytest.raises(ValueError):
        ComponentStatus("unknown")


def test_unknown_context_mode_is_rejected():
    with pytest.raises(ValueError):
        ContextMode("unknown")
    with pytest.raises(TypeError):
        _identity(context_mode="with_context")


def test_json_serialization_distinguishes_unavailable_from_measured_zero():
    unavailable = MetricResult(
        value=None,
        status=MetricStatus.NOT_COMPUTED,
        reason=ReasonCode.EVALUATOR_UNAVAILABLE,
    )
    measured_zero = MetricResult(value=0.0, status=MetricStatus.MEASURED)

    unavailable_json = metric_result_to_json(unavailable)
    measured_json = metric_result_to_json(measured_zero)
    assert unavailable_json["value"] is None
    assert measured_json["value"] == 0.0
    assert json.loads(json.dumps(unavailable_json))["value"] is None
    assert json.loads(json.dumps(measured_json))["value"] == 0.0
    assert unavailable_json["status"] == "not_computed"
    assert unavailable_json["reason"] == "evaluator_unavailable"


def test_csv_projection_distinguishes_unavailable_from_measured_zero():
    unavailable = MetricResult(
        value=None,
        status=MetricStatus.NOT_APPLICABLE,
        reason=ReasonCode.NOT_DEFINED_FOR_CONTEXT_MODE,
    )
    measured_zero = MetricResult(value=0.0, status=MetricStatus.MEASURED)

    unavailable_csv = metric_result_to_csv(unavailable)
    measured_csv = metric_result_to_csv(measured_zero)
    assert math.isnan(unavailable_csv["value"])
    assert measured_csv["value"] == 0.0
    assert unavailable_csv["status"] == "not_applicable"
    assert unavailable_csv["reason"] == "not_defined_for_context_mode"


@pytest.mark.parametrize("schema_version", ["", "sprint3.eval.v2", None])
def test_row_identity_requires_exact_schema_version(schema_version):
    with pytest.raises(ValueError):
        _identity(schema_version=schema_version)


@pytest.mark.parametrize(
    "field_name",
    [
        "run_id",
        "dataset_id",
        "dataset_split",
        "sample_id",
        "model_provider",
        "model_id",
    ],
)
@pytest.mark.parametrize("empty_value", ["", "   ", None])
def test_row_identity_rejects_empty_required_identifiers(field_name, empty_value):
    with pytest.raises(ValueError):
        _identity(**{field_name: empty_value})


def test_row_identity_allows_missing_model_revision():
    identity = _identity(model_revision=None)
    assert identity.model_revision is None


def test_row_identity_enforces_context_specific_retrieval_artifact():
    assert _identity().retrieval_artifact_id == "retrieval-001"
    with pytest.raises(ValueError, match="with_context"):
        _identity(retrieval_artifact_id=None)

    without_context = _identity(
        context_mode=ContextMode.WITHOUT_CONTEXT,
        retrieval_artifact_id=None,
    )
    assert without_context.retrieval_artifact_id is None
    with pytest.raises(ValueError, match="without_context"):
        _identity(context_mode=ContextMode.WITHOUT_CONTEXT)


@pytest.mark.parametrize("generation_replica", [0, 1, 12, np.int64(2)])
def test_generation_replica_accepts_nonnegative_integers(generation_replica):
    assert _identity(generation_replica=generation_replica).generation_replica == generation_replica


@pytest.mark.parametrize("generation_replica", [-1, 0.0, 1.5, True, np.bool_(True)])
def test_generation_replica_rejects_invalid_values(generation_replica):
    with pytest.raises(ValueError):
        _identity(generation_replica=generation_replica)


def test_logical_key_is_deterministic_and_tracks_pairing_dimensions():
    identity = _identity()
    assert identity.logical_key() == _identity().logical_key()
    assert identity.logical_key() != _identity(
        context_mode=ContextMode.WITHOUT_CONTEXT,
        retrieval_artifact_id=None,
    ).logical_key()
    assert identity.logical_key() != _identity(model_id="model-b").logical_key()
    assert identity.logical_key() != _identity(sample_id="sample-002").logical_key()


def test_no_context_logical_key_has_no_fake_retrieval_dependency():
    identity = _identity(
        context_mode=ContextMode.WITHOUT_CONTEXT,
        retrieval_artifact_id=None,
    )
    assert identity.logical_key()[5] is None
    assert "retrieval-001" not in identity.logical_key()
