"""Tests for the regression metrics."""

import math

import pytest

from crab_age.model.evaluate import regression_metrics


def test_perfect_predictions() -> None:
    assert regression_metrics([5.0, 10.0, 15.0], [5.0, 10.0, 15.0]) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


@pytest.mark.parametrize(
    ("y_pred", "mae", "rmse"),
    [
        ([6.0, 11.0, 16.0], 1.0, 1.0),
        ([5.0, 10.0, 18.0], 1.0, math.sqrt(3.0)),
    ],
)
def test_known_errors(y_pred: list[float], mae: float, rmse: float) -> None:
    metrics = regression_metrics([5.0, 10.0, 15.0], y_pred)
    assert metrics["mae"] == pytest.approx(mae)
    assert metrics["rmse"] == pytest.approx(rmse)


def test_r2_of_mean_prediction_is_zero() -> None:
    assert regression_metrics([5.0, 10.0, 15.0], [10.0, 10.0, 10.0])["r2"] == pytest.approx(0.0)


def test_metrics_are_plain_floats() -> None:
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert all(type(value) is float for value in metrics.values())


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])
