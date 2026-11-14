"""Tests for the drift report and the dashboard data."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crab_age.data.schema import FEATURES, NUMERIC_FEATURES, TARGET
from crab_age.monitoring.drift import (
    TRAFFIC_HINT,
    build_drift_report,
    dashboard_data,
    load_reference,
    parse_drift_metrics,
    summarize,
)
from crab_age.monitoring.prediction_log import PredictionRecord, PredictionStore
from crab_age.monitoring.simulate_traffic import Mode, generate_crab


def _crabs(n: int, mode: Mode, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([generate_crab(rng, mode) for _ in range(n)])


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    """Reference data drawn like the training data."""
    return _crabs(300, "normal", seed=1)


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    """Empty prediction log in SQLite."""
    prediction_store = PredictionStore(f"sqlite:///{tmp_path / 'log.db'}")
    prediction_store.init_schema()
    return prediction_store


def _log(store: PredictionStore, crabs: pd.DataFrame, version: str = "1") -> None:
    for i, row in enumerate(crabs.to_dict("records")):
        store.insert(
            PredictionRecord(
                created_at=datetime.now(timezone.utc),
                request_id=f"r{i}",
                model_version=version,
                features=row,
                prediction=8.0 + i % 3,
            )
        )


def test_load_reference_keeps_only_features(tmp_path: Path, reference: pd.DataFrame) -> None:
    path = tmp_path / "train.csv"
    reference.assign(**{TARGET: 9}).to_csv(path, index=False)
    assert list(load_reference(path).columns) == FEATURES


def test_no_drift_on_same_distribution(reference: pd.DataFrame) -> None:
    result = build_drift_report(reference, _crabs(200, "normal", seed=2))
    assert result.drifted_columns <= 1
    assert result.dataset_drift is False
    assert set(result.column_scores) == set(FEATURES)
    assert "<html" in result.html.lower()


def test_drift_mode_is_detected(reference: pd.DataFrame) -> None:
    result = build_drift_report(reference, _crabs(200, "drift", seed=3))
    assert result.drifted_columns >= len(NUMERIC_FEATURES) // 2
    assert result.dataset_drift is True
    assert result.drift_share == pytest.approx(result.drifted_columns / len(FEATURES))
    assert "html" not in repr(result)


def test_summarize_empty_and_filled() -> None:
    empty = summarize(pd.DataFrame(columns=["model_version", "prediction"]), total_predictions=0)
    assert (empty.window_rows, empty.model_versions, empty.mean_predicted_age) == (0, [], None)
    log = pd.DataFrame({"model_version": ["2", "1", "2"], "prediction": [6.0, 9.0, 12.0]})
    filled = summarize(log, total_predictions=10)
    assert filled.model_versions == ["1", "2"]
    assert filled.mean_predicted_age == 9.0
    assert filled.total_predictions == 10


def test_dashboard_data_with_too_few_rows(store: PredictionStore, reference: pd.DataFrame) -> None:
    _log(store, _crabs(5, "normal", seed=4))
    data = dashboard_data(store, reference, window=100, min_rows=30)
    assert data.drift is None
    assert data.message is not None
    assert "5 of at least 30" in data.message
    assert TRAFFIC_HINT in data.message
    assert data.summary.total_predictions == 5


def test_dashboard_data_uses_latest_window(store: PredictionStore, reference: pd.DataFrame) -> None:
    _log(store, _crabs(40, "normal", seed=5), version="1")
    _log(store, _crabs(40, "drift", seed=6), version="2")
    data = dashboard_data(store, reference, window=40, min_rows=30)
    assert data.message is None
    assert data.drift is not None
    assert data.summary.window_rows == 40
    assert data.summary.total_predictions == 80
    assert data.summary.model_versions == ["2"]
    assert data.drift.drifted_columns >= len(NUMERIC_FEATURES) // 2


def test_report_ignores_extra_log_columns(reference: pd.DataFrame) -> None:
    current = _crabs(50, "normal", seed=7).assign(prediction=1.0, request_id="x")
    assert set(build_drift_report(reference, current).column_scores) == set(FEATURES)


def test_parse_drift_metrics_ignores_other_metric_types() -> None:
    metrics = [
        {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 2.0, "share": 0.25}},
        {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "length"}, "value": 0.4},
        {"config": {"type": "evidently:metric_v2:RowCount"}, "value": 100},
    ]
    assert parse_drift_metrics(metrics) == (2, 0.25, {"length": 0.4})
