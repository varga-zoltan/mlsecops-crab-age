"""Tests for the prediction log table."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from crab_age.data.schema import FEATURES
from crab_age.monitoring.prediction_log import LOG_COLUMNS, PredictionRecord, PredictionStore


def _record(i: int) -> PredictionRecord:
    features: dict[str, object] = {name: 0.5 + i for name in FEATURES}
    features["sex"] = "F"
    return PredictionRecord(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        request_id=f"req-{i}",
        model_version="2",
        features=features,
        prediction=8.0 + i,
    )


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    """Empty prediction log in SQLite."""
    prediction_store = PredictionStore(f"sqlite:///{tmp_path / 'log.db'}")
    prediction_store.init_schema()
    return prediction_store


def test_empty_log_has_columns(store: PredictionStore) -> None:
    log = store.read_all()
    assert log.empty
    assert list(log.columns) == LOG_COLUMNS
    assert store.count_unexported() == 0


def test_insert_and_read(store: PredictionStore) -> None:
    ids = [store.insert(_record(i)) for i in range(3)]
    assert ids == [1, 2, 3]
    log = store.read_all()
    assert list(log["request_id"]) == ["req-0", "req-1", "req-2"]
    assert list(log["prediction"]) == [8.0, 9.0, 10.0]
    assert set(log["sex"]) == {"F"}


def test_mark_exported(store: PredictionStore) -> None:
    for i in range(4):
        store.insert(_record(i))
    store.mark_exported([1, 2], "run-a")
    assert store.count_unexported() == 2
    assert list(store.unexported()["id"]) == [3, 4]
    assert len(store.read_all()) == 4


def test_init_schema_is_idempotent(store: PredictionStore) -> None:
    store.insert(_record(0))
    store.init_schema()
    assert len(store.read_all()) == 1
