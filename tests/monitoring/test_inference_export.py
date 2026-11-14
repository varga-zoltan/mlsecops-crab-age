"""Tests for exporting prediction batches to MLflow."""

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from pytest_mock import MockerFixture

from crab_age.data.schema import FEATURES
from crab_age.monitoring.inference_export import InferenceExporter
from crab_age.monitoring.prediction_log import PredictionRecord, PredictionStore

EXPERIMENT = "crab-age-inference-test"


@pytest.fixture
def mlflow_client(tmp_path: Path) -> Iterator[MlflowClient]:
    """Empty local MLflow store (SQLite), no server."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    """Empty prediction log in SQLite."""
    prediction_store = PredictionStore(f"sqlite:///{tmp_path / 'log.db'}")
    prediction_store.init_schema()
    return prediction_store


def _add(store: PredictionStore, n: int) -> None:
    for i in range(n):
        features: dict[str, object] = {name: 1.0 for name in FEATURES}
        features["sex"] = "I"
        store.insert(
            PredictionRecord(
                created_at=datetime.now(timezone.utc),
                request_id=f"r{i}",
                model_version="4",
                features=features,
                prediction=float(i),
            )
        )


def test_no_export_below_batch_size(store: PredictionStore, mlflow_client: MlflowClient) -> None:
    _add(store, 2)
    assert InferenceExporter(store, EXPERIMENT, batch_size=3).export_ready() is None
    assert mlflow_client.get_experiment_by_name(EXPERIMENT) is None


def test_full_batch_is_logged_as_artifact(store: PredictionStore, mlflow_client: MlflowClient, tmp_path: Path) -> None:
    _add(store, 3)
    exporter = InferenceExporter(store, EXPERIMENT, batch_size=3)

    run_id = exporter.export_ready()

    assert run_id is not None
    run = mlflow_client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["n_predictions"] == 3
    assert run.data.tags["model_versions"] == "4"
    artifact = mlflow_client.download_artifacts(run_id, "predictions/predictions_1_3.csv", str(tmp_path))
    assert list(pd.read_csv(artifact)["request_id"]) == ["r0", "r1", "r2"]
    assert store.count_unexported() == 0
    assert exporter.export_ready() is None


def test_second_batch_reuses_experiment(store: PredictionStore, mlflow_client: MlflowClient) -> None:
    exporter = InferenceExporter(store, EXPERIMENT, batch_size=2)
    _add(store, 2)
    first = exporter.export_ready()
    _add(store, 2)
    second = exporter.export_ready()
    assert first is not None and second is not None and first != second
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None
    assert len(mlflow_client.search_runs([experiment.experiment_id])) == 2


def test_failed_export_keeps_rows_for_retry(
    store: PredictionStore, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    _add(store, 2)
    mocker.patch("crab_age.monitoring.inference_export.MlflowClient", side_effect=RuntimeError("mlflow down"))
    exporter = InferenceExporter(store, EXPERIMENT, batch_size=2)
    assert exporter.export_ready_safely() is None
    assert "inference batch export failed" in caplog.text
    assert store.count_unexported() == 2
