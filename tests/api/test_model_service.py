"""Tests for loading the champion model and the prediction service."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from crab_age.api.model_service import (
    LoadedModel,
    ModelHolder,
    ModelNotLoadedError,
    PredictionService,
    load_champion,
)
from crab_age.api.schemas import CrabFeatures
from crab_age.model.train import CHAMPION_ALIAS, SKOPS_TRUSTED_TYPES
from crab_age.monitoring.prediction_log import PredictionStore


@pytest.fixture
def mlflow_client(tmp_path: Path) -> Iterator[MlflowClient]:
    """Empty local MLflow store (SQLite), no server."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def test_load_champion_from_registry(
    mlflow_client: MlflowClient, fitted_pipeline: Pipeline, valid_crab: dict[str, object]
) -> None:
    for _ in range(2):
        with mlflow.start_run():
            mlflow_sklearn.log_model(
                fitted_pipeline, name="model", registered_model_name="crab-age", skops_trusted_types=SKOPS_TRUSTED_TYPES
            )
    mlflow_client.set_registered_model_alias("crab-age", CHAMPION_ALIAS, "1")

    loaded = load_champion("crab-age")

    assert loaded.version == "1"
    store = PredictionStore("sqlite://")
    store.init_schema()
    holder = ModelHolder("crab-age", loader=lambda _: loaded)
    holder.reload()
    result = PredictionService(holder, store).predict(CrabFeatures.model_validate(valid_crab))
    assert result.model_version == "1"


def test_load_champion_without_alias_fails(mlflow_client: MlflowClient) -> None:
    mlflow_client.create_registered_model("crab-age")
    with pytest.raises(mlflow.exceptions.MlflowException):
        load_champion("crab-age")


def test_failed_reload_keeps_previous_model(fitted_pipeline: Pipeline) -> None:
    versions = iter(["1", "boom"])

    def loader(_: str) -> LoadedModel:
        version = next(versions)
        if version == "boom":
            raise RuntimeError("cannot load")
        return LoadedModel(model=fitted_pipeline, version=version)

    holder = ModelHolder("crab-age", loader=loader)
    assert holder.reload() is True
    assert holder.reload() is False
    assert holder.current is not None
    assert holder.current.version == "1"


def test_predict_without_model_raises(valid_crab: dict[str, object]) -> None:
    store = PredictionStore("sqlite://")
    store.init_schema()
    service = PredictionService(ModelHolder("crab-age"), store)
    with pytest.raises(ModelNotLoadedError):
        service.predict(CrabFeatures.model_validate(valid_crab))
