"""Tests for the training job.

The end-to-end tests use a throw-away local MLflow store (SQLite in tmp_path):
real MLflow code, but no MLflow server and no Docker. Error paths that need a
broken server are tested with a mocked MlflowClient.
"""

import importlib
import runpy
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import create_autospec

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from pytest_mock import MockerFixture

from crab_age.data.schema import FEATURES, TARGET
from crab_age.model.promote import GateDecision
from crab_age.model.search import SearchParams
from crab_age.model.train import (
    CHAMPION_ALIAS,
    TrainParams,
    TrainResult,
    champion_mae,
    main,
    read_train_params,
    run,
    split_xy,
)

MODEL_NAME = "crab-age-test"
GOOD_MODELS: dict[str, dict[str, list[object]]] = {
    "ridge": {"alpha": [0.1, 1.0]},
    "hist_gradient_boosting": {"max_iter": [20]},
}
BAD_MODELS: dict[str, dict[str, list[object]]] = {"ridge": {"alpha": [1e6]}}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[MlflowClient]:
    """Point MLflow at an empty SQLite store inside tmp_path."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def data_paths(tmp_path: Path, train_df: pd.DataFrame) -> tuple[Path, Path]:
    """train.csv and test.csv written from the synthetic crabs."""
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train_df.iloc[:45].to_csv(train_path, index=False)
    train_df.iloc[45:].to_csv(test_path, index=False)
    return train_path, test_path


def _params(data_paths: tuple[Path, Path], models: dict[str, dict[str, list[object]]], max_mae: float) -> TrainParams:
    return TrainParams(
        train_path=data_paths[0],
        test_path=data_paths[1],
        experiment_name="crab-age-test",
        registered_model_name=MODEL_NAME,
        max_mae=max_mae,
        search=SearchParams(cv_folds=2, n_iter=2, random_state=0, models=models),
    )


def test_first_run_becomes_champion(client: MlflowClient, data_paths: tuple[Path, Path]) -> None:
    result = run(_params(data_paths, GOOD_MODELS, max_mae=5.0))

    assert result.decision.accepted
    assert result.decision.champion_mae is None
    assert result.model_version == "1"
    assert client.get_registered_model(MODEL_NAME).aliases == {CHAMPION_ALIAS: 1}

    parent = client.get_run(result.run_id)
    assert {"git_sha", "dvc_data_hash", "dvc_raw_hash"} <= set(parent.data.tags)
    assert parent.data.tags["gate"] == "accepted"
    assert {"cv_mae", "test_mae", "test_rmse", "test_r2"} <= set(parent.data.metrics)
    nested = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{result.run_id}'"
    )
    assert sorted(r.data.tags["model_name"] for r in nested) == sorted(GOOD_MODELS)


def test_champion_loads_with_signature(client: MlflowClient, data_paths: tuple[Path, Path]) -> None:
    run(_params(data_paths, GOOD_MODELS, max_mae=5.0))
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    signature = model.metadata.signature
    assert signature is not None
    assert signature.inputs.input_names() == FEATURES
    X, _ = split_xy(pd.read_csv(data_paths[1]))
    assert np.isfinite(model.predict(X)).all()


def test_worse_challenger_is_rejected(client: MlflowClient, data_paths: tuple[Path, Path]) -> None:
    first = run(_params(data_paths, GOOD_MODELS, max_mae=5.0))
    second = run(_params(data_paths, BAD_MODELS, max_mae=5.0))

    assert not second.decision.accepted
    assert second.decision.champion_mae == pytest.approx(first.test_metrics["mae"])
    assert "worse than champion" in second.decision.reason
    assert client.get_registered_model(MODEL_NAME).aliases == {CHAMPION_ALIAS: 1}
    assert client.get_model_version(MODEL_NAME, second.model_version).tags["gate"] == "rejected"
    assert client.get_run(second.run_id).data.metrics["champion_test_mae"] == pytest.approx(first.test_metrics["mae"])


def test_first_run_above_limit_gets_no_champion(client: MlflowClient, data_paths: tuple[Path, Path]) -> None:
    result = run(_params(data_paths, GOOD_MODELS, max_mae=1e-6))
    assert not result.decision.accepted
    assert "above the limit" in result.decision.reason
    assert client.get_registered_model(MODEL_NAME).aliases == {}


def test_champion_mae_without_alias(client: MlflowClient, train_df: pd.DataFrame) -> None:
    client.create_registered_model(MODEL_NAME)
    X, y = split_xy(train_df)
    assert champion_mae(client, MODEL_NAME, X, y) is None


def test_champion_mae_missing_model() -> None:
    fake = create_autospec(MlflowClient, instance=True)
    fake.get_registered_model.side_effect = MlflowException("no model", error_code=RESOURCE_DOES_NOT_EXIST)
    assert champion_mae(fake, MODEL_NAME, pd.DataFrame(), pd.Series(dtype=float)) is None


def test_champion_mae_server_error_is_raised() -> None:
    fake = create_autospec(MlflowClient, instance=True)
    fake.get_registered_model.side_effect = MlflowException("server down", error_code=INTERNAL_ERROR)
    with pytest.raises(MlflowException, match="server down"):
        champion_mae(fake, MODEL_NAME, pd.DataFrame(), pd.Series(dtype=float))


def test_split_xy(train_df: pd.DataFrame) -> None:
    X, y = split_xy(train_df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert y.dtype == float


def test_read_project_params() -> None:
    params = read_train_params(Path("params.yaml"))
    assert params.max_mae > 0
    assert params.train_path.name == "train.csv"
    assert params.test_path.name == "test.csv"
    assert params.search.models


def _result(accepted: bool) -> TrainResult:
    decision = GateDecision(accepted=accepted, reason="r", challenger_mae=1.0, champion_mae=None)
    return TrainResult(run_id="x", model_name="ridge", model_version="1", test_metrics={}, decision=decision)


@pytest.mark.parametrize(("accepted", "exit_code"), [(True, 0), (False, 1)])
def test_main_exit_code(mocker: MockerFixture, tmp_path: Path, accepted: bool, exit_code: int) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text(Path("params.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    fake_run = mocker.patch("crab_age.model.train.run", return_value=_result(accepted))
    assert main(["--params", str(params_file)]) == exit_code
    assert fake_run.call_args.args[0].registered_model_name == yaml.safe_load(
        params_file.read_text(encoding="utf-8")
    )["train"]["registered_model_name"]


def test_module_import_does_not_run_training(mocker: MockerFixture) -> None:
    fake_main = mocker.patch("crab_age.model.train.main")
    sys.modules.pop("crab_age.train", None)
    importlib.import_module("crab_age.train")
    fake_main.assert_not_called()


def test_module_entry_point_exits_with_main_code(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    mocker.patch("crab_age.model.train.main", return_value=1)
    monkeypatch.setattr(sys, "argv", ["crab_age.train"])
    sys.modules.pop("crab_age.train", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crab_age.train", run_name="__main__")
    assert exc.value.code == 1
