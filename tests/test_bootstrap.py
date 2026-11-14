"""Tests for the init job (DVC and MLflow are stubbed)."""

import runpy
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from crab_age import bootstrap


@pytest.fixture
def fakes(mocker: MockerFixture) -> dict[str, MagicMock]:
    """Stub dvc, the MLflow client and the training entry point."""
    return {
        "run": mocker.patch("crab_age.bootstrap.subprocess.run"),
        "client": mocker.patch("crab_age.bootstrap.MlflowClient"),
        "version": mocker.patch("crab_age.bootstrap.train.champion_version"),
        "train": mocker.patch("crab_age.bootstrap.train.main", return_value=0),
    }


def test_existing_champion_skips_training(fakes: dict[str, MagicMock]) -> None:
    fakes["version"].return_value = "3"
    assert bootstrap.main([]) == 0
    fakes["run"].assert_called_once_with(["dvc", "pull"], check=True)
    fakes["train"].assert_not_called()


@pytest.mark.parametrize("train_exit_code", [0, 1])
def test_without_champion_trains_and_returns_its_exit_code(
    fakes: dict[str, MagicMock], train_exit_code: int
) -> None:
    fakes["version"].return_value = None
    fakes["train"].return_value = train_exit_code
    assert bootstrap.main(["--params", "params.yaml"]) == train_exit_code
    fakes["train"].assert_called_once_with(["--params", "params.yaml"])


def test_failed_dvc_pull_stops_the_job(fakes: dict[str, MagicMock]) -> None:
    fakes["run"].side_effect = subprocess.CalledProcessError(1, ["dvc", "pull"])
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main([])
    fakes["version"].assert_not_called()


def test_module_entry_point(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # runpy executes the module again, so the stubs go to the modules it imports from.
    mocker.patch("subprocess.run")
    mocker.patch("mlflow.MlflowClient")
    mocker.patch("crab_age.model.train.champion_version", return_value="1")
    monkeypatch.setattr(sys, "argv", ["crab_age.bootstrap"])
    sys.modules.pop("crab_age.bootstrap", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crab_age.bootstrap", run_name="__main__")
    assert exc.value.code == 0
