"""Training job: search, evaluate, log to MLflow, register, and gate.

Contract (artifact + exit code):

* every run logs a parent MLflow run with one nested run per model type,
* the best candidate (the challenger) is registered with its signature,
* the challenger gets the ``champion`` alias only if it passes the
  acceptance gate; otherwise the old champion stays and the exit code is 1.

Run with: python -m crab_age.train --params params.yaml
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from crab_age.data.schema import FEATURES, TARGET
from crab_age.lineage import lineage_tags
from crab_age.logging_setup import setup_logging
from crab_age.model.evaluate import regression_metrics
from crab_age.model.promote import GateDecision, acceptance_gate
from crab_age.model.search import Candidate, SearchParams, read_search_params, search_model, select_best

logger = logging.getLogger(__name__)

CHAMPION_ALIAS: Final = "champion"

# The model is saved with skops (not pickle): loading it cannot run arbitrary code.
# skops only loads types on an allow-list. These two tree types are needed by the
# random forest and the gradient boosting model; we create these files ourselves.
SKOPS_TRUSTED_TYPES: Final[list[str]] = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


@dataclass(frozen=True)
class TrainParams:
    """Settings of the training job, read from params.yaml."""

    train_path: Path
    test_path: Path
    experiment_name: str
    registered_model_name: str
    max_mae: float
    search: SearchParams


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one training run."""

    run_id: str
    model_name: str
    model_version: str
    test_metrics: dict[str, float]
    decision: GateDecision


def read_train_params(path: Path) -> TrainParams:
    """Read the 'train' section of params.yaml."""
    with path.open(encoding="utf-8") as f:
        section: dict[str, Any] = yaml.safe_load(f)["train"]
    return TrainParams(
        train_path=Path(section["train_path"]),
        test_path=Path(section["test_path"]),
        experiment_name=str(section["experiment_name"]),
        registered_model_name=str(section["registered_model_name"]),
        max_mae=float(section["gate"]["max_mae"]),
        search=read_search_params(path),
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed data frame into features and target."""
    return df.loc[:, FEATURES], pd.Series(df[TARGET], dtype=float, name=TARGET)


def champion_version(client: MlflowClient, model_name: str) -> str | None:
    """Return the registry version that has the champion alias, or None.

    Any MLflow error other than "model does not exist" is raised: an
    unreachable server must never look like "no champion".
    """
    try:
        registered = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise
    version = registered.aliases.get(CHAMPION_ALIAS)
    return None if version is None else str(version)


def champion_mae(client: MlflowClient, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """Evaluate the current champion on the test set; None if there is no champion.

    The champion is re-evaluated on the *current* test set, so the comparison is
    fair even if the data changed since the champion was trained.
    """
    version = champion_version(client, model_name)
    if version is None:
        return None
    champion = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    return regression_metrics(y_test, champion.predict(X_test))["mae"]


def _search_with_nested_runs(X: pd.DataFrame, y: pd.Series, params: SearchParams) -> list[Candidate]:
    candidates: list[Candidate] = []
    for model_name, grid in params.models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            candidate = search_model(model_name, grid, X, y, params)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(candidate.params)
            mlflow.log_metric("cv_mae", candidate.cv_mae)
        candidates.append(candidate)
    return candidates


def run(params: TrainParams) -> TrainResult:
    """Train, log, register and gate one challenger. Returns the outcome."""
    train_df = pd.read_csv(params.train_path)
    test_df = pd.read_csv(params.test_path)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    client = MlflowClient()
    mlflow.set_experiment(params.experiment_name)
    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags(lineage_tags())
        mlflow.log_input(from_pandas(train_df, targets=TARGET, name="train"), "training")
        mlflow.log_input(from_pandas(test_df, targets=TARGET, name="test"), "testing")
        mlflow.log_params(
            {
                "cv_folds": params.search.cv_folds,
                "n_iter": params.search.n_iter,
                "random_state": params.search.random_state,
                "model_types": ",".join(params.search.models),
                "gate_max_mae": params.max_mae,
                "train_path": str(params.train_path),
                "test_path": str(params.test_path),
            }
        )

        best = select_best(_search_with_nested_runs(X_train, y_train, params.search))
        predictions = best.pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, predictions)
        mlflow.log_params({"best_model": best.model_name, **{f"best_{k}": v for k, v in best.params.items()}})
        mlflow.log_metrics({"cv_mae": best.cv_mae, **{f"test_{k}": v for k, v in test_metrics.items()}})

        current_mae = champion_mae(client, params.registered_model_name, X_test, y_test)
        model_info = mlflow_sklearn.log_model(
            best.pipeline,
            name="model",
            signature=infer_signature(X_test, predictions),
            input_example=X_train.head(5),
            registered_model_name=params.registered_model_name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = str(model_info.registered_model_version)

        decision = acceptance_gate(test_metrics["mae"], current_mae, params.max_mae)
        if current_mae is not None:
            mlflow.log_metric("champion_test_mae", current_mae)
        gate = "accepted" if decision.accepted else "rejected"
        mlflow.set_tags({"gate": gate, "gate_reason": decision.reason})
        client.set_model_version_tag(params.registered_model_name, version, "gate", gate)
        if decision.accepted:
            client.set_registered_model_alias(params.registered_model_name, CHAMPION_ALIAS, version)

    logger.info(
        "training finished",
        extra={
            "run_id": parent.info.run_id,
            "best_model": best.model_name,
            "model_version": version,
            "test_metrics": test_metrics,
            "gate": gate,
            "gate_reason": decision.reason,
        },
    )
    return TrainResult(
        run_id=parent.info.run_id,
        model_name=best.model_name,
        model_version=version,
        test_metrics=test_metrics,
        decision=decision,
    )


def main(argv: list[str] | None = None) -> int:
    """Command line entry point. Exit code 0: new champion, 1: rejected by the gate."""
    parser = argparse.ArgumentParser(description="Train, register and gate a crab age model.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    setup_logging()
    result = run(read_train_params(args.params))
    return 0 if result.decision.accepted else 1
