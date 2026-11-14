"""Automatic model improvement: hyperparameter search over several model types.

Each model type listed in params.yaml gets its own RandomizedSearchCV with
k-fold cross-validation. The candidate with the lowest CV MAE becomes the
challenger. Nothing here touches the test set: it is kept for the final,
unbiased comparison with the current champion.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from crab_age.model.pipeline import build_pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchParams:
    """Settings of the hyperparameter search, read from params.yaml."""

    cv_folds: int
    n_iter: int
    random_state: int
    models: dict[str, dict[str, list[Any]]]


@dataclass(frozen=True)
class Candidate:
    """The best pipeline found for one model type."""

    model_name: str
    params: dict[str, Any]
    cv_mae: float
    pipeline: Pipeline


def read_search_params(path: Path) -> SearchParams:
    """Read the 'train.search' section of params.yaml."""
    with path.open(encoding="utf-8") as f:
        section: dict[str, Any] = yaml.safe_load(f)["train"]["search"]
    models: dict[str, dict[str, list[Any]]] = {
        name: {key: list(values) for key, values in grid.items()}
        for name, grid in section["models"].items()
    }
    if not models:
        raise ValueError("train.search.models must list at least one model")
    return SearchParams(
        cv_folds=int(section["cv_folds"]),
        n_iter=int(section["n_iter"]),
        random_state=int(section["random_state"]),
        models=models,
    )


def search_model(
    model_name: str, grid: dict[str, list[Any]], X: pd.DataFrame, y: pd.Series, params: SearchParams
) -> Candidate:
    """Run a randomized search for one model type and return its best pipeline."""
    pipeline_grid = {f"model__{key}": values for key, values in grid.items()}
    n_iter = min(params.n_iter, len(ParameterGrid(pipeline_grid)))
    search = RandomizedSearchCV(
        estimator=build_pipeline(model_name, params.random_state),
        param_distributions=pipeline_grid,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=params.cv_folds, shuffle=True, random_state=params.random_state),
        random_state=params.random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    candidate = Candidate(
        model_name=model_name,
        params=best_params,
        cv_mae=-float(search.best_score_),
        pipeline=search.best_estimator_,
    )
    logger.info(
        "search finished",
        extra={"model_name": model_name, "cv_mae": candidate.cv_mae, "best_params": best_params, "n_iter": n_iter},
    )
    return candidate


def search_candidates(X: pd.DataFrame, y: pd.Series, params: SearchParams) -> list[Candidate]:
    """Search every model type of ``params`` and return one candidate each."""
    return [search_model(name, grid, X, y, params) for name, grid in params.models.items()]


def select_best(candidates: list[Candidate]) -> Candidate:
    """Return the candidate with the lowest cross-validated MAE."""
    if not candidates:
        raise ValueError("no candidates to select from")
    return min(candidates, key=lambda candidate: candidate.cv_mae)
