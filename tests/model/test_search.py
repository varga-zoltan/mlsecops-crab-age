"""Tests for the hyperparameter search over several model types."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from crab_age.model.pipeline import MODEL_FACTORIES, build_pipeline
from crab_age.model.search import (
    Candidate,
    SearchParams,
    read_search_params,
    search_candidates,
    search_model,
    select_best,
)


@pytest.fixture
def search_params() -> SearchParams:
    """A tiny, fast search space for all three model types."""
    return SearchParams(
        cv_folds=2,
        n_iter=2,
        random_state=0,
        models={
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "random_forest": {"n_estimators": [10], "max_depth": [3]},
            "hist_gradient_boosting": {"max_iter": [20], "learning_rate": [0.1]},
        },
    )


def test_read_search_params(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump(
            {"train": {"search": {"cv_folds": 3, "n_iter": 4, "random_state": 7, "models": {"ridge": {"alpha": [1]}}}}}
        ),
        encoding="utf-8",
    )
    assert read_search_params(path) == SearchParams(
        cv_folds=3, n_iter=4, random_state=7, models={"ridge": {"alpha": [1]}}
    )


def test_read_search_params_rejects_empty_model_list(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump({"train": {"search": {"cv_folds": 3, "n_iter": 4, "random_state": 7, "models": {}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one model"):
        read_search_params(path)


def test_project_params_yaml_is_valid() -> None:
    params = read_search_params(Path("params.yaml"))
    assert set(params.models) <= set(MODEL_FACTORIES)
    assert params.cv_folds >= 2


def test_search_model_returns_best_params(X: pd.DataFrame, y: pd.Series, search_params: SearchParams) -> None:
    candidate = search_model("ridge", search_params.models["ridge"], X, y, search_params)
    assert candidate.model_name == "ridge"
    assert candidate.params["alpha"] in {0.1, 1.0, 10.0}
    assert candidate.cv_mae > 0.0
    assert candidate.pipeline.named_steps["model"].alpha == candidate.params["alpha"]
    assert len(candidate.pipeline.predict(X)) == len(X)


def test_search_candidates_covers_every_model(X: pd.DataFrame, y: pd.Series, search_params: SearchParams) -> None:
    candidates = search_candidates(X, y, search_params)
    assert [c.model_name for c in candidates] == list(search_params.models)


def test_search_beats_a_constant_guess(X: pd.DataFrame, y: pd.Series, search_params: SearchParams) -> None:
    best = select_best(search_candidates(X, y, search_params))
    constant_mae = float((y - y.median()).abs().mean())
    assert best.cv_mae < constant_mae


def _candidate(name: str, cv_mae: float) -> Candidate:
    return Candidate(model_name=name, params={}, cv_mae=cv_mae, pipeline=build_pipeline("ridge"))


def test_select_best_picks_lowest_mae() -> None:
    candidates = [_candidate("a", 1.6), _candidate("b", 1.4), _candidate("c", 1.5)]
    assert select_best(candidates).model_name == "b"


def test_select_best_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        select_best([])
