"""Tests for the sklearn Pipeline builder."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from crab_age.model.pipeline import MODEL_FACTORIES, build_pipeline, build_preprocessor


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
def test_every_model_fits_and_predicts(model_name: str, X: pd.DataFrame, y: pd.Series) -> None:
    pipeline = build_pipeline(model_name, random_state=0).fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_pipeline_steps_contain_preprocessing() -> None:
    pipeline = build_pipeline("ridge")
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["preprocess"], ColumnTransformer)


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown model 'svm'"):
        build_pipeline("svm")


def test_preprocessor_scales_and_one_hot_encodes(X: pd.DataFrame) -> None:
    out = build_preprocessor().fit_transform(X)
    assert out.shape == (len(X), 3 + 7)
    one_hot = out[:, :3]
    assert set(np.unique(one_hot)) <= {0.0, 1.0}
    assert np.allclose(one_hot.sum(axis=1), 1.0)
    assert np.allclose(out[:, 3:].mean(axis=0), 0.0)


def test_unknown_sex_at_serving_time_does_not_crash(X: pd.DataFrame, y: pd.Series) -> None:
    pipeline = build_pipeline("ridge").fit(X, y)
    row = X.iloc[[0]].copy()
    row["sex"] = "X"
    assert np.isfinite(pipeline.predict(row)).all()


def test_same_random_state_gives_same_model(X: pd.DataFrame, y: pd.Series) -> None:
    first = build_pipeline("random_forest", random_state=1).fit(X, y).predict(X)
    second = build_pipeline("random_forest", random_state=1).fit(X, y).predict(X)
    assert np.array_equal(first, second)
