"""The sklearn Pipeline: preprocessing + regressor in one object.

Scaling and one-hot encoding live inside the Pipeline, so the exact same
transformation runs at training and at serving time (no training/serving skew,
no separate scaler file).
"""

from collections.abc import Callable
from typing import Any, Final

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crab_age.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_FACTORIES: Final[dict[str, Callable[[int], Any]]] = {
    "ridge": lambda random_state: Ridge(random_state=random_state),
    "random_forest": lambda random_state: RandomForestRegressor(random_state=random_state),
    "hist_gradient_boosting": lambda random_state: HistGradientBoostingRegressor(random_state=random_state),
}


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode 'sex' and standardize the numeric measurements."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Return an unfitted Pipeline with the preprocessor and the named regressor.

    Raises ValueError for an unknown model name.
    """
    if model_name not in MODEL_FACTORIES:
        known = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"unknown model {model_name!r}, expected one of: {known}")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", MODEL_FACTORIES[model_name](random_state)),
        ]
    )
