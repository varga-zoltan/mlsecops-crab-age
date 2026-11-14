"""Regression metrics used for model selection and the acceptance gate."""

import math

from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and R² of the predictions.

    MAE (mean absolute error, in months) is the main metric: it decides model
    selection and the acceptance gate.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }
