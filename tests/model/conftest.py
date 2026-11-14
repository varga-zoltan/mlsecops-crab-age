"""Shared fixtures for the model tests."""

import numpy as np
import pandas as pd
import pytest

from crab_age.data.schema import FEATURES, TARGET


@pytest.fixture
def train_df() -> pd.DataFrame:
    """60 synthetic crabs: age grows with size, plus a little noise."""
    rng = np.random.default_rng(0)
    n = 60
    size = rng.uniform(0.5, 1.5, n)
    df = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F", "I"], n),
            "length": 1.3 * size,
            "diameter": 1.0 * size,
            "height": 0.35 * size,
            "weight": 23.0 * size,
            "shucked_weight": 10.0 * size,
            "viscera_weight": 5.0 * size,
            "shell_weight": 6.5 * size,
        }
    )
    df[TARGET] = 4.0 + 8.0 * size + rng.normal(0.0, 0.3, n)
    return df


@pytest.fixture
def X(train_df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns of train_df."""
    return train_df.loc[:, FEATURES]


@pytest.fixture
def y(train_df: pd.DataFrame) -> pd.Series:
    """Target column of train_df."""
    return train_df[TARGET]
