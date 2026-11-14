"""Shared fixtures for the data tests."""

from pathlib import Path

import pandas as pd
import pytest

from crab_age.data.schema import RAW_TO_CLEAN_COLUMNS


def _valid_row(sex: str, scale: float, age: int) -> dict[str, object]:
    return {
        "sex": sex,
        "length": 1.3 * scale,
        "diameter": 1.0 * scale,
        "height": 0.35 * scale,
        "weight": 23.0 * scale,
        "shucked_weight": 10.0 * scale,
        "viscera_weight": 5.0 * scale,
        "shell_weight": 6.5 * scale,
        "age": age,
    }


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Ten plausible rows with clean column names."""
    rows: list[dict[str, object]] = [
        _valid_row(sex, 0.8 + 0.05 * i, 5 + i) for i, sex in enumerate("MFIMFIMFIM")
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv(tmp_path: Path, clean_df: pd.DataFrame) -> Path:
    """The clean_df rows written as a raw Kaggle-style CSV."""
    clean_to_raw: dict[str, str] = {v: k for k, v in RAW_TO_CLEAN_COLUMNS.items()}
    path: Path = tmp_path / "raw.csv"
    clean_df.rename(columns=clean_to_raw).to_csv(path, index=False)
    return path
