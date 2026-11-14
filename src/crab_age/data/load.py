"""Load the raw CSV and give the columns clean snake_case names."""

from pathlib import Path

import pandas as pd

from crab_age.data.schema import COLUMNS, RAW_TO_CLEAN_COLUMNS


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw Kaggle CSV and rename its columns to snake_case.

    Raises:
        ValueError: if a required raw column is missing.
    """
    df: pd.DataFrame = pd.read_csv(path)
    missing: list[str] = [col for col in RAW_TO_CLEAN_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    renamed: pd.DataFrame = df.rename(columns=RAW_TO_CLEAN_COLUMNS)
    return renamed.loc[:, COLUMNS]
