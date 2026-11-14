"""Tests for crab_age.data.load."""

from pathlib import Path

import pandas as pd
import pytest

from crab_age.data.load import load_raw
from crab_age.data.schema import COLUMNS


def test_load_raw_renames_columns(raw_csv: Path) -> None:
    df: pd.DataFrame = load_raw(raw_csv)
    assert list(df.columns) == COLUMNS
    assert len(df) == 10


def test_load_raw_drops_extra_columns(tmp_path: Path, raw_csv: Path) -> None:
    raw: pd.DataFrame = pd.read_csv(raw_csv)
    raw["Extra"] = 1
    path: Path = tmp_path / "extra.csv"
    raw.to_csv(path, index=False)
    assert list(load_raw(path).columns) == COLUMNS


@pytest.mark.parametrize("dropped", ["Sex", "Shucked Weight", "Age"])
def test_load_raw_missing_column_raises(tmp_path: Path, raw_csv: Path, dropped: str) -> None:
    path: Path = tmp_path / "broken.csv"
    pd.read_csv(raw_csv).drop(columns=[dropped]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=dropped):
        load_raw(path)
