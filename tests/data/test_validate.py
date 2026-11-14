"""Tests for crab_age.data.validate."""

import pandas as pd
import pytest

from crab_age.data.validate import DataValidationError, invalid_row_masks, validate_schema


def test_validate_schema_accepts_valid_data(clean_df: pd.DataFrame) -> None:
    validate_schema(clean_df)


def test_validate_schema_missing_column(clean_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate_schema(clean_df.drop(columns=["height"]))


def test_validate_schema_missing_value(clean_df: pd.DataFrame) -> None:
    clean_df.loc[0, "weight"] = None
    with pytest.raises(DataValidationError, match="missing values"):
        validate_schema(clean_df)


@pytest.mark.parametrize("column", ["length", "age"])
def test_validate_schema_non_numeric(clean_df: pd.DataFrame, column: str) -> None:
    clean_df[column] = clean_df[column].astype(str)
    with pytest.raises(DataValidationError, match=column):
        validate_schema(clean_df)


@pytest.mark.parametrize("bad_sex", ["X", "m", ""])
def test_validate_schema_unknown_sex(clean_df: pd.DataFrame, bad_sex: str) -> None:
    clean_df.loc[0, "sex"] = bad_sex
    with pytest.raises(DataValidationError, match="Unknown sex"):
        validate_schema(clean_df)


def test_invalid_row_masks_valid_data_has_no_hits(clean_df: pd.DataFrame) -> None:
    masks: dict[str, pd.Series] = invalid_row_masks(clean_df, max_height_to_diameter=0.8)
    assert all(int(mask.sum()) == 0 for mask in masks.values())


@pytest.mark.parametrize(
    ("column", "value", "rule"),
    [
        ("height", 0.0, "non_positive_value"),
        ("weight", -1.0, "non_positive_value"),
        ("height", 2.8, "height_to_diameter_too_large"),
        ("shucked_weight", 100.0, "shucked_weight_above_weight"),
        ("diameter", 5.0, "diameter_above_length"),
    ],
)
def test_invalid_row_masks_flags_bad_row(clean_df: pd.DataFrame, column: str, value: float, rule: str) -> None:
    clean_df.loc[3, column] = value
    mask: pd.Series = invalid_row_masks(clean_df, max_height_to_diameter=0.8)[rule]
    assert mask.tolist() == [i == 3 for i in range(len(clean_df))]
