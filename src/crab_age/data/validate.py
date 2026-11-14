"""Schema checks and row-level plausibility rules."""

import pandas as pd

from crab_age.data.schema import COLUMNS, NUMERIC_FEATURES, SEX_VALUES, TARGET


class DataValidationError(ValueError):
    """The data does not match the expected schema."""


def validate_schema(df: pd.DataFrame) -> None:
    """Check columns, missing values, numeric types and allowed categories.

    Raises:
        DataValidationError: on the first problem found.
    """
    missing: list[str] = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if bool(df[COLUMNS].isna().to_numpy().any()):
        raise DataValidationError("Data contains missing values")
    for col in NUMERIC_FEATURES + [TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' is not numeric")
    unknown: set[str] = set(df["sex"].unique()) - SEX_VALUES
    if unknown:
        raise DataValidationError(f"Unknown sex values: {sorted(unknown)}")


def invalid_row_masks(df: pd.DataFrame, max_height_to_diameter: float) -> dict[str, pd.Series]:
    """Return one boolean mask per rule; True marks a physically impossible row."""
    return {
        "non_positive_value": (df[NUMERIC_FEATURES] <= 0).any(axis=1),
        "height_to_diameter_too_large": df["height"] > max_height_to_diameter * df["diameter"],
        "shucked_weight_above_weight": df["shucked_weight"] > df["weight"],
        "diameter_above_length": df["diameter"] > df["length"],
    }
