"""Column names and allowed values of the crab age dataset."""

from typing import Final

RAW_TO_CLEAN_COLUMNS: Final[dict[str, str]] = {
    "Sex": "sex",
    "Length": "length",
    "Diameter": "diameter",
    "Height": "height",
    "Weight": "weight",
    "Shucked Weight": "shucked_weight",
    "Viscera Weight": "viscera_weight",
    "Shell Weight": "shell_weight",
    "Age": "age",
}

CATEGORICAL_FEATURES: Final[list[str]] = ["sex"]
NUMERIC_FEATURES: Final[list[str]] = [
    "length",
    "diameter",
    "height",
    "weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
FEATURES: Final[list[str]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET: Final[str] = "age"
COLUMNS: Final[list[str]] = FEATURES + [TARGET]

SEX_VALUES: Final[frozenset[str]] = frozenset({"M", "F", "I"})
