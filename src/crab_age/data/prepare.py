"""DVC 'prepare' stage: validate, clean and split the raw data.

Run with: python -m crab_age.data.prepare --params params.yaml
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crab_age.data.load import load_raw
from crab_age.data.validate import invalid_row_masks, validate_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareParams:
    """Settings of the prepare stage, read from params.yaml."""

    raw_path: Path
    out_dir: Path
    report_path: Path
    max_height_to_diameter: float
    test_size: float
    random_state: int


def read_params(path: Path) -> PrepareParams:
    """Read the 'prepare' section of params.yaml."""
    with path.open(encoding="utf-8") as f:
        section: dict[str, Any] = yaml.safe_load(f)["prepare"]
    return PrepareParams(
        raw_path=Path(section["raw_path"]),
        out_dir=Path(section["out_dir"]),
        report_path=Path(section["report_path"]),
        max_height_to_diameter=float(section["max_height_to_diameter"]),
        test_size=float(section["test_size"]),
        random_state=int(section["random_state"]),
    )


def clean(df: pd.DataFrame, max_height_to_diameter: float) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop physically impossible rows.

    Returns the cleaned frame and a report: rows matched by each rule, rows in, rows out.
    One row can match more than one rule.
    """
    masks: dict[str, pd.Series] = invalid_row_masks(df, max_height_to_diameter)
    drop: pd.Series = pd.Series(False, index=df.index)
    for mask in masks.values():
        drop = drop | mask
    report: dict[str, int] = {name: int(mask.sum()) for name, mask in masks.items()}
    report["rows_in"] = len(df)
    report["rows_out"] = int((~drop).sum())
    return df.loc[~drop].reset_index(drop=True), report


def split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into a fixed train and test set (same seed gives the same split)."""
    test: pd.DataFrame = df.sample(frac=test_size, random_state=random_state)
    train: pd.DataFrame = df.loc[~df.index.isin(test.index)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run(params: PrepareParams) -> dict[str, int]:
    """Run the whole stage and write the outputs. Returns the report."""
    df: pd.DataFrame = load_raw(params.raw_path)
    validate_schema(df)
    cleaned, report = clean(df, params.max_height_to_diameter)
    train, test = split(cleaned, params.test_size, params.random_state)
    report["train_rows"] = len(train)
    report["test_rows"] = len(test)

    params.out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(params.out_dir / "crab_age_clean.csv", index=False)
    train.to_csv(params.out_dir / "train.csv", index=False)
    test.to_csv(params.out_dir / "test.csv", index=False)
    params.report_path.parent.mkdir(parents=True, exist_ok=True)
    params.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info("prepare finished: %s", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Command line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Validate, clean and split the crab age data.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(read_params(args.params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
