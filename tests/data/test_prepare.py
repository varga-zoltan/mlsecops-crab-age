"""Tests for crab_age.data.prepare (the DVC prepare stage)."""

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from crab_age.data.prepare import PrepareParams, clean, main, read_params, run, split


@pytest.fixture
def params(tmp_path: Path, raw_csv: Path) -> PrepareParams:
    return PrepareParams(
        raw_path=raw_csv,
        out_dir=tmp_path / "processed",
        report_path=tmp_path / "metrics" / "prepare.json",
        max_height_to_diameter=0.8,
        test_size=0.2,
        random_state=42,
    )


@pytest.fixture
def params_file(tmp_path: Path, params: PrepareParams) -> Path:
    content: dict[str, object] = {
        "prepare": {
            "raw_path": str(params.raw_path),
            "out_dir": str(params.out_dir),
            "report_path": str(params.report_path),
            "max_height_to_diameter": params.max_height_to_diameter,
            "test_size": params.test_size,
            "random_state": params.random_state,
        }
    }
    path: Path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_read_params(params_file: Path, params: PrepareParams) -> None:
    assert read_params(params_file) == params


def test_clean_keeps_valid_rows(clean_df: pd.DataFrame) -> None:
    cleaned, report = clean(clean_df, max_height_to_diameter=0.8)
    assert len(cleaned) == 10
    assert report["rows_in"] == 10
    assert report["rows_out"] == 10


def test_clean_drops_invalid_rows_and_counts_rules(clean_df: pd.DataFrame) -> None:
    clean_df.loc[0, "height"] = 0.0
    clean_df.loc[1, "height"] = 3.0
    clean_df.loc[1, "shucked_weight"] = 100.0  # same row, two rules
    cleaned, report = clean(clean_df, max_height_to_diameter=0.8)
    assert len(cleaned) == 8
    assert report["non_positive_value"] == 1
    assert report["height_to_diameter_too_large"] == 1
    assert report["shucked_weight_above_weight"] == 1
    assert report["rows_out"] == 8
    assert list(cleaned.index) == list(range(8))


def test_split_sizes_and_reproducible(clean_df: pd.DataFrame) -> None:
    train, test = split(clean_df, test_size=0.2, random_state=42)
    train2, test2 = split(clean_df, test_size=0.2, random_state=42)
    assert (len(train), len(test)) == (8, 2)
    pd.testing.assert_frame_equal(test, test2)
    pd.testing.assert_frame_equal(train, train2)


def test_split_has_no_overlap(clean_df: pd.DataFrame) -> None:
    train, test = split(clean_df, test_size=0.3, random_state=1)
    assert set(train["age"]).isdisjoint(set(test["age"]))


def test_run_writes_outputs(params: PrepareParams) -> None:
    report: dict[str, int] = run(params)
    for name in ("crab_age_clean.csv", "train.csv", "test.csv"):
        assert (params.out_dir / name).is_file()
    assert json.loads(params.report_path.read_text(encoding="utf-8")) == report
    assert report["train_rows"] + report["test_rows"] == report["rows_out"]


def test_main_returns_zero(params_file: Path, params: PrepareParams) -> None:
    assert main(["--params", str(params_file)]) == 0
    assert (params.out_dir / "train.csv").is_file()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_module_entrypoint_exits_zero(
    params_file: Path, params: PrepareParams, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare", "--params", str(params_file)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crab_age.data.prepare", run_name="__main__")
    assert exc.value.code == 0
    assert (params.out_dir / "test.csv").is_file()
