"""Tests for the traffic simulator (HTTP is stubbed, no running API needed)."""

import io
import json
import runpy
import sys
import urllib.error
from collections import Counter
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from crab_age.api.schemas import MAX_WEIGHT, CrabFeatures
from crab_age.data.schema import FEATURES
from crab_age.monitoring import simulate_traffic as sim


@pytest.mark.parametrize("mode", ["normal", "drift"])
def test_generated_crabs_pass_api_validation(mode: sim.Mode) -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        CrabFeatures.model_validate(sim.generate_crab(rng, mode))


def test_drift_mode_makes_bigger_crabs() -> None:
    rng_n, rng_d = np.random.default_rng(2), np.random.default_rng(2)
    normal = [sim.generate_crab(rng_n, "normal")["length"] for _ in range(300)]
    drift = [sim.generate_crab(rng_d, "drift")["length"] for _ in range(300)]
    assert np.mean(drift) > 1.2 * np.mean(normal)


def test_same_seed_same_crabs() -> None:
    first = sim.generate_crab(np.random.default_rng(5))
    assert first == sim.generate_crab(np.random.default_rng(5))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x", "localhost:8000"])
def test_only_http_urls_are_allowed(url: str) -> None:
    with pytest.raises(ValueError, match="only http"):
        sim.get_token(url, "crab", "pw")


class FakeResponse(io.BytesIO):
    """Minimal stand-in for an HTTP response (context manager with .status)."""

    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        super().__init__(json.dumps(body).encode())
        self.status = status


def test_get_token_posts_form(mocker: MockerFixture) -> None:
    urlopen = mocker.patch("urllib.request.urlopen", return_value=FakeResponse({"access_token": "tok"}))
    assert sim.get_token("http://api:8000/", "crab", "pw") == "tok"
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://api:8000/token"
    assert request.data == b"username=crab&password=pw"


def test_predict_returns_status_also_on_http_error(mocker: MockerFixture) -> None:
    error = urllib.error.HTTPError("http://api/predict", 422, "Unprocessable", Message(), None)
    urlopen = mocker.patch("urllib.request.urlopen", side_effect=[FakeResponse({}), error])
    assert sim.predict("http://api", "tok", {"sex": "M"}) == 200
    assert sim.predict("http://api", "tok", {"sex": "M"}) == 422
    request = urlopen.call_args.args[0]
    assert request.headers["Authorization"] == "Bearer tok"


def test_simulate_counts_statuses(mocker: MockerFixture) -> None:
    mocker.patch.object(sim, "get_token", return_value="tok")
    sent = mocker.patch.object(sim, "predict", side_effect=[200, 200, 422])
    assert sim.simulate("http://api", "crab", "pw", n=3, mode="drift", seed=0) == Counter({200: 2, 422: 1})
    assert sent.call_count == 3


@pytest.mark.parametrize(("statuses", "exit_code"), [(Counter({200: 5}), 0), (Counter({200: 4, 503: 1}), 1)])
def test_main_exit_code(mocker: MockerFixture, statuses: Counter[int], exit_code: int) -> None:
    simulate = mocker.patch.object(sim, "simulate", return_value=statuses)
    assert sim.main(["--n", "5", "--mode", "drift", "--url", "http://api:8000", "--data", "missing.csv"]) == exit_code
    assert simulate.call_args.args[:6] == ("http://api:8000", "crab", "crab-dev-password", 5, "drift", 0)


def test_module_entry_point(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    mocker.patch("urllib.request.urlopen", side_effect=lambda *a, **k: FakeResponse({"access_token": "t"}))
    monkeypatch.setattr(sys, "argv", ["simulate_traffic", "--n", "2"])
    sys.modules.pop("crab_age.monitoring.simulate_traffic", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("crab_age.monitoring.simulate_traffic", run_name="__main__")
    assert exc.value.code == 0


@pytest.fixture
def pool() -> pd.DataFrame:
    """Two real-looking crabs, one very heavy."""
    return pd.DataFrame(
        [
            {"sex": "M", "length": 1.4, "diameter": 1.1, "height": 0.4,
             "weight": 25.0, "shucked_weight": 11.0, "viscera_weight": 5.0, "shell_weight": 7.0},
            {"sex": "F", "length": 1.0, "diameter": 0.8, "height": 0.3,
             "weight": 80.0, "shucked_weight": 40.0, "viscera_weight": 20.0, "shell_weight": 28.0},
        ]
    )


def test_load_pool(tmp_path: Path, pool: pd.DataFrame) -> None:
    assert sim.load_pool(tmp_path / "missing.csv") is None
    path = tmp_path / "test.csv"
    pool.assign(age=9).to_csv(path, index=False)
    loaded = sim.load_pool(path)
    assert loaded is not None
    assert list(loaded.columns) == FEATURES


def test_normal_mode_sends_real_rows(pool: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    rows = [sim.generate_crab(rng, "normal", pool) for _ in range(20)]
    assert all(row in pool.to_dict("records") for row in rows)


def test_drift_mode_scales_real_rows_within_limits(pool: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        crab = sim.generate_crab(rng, "drift", pool)
        CrabFeatures.model_validate(crab)
        assert crab["length"] in {round(1.4 * sim.DRIFT_LENGTH_FACTOR, 4), round(1.0 * sim.DRIFT_LENGTH_FACTOR, 4)}
        assert crab["weight"] <= MAX_WEIGHT
