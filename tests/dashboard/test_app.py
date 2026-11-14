"""Smoke tests of the thin Streamlit app with Streamlit's AppTest (no browser, no server)."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from crab_age.data.schema import TARGET
from crab_age.monitoring.prediction_log import PredictionRecord, PredictionStore
from crab_age.monitoring.simulate_traffic import generate_crab

APP = str(Path(__file__).parents[2] / "dashboard" / "app.py")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PredictionStore:
    """Point the dashboard at a temporary prediction log and reference file."""
    st.cache_data.clear()  # Streamlit caches live for the whole process
    st.cache_resource.clear()
    rng = np.random.default_rng(0)
    pd.DataFrame([generate_crab(rng) for _ in range(200)]).assign(**{TARGET: 9}).to_csv(
        tmp_path / "train.csv", index=False
    )
    url = f"sqlite:///{tmp_path / 'log.db'}"
    monkeypatch.setenv("CRAB_MONITORING_DB_URL", url)
    monkeypatch.setenv("CRAB_REFERENCE_PATH", str(tmp_path / "train.csv"))
    store = PredictionStore(url)
    store.init_schema()
    return store


def _log(store: PredictionStore, n: int) -> None:
    rng = np.random.default_rng(1)
    for i in range(n):
        store.insert(
            PredictionRecord(datetime.now(timezone.utc), f"r{i}", "3", generate_crab(rng, "drift"), 10.0)
        )


def test_empty_log_shows_how_to_generate_traffic(env: PredictionStore) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "simulate_traffic" in app.info[0].value
    assert app.metric[0].value == "0"


def test_report_is_shown_with_enough_data(env: PredictionStore) -> None:
    _log(env, 60)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert not app.info
    assert app.metric[0].value == "60"
    assert app.metric[1].value == "3"
    assert app.metric[3].value in {"Yes", "No"}
    assert app.subheader[0].value.startswith("Evidently data drift report")
