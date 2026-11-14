"""Streamlit monitoring dashboard: a thin display layer.

The logic (reading the prediction log, the Evidently drift report, the summary)
is in crab_age.monitoring.drift and is tested there.

Run: streamlit run dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from crab_age.config import DashboardSettings
from crab_age.monitoring.drift import DashboardData, dashboard_data, load_reference
from crab_age.monitoring.prediction_log import PredictionStore


@st.cache_resource
def get_settings() -> DashboardSettings:
    """Settings, read once from the environment."""
    return DashboardSettings()


@st.cache_resource
def get_store(url: str) -> PredictionStore:
    """One database connection pool for the whole app (creates the table if the API has not yet)."""
    store = PredictionStore(url)
    store.init_schema()
    return store


@st.cache_data
def get_reference(path: Path) -> pd.DataFrame:
    """The training features (reference data), read once."""
    return load_reference(path)


@st.cache_data(ttl=30, show_spinner="Building the drift report...")
def get_data(window: int, total: int) -> DashboardData:
    """Recomputed when the window or the number of logged predictions changes (at most every 30 s)."""
    settings = get_settings()
    store = get_store(settings.monitoring_db_url.get_secret_value())
    return dashboard_data(store, get_reference(settings.reference_path), window, settings.drift_min_rows)


st.set_page_config(page_title="Crab Age monitoring", layout="wide")
settings = get_settings()
store = get_store(settings.monitoring_db_url.get_secret_value())

st.title("Crab Age monitoring")
st.caption("Reference: training data. Current: the latest logged /predict requests.")

window = st.slider("Latest N predictions", min_value=50, max_value=5000, value=settings.drift_window, step=50)
data = get_data(window, store.count_all())
summary = data.summary

columns = st.columns(4)
columns[0].metric("Logged predictions", summary.total_predictions)
columns[1].metric("Model version(s) in window", ", ".join(summary.model_versions) or "-")
columns[2].metric(
    "Mean predicted age (months)",
    "-" if summary.mean_predicted_age is None else f"{summary.mean_predicted_age:.1f}",
)
if data.drift is None:
    columns[3].metric("Dataset drift", "-")
    st.info(data.message)
else:
    columns[3].metric(
        "Dataset drift",
        "Yes" if data.drift.dataset_drift else "No",
        f"{data.drift.drifted_columns} / {len(data.drift.column_scores)} columns ({data.drift.drift_share:.0%})",
        delta_color="off",
    )
    st.caption("Dataset drift is reported when at least half of the columns drift (Evidently's default rule).")
    st.subheader(f"Evidently data drift report ({summary.window_rows} requests)")
    components.html(data.drift.html, height=1400, scrolling=True)
