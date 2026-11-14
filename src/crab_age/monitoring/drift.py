"""Data drift report: training data (reference) vs. logged requests (current).

All logic of the dashboard lives here, so it is tested; the Streamlit app only
displays the result. Evidently's ``DataDriftPreset`` compares the distribution
of every input feature and reports how many columns drifted.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from crab_age.data.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from crab_age.monitoring.prediction_log import PredictionStore

# Evidently's DataDriftPreset default: the dataset drifts if at least half of the columns drift.
DATASET_DRIFT_SHARE: Final = 0.5

TRAFFIC_HINT: Final = (
    "Generate requests with: docker compose exec api python -m crab_age.monitoring.simulate_traffic "
    "--mode normal (or --mode drift), or run scripts/simulate_traffic.py from the host."
)


@dataclass(frozen=True)
class LogSummary:
    """Key numbers of the prediction log window shown on the dashboard."""

    total_predictions: int
    window_rows: int
    model_versions: list[str]
    mean_predicted_age: float | None


@dataclass(frozen=True)
class DriftResult:
    """Outcome of one drift report."""

    drifted_columns: int
    drift_share: float
    column_scores: dict[str, float]
    html: str = field(repr=False)

    @property
    def dataset_drift(self) -> bool:
        """True if at least half of the columns drifted (Evidently's dataset drift rule)."""
        return self.drift_share >= DATASET_DRIFT_SHARE


@dataclass(frozen=True)
class DashboardData:
    """Everything the dashboard shows. ``drift`` is None if there is too little data."""

    summary: LogSummary
    drift: DriftResult | None
    message: str | None


def _data_definition() -> DataDefinition:
    return DataDefinition(numerical_columns=list(NUMERIC_FEATURES), categorical_columns=list(CATEGORICAL_FEATURES))


def load_reference(path: Path) -> pd.DataFrame:
    """The training features: the reference distribution of the drift report."""
    return pd.read_csv(path).loc[:, FEATURES]


def summarize(log: pd.DataFrame, total_predictions: int) -> LogSummary:
    """Count, model versions and mean prediction of a prediction log window."""
    versions = sorted(str(version) for version in log["model_version"].unique()) if len(log) else []
    mean = float(log["prediction"].mean()) if len(log) else None
    return LogSummary(
        total_predictions=total_predictions,
        window_rows=len(log),
        model_versions=versions,
        mean_predicted_age=mean,
    )


def parse_drift_metrics(metrics: list[dict[str, Any]]) -> tuple[int, float, dict[str, float]]:
    """Drifted column count, drift share and per-column drift scores from Evidently's metric list.

    Metric types other than the two of DataDriftPreset are ignored.
    """
    drifted, share, scores = 0, 0.0, {}
    for metric in metrics:
        config: dict[str, Any] = metric["config"]
        if config["type"].endswith("DriftedColumnsCount"):
            drifted, share = int(metric["value"]["count"]), float(metric["value"]["share"])
        elif config["type"].endswith("ValueDrift"):
            scores[str(config["column"])] = float(metric["value"])
    return drifted, share, scores


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    """Run Evidently's DataDriftPreset on the feature columns."""
    definition = _data_definition()
    snapshot = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current.loc[:, FEATURES], data_definition=definition),
        reference_data=Dataset.from_pandas(reference.loc[:, FEATURES], data_definition=definition),
    )
    drifted, share, scores = parse_drift_metrics(snapshot.dict()["metrics"])
    return DriftResult(
        drifted_columns=drifted,
        drift_share=share,
        column_scores=scores,
        html=snapshot.get_html_str(as_iframe=False),
    )


def dashboard_data(store: PredictionStore, reference: pd.DataFrame, window: int, min_rows: int) -> DashboardData:
    """Collect the summary and, if there are at least ``min_rows`` logged requests, the drift report."""
    log = store.read_latest(window)
    summary = summarize(log, store.count_all())
    if len(log) < min_rows:
        message = (
            f"Not enough logged predictions for a drift report yet: {len(log)} of at least {min_rows}. "
            + TRAFFIC_HINT
        )
        return DashboardData(summary=summary, drift=None, message=message)
    return DashboardData(summary=summary, drift=build_drift_report(reference, log), message=None)
