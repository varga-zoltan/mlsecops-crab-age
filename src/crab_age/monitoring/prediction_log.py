"""Prediction log: every /predict request and its result, stored in a database.

In Docker this is a separate Postgres database (``monitoring``), not the MLflow
database, so it never mixes with MLflow's own tables and migrations. The same
code runs on SQLite in the tests. The table is the "current" data of the drift
report and the source of the MLflow inference batches.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func, select, update
from sqlalchemy.engine import Engine

from crab_age.data.schema import FEATURES, NUMERIC_FEATURES

metadata = MetaData()

predictions: Final = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("request_id", String(64), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("sex", String(1), nullable=False),
    *[Column(name, Float, nullable=False) for name in NUMERIC_FEATURES],
    Column("prediction", Float, nullable=False),
    Column("export_run_id", String(64), nullable=True, index=True),
)

LOG_COLUMNS: Final[list[str]] = ["id", "created_at", "request_id", "model_version", *FEATURES, "prediction"]


@dataclass(frozen=True)
class PredictionRecord:
    """One logged prediction."""

    created_at: datetime
    request_id: str
    model_version: str
    features: dict[str, Any]
    prediction: float


class PredictionStore:
    """Read and write the prediction log table."""

    def __init__(self, url: str) -> None:
        self.engine: Engine = create_engine(url, pool_pre_ping=True)

    def init_schema(self) -> None:
        """Create the table if it does not exist yet."""
        metadata.create_all(self.engine)

    def insert(self, record: PredictionRecord) -> int:
        """Store one prediction and return its id."""
        row = {
            "created_at": record.created_at,
            "request_id": record.request_id,
            "model_version": record.model_version,
            **{name: record.features[name] for name in FEATURES},
            "prediction": record.prediction,
        }
        with self.engine.begin() as conn:
            result = conn.execute(predictions.insert().values(**row))
            key = result.inserted_primary_key
        assert key is not None  # always set for a single-row insert
        return int(key[0])

    def count_unexported(self) -> int:
        """Number of rows not yet exported to MLflow."""
        query = select(func.count()).select_from(predictions).where(predictions.c.export_run_id.is_(None))
        with self.engine.connect() as conn:
            return int(conn.execute(query).scalar_one())

    def unexported(self) -> pd.DataFrame:
        """All rows not yet exported to MLflow, oldest first."""
        query = select(*[predictions.c[name] for name in LOG_COLUMNS]).where(
            predictions.c.export_run_id.is_(None)
        ).order_by(predictions.c.id)
        return self._frame(query)

    def mark_exported(self, ids: list[int], run_id: str) -> None:
        """Remember that these rows were exported in MLflow run ``run_id``."""
        statement = update(predictions).where(predictions.c.id.in_(ids)).values(export_run_id=run_id)
        with self.engine.begin() as conn:
            conn.execute(statement)

    def read_all(self) -> pd.DataFrame:
        """The whole prediction log, oldest first (input of the drift report)."""
        query = select(*[predictions.c[name] for name in LOG_COLUMNS]).order_by(predictions.c.id)
        return self._frame(query)

    def _frame(self, query: Any) -> pd.DataFrame:
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return pd.DataFrame([dict(row) for row in rows], columns=pd.Index(LOG_COLUMNS))
