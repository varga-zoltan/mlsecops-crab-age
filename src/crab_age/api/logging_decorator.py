"""Parameterized decorator that logs every prediction.

Usage::

    @log_prediction(persist=True, log_inputs=True)
    def predict(self, features: CrabFeatures) -> PredictionResult: ...

For each call it writes one JSON log line (request id, inputs, prediction,
model version, duration) and, if ``persist`` is True, stores the record in the
prediction log database. A database failure is logged but does not fail the
prediction: serving stays available, and the JSON log line still has the data.
"""

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, TypeVar

from crab_age.api.schemas import CrabFeatures
from crab_age.logging_setup import get_request_id, new_request_id
from crab_age.monitoring.prediction_log import PredictionRecord, PredictionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Prediction of one request and the model version that made it."""

    predicted_age_months: float
    model_version: str


class HasPredictionStore(Protocol):
    """Anything with a prediction log store (the decorated method's ``self``)."""

    store: PredictionStore


S = TypeVar("S", bound=HasPredictionStore)
PredictMethod = Callable[[S, CrabFeatures], PredictionResult]


def log_prediction(
    *, persist: bool = True, log_inputs: bool = True, level: int = logging.INFO
) -> Callable[[PredictMethod[S]], PredictMethod[S]]:
    """Return a decorator that logs (and optionally stores) each prediction.

    Args:
        persist: also write the record to the prediction log database.
        log_inputs: include the input features in the JSON log line.
        level: logging level of the log line.
    """

    def decorator(method: PredictMethod[S]) -> PredictMethod[S]:
        @functools.wraps(method)
        def wrapper(self: S, features: CrabFeatures) -> PredictionResult:
            start = time.perf_counter()
            result = method(self, features)
            duration_ms = (time.perf_counter() - start) * 1000
            record = PredictionRecord(
                created_at=datetime.now(timezone.utc),
                request_id=get_request_id() or new_request_id(),
                model_version=result.model_version,
                features=features.model_dump(),
                prediction=result.predicted_age_months,
            )
            extra: dict[str, object] = {
                "prediction": record.prediction,
                "model_version": record.model_version,
                "duration_ms": round(duration_ms, 2),
            }
            if log_inputs:
                extra["inputs"] = record.features
            logger.log(level, "prediction", extra=extra)
            if persist:
                try:
                    self.store.insert(record)
                except Exception:  # noqa: BLE001 - logging must not break serving; the JSON line has the data
                    logger.exception("could not store prediction in the prediction log")
            return result

        return wrapper

    return decorator
