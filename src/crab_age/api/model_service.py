"""Holds the champion model and makes predictions with it."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import mlflow
import pandas as pd
from mlflow import MlflowClient

from crab_age.api.logging_decorator import PredictionResult, log_prediction
from crab_age.api.schemas import CrabFeatures
from crab_age.data.schema import FEATURES
from crab_age.model.train import CHAMPION_ALIAS
from crab_age.monitoring.prediction_log import PredictionStore

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """What we need from a model: ``predict(DataFrame) -> array``."""

    def predict(self, data: pd.DataFrame) -> Any:
        """Return one prediction per row."""
        ...


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model and its registry version."""

    model: Predictor
    version: str


class ModelNotLoadedError(RuntimeError):
    """There is no champion model to predict with."""


def load_champion(model_name: str) -> LoadedModel:
    """Load the model version that has the ``champion`` alias in the registry.

    The version is resolved first and then loaded by number, so the reported
    version always matches the loaded model even if the alias moves meanwhile.
    """
    version = MlflowClient().get_model_version_by_alias(model_name, CHAMPION_ALIAS).version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    return LoadedModel(model=model, version=str(version))


class ModelHolder:
    """Keeps the current champion; a failed reload keeps the previous model."""

    def __init__(self, model_name: str, loader: Callable[[str], LoadedModel] = load_champion) -> None:
        self.model_name = model_name
        self._loader = loader
        self.current: LoadedModel | None = None

    def reload(self) -> bool:
        """Try to load the current champion. Returns True on success."""
        try:
            loaded = self._loader(self.model_name)
        except Exception:  # noqa: BLE001 - any load error must keep the old model serving
            logger.exception("could not load champion model, keeping the previous one")
            return False
        self.current = loaded
        logger.info("champion model loaded", extra={"model_name": self.model_name, "model_version": loaded.version})
        return True


class PredictionService:
    """Turns validated features into a prediction with the current champion."""

    def __init__(self, holder: ModelHolder, store: PredictionStore) -> None:
        self.holder = holder
        self.store = store

    @log_prediction(persist=True, log_inputs=True)
    def predict(self, features: CrabFeatures) -> PredictionResult:
        """Predict the age (months) of one crab. Raises ModelNotLoadedError without a model."""
        current = self.holder.current
        if current is None:
            raise ModelNotLoadedError("no champion model is loaded")
        frame = pd.DataFrame([features.model_dump()]).loc[:, FEATURES]
        value = float(current.model.predict(frame)[0])
        return PredictionResult(predicted_age_months=value, model_version=current.version)
