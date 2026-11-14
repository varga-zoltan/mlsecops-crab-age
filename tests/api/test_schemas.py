"""Tests for request validation, including a property-based test."""

import math

import pytest
from hypothesis import given, settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from crab_age.api.model_service import LoadedModel, ModelHolder, PredictionService
from crab_age.api.schemas import MAX_SIZE, MAX_WEIGHT, CrabFeatures
from crab_age.monitoring.prediction_log import PredictionStore


def test_valid_crab_is_accepted(valid_crab: dict[str, object]) -> None:
    assert CrabFeatures.model_validate(valid_crab).sex == "M"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length", 0),
        ("length", -1.0),
        ("height", MAX_SIZE + 0.01),
        ("weight", MAX_WEIGHT + 1),
        ("weight", 1e308),
        ("diameter", "1.2"),
        ("diameter", "abc"),
        ("diameter", None),
        ("diameter", True),
        ("sex", "X"),
        ("sex", "m"),
        ("sex", 1),
    ],
)
def test_invalid_values_are_rejected(valid_crab: dict[str, object], field: str, value: object) -> None:
    valid_crab[field] = value
    with pytest.raises(ValidationError):
        CrabFeatures.model_validate(valid_crab)


def test_missing_field_is_rejected(valid_crab: dict[str, object]) -> None:
    del valid_crab["shell_weight"]
    with pytest.raises(ValidationError, match="shell_weight"):
        CrabFeatures.model_validate(valid_crab)


def test_unknown_field_is_rejected(valid_crab: dict[str, object]) -> None:
    valid_crab["age"] = 10
    with pytest.raises(ValidationError, match="age"):
        CrabFeatures.model_validate(valid_crab)


def _size() -> st.SearchStrategy[float]:
    return st.floats(min_value=1e-3, max_value=MAX_SIZE, allow_nan=False, allow_infinity=False)


def _weight() -> st.SearchStrategy[float]:
    return st.floats(min_value=1e-3, max_value=MAX_WEIGHT, allow_nan=False, allow_infinity=False)


crabs = st.builds(
    CrabFeatures,
    sex=st.sampled_from(["M", "F", "I"]),
    length=_size(),
    diameter=_size(),
    height=_size(),
    weight=_weight(),
    shucked_weight=_weight(),
    viscera_weight=_weight(),
    shell_weight=_weight(),
)


@hypothesis_settings(max_examples=50, deadline=None)
@given(features=crabs)
def test_any_valid_crab_gets_a_finite_positive_age(fitted_pipeline: Pipeline, features: CrabFeatures) -> None:
    holder = ModelHolder("crab-age", loader=lambda _: LoadedModel(model=fitted_pipeline, version="1"))
    holder.reload()
    store = PredictionStore("sqlite://")
    store.init_schema()
    result = PredictionService(holder, store).predict(features)
    assert math.isfinite(result.predicted_age_months)
    assert result.predicted_age_months > 0
