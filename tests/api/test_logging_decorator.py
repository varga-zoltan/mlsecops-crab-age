"""Tests for the parameterized prediction logging decorator."""

import logging

import pytest
from pytest_mock import MockerFixture

from crab_age.api.logging_decorator import PredictionResult, log_prediction
from crab_age.api.schemas import CrabFeatures
from crab_age.logging_setup import reset_request_id, set_request_id
from crab_age.monitoring.prediction_log import PredictionStore


class FakeService:
    """Minimal object with a store, like PredictionService."""

    def __init__(self, store: PredictionStore) -> None:
        self.store = store

    @log_prediction()
    def default(self, features: CrabFeatures) -> PredictionResult:
        return PredictionResult(predicted_age_months=10.0, model_version="3")

    @log_prediction(persist=False, log_inputs=False, level=logging.DEBUG)
    def quiet(self, features: CrabFeatures) -> PredictionResult:
        return PredictionResult(predicted_age_months=11.0, model_version="3")

    @log_prediction()
    def failing(self, features: CrabFeatures) -> PredictionResult:
        raise ValueError("model exploded")


@pytest.fixture
def service(store: PredictionStore) -> FakeService:
    """FakeService with an empty SQLite prediction log."""
    return FakeService(store)


@pytest.fixture
def crab(valid_crab: dict[str, object]) -> CrabFeatures:
    """Validated features."""
    return CrabFeatures.model_validate(valid_crab)


def test_default_logs_and_persists(
    service: FakeService, crab: CrabFeatures, caplog: pytest.LogCaptureFixture
) -> None:
    token = set_request_id("req-1")
    try:
        with caplog.at_level(logging.INFO):
            result = service.default(crab)
    finally:
        reset_request_id(token)
    assert result.predicted_age_months == 10.0
    (record,) = [r for r in caplog.records if r.getMessage() == "prediction"]
    assert record.__dict__["inputs"]["sex"] == "M"
    assert record.__dict__["model_version"] == "3"
    log = service.store.read_all()
    assert list(log["request_id"]) == ["req-1"]
    assert log.loc[0, "prediction"] == 10.0


def test_parameters_switch_off_persist_and_inputs(
    service: FakeService, crab: CrabFeatures, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        service.quiet(crab)
    (record,) = [r for r in caplog.records if r.getMessage() == "prediction"]
    assert record.levelno == logging.DEBUG
    assert "inputs" not in record.__dict__
    assert service.store.read_all().empty


def test_request_id_is_generated_outside_a_request(service: FakeService, crab: CrabFeatures) -> None:
    service.default(crab)
    assert len(service.store.read_all().loc[0, "request_id"]) == 16


def test_store_failure_does_not_break_prediction(
    service: FakeService, crab: CrabFeatures, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch.object(service.store, "insert", side_effect=RuntimeError("database down"))
    with caplog.at_level(logging.ERROR):
        result = service.default(crab)
    assert result.predicted_age_months == 10.0
    assert "could not store prediction" in caplog.text


def test_model_error_is_not_logged_as_prediction(service: FakeService, crab: CrabFeatures) -> None:
    with pytest.raises(ValueError, match="model exploded"):
        service.failing(crab)
    assert service.store.read_all().empty


def test_wraps_keeps_name_and_docstring() -> None:
    assert FakeService.default.__name__ == "default"
