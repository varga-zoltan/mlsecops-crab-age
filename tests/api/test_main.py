"""Endpoint tests with FastAPI's TestClient (no server, no MLflow, no Docker)."""

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sklearn.pipeline import Pipeline

from crab_age.api.auth import get_current_user
from crab_age.api.main import create_app
from crab_age.api.model_service import LoadedModel, ModelHolder
from crab_age.config import Settings, get_settings
from crab_age.model.promote import GateDecision
from crab_age.model.train import TrainResult
from crab_age.monitoring.prediction_log import PredictionStore


def _train_result(accepted: bool) -> TrainResult:
    decision = GateDecision(
        accepted=accepted, reason="accepted" if accepted else "worse", challenger_mae=1.5, champion_mae=1.6
    )
    return TrainResult(
        run_id="run-1",
        model_name="ridge",
        model_version="8",
        test_metrics={"mae": 1.5, "rmse": 2.0, "r2": 0.5},
        decision=decision,
    )


# --- info endpoints -------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_version(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["name"] == "Crab Age API"
    assert body["model_name"] == "crab-age"
    assert body["model_version"] == "7"


def test_openapi_docs_are_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


# --- /token ---------------------------------------------------------------


@pytest.mark.parametrize(("username", "password"), [("crab", "wrong"), ("nobody", "test-password")])
def test_token_with_wrong_credentials(client: TestClient, username: str, password: str) -> None:
    response = client.post("/token", data={"username": username, "password": password})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_token_without_form_fields(client: TestClient) -> None:
    assert client.post("/token", data={}).status_code == 422


# --- /predict -------------------------------------------------------------


def test_predict_returns_age_and_logs_it(
    client: TestClient, auth_headers: dict[str, str], store: PredictionStore, valid_crab: dict[str, object]
) -> None:
    response = client.post("/predict", json=valid_crab, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert 4.0 <= body["predicted_age_months"] <= 13.0
    assert body["model_version"] == "7"
    assert body["request_id"] == response.headers["X-Request-ID"]

    log = store.read_all()
    assert len(log) == 1
    assert log.loc[0, "request_id"] == body["request_id"]
    assert log.loc[0, "sex"] == "M"
    assert log.loc[0, "prediction"] == pytest.approx(body["predicted_age_months"])


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-token"}, {"Authorization": "Basic abc"}])
def test_predict_without_valid_token_is_401(
    client: TestClient, valid_crab: dict[str, object], headers: dict[str, str]
) -> None:
    assert client.post("/predict", json=valid_crab, headers=headers).status_code == 401


@pytest.mark.parametrize(
    ("field", "value"),
    [("length", -1.0), ("height", 0), ("weight", "heavy"), ("sex", "X"), ("diameter", 99.0)],
)
def test_predict_invalid_input_is_422(
    client: TestClient, auth_headers: dict[str, str], valid_crab: dict[str, object], field: str, value: object
) -> None:
    valid_crab[field] = value
    response = client.post("/predict", json=valid_crab, headers=auth_headers)
    assert response.status_code == 422
    assert field in response.text


def test_predict_missing_field_is_422(
    client: TestClient, auth_headers: dict[str, str], valid_crab: dict[str, object]
) -> None:
    del valid_crab["sex"]
    assert client.post("/predict", json=valid_crab, headers=auth_headers).status_code == 422


def test_predict_too_large_payload_is_413(
    client: TestClient, auth_headers: dict[str, str], valid_crab: dict[str, object]
) -> None:
    valid_crab["padding"] = "x" * 10_000
    response = client.post("/predict", json=valid_crab, headers=auth_headers)
    assert response.status_code == 413


def test_predict_chunked_body_is_411(client: TestClient, auth_headers: dict[str, str]) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"sex": "M"}'  # pragma: no cover - the server refuses before reading the body

    headers = {**auth_headers, "Content-Type": "application/json"}
    response = client.post("/predict", content=chunks(), headers=headers)
    assert response.status_code == 411


def test_invalid_content_length_is_400(client: TestClient) -> None:
    assert client.get("/health", headers={"Content-Length": "abc"}).status_code == 400


def test_predict_without_model_is_503(
    settings: Settings, store: PredictionStore, valid_crab: dict[str, object]
) -> None:
    def broken_loader(_: str) -> LoadedModel:
        raise RuntimeError("registry unreachable")

    app = create_app(settings, holder=ModelHolder("crab-age", loader=broken_loader), store=store)
    app.dependency_overrides[get_current_user] = lambda: "crab"
    try:
        with TestClient(app) as test_client:
            assert test_client.get("/health").json()["model_loaded"] is False
            assert test_client.get("/version").json()["model_version"] is None
            assert test_client.post("/predict", json=valid_crab).status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_predict_triggers_batch_export(
    client: TestClient, auth_headers: dict[str, str], valid_crab: dict[str, object], mocker: MockerFixture
) -> None:
    export = mocker.patch("crab_age.monitoring.inference_export.InferenceExporter.export_ready_safely")
    client.post("/predict", json=valid_crab, headers=auth_headers)
    export.assert_called_once()


# --- request id middleware -------------------------------------------------


def test_safe_request_id_is_reused(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


@pytest.mark.parametrize("sent", ["has spaces", "x" * 65, "evil line"])
def test_unsafe_request_id_is_replaced(client: TestClient, sent: str) -> None:
    response = client.get("/health", headers={"X-Request-ID": sent.encode("utf-8")})
    returned = response.headers["X-Request-ID"]
    assert returned != sent
    assert len(returned) == 16


# --- /train -----------------------------------------------------------------


@pytest.fixture
def train_app(
    settings: Settings, holder: ModelHolder, store: PredictionStore
) -> Iterator[tuple[TestClient, list[bool]]]:
    """App whose training runner returns the next value of ``outcomes``."""
    outcomes: list[bool] = []
    app = create_app(settings, holder=holder, store=store, train_runner=lambda: _train_result(outcomes.pop(0)))
    app.dependency_overrides[get_current_user] = lambda: "crab"
    with TestClient(app) as test_client:
        yield test_client, outcomes
    app.dependency_overrides.clear()


def test_train_accepted_reloads_model(train_app: tuple[TestClient, list[bool]], mocker: MockerFixture) -> None:
    client, outcomes = train_app
    outcomes.append(True)
    reload = mocker.spy(ModelHolder, "reload")
    body = client.post("/train").json()
    assert body["accepted"] is True
    assert body["model_reloaded"] is True
    assert body["test_mae"] == 1.5
    assert body["champion_mae"] == 1.6
    reload.assert_called_once()


def test_train_rejected_keeps_model(train_app: tuple[TestClient, list[bool]], mocker: MockerFixture) -> None:
    client, outcomes = train_app
    outcomes.append(False)
    reload = mocker.spy(ModelHolder, "reload")
    body = client.post("/train").json()
    assert body["accepted"] is False
    assert body["model_reloaded"] is False
    reload.assert_not_called()


def test_train_reload_failure_keeps_old_model(
    settings: Settings, store: PredictionStore, fitted_pipeline: Pipeline, valid_crab: dict[str, object]
) -> None:
    calls = {"n": 0}

    def flaky_loader(_: str) -> LoadedModel:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("new model is broken")
        return LoadedModel(model=fitted_pipeline, version="7")

    app = create_app(
        settings,
        holder=ModelHolder("crab-age", loader=flaky_loader),
        store=store,
        train_runner=lambda: _train_result(True),
    )
    app.dependency_overrides[get_current_user] = lambda: "crab"
    try:
        with TestClient(app) as test_client:
            body = test_client.post("/train").json()
            assert body["accepted"] is True
            assert body["model_reloaded"] is False
            predict = test_client.post("/predict", json=valid_crab)
            assert predict.status_code == 200
            assert predict.json()["model_version"] == "7"
    finally:
        app.dependency_overrides.clear()


def test_parallel_training_is_409(settings: Settings, holder: ModelHolder, store: PredictionStore) -> None:
    started, release = threading.Event(), threading.Event()

    def slow_runner() -> TrainResult:
        started.set()
        release.wait(timeout=5)
        return _train_result(False)

    app = create_app(settings, holder=holder, store=store, train_runner=slow_runner)
    app.dependency_overrides[get_current_user] = lambda: "crab"
    try:
        with TestClient(app) as test_client:
            first: dict[str, int] = {}
            worker = threading.Thread(target=lambda: first.update(code=test_client.post("/train").status_code))
            worker.start()
            assert started.wait(timeout=5)
            assert test_client.post("/train").status_code == 409
            release.set()
            worker.join(timeout=5)
            assert first["code"] == 200
    finally:
        app.dependency_overrides.clear()


def test_train_without_token_is_401(client: TestClient) -> None:
    assert client.post("/train").status_code == 401


def test_default_settings_and_store_are_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, password_hash: str, holder: ModelHolder
) -> None:
    monkeypatch.setenv("CRAB_API_PASSWORD_HASH", password_hash)
    monkeypatch.setenv("CRAB_JWT_SECRET_KEY", "k" * 64)
    monkeypatch.setenv("CRAB_MONITORING_DB_URL", f"sqlite:///{tmp_path / 'm.db'}")
    get_settings.cache_clear()
    try:
        app = create_app(holder=holder)
        with TestClient(app) as test_client:
            assert test_client.get("/health").status_code == 200
        assert (tmp_path / "m.db").exists()
    finally:
        get_settings.cache_clear()
