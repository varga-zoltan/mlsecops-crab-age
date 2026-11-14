"""Shared fixtures for the API tests: settings, a small model and a test client."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from crab_age.api.auth import hash_password
from crab_age.api.main import create_app
from crab_age.api.model_service import LoadedModel, ModelHolder
from crab_age.config import Settings
from crab_age.data.schema import FEATURES, TARGET
from crab_age.model.pipeline import build_pipeline
from crab_age.model.train import TrainResult
from crab_age.monitoring.prediction_log import PredictionStore

PASSWORD = "test-password"
VALID_CRAB: dict[str, object] = {
    "sex": "M",
    "length": 1.4375,
    "diameter": 1.175,
    "height": 0.4125,
    "weight": 24.635715,
    "shucked_weight": 12.332033,
    "viscera_weight": 5.584852,
    "shell_weight": 6.747181,
}


@pytest.fixture(scope="session")
def password_hash() -> str:
    """Argon2 hash of the test password (hashing is slow, so done once)."""
    return hash_password(PASSWORD)


@pytest.fixture
def settings(password_hash: str) -> Settings:
    """Test settings; the batch size is high so no MLflow export is triggered."""
    return Settings(
        api_password_hash=password_hash,
        jwt_secret_key="test-secret-key-" + "k" * 64,
        inference_batch_size=1000,
    )


@pytest.fixture(scope="session")
def fitted_pipeline() -> Pipeline:
    """A gradient boosting pipeline trained on synthetic crabs (age 4..12 months)."""
    rng = np.random.default_rng(0)
    n = 80
    size = rng.uniform(0.3, 1.8, n)
    df = pd.DataFrame(
        {
            "sex": rng.choice(["M", "F", "I"], n),
            "length": size,
            "diameter": 0.8 * size,
            "height": 0.3 * size,
            "weight": 20 * size,
            "shucked_weight": 9 * size,
            "viscera_weight": 4 * size,
            "shell_weight": 6 * size,
        }
    )
    df[TARGET] = 4.0 + 5.0 * size
    return build_pipeline("hist_gradient_boosting", random_state=0).fit(df[FEATURES], df[TARGET])


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    """Prediction log in a throw-away SQLite file."""
    prediction_store = PredictionStore(f"sqlite:///{tmp_path / 'monitoring.db'}")
    prediction_store.init_schema()
    return prediction_store


@pytest.fixture
def holder(fitted_pipeline: Pipeline) -> ModelHolder:
    """A model holder that 'loads' the fitted pipeline as version 7."""
    return ModelHolder("crab-age", loader=lambda _: LoadedModel(model=fitted_pipeline, version="7"))


def _no_training() -> TrainResult:
    raise AssertionError("training must be stubbed in this test")  # pragma: no cover - fails only on a test bug


@pytest.fixture
def client(settings: Settings, holder: ModelHolder, store: PredictionStore) -> Iterator[TestClient]:
    """A TestClient whose lifespan has loaded the model."""
    app = create_app(settings, holder=holder, store=store, train_runner=_no_training)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Authorization header with a token from the real /token endpoint."""
    response = client.post("/token", data={"username": "crab", "password": PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def valid_crab() -> dict[str, object]:
    """A valid /predict request body (a fresh copy for every test)."""
    return dict(VALID_CRAB)
