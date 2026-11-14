"""Request and response models of the API.

Validation happens here, before the model sees any data: wrong types, unknown
fields, zero or negative sizes, unknown sex and absurdly large values are all
rejected with HTTP 422. ``strict=True`` means "1.2" (a string) is not accepted
as a number.
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_SIZE: Final = 5.0  # length / diameter / height; largest seen value is about 2.0
MAX_WEIGHT: Final = 200.0  # all weights; largest seen value is about 80


class CrabFeatures(BaseModel):
    """Measurements of one crab (same units as the training data)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sex: Literal["M", "F", "I"] = Field(description="M = male, F = female, I = indeterminate")
    length: float = Field(gt=0, le=MAX_SIZE, examples=[1.4375])
    diameter: float = Field(gt=0, le=MAX_SIZE, examples=[1.175])
    height: float = Field(gt=0, le=MAX_SIZE, examples=[0.4125])
    weight: float = Field(gt=0, le=MAX_WEIGHT, examples=[24.635715])
    shucked_weight: float = Field(gt=0, le=MAX_WEIGHT, examples=[12.332033])
    viscera_weight: float = Field(gt=0, le=MAX_WEIGHT, examples=[5.584852])
    shell_weight: float = Field(gt=0, le=MAX_WEIGHT, examples=[6.747181])


class PredictResponse(BaseModel):
    """Predicted age of one crab."""

    predicted_age_months: float
    model_version: str
    request_id: str


class TrainResponse(BaseModel):
    """Outcome of a training run and of the acceptance gate."""

    accepted: bool
    reason: str
    best_model: str
    model_version: str
    run_id: str
    test_mae: float
    test_rmse: float
    test_r2: float
    champion_mae: float | None
    model_reloaded: bool


class Token(BaseModel):
    """OAuth2 bearer token."""

    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    """Liveness information."""

    status: Literal["ok"] = "ok"
    model_loaded: bool


class VersionResponse(BaseModel):
    """Version of the API and of the served model."""

    name: str
    version: str
    git_sha: str
    model_name: str
    model_version: str | None


class ErrorResponse(BaseModel):
    """Error body (FastAPI's default shape)."""

    detail: str
