"""Runtime configuration of the API and the dashboard, read from environment variables.

Every variable has the ``CRAB_`` prefix (e.g. ``CRAB_JWT_SECRET_KEY``).
Settings come only from real environment variables (in Docker: the compose
``environment:`` section); the ``.env`` file is left to docker compose, which
also keeps non-CRAB variables there. Typos are errors, not silent defaults:
unknown arguments are rejected (``extra="forbid"``) and so is any unknown
``CRAB_*`` environment variable. Secrets have no default in code: without them
the API does not start.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_PREFIX: Final = "CRAB_"


def _reject_unknown_env_vars() -> None:
    """Raise if a CRAB_* environment variable matches no setting of the API or the dashboard."""
    fields = set(Settings.model_fields) | set(DashboardSettings.model_fields)
    known = {ENV_PREFIX + name.upper() for name in fields}
    unknown = sorted(key for key in os.environ if key.upper().startswith(ENV_PREFIX) and key.upper() not in known)
    if unknown:
        raise ValueError(f"unknown environment variables (typo?): {', '.join(unknown)}")


class Settings(BaseSettings):
    """All settings of the serving API."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid", frozen=True)

    # Authentication: one API user; only the Argon2 hash of the password is stored.
    api_username: str = Field(default="crab", min_length=1, max_length=64)
    api_password_hash: SecretStr
    jwt_secret_key: SecretStr
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS512"
    jwt_expire_minutes: int = Field(default=30, ge=1, le=1440)

    # Model registry and training.
    registered_model_name: str = "crab-age"
    params_path: Path = Path("params.yaml")

    # Monitoring: prediction log database and MLflow inference batches.
    monitoring_db_url: SecretStr = SecretStr("sqlite:///monitoring.db")
    inference_experiment: str = "crab-age-inference"
    inference_batch_size: int = Field(default=10, ge=1, le=10_000)

    # Request hardening and lineage.
    max_body_bytes: int = Field(default=4096, ge=256, le=1_048_576)
    git_sha: str = "unknown"  # CRAB_GIT_SHA build argument; read by crab_age.lineage as a fallback

    @model_validator(mode="before")
    @classmethod
    def _no_unknown_env_vars(cls, data: Any) -> Any:
        _reject_unknown_env_vars()
        return data

    @field_validator("api_password_hash")
    @classmethod
    def _must_be_argon2(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("$argon2"):
            raise ValueError("CRAB_API_PASSWORD_HASH must be an Argon2 hash, not a plain password")
        return value

    @field_validator("jwt_secret_key")
    @classmethod
    def _long_enough(cls, value: SecretStr) -> SecretStr:
        # RFC 7518: an HMAC key must be at least as long as the hash output (64 bytes for HS512).
        if len(value.get_secret_value().encode()) < 64:
            raise ValueError("CRAB_JWT_SECRET_KEY must be at least 64 bytes long")
        return value


class DashboardSettings(BaseSettings):
    """Settings of the Streamlit monitoring dashboard (no secrets needed)."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid", frozen=True)

    monitoring_db_url: SecretStr = SecretStr("sqlite:///monitoring.db")
    reference_path: Path = Path("data/processed/train.csv")
    drift_min_rows: int = Field(default=30, ge=5, le=10_000)
    drift_window: int = Field(default=500, ge=10, le=100_000)

    @model_validator(mode="before")
    @classmethod
    def _no_unknown_env_vars(cls, data: Any) -> Any:
        _reject_unknown_env_vars()
        return data


@lru_cache
def get_settings() -> Settings:
    """Return the settings of this process (read once from the environment)."""
    return Settings()  # pyright: ignore[reportCallIssue] - required fields are read from the environment
