"""Tests for the pydantic-settings configuration."""

import pytest
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pydantic import ValidationError

from crab_age.config import DashboardSettings, Settings, get_settings

SECRET = "x" * 64


def hash_password(password: str) -> str:
    """Argon2 hash, made directly with pwdlib (independent of the API code)."""
    return PasswordHash((Argon2Hasher(),)).hash(password)


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_API_PASSWORD_HASH", hash_password("pw"))
    monkeypatch.setenv("CRAB_JWT_SECRET_KEY", SECRET)
    monkeypatch.setenv("CRAB_INFERENCE_BATCH_SIZE", "25")
    get_settings.cache_clear()
    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()
    assert settings.inference_batch_size == 25
    assert settings.api_username == "crab"
    assert settings.jwt_secret_key.get_secret_value() == SECRET


def test_secrets_are_hidden_in_repr() -> None:
    settings = Settings(api_password_hash=hash_password("pw"), jwt_secret_key=SECRET)
    assert SECRET not in repr(settings)
    assert "sqlite" not in repr(settings)


def test_missing_secrets_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRAB_API_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("CRAB_JWT_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings()  # pyright: ignore[reportCallIssue] - testing that required fields are enforced


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_password_hash": "plain-password"}, "Argon2 hash"),
        ({"jwt_secret_key": "short"}, "at least 64"),
        ({"jwt_algorithm": "none"}, "jwt_algorithm"),
        ({"inference_batch_size": 0}, "inference_batch_size"),
        ({"unknown_setting": 1}, "unknown_setting"),
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"api_password_hash": hash_password("pw"), "jwt_secret_key": SECRET, **overrides}
    with pytest.raises(ValidationError, match=message):
        Settings(**values)  # pyright: ignore[reportArgumentType] - invalid values on purpose


def test_unknown_crab_env_var_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_JWT_SECRET", "typo of CRAB_JWT_SECRET_KEY")
    with pytest.raises(ValidationError, match="CRAB_JWT_SECRET"):
        Settings(api_password_hash=hash_password("pw"), jwt_secret_key=SECRET)


def test_other_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert Settings(api_password_hash=hash_password("pw"), jwt_secret_key=SECRET).api_username == "crab"


def test_dashboard_settings_need_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_DRIFT_MIN_ROWS", "50")
    settings = DashboardSettings()
    assert settings.drift_min_rows == 50
    assert settings.drift_window == 500


def test_dashboard_variables_do_not_break_api_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_DRIFT_WINDOW", "100")
    assert Settings(api_password_hash=hash_password("pw"), jwt_secret_key=SECRET).api_username == "crab"


def test_dashboard_settings_reject_unknown_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_DRIFT_WINDOWS", "100")
    with pytest.raises(ValidationError, match="CRAB_DRIFT_WINDOWS"):
        DashboardSettings()
