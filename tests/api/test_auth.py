"""Tests for password checking and JWT handling."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from crab_age.api.auth import authenticate, create_access_token, get_current_user, hash_password
from crab_age.config import Settings


@pytest.mark.parametrize(
    ("username", "password", "expected"),
    [
        ("crab", "test-password", True),
        ("crab", "wrong-password", False),
        ("crab", "", False),
        ("alice", "test-password", False),
        ("", "", False),
    ],
)
def test_authenticate(settings: Settings, username: str, password: str, expected: bool) -> None:
    assert authenticate(settings, username, password) is expected


def test_hash_is_salted_argon2() -> None:
    first, second = hash_password("same"), hash_password("same")
    assert first.startswith("$argon2id$")
    assert first != second


def test_valid_token_returns_user(settings: Settings) -> None:
    token = create_access_token(settings, "crab")
    assert get_current_user(token, settings) == "crab"


def _expect_401(token: str, settings: Settings) -> None:
    with pytest.raises(HTTPException) as exc:
        get_current_user(token, settings)
    assert exc.value.status_code == 401
    assert exc.value.headers == {"WWW-Authenticate": "Bearer"}


def test_expired_token_is_rejected(settings: Settings) -> None:
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    _expect_401(create_access_token(settings, "crab", now=past), settings)


def test_token_signed_with_other_key_is_rejected(settings: Settings) -> None:
    forged = jwt.encode(
        {"sub": "crab", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "another-secret-key-" + "z" * 64,
        algorithm="HS512",
    )
    _expect_401(forged, settings)


def test_unsigned_token_is_rejected(settings: Settings) -> None:
    unsigned = jwt.encode({"sub": "crab", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, "", "none")
    _expect_401(unsigned, settings)


def test_token_without_expiry_is_rejected(settings: Settings) -> None:
    no_exp = jwt.encode({"sub": "crab"}, settings.jwt_secret_key.get_secret_value(), algorithm="HS512")
    _expect_401(no_exp, settings)


def test_token_for_other_user_is_rejected(settings: Settings) -> None:
    _expect_401(create_access_token(settings, "mallory"), settings)


@pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b.c"])
def test_garbage_token_is_rejected(settings: Settings, token: str) -> None:
    _expect_401(token, settings)


def test_malformed_stored_hash_fails_closed(settings: Settings) -> None:
    broken = settings.model_copy(update={"api_password_hash": SecretStr("$argon2id$x")})
    assert authenticate(broken, "crab", "test-password") is False
