"""JWT authentication with an Argon2-hashed password.

Flow: POST /token with username + password -> signed JWT (short lifetime).
Protected endpoints require ``Authorization: Bearer <token>``; a missing,
expired or forged token gives HTTP 401.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher

from crab_age.config import Settings

password_hash = PasswordHash((Argon2Hasher(),))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Verifying against a dummy hash for unknown users keeps the response time the
# same, so the API does not reveal which usernames exist.
_DUMMY_HASH = password_hash.hash("dummy-password-for-timing")


def hash_password(password: str) -> str:
    """Return the Argon2 hash of ``password`` (for creating CRAB_API_PASSWORD_HASH)."""
    return password_hash.hash(password)


def authenticate(settings: Settings, username: str, password: str) -> bool:
    """Return True if username and password match the configured API user."""
    known_user = username == settings.api_username
    stored = settings.api_password_hash.get_secret_value() if known_user else _DUMMY_HASH
    try:
        valid = password_hash.verify(password, stored)
    except UnknownHashError:
        valid = False
    return known_user and valid


def create_access_token(settings: Settings, username: str, now: datetime | None = None) -> str:
    """Return a signed JWT for ``username`` that expires after the configured minutes."""
    issued = now or datetime.now(timezone.utc)
    claims = {"sub": username, "iat": issued, "exp": issued + timedelta(minutes=settings.jwt_expire_minutes)}
    return jwt.encode(claims, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def get_settings_from_app(request: Request) -> Settings:
    """FastAPI dependency: the settings the app was created with."""
    settings: Settings = request.app.state.settings
    return settings


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    settings: Annotated[Settings, Depends(get_settings_from_app)],
) -> str:
    """FastAPI dependency: the username in a valid token, else HTTP 401."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp"]},
        )
    except jwt.InvalidTokenError:
        raise unauthorized from None
    if claims["sub"] != settings.api_username:
        raise unauthorized
    return str(claims["sub"])
