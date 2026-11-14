"""Smoke tests for the package itself."""

import crab_age


def test_version_is_semver_string() -> None:
    """The package exposes a semantic version string."""
    parts: list[str] = crab_age.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
