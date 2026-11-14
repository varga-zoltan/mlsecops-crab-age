"""Shared fixtures for all tests."""

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Undo any change a test makes to the root logger (handlers, level)."""
    root = logging.getLogger()
    handlers: list[logging.Handler] = list(root.handlers)
    level: int = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
