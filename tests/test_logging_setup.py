"""Tests for the JSON logging setup."""

import io
import json
import logging
from typing import Any

import pytest

from crab_age.logging_setup import (
    JsonFormatter,
    get_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
    setup_logging,
)


def _lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.fixture
def stream() -> io.StringIO:
    """In-memory stream that receives the JSON log lines."""
    out = io.StringIO()
    setup_logging(logging.DEBUG, stream=out)
    return out


def test_log_line_is_valid_json_with_base_fields(stream: io.StringIO) -> None:
    logging.getLogger("crab_age.test").info("hello %s", "crab")
    (line,) = _lines(stream)
    assert line["message"] == "hello crab"
    assert line["level"] == "INFO"
    assert line["logger"] == "crab_age.test"
    assert line["timestamp"].endswith("+00:00")
    assert "request_id" not in line


def test_extra_fields_are_included(stream: io.StringIO) -> None:
    logging.getLogger("x").info("done", extra={"report": {"rows": 3}, "model_version": 2})
    (line,) = _lines(stream)
    assert line["report"] == {"rows": 3}
    assert line["model_version"] == 2


def test_non_json_values_are_converted_to_text(stream: io.StringIO) -> None:
    logging.getLogger("x").info("path", extra={"path": io.StringIO})
    (line,) = _lines(stream)
    assert "StringIO" in line["path"]


def test_request_id_is_added_and_reset(stream: io.StringIO) -> None:
    token = set_request_id("abc123")
    logging.getLogger("x").info("inside")
    reset_request_id(token)
    logging.getLogger("x").info("outside")
    inside, outside = _lines(stream)
    assert inside["request_id"] == "abc123"
    assert "request_id" not in outside
    assert get_request_id() is None


def test_exception_is_logged(stream: io.StringIO) -> None:
    try:
        raise ValueError("bad value")
    except ValueError:
        logging.getLogger("x").exception("failed")
    (line,) = _lines(stream)
    assert line["level"] == "ERROR"
    assert "ValueError: bad value" in line["exception"]


@pytest.mark.parametrize(
    ("level", "expected"),
    [(logging.DEBUG, 2), (logging.INFO, 1), ("WARNING", 0)],
)
def test_level_filters_records(level: int | str, expected: int) -> None:
    out = io.StringIO()
    setup_logging(level, stream=out)
    logging.getLogger("x").debug("debug")
    logging.getLogger("x").info("info")
    assert len(_lines(out)) == expected


def test_setup_twice_does_not_duplicate_lines() -> None:
    first, second = io.StringIO(), io.StringIO()
    setup_logging(stream=first)
    setup_logging(stream=second)
    logging.getLogger("x").info("once")
    assert first.getvalue() == ""
    assert len(_lines(second)) == 1


def test_default_stream_is_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()
    logging.getLogger("x").warning("to stderr")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip())["message"] == "to stderr"


def test_new_request_id_is_random_hex() -> None:
    first, second = new_request_id(), new_request_id()
    assert first != second
    assert len(first) == 16
    int(first, 16)


def test_formatter_output_is_single_line() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "line1\nline2", None, None)
    text = JsonFormatter().format(record)
    assert "\n" not in text
    assert json.loads(text)["message"] == "line1\nline2"
