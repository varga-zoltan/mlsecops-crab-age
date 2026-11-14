"""JSON logging for the whole package.

Every log record is written as one JSON object per line, so the logs of the
training job, the API and the dashboard can be searched and parsed by tools.
A request id (correlation id) is kept in a context variable: the API middleware
sets it once per request and every log line written while serving that request
carries the same id.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, TextIO

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has. Anything else was passed via ``extra=``.
_STANDARD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}

_HANDLER_NAME = "crab_age_json"


def new_request_id() -> str:
    """Return a new random request id (16 hex characters)."""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str | None:
    """Return the request id of the current context, or None."""
    return _request_id.get()


def set_request_id(request_id: str | None) -> Token[str | None]:
    """Set the request id for the current context.

    Returns a token; pass it to :func:`reset_request_id` to restore the
    previous value (e.g. at the end of an HTTP request).
    """
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request id that was active before :func:`set_request_id`."""
    _request_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Format a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON text of ``record``."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id is not None:
            payload["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: int | str = logging.INFO, stream: TextIO | None = None) -> None:
    """Send all logs of the process to ``stream`` (default: stderr) as JSON lines.

    Safe to call more than once: the handler added by an earlier call is
    replaced, so log lines are never duplicated.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
