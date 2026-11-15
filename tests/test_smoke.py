"""Tests for the smoke test logic (no running stack: HTTP is faked)."""

import io
import json
import urllib.error
from email.message import Message
from typing import Any

import pytest
from pytest_mock import MockerFixture

from crab_age.smoke import VALID_CRAB, CheckResult, Response, format_report, http, run_smoke


class FakeApi:
    """Answers like a healthy stack; ``broken`` lists the request keys that should misbehave."""

    def __init__(self, broken: frozenset[str] = frozenset()) -> None:
        self.broken = broken
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, **kwargs: Any) -> Response:
        path = url.split(":8000", 1)[-1] if ":8000" in url else url
        self.calls.append((method, path))
        key = f"{method} {path}"
        if key in self.broken:
            return Response(500, None)
        token = kwargs.get("token")
        if path == "/token":
            form: dict[str, str] = kwargs["form"]
            ok = form["password"] == "pw"
            return Response(200, {"access_token": "tok"}) if ok else Response(401, {"detail": "no"})
        if path in {"/predict", "/train"} and token is None:
            return Response(401, {"detail": "no"})
        if path == "/predict":
            valid = kwargs["json_body"]["sex"] != "X"
            return Response(200, {"predicted_age_months": 9.2, "model_version": "3"}) if valid else Response(422, {})
        if path == "/train":
            return Response(200, {"accepted": True, "best_model": "ridge", "model_version": "3", "test_mae": 1.5})
        if path == "/version":
            return Response(200, {"git_sha": "a" * 40, "model_version": "3"})
        return Response(200, {"status": "ok"})


def test_healthy_stack_passes_every_check() -> None:
    results = run_smoke("http://api:8000/", "crab", "pw", "http://mlflow:5000", "http://dash:8501", send=FakeApi())
    assert [r.name for r in results if not r.passed] == []
    assert len(results) == 10
    assert any("age=9.2" in r.detail for r in results)


def test_optional_services_can_be_skipped() -> None:
    results = run_smoke("http://api:8000", "crab", "pw", send=FakeApi())
    assert len(results) == 8
    assert all(r.passed for r in results)


def test_wrong_password_stops_after_token() -> None:
    api = FakeApi()
    results = run_smoke("http://api:8000", "crab", "nope", send=api)
    assert results[-1] == CheckResult("remaining API checks", False, "skipped: no token")
    assert ("POST", "/train") not in api.calls


@pytest.mark.parametrize("broken", ["GET /health", "POST /train", "GET /version"])
def test_a_broken_endpoint_fails_its_check(broken: str) -> None:
    results = run_smoke("http://api:8000", "crab", "pw", send=FakeApi(frozenset({broken})))
    failed = [r.name for r in results if not r.passed]
    assert len(failed) == 1
    assert broken.split()[1] in failed[0]


def test_format_report() -> None:
    report = format_report([CheckResult("a", True, "ok"), CheckResult("longer name", False, "500")])
    assert "PASS  a            ok" in report
    assert "FAIL  longer name  500" in report
    assert report.endswith("1/2 checks passed")


class FakeHttpResponse(io.BytesIO):
    """Context manager with .status, like urllib's response."""

    def __init__(self, raw: bytes, status: int = 200) -> None:
        super().__init__(raw)
        self.status = status


def test_http_sends_json_and_token(mocker: MockerFixture) -> None:
    urlopen = mocker.patch("urllib.request.urlopen", return_value=FakeHttpResponse(b'{"x": 1}'))
    response = http("POST", "http://api/predict", json_body=VALID_CRAB, token="tok")
    assert response == Response(200, {"x": 1})
    request = urlopen.call_args.args[0]
    assert request.headers["Content-type"] == "application/json"
    assert request.headers["Authorization"] == "Bearer tok"
    assert json.loads(request.data)["sex"] == "F"


def test_http_sends_form(mocker: MockerFixture) -> None:
    urlopen = mocker.patch("urllib.request.urlopen", return_value=FakeHttpResponse(b""))
    assert http("POST", "http://api/token", form={"username": "crab"}) == Response(200, None)
    assert urlopen.call_args.args[0].data == b"username=crab"


def test_http_returns_errors_and_non_json(mocker: MockerFixture) -> None:
    error = urllib.error.HTTPError("http://api/x", 401, "no", Message(), io.BytesIO(b'{"detail": "no"}'))
    mocker.patch("urllib.request.urlopen", side_effect=[error, FakeHttpResponse(b"<html>")])
    assert http("GET", "http://api/x") == Response(401, {"detail": "no"})
    assert http("GET", "http://api/x") == Response(200, None)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x"])
def test_http_rejects_other_schemes(url: str) -> None:
    with pytest.raises(ValueError, match="only http"):
        http("GET", url)
