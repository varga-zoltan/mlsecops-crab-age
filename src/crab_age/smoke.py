"""Smoke test of the running stack: /token -> /train -> /predict, plus the negative cases.

Standard library only, so ``scripts/smoke_test.py`` runs with any Python 3.12
without installing the project. Every check returns a :class:`CheckResult`; the
script prints them as a table and exits with 1 if any check failed.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

VALID_CRAB: Final[dict[str, Any]] = {
    "sex": "F",
    "length": 1.4375,
    "diameter": 1.175,
    "height": 0.4125,
    "weight": 24.635715,
    "shucked_weight": 12.332033,
    "viscera_weight": 5.584852,
    "shell_weight": 6.747181,
}


@dataclass(frozen=True)
class Response:
    """Status code and decoded JSON body (None if the body is not JSON)."""

    status: int
    body: Any


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one smoke check."""

    name: str
    passed: bool
    detail: str


def http(
    method: str,
    url: str,
    *,
    json_body: Any = None,
    form: dict[str, str] | None = None,
    token: str | None = None,
    timeout: float = 10,
) -> Response:
    """Send one HTTP request and return status + JSON; HTTP errors are returned, not raised."""
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"only http(s) URLs are allowed, got {url!r}")
    headers: dict[str, str] = {}
    data: bytes | None = None
    if json_body is not None:
        data, headers["Content-Type"] = json.dumps(json_body).encode(), "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # scheme checked above
            status, raw = int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        status, raw = int(exc.code), exc.read()
    try:
        body: Any = json.loads(raw) if raw else None
    except ValueError:
        body = None
    return Response(status=status, body=body)


def _check(name: str, condition: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=condition, detail=detail)


def run_smoke(
    api_url: str,
    username: str,
    password: str,
    mlflow_url: str | None = None,
    dashboard_url: str | None = None,
    send: Callable[..., Response] = http,
) -> list[CheckResult]:
    """Run all checks against a running stack and return their results.

    Order: info endpoints, auth failures, /token, invalid input, /train,
    /predict, then (optional) MLflow and dashboard health. Later checks that
    need a token are skipped as failed if /token does not work.
    """
    api = api_url.rstrip("/")
    results: list[CheckResult] = []

    health = send("GET", f"{api}/health")
    results.append(_check("GET /health", health.status == 200, f"{health.status} {health.body}"))
    version = send("GET", f"{api}/version")
    body = version.body if isinstance(version.body, dict) else {}
    results.append(
        _check(
            "GET /version", version.status == 200,
            f"{version.status} git_sha={str(body.get('git_sha', '?'))[:12]} model_version={body.get('model_version')}",
        )
    )

    no_token = send("POST", f"{api}/predict", json_body=VALID_CRAB)
    results.append(_check("POST /predict without token -> 401", no_token.status == 401, str(no_token.status)))
    wrong = send("POST", f"{api}/token", form={"username": username, "password": password + "-wrong"})
    results.append(_check("POST /token wrong password -> 401", wrong.status == 401, str(wrong.status)))

    login = send("POST", f"{api}/token", form={"username": username, "password": password})
    token = login.body.get("access_token") if isinstance(login.body, dict) else None
    results.append(_check("POST /token", login.status == 200 and bool(token), str(login.status)))
    if not token:
        results.append(_check("remaining API checks", False, "skipped: no token"))
        return results

    bad = send("POST", f"{api}/predict", json_body={**VALID_CRAB, "sex": "X", "length": -1}, token=token)
    results.append(_check("POST /predict invalid input -> 422", bad.status == 422, str(bad.status)))

    train = send("POST", f"{api}/train", token=token, timeout=600)
    tb = train.body if isinstance(train.body, dict) else {}
    results.append(
        _check(
            "POST /train", train.status == 200,
            f"{train.status} accepted={tb.get('accepted')} model={tb.get('best_model')} "
            f"version={tb.get('model_version')} test_mae={tb.get('test_mae')}",
        )
    )

    predict = send("POST", f"{api}/predict", json_body=VALID_CRAB, token=token)
    pb = predict.body if isinstance(predict.body, dict) else {}
    age = pb.get("predicted_age_months")
    ok = predict.status == 200 and isinstance(age, (int, float)) and age > 0
    results.append(
        _check("POST /predict", ok, f"{predict.status} age={age} months, model_version={pb.get('model_version')}")
    )

    if mlflow_url:
        mlflow = send("GET", f"{mlflow_url.rstrip('/')}/health")
        results.append(_check("MLflow /health", mlflow.status == 200, str(mlflow.status)))
    if dashboard_url:
        dashboard = send("GET", f"{dashboard_url.rstrip('/')}/_stcore/health")
        results.append(_check("Dashboard /_stcore/health", dashboard.status == 200, str(dashboard.status)))
    return results


def format_report(results: list[CheckResult]) -> str:
    """Plain-text table of the results, one line per check, and a summary line."""
    width = max(len(result.name) for result in results)
    lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.name.ljust(width)}  {r.detail}" for r in results]
    failed = sum(not r.passed for r in results)
    lines.append(f"\n{len(results) - failed}/{len(results)} checks passed")
    return "\n".join(lines)
