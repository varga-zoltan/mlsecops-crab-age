"""Send /predict requests with realistic or drifted crabs, so the drift report has data.

Two modes:

* ``normal`` - real crabs from the held-out test set (``data/processed/test.csv``),
  so the drift report should show no dataset drift;
* ``drift``  - the same crabs made bigger and heavier, with more juveniles
  (sex "I"), so the drift report clearly shows drift.

If the test set is not available (e.g. on a host without ``dvc pull``), crabs are
generated from the body proportions of the training data instead.

Run inside the stack (no local Python needed):
    docker compose exec api python -m crab_age.monitoring.simulate_traffic --mode drift
or from the host: python scripts/simulate_traffic.py --mode drift
"""

import argparse
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from crab_age.api.schemas import MAX_SIZE, MAX_WEIGHT
from crab_age.data.schema import FEATURES
from crab_age.logging_setup import setup_logging

logger = logging.getLogger(__name__)

Mode = Literal["normal", "drift"]

# Proportions measured on the cleaned training data.
SEX_SHARE: Final[dict[str, float]] = {"M": 0.369, "F": 0.315, "I": 0.316}
LENGTH_MEAN, LENGTH_STD = 1.312, 0.300
RATIO_TO_LENGTH: Final[dict[str, tuple[float, float]]] = {"diameter": (0.776, 0.033), "height": (0.264, 0.033)}
RATIO_TO_LENGTH_CUBED: Final[dict[str, tuple[float, float]]] = {
    "weight": (9.046, 1.400),
    "shucked_weight": (3.882, 0.779),
    "viscera_weight": (1.974, 0.416),
    "shell_weight": (2.660, 0.533),
}
# Drift mode: 35% longer, 30% heavier for the size, mostly juveniles.
DRIFT_LENGTH_FACTOR, DRIFT_WEIGHT_FACTOR = 1.35, 1.30
DRIFT_SEX_SHARE: Final[dict[str, float]] = {"M": 0.15, "F": 0.15, "I": 0.70}
SIZE_COLUMNS: Final = ("length", "diameter", "height")
WEIGHT_COLUMNS: Final = ("weight", "shucked_weight", "viscera_weight", "shell_weight")


def load_pool(path: Path) -> pd.DataFrame | None:
    """Real crabs to sample from, or None if the file is missing."""
    if not path.exists():
        logger.warning("no data file, generating synthetic crabs", extra={"path": str(path)})
        return None
    return pd.read_csv(path).loc[:, FEATURES]


def _drifted(rng: np.random.Generator, crab: dict[str, Any]) -> dict[str, Any]:
    """Make a crab bigger and heavier and redraw its sex with more juveniles."""
    shifted = dict(crab)
    shifted["sex"] = str(rng.choice(list(DRIFT_SEX_SHARE), p=list(DRIFT_SEX_SHARE.values())))
    for name in SIZE_COLUMNS:
        shifted[name] = round(min(float(crab[name]) * DRIFT_LENGTH_FACTOR, MAX_SIZE), 4)
    weight_factor = DRIFT_LENGTH_FACTOR**3 * DRIFT_WEIGHT_FACTOR
    for name in WEIGHT_COLUMNS:
        shifted[name] = round(min(float(crab[name]) * weight_factor, MAX_WEIGHT), 4)
    return shifted


def generate_crab(
    rng: np.random.Generator, mode: Mode = "normal", pool: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Return one valid /predict request body (from ``pool`` if given, else synthetic)."""
    if pool is not None:
        row = pool.iloc[int(rng.integers(len(pool)))]
        crab: dict[str, Any] = {name: (str(row[name]) if name == "sex" else float(row[name])) for name in FEATURES}
        return _drifted(rng, crab) if mode == "drift" else crab
    drift = mode == "drift"
    sex_share = DRIFT_SEX_SHARE if drift else SEX_SHARE
    sex = str(rng.choice(list(sex_share), p=list(sex_share.values())))
    length = rng.normal(LENGTH_MEAN * (DRIFT_LENGTH_FACTOR if drift else 1.0), LENGTH_STD)
    length = float(np.clip(length, 0.2, MAX_SIZE / 2))
    crab = {"sex": sex, "length": round(length, 4)}
    for name, (mean, std) in RATIO_TO_LENGTH.items():
        crab[name] = round(float(np.clip(length * rng.normal(mean, std), 0.01, MAX_SIZE)), 4)
    weight_factor = DRIFT_WEIGHT_FACTOR if drift else 1.0
    for name, (mean, std) in RATIO_TO_LENGTH_CUBED.items():
        value = length**3 * max(rng.normal(mean, std), 0.1) * weight_factor
        crab[name] = round(float(np.clip(value, 0.01, MAX_WEIGHT)), 4)
    return crab


def _check_url(base_url: str) -> str:
    scheme = urllib.parse.urlparse(base_url).scheme
    if scheme not in {"http", "https"}:
        raise ValueError(f"only http(s) URLs are allowed, got {base_url!r}")
    return base_url.rstrip("/")


def get_token(base_url: str, username: str, password: str) -> str:
    """Log in at /token and return the bearer token."""
    form = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(f"{_check_url(base_url)}/token", data=form, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:  # scheme checked in _check_url
        token: str = json.load(response)["access_token"]
    return token


def predict(base_url: str, token: str, crab: dict[str, Any]) -> int:
    """POST one crab to /predict and return the HTTP status code."""
    request = urllib.request.Request(
        f"{_check_url(base_url)}/predict",
        data=json.dumps(crab).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # scheme checked in _check_url
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def simulate(
    base_url: str, username: str, password: str, n: int, mode: Mode, seed: int, pool: pd.DataFrame | None = None
) -> Counter[int]:
    """Send ``n`` requests and count the response status codes."""
    rng = np.random.default_rng(seed)
    token = get_token(base_url, username, password)
    statuses: Counter[int] = Counter(predict(base_url, token, generate_crab(rng, mode, pool)) for _ in range(n))
    logger.info("traffic sent", extra={"mode": mode, "requests": n, "statuses": dict(statuses)})
    return statuses


def main(argv: list[str] | None = None) -> int:
    """Command line entry point. Exit code 0 if every request returned 200."""
    parser = argparse.ArgumentParser(description="Send synthetic /predict requests to the Crab Age API.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--user", default="crab")
    parser.add_argument("--password", default="crab-dev-password", help="local development default")
    parser.add_argument("--n", type=int, default=300, help="number of requests")
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.csv"), help="real crabs to sample")
    args = parser.parse_args(argv)
    setup_logging()
    pool = load_pool(args.data)
    statuses = simulate(args.url, args.user, args.password, args.n, args.mode, args.seed, pool)
    return 0 if set(statuses) == {200} else 1


if __name__ == "__main__":
    raise SystemExit(main())
