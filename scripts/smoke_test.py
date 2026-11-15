"""Smoke test of the running stack (docker compose up): /token -> /train -> /predict.

Runs with plain Python 3.12, no project install needed:
    python3 scripts/smoke_test.py
    python3 scripts/smoke_test.py --api http://localhost:8000 --password crab-dev-password

Prints one PASS/FAIL line per check and exits with 1 if any check failed.
The logic is in src/crab_age/smoke.py (standard library only, unit tested).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crab_age.smoke import format_report, run_smoke  # noqa: E402 - needs the src path above


def main() -> int:
    """Run the checks, print the report and return the exit code."""
    parser = argparse.ArgumentParser(description="Smoke test of the running Crab Age stack.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--mlflow", default="http://localhost:5500")
    parser.add_argument("--dashboard", default="http://localhost:8501")
    parser.add_argument("--user", default="crab")
    parser.add_argument("--password", default="crab-dev-password", help="local development default")
    args = parser.parse_args()
    results = run_smoke(args.api, args.user, args.password, args.mlflow, args.dashboard)
    print(format_report(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
