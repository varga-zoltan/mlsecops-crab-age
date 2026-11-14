"""Send synthetic /predict requests to the running stack (normal or drifted data).

Usage (with the project venv active):
    python scripts/simulate_traffic.py --mode normal --n 300
    python scripts/simulate_traffic.py --mode drift --n 300

The logic is in crab_age.monitoring.simulate_traffic; without a local venv run it in the
API container: docker compose exec api python -m crab_age.monitoring.simulate_traffic --mode drift
"""

from crab_age.monitoring.simulate_traffic import main

if __name__ == "__main__":
    raise SystemExit(main())
