"""One-shot init job of ``docker compose up``.

1. ``dvc pull``: download the raw and processed data from the public DVC remote.
2. If the registry already has a champion model, stop: nothing to do.
3. Otherwise run the first training (search + gate), so /predict works at once.

The exit code is the contract with docker compose: the API only starts after
this job finished with 0 (``service_completed_successfully``).

Run with: python -m crab_age.bootstrap --params params.yaml
"""

import argparse
import logging
import subprocess
from pathlib import Path

from mlflow import MlflowClient

from crab_age.logging_setup import setup_logging
from crab_age.model import train

logger = logging.getLogger(__name__)


def pull_data() -> None:
    """Download the DVC-tracked data (raises if ``dvc pull`` fails)."""
    subprocess.run(["dvc", "pull"], check=True)
    logger.info("dvc pull finished")


def main(argv: list[str] | None = None) -> int:
    """Pull the data and train a first champion if there is none."""
    parser = argparse.ArgumentParser(description="Init job: pull data and train the first champion if needed.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    setup_logging()

    pull_data()
    model_name = train.read_train_params(args.params).registered_model_name
    version = train.champion_version(MlflowClient(), model_name)
    if version is not None:
        logger.info("champion already exists, skipping training", extra={"model_version": version})
        return 0
    logger.info("no champion yet, running the first training")
    return train.main(["--params", str(args.params)])


if __name__ == "__main__":
    raise SystemExit(main())
