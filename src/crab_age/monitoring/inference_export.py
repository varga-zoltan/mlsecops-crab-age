"""Export logged predictions to MLflow as CSV artifacts, batch by batch.

The batch is built from the database rows that were not exported yet, so
nothing is lost when the API restarts. A batch is written once at least
``batch_size`` rows are waiting; each batch is a new run in the inference
experiment. The MlflowClient API is used (not the fluent ``mlflow.start_run``),
so the export never interferes with an active training run.
"""

import logging
import tempfile
import threading
from pathlib import Path

from mlflow import MlflowClient

from crab_age.monitoring.prediction_log import PredictionStore

logger = logging.getLogger(__name__)


class InferenceExporter:
    """Writes full batches of the prediction log to an MLflow experiment."""

    def __init__(self, store: PredictionStore, experiment_name: str, batch_size: int) -> None:
        self.store = store
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def _experiment_id(self, client: MlflowClient) -> str:
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        return str(client.create_experiment(self.experiment_name))

    def export_ready(self) -> str | None:
        """Export the waiting rows if there are at least ``batch_size``.

        Returns the MLflow run id of the batch, or None if nothing was exported.
        """
        with self._lock:
            if self.store.count_unexported() < self.batch_size:
                return None
            batch = self.store.unexported()
            client = MlflowClient()
            first_id, last_id = int(batch["id"].iloc[0]), int(batch["id"].iloc[-1])
            run = client.create_run(
                self._experiment_id(client),
                run_name=f"predictions-{first_id}-{last_id}",
                tags={
                    "first_prediction_id": str(first_id),
                    "last_prediction_id": str(last_id),
                    "model_versions": ",".join(sorted(batch["model_version"].astype(str).unique())),
                },
            )
            run_id = run.info.run_id
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"predictions_{first_id}_{last_id}.csv"
                batch.to_csv(path, index=False)
                client.log_artifact(run_id, str(path), artifact_path="predictions")
            client.log_metric(run_id, "n_predictions", float(len(batch)))
            client.log_metric(run_id, "mean_predicted_age", float(batch["prediction"].mean()))
            client.set_terminated(run_id)
            self.store.mark_exported([int(i) for i in batch["id"]], run_id)
        logger.info("inference batch exported", extra={"mlflow_run_id": run_id, "n_predictions": len(batch)})
        return run_id

    def export_ready_safely(self) -> str | None:
        """Like :meth:`export_ready`, but never raises (runs after the response).

        On failure the rows stay unexported and are retried with the next batch.
        """
        try:
            return self.export_ready()
        except Exception:  # noqa: BLE001 - a monitoring failure must not break serving; rows are retried
            logger.exception("inference batch export failed")
            return None
