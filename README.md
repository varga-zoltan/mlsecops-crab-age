# Crab Age MLOps

MLSecOps semester project: crab age regression with an automated MLOps pipeline
(DVC, MLflow, FastAPI, Docker Compose, Evidently, Streamlit).

Work in progress. The full documentation will be added here.

## Quick start

```bash
git clone https://github.com/varga-zoltan/mlsecops-crab-age.git
cd mlsecops-crab-age
docker compose up
```

No `.env` file or secrets are needed. The first start builds the images, pulls the data
with DVC and trains the first model, so it takes a few minutes. The API starts when this is done.

| Service | URL | Notes |
|---|---|---|
| API (Swagger UI) | http://localhost:8000/docs | `/token`, `/health`, `/version`, `/train`, `/predict` |
| MLflow | http://localhost:5500 | experiments, model registry (`crab-age@champion`) |
| Monitoring dashboard | http://localhost:8501 | Streamlit + Evidently data drift report |

The ports are bound to `127.0.0.1`, so the services are reachable only from this machine,
not from other machines on the network. Postgres is only reachable from the containers.

**Login (local development only):** user `crab`, password `crab-dev-password`.
These are local defaults in `docker-compose.yaml`. For any real use, override
`CRAB_API_PASSWORD_HASH`, `CRAB_JWT_SECRET_KEY` and `POSTGRES_PASSWORD` with environment
variables or a `.env` file (see `.env.example`).

Stop: `docker compose down`. Reset everything (data, models, logs): `docker compose down -v`.

## Dataset

[Crab Age Prediction](https://www.kaggle.com/datasets/sidhus/crab-age-prediction) (Kaggle, CC0).
Target: `Age` (months), regression.

## Data preprocessing

DVC stage `prepare` (`dvc repro`), code in `src/crab_age/data/`, settings in `params.yaml`.

1. Load the raw CSV and rename columns to snake_case (`Shucked Weight` → `shucked_weight`).
2. Validate the schema: all columns present, no missing values, numeric types, `sex` in M/F/I.
3. Drop physically impossible rows:

   | Rule | Rows |
   |---|---|
   | a size or weight is 0 or negative | 2 |
   | `height` > 0.8 × `diameter` (typical ratio: 0.34) | 2 |
   | `shucked_weight` > `weight` | 4 |
   | `diameter` > `length` | 1 |

   3893 rows in, 3884 rows out. The counts are written to `metrics/prepare.json`.
4. Split into a fixed train (80%) and test (20%) set with a fixed seed.
   Every model is evaluated on the same test set, so champion and challenger are comparable.

Outputs in `data/processed/` (tracked by DVC): `crab_age_clean.csv`, `train.csv`, `test.csv`.

Scaling and encoding are not done here. They are part of the sklearn `Pipeline`,
so training and serving use the same transformations.

## Monitoring and drift report

Every `/predict` request and its prediction is logged in the Postgres database `monitoring`.
The dashboard (http://localhost:8501) compares the latest N logged requests (slider, default 500)
with the training data using Evidently's `DataDriftPreset`. Dataset drift is reported when at least
half of the feature columns drift (Evidently's default rule).

Right after the first start the log is empty, so the dashboard asks for data. Generate traffic
inside the running stack (no local Python needed):

```bash
docker compose exec api python -m crab_age.monitoring.simulate_traffic --mode normal --n 500
docker compose exec api python -m crab_age.monitoring.simulate_traffic --mode drift --n 500
```

`normal` sends real crabs from the held-out test set (expected: no dataset drift),
`drift` sends the same crabs bigger and heavier, with more juveniles (expected: dataset drift).
With the project venv active, the same works from the host: `python scripts/simulate_traffic.py --mode drift`.

Evidently and Streamlit usage telemetry is switched off (`DO_NOT_TRACK=1`, `--browser.gatherUsageStats=false`).

## Technical decisions

- MLflow stores artifacts on a Docker volume, not in MinIO: the official MinIO image is no longer
  published and the third-party rebuild is not maintained, which is a supply chain risk.

## Assignment

See [docs/assignment.md](docs/assignment.md).
