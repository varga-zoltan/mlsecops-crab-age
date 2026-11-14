# Progress and key decisions

## Key decisions

| Topic | Decision | Reason |
|---|---|---|
| DVC remote | Public GitHub repo `mlsecops-crab-age-data`, read over HTTP | `dvc pull` works on a fresh clone without credentials |
| Preprocessing | Invalid rows dropped in the `prepare` stage; scaling and encoding inside the sklearn Pipeline | No training/serving skew |
| Train/test split | Fixed split in the `prepare` stage | Champion and challenger are compared on the same test set |
| Model search | Ridge, RandomForest, HistGradientBoosting, each with `RandomizedSearchCV` (5-fold CV) | Automatic improvement of model quality |
| Main metric | MAE (months) | Easy to explain, robust to outliers |
| Acceptance gate | Challenger MAE <= champion MAE **and** MAE <= 2.0 | A worse model never becomes champion |
| MLflow version | `mlflow==3.18.0` everywhere | `3.1.0` conflicts with the `packaging` version that `pre-commit` needs; newer release has security fixes |
| Model format | MLflow sklearn flavor with `skops` serialization (not pickle) and a signature | Loading a skops file cannot run arbitrary code. Only two tree types are on the trusted list (`SKOPS_TRUSTED_TYPES` in `model/train.py`) |
| Champion comparison | The current champion is re-evaluated on the current test set | Fair comparison even if the data changed |
| Gate exit code | `python -m crab_age.train` exits 0 (new champion) or 1 (rejected) | Training contract: artifact + exit code |
| Prediction log | Postgres table `predictions` in a separate database `monitoring` (not the MLflow DB); SQLite in tests | Does not mix with MLflow's tables and migrations |
| Inference batches | Built from unexported DB rows, exported to MLflow (`crab-age-inference`) when >= 10 rows wait | Nothing is lost on restart; batch size in `CRAB_INFERENCE_BATCH_SIZE` |
| `/train` | Synchronous, one run at a time (409 otherwise); a failed reload keeps the old champion serving | Simple and testable; serving never breaks |
| API user | One user from env: `CRAB_API_USERNAME` + Argon2 hash in `CRAB_API_PASSWORD_HASH`; `CRAB_JWT_SECRET_KEY` >= 64 bytes (HS512) | No password in code |
| Settings source | Environment variables only (no `.env` reading in Python); unknown `CRAB_*` variables are an error | Compose's `.env` may hold non-CRAB variables; typos must not fall back silently |

## Open items to remember

- **Docker:** the MLflow server image must be the same version as the client:
  `ghcr.io/mlflow/mlflow:v3.18.0` (check the arm64 variant).
- **Docker:** Postgres needs a second database `monitoring` (init script), the API gets
  `CRAB_MONITORING_DB_URL=postgresql+psycopg://...@postgres:5432/monitoring`.
- **Docker:** the Argon2 hash contains `$` characters: write them as `$$` in `docker-compose.yaml`.
- **Docker:** API start command: `uvicorn crab_age.api.main:create_app --factory --host 0.0.0.0 --port 8000`.
- **README:** document the dev login (user `crab`, dev password) and that it must be overridden via env outside local dev.
- **Tests:** Starlette warns that `httpx` with its TestClient is deprecated (`httpx2` recommended); revisit when FastAPI follows.
