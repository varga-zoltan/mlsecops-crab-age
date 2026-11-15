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
| MLflow artifacts | Docker volume served by the MLflow server (`--serve-artifacts`), no MinIO | Official MinIO image is no longer published; the third-party rebuild is unmaintained (supply chain risk) |
| Host ports | API 8000, MLflow 5500, Streamlit 8501; Postgres not published | Port 5000 is used by AirPlay on macOS; smaller attack surface |
| Images | One `Dockerfile`, targets `runtime` (API, no DVC) and `init` (+ DVC); MLflow server: `ghcr.io/mlflow/mlflow:v3.18.0` + psycopg, non-root | Smaller API image; CI builds both targets |
| Drift report | Evidently 0.7.23 `DataDriftPreset` (new API); reference = `train.csv`, current = latest N logged requests (default 500, min 30) | Recent drift is not hidden by old traffic; too few rows give noisy tests |
| Dataset drift rule | Drift if >= 50% of feature columns drift (Evidently default) | Single columns drift by chance even on real held-out data |
| Traffic simulator | `crab_age.monitoring.simulate_traffic`: `normal` samples real test rows, `drift` scales them; runs inside the api container | Reproducible demo without local Python |
| Telemetry | `DO_NOT_TRACK=1` (Evidently), `--browser.gatherUsageStats=false` (Streamlit) | No usage data leaves the stack |
| Lineage in Docker | compose mounts `.git` read-only into `init` and `api`; `crab_age.lineage` reads HEAD (loose ref, `packed-refs`, detached HEAD) without the git program | `git_sha` works with a plain `docker compose up`; uncommitted changes (`-dirty`) cannot be detected without git |
| Settings source | Environment variables only (no `.env` reading in Python); unknown `CRAB_*` variables are an error | Compose's `.env` may hold non-CRAB variables; typos must not fall back silently |

## Open items to remember

- **Image size:** the app image is ~1.5 GB (full `mlflow` package); `mlflow-skinny` could shrink it.
- **Tests:** Starlette warns that `httpx` with its TestClient is deprecated (`httpx2` recommended); revisit when FastAPI follows.
