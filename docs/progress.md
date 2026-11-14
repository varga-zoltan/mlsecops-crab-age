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

## Open items to remember

- **Docker:** the MLflow server image must be the same version as the client:
  `ghcr.io/mlflow/mlflow:v3.18.0` (check the arm64 variant).
