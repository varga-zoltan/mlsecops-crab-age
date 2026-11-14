# MLflow tracking server + model registry.
# Same version as the mlflow client in requirements.txt (client == server).
FROM ghcr.io/mlflow/mlflow:v3.18.0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Postgres driver for the backend store (same version as in requirements.txt).
RUN pip install "psycopg[binary]==3.3.9"

# Run as a non-root user; artifacts are stored on a volume mounted at /mlartifacts.
RUN useradd --create-home --uid 10001 mlflow \
 && mkdir -p /mlartifacts \
 && chown mlflow:mlflow /mlartifacts
USER mlflow

EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=12 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=4)"]
