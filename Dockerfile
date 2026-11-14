# Crab Age application image, two targets:
#   runtime - the FastAPI service and the Streamlit dashboard; no DVC inside
#   init    - runtime + DVC, for the one-shot job: dvc pull + first training
#
# Build: docker build --target runtime -t crab-age-api:local .
#        docker build --target init    -t crab-age-init:local .

FROM python:3.12.16-slim AS runtime

# DO_NOT_TRACK switches off Evidently's usage telemetry.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    DO_NOT_TRACK=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Then the code (changes often) and the files the jobs read at runtime.
COPY pyproject.toml params.yaml dvc.yaml dvc.lock ./
COPY src/ ./src/
COPY dashboard/ ./dashboard/
RUN pip install --no-deps . \
 && rm -rf build \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /app/data/processed \
 && chown -R app:app /app

# Lineage: there is no .git in the image, so the commit comes in as a build argument.
ARG GIT_SHA=unknown
ENV CRAB_GIT_SHA=${GIT_SHA}

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
CMD ["uvicorn", "crab_age.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime AS init

USER root
# Same DVC version as in requirements_dev.txt.
RUN pip install dvc==3.67.1
COPY --chown=app:app .dvc/config .dvc/config
COPY --chown=app:app .dvcignore ./
COPY --chown=app:app data/raw/*.dvc data/raw/
USER app
# The image has no git repository, so DVC runs without SCM.
RUN dvc config --local core.no_scm true

HEALTHCHECK NONE
CMD ["python", "-m", "crab_age.bootstrap", "--params", "params.yaml"]
