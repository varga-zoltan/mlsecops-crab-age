# Crab Age application images, three targets on a common base:
#   runtime   - the FastAPI service (requirements.txt: mlflow, scikit-learn, FastAPI); no DVC
#   init      - runtime + DVC, for the one-shot job: dvc pull + first training
#   dashboard - the Streamlit + Evidently dashboard (requirements_dashboard.txt); no mlflow, no FastAPI
#
# Build: docker build --target runtime   -t crab-age-api:local .
#        docker build --target init      -t crab-age-init:local .
#        docker build --target dashboard -t crab-age-dashboard:local .

FROM python:3.12.16-slim AS base

# Slow networks: pip waits 60 s per read and retries 10 times (defaults: 15 s, 5).
# DO_NOT_TRACK switches off Evidently's usage telemetry. There is no git program in the images
# (the commit id is read from the mounted .git folder), so MLflow's GitPython warning is silenced.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    DO_NOT_TRACK=1 \
    GIT_PYTHON_REFRESH=quiet \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10

WORKDIR /app
# The build backend (same version as in pyproject.toml) is installed once here, so installing
# the project later (--no-build-isolation) does not download it again on every code change.
RUN useradd --create-home --uid 10001 app \
 && pip install setuptools==80.9.0

# Lineage fallback: if no .git folder is mounted, the commit comes from this build argument.
ARG GIT_SHA=unknown
ENV CRAB_GIT_SHA=${GIT_SHA}


FROM base AS runtime

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Then the code (changes often) and the files the jobs read at runtime.
COPY pyproject.toml params.yaml dvc.yaml dvc.lock ./
COPY src/ ./src/
RUN pip install --no-deps --no-build-isolation . \
 && rm -rf build \
 && mkdir -p /app/data/processed \
 && chown -R app:app /app

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
# The image has no git program, so DVC runs without SCM.
RUN dvc config --local core.no_scm true

HEALTHCHECK NONE
CMD ["python", "-m", "crab_age.bootstrap", "--params", "params.yaml"]


FROM base AS dashboard

# requirements.txt is only a constraint here: shared packages get the API's versions,
# but mlflow, FastAPI and the auth packages are not installed. pip constraints cannot
# have extras, so "[binary]" etc. are stripped first.
COPY requirements.txt requirements_dashboard.txt ./
RUN sed 's/\[[^]]*\]//' requirements.txt > /tmp/constraints.txt \
 && pip install -r requirements_dashboard.txt -c /tmp/constraints.txt \
 && rm /tmp/constraints.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY dashboard/ ./dashboard/
RUN pip install --no-deps --no-build-isolation . \
 && rm -rf build \
 && mkdir -p /app/data/processed \
 && chown -R app:app /app

USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer"]
