"""FastAPI application: /token, /health, /version, /train, /predict.

Start with: uvicorn crab_age.api.main:create_app --factory --host 0.0.0.0 --port 8000
"""

import logging
import re
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version as package_version
from typing import Annotated, Final

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

from crab_age.api.auth import authenticate, create_access_token, get_current_user
from crab_age.api.model_service import ModelHolder, ModelNotLoadedError, PredictionService
from crab_age.api.schemas import (
    CrabFeatures,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
    Token,
    TrainResponse,
    VersionResponse,
)
from crab_age.config import Settings, get_settings
from crab_age.lineage import git_sha
from crab_age.logging_setup import get_request_id, new_request_id, reset_request_id, set_request_id, setup_logging
from crab_age.model.train import TrainResult, read_train_params, run
from crab_age.monitoring.inference_export import InferenceExporter
from crab_age.monitoring.prediction_log import PredictionStore

logger = logging.getLogger(__name__)

API_NAME: Final = "Crab Age API"
REQUEST_ID_HEADER: Final = "X-Request-ID"
# A client-sent request id is only reused if it is short and harmless (no log injection).
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

TrainRunner = Callable[[], TrainResult]
CurrentUser = Annotated[str, Depends(get_current_user)]


def _default_train_runner(settings: Settings) -> TrainRunner:
    return lambda: run(read_train_params(settings.params_path))


def _check_body_size(request: Request, max_bytes: int) -> JSONResponse | None:
    """Reject a body above ``max_bytes`` before it is read (simple DoS protection).

    Chunked uploads (no Content-Length) are refused, otherwise the limit could
    be bypassed. A request without any body (e.g. POST /train) is fine.
    """
    length = request.headers.get("content-length")
    if length is None:
        if "transfer-encoding" in request.headers:
            return JSONResponse(status_code=status.HTTP_411_LENGTH_REQUIRED, content={"detail": "Length required"})
        return None
    if not length.isdigit():
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": "Invalid Content-Length"})
    if int(length) > max_bytes:
        return JSONResponse(status_code=status.HTTP_413_CONTENT_TOO_LARGE, content={"detail": "Request body too large"})
    return None


def create_app(
    settings: Settings | None = None,
    *,
    holder: ModelHolder | None = None,
    store: PredictionStore | None = None,
    train_runner: TrainRunner | None = None,
) -> FastAPI:
    """Build the FastAPI app. Arguments are for tests; production uses the defaults."""
    settings = settings or get_settings()
    store = store or PredictionStore(settings.monitoring_db_url.get_secret_value())
    holder = holder or ModelHolder(settings.registered_model_name)
    service = PredictionService(holder, store)
    exporter = InferenceExporter(store, settings.inference_experiment, settings.inference_batch_size)
    runner = train_runner or _default_train_runner(settings)
    train_lock = threading.Lock()
    commit = git_sha()  # read once: the code does not change while the API runs

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        setup_logging()
        store.init_schema()
        if not holder.reload():
            logger.warning("API started without a champion model; /predict returns 503 until one is trained")
        yield

    app = FastAPI(
        title=API_NAME,
        version=package_version("crab-age"),
        summary="Predicts the age of a crab (months) from its measurements.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Request id, body size limit, timing and one access log line per request."""
        sent_id = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = sent_id if _SAFE_REQUEST_ID.match(sent_id) else new_request_id()
        token = set_request_id(request_id)
        start = time.perf_counter()
        try:
            rejected = _check_body_size(request, settings.max_body_bytes)
            response: Response = rejected if rejected is not None else await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 1),
                },
            )
            return response
        finally:
            reset_request_id(token)

    @app.post("/token", response_model=Token, tags=["auth"], responses={401: {"model": ErrorResponse}})
    def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
        """Exchange username + password for a bearer token."""
        if not authenticate(settings, form.username, form.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Token(access_token=create_access_token(settings, form.username))

    @app.get("/health", response_model=HealthResponse, tags=["info"])
    def health() -> HealthResponse:
        """Liveness check (used by the Docker healthcheck)."""
        return HealthResponse(model_loaded=holder.current is not None)

    @app.get("/version", response_model=VersionResponse, tags=["info"])
    def api_version() -> VersionResponse:
        """API version, git commit and the served model version."""
        current = holder.current
        return VersionResponse(
            name=API_NAME,
            version=package_version("crab-age"),
            git_sha=commit,
            model_name=settings.registered_model_name,
            model_version=current.version if current else None,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid token"},
            413: {"model": ErrorResponse, "description": "Request body too large"},
            503: {"model": ErrorResponse, "description": "No champion model loaded"},
        },
    )
    def predict(features: CrabFeatures, background: BackgroundTasks, _: CurrentUser) -> PredictResponse:
        """Predict the age of one crab with the champion model."""
        try:
            result = service.predict(features)
        except ModelNotLoadedError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No champion model loaded")
        background.add_task(exporter.export_ready_safely)
        return PredictResponse(
            predicted_age_months=result.predicted_age_months,
            model_version=result.model_version,
            request_id=get_request_id() or new_request_id(),
        )

    @app.post(
        "/train",
        response_model=TrainResponse,
        tags=["training"],
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid token"},
            409: {"model": ErrorResponse, "description": "A training run is already in progress"},
        },
    )
    def train(_: CurrentUser) -> TrainResponse:
        """Search, train, register and gate a new model (takes ~15-30 s).

        If the challenger passes the acceptance gate it becomes the champion and
        is loaded immediately; if loading fails, the previous model keeps serving.
        """
        if not train_lock.acquire(blocking=False):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A training run is already in progress")
        try:
            result = runner()
            reloaded = holder.reload() if result.decision.accepted else False
        finally:
            train_lock.release()
        return TrainResponse(
            accepted=result.decision.accepted,
            reason=result.decision.reason,
            best_model=result.model_name,
            model_version=result.model_version,
            run_id=result.run_id,
            test_mae=result.test_metrics["mae"],
            test_rmse=result.test_metrics["rmse"],
            test_r2=result.test_metrics["r2"],
            champion_mae=result.decision.champion_mae,
            model_reloaded=reloaded,
        )

    return app
