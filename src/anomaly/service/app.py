import base64
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse

from anomaly.logging_setup import configure_logging
from anomaly.service.config import ServiceSettings
from anomaly.service.demo import DEMO_HTML
from anomaly.service.inferance import InvalidImageError, decode_image, predict
from anomaly.service.metrics import ServiceMetrics
from anomaly.service.registry import ModelRegistry, UnknownCategoryError
from anomaly.service.schemas import HealthResponse, ModelInfo, PredictionResponse, ReadyResponse

log = structlog.get_logger()
QUIET_PATHS = {"/healthz", "/readyz", "/metrics"}


def create_app(
    settings: ServiceSettings | None = None, registry: ModelRegistry | None = None
) -> FastAPI:
    settings = settings or ServiceSettings()
    configure_logging(settings.log_level, settings.json_logs)
    registry = registry or ModelRegistry(settings)
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        registry.load()  # fail fast: a service without models should not start
        metrics.model_info.labels(registry.manifest.version).set(1)
        yield

    app = FastAPI(title="Anomaly detection service", version="1.1.0", lifespan=lifespan)

    async def require_api_key(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        secret = settings.api_key
        if secret is None:
            return
        supplied = x_api_key
        if supplied is None and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()  # lets Prometheus use its standard auth setting
        if not secrets.compare_digest((supplied or "").encode(), secret.get_secret_value().encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:64]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_failed", method=request.method, path=request.url.path)
            raise
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        path_label = getattr(route, "path", "unmatched")  # route template keeps cardinality low
        metrics.requests.labels(request.method, path_label, str(response.status_code)).inc()
        metrics.request_seconds.labels(path_label).observe(duration)
        response.headers["X-Request-ID"] = request_id
        emit = log.debug if request.url.path in QUIET_PATHS else log.info
        emit(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration * 1000.0, 1),
        )
        return response

    def reject(reason: str, status_code: int, detail: str, headers: dict | None = None):
        metrics.rejected.labels(reason).inc()
        raise HTTPException(status_code=status_code, detail=detail, headers=headers)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        """Liveness: the process is up."""
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    def readyz() -> ReadyResponse:
        """Readiness: models are loaded and the service can take traffic."""
        if not registry.ready or registry.manifest is None:
            raise HTTPException(status_code=503, detail="Models are not loaded yet.")
        return ReadyResponse(
            status="ready",
            model_version=registry.manifest.version,
            categories=registry.categories,
            device=str(registry.device),
        )

    @app.get("/v1/models", response_model=list[ModelInfo], dependencies=[Depends(require_api_key)])
    def list_models() -> list[ModelInfo]:
        if not registry.ready or registry.manifest is None:
            raise HTTPException(status_code=503, detail="Models are not loaded yet.")
        return [
            ModelInfo(
                category=name,
                threshold=registry.manifest.categories[name].threshold,
                bank_size=registry.manifest.categories[name].bank_size,
                test_image_auroc=registry.manifest.categories[name].test_image_auroc,
                test_pixel_auroc=registry.manifest.categories[name].test_pixel_auroc,
            )
            for name in registry.categories
        ]

    @app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
    def predict_image(
        category: str = Query(..., description="Product category of the image"),
        include_heatmap: bool = Query(False, description="Also return an overlay PNG (base64)"),
        file: UploadFile = File(...),
    ) -> PredictionResponse:
        if not registry.ready or registry.manifest is None:
            raise HTTPException(status_code=503, detail="Models are not loaded yet.")
        try:
            loaded = registry.get(category)
        except UnknownCategoryError:
            reject(
                "unknown_category", 404,
                f"Unknown category '{category}'. Available: {registry.categories}",
            )

        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            reject("too_large", 413, "Uploaded file is too large.")
        try:
            image = decode_image(data, settings.max_image_pixels)
        except InvalidImageError as exc:
            reject("invalid_image", 400, str(exc))

        if not registry.semaphore.acquire(timeout=settings.queue_timeout_seconds):
            reject("busy", 503, "Server busy.", headers={"Retry-After": "1"})
        try:
            with metrics.in_flight.track_inprogress():
                result = predict(loaded, image, registry.transform, registry.device, include_heatmap)
        finally:
            registry.semaphore.release()

        is_anomalous = result.score >= loaded.threshold
        ratio = result.score / max(loaded.threshold, 1e-8)
        metrics.predictions.labels(category, "anomalous" if is_anomalous else "normal").inc()
        metrics.inference_seconds.labels(category).observe(result.latency_ms / 1000.0)
        metrics.score_ratio.labels(category).observe(ratio)
        snapshot = registry.monitor(category).observe(ratio, result.input_mean)
        if snapshot is not None:
            metrics.update_drift(category, snapshot)

        log.info(
            "prediction",
            category=category,
            score=round(result.score, 4),
            ratio=round(ratio, 3),
            is_anomalous=is_anomalous,
            latency_ms=round(result.latency_ms, 1),
        )
        return PredictionResponse(
            category=category,
            anomaly_score=round(result.score, 6),
            threshold=loaded.threshold,
            is_anomalous=is_anomalous,
            latency_ms=round(result.latency_ms, 1),
            model_version=registry.manifest.version,
            heatmap_png_base64=(
                base64.b64encode(result.overlay_png).decode() if result.overlay_png else None
            ),
        )

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
        def metrics_endpoint() -> Response:
            return Response(content=metrics.render(), media_type=metrics.content_type)

    if settings.demo_enabled:

        @app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
        def demo() -> str:
            return DEMO_HTML

    return app