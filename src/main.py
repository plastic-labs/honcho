import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import psycopg
import sentry_sdk
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_pagination import add_pagination
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sqlalchemy.exc import DataError

from src._version import HONCHO_VERSION
from src.backlog import DeriverMetricsPoller
from src.cache.client import close_cache, init_cache
from src.config import settings
from src.db import (
    engine,
    register_db_connection_instrumentation,
    register_db_query_instrumentation,
    request_context,
)
from src.exceptions import HonchoException
from src.reconciler import ReconcilerScheduler, set_reconciler_scheduler
from src.routers import (
    conclusions,
    deriver_metrics,
    keys,
    messages,
    peers,
    scopes,
    sessions,
    webhooks,
    workspaces,
)
from src.startup import validate_embedding_schema
from src.telemetry import (
    initialize_telemetry_async,
    metrics_endpoint,
    prometheus_metrics,
    register_db_pool_collector,
    shutdown_telemetry,
)
from src.telemetry.client_context import (
    HEADER_AGENT_MODEL,
    HEADER_HOST,
    HEADER_PLUGIN,
    HEADER_USER_AGENT,
    reset_client_context,
    set_client_context,
)
from src.telemetry.logging import get_route_template
from src.telemetry.sentry import initialize_sentry


def get_log_level() -> int:
    """
    Convert log level string from settings to logging module constant.

    Returns:
        int: The logging level constant (e.g., logging.INFO)
    """
    log_level_str = settings.LOG_LEVEL.upper()

    log_levels = {
        "CRITICAL": logging.CRITICAL,  # 50
        "ERROR": logging.ERROR,  # 40
        "WARNING": logging.WARNING,  # 30
        "INFO": logging.INFO,  # 20
        "DEBUG": logging.DEBUG,  # 10
        "NOTSET": logging.NOTSET,  # 0
    }

    return log_levels.get(log_level_str, logging.INFO)


# Configure logging
logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress cashews Redis error logs (NoScriptError, ConnectionError, etc.)
# These are handled gracefully by SafeRedis and don't need full tracebacks
logging.getLogger("cashews.backends.redis.client").setLevel(logging.CRITICAL)


class MetricsAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "GET /metrics" not in msg


logging.getLogger("uvicorn.access").addFilter(MetricsAccessFilter())


# Sentry Setup
SENTRY_ENABLED = settings.SENTRY.ENABLED
if SENTRY_ENABLED:
    # before_send defaults to sentry.default_before_send (shared with the deriver).
    initialize_sentry(
        integrations=[
            StarletteIntegration(
                transaction_style="endpoint",
            ),
            FastApiIntegration(
                transaction_style="endpoint",
            ),
            # Explicit so DB-query spans are not reliant on auto-enabling.
            SqlalchemyIntegration(),
        ],
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize CloudEvents telemetry
    await initialize_telemetry_async()

    # Expose DB connection-pool stats for this API instance (no-op if metrics off)
    register_db_pool_collector("api")
    register_db_query_instrumentation("api")
    register_db_connection_instrumentation("api")

    # region ai
    # Zero-init bounded-label counters so a missing series signals a broken scrape,
    # not "no events" — see initialize_bounded_metrics. No-op if metrics off.
    # endregion
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")

    # Validate embedding schema before serving any traffic. Fails closed: if
    # the configured EMBEDDING_VECTOR_DIMENSIONS does not match the physical
    # pgvector columns, the process refuses to start rather than silently
    # writing wrong-dim vectors.
    await validate_embedding_schema(engine)

    try:
        await init_cache()
    except Exception as e:
        logger.warning(
            "Error initializing cache in api process; proceeding without cache: %s", e
        )

    deriver_metrics_poller = DeriverMetricsPoller()
    deriver_metrics.set_deriver_metrics_poller(deriver_metrics_poller)
    try:
        await deriver_metrics_poller.start()
    except Exception as e:
        logger.error("Failed to start backlog metrics poller: %s", e)

    reconciler_scheduler = None
    if settings.DERIVER.SCHEDULER == "api":
        reconciler_scheduler = ReconcilerScheduler()
        set_reconciler_scheduler(reconciler_scheduler)
        try:
            await reconciler_scheduler.start()
        except Exception as e:
            logger.error("Failed to start reconciler scheduler: %s", e)

    try:
        yield
    finally:
        # Import here to avoid circular import at module load time
        from src.vector_store import close_external_vector_store

        if reconciler_scheduler is not None:
            await reconciler_scheduler.shutdown()
        await deriver_metrics_poller.shutdown()
        deriver_metrics.set_deriver_metrics_poller(None)
        await close_external_vector_store()
        await close_cache()
        await engine.dispose()
        # Shutdown telemetry (flush CloudEvents buffer)
        await shutdown_telemetry()


app = FastAPI(
    lifespan=lifespan,
    servers=[{"url": "/"}],
    title="Honcho API",
    summary="The Identity Layer for the Agentic World",
    description="""Honcho is a platform for giving agents user-centric memory and social cognition.""",
    version=HONCHO_VERSION,
    contact={
        "name": "Plastic Labs",
        "url": "https://honcho.dev",
        "email": "hello@plasticlabs.ai",
    },
    license_info={
        # The 3.1 License Object treats `identifier` and `url` as mutually
        # exclusive, and emitting both makes the schema fail validation.
        "name": "GNU Affero General Public License v3.0",
        "url": "https://github.com/plastic-labs/honcho/blob/main/LICENSE",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


add_pagination(app)

app.include_router(workspaces.router, prefix="/v3")
app.include_router(peers.router, prefix="/v3")
app.include_router(sessions.router, prefix="/v3")
app.include_router(scopes.router, prefix="/v3")
app.include_router(messages.router, prefix="/v3")
app.include_router(conclusions.router, prefix="/v3")
app.include_router(keys.router, prefix="/v3")
app.include_router(webhooks.router, prefix="/v3")
app.include_router(deriver_metrics.router)

# Prometheus metrics endpoint
app.add_route("/metrics", metrics_endpoint, methods=["GET"])


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and container orchestration."""
    return {"status": "ok"}


# Global exception handlers
@app.exception_handler(HonchoException)
async def honcho_exception_handler(_request: Request, exc: HonchoException):
    """Handle all Honcho-specific exceptions."""
    logger.error(f"{exc.__class__.__name__}: {exc.detail}", exc_info=exc)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


class _AsciiJSONResponse(JSONResponse):
    """JSONResponse that escapes non-ASCII, so lone surrogates encode cleanly."""

    def render(self, content: object) -> bytes:
        return json.dumps(
            content, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("ascii")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Same body as FastAPI's default handler, but safe to encode.

    Validation errors echo the rejected `input`. When that input contains a lone
    surrogate (e.g. an emoji split mid-pair by a client), the default UTF-8
    render raises and the 422 turns into a 500.
    """
    return _AsciiJSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


# SQLSTATEs for input Postgres cannot store: a value out of range for its
# column type, and a `\u0000` escape in JSONB.
_INPUT_DATA_SQLSTATES = {"22003", "22P05"}

# A lone surrogate escape inside a JSONB value (free-form metadata or
# configuration) is 22P02, a SQLSTATE that also covers server-side bugs such as
# a bad cast, so only the surrogate message is treated as caller input.
_JSON_SURROGATE_SQLSTATE = "22P02"


def _is_input_data_error(exc: DataError) -> bool:
    orig = exc.orig
    sqlstate = getattr(orig, "sqlstate", None)
    if sqlstate in _INPUT_DATA_SQLSTATES:
        return True
    if sqlstate == _JSON_SURROGATE_SQLSTATE and "surrogate" in str(orig):
        return True
    # psycopg rejects NUL in text parameters client-side, without a SQLSTATE
    return isinstance(orig, psycopg.DataError) and "NUL (0x00)" in str(orig)


@app.exception_handler(DataError)
async def data_exception_handler(request: Request, exc: DataError):
    """422 for request values Postgres cannot store; anything else is a 500."""
    if _is_input_data_error(exc):
        sqlstate = getattr(exc.orig, "sqlstate", None) or "unknown"
        route = get_route_template(request)
        # Driver messages can echo SQL, parameters, or user data. Log only the
        # classification and static route template, never the exception/input.
        logger.warning(
            "Rejected unstorable input: sqlstate=%s route=%s",
            sqlstate,
            route,
            extra={"sqlstate": sqlstate, "route": route},
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Request contains a value that cannot be stored"},
        )
    return await global_exception_handler(request, exc)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    if SENTRY_ENABLED:
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )


@app.middleware("http")
async def track_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    # Create a request ID that includes endpoint information
    endpoint = re.sub(r"/[A-Za-z0-9_-]{21}", "", request.url.path).replace("/", "_")
    request_id = f"{request.method}:{endpoint}:{str(uuid.uuid4())[:8]}"

    # Store in request state and context var
    request.state.request_id = request_id
    token = request_context.set(f"api:{request_id}")
    # Optional client identity headers; the telemetry emitter injects these
    # into every event body emitted during this request.
    client_tokens = set_client_context(
        host=request.headers.get(HEADER_HOST),
        plugin=request.headers.get(HEADER_PLUGIN),
        agent_model=request.headers.get(HEADER_AGENT_MODEL),
        user_agent=request.headers.get(HEADER_USER_AGENT),
    )

    try:
        start_time = time.perf_counter()
        response = await call_next(request)

        # Track metrics if enabled
        if settings.METRICS.ENABLED:
            template = get_route_template(request)
            prometheus_metrics.record_api_request(
                method=request.method,
                endpoint=template,
                status_code=str(response.status_code),
                duration_seconds=time.perf_counter() - start_time,
            )

        return response
    finally:
        reset_client_context(client_tokens)
        request_context.reset(token)
