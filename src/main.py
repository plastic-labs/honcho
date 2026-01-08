import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import sentry_sdk
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_pagination import add_pagination
from pydantic import ValidationError
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src import prometheus
from src.cache.client import close_cache, init_cache
from src.config import settings
from src.db import engine, request_context
from src.exceptions import HonchoException
from src.routers import (
    keys,
    messages,
    observations,
    peers,
    sessions,
    webhooks,
    workspaces,
)
from src.security import create_admin_jwt
from src.sentry import initialize_sentry
from src.utils.logging import get_route_template

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint


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


# JWT Setup
async def setup_admin_jwt():
    token = create_admin_jwt()
    print(f"\n    ADMIN JWT: {token}\n")


def before_send(event: "Event", hint: "Hint | None") -> "Event | None":
    """Filter out events raised from known non-actionable exceptions before Sentry sees them."""
    if not hint:
        return event

    exc_info = hint.get("exc_info")
    if not exc_info:
        return event

    _, exc_value, _ = exc_info
    if isinstance(exc_value, HonchoException):
        return None

    # Filters out ValidationErrors and RequestValidationErrors (typically coming from Pydantic)
    if isinstance(exc_value, ValidationError | RequestValidationError):
        logger.info(f"Filtering out validation error from Sentry: {exc_value}")
        return None

    return event


# Sentry Setup
SENTRY_ENABLED = settings.SENTRY.ENABLED
if SENTRY_ENABLED:
    initialize_sentry(
        integrations=[
            StarletteIntegration(
                transaction_style="endpoint",
            ),
            FastApiIntegration(
                transaction_style="endpoint",
            ),
        ],
        before_send=before_send,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await init_cache()
    except Exception as e:
        logger.warning(
            "Error initializing cache in api process; proceeding without cache: %s", e
        )

    try:
        yield
    finally:
        # Import here to avoid circular import at module load time
        from src.vector_store import close_vector_store

        await close_vector_store()
        await close_cache()
        await engine.dispose()


app = FastAPI(
    lifespan=lifespan,
    servers=[
        {"url": "http://localhost:8000", "description": "Local Development Server"},
        {"url": "https://demo.honcho.dev", "description": "Demo Server"},
        {"url": "https://api.honcho.dev", "description": "Production SaaS Platform"},
    ],
    title="Honcho API",
    summary="The Identity Layer for the Agentic World",
    description="""Honcho is a platform for giving agents user-centric memory and social cognition""",
    version="2.5.0",
    contact={
        "name": "Plastic Labs",
        "url": "https://honcho.dev",
        "email": "hello@plasticlabs.ai",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0",
        "identifier": "AGPL-3.0-only",
        "url": "https://github.com/plastic-labs/honcho/blob/main/LICENSE",
    },
)

origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "https://demo.honcho.dev",
    "https://api.honcho.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


add_pagination(app)

app.include_router(workspaces.router, prefix="/v2")
app.include_router(peers.router, prefix="/v2")
app.include_router(sessions.router, prefix="/v2")
app.include_router(messages.router, prefix="/v2")
app.include_router(observations.router, prefix="/v2")
app.include_router(keys.router, prefix="/v2")
app.include_router(webhooks.router, prefix="/v2")

app.add_api_route("/metrics", prometheus.metrics, methods=["GET"])


# Global exception handlers
@app.exception_handler(HonchoException)
async def honcho_exception_handler(request: Request, exc: HonchoException):
    """Handle all Honcho-specific exceptions."""
    logger.error(f"{exc.__class__.__name__}: {exc.detail}", exc_info=exc)

    if prometheus.METRICS_ENABLED and request.url.path != "/metrics":
        template = get_route_template(request)
        prometheus.API_REQUESTS.labels(
            method=request.method,
            endpoint=template,
            status_code=str(exc.status_code),
        ).inc()

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    if prometheus.METRICS_ENABLED and request.url.path != "/metrics":
        template = get_route_template(request)
        prometheus.API_REQUESTS.labels(
            method=request.method,
            endpoint=template,
            status_code="500",
        ).inc()

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

    try:
        response = await call_next(request)

        # Track Prometheus metrics if enabled
        if prometheus.METRICS_ENABLED and request.url.path != "/metrics":
            template = get_route_template(request)
            prometheus.API_REQUESTS.labels(
                method=request.method,
                endpoint=template,
                status_code=str(response.status_code),
            ).inc()

        return response
    finally:
        request_context.reset(token)
