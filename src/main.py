import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import sentry_sdk
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_pagination import add_pagination

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src.config import settings
from src.db import engine, request_context
from src.exceptions import HonchoException
from src.routers import (
    keys,
    messages,
    peers,
    sessions,
    workspaces,
)
from src.security import create_admin_jwt


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


# JWT Setup
async def setup_admin_jwt():
    token = create_admin_jwt()
    print(f"\n    ADMIN JWT: {token}\n")


# Sentry Setup
SENTRY_ENABLED = settings.SENTRY.ENABLED
if SENTRY_ENABLED:

    def before_send(event: "Event", hint: "Hint") -> "Event | None":
        if "exc_info" in hint:
            _, exc_value, _ = hint["exc_info"]
            # Filter out HonchoExceptions from being sent to Sentry
            if isinstance(exc_value, HonchoException):
                return None

        return event

    # Sentry SDK's default behavior:
    # - Captures INFO+ level logs as breadcrumbs
    # - Captures ERROR+ level logs as Sentry events
    #
    # For custom log levels, use the LoggingIntegration class:
    # sentry_sdk.init(..., integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)])
    sentry_sdk.init(
        dsn=settings.SENTRY.DSN,
        traces_sample_rate=settings.SENTRY.TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY.PROFILES_SAMPLE_RATE,
        before_send=before_send,
        integrations=[
            StarletteIntegration(
                transaction_style="endpoint",
            ),
            FastApiIntegration(
                transaction_style="endpoint",
            ),
        ],
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
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
    version="2.0.2",
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
app.include_router(keys.router, prefix="/v2")


# Global exception handlers
@app.exception_handler(HonchoException)
async def honcho_exception_handler(_request: Request, exc: HonchoException):
    """Handle all Honcho-specific exceptions."""
    logger.error(f"{exc.__class__.__name__}: {exc.detail}", exc_info=exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


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

    try:
        return await call_next(request)
    finally:
        request_context.reset(token)
