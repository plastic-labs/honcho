import logging
import os
import re
import uuid
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_pagination import add_pagination
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src.db import engine, request_context
from src.exceptions import HonchoException
from src.routers import (
    apps,
    collections,
    documents,
    keys,
    messages,
    metamessages,
    sessions,
    users,
)
from src.security import create_admin_jwt
from src.utils.logging import console, log_error_panel, setup_rich_logging

# Setup beautiful rich logging
setup_rich_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# JWT Setup
async def setup_admin_jwt():
    token = create_admin_jwt()
    console.print(f"\n[bold green]🔑 ADMIN JWT:[/] [dim]{token}[/]\n")


# Sentry Setup
SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:

    def before_send(event, hint):
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            # Filter out HonchoExceptions from being sent to Sentry
            if isinstance(exc_value, HonchoException):
                return None

        return event

    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        traces_sample_rate=0.4,
        profiles_sample_rate=0.4,
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
async def lifespan(app: FastAPI):
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
    version="1.1.0",
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

origins = ["http://localhost", "http://127.0.0.1:8000", "https://demo.honcho.dev"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/apps/{app_id}/users/{user_id}")

add_pagination(app)

app.include_router(apps.router, prefix="/v1")
app.include_router(users.router, prefix="/v1")
app.include_router(sessions.router, prefix="/v1")
app.include_router(messages.router, prefix="/v1")
app.include_router(metamessages.router, prefix="/v1")
app.include_router(collections.router, prefix="/v1")
app.include_router(documents.router, prefix="/v1")
app.include_router(keys.router, prefix="/v1")


# Global exception handlers
@app.exception_handler(HonchoException)
async def honcho_exception_handler(request: Request, exc: HonchoException):
    """Handle all Honcho-specific exceptions."""
    log_error_panel(logger, exc, f"Request: {request.method} {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    log_error_panel(
        logger, exc, f"Unhandled exception in {request.method} {request.url}"
    )
    if SENTRY_ENABLED:
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )


@app.middleware("http")
async def track_request(request: Request, call_next):
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
