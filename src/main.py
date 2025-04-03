import logging
import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_pagination import add_pagination
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src.db import engine, scaffold_db
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



def get_log_level(env_var="LOG_LEVEL", default="INFO"):
    """
    Convert log level string from environment variable to logging module constant.

    Args:
        env_var: Name of the environment variable to check
        default: Default log level if environment variable is not set

    Returns:
        int: The logging level constant (e.g., logging.INFO)
    """
    log_level_str = os.getenv(env_var, default).upper()

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
SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
        traces_sample_rate=0.4,
        profiles_sample_rate=0.4,
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
    ],
    title="Honcho API",
    summary="An API for adding personalization to AI Apps",
    description="""This API is used to store data and get insights about users for AI
    applications""",
    version="0.0.17",
    contact={
        "name": "Plastic Labs",
        "url": "https://plasticlabs.ai",
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
    logger.error(f"{exc.__class__.__name__}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    if SENTRY_ENABLED:
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )
