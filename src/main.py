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

from src import crud
from src.db import SessionLocal, engine, scaffold_db
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# JWT Setup
async def setup_admin_jwt():
    db = SessionLocal()
    try:
        token = create_admin_jwt()

        # if admin key is not already in the database, save it
        key = await crud.get_key(db, token)
        if key:
            logger.info("Admin key already exists in database")
        else:
            logger.info("Creating new admin key in database")
            await crud.create_key(db, token)
        print(f"\n    ADMIN JWT: {token}\n")
    finally:
        await db.close()


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
    scaffold_db()  # Scaffold Database on Startup
    await setup_admin_jwt()  # Add JWT setup
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
    version="0.0.16",
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
app.include_router(metamessages.router_user_level, prefix="/v1")
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
