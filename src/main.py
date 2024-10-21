import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination

from src.routers import (
    apps,
    collections,
    documents,
    messages,
    metamessages,
    sessions,
    users,
)

from .db import engine, scaffold_db

# Sentry Setup

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
        traces_sample_rate=0.4,
        profiles_sample_rate=0.4,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    scaffold_db()  # Scaffold Database on Startup
    yield
    await engine.dispose()


app = FastAPI(
    lifespan=lifespan,
    servers=[
        {"url": "http://127.0.0.1:8000", "description": "Local Development Server"},
        {"url": "https:/demo.honcho.dev", "description": "Demo Server"},
    ],
    title="Honcho API",
    summary="An API for adding personalization to AI Apps",
    description="""This API is used to store data and get insights about users for AI
    applications""",
    version="0.0.12",
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
