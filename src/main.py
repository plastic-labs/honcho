import json
import logging
import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi_pagination import add_pagination
from opentelemetry import trace
from opentelemetry._logs import (
    set_logger_provider,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

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

# Otel Setup

DEBUG_LOG_OTEL_TO_PROVIDER = (
    os.getenv("DEBUG_LOG_OTEL_TO_PROVIDER", "False").lower() == "true"
)
DEBUG_LOG_OTEL_TO_CONSOLE = (
    os.getenv("DEBUG_LOG_OTEL_TO_CONSOLE", "False").lower() == "true"
)


def otel_get_env_vars():
    otel_http_headers = {}
    try:
        decoded_http_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        key_values = decoded_http_headers.split(",")
        for key_value in key_values:
            key, value = key_value.split("=")
            otel_http_headers[key] = value

    except Exception as e:
        print(f"Error parsing OTEL_ENDPOINT_HTTP_HEADERS: {str(e)}")
    otel_endpoint_url = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    return otel_endpoint_url, otel_http_headers


def otel_trace_init():
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create({}),
        ),
    )
    if DEBUG_LOG_OTEL_TO_PROVIDER:
        otel_endpoint_url, otel_http_headers = otel_get_env_vars()
        if otel_endpoint_url is not None:
            otel_endpoint_url = otel_endpoint_url + "/v1/traces"
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=otel_endpoint_url, headers=otel_http_headers
        )
        trace.get_tracer_provider().add_span_processor(  # type: ignore
            BatchSpanProcessor(otlp_span_exporter)
        )
    if DEBUG_LOG_OTEL_TO_CONSOLE:
        trace.get_tracer_provider().add_span_processor(  # type: ignore
            SimpleSpanProcessor(ConsoleSpanExporter())
        )


def otel_logging_init():
    # ------------Logging
    # Set logging level
    # CRITICAL = 50
    # ERROR = 40
    # WARNING = 30
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0
    # default = WARNING
    log_level = str(os.getenv("OTEL_PYTHON_LOG_LEVEL", "INFO")).upper()
    if log_level == "CRITICAL":
        log_level = logging.CRITICAL
        print(f"Using log level: CRITICAL / {log_level}")
    elif log_level == "ERROR":
        log_level = logging.ERROR
        print(f"Using log level: ERROR / {log_level}")
    elif log_level == "WARNING":
        log_level = logging.WARNING
        print(f"Using log level: WARNING / {log_level}")
    elif log_level == "INFO":
        log_level = logging.INFO
        print(f"Using log level: INFO / {log_level}")
    elif log_level == "DEBUG":
        log_level = logging.DEBUG
        print(f"Using log level: DEBUG / {log_level}")
    elif log_level == "NOTSET":
        log_level = logging.INFO
        print(f"Using log level: NOTSET / {log_level}")
    # ------------ Opentelemetry logging initialization

    logger_provider = LoggerProvider(resource=Resource.create({}))
    set_logger_provider(logger_provider)
    if DEBUG_LOG_OTEL_TO_CONSOLE:
        console_log_exporter = ConsoleLogExporter()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(console_log_exporter)
        )
    if DEBUG_LOG_OTEL_TO_PROVIDER:
        otel_endpoint_url, otel_http_headers = otel_get_env_vars()
        if otel_endpoint_url is not None:
            otel_endpoint_url = otel_endpoint_url + "/v1/logs"
        otlp_log_exporter = OTLPLogExporter(
            endpoint=otel_endpoint_url, headers=otel_http_headers
        )
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(otlp_log_exporter)
        )

    # otel_log_handler = FormattedLoggingHandler(logger_provider=logger_provider)
    otel_log_handler = LoggingHandler(
        level=logging.NOTSET, logger_provider=logger_provider
    )

    otel_log_handler.setLevel(log_level)
    # This has to be called first before logger.getLogger().addHandler() so that it can call logging.basicConfig first to set the logging format
    # based on the environment variable OTEL_PYTHON_LOG_FORMAT
    LoggingInstrumentor(log_level=log_level).instrument(log_level=log_level)
    # logFormatter = logging.Formatter(os.getenv("OTEL_PYTHON_LOG_FORMAT", None))
    # otel_log_handler.setFormatter(logFormatter)
    logging.getLogger().addHandler(otel_log_handler)


OPENTELEMTRY_ENABLED = os.getenv("OPENTELEMETRY_ENABLED", "False").lower() == "true"

# Instrument SQLAlchemy
if OPENTELEMTRY_ENABLED:
    otel_trace_init()
    otel_logging_init()

    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


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
    version="0.0.11",
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

if OPENTELEMTRY_ENABLED:
    FastAPIInstrumentor().instrument_app(app)

router = APIRouter(prefix="/apps/{app_id}/users/{user_id}")

add_pagination(app)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    current_span = trace.get_current_span()
    if (current_span is not None) and (current_span.is_recording()):
        current_span.set_attributes({
            "http.status_text": str(exc.detail),
            "otel.status_description": f"{exc.status_code} / {str(exc.detail)}",
            "otel.status_code": "ERROR",
        })
    return PlainTextResponse(
        json.dumps({"detail": str(exc.detail)}), status_code=exc.status_code
    )


app.include_router(apps.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(messages.router)
app.include_router(metamessages.router)
app.include_router(collections.router)
app.include_router(documents.router)
