import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Sequence

import sentry_sdk
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import PlainTextResponse
from fastapi_pagination import Page, add_pagination
from fastapi_pagination.ext.sqlalchemy import paginate
from opentelemetry import trace
from opentelemetry._logs import (
    SeverityNumber,
    get_logger,
    get_logger_provider,
    set_logger_provider,
    std_to_otel,
)

# from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import crud, schemas
from .db import SessionLocal, engine, scaffold_db

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
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(otlp_span_exporter)
        )
    if DEBUG_LOG_OTEL_TO_CONSOLE:
        trace.get_tracer_provider().add_span_processor(
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

    SQLAlchemyInstrumentor().instrument(engine=engine)

# Sentry Setup

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    scaffold_db()  # Scaffold Database on Startup
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

if OPENTELEMTRY_ENABLED:
    FastAPIInstrumentor().instrument_app(app)


router = APIRouter(prefix="/apps/{app_id}/users/{user_id}")

# Create a Limiter instance
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# Add SlowAPI middleware to the application
app.state.limiter = limiter
app.add_exception_handler(
    exc_class_or_status_code=RateLimitExceeded,
    handler=_rate_limit_exceeded_handler,  # type: ignore
)
app.add_middleware(SlowAPIMiddleware)


add_pagination(app)


async def get_db():
    """FastAPI Dependency Generator for Database"""
    db: AsyncSession = SessionLocal()
    try:
        yield db
    finally:
        await db.close()


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    current_span = trace.get_current_span()
    if (current_span is not None) and (current_span.is_recording()):
        current_span.set_attributes(
            {
                "http.status_text": str(exc.detail),
                "otel.status_description": f"{exc.status_code} / {str(exc.detail)}",
                "otel.status_code": "ERROR",
            }
        )
    return PlainTextResponse(
        json.dumps({"detail": str(exc.detail)}), status_code=exc.status_code
    )


########################################################
# App Routes
########################################################
@app.get("/apps/{app_id}", response_model=schemas.App)
async def get_app(
    request: Request,
    app_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get an App by ID

    Args:
        app_id (uuid.UUID): The ID of the app

    Returns:
        schemas.App: App object

    """
    app = await crud.get_app(db, app_id=app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@app.get("/apps/name/{name}", response_model=schemas.App)
async def get_app_by_name(
    request: Request,
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get an App by Name

    Args:
        app_name (str): The name of the app

    Returns:
        schemas.App: App object

    """
    app = await crud.get_app_by_name(db, name=name)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


@app.post("/apps", response_model=schemas.App)
async def create_app(
    request: Request,
    app: schemas.AppCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create an App

    Args:
        app (schemas.AppCreate): The App object containing any metadata

    Returns:
        schemas.App: Created App object

    """

    return await crud.create_app(db, app=app)


@app.get("/apps/get_or_create/{name}", response_model=schemas.App)
async def get_or_create_app(
    request: Request,
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get or Create an App

    Args:
        app_name (str): The name of the app

    Returns:
        schemas.App: App object

    """
    app = await crud.get_app_by_name(db, name=name)
    if app is None:
        app = await crud.create_app(db, app=schemas.AppCreate(name=name))
    return app


@app.put("/apps/{app_id}", response_model=schemas.App)
async def update_app(
    request: Request,
    app_id: uuid.UUID,
    app: schemas.AppUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an App

    Args:
        app_id (uuid.UUID): The ID of the app to update
        app (schemas.AppUpdate): The App object containing any new metadata

    Returns:
        schemas.App: The App object of the updated App

    """
    honcho_app = await crud.update_app(db, app_id=app_id, app=app)
    if honcho_app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return honcho_app


########################################################
# User Routes
########################################################


@app.post("/apps/{app_id}/users", response_model=schemas.User)
async def create_user(
    request: Request,
    app_id: uuid.UUID,
    user: schemas.UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user (schemas.UserCreate): The User object containing any metadata

    Returns:
        schemas.User: Created User object

    """
    return await crud.create_user(db, app_id=app_id, user=user)


@app.get("/apps/{app_id}/users", response_model=Page[schemas.User])
async def get_users(
    request: Request,
    app_id: uuid.UUID,
    reverse: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Get All Users for an App

    Args:
        app_id (uuid.UUID): The ID of the app representing the client
        application using honcho

    Returns:
        list[schemas.User]: List of User objects

    """
    return await paginate(db, await crud.get_users(db, app_id=app_id, reverse=reverse))


@app.get("/apps/{app_id}/users/{name}", response_model=schemas.User)
async def get_user_by_name(
    request: Request,
    app_id: uuid.UUID,
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        schemas.User: User object

    """
    return await crud.get_user_by_name(db, app_id=app_id, name=name)


@app.get("/apps/{app_id}/users/get_or_create/{name}", response_model=schemas.User)
async def get_or_create_user(
    request: Request, app_id: uuid.UUID, name: str, db: AsyncSession = Depends(get_db)
):
    """Get or Create a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        schemas.User: User object

    """
    user = await crud.get_user_by_name(db, app_id=app_id, name=name)
    if user is None:
        user = await crud.create_user(
            db, app_id=app_id, user=schemas.UserCreate(name=name)
        )
    return user


@app.put("/apps/{app_id}/users/{user_id}", response_model=schemas.User)
async def update_user(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    user: schemas.UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        user (schemas.UserCreate): The User object containing any metadata

    Returns:
        schemas.User: Updated User object

    """
    return await crud.update_user(db, app_id=app_id, user_id=user_id, user=user)


########################################################
# Session Routes
########################################################


@router.get("/sessions", response_model=Page[schemas.Session])
async def get_sessions(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    location_id: Optional[str] = None,
    is_active: Optional[bool] = False,
    reverse: Optional[bool] = False,
    db: AsyncSession = Depends(get_db),
):
    """Get All Sessions for a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        location_id (str, optional): Optional Location ID representing the location of a
        session

    Returns:
        list[schemas.Session]: List of Session objects

    """
    return await paginate(
        db,
        await crud.get_sessions(
            db,
            app_id=app_id,
            user_id=user_id,
            location_id=location_id,
            reverse=reverse,
            is_active=is_active,
        ),
    )


@router.post("/sessions", response_model=schemas.Session)
async def create_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session: schemas.SessionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a Session for a User

    Args:
        app_id (uuid.UUID): The ID of the app representing the client
        application using honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session (schemas.SessionCreate): The Session object containing any
        metadata and a location ID

    Returns:
        schemas.Session: The Session object of the new Session

    """
    value = await crud.create_session(
        db, app_id=app_id, user_id=user_id, session=session
    )
    return value


@router.put("/sessions/{session_id}", response_model=schemas.Session)
async def update_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    session: schemas.SessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update the metadata of a Session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to update
        session (schemas.SessionUpdate): The Session object containing any new metadata

    Returns:
        schemas.Session: The Session object of the updated Session

    """
    if session.metadata is None:
        raise HTTPException(
            status_code=400, detail="Session metadata cannot be empty"
        )  # TODO TEST if I can set the metadata to be blank with this
    try:
        return await crud.update_session(
            db, app_id=app_id, user_id=user_id, session_id=session_id, session=session
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.delete("/sessions/{session_id}")
async def delete_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a session by marking it as inactive

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to delete

    Returns:
        dict: A message indicating that the session was deleted

    Raises:
        HTTPException: If the session is not found

    """
    response = await crud.delete_session(
        db, app_id=app_id, user_id=user_id, session_id=session_id
    )
    if response:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}", response_model=schemas.Session)
async def get_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific session for a user by ID

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (uuid.UUID): The User ID representing the user, managed by the user
        session_id (uuid.UUID): The ID of the Session to retrieve

    Returns:
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
    honcho_session = await crud.get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_session


########################################################
# Message Routes
########################################################


@router.post("/sessions/{session_id}/messages", response_model=schemas.Message)
async def create_message_for_session(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message: schemas.MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Adds a message to a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        message (schemas.MessageCreate): The Message object to add containing the
        message content and type

    Returns:
        schemas.Message: The Message object of the added message

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await crud.create_message(
            db, message=message, app_id=app_id, user_id=user_id, session_id=session_id
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get("/sessions/{session_id}/messages", response_model=Page[schemas.Message])
async def get_messages(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    reverse: Optional[bool] = False,
    db: AsyncSession = Depends(get_db),
):
    """Get all messages for a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve
        reverse (bool): Whether to reverse the order of the messages

    Returns:
        list[schemas.Message]: List of Message objects

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await paginate(
            db,
            await crud.get_messages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get(
    "sessions/{session_id}/messages/{message_id}", response_model=schemas.Message
)
async def get_message(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """ """
    honcho_message = await crud.get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
    )
    if honcho_message is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_message


########################################################
# metamessage routes
########################################################


@router.post("/sessions/{session_id}/metamessages", response_model=schemas.Metamessage)
async def create_metamessage(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    metamessage: schemas.MetamessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Adds a message to a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        message (schemas.MessageCreate): The Message object to add containing the
        message content and type

    Returns:
        schemas.Message: The Message object of the added message

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await crud.create_metamessage(
            db,
            metamessage=metamessage,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get(
    "/sessions/{session_id}/metamessages", response_model=Page[schemas.Metamessage]
)
async def get_metamessages(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: Optional[uuid.UUID] = None,
    metamessage_type: Optional[str] = None,
    reverse: Optional[bool] = False,
    db: AsyncSession = Depends(get_db),
):
    """Get all messages for a session

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve
        reverse (bool): Whether to reverse the order of the metamessages

    Returns:
        list[schemas.Message]: List of Message objects

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return await paginate(
            db,
            await crud.get_metamessages(
                db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                metamessage_type=metamessage_type,
                reverse=reverse,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.get(
    "/sessions/{session_id}/metamessages/{metamessage_id}",
    response_model=schemas.Metamessage,
)
async def get_metamessage(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    metamessage_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific session for a user by ID

    Args:
        app_id (uuid.UUID): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve

    Returns:
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
    honcho_metamessage = await crud.get_metamessage(
        db,
        app_id=app_id,
        session_id=session_id,
        user_id=user_id,
        message_id=message_id,
        metamessage_id=metamessage_id,
    )
    if honcho_metamessage is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_metamessage


########################################################
# collection routes
########################################################


@router.get("/collections", response_model=Page[schemas.Collection])
async def get_collections(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    reverse: Optional[bool] = False,
    db: AsyncSession = Depends(get_db),
):
    return await paginate(
        db,
        await crud.get_collections(db, app_id=app_id, user_id=user_id, reverse=reverse),
    )


# @router.get("/collections/id/{collection_id}", response_model=schemas.Collection)
# def get_collection_by_id(
#     request: Request,
#     app_id: uuid.UUID,
#     user_id: uuid.UUID,
#     collection_id: uuid.UUID,
#     db: AsyncSession = Depends(get_db),
# ) -> schemas.Collection:
#     honcho_collection = crud.get_collection_by_id(
#         db, app_id=app_id, user_id=user_id, collection_id=collection_id
#     )
#     if honcho_collection is None:
#         raise HTTPException(
#             status_code=404, detail="collection not found or does not belong to user"
#         )
#     return honcho_collection


@router.get("/collections/{name}", response_model=schemas.Collection)
async def get_collection_by_name(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    name: str,
    db: AsyncSession = Depends(get_db),
) -> schemas.Collection:
    honcho_collection = await crud.get_collection_by_name(
        db, app_id=app_id, user_id=user_id, name=name
    )
    if honcho_collection is None:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )
    return honcho_collection


@router.post("/collections", response_model=schemas.Collection)
async def create_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection: schemas.CollectionCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await crud.create_collection(
            db, collection=collection, app_id=app_id, user_id=user_id
        )
    except ValueError:
        raise HTTPException(
            status_code=406,
            detail="Error invalid collection configuration - name may already exist",
        ) from None


@router.put("/collections/{collection_id}", response_model=schemas.Collection)
async def update_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    collection: schemas.CollectionUpdate,
    db: AsyncSession = Depends(get_db),
):
    if collection.name is None:
        raise HTTPException(
            status_code=400, detail="invalid request - name cannot be None"
        )
    try:
        honcho_collection = await crud.update_collection(
            db,
            collection=collection,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=406,
            detail="Error invalid collection configuration - name may already exist",
        ) from None
    return honcho_collection


@router.delete("/collections/{collection_id}")
async def delete_collection(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    response = await crud.delete_collection(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    if response:
        return {"message": "Collection deleted successfully"}
    else:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        )


########################################################
# Document routes
########################################################


@router.get(
    "/collections/{collection_id}/documents", response_model=Page[schemas.Document]
)
async def get_documents(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    reverse: Optional[bool] = False,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await paginate(
            db,
            await crud.get_documents(
                db,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                reverse=reverse,
            ),
        )
    except (
        ValueError
    ):  # TODO can probably remove this exception ok to return empty here
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None


router.get(
    "/collections/{collection_id}/documents/{document_id}",
    response_model=schemas.Document,
)


async def get_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    honcho_document = await crud.get_document(
        db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise HTTPException(
            status_code=404, detail="document not found or does not belong to user"
        )
    return honcho_document


@router.get(
    "/collections/{collection_id}/query", response_model=Sequence[schemas.Document]
)
async def query_documents(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    db: AsyncSession = Depends(get_db),
):
    if top_k is not None and top_k > 50:
        top_k = 50  # TODO see if we need to paginate this
    return await crud.query_documents(
        db=db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        query=query,
        top_k=top_k,
    )


@router.post("/collections/{collection_id}/documents", response_model=schemas.Document)
async def create_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document: schemas.DocumentCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await crud.create_document(
            db,
            document=document,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
        )
    except ValueError:
        raise HTTPException(
            status_code=404, detail="collection not found or does not belong to user"
        ) from None


@router.put(
    "/collections/{collection_id}/documents/{document_id}",
    response_model=schemas.Document,
)
async def update_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    document: schemas.DocumentUpdate,
    db: AsyncSession = Depends(get_db),
):
    if document.content is None and document.metadata is None:
        raise HTTPException(
            status_code=400, detail="content and metadata cannot both be None"
        )
    return await crud.update_document(
        db,
        document=document,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        document_id=document_id,
    )


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(
    request: Request,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    response = await crud.delete_document(
        db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        document_id=document_id,
    )
    if response:
        return {"message": "Document deleted successfully"}
    else:
        raise HTTPException(
            status_code=404, detail="document not found or does not belong to user"
        )


app.include_router(router)
