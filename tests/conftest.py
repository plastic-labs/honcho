import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis
import jwt
import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
    database_exists,  # pyright: ignore[reportUnknownVariableType]
    drop_database,  # pyright: ignore[reportUnknownVariableType]
)

from src import models
from src.cache import client as cache_client
from src.config import settings
from src.db import Base
from src.dependencies import get_db
from src.exceptions import HonchoException
from src.main import app
from src.models import Peer, Workspace
from src.security import JWTParams, create_admin_jwt, create_jwt


# Create a custom handler that doesn't get closed prematurely
class TestHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)


# Setup logging with our custom handler
test_handler = TestHandler()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[test_handler],
)
logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

# Test database URL
# TODO use environment variable
DB_URI = (
    settings.DB.CONNECTION_URI
    or "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
)
CONNECTION_URI = make_url(DB_URI)
TEST_DB_URL = CONNECTION_URI.set(database="test_db")
DEFAULT_DB_URL = str(CONNECTION_URI.set(database="postgres"))

# Test API authorization - no longer needed as module-level constants
# We'll use settings.AUTH directly where needed


def create_test_database(db_url: URL):
    """Helper function create a database if it does not already exist
    uses the `sqlalchemy_utils` library to create the database and takes a DB URL
    as the input

    Args:
        db_url (str): Database URL
    """
    try:
        logger.debug(f"Checking if database exists: {db_url.database}")
        if not database_exists(db_url):
            logger.info(f"Creating test database: {db_url.database}")
            create_database(db_url)
            logger.info(f"Test database created successfully: {db_url.database}")
        else:
            logger.info(f"Database already exists: {db_url.database}")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


async def setup_test_database(db_url: URL):
    """Helper function to setup the test database
    takes a DB URL as input and returns a SQLAlchemy engine

    Args:
        db_url (str): Database URL

    Returns:
        engine: SQLAlchemy engine
    """
    engine = create_async_engine(str(db_url), echo=True)
    async with engine.connect() as conn:
        try:
            logger.info("Attempting to create pgvector extension...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
            logger.info("pgvector extension created successfully.")
        except ProgrammingError as e:
            logger.error(f"ProgrammingError: {e}")
            raise RuntimeError(
                "Failed to create pgvector extension. Make sure it's installed on the PostgreSQL server."
            ) from e
        except OperationalError as e:
            logger.error(f"OperationalError: {e}")
            raise RuntimeError(
                "Failed to connect to the database. Check your connection settings."
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    return engine


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    create_test_database(TEST_DB_URL)
    engine = await setup_test_database(TEST_DB_URL)

    # Force the schema to 'public' for tests
    # Save the original schema to restore later
    original_schema = Base.metadata.schema
    Base.metadata.schema = "public"

    # Update all table schemas to public
    for table in Base.metadata.tables.values():
        table.schema = "public"

    # Drop all tables first to ensure clean state
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        # Then create all tables with current models
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()

    # Restore original schema
    Base.metadata.schema = original_schema
    for table in Base.metadata.tables.values():
        table.schema = original_schema

    drop_database(TEST_DB_URL)


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine: AsyncEngine):
    """Create a database session for the scope of a single test function"""
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with Session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def fake_redis(monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated fakeredis client for each test."""

    original_client = cache_client._client  # pyright: ignore[reportPrivateUsage]
    original_enabled = settings.CACHE.ENABLED
    original_url = settings.CACHE.URL

    fake_client = fakeredis.FakeRedis(decode_responses=False)

    def _fake_from_url(*_: Any, **__: Any):  # pyright: ignore[reportUnusedParameter]
        return fake_client

    monkeypatch.setattr(cache_client.redis, "from_url", _fake_from_url)  # pyright: ignore[reportPrivateLocalImportUsage]
    monkeypatch.setattr("redis.asyncio.from_url", _fake_from_url)

    settings.CACHE.ENABLED = original_enabled
    cache_client._client = None  # pyright: ignore[reportPrivateUsage]

    await fake_client.flushall()  # pyright: ignore[reportUnknownMemberType]
    if original_enabled:
        await cache_client.init_cache()

    try:
        yield fake_client
    finally:
        await cache_client.close_cache()
        await fake_client.flushall()  # pyright: ignore[reportUnknownMemberType]
        await fake_client.aclose()
        cache_client._client = original_client  # pyright: ignore[reportPrivateUsage]
        settings.CACHE.ENABLED = original_enabled
        settings.CACHE.URL = original_url


@pytest.fixture(scope="function")
async def client(db_session: AsyncSession):
    """Create a FastAPI TestClient for the scope of a single test function"""

    # Register exception handlers for tests
    @app.exception_handler(HonchoException)
    async def test_exception_handler(  # pyright: ignore
        _: Request, exc: HonchoException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        if settings.AUTH.USE_AUTH:
            # give the test client the admin JWT
            c.headers["Authorization"] = f"Bearer {create_admin_jwt()}"
        yield c


def create_invalid_jwt() -> str:
    return jwt.encode({"ad": "invalid"}, "this is not the secret", algorithm="HS256")


class AuthClient(TestClient):
    auth_type: str | None = None


@pytest.fixture(
    params=[
        ("none", None),  # No auth
        ("invalid", create_invalid_jwt),  # Invalid JWT
        ("empty", lambda: create_jwt(JWTParams())),  # Empty JWT
        ("admin", create_admin_jwt),  # Admin JWT
    ]
)
def auth_client(
    client: AuthClient,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Fixture that provides a client with different authentication states.
    Always ensures USE_AUTH is set to True.
    """
    # Ensure USE_AUTH is always True for this fixture
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")

    # Clear any existing Authorization header
    client.headers.pop("Authorization", None)

    auth_type, token_func = request.param
    client.auth_type = auth_type

    if token_func is not None:
        token = token_func()
        client.headers["Authorization"] = f"Bearer {token}"

    return client


@pytest_asyncio.fixture(scope="function")
async def sample_data(
    db_session: AsyncSession,
) -> AsyncGenerator[tuple[Workspace, Peer], Any]:
    """Helper function to create test data"""
    # Create test app
    test_workspace = models.Workspace(name=str(generate_nanoid()))
    db_session.add(test_workspace)
    await db_session.flush()

    # Create test user
    test_peer = models.Peer(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    db_session.add(test_peer)
    await db_session.flush()

    yield test_workspace, test_peer

    await db_session.rollback()


@pytest.fixture(autouse=True)
def mock_langfuse():
    """Mock Langfuse decorator and context during tests"""
    with (
        patch("langfuse.observe") as mock_observe,
    ):
        # Mock the decorator to just return the function
        def return_value(func: Callable[..., Any]):
            return func

        mock_observe.return_value = return_value

        # Disable httpx logging during tests
        logging.getLogger("httpx").setLevel(logging.WARNING)

        yield

        # Clean up logging handlers
        for handler in logging.getLogger().handlers[:]:
            if isinstance(handler, TestHandler):
                handler.close()
                logging.getLogger().removeHandler(handler)


@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Mock OpenAI embeddings API calls for testing"""
    with (
        patch("src.embedding_client.embedding_client.embed") as mock_embed,
        patch("src.embedding_client.embedding_client.batch_embed") as mock_batch_embed,
    ):
        # Mock the embed method to return a fake embedding vector
        mock_embed.return_value = [0.1] * 1536

        # Mock the batch_embed method to return a dict of fake embedding vectors
        # Updated to support chunking - each text_id maps to a list of embedding vectors
        async def mock_batch_embed_func(
            id_resource_dict: dict[str, tuple[str, list[int]]],
        ) -> dict[str, list[list[float]]]:
            return {
                text_id: [[0.1] * 1536] for text_id in id_resource_dict
            }  # Single chunk per text

        mock_batch_embed.side_effect = mock_batch_embed_func

        yield {"embed": mock_embed, "batch_embed": mock_batch_embed}


@pytest.fixture(autouse=True)
def mock_llm_call_functions():
    """Mock LLM functions to avoid needing API keys during tests"""

    # Create mock responses for different function types
    with (
        patch(
            "src.utils.summarizer.create_short_summary", new_callable=AsyncMock
        ) as mock_short_summary,
        patch(
            "src.utils.summarizer.create_long_summary", new_callable=AsyncMock
        ) as mock_long_summary,
        patch(
            "src.deriver.deriver.critical_analysis_call", new_callable=AsyncMock
        ) as mock_critical_analysis,
        patch(
            "src.dialectic.chat.dialectic_call", new_callable=AsyncMock
        ) as mock_dialectic_call,
        patch(
            "src.dialectic.chat.dialectic_stream", new_callable=AsyncMock
        ) as mock_dialectic_stream,
    ):
        # Import the required models for proper mocking
        from src.utils.representation import (
            DeductiveObservationBase,
            ExplicitObservationBase,
            PromptRepresentation,
        )

        # Mock return values for different function types
        mock_short_summary.return_value = "Test short summary content"
        mock_long_summary.return_value = "Test long summary content"

        # Mock critical_analysis_call to return a proper object with _response attribute
        _rep = PromptRepresentation(
            explicit=[ExplicitObservationBase(content="Test explicit observation")],
            deductive=[
                DeductiveObservationBase(
                    conclusion="Test deductive conclusion",
                    premises=["Test premise 1", "Test premise 2"],
                )
            ],
        )
        mock_critical_analysis_result = MagicMock(wraps=_rep)
        # Add the _response attribute that contains thinking (used in the actual code)
        mock_response = MagicMock()
        mock_response.thinking = "Test thinking content"
        mock_critical_analysis_result._response = mock_response
        mock_critical_analysis.return_value = mock_critical_analysis_result

        # Mock dialectic_call to return a string (matching actual return type)
        mock_dialectic_call.return_value = "Test dialectic response"

        mock_dialectic_stream.return_value = AsyncMock()

        yield {
            "short_summary": mock_short_summary,
            "long_summary": mock_long_summary,
            "critical_analysis": mock_critical_analysis,
            "dialectic_call": mock_dialectic_call,
            "dialectic_stream": mock_dialectic_stream,
        }


@pytest.fixture(autouse=True)
def mock_honcho_llm_call():
    """Generic mock for the honcho_llm_call decorator to avoid actual LLM calls during tests"""
    from unittest.mock import AsyncMock, MagicMock

    from src.utils.representation import (
        DeductiveObservationBase,
        ExplicitObservationBase,
        PromptRepresentation,
    )

    def create_mock_response(
        response_model: Any = None,
        stream: bool = False,
        return_call_response: bool = False,
    ) -> Any:
        """Create a mock response based on the expected return type"""
        if stream:
            # For streaming responses, return an async mock
            mock_stream = AsyncMock()
            mock_stream.__aiter__.return_value = iter([])
            return mock_stream
        elif response_model:
            # For structured responses, create appropriate mock objects
            if getattr(response_model, "__name__", "") == "ReasoningResponse":
                _rep = PromptRepresentation(
                    explicit=[
                        ExplicitObservationBase(content="Test explicit observation")
                    ],
                    deductive=[
                        DeductiveObservationBase(
                            conclusion="Test deductive conclusion",
                            premises=["Test premise 1", "Test premise 2"],
                        ),
                    ],
                )
                mock_response = MagicMock(wraps=_rep)
                # Add the _response attribute that contains thinking (used in the actual code)
                mock_response._response = MagicMock()
                mock_response._response.thinking = "Test thinking content"
                return mock_response
            else:
                # Generic response model mock
                mock_response = MagicMock(spec=response_model)
                # Set some default attributes for common use cases
                if hasattr(mock_response, "content"):
                    mock_response.content = "Test response content"
                return mock_response
        elif return_call_response:
            # For CallResponse objects, create a mock with content and usage
            mock_response = MagicMock()
            mock_response.content = "Test response content"
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            return mock_response
        else:
            # For string responses, return a simple string
            return "Test response content"

    # Patch the honcho_llm_call decorator to prevent actual LLM calls at module level
    original_decorator = None
    try:
        import src.utils.clients

        original_decorator = src.utils.clients.honcho_llm_call
        src.utils.clients.honcho_llm_call = lambda *args, **kwargs: lambda func: func  # pyright: ignore[reportUnknownLambdaType]
    except ImportError:
        pass

    def decorator_factory(*args: Any, **kwargs: Any) -> Callable[..., Any]:  # pyright: ignore[reportUnusedParameter]
        """Factory function that creates the mock decorator"""

        def mock_llm_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            async def async_wrapper(*func_args: Any, **func_kwargs: Any) -> Any:  # pyright: ignore[reportUnusedParameter]
                # Create and return appropriate mock response
                return create_mock_response(
                    response_model=kwargs.get("response_model"),
                    stream=kwargs.get("stream", False),
                    return_call_response=kwargs.get("return_call_response", False),
                )

            def sync_wrapper(*func_args: Any, **func_kwargs: Any) -> Any:  # pyright: ignore[reportUnusedParameter]
                # Create and return appropriate mock response
                return create_mock_response(
                    response_model=kwargs.get("response_model"),
                    stream=kwargs.get("stream", False),
                    return_call_response=kwargs.get("return_call_response", False),
                )

            # Check if the original function is async
            import inspect

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return mock_llm_decorator

    with patch("src.utils.clients.honcho_llm_call", side_effect=decorator_factory):
        yield decorator_factory

    # Restore the original decorator
    if original_decorator:
        try:
            import src.utils.clients

            src.utils.clients.honcho_llm_call = original_decorator
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def mock_tracked_db(db_session: AsyncSession):
    """Mock tracked_db to use the test database session"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_tracked_db_context(_: str | None = None):
        yield db_session

    with (
        patch("src.dependencies.tracked_db", mock_tracked_db_context),
        patch("src.deriver.queue_manager.tracked_db", mock_tracked_db_context),
        patch("src.routers.sessions.tracked_db", mock_tracked_db_context),
        patch("src.routers.peers.tracked_db", mock_tracked_db_context),
        patch("src.crud.representation.tracked_db", mock_tracked_db_context),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_crud_collection_operations():
    """Mock CRUD operations that try to commit to database during tests"""
    from nanoid import generate as generate_nanoid

    from src import models

    async def mock_get_or_create_collection(
        _: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
    ):
        # Create a mock collection object that doesn't require database commit
        mock_collection = models.Collection(
            observer=observer,
            observed=observed,
            workspace_name=workspace_name,
        )
        mock_collection.id = generate_nanoid()
        return mock_collection

    with patch(
        "src.crud.get_or_create_collection",
        mock_get_or_create_collection,
    ):
        yield
