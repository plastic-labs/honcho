import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
import pytest_asyncio
from cashews.backends.interface import ControlMixin
from cashews.picklers import PicklerType
from fakeredis import FakeAsyncRedis
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
from src.cache.client import cache
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


def _get_test_db_url(worker_id: str) -> URL:
    """Get a worker-specific test database URL for pytest-xdist parallelism."""

    db_name = "test_db" if worker_id == "master" else f"test_db_{worker_id}"
    return CONNECTION_URI.set(database=db_name)


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
    engine = create_async_engine(str(db_url), echo=False)
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


async def _truncate_all_tables(engine: AsyncEngine) -> None:
    """Remove all data from every mapped table while resetting identities."""

    table_names: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.schema:
            table_names.append(f'"{table.schema}"."{table.name}"')
        else:
            table_names.append(f'"{table.name}"')

    if not table_names:
        return

    joined_names = ", ".join(table_names)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {joined_names} RESTART IDENTITY CASCADE"))


@pytest_asyncio.fixture(scope="session")
async def db_engine(worker_id: str):
    test_db_url = _get_test_db_url(worker_id)
    create_test_database(test_db_url)
    engine = await setup_test_database(test_db_url)

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

    try:
        yield engine
    finally:
        await engine.dispose()

        # Restore original schema
        Base.metadata.schema = original_schema
        for table in Base.metadata.tables.values():
            table.schema = original_schema

        drop_database(test_db_url)


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine: AsyncEngine):
    """Create a database session for the scope of a single test function"""
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    try:
        async with Session() as session:
            try:
                yield session
            finally:
                await session.rollback()
    finally:
        await _truncate_all_tables(db_engine)


@pytest_asyncio.fixture(scope="session")
async def fake_cache_session():
    """Set up fakeredis for caching once per test session."""
    # Store original settings
    original_enabled = settings.CACHE.ENABLED
    original_url = settings.CACHE.URL

    # Create a fake redis instance that persists for the session
    fake_redis = FakeAsyncRedis(decode_responses=True)

    # Patch redis creation to use fakeredis
    # Cashews uses redis.asyncio.from_url to create connections
    def fake_redis_from_url(*_args: Any, **_kwargs: Any):
        return fake_redis

    # Patch the cashews backend's _disable property to avoid ContextVar issues
    # This works around cashews' ContextVar not being properly initialized in TestClient context

    original_disable_property = ControlMixin._disable  # pyright: ignore[reportPrivateUsage]

    @property  # type: ignore
    def patched_disable_property(self):  # pyright: ignore
        try:
            return original_disable_property.fget(self)  # pyright: ignore[reportOptionalCall]
        except LookupError:
            # Return empty set as default if ContextVar not set in current context
            return set()  # pyright: ignore

    # Start patching
    redis_patch = patch("redis.asyncio.from_url", fake_redis_from_url)
    redis_patch.start()
    ControlMixin._disable = patched_disable_property  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    try:
        # Enable caching and set URL for tests
        settings.CACHE.ENABLED = True
        settings.CACHE.URL = "redis://fake-redis:6379/0"

        # Setup cache for tests that don't use TestClient (direct CRUD tests)
        # For TestClient tests, the app's lifespan handler will also call cache.setup()
        # The ContextVar patch above handles any context issues
        cache.setup(
            "redis://fake-redis:6379/0", pickle_type=PicklerType.SQLALCHEMY, enable=True
        )

        yield fake_redis
    finally:
        # Stop the patches
        redis_patch.stop()
        ControlMixin._disable = original_disable_property  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

        # Restore original settings
        settings.CACHE.ENABLED = original_enabled
        settings.CACHE.URL = original_url


@pytest_asyncio.fixture(scope="function", autouse=True)
async def fake_cache(fake_cache_session: FakeAsyncRedis):
    """Clear cache between tests."""
    # Clear cache before each test
    await fake_cache_session.flushall()  # pyright: ignore[reportUnknownMemberType]

    yield cache

    # Clear cache after each test
    await fake_cache_session.flushall()  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture(scope="function")
async def client(
    db_session: AsyncSession,
    fake_cache_session: FakeAsyncRedis,  # pyright: ignore[reportUnusedParameter]
) -> AsyncGenerator[TestClient, Any]:
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

    # Create test user
    test_peer = models.Peer(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    db_session.add(test_peer)

    # Commit so data is visible to independent tracked_db sessions.
    # _truncate_all_tables handles cleanup between tests.
    await db_session.commit()

    yield test_workspace, test_peer


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


def _content_to_embedding(content: str) -> list[float]:
    """Generate a deterministic embedding from content hash.

    This ensures different content produces different embeddings,
    which is critical for deduplication logic to work correctly in tests.
    """
    import hashlib

    # Hash the content to get a deterministic seed
    content_hash = hashlib.sha256(content.encode()).digest()
    # Use hash bytes to generate 1536 floats between -1 and 1
    embedding: list[float] = []
    for i in range(1536):
        # Use different bytes from hash (cycling through)
        byte_val = content_hash[i % len(content_hash)]
        # Normalize to [-1, 1] range
        embedding.append((byte_val / 255.0) * 2 - 1)
    return embedding


@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Mock OpenAI embeddings API calls for testing"""
    with (
        patch("src.embedding_client.embedding_client.embed") as mock_embed,
        patch("src.embedding_client.embedding_client.batch_embed") as mock_batch_embed,
    ):
        # Mock the embed method to return content-dependent embedding
        def embed_side_effect(content: str) -> list[float]:
            return _content_to_embedding(content)

        mock_embed.side_effect = embed_side_effect

        # Mock the batch_embed method to return content-dependent embeddings
        async def mock_batch_embed_func(
            id_resource_dict: dict[str, tuple[str, list[int]]],
        ) -> dict[str, list[list[float]]]:
            return {
                text_id: [_content_to_embedding(resource[0])]
                for text_id, resource in id_resource_dict.items()
            }

        mock_batch_embed.side_effect = mock_batch_embed_func

        yield {"embed": mock_embed, "batch_embed": mock_batch_embed}


@pytest.fixture(autouse=True)
def mock_vector_store():
    """Mock vector store operations for testing"""
    from unittest.mock import AsyncMock, MagicMock

    from src.vector_store import (
        VectorQueryResult,
        VectorRecord,
        VectorUpsertResult,
        _hash_namespace_components,  # pyright: ignore[reportPrivateUsage]
    )

    # Create a mock vector store that stores vectors in memory
    vector_storage: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}

    async def mock_upsert_many(
        namespace: str, vectors: list[VectorRecord]
    ) -> VectorUpsertResult:
        if namespace not in vector_storage:
            vector_storage[namespace] = {}
        for vector in vectors:
            vector_storage[namespace][vector.id] = (vector.embedding, vector.metadata)
        return VectorUpsertResult(ok=True)

    async def mock_query(
        namespace: str, embedding: list[float], **kwargs: Any
    ) -> list[VectorQueryResult]:
        _ = embedding  # unused in mock
        if namespace not in vector_storage:
            return []

        # Simple mock: return all vectors in the namespace as results
        results: list[VectorQueryResult] = []
        for vec_id, (_vec_embedding, metadata) in vector_storage[namespace].items():
            results.append(
                VectorQueryResult(
                    id=vec_id,
                    score=0.1,  # Mock score
                    metadata=metadata,
                )
            )
        top_k: int = kwargs.get("top_k", 10)
        return results[:top_k]

    async def mock_delete_many(namespace: str, ids: list[str]) -> None:
        if namespace in vector_storage:
            for vec_id in ids:
                vector_storage[namespace].pop(vec_id, None)

    async def mock_delete_namespace(namespace: str) -> None:
        vector_storage.pop(namespace, None)

    # Clear the cache on get_external_vector_store before patching
    from src.vector_store import get_external_vector_store

    get_external_vector_store.cache_clear()  # type: ignore

    # Create the mock vector store
    mock_vs = MagicMock()
    mock_vs.upsert_many = AsyncMock(side_effect=mock_upsert_many)
    mock_vs.query = AsyncMock(side_effect=mock_query)
    mock_vs.delete_many = AsyncMock(side_effect=mock_delete_many)
    mock_vs.delete_namespace = AsyncMock(side_effect=mock_delete_namespace)

    def mock_get_vector_namespace(
        namespace_type: str,
        workspace_name: str,
        observer: str | None = None,
        observed: str | None = None,
    ) -> str:
        # Uses real hash function for consistency with production
        if namespace_type == "document":
            if observer is None or observed is None:
                raise ValueError(
                    "observer and observed are required for document namespaces"
                )
            return f"honcho2345.doc.{_hash_namespace_components(workspace_name, observer, observed)}"
        if namespace_type == "message":
            return f"honcho2345.msg.{_hash_namespace_components(workspace_name)}"
        raise ValueError(f"Unknown namespace type: {namespace_type}")

    mock_vs.get_vector_namespace = mock_get_vector_namespace

    with (
        patch("src.crud.document.get_external_vector_store", return_value=mock_vs),
        patch("src.crud.workspace.get_external_vector_store", return_value=mock_vs),
        patch("src.crud.session.get_external_vector_store", return_value=mock_vs),
        patch("src.crud.message.get_external_vector_store", return_value=mock_vs),
        patch(
            "src.reconciler.sync_vectors.get_external_vector_store",
            return_value=mock_vs,
        ),
        patch("src.utils.search.get_external_vector_store", return_value=mock_vs),
    ):
        yield mock_vs

        # Clear cache after test as well for cleanliness
        get_external_vector_store.cache_clear()  # type: ignore


@pytest.fixture(autouse=True)
def mock_llm_call_functions():
    """Mock LLM functions to avoid needing API keys during tests"""

    # Create an async generator for streaming responses
    async def mock_stream(*args, **kwargs):  # pyright: ignore[reportUnusedParameter, reportMissingParameterType, reportUnknownParameterType]
        """Mock streaming response that yields chunks"""
        chunks = ["Test ", "streaming ", "response"]
        for chunk in chunks:
            yield chunk

    # Create mock responses for different function types
    # Note: critical_analysis_call was removed as the deriver now uses agentic approach
    # Note: dialectic_call/dialectic_stream were replaced with agentic_chat
    with (
        patch(
            "src.utils.summarizer.create_short_summary", new_callable=AsyncMock
        ) as mock_short_summary,
        patch(
            "src.utils.summarizer.create_long_summary", new_callable=AsyncMock
        ) as mock_long_summary,
        patch(
            "src.routers.peers.agentic_chat", new_callable=AsyncMock
        ) as mock_agentic_chat,
        patch(
            "src.routers.peers.agentic_chat_stream", side_effect=mock_stream
        ) as mock_agentic_chat_stream,
    ):
        # Mock return values for different function types
        mock_short_summary.return_value = "Test short summary content"
        mock_long_summary.return_value = "Test long summary content"

        # Mock agentic_chat to return a string (matching actual return type)
        mock_agentic_chat.return_value = "Test dialectic response"

        yield {
            "short_summary": mock_short_summary,
            "long_summary": mock_long_summary,
            "agentic_chat": mock_agentic_chat,
            "agentic_chat_stream": mock_agentic_chat_stream,
        }


@pytest.fixture(autouse=True)
def mock_honcho_llm_call():
    """Generic mock for the honcho_llm_call decorator to avoid actual LLM calls during tests"""
    from unittest.mock import AsyncMock

    from src.utils.representation import (
        # DeductiveObservationBase,
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
                    # deductive=[
                    #     DeductiveObservationBase(
                    #         conclusion="Test deductive conclusion",
                    #         premises=["Test premise 1", "Test premise 2"],
                    #     ),
                    # ],
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
def mock_tracked_db(db_engine: AsyncEngine):
    """Mock tracked_db to create fresh sessions per call.

    Using a session factory instead of a shared session avoids asyncio lock
    errors when multiple tracked_db calls run concurrently via asyncio.gather.
    """
    from contextlib import asynccontextmanager

    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def mock_tracked_db_context(_: str | None = None):
        async with session_factory() as session:
            yield session

    with (
        patch("src.dependencies.tracked_db", mock_tracked_db_context),
        patch("src.deriver.queue_manager.tracked_db", mock_tracked_db_context),
        patch("src.deriver.consumer.tracked_db", mock_tracked_db_context),
        patch("src.deriver.enqueue.tracked_db", mock_tracked_db_context),
        patch("src.routers.peers.tracked_db", mock_tracked_db_context),
        patch("src.crud.representation.tracked_db", mock_tracked_db_context),
        patch("src.dreamer.orchestrator.tracked_db", mock_tracked_db_context),
        patch("src.dreamer.dream_scheduler.tracked_db", mock_tracked_db_context),
        patch("src.dialectic.chat.tracked_db", mock_tracked_db_context),
        patch("src.utils.summarizer.tracked_db", mock_tracked_db_context),
        patch("src.webhooks.events.tracked_db", mock_tracked_db_context),
    ):
        yield


@pytest.fixture(autouse=True)
def enable_deriver_for_tests():
    """Enable deriver globally for tests that need queue processing"""
    from src.config import settings

    original_value = settings.DERIVER.ENABLED
    settings.DERIVER.ENABLED = True
    yield
    settings.DERIVER.ENABLED = original_value


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
