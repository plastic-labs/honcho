import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
import pytest_asyncio
from cashews.picklers import PicklerType
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, text
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
)

from src import models
from src.cache.client import cache
from src.config import settings
from src.db import Base
from src.dependencies import get_db, get_read_db
from src.exceptions import HonchoException
from src.models import Peer, Workspace
from src.security import JWTParams, create_admin_jwt, create_jwt

# Disable Langfuse for the whole suite before importing src.main: @conditional_observe
# binds to settings.LANGFUSE_PUBLIC_KEY at import time, so blanking it here keeps mocked
# test calls from emitting traces to a configured Langfuse backend. Tests that exercise
# Langfuse patch settings.LANGFUSE_PUBLIC_KEY themselves.
settings.LANGFUSE_PUBLIC_KEY = None

from src.main import app  # noqa: E402


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

_RUNTIME_MOCK_TEST_BLOCKLIST_PREFIXES = (
    # Benchmarks and migration tests have their own execution/runtime constraints.
    "tests/bench/",
    "tests/alembic/",
    "tests/unified/",
    "tests/live_llm/",
    # Pure llm unit tests should stay isolated from the broader app/runtime fixtures.
    "tests/llm/",
    # LLM transport tests mock providers directly and don't need database/runtime setup.
    "tests/utils/test_length_finish_reason.py",
    "tests/utils/test_clients.py",
    # Session-scope SQL shape — asserts on compiled statements, never executes one.
    "tests/crud/test_session_scope_clauses.py",
    # Pure JWT scope tests — operate on src.security directly, no DB needed.
    "tests/test_security.py",
    "tests/test_generate_jwt_script.py",
    # The mock provider is a standalone ASGI app with no database or LLM of its
    # own; the runtime mocks would patch the very seams it exists to replace.
    "tests/mock_provider/",
)

_LIVE_LLM_MARKER = "live_llm"
_LIVE_LLM_SKIP_REASON = "live LLM tests are disabled; pass --live-llm to run them"


def _requires_runtime_mocks(nodeid: str) -> bool:
    return not any(
        nodeid.startswith(prefix) for prefix in _RUNTIME_MOCK_TEST_BLOCKLIST_PREFIXES
    )


def _get_nodeid(request: pytest.FixtureRequest) -> str:
    node = getattr(request, "node", None)
    nodeid = getattr(node, "nodeid", "")
    return nodeid if isinstance(nodeid, str) else ""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-llm",
        action="store_true",
        default=False,
        help="Run opt-in live LLM integration tests that call provider APIs.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--live-llm"):
        return

    skip_live = pytest.mark.skip(reason=_LIVE_LLM_SKIP_REASON)
    for item in items:
        if _LIVE_LLM_MARKER in item.keywords:
            item.add_marker(skip_live)


_RUN_ID_ENV_VAR = "HONCHO_TEST_RUN_ID"
_RUN_ID_TIME_FORMAT = "%Y%m%d%H%M%S"

# Only a database whose name carries a run-id timestamp this old is swept. Long
# enough that no live suite is ever this stale, short enough that a leak from the
# morning is gone by the afternoon.
_STALE_DB_AGE_SECONDS = 2 * 60 * 60

# test_db_<14-digit timestamp>_<4 hex>[_gwN] -- only names this function minted.
# A pinned HONCHO_TEST_RUN_ID deliberately won't match, so it's never swept.
_SWEEPABLE_DB_NAME = re.compile(r"^test_db_(\d{14})_[0-9a-f]{4}(?:_gw\d+)?$")


def pytest_configure(config: pytest.Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Stamp this pytest run with an id so its databases can't collide with another run's.

    The xdist controller runs this first and its environment is inherited by the
    workers it spawns, so `setdefault` gives every worker in a run the same id
    while separate runs (concurrent worktrees, two agents, a local run alongside
    CI) each get their own. Set the env var yourself to pin a stable name.

    The id leads with a sortable local-time timestamp so leaked databases can be
    aged out (see `_sweep_stale_test_databases`); the random tail keeps two runs
    starting in the same second apart.
    """

    os.environ.setdefault(
        _RUN_ID_ENV_VAR,
        f"{time.strftime(_RUN_ID_TIME_FORMAT)}_{uuid.uuid4().hex[:4]}",
    )

    # Workers inherit the controller's env and would each redo this.
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        _sweep_stale_test_databases()


def _get_test_db_url(worker_id: str) -> URL:
    """Get a worker-specific test database URL for pytest-xdist parallelism."""

    run_id = os.environ.get(_RUN_ID_ENV_VAR, "local")
    suffix = "" if worker_id == "master" else f"_{worker_id}"
    return CONNECTION_URI.set(database=f"test_db_{run_id}{suffix}")


def _drop_database(db_url: URL) -> None:
    """Drop a test database, evicting any connections still holding it open.

    WITH (FORCE) (pg13+) is what makes this reliable: a pooled connection that
    outlives engine disposal, or an xdist worker killed mid-query, otherwise
    leaves the drop failing with "database is being accessed by other users".
    """

    name = db_url.database
    if not name:
        return

    # Maintenance connection: you cannot drop the database you're connected to.
    engine = create_engine(
        db_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        engine.dispose()


def _sweep_stale_test_databases() -> None:
    """Reclaim test databases left behind by runs that died before teardown.

    A run killed by SIGKILL, an IDE stop button, an OOM'd worker or `-x` on a hang
    never reaches the `db_engine` teardown, and since every run mints its own
    database name nothing later reuses (and thus cleans) it.

    Two guards keep this from touching a suite that is currently running, which is
    the whole point of per-run names:

    - the run-id timestamp in the name must be older than `_STALE_DB_AGE_SECONDS`
    - the database must have no backends connected to it right now

    Each covers the other's blind spot: the age check is immune to the race where
    a database has been created but its first worker hasn't connected yet, and the
    connection check catches a genuinely long-running suite. Failure to sweep is
    logged and ignored -- it must never fail a test session.
    """

    cutoff = time.strftime(
        _RUN_ID_TIME_FORMAT, time.localtime(time.time() - _STALE_DB_AGE_SECONDS)
    )

    try:
        engine = create_engine(
            CONNECTION_URI.set(database="postgres"), isolation_level="AUTOCOMMIT"
        )
        try:
            with engine.connect() as conn:
                names = [
                    row[0]
                    for row in conn.exec_driver_sql(
                        "SELECT datname FROM pg_database d "
                        + "WHERE NOT EXISTS ("
                        + "  SELECT 1 FROM pg_stat_activity WHERE datname = d.datname"
                        + ")"
                    )
                ]
        finally:
            engine.dispose()

        for name in names:
            match = _SWEEPABLE_DB_NAME.match(name)
            if match is None or match.group(1) >= cutoff:
                continue
            logger.info(f"Dropping stale test database: {name}")
            _drop_database(CONNECTION_URI.set(database=name))
    except Exception as e:
        logger.warning(f"Could not sweep stale test databases: {e}")


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


async def _clear_all_tables(engine: AsyncEngine) -> None:
    """Remove all data from every mapped table between tests.

    Uses DELETE rather than TRUNCATE: TRUNCATE rewrites the relfilenode of every
    table and index it touches, so it costs a flat ~33ms for this schema's 11
    tables / 41 indexes no matter how few rows a test actually wrote. DELETE of
    the same (near-empty) tables, batched into one round trip, is ~3ms. Tables go
    in reverse dependency order so foreign keys are satisfied without CASCADE.

    This does not reset identity sequences, so tests must not assert on absolute
    generated id values -- compare against the ids the test itself created.
    """

    table_names: list[str] = []
    for table in reversed(Base.metadata.sorted_tables):
        if table.schema:
            table_names.append(f'"{table.schema}"."{table.name}"')
        else:
            table_names.append(f'"{table.name}"')

    if not table_names:
        return

    statement = "; ".join(f"DELETE FROM {name}" for name in table_names)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(statement)


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

        _drop_database(test_db_url)


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
        await _clear_all_tables(db_engine)


@pytest_asyncio.fixture(scope="session")
async def fake_cache_session():
    """Set up a taskless in-memory cache once per test session.

    Cashews' normal memory backend starts a periodic expiry task on whichever
    event loop first uses it. Tests use both pytest-asyncio loops and TestClient
    portal loops, so that task can be cancelled when its originating loop closes
    and then leak a CancelledError into the next app startup. Disabling the
    periodic sweep keeps the backend loop-agnostic; expired entries are still
    discarded lazily when read.
    """
    # Store original settings
    original_enabled = settings.CACHE.ENABLED
    original_url = settings.CACHE.URL

    try:
        # Use the same backend from pytest-asyncio and TestClient event loops.
        settings.CACHE.ENABLED = True
        settings.CACHE.URL = "mem://?check_interval=0"
        cache.setup(
            settings.CACHE.URL,
            pickle_type=PicklerType.SQLALCHEMY,
            enable=True,
        )

        yield cache
    finally:
        await cache.close()

        # Restore original settings
        settings.CACHE.ENABLED = original_enabled
        settings.CACHE.URL = original_url


@pytest_asyncio.fixture(scope="function", autouse=True)
async def fake_cache(fake_cache_session: Any):  # pyright: ignore[reportUnusedParameter]
    """Clear cache between tests."""
    # Clear cache before each test
    await cache.clear()

    yield cache

    # Clear cache after each test
    await cache.clear()


@pytest.fixture(scope="function")
async def client(
    db_session: AsyncSession,
    fake_cache_session: Any,  # pyright: ignore[reportUnusedParameter]
    monkeypatch: pytest.MonkeyPatch,
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
    # Read-only routes use get_read_db (AUTOCOMMIT engine) in production; in
    # tests they must see the same per-test database/session as writes, both
    # for isolation and so data written by a test is visible to its reads.
    app.dependency_overrides[get_read_db] = override_get_db

    # No-op the startup embedding-schema validator inside the lifespan. The
    # global `engine` it would inspect points to a DB that isn't migrated in
    # CI (per-worker test DBs are migrated separately by db_engine), and we
    # don't want the validator to dispose the test engine via the lifespan
    # finally block either. The validator has its own dedicated coverage in
    # tests/startup/test_embedding_validator.py against db_engine directly.
    async def _skip_validate(_engine: object) -> None:
        return None

    monkeypatch.setattr("src.main.validate_embedding_schema", _skip_validate)

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
    # _clear_all_tables handles cleanup between tests.
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
    vector_dimensions = settings.EMBEDDING.VECTOR_DIMENSIONS
    # Use hash bytes to generate deterministic floats between -1 and 1
    embedding: list[float] = []
    for i in range(vector_dimensions):
        # Use different bytes from hash (cycling through)
        byte_val = content_hash[i % len(content_hash)]
        # Normalize to [-1, 1] range
        embedding.append((byte_val / 255.0) * 2 - 1)
    return embedding


@pytest.fixture(autouse=True)
def mock_openai_embeddings(request: pytest.FixtureRequest):
    """Mock OpenAI embeddings API calls for testing"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

    with (
        patch("src.embedding_client.embedding_client.embed") as mock_embed,
        patch(
            "src.embedding_client.embedding_client.simple_batch_embed"
        ) as mock_simple_batch_embed,
        patch(
            "src.embedding_client.embedding_client.prepare_chunks"
        ) as mock_prepare_chunks,
        patch("src.embedding_client.embedding_client.batch_embed") as mock_batch_embed,
    ):
        # Mock the embed method to return content-dependent embedding
        def embed_side_effect(content: str) -> list[float]:
            return _content_to_embedding(content)

        mock_embed.side_effect = embed_side_effect

        async def mock_simple_batch_embed_func(
            texts: list[str], **_kwargs: object
        ) -> list[list[float]]:
            return [_content_to_embedding(text) for text in texts]

        mock_simple_batch_embed.side_effect = mock_simple_batch_embed_func

        def mock_prepare_chunks_func(
            id_resource_dict: dict[str, str],
        ) -> dict[str, list[str]]:
            # No real tokenizer in mocks: treat each input as a single chunk.
            return {text_id: [text] for text_id, text in id_resource_dict.items()}

        mock_prepare_chunks.side_effect = mock_prepare_chunks_func

        # Mock the batch_embed method to return content-dependent embeddings
        async def mock_batch_embed_func(
            id_resource_dict: dict[str, str],
        ) -> dict[str, list[list[float]]]:
            return {
                text_id: [_content_to_embedding(content)]
                for text_id, content in id_resource_dict.items()
            }

        mock_batch_embed.side_effect = mock_batch_embed_func

        yield {
            "embed": mock_embed,
            "simple_batch_embed": mock_simple_batch_embed,
            "prepare_chunks": mock_prepare_chunks,
            "batch_embed": mock_batch_embed,
        }


@pytest.fixture(autouse=True)
def mock_vector_store(request: pytest.FixtureRequest):
    """Mock vector store operations for testing"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

    from unittest.mock import AsyncMock, MagicMock

    from src.vector_store import (
        VectorQueryResult,
        VectorRecord,
        _hash_namespace_components,
    )

    # Create a mock vector store that stores vectors in memory
    vector_storage: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}

    async def mock_upsert_many(namespace: str, vectors: list[VectorRecord]) -> None:
        if namespace not in vector_storage:
            vector_storage[namespace] = {}
        for vector in vectors:
            vector_storage[namespace][vector.id] = (vector.embedding, vector.metadata)
        return

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
def mock_llm_call_functions(request: pytest.FixtureRequest):
    """Mock LLM functions to avoid needing API keys during tests"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

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
        patch(
            "src.routers.workspaces.workspace_chat", new_callable=AsyncMock
        ) as mock_workspace_chat,
        patch(
            "src.routers.workspaces.workspace_chat_stream", side_effect=mock_stream
        ) as mock_workspace_chat_stream,
    ):
        # Mock return values for different function types
        mock_short_summary.return_value = "Test short summary content"
        mock_long_summary.return_value = "Test long summary content"

        # Mock agentic_chat to return a string (matching actual return type).
        # With a response_model (structured output) the real function returns
        # a JSON string, so mirror that for SDK clients that parse content.
        async def _agentic_chat_response(*_args: object, **kwargs: object) -> str:
            if kwargs.get("response_model") is not None:
                return "{}"
            return "Test dialectic response"

        mock_agentic_chat.side_effect = _agentic_chat_response

        async def _workspace_chat_response(*_args: object, **kwargs: object) -> str:
            if kwargs.get("response_model") is not None:
                return "{}"
            return "Test workspace chat response"

        mock_workspace_chat.side_effect = _workspace_chat_response

        yield {
            "short_summary": mock_short_summary,
            "long_summary": mock_long_summary,
            "agentic_chat": mock_agentic_chat,
            "agentic_chat_stream": mock_agentic_chat_stream,
            "workspace_chat": mock_workspace_chat,
            "workspace_chat_stream": mock_workspace_chat_stream,
        }


@pytest.fixture(autouse=True)
def mock_honcho_llm_call(request: pytest.FixtureRequest):
    """Generic mock for the honcho_llm_call decorator to avoid actual LLM calls during tests"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

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
        import src.llm

        original_decorator = src.llm.honcho_llm_call
        src.llm.honcho_llm_call = lambda *args, **kwargs: lambda func: func  # pyright: ignore[reportUnknownLambdaType]
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

    with patch("src.llm.honcho_llm_call", side_effect=decorator_factory):
        yield decorator_factory

    # Restore the original decorator
    if original_decorator:
        try:
            import src.llm

            src.llm.honcho_llm_call = original_decorator
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def mock_tracked_db(request: pytest.FixtureRequest):
    """Mock tracked_db to create fresh sessions per call.

    Using a session factory instead of a shared session avoids asyncio lock
    errors when multiple tracked_db calls run concurrently via asyncio.gather.
    """
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

    from contextlib import ExitStack, asynccontextmanager

    db_engine = request.getfixturevalue("db_engine")
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def mock_tracked_db_context(_: str | None = None, *, read_only: bool = False):
        # read_only is accepted (and ignored): in tests both engines resolve to
        # the same per-test database session.
        del read_only
        async with session_factory() as session:
            yield session

    # Each module imports tracked_db by name, so patch every import site.
    # Use ExitStack (not a parenthesized `with`) to stay under CPython's
    # 20-statically-nested-block limit as this list grows.
    tracked_db_targets = [
        "src.dependencies.tracked_db",
        "src.deriver.queue_manager.tracked_db",
        "src.deriver.consumer.tracked_db",
        "src.deriver.enqueue.tracked_db",
        "src.routers.peers.tracked_db",
        "src.routers.workspaces.tracked_db",
        "src.crud.representation.tracked_db",
        "src.dreamer.orchestrator.tracked_db",
        "src.dreamer.dream_scheduler.tracked_db",
        "src.dialectic.chat.tracked_db",
        "src.utils.summarizer.tracked_db",
        "src.webhooks.events.tracked_db",
        "src.webhooks.webhook_delivery.tracked_db",
        "src.utils.agent_tools.tracked_db",
        "src.utils.search.tracked_db",
        "src.crud.document.tracked_db",
        "src.crud.message.tracked_db",
        "src.reconciler.sync_vectors.tracked_db",
        "src.reconciler.embed_now.tracked_db",
        "src.dialectic.core.tracked_db",
        "src.dreamer.specialists.tracked_db",
        "src.dreamer.surprisal.tracked_db",
        "src.deriver.scope_backfill.tracked_db",
    ]
    with ExitStack() as stack:
        for target in tracked_db_targets:
            stack.enter_context(patch(target, mock_tracked_db_context))
        yield


@pytest.fixture(autouse=True)
def enable_deriver_for_tests(request: pytest.FixtureRequest):
    """Enable deriver globally for tests that need queue processing"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

    from src.config import settings

    original_value = settings.DERIVER.ENABLED
    settings.DERIVER.ENABLED = True
    yield
    settings.DERIVER.ENABLED = original_value


@pytest.fixture(autouse=True)
def mock_crud_collection_operations(request: pytest.FixtureRequest):
    """Mock CRUD operations that try to commit to database during tests"""
    if not _requires_runtime_mocks(_get_nodeid(request)):
        yield
        return

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
