import logging  # noqa: I001
from collections.abc import AsyncGenerator
from typing import Any
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

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
from src.db import Base
from src.dependencies import get_db
from src.exceptions import HonchoException
from src.main import app
from src.config import settings
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
    return jwt.encode(  # pyright: ignore[reportUnknownMemberType]
        {"ad": "invalid"}, "this is not the secret", algorithm="HS256"
    )


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
        patch("langfuse.decorators.observe") as mock_observe,
        patch("langfuse.decorators.langfuse_context") as mock_context,
    ):
        # Mock the decorator to just return the function
        def return_value(func: Callable[..., Any]):
            return func

        mock_observe.return_value = return_value

        # Mock the context object
        mock_context_obj = MagicMock()
        mock_context_obj.update_current_observation = MagicMock()
        mock_context_obj.update_current_trace = MagicMock()
        mock_context.return_value = mock_context_obj

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
    with patch("src.crud.openai_client.embeddings.create") as mock_create:
        mock_response = AsyncMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_create.return_value = mock_response
        yield mock_create


@pytest.fixture(autouse=True)
def mock_mirascope_functions():
    """Mock Mirascope LLM functions to avoid needing API keys during tests"""

    # Create mock responses for different function types
    with (
        patch("src.utils.history.create_short_summary") as mock_short_summary,
        patch("src.utils.history.create_long_summary") as mock_long_summary,
        patch(
            "src.deriver.tom.single_prompt.tom_inference", new_callable=AsyncMock
        ) as mock_tom_inference,
        patch(
            "src.deriver.tom.single_prompt.user_representation_inference"
        ) as mock_user_rep_inference,
        patch(
            "src.deriver.tom.single_prompt.get_tom_inference_single_prompt",
            new_callable=AsyncMock,
        ) as mock_single_tom,
        patch(
            "src.deriver.tom.single_prompt.get_user_representation_single_prompt"
        ) as mock_single_rep,
        patch(
            "src.deriver.tom.long_term.get_user_representation_long_term"
        ) as mock_long_rep,
        patch(
            "src.deriver.tom.long_term.extract_facts_long_term"
        ) as mock_extract_facts,
        patch(
            "src.agent.dialectic_call", new_callable=AsyncMock
        ) as mock_dialectic_call,
        patch("src.agent.dialectic_stream") as mock_dialectic_stream,
        patch("src.agent.generate_semantic_queries_llm") as mock_semantic_queries,
    ):
        # Mock return values for different function types
        mock_short_summary.return_value = "Test short summary content"
        mock_long_summary.return_value = "Test long summary content"
        mock_tom_inference.return_value = MagicMock(inference="Test tom inference")
        mock_user_rep_inference.return_value = "Test user representation"
        # Mock single_tom to return a proper Pydantic object
        from src.deriver.tom.single_prompt import (
            CurrentState,
            TentativeInference,
            TomInferenceOutput,
        )

        mock_tom_obj = TomInferenceOutput(
            current_state=CurrentState(
                immediate_context="test context",
                active_goals="test goals",
                present_mood="test mood",
            ),
            tentative_inferences=[
                TentativeInference(interpretation="test inference", basis="test basis")
            ],
            knowledge_gaps=[],
            expectation_violations=[],
        )
        mock_single_tom.return_value = mock_tom_obj
        mock_single_rep.return_value = "Test user representation"
        mock_long_rep.return_value = MagicMock(
            current_state="Test state",
            tentative_patterns=[],
            knowledge_gaps=[],
            expectation_violations=[],
            updates=[],
        )
        mock_extract_facts.return_value = MagicMock(facts=["fact 1", "fact 2"])

        # Create a proper async mock result for dialectic_call
        mock_dialectic_result = MagicMock()
        mock_dialectic_result.content = "Test dialectic response"
        mock_dialectic_call.return_value = mock_dialectic_result

        mock_dialectic_stream.return_value = AsyncMock()
        mock_semantic_queries.return_value = ["test query 1", "test query 2"]

        yield {
            "short_summary": mock_short_summary,
            "long_summary": mock_long_summary,
            "tom_inference": mock_tom_inference,
            "user_rep_inference": mock_user_rep_inference,
            "single_tom": mock_single_tom,
            "single_rep": mock_single_rep,
            "long_rep": mock_long_rep,
            "extract_facts": mock_extract_facts,
            "dialectic_call": mock_dialectic_call,
            "dialectic_stream": mock_dialectic_stream,
            "semantic_queries": mock_semantic_queries,
        }


@pytest.fixture(autouse=True)
def mock_tracked_db(db_session: AsyncSession):
    """Mock tracked_db to use the test database session"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_tracked_db_context(_: str | None = None):
        yield db_session

    with patch("src.dependencies.tracked_db", mock_tracked_db_context):
        yield


@pytest.fixture(autouse=True)
def mock_crud_collection_operations():
    """Mock CRUD operations that try to commit to database during tests"""
    from nanoid import generate as generate_nanoid

    from src import models

    async def mock_get_or_create_collection(
        _: AsyncSession, workspace_name: str, peer_name: str, collection_name: str
    ):
        # Create a mock collection object that doesn't require database commit
        mock_collection = models.Collection(
            name=collection_name,
            workspace_name=workspace_name,
            peer_name=peer_name,
        )
        mock_collection.id = generate_nanoid()
        return mock_collection

    with patch(
        "src.crud.get_or_create_collection",
        mock_get_or_create_collection,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_agent_api_calls():
    """Mock API calls made by the agent during tests"""
    # Mock the agent-specific functions
    with (
        patch("src.agent.generate_semantic_queries") as mock_generate_queries,
        patch("src.agent.get_user_representation_long_term") as mock_user_rep,
        patch(
            "src.deriver.tom.embeddings.CollectionEmbeddingStore.get_relevant_facts"
        ) as mock_get_facts,
        patch(
            "src.agent.dialectic_call", new_callable=AsyncMock
        ) as mock_dialectic_call,
        patch("src.agent.dialectic_stream") as mock_dialectic_stream,
    ):
        # Mock semantic query generation
        mock_generate_queries.return_value = ["test query 1", "test query 2"]

        # Mock user representation generation
        mock_user_rep.return_value = (
            "<representation>Test user representation</representation>"
        )

        # Mock embedding store facts retrieval
        mock_get_facts.return_value = ["fact 1", "fact 2", "fact 3"]

        # Mock Dialectic API calls
        mock_dialectic_result = MagicMock()
        mock_dialectic_result.content = "Test dialectic response"
        mock_dialectic_call.return_value = mock_dialectic_result
        mock_dialectic_stream.return_value = AsyncMock()

        yield {
            "generate_queries": mock_generate_queries,
            "user_rep": mock_user_rep,
            "get_facts": mock_get_facts,
            "dialectic_call": mock_dialectic_call,
            "dialectic_stream": mock_dialectic_stream,
        }
