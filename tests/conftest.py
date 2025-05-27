import logging  # noqa: I001
import os
import jwt
from nanoid import generate as generate_nanoid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy_utils import create_database, database_exists, drop_database

from src import models
from src.db import Base
from src.dependencies import get_db
from src.exceptions import HonchoException
from src.security import create_admin_jwt, create_jwt, JWTParams
from src.main import app
from src.config import settings


# Create a custom handler that doesn't get closed prematurely
class TestHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
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


def create_test_database(db_url):
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


async def setup_test_database(db_url):
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

    # Drop all tables first to ensure clean state
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        # Then create all tables with current models
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()

    drop_database(TEST_DB_URL)


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    """Create a database session for the scope of a single test function"""
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with Session() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
async def client(db_session):
    """Create a FastAPI TestClient for the scope of a single test function"""

    # Register exception handlers for tests
    @app.exception_handler(HonchoException)
    async def test_exception_handler(request: Request, exc: HonchoException):
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


@pytest.fixture(
    params=[
        ("none", None),  # No auth
        ("invalid", create_invalid_jwt),  # Invalid JWT
        ("empty", lambda: create_jwt(JWTParams())),  # Empty JWT
        ("admin", create_admin_jwt),  # Admin JWT
    ]
)
def auth_client(client, request, monkeypatch):
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
async def sample_data(db_session):
    """Helper function to create test data"""
    # Create test app
    test_app = models.App(name=str(generate_nanoid()))
    db_session.add(test_app)
    await db_session.flush()

    # Create test user
    test_user = models.User(name=str(generate_nanoid()), app_id=test_app.public_id)
    db_session.add(test_user)
    await db_session.flush()

    yield test_app, test_user

    await db_session.rollback()


@pytest.fixture(autouse=True)
def mock_langfuse():
    """Mock Langfuse decorator and context during tests"""
    with (
        patch("langfuse.decorators.observe") as mock_observe,
        patch("langfuse.decorators.langfuse_context") as mock_context,
    ):
        # Mock the decorator to just return the function
        mock_observe.return_value = lambda func: func

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
