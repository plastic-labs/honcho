import logging  # noqa: I001
import os
import sys
import jwt
from nanoid import generate as generate_nanoid

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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # This ensures the output goes to stdout
)
logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

# Test database URL
# TODO use environment variable
CONNECTION_URI = make_url(os.getenv("CONNECTION_URI", ""))
TEST_DB_URL = CONNECTION_URI.set(database="test_db")
DEFAULT_DB_URL = str(CONNECTION_URI.set(database="postgres"))

# Test API authorization
USE_AUTH = os.getenv("USE_AUTH", "False").lower() == "true"
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "test-secret")


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
    engine = create_async_engine(str(db_url))
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

    async with engine.begin() as conn:
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
        if USE_AUTH:
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
    import src.routers.keys as keys_module
    import src.security as security

    monkeypatch.setattr(keys_module, "USE_AUTH", "true")
    monkeypatch.setattr(security, "USE_AUTH", "true")

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
