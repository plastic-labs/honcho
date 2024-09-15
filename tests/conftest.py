import logging  # noqa: I001
import os
import sys
import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy_utils import create_database, database_exists, drop_database

from src import models
from src.db import Base
from src.dependencies import get_db
from src.main import app

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # This ensures the output goes to stdout
)
logger = logging.getLogger(__name__)

# Test database URL
# TODO use environment variable
CONNECTION_URI = make_url(os.getenv("CONNECTION_URI"))
TEST_DB_URL = CONNECTION_URI.set(database="postgres")
DEFAULT_DB_URL = str(CONNECTION_URI.set(database="postgres"))


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
def client(db_session):
    """Create a FastAPI TestClient for the scope of a single test function"""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(scope="function")
async def sample_data(db_session):
    """Helper function to create test data"""
    # Create test app
    test_app = models.App(name=str(uuid.uuid4()))
    db_session.add(test_app)
    await db_session.flush()

    # Create test user
    test_user = models.User(name=str(uuid.uuid4()), app_id=test_app.id)
    db_session.add(test_user)
    await db_session.flush()

    yield test_app, test_user

    await db_session.rollback()
