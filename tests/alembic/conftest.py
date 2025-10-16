"""Pytest configuration for Alembic-focused tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from pytest_alembic.runner import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy_utils import (
    create_database,  # pyright: ignore
    database_exists,  # pyright: ignore
    drop_database,  # pyright: ignore
)

from src.config import settings
from tests.conftest import CONNECTION_URI

ALEMBIC_CONFIG_PATH = Path(__file__).resolve().parents[2] / "alembic.ini"
ALEMBIC_TEST_DB_URL: URL = CONNECTION_URI.set(database="alembic_migration_tests")


@pytest.fixture(scope="session")
def alembic_database() -> Generator[str, None, None]:
    """Provision a dedicated DB for Alembic verification tests."""

    if database_exists(ALEMBIC_TEST_DB_URL):
        drop_database(ALEMBIC_TEST_DB_URL)  # start fresh
    create_database(ALEMBIC_TEST_DB_URL)

    engine = create_engine(str(ALEMBIC_TEST_DB_URL))
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        yield str(ALEMBIC_TEST_DB_URL)
    finally:
        engine.dispose()
        if database_exists(ALEMBIC_TEST_DB_URL):
            drop_database(ALEMBIC_TEST_DB_URL)


@pytest.fixture(scope="session", autouse=True)
def configure_alembic_settings(alembic_database: str) -> Generator[None, None, None]:
    """Point application settings at the Alembic test database."""

    previous_uri = settings.DB.CONNECTION_URI
    os.environ["DB_CONNECTION_URI"] = alembic_database
    settings.DB.CONNECTION_URI = alembic_database

    try:
        yield
    finally:
        settings.DB.CONNECTION_URI = previous_uri
        if previous_uri:
            os.environ["DB_CONNECTION_URI"] = previous_uri
        else:
            os.environ.pop("DB_CONNECTION_URI", None)


@pytest.fixture
def alembic_config(alembic_database: str):
    """Provide pytest-alembic with the project configuration."""
    return {
        "file": str(ALEMBIC_CONFIG_PATH),
        "script_location": str(ALEMBIC_CONFIG_PATH.parent / "migrations"),
        "sqlalchemy.url": alembic_database,
    }


@pytest.fixture
def alembic_engine(alembic_database: str) -> Generator[Engine, None, None]:
    """Yield an engine bound to the Alembic test database."""

    engine = create_engine(alembic_database)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_database_between_tests(
    alembic_runner: MigrationContext,
    alembic_engine: Engine,  # pyright: ignore[reportUnusedParameter]
) -> Generator[None, None, None]:
    """Ensure each test starts and ends from a clean base migration state."""

    alembic_runner.migrate_down_to("base")  # pyright: ignore[reportUnknownMemberType]
    yield
    alembic_runner.migrate_down_to("base")  # pyright: ignore[reportUnknownMemberType]
