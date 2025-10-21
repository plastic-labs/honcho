"""Pytest configuration for Alembic-focused tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from alembic.config import Config
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
def alembic_cfg(alembic_database: str) -> Config:
    """Provide an Alembic Config bound to the alembic test database."""
    cfg = Config(str(ALEMBIC_CONFIG_PATH))
    cfg.set_main_option(
        "script_location", str(ALEMBIC_CONFIG_PATH.parent / "migrations")
    )
    cfg.set_main_option("sqlalchemy.url", alembic_database)
    return cfg


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
    alembic_engine: Engine,
) -> Generator[None, None, None]:
    """Drop and recreate the schema for a clean slate each test (fast reset)."""

    schema = settings.DB.SCHEMA

    def _reset_schema() -> None:
        with alembic_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    _reset_schema()
    yield
    _reset_schema()
