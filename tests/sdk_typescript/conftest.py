"""
TypeScript SDK Integration Tests

This module provides fixtures for running TypeScript SDK tests against a real HTTP server.
The server uses the same test database and mocks as the Python tests.
"""

import socket
import threading
import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any
from unittest.mock import patch

import pytest
import uvicorn
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from uvicorn.config import Config
from uvicorn.server import Server

from src.dependencies import get_db
from src.main import app


class TestServer:
    """A test server that runs uvicorn in a background thread."""

    def __init__(self, app: Any, port: int):
        self.config: Config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"
        )
        self.server: Server = uvicorn.Server(self.config)
        self.thread: Thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        # Wait for server to be ready
        deadline = time.time() + 10  # 10 second timeout
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not self.server.started:
            raise RuntimeError("Test server failed to start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def ts_db_session(
    db_engine: AsyncEngine,
) -> Generator[async_sessionmaker[AsyncSession], None, None]:
    """Create a session factory for the TypeScript test module."""
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    yield Session


# Store the session factory at module level so tracked_db override can access it
_ts_session_factory: async_sessionmaker[AsyncSession] | None = None


@pytest.fixture(scope="module")
def ts_test_server(
    ts_db_session: async_sessionmaker[AsyncSession],
) -> Generator[str, None, None]:
    """
    Start a real HTTP server for TypeScript SDK tests.

    This fixture:
    1. Uses the test database from db_engine
    2. Starts uvicorn on a random port
    3. Yields the server URL
    4. Cleans up on teardown

    Note: tracked_db mocking is handled by the mock_tracked_db fixture
    which creates fresh sessions for concurrent requests.
    """
    global _ts_session_factory
    _ts_session_factory = ts_db_session

    port = find_free_port()

    # Override database dependency to use test database
    async def override_get_db():
        async with ts_db_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Start the server
    server = TestServer(app, port)
    server.start()

    yield f"http://127.0.0.1:{port}"

    # Cleanup
    server.stop()
    app.dependency_overrides.clear()
    _ts_session_factory = None


@pytest.fixture(autouse=True)
def mock_tracked_db(ts_db_session: async_sessionmaker[AsyncSession]):
    """
    Override the main conftest's mock_tracked_db fixture.

    The main fixture uses a single shared db_session which gets corrupted
    by concurrent HTTP requests in the TypeScript tests. This override
    creates fresh sessions for each tracked_db call instead.
    """

    # Create a tracked_db that uses fresh sessions (not shared)
    @asynccontextmanager
    async def ts_tracked_db(_: str | None = None):
        async with ts_db_session() as session:
            yield session

    with (
        patch("src.dependencies.tracked_db", ts_tracked_db),
        patch("src.deriver.queue_manager.tracked_db", ts_tracked_db),
        patch("src.deriver.consumer.tracked_db", ts_tracked_db),
        patch("src.deriver.enqueue.tracked_db", ts_tracked_db),
        patch("src.routers.peers.tracked_db", ts_tracked_db),
        patch("src.crud.representation.tracked_db", ts_tracked_db),
        patch("src.dreamer.dream_scheduler.tracked_db", ts_tracked_db),
        patch("src.dreamer.orchestrator.tracked_db", ts_tracked_db),
        patch("src.dialectic.chat.tracked_db", ts_tracked_db),
        patch("src.utils.summarizer.tracked_db", ts_tracked_db),
        patch("src.webhooks.events.tracked_db", ts_tracked_db),
    ):
        yield
