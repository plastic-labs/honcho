"""Fixtures for property-based API tests.

The API schema comes from the live FastAPI app, so these tests always fuzz the
routes that are actually mounted rather than a committed spec that can drift.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import schemathesis
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.dependencies import get_db, get_read_db
from src.main import app

API_SCHEMA = schemathesis.openapi.from_dict(app.openapi())
app.openapi_schema = None


@pytest.fixture
def fuzz_client(
    client: TestClient, db_engine: AsyncEngine
) -> Generator[TestClient, Any]:
    """`client`, but with a fresh DB session per request, as in production.

    The shared `client` fixture hands every request one session. Hypothesis
    sends hundreds of requests per test, and a single failed statement would
    poison that session for every request after it.
    """
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def per_request_db() -> AsyncGenerator[AsyncSession, Any]:
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.rollback()

    app.dependency_overrides[get_db] = per_request_db
    app.dependency_overrides[get_read_db] = per_request_db
    yield client


@pytest.fixture
def seeded_ids(fuzz_client: TestClient) -> dict[str, str]:
    """Create one of each resource through the API and return their IDs.

    Fuzzed requests mix these in for path parameters, so they reach code past
    the "does the parent exist" check instead of stopping at a 404.
    """
    workspace, session = "fuzz-ws", "fuzz-session"
    base = f"/v3/workspaces/{workspace}"

    def ok(response: Any) -> Any:
        assert response.status_code < 300, response.text
        return response.json()

    ok(fuzz_client.post("/v3/workspaces", json={"id": workspace}))
    for peer in ("fuzz-alice", "fuzz-bob"):
        ok(fuzz_client.post(f"{base}/peers", json={"id": peer}))
    ok(
        fuzz_client.post(
            f"{base}/sessions",
            json={"id": session, "peers": {"fuzz-alice": {}, "fuzz-bob": {}}},
        )
    )
    messages = ok(
        fuzz_client.post(
            f"{base}/sessions/{session}/messages",
            json={
                "messages": [
                    {"content": "I like hiking", "peer_id": "fuzz-alice"},
                    {"content": "Me too", "peer_id": "fuzz-bob"},
                ]
            },
        )
    )
    conclusions = ok(
        fuzz_client.post(
            f"{base}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "alice likes hiking",
                        "observer_id": "fuzz-bob",
                        "observed_id": "fuzz-alice",
                        "session_id": session,
                    }
                ]
            },
        )
    )
    scope = ok(fuzz_client.post(f"{base}/scopes", json={"id": "fuzz-scope"}))
    webhook = ok(
        fuzz_client.post(f"{base}/webhooks", json={"url": "https://example.com/hook"})
    )
    return {
        "workspace_id": workspace,
        "peer_id": "fuzz-alice",
        "session_id": session,
        "message_id": messages[0]["id"],
        "conclusion_id": conclusions[0]["id"],
        "scope_id": scope["id"],
        "endpoint_id": webhook["id"],
    }
