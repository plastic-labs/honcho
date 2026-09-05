"""Workspace delete must return 404, not 500, once the row is gone.

The workspace lookup used elsewhere goes through the Redis-backed
`_fetch_workspace`. In production the deriver runs with the cache disabled, so
its cascade never clears Redis and the API keeps serving the deleted workspace
until TTL. A DELETE that trusted that entry passed its existence check and then
failed the queue insert on `fk_queue_workspace_name` (Sentry HONCHO-19W/19X,
DEV-2646).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.cache.client import cache
from src.crud.workspace import workspace_cache_key


def _create_and_warm(client: TestClient, name: str) -> None:
    assert client.post("/v3/workspaces", json={"name": name}).status_code in (200, 201)
    # Second get-or-create goes through the @cache-decorated fetch.
    assert client.post("/v3/workspaces", json={"name": name}).status_code in (200, 201)


@pytest.mark.asyncio
async def test_delete_after_cascade_clears_cache_and_returns_404(
    client: TestClient, db_session: AsyncSession
):
    name = str(generate_nanoid())
    _create_and_warm(client, name)
    assert await cache.get(workspace_cache_key(name)) is not None

    assert client.delete(f"/v3/workspaces/{name}").status_code == 202

    # What the deriver's process_deletion runs.
    await crud.delete_workspace(db_session, workspace_name=name)

    assert await cache.get(workspace_cache_key(name)) is None
    response = client.delete(f"/v3/workspaces/{name}")
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_delete_with_stale_cache_returns_404_and_evicts(
    client: TestClient, db_session: AsyncSession
):
    """Models a deriver whose cache invalidation never reaches the API's Redis."""
    name = str(generate_nanoid())
    _create_and_warm(client, name)
    assert client.delete(f"/v3/workspaces/{name}").status_code == 202

    with patch("src.crud.workspace.cache.delete_match", new=AsyncMock()):
        await crud.delete_workspace(db_session, workspace_name=name)

    # Row is gone, Redis still says the workspace exists.
    assert await cache.get(workspace_cache_key(name)) is not None

    response = client.delete(f"/v3/workspaces/{name}")
    assert response.status_code == 404, response.text
    assert await cache.get(workspace_cache_key(name)) is None

    # With the ghost evicted, get-or-create recreates the workspace for real.
    assert client.post("/v3/workspaces", json={"name": name}).status_code in (200, 201)
    assert client.delete(f"/v3/workspaces/{name}").status_code == 202


@pytest.mark.asyncio
async def test_delete_racing_cascade_returns_404(
    client: TestClient, db_session: AsyncSession
):
    """The deriver drops the row between the existence check and the queue insert."""
    name = str(generate_nanoid())
    _create_and_warm(client, name)
    assert client.delete(f"/v3/workspaces/{name}").status_code == 202

    with patch("src.crud.workspace.cache.delete_match", new=AsyncMock()):
        await crud.delete_workspace(db_session, workspace_name=name)

    with patch(
        "src.routers.workspaces.crud.workspace_exists", new=AsyncMock(return_value=True)
    ):
        response = client.delete(f"/v3/workspaces/{name}")

    assert response.status_code == 404, response.text
    assert await cache.get(workspace_cache_key(name)) is None
