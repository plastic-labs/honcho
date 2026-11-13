"""Auto workspaces: `auto` is chosen when a workspace is created and never changes."""

import datetime
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.cache.client import cache
from src.crud import workspace as workspace_crud


def _name() -> str:
    return f"ws-{generate_nanoid(size=10).replace('_', 'a')}"


def _workspace(client: TestClient, **body: Any) -> str:
    name = _name()
    response = client.post("/v3/workspaces", json={"id": name, **body})
    assert response.status_code == 201, response.text
    return name


async def _auto_enabled_at(
    db: AsyncSession, workspace: str
) -> datetime.datetime | None:
    return await db.scalar(
        select(models.Workspace.auto_enabled_at).where(
            models.Workspace.name == workspace
        )
    )


@pytest.mark.asyncio
async def test_auto_on_create_makes_an_auto_workspace(
    client: TestClient, db_session: AsyncSession
) -> None:
    name = _name()
    created = client.post("/v3/workspaces", json={"id": name, "auto": True})
    assert created.status_code == 201, created.text
    assert created.json()["auto"] is True
    enabled_at = await _auto_enabled_at(db_session, name)
    assert enabled_at is not None

    # Getting it again, with or without `auto`, returns it as it is.
    for body in ({"id": name}, {"id": name, "auto": True}):
        again = client.post("/v3/workspaces", json=body)
        assert again.status_code == 200, again.text
        assert again.json()["auto"] is True
    assert await _auto_enabled_at(db_session, name) == enabled_at


@pytest.mark.asyncio
async def test_a_workspace_is_not_auto_by_default(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    assert client.post("/v3/workspaces", json={"id": workspace}).json()["auto"] is False
    assert await _auto_enabled_at(db_session, workspace) is None


@pytest.mark.asyncio
async def test_auto_on_an_existing_workspace_that_isnt_auto_is_409(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Only a new workspace can be made auto, so asking an existing one fails loudly."""
    workspace = _workspace(client)
    response = client.post("/v3/workspaces", json={"id": workspace, "auto": True})
    assert response.status_code == 409, response.text
    assert "auto" in response.json()["detail"]
    assert await _auto_enabled_at(db_session, workspace) is None


@pytest.mark.asyncio
async def test_updating_the_workspace_keeps_auto(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client, auto=True)
    enabled_at = await _auto_enabled_at(db_session, workspace)

    response = client.put(
        f"/v3/workspaces/{workspace}",
        json={"metadata": {"team": "core"}, "configuration": {}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["auto"] is True
    assert await _auto_enabled_at(db_session, workspace) == enabled_at


def test_listed_workspaces_show_auto(client: TestClient) -> None:
    auto = _workspace(client, auto=True)
    plain = _workspace(client)
    listed = client.post(
        "/v3/workspaces/list", json={"filters": {"id": {"in": [auto, plain]}}}
    ).json()["items"]
    assert {w["id"]: w["auto"] for w in listed} == {auto: True, plain: False}


# The workspace cache entry's fields in the previous release, which builds
# models.Workspace(**entry) from it and has no auto_enabled_at.
_PREVIOUS_RELEASE_ENTRY = {
    "id",
    "name",
    "h_metadata",
    "internal_metadata",
    "configuration",
    "created_at",
}


@pytest.mark.asyncio
async def test_the_workspace_cache_entry_keeps_the_previous_release_format(
    client: TestClient,
) -> None:
    """A replica of the previous release must still rebuild an entry this one wrote."""
    workspace = _workspace(client, auto=True)
    assert client.post("/v3/workspaces", json={"id": workspace}).json()["auto"]

    entry = await cache.get(workspace_crud.workspace_cache_key(workspace))
    assert isinstance(entry, dict)
    assert set(cast(dict[str, Any], entry)) == _PREVIOUS_RELEASE_ENTRY


@pytest.mark.asyncio
async def test_auto_reads_right_from_an_entry_the_previous_release_wrote(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client, auto=True)
    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == workspace)
    )
    assert row is not None
    previous_release_entry = {
        field: getattr(row, field) for field in _PREVIOUS_RELEASE_ENTRY
    }
    await cache.set(
        workspace_crud.workspace_cache_key(workspace), previous_release_entry
    )

    assert client.post("/v3/workspaces", json={"id": workspace}).json()["auto"]


@pytest.mark.asyncio
async def test_a_reused_name_never_reads_the_old_workspaces_flag(
    client: TestClient, db_session: AsyncSession
) -> None:
    """A flag cached for a deleted workspace doesn't answer for a new one.

    The race: a read fills the flag's entry after deletion cleared the cache.
    Simulated by deleting the row and only the main entry, leaving the flag's.
    """
    name = _workspace(client, auto=True)
    assert client.post("/v3/workspaces", json={"id": name}).json()["auto"] is True

    await db_session.execute(
        delete(models.Workspace).where(models.Workspace.name == name)
    )
    await db_session.commit()
    await cache.delete(workspace_crud.workspace_cache_key(name))

    recreated = client.post("/v3/workspaces", json={"id": name})
    assert recreated.status_code == 201, recreated.text
    assert recreated.json()["auto"] is False
    again = client.post("/v3/workspaces", json={"id": name})
    assert again.json()["auto"] is False
    refused = client.post("/v3/workspaces", json={"id": name, "auto": True})
    assert refused.status_code == 409, refused.text
