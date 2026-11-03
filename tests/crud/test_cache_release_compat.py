"""Cached rows stay in the previous release's format.

During a rolling deploy both releases share the cache. Each must read the other's
entries, and a write by either must clear the key the other reads, or the other
keeps serving the row as it was before the write until the entry expires. So the
key templates keep their version and the cached dicts keep their fields: tenant_id
is left out and restored on read (models.with_cached_tenant).
"""

from typing import Any, cast

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.cache.client import cache
from src.crud.session import session_cache_key

# The session entry's fields in the previous release; tenant_id is not one of them.
_PREVIOUS_RELEASE_SESSION_FIELDS = {
    "id",
    "name",
    "workspace_name",
    "is_active",
    "h_metadata",
    "internal_metadata",
    "configuration",
    "created_at",
}


@pytest.mark.asyncio
async def test_a_cached_session_has_the_previous_releases_fields(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _ = sample_data
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()

    await crud.get_session(db_session, session.name, workspace.name)

    cached: dict[str, Any] | None = await cache.get(
        session_cache_key(workspace.name, session.name)
    )
    assert cached is not None
    assert set(cached) == _PREVIOUS_RELEASE_SESSION_FIELDS


@pytest.mark.asyncio
async def test_a_previous_release_entry_reads_and_updates(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """An entry the previous release wrote rebuilds a row this release can update."""
    workspace, _ = sample_data
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    entry: dict[str, Any] = {
        "id": session.id,
        "name": session.name,
        "workspace_name": session.workspace_name,
        "is_active": True,
        "h_metadata": {},
        "internal_metadata": {},
        "configuration": {},
        "created_at": session.created_at,
    }
    await cache.set(session_cache_key(workspace.name, session.name), entry, expire=60)

    read = await crud.get_session(db_session, session.name, workspace.name)
    assert read.tenant_id == models.instance_tenant_id()

    updated = await crud.update_session(
        db_session,
        schemas.SessionUpdate(metadata={"k": "v"}),
        workspace.name,
        session.name,
    )
    assert updated.h_metadata == {"k": "v"}
    # The write cleared the key the previous release reads.
    assert await cache.get(session_cache_key(workspace.name, session.name)) is None


@pytest.mark.asyncio
async def test_no_create_path_caches_tenant_id(db_session: AsyncSession) -> None:
    """The create paths write the cache directly, not through the _fetch_* readers."""
    workspace = await crud.get_or_create_workspace(
        db_session, schemas.WorkspaceCreate(name=str(generate_nanoid()))
    )
    await workspace.post_commit()
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(name=str(generate_nanoid())),
        workspace.resource.name,
    )
    await session.post_commit()
    peer_name = str(generate_nanoid())
    db_session.add(models.Peer(name=peer_name, workspace_name=workspace.resource.name))
    await db_session.commit()
    collection = await crud.get_or_create_collection(
        db_session, workspace.resource.name, observer=peer_name, observed=peer_name
    )
    assert collection is not None

    entries: list[Any] = [await cache.get(key) async for key in cache.scan("*")]
    rows: list[dict[str, Any]] = [
        cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict) and "id" in entry
    ]
    assert len(rows) >= 2  # workspace and session; collection only on some paths
    assert not [row for row in rows if "tenant_id" in row]
