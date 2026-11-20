"""Sync states: where a connector following a source has got to, and its health.

A connector that follows a source live (catching up from a saved position, then
taking in new events) saves its position here, so a restart on any machine
carries on from it, and `GET /ingest/status` can show each source's last sync,
newest record and last error. The position is the connector's own cursor,
stored as sent.

Each save writes a fresh opaque `revision`. A save that names the revision it
read is refused if another write came between, or if the state was forgotten:
two runners on one source notice each other, and a runner that read a position
before a reset can't put it back. A reset of the import forgets every state, so
connectors read their sources again from the start.

Writes take the workspace fence shared and check the workspace again under it,
so a save can't race the workspace's deletion and leave a row behind.

Experimental, like the rest of ingest.
"""

import datetime
import json
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.ingest_fence import take_workspace_fence
from src.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)

# The largest position a connector may save, serialized.
MAX_POSITION_BYTES = 1_000_000


def _match(workspace_name: str, source: str, source_instance: str) -> list[Any]:
    state = models.IngestSyncState
    return [
        state.tenant_id == models.scoped_tenant_id(),
        state.workspace_name == workspace_name,
        state.source == source,
        state.source_instance == source_instance,
    ]


async def list_sync_states(
    db: AsyncSession, workspace_name: str
) -> list[models.IngestSyncState]:
    """Every source the workspace syncs, by source and instance."""
    state = models.IngestSyncState
    return list(
        (
            await db.scalars(
                select(state)
                .where(
                    state.tenant_id == models.scoped_tenant_id(),
                    state.workspace_name == workspace_name,
                )
                .order_by(state.source, state.source_instance, state.id)
            )
        ).all()
    )


async def get_sync_state(
    db: AsyncSession, workspace_name: str, source: str, source_instance: str
) -> models.IngestSyncState | None:
    """One source's state, or None if nothing has been saved (or it was forgotten)."""
    return await db.scalar(
        select(models.IngestSyncState).where(
            *_match(workspace_name, source, source_instance)
        )
    )


async def _require_auto_workspace(db: AsyncSession, workspace_name: str) -> None:
    """Under the fence: the workspace exists now and is auto."""
    auto_enabled_at = (
        await db.execute(
            select(models.Workspace.auto_enabled_at).where(
                models.Workspace.tenant_id == models.scoped_tenant_id(),
                models.Workspace.name == workspace_name,
            )
        )
    ).one_or_none()
    if auto_enabled_at is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")
    if auto_enabled_at[0] is None:
        raise ConflictException(
            f"Workspace {workspace_name} isn't an auto workspace: only auto"
            + " workspaces sync sources."
        )


async def save_sync_state(
    db: AsyncSession,
    workspace_name: str,
    source: str,
    source_instance: str,
    *,
    position: dict[str, Any],
    expected_revision: str | None,
    last_record_at: datetime.datetime | None,
    error: str | None,
) -> models.IngestSyncState:
    """Save a source's position and health; commit.

    With `expected_revision`, the save happens only if the stored state still has
    that revision; without, it replaces whatever is there. `last_record_at` only
    moves forward: a save without one keeps the stored value.

    Raises:
        ValidationException: The position is larger than MAX_POSITION_BYTES.
        ResourceNotFoundException: The workspace doesn't exist.
        ConflictException: The workspace isn't auto, or `expected_revision` isn't
            the stored revision (another write came between, or the state was
            forgotten).
    """
    if len(json.dumps(position).encode()) > MAX_POSITION_BYTES:
        raise ValidationException(
            f"A sync position is limited to {MAX_POSITION_BYTES:,} bytes"
        )
    await take_workspace_fence(db, workspace_name, mode="shared")
    await _require_auto_workspace(db, workspace_name)

    state = models.IngestSyncState
    revision = generate_nanoid()
    now = func.statement_timestamp()
    if expected_revision is None:
        # The unique key serializes concurrent first saves: the second updates.
        saved = await db.scalar(
            insert(state)
            .values(
                tenant_id=models.scoped_tenant_id(),
                id=generate_nanoid(),
                workspace_name=workspace_name,
                source=source,
                source_instance=source_instance,
                position=position,
                revision=revision,
                last_synced_at=now,
                last_record_at=last_record_at,
                error=error,
            )
            .on_conflict_do_update(
                constraint="uq_ingest_sync_states_source",
                set_={
                    "position": position,
                    "revision": revision,
                    "last_synced_at": now,
                    "last_record_at": func.coalesce(
                        last_record_at, state.last_record_at
                    ),
                    "error": error,
                },
            )
            .returning(state)
            .execution_options(populate_existing=True)
        )
    else:
        saved = await db.scalar(
            update(state)
            .where(
                *_match(workspace_name, source, source_instance),
                state.revision == expected_revision,
            )
            .values(
                position=position,
                revision=revision,
                last_synced_at=now,
                last_record_at=func.coalesce(last_record_at, state.last_record_at),
                error=error,
            )
            .returning(state)
            .execution_options(populate_existing=True)
        )
        if saved is None:
            await db.rollback()
            raise ConflictException(
                f"The sync state of {source}/{source_instance} changed since revision"
                + f" {expected_revision} was read, or was forgotten (a reset): read"
                + " it again before saving"
            )
    await db.commit()
    assert saved is not None
    return saved


async def forget_sync_state(
    db: AsyncSession, workspace_name: str, source: str, source_instance: str
) -> bool:
    """Forget a source's state, so its connector reads it from the start; commit.

    Returns whether there was one.
    """
    await take_workspace_fence(db, workspace_name, mode="shared")
    result = await db.execute(
        delete(models.IngestSyncState)
        .where(*_match(workspace_name, source, source_instance))
        .returning(models.IngestSyncState.id)
    )
    forgotten = result.first() is not None
    await db.commit()
    return forgotten


async def forget_all_sync_states(db: AsyncSession, workspace_name: str) -> None:
    """Forget every source's state, in the caller's transaction (a reset's)."""
    await db.execute(
        delete(models.IngestSyncState).where(
            models.IngestSyncState.tenant_id == models.scoped_tenant_id(),
            models.IngestSyncState.workspace_name == workspace_name,
        )
    )


__all__ = [
    "MAX_POSITION_BYTES",
    "forget_all_sync_states",
    "forget_sync_state",
    "get_sync_state",
    "list_sync_states",
    "save_sync_state",
]
