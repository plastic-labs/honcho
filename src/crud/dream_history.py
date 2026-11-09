"""Representation History: what each dream changed in a representation.

A dream's writes record their own history rows in the same transaction as the
write, through the ``record_*`` helpers here. The orchestrator opens a run
before its specialists start and closes it when they finish, so a dream that
fails partway still shows the changes it committed.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import (
    CursorResult,
    Select,
    and_,
    delete,
    literal,
    select,
    tuple_,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement, func

from src import models, schemas
from src.config import settings
from src.crud.lifecycle import (
    live_document_row,
    live_dream_run_change_row,
    live_dream_run_row,
)
from src.dependencies import service_db, tracked_db
from src.exceptions import ResourceNotFoundException
from src.utils.types import DocumentLevel

logger = logging.getLogger(__name__)

# A run still marked running after this long was cut off (worker crash or
# shutdown) without closing, and is reported as interrupted.
_STALE_RUN_AFTER = datetime.timedelta(hours=24)
_PRUNE_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class DreamHistoryTarget:
    """Where a dream's writes record their changes: an open run and its specialist."""

    run_id: str
    specialist: str | None = None


@dataclass(frozen=True, slots=True)
class CreatedConclusion:
    id: str
    level: DocumentLevel
    source_ids: list[str]


@dataclass(frozen=True, slots=True)
class RemovedConclusion:
    id: str
    level: DocumentLevel
    content: str
    session_name: str | None
    replaced_by: str | None = None


def _run_key(run_id: str) -> tuple[ColumnElement[bool], ColumnElement[bool]]:
    """The predicate for one run in the bound tenant."""
    return (
        models.DreamRun.tenant_id == models.scoped_tenant_id(),
        models.DreamRun.id == run_id,
    )


async def open_run(
    *,
    run_id: str,
    workspace_name: str,
    observer: str,
    observed: str,
    dream_type: str | None,
    trigger_reason: str | None,
) -> DreamHistoryTarget | None:
    """Record the start of a dream. Returns None when history is off or the insert fails.

    A failure here never stops the dream: it runs without recording history.
    """
    if not settings.DREAM.HISTORY_ENABLED:
        return None
    try:
        async with tracked_db("dream_history.open_run") as db:
            db.add(
                models.DreamRun(
                    id=run_id,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    dream_type=dream_type,
                    trigger_reason=trigger_reason,
                )
            )
            await db.commit()
    except Exception:
        logger.warning(
            "[%s] Could not open a dream history run; the dream runs unrecorded",
            run_id,
            exc_info=True,
        )
        return None
    return DreamHistoryTarget(run_id=run_id)


async def close_run(run_id: str, *, succeeded: bool) -> None:
    """Mark a run completed or failed. Never raises."""
    try:
        async with tracked_db("dream_history.close_run") as db:
            await db.execute(
                update(models.DreamRun)
                .where(
                    *_run_key(run_id),
                    models.DreamRun.status == "running",
                )
                .values(
                    status="completed" if succeeded else "failed",
                    finished_at=func.now(),
                )
            )
            await db.commit()
    except Exception:
        logger.warning(
            "[%s] Could not close the dream history run", run_id, exc_info=True
        )


async def record_created(
    db: AsyncSession,
    target: DreamHistoryTarget,
    workspace_name: str,
    created: Sequence[CreatedConclusion],
) -> None:
    """Add `created` rows and bump the run's count. The caller commits."""
    if not created:
        return
    db.add_all(
        models.DreamRunChange(
            run_id=target.run_id,
            workspace_name=workspace_name,
            kind="created",
            specialist=target.specialist,
            conclusion_id=conclusion.id,
            level=conclusion.level,
            source_ids=conclusion.source_ids or None,
        )
        for conclusion in created
    )
    await db.execute(
        update(models.DreamRun)
        .where(*_run_key(target.run_id))
        .values(created_count=models.DreamRun.created_count + len(created))
    )


async def record_removed(
    db: AsyncSession,
    target: DreamHistoryTarget,
    workspace_name: str,
    removed: Sequence[RemovedConclusion],
) -> None:
    """Add `removed` rows, copying each conclusion's content. The caller commits."""
    if not removed:
        return
    db.add_all(
        models.DreamRunChange(
            run_id=target.run_id,
            workspace_name=workspace_name,
            kind="removed",
            specialist=target.specialist,
            conclusion_id=conclusion.id,
            level=conclusion.level,
            content=conclusion.content,
            session_name=conclusion.session_name,
            replaced_by=conclusion.replaced_by,
        )
        for conclusion in removed
    )
    await db.execute(
        update(models.DreamRun)
        .where(*_run_key(target.run_id))
        .values(removed_count=models.DreamRun.removed_count + len(removed))
    )


async def record_peer_card(
    db: AsyncSession,
    target: DreamHistoryTarget,
    workspace_name: str,
    *,
    before: list[str] | None,
    after: list[str],
) -> None:
    """Add a `peer_card` row with the card before and after. The caller commits."""
    db.add(
        models.DreamRunChange(
            run_id=target.run_id,
            workspace_name=workspace_name,
            kind="peer_card",
            specialist=target.specialist,
            peer_card_before=before,
            peer_card_after=after,
        )
    )
    await db.execute(
        update(models.DreamRun)
        .where(*_run_key(target.run_id))
        .values(peer_card_updated=True)
    )


def list_runs_stmt(
    workspace_name: str,
    *,
    observed: str | None = None,
    observer: str | None = None,
) -> Select[tuple[models.DreamRun]]:
    """Runs in a workspace, newest first, optionally for one observed/observer peer."""
    stmt = select(models.DreamRun).where(
        models.DreamRun.tenant_id == models.scoped_tenant_id(),
        models.DreamRun.workspace_name == workspace_name,
        live_dream_run_row(),
    )
    if observed is not None:
        stmt = stmt.where(models.DreamRun.observed == observed)
    if observer is not None:
        stmt = stmt.where(models.DreamRun.observer == observer)
    return stmt.order_by(models.DreamRun.started_at.desc(), models.DreamRun.id.desc())


async def get_run_detail(
    db: AsyncSession, workspace_name: str, run_id: str
) -> schemas.DreamRunDetail:
    """A run and its changes, with conclusion content resolved for display.

    Created conclusions and sources are looked up live first. One a later
    dream removed resolves from that dream's copy; one deleted through the
    API, or by deleting its session, resolves to `unavailable`. A run whose
    workspace, observer or observed peer is deleted is not found.
    """
    run = await db.scalar(
        select(models.DreamRun).where(
            *_run_key(run_id),
            models.DreamRun.workspace_name == workspace_name,
            live_dream_run_row(),
        )
    )
    if run is None:
        raise ResourceNotFoundException(f"Dream run {run_id} not found")

    changes = (
        await db.scalars(
            select(models.DreamRunChange)
            .where(
                models.DreamRunChange.tenant_id == run.tenant_id,
                models.DreamRunChange.run_id == run_id,
                live_dream_run_change_row(),
            )
            .order_by(models.DreamRunChange.id)
        )
    ).all()

    created_changes = [c for c in changes if c.kind == "created" and c.conclusion_id]
    to_resolve: set[str] = set()
    for change in created_changes:
        to_resolve.add(cast(str, change.conclusion_id))
        to_resolve.update(change.source_ids or [])
    resolved = await _resolve_conclusions(
        db,
        workspace_name,
        observer=run.observer,
        observed=run.observed,
        conclusion_ids=to_resolve,
    )

    created = [
        schemas.DreamCreatedConclusion(
            conclusion=resolved[cast(str, change.conclusion_id)],
            sources=[resolved[source_id] for source_id in change.source_ids or []],
            specialist=change.specialist,
        )
        for change in created_changes
    ]
    removed = [
        schemas.DreamRemovedConclusion(
            id=change.conclusion_id,
            content=cast(str, change.content),
            level=change.level,
            replaced_by=change.replaced_by,
            specialist=change.specialist,
        )
        for change in changes
        if change.kind == "removed" and change.conclusion_id
    ]
    card_changes = [c for c in changes if c.kind == "peer_card"]
    peer_card = (
        schemas.DreamPeerCardChange(
            before=card_changes[0].peer_card_before,
            after=card_changes[-1].peer_card_after or [],
        )
        if card_changes
        else None
    )
    return schemas.DreamRunDetail(
        **schemas.DreamRun.model_validate(run).model_dump(),
        created=created,
        removed=removed,
        peer_card=peer_card,
    )


async def _resolve_conclusions(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    conclusion_ids: set[str],
) -> dict[str, schemas.DreamConclusionRef]:
    resolved = {
        conclusion_id: schemas.DreamConclusionRef(
            id=conclusion_id, status="unavailable"
        )
        for conclusion_id in conclusion_ids
    }
    if not conclusion_ids:
        return resolved

    live = await db.execute(
        select(models.Document.id, models.Document.content, models.Document.level)
        .where(
            models.Document.id.in_(conclusion_ids),
            models.Document.observer == observer,
            models.Document.observed == observed,
            live_document_row(workspace_name),
        )
        .order_by(models.Document.id)
    )
    for row in live.all():
        resolved[row.id] = schemas.DreamConclusionRef(
            id=row.id, content=row.content, level=row.level, status="current"
        )

    missing = [cid for cid, ref in resolved.items() if ref.status == "unavailable"]
    if missing:
        removed = await db.execute(
            select(
                models.DreamRunChange.conclusion_id,
                models.DreamRunChange.content,
                models.DreamRunChange.level,
                models.DreamRunChange.run_id,
            )
            # The copy's own run must be live too: a source can come from
            # another peer pair, whose runs may be hidden.
            .join(
                models.DreamRun,
                and_(
                    models.DreamRun.tenant_id == models.DreamRunChange.tenant_id,
                    models.DreamRun.id == models.DreamRunChange.run_id,
                ),
            )
            .where(
                models.DreamRunChange.tenant_id == models.scoped_tenant_id(),
                models.DreamRunChange.workspace_name == workspace_name,
                models.DreamRunChange.kind == "removed",
                models.DreamRunChange.conclusion_id.in_(missing),
                live_dream_run_change_row(),
                live_dream_run_row(),
            )
            .order_by(models.DreamRunChange.id)
        )
        for row in removed.all():
            resolved[cast(str, row.conclusion_id)] = schemas.DreamConclusionRef(
                id=cast(str, row.conclusion_id),
                content=row.content,
                level=row.level,
                status="removed",
                removed_in_run=row.run_id,
            )
    return resolved


async def delete_session_history(
    db: AsyncSession, workspace_name: str, session_name: str
) -> None:
    """Delete removed-conclusion copies that came from a session. The caller commits.

    Run it after the session's documents are deleted, so a dream still committing
    a copy of one of them has finished first.
    """
    tenant_id = models.scoped_tenant_id()
    # Locked in id order: an IN predicate alone imposes none.
    await db.execute(
        delete(models.DreamRunChange).where(
            models.DreamRunChange.tenant_id == tenant_id,
            models.DreamRunChange.id.in_(
                select(models.DreamRunChange.id)
                .where(
                    models.DreamRunChange.tenant_id == tenant_id,
                    models.DreamRunChange.workspace_name == workspace_name,
                    models.DreamRunChange.session_name == session_name,
                )
                .order_by(models.DreamRunChange.id)
                .with_for_update()
            ),
        )
    )


async def erase_conclusion_history(
    db: AsyncSession, workspace_name: str, conclusion_id: str
) -> int:
    """Delete the copies history keeps of a removed conclusion. The caller commits.

    For a user-requested deletion: afterwards the conclusion reads as
    `unavailable` everywhere history references it. Returns the copies deleted.
    """
    tenant_id = models.scoped_tenant_id()
    copies = (
        models.DreamRunChange.tenant_id == tenant_id,
        models.DreamRunChange.workspace_name == workspace_name,
        models.DreamRunChange.kind == "removed",
        models.DreamRunChange.conclusion_id == conclusion_id,
    )
    # The copies' runs first, in id order: workspace deletion and the prune lock
    # runs before the changes that cascade from them.
    await db.execute(
        select(models.DreamRun.id)
        .where(
            models.DreamRun.tenant_id == tenant_id,
            models.DreamRun.id.in_(select(models.DreamRunChange.run_id).where(*copies)),
        )
        .order_by(models.DreamRun.id)
        .with_for_update()
    )
    # Then the copies in id order, as session deletion and the prune take them.
    result = cast(
        CursorResult[Any],
        await db.execute(
            delete(models.DreamRunChange).where(
                models.DreamRunChange.tenant_id == tenant_id,
                models.DreamRunChange.id.in_(
                    select(models.DreamRunChange.id)
                    .where(*copies)
                    .order_by(models.DreamRunChange.id)
                    .with_for_update()
                ),
            )
        ),
    )
    return result.rowcount


async def delete_workspace_history(db: AsyncSession, workspace_name: str) -> None:
    """Delete every run in a workspace; their changes cascade. The caller commits.

    Runs are locked in id order, the same order the retention prune takes them.
    """
    tenant_id = models.scoped_tenant_id()
    await db.execute(
        delete(models.DreamRun).where(
            models.DreamRun.tenant_id == tenant_id,
            models.DreamRun.id.in_(
                select(models.DreamRun.id)
                .where(
                    models.DreamRun.tenant_id == tenant_id,
                    models.DreamRun.workspace_name == workspace_name,
                )
                .order_by(models.DreamRun.id)
                .with_for_update()
            ),
        )
    )


async def prune_dream_history() -> int:
    """Mark stale running runs interrupted and apply the retention window.

    Past the window, a run keeps only its `created` changes whose conclusion
    still exists, so which dream made a live conclusion stays readable. Its
    other changes are deleted and the run gets `pruned_at`. A run with nothing
    left is deleted. Returns the number of runs deleted.
    """
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(days=settings.DREAM.HISTORY_RETENTION_DAYS)
    deleted = 0
    run_key = tuple_(models.DreamRun.tenant_id, models.DreamRun.id)
    change_run_key = tuple_(
        models.DreamRunChange.tenant_id, models.DreamRunChange.run_id
    )
    change_key = tuple_(models.DreamRunChange.tenant_id, models.DreamRunChange.id)
    conclusion_exists = (
        select(models.Document.id)
        .where(
            models.Document.tenant_id == models.DreamRunChange.tenant_id,
            models.Document.id == models.DreamRunChange.conclusion_id,
            models.Document.deleted_at.is_(None),
        )
        .exists()
    )
    has_changes = (
        select(models.DreamRunChange.id)
        .where(
            models.DreamRunChange.tenant_id == models.DreamRun.tenant_id,
            models.DreamRunChange.run_id == models.DreamRun.id,
        )
        .exists()
    )
    # Runs across every tenant, so it uses the cross-tenant service session.
    async with service_db("dream_history.prune") as db:
        await db.execute(
            update(models.DreamRun)
            .where(
                run_key.in_(
                    select(models.DreamRun.tenant_id, models.DreamRun.id)
                    .where(
                        models.DreamRun.status == "running",
                        models.DreamRun.started_at < now - _STALE_RUN_AFTER,
                    )
                    .order_by(models.DreamRun.tenant_id, models.DreamRun.id)
                    .with_for_update()
                )
            )
            .values(status="interrupted")
        )
        await db.commit()

        after: tuple[str, str] | None = None
        while True:
            # Runs are locked before their changes, the order workspace
            # deletion takes them in. Within a tenant that is id order.
            batch_stmt = (
                select(models.DreamRun.tenant_id, models.DreamRun.id)
                .where(models.DreamRun.started_at < cutoff)
                .order_by(models.DreamRun.tenant_id, models.DreamRun.id)
                .limit(_PRUNE_BATCH_SIZE)
                .with_for_update()
            )
            if after is not None:
                batch_stmt = batch_stmt.where(run_key > tuple_(*map(literal, after)))
            batch = [(row[0], row[1]) for row in (await db.execute(batch_stmt)).all()]
            if not batch:
                break
            after = batch[-1]

            # Changes are locked in id order, as session deletion takes them.
            pruned = await db.execute(
                delete(models.DreamRunChange)
                .where(
                    change_key.in_(
                        select(
                            models.DreamRunChange.tenant_id, models.DreamRunChange.id
                        )
                        .where(
                            change_run_key.in_(batch),
                            ~and_(
                                models.DreamRunChange.kind == "created",
                                conclusion_exists,
                            ),
                        )
                        .order_by(
                            models.DreamRunChange.tenant_id, models.DreamRunChange.id
                        )
                        .with_for_update(of=models.DreamRunChange)
                    )
                )
                .returning(
                    models.DreamRunChange.tenant_id, models.DreamRunChange.run_id
                )
            )
            pruned_runs = {(row[0], row[1]) for row in pruned.all()}
            if pruned_runs:
                await db.execute(
                    update(models.DreamRun)
                    .where(
                        run_key.in_(sorted(pruned_runs)),
                        models.DreamRun.pruned_at.is_(None),
                    )
                    .values(pruned_at=now)
                )
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(models.DreamRun).where(run_key.in_(batch), ~has_changes)
                ),
            )
            await db.commit()
            deleted += result.rowcount
            if len(batch) < _PRUNE_BATCH_SIZE:
                break
    return deleted
