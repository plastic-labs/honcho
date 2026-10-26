"""Read-only listing of the collections whose next dream is due. Enqueues nothing."""

from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, NamedTuple, cast

from sqlalchemy import (
    Text,
    and_,
    column,
    func,
    literal_column,
    or_,
    select,
    true,
    tuple_,
    values,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dreamer.dream_scheduler import WORK_LIST_BACKLOG_KEY
from src.schemas import DreamType
from src.utils.config_helpers import get_configuration
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key

logger = getLogger(__name__)


def _observed_key(
    tenant_id: str | None, workspace_name: str | None, observed: str | None
) -> tuple[str | None, ...]:
    # region ai
    # Keys the "is representation work still pending for this collection?" check.
    # Under MULTI_TENANT the same (workspace, observed) names recur across tenants, so
    # the tenant must be part of the key — otherwise one tenant's pending work would
    # suppress another tenant's due dream. Flag off (unnamespaced keys) the tenant is
    # dropped, so a parsed key (tenant_id None) and the collection's tenant still align.
    # endregion
    if settings.MULTI_TENANT:
        return (tenant_id, workspace_name, observed)
    return (workspace_name, observed)


class DueDream(NamedTuple):
    """One collection whose next dream is due, with everything needed to enqueue it."""

    workspace_name: str
    observer: str
    observed: str
    dream_type: DreamType
    session_name: str
    # region ai
    # The collection's tenant. Under MULTI_TENANT this scheduler runs cross-tenant on
    # the service session, so the tenant is carried explicitly (like the deriver's
    # work-unit key) to namespace the enqueued dream and to key the pending-work check
    # per tenant. "default" under flag-off (the model default).
    # endregion
    tenant_id: str


async def count_due_dreams(db: AsyncSession) -> int:
    """Count collections whose next dream is due."""
    return len(await list_due_dreams(db))


async def list_due_dreams(db: AsyncSession) -> list[DueDream]:
    """List collections past the threshold, the idle timeout, the min-hours gate, any earlier attempt, pending representation work, and the session's dream setting."""
    dream_types = [
        DreamType(dream_type)
        for dream_type in settings.DREAM.ENABLED_TYPES
        if dream_type == DreamType.OMNI.value
    ]
    if not settings.DREAM.ENABLED or not dream_types:
        return []

    now = datetime.now(UTC)
    idle_cutoff = now - timedelta(minutes=settings.DREAM.IDLE_TIMEOUT_MINUTES)
    rows = (
        await db.execute(
            select(
                models.Collection.workspace_name,
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.internal_metadata,
                models.Collection.explicit_count,
                models.Collection.newest_explicit_at,
                models.Collection.tenant_id,
            ).where(
                or_(
                    and_(
                        literal_column(f"({models.DREAM_DOCUMENTS_SINCE_SQL})")
                        >= settings.DREAM.DOCUMENT_THRESHOLD,
                        models.Collection.newest_explicit_at <= idle_cutoff,
                    ),
                    # Any documents that were selected but not used in the last dream
                    literal_column(models.DREAM_BACKLOG_SQL) > literal_column("0"),
                )
            )
        )
    ).all()

    # A candidate's newest explicit conclusion, or, for a pair whose work list
    # is still behind, its last completed dream: an attempt after that time
    # means the dream is already queued or was tried.
    candidates: dict[str, tuple[DueDream, datetime]] = {}

    for row in rows:
        workspace_name = cast(str, row[0])
        observer = cast(str, row[1])
        observed = cast(str, row[2])
        internal_metadata = cast("dict[str, Any] | None", row[3])
        explicit_count = cast(int, row[4])
        newest_created_at = cast("datetime | None", row[5])
        tenant_id = cast(str, row[6])

        dream_metadata: dict[str, Any] = (internal_metadata or {}).get("dream", {})
        since_last_dream = explicit_count - int(
            dream_metadata.get("last_dream_document_count", 0)
        )
        # The work list's backlog is due even without new conclusions: without
        # them, the attempt check below would never let the pair through again.
        backlog_since = (
            _parse_last_dream_at(dream_metadata.get("last_dream_at"))
            if int(dream_metadata.get(WORK_LIST_BACKLOG_KEY, 0) or 0) > 0
            else None
        )
        if (
            since_last_dream < settings.DREAM.DOCUMENT_THRESHOLD
            and backlog_since is None
        ):
            continue

        if newest_created_at is None or newest_created_at > idle_cutoff:
            continue

        last_dream_at = cast("str | None", dream_metadata.get("last_dream_at"))
        if last_dream_at and _within_min_hours_gate(last_dream_at, now):
            continue

        for dream_type in dream_types:
            work_unit_key = construct_work_unit_key(
                workspace_name,
                {
                    "task_type": "dream",
                    "observer": observer,
                    "observed": observed,
                    "dream_type": dream_type.value,
                },
                tenant_id=tenant_id,
            )
            candidates[work_unit_key] = (
                DueDream(
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    dream_type=dream_type,
                    # Hydrated later on
                    session_name="",
                    tenant_id=tenant_id,
                ),
                max(newest_created_at, backlog_since)
                if backlog_since is not None
                else newest_created_at,
            )

    if not candidates:
        return []

    attempt_rows = (
        await db.execute(
            select(
                models.QueueItem.work_unit_key,
                func.max(models.QueueItem.created_at),
            )
            .where(
                models.QueueItem.task_type == "dream",
                models.QueueItem.work_unit_key.in_(candidates.keys()),
            )
            .group_by(models.QueueItem.work_unit_key)
        )
    ).all()
    newest_attempts: dict[str, datetime] = {
        cast(str, row[0]): cast(datetime, row[1]) for row in attempt_rows
    }

    pending_representation = select(models.QueueItemBatch.work_unit_key).where(
        models.QueueItemBatch.task_type == "representation"
    )
    if settings.MULTI_TENANT:
        tenants = sorted({due_dream.tenant_id for due_dream, _ in candidates.values()})
        pending_representation = pending_representation.where(
            models.QueueItemBatch.tenant_id.in_(tenants)
            | models.QueueItemBatch.tenant_id.is_(None)
        )
    pending_representation_keys = (
        (await db.execute(pending_representation)).scalars().all()
    )
    active_observed = {
        _observed_key(parsed.tenant_id, parsed.workspace_name, parsed.observed)
        for parsed in (parse_work_unit_key(key) for key in pending_representation_keys)
    }

    unattempted = [
        due_dream
        for work_unit_key, (due_dream, newest_created_at) in candidates.items()
        if (
            work_unit_key not in newest_attempts
            or newest_attempts[work_unit_key] < newest_created_at
        )
        and _observed_key(
            due_dream.tenant_id, due_dream.workspace_name, due_dream.observed
        )
        not in active_observed
    ]
    if not unattempted:
        return []

    newest_sessions = await _newest_session_names(db, unattempted)
    with_sessions = [
        due_dream._replace(session_name=session_name)
        for due_dream in unattempted
        if (
            session_name := newest_sessions.get(
                (
                    due_dream.tenant_id,
                    due_dream.workspace_name,
                    due_dream.observer,
                    due_dream.observed,
                )
            )
        )
        is not None
    ]
    if not with_sessions:
        return []

    return await _filter_dreams_enabled(db, with_sessions)


async def _newest_session_names(
    db: AsyncSession, candidates: list[DueDream]
) -> dict[tuple[str, str, str, str], str | None]:
    """The session of each candidate collection's newest explicit document."""
    keys = values(
        column("tenant_id", Text),
        column("workspace_name", Text),
        column("observer", Text),
        column("observed", Text),
        name="keys",
    ).data(
        list(
            {
                (
                    due_dream.tenant_id,
                    due_dream.workspace_name,
                    due_dream.observer,
                    due_dream.observed,
                )
                for due_dream in candidates
            }
        )
    )
    newest = (
        select(models.Document.session_name)
        .where(
            models.Document.tenant_id == keys.c.tenant_id,
            models.Document.observer == keys.c.observer,
            models.Document.observed == keys.c.observed,
            models.Document.workspace_name == keys.c.workspace_name,
            models.Document.level == "explicit",
        )
        .order_by(models.Document.created_at.desc())
        .limit(1)
        .lateral("newest")
    )
    rows = (
        await db.execute(
            select(
                keys.c.tenant_id,
                keys.c.workspace_name,
                keys.c.observer,
                keys.c.observed,
                newest.c.session_name,
            ).select_from(keys.join(newest, true()))
        )
    ).all()
    return {
        (
            cast(str, row[0]),
            cast(str, row[1]),
            cast(str, row[2]),
            cast(str, row[3]),
        ): cast("str | None", row[4])
        for row in rows
    }


async def _filter_dreams_enabled(
    db: AsyncSession, candidates: list[DueDream]
) -> list[DueDream]:
    """Drop candidates whose resolved configuration has dreams turned off."""
    workspace_keys = sorted(
        {(due_dream.tenant_id, due_dream.workspace_name) for due_dream in candidates}
    )
    session_keys = sorted(
        {
            (due_dream.tenant_id, due_dream.workspace_name, due_dream.session_name)
            for due_dream in candidates
        }
    )

    # ai: lookups are keyed by tenant as well as name — this runs on the
    # cross-tenant service session, so a name-only key would let one tenant's
    # same-named workspace/session decide another tenant's dream.enabled config.
    # The tenant goes in the SQL, not only in the dicts below, so each lookup is a
    # probe of the (tenant_id, name[, workspace_name]) unique index instead of a
    # scan of every tenant's same-named rows.
    workspaces = {
        (workspace.tenant_id, workspace.name): workspace
        for workspace in (
            await db.execute(
                select(models.Workspace).where(
                    tuple_(models.Workspace.tenant_id, models.Workspace.name).in_(
                        workspace_keys
                    )
                )
            )
        )
        .scalars()
        .all()
    }

    sessions: dict[tuple[str, str, str], models.Session] = {}
    if session_keys:
        session_rows = (
            (
                await db.execute(
                    select(models.Session).where(
                        tuple_(
                            models.Session.tenant_id,
                            models.Session.workspace_name,
                            models.Session.name,
                        ).in_(session_keys)
                    )
                )
            )
            .scalars()
            .all()
        )
        sessions = {
            (session.tenant_id, session.workspace_name, session.name): session
            for session in session_rows
        }

    enabled: list[DueDream] = []
    for due_dream in candidates:
        configuration = get_configuration(
            None,
            sessions.get(
                (
                    due_dream.tenant_id,
                    due_dream.workspace_name,
                    due_dream.session_name,
                )
            ),
            workspaces.get((due_dream.tenant_id, due_dream.workspace_name)),
        )
        if configuration.dream.enabled:
            enabled.append(due_dream)
    return enabled


def _parse_last_dream_at(last_dream_at: Any) -> datetime | None:
    """The stored `last_dream_at` as an aware datetime, or None if unusable."""
    try:
        last_dream_time = datetime.fromisoformat(last_dream_at)
    except (ValueError, TypeError):
        return None
    if last_dream_time.tzinfo is None:
        last_dream_time = last_dream_time.replace(tzinfo=UTC)
    return last_dream_time


def _within_min_hours_gate(last_dream_at: str, now: datetime) -> bool:
    """True when the last dream is too recent for another one."""
    last_dream_time = _parse_last_dream_at(last_dream_at)
    if last_dream_time is None:
        return False

    hours_since = (now - last_dream_time).total_seconds() / 3600
    return hours_since < settings.DREAM.MIN_HOURS_BETWEEN_DREAMS
