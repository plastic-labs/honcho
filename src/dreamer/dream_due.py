"""Read-only count of the collections whose next dream is due. Enqueues nothing."""

from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.schemas import DreamType
from src.utils.config_helpers import get_configuration
from src.utils.work_unit import construct_work_unit_key

logger = getLogger(__name__)


async def count_due_dreams(db: AsyncSession) -> int:
    """Count collections past the threshold, the idle timeout, the min-hours gate, any earlier attempt, and the session's dream setting."""
    dream_types = [
        DreamType(dream_type)
        for dream_type in settings.DREAM.ENABLED_TYPES
        if dream_type == DreamType.OMNI.value
    ]
    if not settings.DREAM.ENABLED or not dream_types:
        return 0

    explicit_counts = (
        select(
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
            func.count(models.Document.id).label("explicit_count"),
            func.max(models.Document.created_at).label("newest_created_at"),
            func.array_agg(
                aggregate_order_by(
                    models.Document.session_name, models.Document.created_at.desc()
                )
            )[1].label("newest_session_name"),
        )
        .where(models.Document.level == "explicit")
        .group_by(
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
        )
        .subquery()
    )

    rows = (
        await db.execute(
            select(
                models.Collection.workspace_name,
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.internal_metadata,
                func.coalesce(explicit_counts.c.explicit_count, 0),
                explicit_counts.c.newest_created_at,
                explicit_counts.c.newest_session_name,
            ).outerjoin(
                explicit_counts,
                (models.Collection.workspace_name == explicit_counts.c.workspace_name)
                & (models.Collection.observer == explicit_counts.c.observer)
                & (models.Collection.observed == explicit_counts.c.observed),
            )
        )
    ).all()

    now = datetime.now(UTC)
    idle_cutoff = now - timedelta(minutes=settings.DREAM.IDLE_TIMEOUT_MINUTES)
    candidates: dict[str, tuple[str, str, datetime]] = {}

    for row in rows:
        workspace_name = cast(str, row[0])
        observer = cast(str, row[1])
        observed = cast(str, row[2])
        internal_metadata = cast("dict[str, Any] | None", row[3])
        explicit_count = cast(int, row[4])
        newest_created_at = cast("datetime | None", row[5])
        newest_session_name = cast("str | None", row[6])

        dream_metadata: dict[str, Any] = (internal_metadata or {}).get("dream", {})
        since_last_dream = explicit_count - int(
            dream_metadata.get("last_dream_document_count", 0)
        )
        if since_last_dream < settings.DREAM.DOCUMENT_THRESHOLD:
            continue

        if newest_created_at is None or newest_created_at > idle_cutoff:
            continue

        if newest_session_name is None:
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
            )
            candidates[work_unit_key] = (
                workspace_name,
                newest_session_name,
                newest_created_at,
            )

    if not candidates:
        return 0

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

    unattempted = [
        (workspace_name, session_name)
        for work_unit_key, (
            workspace_name,
            session_name,
            newest_created_at,
        ) in candidates.items()
        if work_unit_key not in newest_attempts
        or newest_attempts[work_unit_key] < newest_created_at
    ]
    if not unattempted:
        return 0

    return await _count_with_dreams_enabled(db, unattempted)


async def _count_with_dreams_enabled(
    db: AsyncSession, candidates: list[tuple[str, str]]
) -> int:
    """Drop candidates whose resolved configuration has dreams turned off."""
    workspace_names = {workspace_name for workspace_name, _ in candidates}
    session_keys = set(candidates)

    workspaces = {
        workspace.name: workspace
        for workspace in (
            await db.execute(
                select(models.Workspace).where(
                    models.Workspace.name.in_(workspace_names)
                )
            )
        )
        .scalars()
        .all()
    }

    sessions: dict[tuple[str, str], models.Session] = {}
    if session_keys:
        session_rows = (
            (
                await db.execute(
                    select(models.Session).where(
                        models.Session.workspace_name.in_(workspace_names),
                        models.Session.name.in_(
                            {session_name for _, session_name in candidates}
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        sessions = {
            (session.workspace_name, session.name): session for session in session_rows
        }

    enabled = 0
    for workspace_name, session_name in candidates:
        configuration = get_configuration(
            None,
            sessions.get((workspace_name, session_name)),
            workspaces.get(workspace_name),
        )
        if configuration.dream.enabled:
            enabled += 1
    return enabled


def _within_min_hours_gate(last_dream_at: str, now: datetime) -> bool:
    """True when the last dream is too recent for another one."""
    try:
        last_dream_time = datetime.fromisoformat(last_dream_at)
    except (ValueError, TypeError):
        return False

    if last_dream_time.tzinfo is None:
        last_dream_time = last_dream_time.replace(tzinfo=UTC)

    hours_since = (now - last_dream_time).total_seconds() / 3600
    return hours_since < settings.DREAM.MIN_HOURS_BETWEEN_DREAMS
