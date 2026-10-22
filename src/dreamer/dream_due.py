"""Read-only listing of the collections whose next dream is due. Enqueues nothing."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, NamedTuple, cast

from sqlalchemy import BigInteger, ColumnElement, func, select, text, tuple_
from sqlalchemy import cast as sa_cast
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
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
    # True when only the slow sweep, not the lookback window, found it due.
    found_by_sweep: bool = False


# Max rows bound into one IN list or unnest batch.
_CHUNK = 1000


def _chunks[T](items: Sequence[T], size: int = _CHUNK) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass
class _Candidate:
    """A collection under consideration, with what the due rule needs so far."""

    tenant_id: str
    workspace_name: str
    observer: str
    observed: str
    internal_metadata: dict[str, Any]
    newest_created_at: datetime | None
    newest_session_name: str | None
    from_sweep: bool

    @property
    def collection_key(self) -> tuple[str, str, str, str]:
        return (self.tenant_id, self.workspace_name, self.observer, self.observed)

    @property
    def dream_metadata(self) -> dict[str, Any]:
        return self.internal_metadata.get("dream", {})

    def work_unit_key(self, dream_type: DreamType) -> str:
        return construct_work_unit_key(
            self.workspace_name,
            {
                "task_type": "dream",
                "observer": self.observer,
                "observed": self.observed,
                "dream_type": dream_type.value,
            },
            tenant_id=self.tenant_id,
        )


def sweep_slots() -> int:
    """How many polls one sweep cycle spans; 0 when the sweep is off."""
    if not settings.DREAM.DUE_SWEEP_HOURS:
        return 0
    return max(
        1,
        settings.DREAM.DUE_SWEEP_HOURS
        * 3600
        // settings.DREAM.DUE_POLL_INTERVAL_SECONDS,
    )


def sweep_slot_at(epoch_seconds: float) -> int | None:
    """The sweep slot a poll at this time visits, or None when the sweep is off."""
    slots = sweep_slots()
    if not slots:
        return None
    return int(epoch_seconds // settings.DREAM.DUE_POLL_INTERVAL_SECONDS) % slots


def collection_sweep_slot(slots: int) -> ColumnElement[int]:
    """SQL for a collection's sweep slot: a stable hash of its (tenant_id, id)."""
    # region ai
    # Stateless on purpose: every poll derives its slot from the clock and every
    # collection's slot from its own key, so a lease handover neither skips nor
    # repeats part of the cycle and no cursor has to be stored. hashtext is int4,
    # so it's widened before the modulo, and the double modulo keeps a negative
    # hash in [0, slots).
    # endregion
    hashed = sa_cast(
        func.hashtext(
            models.Collection.tenant_id.concat(":").concat(models.Collection.id)
        ),
        BigInteger,
    )
    return ((hashed % slots) + slots) % slots


async def count_due_dreams(db: AsyncSession) -> int:
    """Count collections whose next dream is due (the lookback window only)."""
    return len(await list_due_dreams(db))


async def list_due_dreams(
    db: AsyncSession,
    *,
    sweep_slot: int | None = None,
    now: datetime | None = None,
) -> list[DueDream]:
    """List collections past the threshold, the idle timeout, the min-hours gate, any earlier attempt, pending representation work, and the session's dream setting.

    Candidates come from the lookback window (collections with a recent explicit
    document) and, when `sweep_slot` is given, from that slot of the slow sweep.
    Due dreams come back oldest newest-document first.
    """
    dream_types = [
        DreamType(dream_type)
        for dream_type in settings.DREAM.ENABLED_TYPES
        if dream_type == DreamType.OMNI.value
    ]
    if not settings.DREAM.ENABLED or not dream_types:
        return []

    now = now or datetime.now(UTC)
    idle_cutoff = now - timedelta(minutes=settings.DREAM.IDLE_TIMEOUT_MINUTES)

    candidates = await _window_candidates(db, now=now, idle_cutoff=idle_cutoff)
    if sweep_slot is not None and (slots := sweep_slots()):
        seen = {c.collection_key for c in candidates}
        candidates.extend(
            c
            for c in await _sweep_candidates(db, slot=sweep_slot, slots=slots)
            if c.collection_key not in seen
        )
    candidates = [c for c in candidates if not _min_hours_gated(c, now)]
    if not candidates:
        return []
    candidates = await _drop_pending_representation(db, candidates)

    # region ai
    # Cheapest first. A window candidate already knows its newest document, so the
    # prior-attempt and dream.enabled checks run before the per-collection count.
    # A sweep candidate learns its newest document only after the count, so for it
    # the count comes first and only the survivors get the newest-document lookup.
    # endregion
    window = [c for c in candidates if not c.from_sweep]
    window = await _drop_disabled(db, await _drop_attempted(db, window, dream_types))
    window = await _drop_below_threshold(db, window)

    swept = await _drop_below_threshold(db, [c for c in candidates if c.from_sweep])
    swept = [
        c
        for c in await _fill_newest(db, swept)
        if c.newest_created_at is not None and c.newest_created_at <= idle_cutoff
    ]
    swept = await _drop_disabled(db, await _drop_attempted(db, swept, dream_types))

    due = sorted(
        window + swept,
        key=lambda c: cast(datetime, c.newest_created_at),
    )
    return [
        DueDream(
            workspace_name=c.workspace_name,
            observer=c.observer,
            observed=c.observed,
            dream_type=dream_type,
            session_name=cast(str, c.newest_session_name),
            tenant_id=c.tenant_id,
            found_by_sweep=c.from_sweep,
        )
        for c in due
        for dream_type in dream_types
    ]


async def _window_candidates(
    db: AsyncSession, *, now: datetime, idle_cutoff: datetime
) -> list[_Candidate]:
    """Collections whose newest explicit document is inside the lookback and idle."""
    # region ai
    # A collection that stopped receiving documents at N is due by
    # N + max(idle, min_hours) (the validator on DUE_LOOKBACK_HOURS), so a
    # collection whose newest explicit document predates the lookback can only be
    # due if something blocked it for longer; the sweep catches those. The newest
    # document is inside the window by construction, so DISTINCT ON gives the
    # exact newest created_at and session. ix_documents_explicit_created_at serves
    # the range. Grouped by tenant too: this runs cross-tenant on the service
    # session, and (workspace, observer, observed) is unique only per tenant.
    # endregion
    window_start = now - timedelta(hours=settings.DREAM.DUE_LOOKBACK_HOURS)
    newest = (
        select(
            models.Document.tenant_id,
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
            models.Document.created_at,
            models.Document.session_name,
        )
        .where(
            models.Document.level == "explicit",
            models.Document.created_at > window_start,
        )
        .distinct(
            models.Document.tenant_id,
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
        )
        .order_by(
            models.Document.tenant_id,
            models.Document.workspace_name,
            models.Document.observer,
            models.Document.observed,
            models.Document.created_at.desc(),
        )
        .subquery()
    )
    rows = (
        await db.execute(select(newest).where(newest.c.created_at <= idle_cutoff))
    ).all()
    newest_by_key = {
        (row[0], row[1], row[2], row[3]): (row[4], row[5])
        for row in rows
        if row[5] is not None
    }
    if not newest_by_key:
        return []

    candidates: list[_Candidate] = []
    for chunk in _chunks(list(newest_by_key)):
        collections = (
            await db.execute(
                select(
                    models.Collection.tenant_id,
                    models.Collection.workspace_name,
                    models.Collection.observer,
                    models.Collection.observed,
                    models.Collection.internal_metadata,
                ).where(
                    tuple_(
                        models.Collection.tenant_id,
                        models.Collection.workspace_name,
                        models.Collection.observer,
                        models.Collection.observed,
                    ).in_(chunk)
                )
            )
        ).all()
        for row in collections:
            key = (row[0], row[1], row[2], row[3])
            newest_created_at, newest_session_name = newest_by_key[key]
            candidates.append(
                _Candidate(
                    tenant_id=row[0],
                    workspace_name=row[1],
                    observer=row[2],
                    observed=row[3],
                    internal_metadata=cast("dict[str, Any] | None", row[4]) or {},
                    newest_created_at=newest_created_at,
                    newest_session_name=newest_session_name,
                    from_sweep=False,
                )
            )
    return candidates


async def _sweep_candidates(
    db: AsyncSession, *, slot: int, slots: int
) -> list[_Candidate]:
    """Every collection in one sweep slot; newest document still unknown."""
    rows = (
        await db.execute(
            select(
                models.Collection.tenant_id,
                models.Collection.workspace_name,
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.internal_metadata,
            ).where(collection_sweep_slot(slots) == slot)
        )
    ).all()
    return [
        _Candidate(
            tenant_id=row[0],
            workspace_name=row[1],
            observer=row[2],
            observed=row[3],
            internal_metadata=cast("dict[str, Any] | None", row[4]) or {},
            newest_created_at=None,
            newest_session_name=None,
            from_sweep=True,
        )
        for row in rows
    ]


def _min_hours_gated(candidate: _Candidate, now: datetime) -> bool:
    last_dream_at = cast("str | None", candidate.dream_metadata.get("last_dream_at"))
    return last_dream_at is not None and _within_min_hours_gate(last_dream_at, now)


async def _drop_pending_representation(
    db: AsyncSession, candidates: list[_Candidate]
) -> list[_Candidate]:
    """Drop collections whose observed peer still has representation work pending."""
    # region ai
    # queue_item_batches holds one row per work unit with pending items and none
    # once it drains, so this reads the pending units, not every unprocessed queue
    # row. Under MULTI_TENANT it reads only the candidates' tenants; rows with no
    # tenant_id are kept as a belt, since the key prefix still carries the tenant.
    # endregion
    stmt = select(models.QueueItemBatch.work_unit_key).where(
        models.QueueItemBatch.task_type == "representation"
    )
    if settings.MULTI_TENANT:
        tenants = sorted({c.tenant_id for c in candidates})
        stmt = stmt.where(
            models.QueueItemBatch.tenant_id.in_(tenants)
            | models.QueueItemBatch.tenant_id.is_(None)
        )
    pending_keys = (await db.execute(stmt)).scalars().all()
    active_observed = {
        _observed_key(parsed.tenant_id, parsed.workspace_name, parsed.observed)
        for parsed in (parse_work_unit_key(key) for key in pending_keys)
    }
    return [
        c
        for c in candidates
        if _observed_key(c.tenant_id, c.workspace_name, c.observed)
        not in active_observed
    ]


async def _drop_attempted(
    db: AsyncSession, candidates: list[_Candidate], dream_types: list[DreamType]
) -> list[_Candidate]:
    """Drop collections with a dream attempt newer than their newest document."""
    keys = {
        c.work_unit_key(dream_type): c for c in candidates for dream_type in dream_types
    }
    newest_attempts: dict[str, datetime] = {}
    for chunk in _chunks(list(keys)):
        attempt_rows = (
            await db.execute(
                select(
                    models.QueueItem.work_unit_key,
                    func.max(models.QueueItem.created_at),
                )
                .where(
                    models.QueueItem.task_type == "dream",
                    models.QueueItem.work_unit_key.in_(chunk),
                )
                .group_by(models.QueueItem.work_unit_key)
            )
        ).all()
        newest_attempts.update(
            (cast(str, row[0]), cast(datetime, row[1])) for row in attempt_rows
        )
    blocked = {
        id(c)
        for key, c in keys.items()
        if key in newest_attempts
        and newest_attempts[key] >= cast(datetime, c.newest_created_at)
    }
    return [c for c in candidates if id(c) not in blocked]


async def _drop_disabled(
    db: AsyncSession, candidates: list[_Candidate]
) -> list[_Candidate]:
    """Drop candidates whose resolved configuration has dreams turned off."""
    if not candidates:
        return []
    # region ai
    # Keyed by tenant as well as name, both in the lookup and the match: this runs
    # on the cross-tenant service session, so a name-only key would let one
    # tenant's same-named workspace or session decide another tenant's
    # dream.enabled, and would read every tenant's same-named rows.
    # endregion
    workspace_keys = sorted({(c.tenant_id, c.workspace_name) for c in candidates})
    session_keys = sorted(
        {
            (c.tenant_id, c.workspace_name, cast(str, c.newest_session_name))
            for c in candidates
        }
    )
    workspaces: dict[tuple[str, str], models.Workspace] = {}
    for chunk in _chunks(workspace_keys):
        for workspace in (
            (
                await db.execute(
                    select(models.Workspace).where(
                        tuple_(models.Workspace.tenant_id, models.Workspace.name).in_(
                            chunk
                        )
                    )
                )
            )
            .scalars()
            .all()
        ):
            workspaces[(workspace.tenant_id, workspace.name)] = workspace
    sessions: dict[tuple[str, str, str], models.Session] = {}
    for chunk in _chunks(session_keys):
        for session in (
            (
                await db.execute(
                    select(models.Session).where(
                        tuple_(
                            models.Session.tenant_id,
                            models.Session.workspace_name,
                            models.Session.name,
                        ).in_(chunk)
                    )
                )
            )
            .scalars()
            .all()
        ):
            sessions[(session.tenant_id, session.workspace_name, session.name)] = (
                session
            )

    enabled: list[_Candidate] = []
    for c in candidates:
        configuration = get_configuration(
            None,
            sessions.get(
                (c.tenant_id, c.workspace_name, cast(str, c.newest_session_name))
            ),
            workspaces.get((c.tenant_id, c.workspace_name)),
        )
        if configuration.dream.enabled:
            enabled.append(c)
    return enabled


def _unnest_keys(candidates: Sequence[_Candidate]) -> dict[str, list[str]]:
    return {
        "tenant_ids": [c.tenant_id for c in candidates],
        "workspace_names": [c.workspace_name for c in candidates],
        "observers": [c.observer for c in candidates],
        "observeds": [c.observed for c in candidates],
    }


_KEY_ARRAYS = (
    "unnest(CAST(:tenant_ids AS text[]), CAST(:workspace_names AS text[]),"
    + " CAST(:observers AS text[]), CAST(:observeds AS text[])"
)
_KEY_MATCH = (
    "d.tenant_id = c.tenant_id AND d.workspace_name = c.workspace_name"
    + " AND d.observer = c.observer AND d.observed = c.observed"
    + " AND d.level = 'explicit'"
)


def _documents_table() -> str:
    return f'"{settings.DB.SCHEMA}".documents'


async def _drop_below_threshold(
    db: AsyncSession, candidates: list[_Candidate]
) -> list[_Candidate]:
    """Keep collections with at least DOCUMENT_THRESHOLD explicit documents since their last dream."""
    # region ai
    # The count stops at what the rule needs: a collection is due once it has
    # last_dream_document_count + DOCUMENT_THRESHOLD explicit documents, so each
    # per-collection scan of ix_documents_tenant_collection is capped there
    # instead of counting the whole collection. Like the query this replaces, it
    # counts soft-deleted explicit documents too (last_dream_document_count was
    # recorded the same way).
    # endregion
    kept: list[_Candidate] = []
    for chunk in _chunks(candidates):
        caps = [
            int(c.dream_metadata.get("last_dream_document_count", 0))
            + settings.DREAM.DOCUMENT_THRESHOLD
            for c in chunk
        ]
        rows = (
            await db.execute(
                text(
                    "SELECT c.ord, k.n FROM "
                    + _KEY_ARRAYS
                    + ", CAST(:caps AS int[])) WITH ORDINALITY"
                    + " AS c(tenant_id, workspace_name, observer, observed, cap, ord)"
                    + " CROSS JOIN LATERAL (SELECT count(*) AS n FROM (SELECT 1 FROM "
                    + _documents_table()
                    + " d WHERE "
                    + _KEY_MATCH
                    + " LIMIT c.cap) capped) k"
                ).bindparams(caps=caps, **_unnest_keys(chunk))
            )
        ).all()
        counts = {int(row[0]): int(row[1]) for row in rows}
        kept.extend(
            c
            for index, (c, cap) in enumerate(zip(chunk, caps, strict=True), start=1)
            if counts.get(index, 0) >= cap
        )
    return kept


async def _fill_newest(
    db: AsyncSession, candidates: list[_Candidate]
) -> list[_Candidate]:
    """Set each candidate's newest explicit document; drop those without a session."""
    filled: list[_Candidate] = []
    for chunk in _chunks(candidates):
        rows = (
            await db.execute(
                text(
                    "SELECT c.ord, n.created_at, n.session_name FROM "
                    + _KEY_ARRAYS
                    + ") WITH ORDINALITY"
                    + " AS c(tenant_id, workspace_name, observer, observed, ord)"
                    + " CROSS JOIN LATERAL (SELECT d.created_at, d.session_name FROM "
                    + _documents_table()
                    + " d WHERE "
                    + _KEY_MATCH
                    + " ORDER BY d.created_at DESC LIMIT 1) n"
                ).bindparams(**_unnest_keys(chunk))
            )
        ).all()
        newest = {int(row[0]): (row[1], row[2]) for row in rows}
        for index, c in enumerate(chunk, start=1):
            created_at, session_name = newest.get(index, (None, None))
            if session_name is None:
                continue
            c.newest_created_at = created_at
            c.newest_session_name = session_name
            filled.append(c)
    return filled


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
