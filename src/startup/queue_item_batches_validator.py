"""Startup validator for the ``queue_item_batches`` trigger set.

The deriver claims work from ``queue_item_batches`` — an aggregate that only
the triggers on ``queue`` maintain (see ``QueueItemBatch`` in ``src/models.py``).
If those triggers are missing or disabled nothing errors: enqueued rows never
reach the aggregate, the claim sees an empty or stale table, and the deriver
silently processes nothing. This validator turns that half-state into a boot
failure for both processes that touch the queue — the API (enqueue) and the
deriver (claim).

Checks, against the configured ``DB.SCHEMA``:

1. ``queue_item_batches`` exists.
2. Each of the three triggers exists on ``queue`` and is enabled for ordinary
   writes — ``pg_trigger.tgenabled`` is ``O`` (origin) or ``A`` (always).
   ``D`` (disabled) and ``R`` (replica-only, which never fires on a primary)
   both fail.

Always on: the aggregate is load-bearing in single-tenant deployments too.
Alembic revision ``b7d2f4a81c39`` installs everything this validator expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from src.config import AppSettings, settings
from src.startup.embedding_validator import StartupValidationError

logger = logging.getLogger(__name__)

_QUEUE_TABLE = "queue"
_AGGREGATE_TABLE = "queue_item_batches"
_INSTALLING_REVISION = "b7d2f4a81c39"

# The triggers the migration installs on ``queue``; the aggregate is exact only
# while all three fire on every ordinary write.
_REQUIRED_TRIGGERS: tuple[str, ...] = (
    "trg_queue_item_batches_insert",
    "trg_queue_item_batches_update",
    "trg_queue_item_batches_delete",
)

# region ai
# pg_trigger.tgenabled: 'O' fires in origin+local mode (the default), 'A' fires
# always, 'D' is disabled, 'R' fires only under session_replication_role =
# replica — on a primary that is indistinguishable from disabled, so it fails.
# endregion
_LIVE_TGENABLED: frozenset[str] = frozenset({"O", "A"})

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class _TriggerState:
    aggregate_exists: bool
    # trigger name -> tgenabled flag, for the required triggers found on queue
    triggers: dict[str, str] = field(default_factory=dict)


async def validate_queue_item_batches(
    engine: AsyncEngine,
    *,
    app_settings: AppSettings | None = None,
) -> None:
    """Fail boot unless the queue_item_batches aggregate and its triggers are live.

    Run after the DB pool is initialized and before serving traffic / processing
    the queue — the same placement as ``validate_embedding_schema``.
    """
    s = app_settings if app_settings is not None else settings
    state = await _introspect_triggers_with_retry(engine, s.DB.SCHEMA)
    _assert_triggers_live(state, schema=s.DB.SCHEMA)


async def _introspect_triggers_with_retry(
    engine: AsyncEngine, schema: str
) -> _TriggerState:
    # ai: fails closed on the last attempt — uncertainty is not a green light.
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_fixed(_RETRY_BACKOFF_SECONDS),
            retry=retry_if_exception_type(SQLAlchemyError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        ):
            with attempt:
                return await _introspect_triggers_once(engine, schema)
    except RetryError as e:
        underlying = e.last_attempt.exception()
        raise StartupValidationError(
            f"could not validate the {_AGGREGATE_TABLE} triggers: {underlying}"
        ) from underlying
    # ai: unreachable — AsyncRetrying either returns from inside the loop or raises.
    raise StartupValidationError(
        f"{_AGGREGATE_TABLE} trigger introspection did not run"
    )


async def _introspect_triggers_once(engine: AsyncEngine, schema: str) -> _TriggerState:
    """Schema-qualified catalog read: the aggregate's existence + the triggers on queue."""
    table_query = text(
        """
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = :table_name
          AND c.relkind IN ('r', 'p')
        """
    )
    trigger_query = text(
        """
        SELECT t.tgname AS trigger_name, t.tgenabled AS enabled
        FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = :queue_table
          AND NOT t.tgisinternal
          AND t.tgname = ANY(:names)
        """
    )
    async with engine.connect() as conn:
        table_row = await conn.execute(
            table_query, {"schema": schema, "table_name": _AGGREGATE_TABLE}
        )
        aggregate_exists = table_row.first() is not None
        result = await conn.execute(
            trigger_query,
            {
                "schema": schema,
                "queue_table": _QUEUE_TABLE,
                "names": list(_REQUIRED_TRIGGERS),
            },
        )
        triggers = {str(row.trigger_name): str(row.enabled) for row in result}
    return _TriggerState(aggregate_exists=aggregate_exists, triggers=triggers)


def _assert_triggers_live(state: _TriggerState, *, schema: str) -> None:
    upgrade_hint = (
        f" Run `alembic upgrade head` (revision {_INSTALLING_REVISION} installs"
        + " the table and its triggers)."
    )
    if not state.aggregate_exists:
        raise StartupValidationError(
            f"{schema}.{_AGGREGATE_TABLE} is absent: the deriver claim reads only"
            + " this trigger-maintained aggregate, so no work could be claimed."
            + upgrade_hint
        )

    missing = [name for name in _REQUIRED_TRIGGERS if name not in state.triggers]
    if missing:
        raise StartupValidationError(
            f"{_AGGREGATE_TABLE} triggers are missing on {schema}.{_QUEUE_TABLE}: "
            + ", ".join(missing)
            + ". Without them enqueued work never reaches the aggregate and the"
            + " deriver silently claims nothing."
            + upgrade_hint
        )

    dead = [
        f"{name} (tgenabled={state.triggers[name]!r})"
        for name in _REQUIRED_TRIGGERS
        if state.triggers[name] not in _LIVE_TGENABLED
    ]
    if dead:
        raise StartupValidationError(
            f"{_AGGREGATE_TABLE} triggers are not live on {schema}.{_QUEUE_TABLE}: "
            + ", ".join(dead)
            + ". 'D' is disabled and 'R' fires only under replica mode, so on a"
            + " primary neither maintains the aggregate. Re-enable with"
            + f" `ALTER TABLE {schema}.{_QUEUE_TABLE} ENABLE TRIGGER <name>`."
        )
