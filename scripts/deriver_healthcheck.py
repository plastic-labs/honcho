"""Queue-progress health probe run by the Deriver supervisor.

The API health endpoint proves only that the API process is alive.  This probe
inspects pending Deriver work and its durable leases from inside the Deriver
container.  It is intentionally read-only; the supervisor is responsible for
the failure action.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg

from src.config import settings
from src.utils.work_unit import parse_work_unit_key

logger = logging.getLogger(__name__)

_QUEUE_HEALTH_QUERY = """
    SELECT
        queue.work_unit_key,
        MIN(queue.created_at) AS oldest_pending_at,
        MAX(active_queue_sessions.last_updated) AS last_progress_at
    FROM queue
    LEFT JOIN active_queue_sessions
        ON active_queue_sessions.work_unit_key = queue.work_unit_key
    WHERE queue.processed = false
    GROUP BY queue.work_unit_key
"""


@dataclass(frozen=True)
class QueueHealthResult:
    """A compact, non-sensitive summary for the supervisor and operator logs."""

    pending_work_units: int
    malformed_work_units: int
    stalled_work_units: int
    oldest_stalled_seconds: int = 0

    @property
    def healthy(self) -> bool:
        return self.malformed_work_units == 0 and self.stalled_work_units == 0


def _to_psycopg_uri(connection_uri: str) -> str:
    """Translate SQLAlchemy's psycopg URL scheme for psycopg directly."""
    return connection_uri.replace("postgresql+psycopg://", "postgresql://", 1)


def _as_utc(value: datetime) -> datetime:
    """Normalize database timestamps before calculating a duration."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def evaluate_pending_work_units(
    rows: Iterable[tuple[str, datetime, datetime | None]],
    *,
    now: datetime,
    max_pending_seconds: int,
) -> QueueHealthResult:
    """Evaluate parser validity and observed queue progress without database I/O."""
    pending_work_units = 0
    malformed_work_units = 0
    stalled_work_units = 0
    oldest_stalled_seconds = 0
    current_time = _as_utc(now)

    for work_unit_key, oldest_pending_at, last_progress_at in rows:
        pending_work_units += 1
        try:
            parse_work_unit_key(work_unit_key)
        except ValueError:
            malformed_work_units += 1
            continue

        if max_pending_seconds <= 0:
            continue

        # An actively leased item is judged by its last observed Deriver progress,
        # not the time it entered the queue.  This avoids calling a healthy worker
        # stuck merely because it is processing an old backlog item.
        progress_at = _as_utc(last_progress_at or oldest_pending_at)
        age_seconds = max(0, int((current_time - progress_at).total_seconds()))
        if age_seconds > max_pending_seconds:
            stalled_work_units += 1
            oldest_stalled_seconds = max(oldest_stalled_seconds, age_seconds)

    return QueueHealthResult(
        pending_work_units=pending_work_units,
        malformed_work_units=malformed_work_units,
        stalled_work_units=stalled_work_units,
        oldest_stalled_seconds=oldest_stalled_seconds,
    )


def inspect_queue_health(
    *,
    connection_uri: str,
    connect_timeout_seconds: int,
    max_pending_seconds: int,
    now: datetime | None = None,
    connect: Callable[..., Any] = psycopg.connect,
) -> QueueHealthResult:
    """Run the read-only probe using the application's resolved database config."""
    with (
        connect(
            _to_psycopg_uri(connection_uri), connect_timeout=connect_timeout_seconds
        ) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(_QUEUE_HEALTH_QUERY)
        rows = cursor.fetchall()

    return evaluate_pending_work_units(
        rows,
        now=now or datetime.now(timezone.utc),
        max_pending_seconds=max_pending_seconds,
    )


def main() -> int:
    """Print a concise status and return a conventional health-check exit code."""
    try:
        result = inspect_queue_health(
            connection_uri=settings.DB.CONNECTION_URI,
            connect_timeout_seconds=settings.DB.CONNECT_TIMEOUT_SECONDS,
            max_pending_seconds=settings.DERIVER.HEALTH_MAX_PENDING_SECONDS,
        )
    except Exception:
        logger.exception("Deriver queue health probe failed")
        print("unhealthy: queue health probe failed", file=sys.stderr)
        return 1

    if result.malformed_work_units:
        print(
            f"unhealthy: malformed_work_units={result.malformed_work_units}, "
            + f"pending_work_units={result.pending_work_units}"
        )
        return 1

    if result.stalled_work_units:
        print(
            f"unhealthy: stalled_work_units={result.stalled_work_units}, "
            + f"oldest_stalled_seconds={result.oldest_stalled_seconds}, "
            + f"pending_work_units={result.pending_work_units}"
        )
        return 1

    print(f"healthy: pending_work_units={result.pending_work_units}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
