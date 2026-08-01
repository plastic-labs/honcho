"""Container healthcheck for Deriver queue progress.

The API health endpoint only proves that the separate API container is alive.
This probe runs inside Deriver and checks the queue state it is responsible for.
"""

from __future__ import annotations

import os
import sys

import psycopg


def main() -> int:
    uri = os.environ["DB_CONNECTION_URI"].replace(
        "postgresql+psycopg://", "postgresql://", 1
    )
    max_pending_seconds = int(
        os.environ.get("DERIVER_HEALTH_MAX_PENDING_SECONDS", "3600")
    )

    with (
        psycopg.connect(uri, connect_timeout=5) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            SELECT
                -- Mirrors parse_work_unit_key(): a representation key must be
                -- prefixed 'representation:' AND have either 4 segments
                -- (workspace:session:observed) or 5 (legacy, with observer).
                -- Any other shape raises in the parser and would be quarantined,
                -- so counting only the prefix mismatch would under-report.
                COUNT(*) FILTER (
                    WHERE task_type = 'representation'
                      AND (
                        work_unit_key NOT LIKE 'representation:%'
                        OR array_length(string_to_array(work_unit_key, ':'), 1)
                             NOT IN (4, 5)
                      )
                ),
                COALESCE(
                    EXTRACT(EPOCH FROM (NOW() - MIN(created_at)))::bigint,
                    0
                ),
                COUNT(*)
            FROM queue
            WHERE processed = false
              AND task_type IN ('representation', 'summary', 'dream')
            """
        )
        malformed, oldest_pending_seconds, pending = cursor.fetchone()

    if malformed:
        print(f"unhealthy: {malformed} malformed representation work unit(s)")
        return 1
    if pending and oldest_pending_seconds > max_pending_seconds:
        print(
            "unhealthy: oldest pending user-facing work unit is "
            f"{oldest_pending_seconds}s old (limit {max_pending_seconds}s)"
        )
        return 1

    print(
        f"healthy: pending={pending}, oldest_pending_seconds={oldest_pending_seconds}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
