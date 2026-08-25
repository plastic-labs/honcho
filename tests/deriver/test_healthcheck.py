from datetime import datetime, timedelta, timezone

from scripts import deriver_healthcheck


def test_evaluate_pending_work_units_accepts_empty_queue() -> None:
    result = deriver_healthcheck.evaluate_pending_work_units(
        [],
        now=datetime(2026, 8, 25, tzinfo=timezone.utc),
        max_pending_seconds=3600,
    )

    assert result.healthy is True
    assert result.pending_work_units == 0


def test_evaluate_pending_work_units_detects_malformed_key() -> None:
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    result = deriver_healthcheck.evaluate_pending_work_units(
        [("representation:workspace:session", now, None)],
        now=now,
        max_pending_seconds=3600,
    )

    assert result.healthy is False
    assert result.malformed_work_units == 1
    assert result.stalled_work_units == 0


def test_evaluate_pending_work_units_uses_active_lease_progress_not_enqueue_age() -> None:
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    result = deriver_healthcheck.evaluate_pending_work_units(
        [
            (
                "summary:workspace:session:None:None",
                now - timedelta(hours=2),
                now - timedelta(seconds=30),
            )
        ],
        now=now,
        max_pending_seconds=60,
    )

    assert result.healthy is True
    assert result.stalled_work_units == 0


def test_evaluate_pending_work_units_detects_unclaimed_stall_and_supports_opt_out() -> None:
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    rows = [
        (
            "summary:workspace:session:None:None",
            now - timedelta(seconds=61),
            None,
        )
    ]

    unhealthy = deriver_healthcheck.evaluate_pending_work_units(
        rows, now=now, max_pending_seconds=60
    )
    disabled = deriver_healthcheck.evaluate_pending_work_units(
        rows, now=now, max_pending_seconds=0
    )

    assert unhealthy.healthy is False
    assert unhealthy.stalled_work_units == 1
    assert unhealthy.oldest_stalled_seconds == 61
    assert disabled.healthy is True


def test_inspect_queue_health_uses_resolved_database_uri_and_connect_timeout() -> None:
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    calls: list[tuple[str, int]] = []

    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str) -> None:
            assert "active_queue_sessions" in query

        def fetchall(self) -> list[tuple[str, datetime, datetime | None]]:
            return []

    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

    def connect(uri: str, *, connect_timeout: int) -> Connection:
        calls.append((uri, connect_timeout))
        return Connection()

    result = deriver_healthcheck.inspect_queue_health(
        connection_uri="postgresql+psycopg://configured-host/honcho",
        connect_timeout_seconds=7,
        max_pending_seconds=3600,
        now=now,
        connect=connect,
    )

    assert result.healthy is True
    assert calls == [("postgresql://configured-host/honcho", 7)]
