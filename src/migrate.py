"""Locked Alembic upgrades; run as ``python -m src.migrate [revision]``."""

import argparse
import hashlib
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError

from src.config import settings

__all__ = [
    "MIGRATION_LOCK_KEY",
    "MigrationLockTimeout",
    "acquire_migration_lock",
    "advisory_lock_key",
    "main",
    "release_migration_lock",
    "run_with_lock_retry",
    "upgrade",
]

logger = logging.getLogger(__name__)


def advisory_lock_key(name: str) -> int:
    """Stable signed 64-bit advisory lock key for a namespaced lock name."""
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


MIGRATION_LOCK_KEY = advisory_lock_key("honcho:alembic-migrations")
LOCK_NOT_AVAILABLE = "55P03"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MigrationLockTimeout(RuntimeError):
    """Another migrator held the migration lock past the wait deadline."""


def _lock_holder_pid(connection: Connection) -> int | None:
    return connection.scalar(
        text(
            "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted"
            + " AND classid::bigint = :hi AND objid::bigint = :lo AND objsubid = 1"
        ),
        {
            "hi": (MIGRATION_LOCK_KEY >> 32) & 0xFFFFFFFF,
            "lo": MIGRATION_LOCK_KEY & 0xFFFFFFFF,
        },
    )


def acquire_migration_lock(
    connection: Connection,
    *,
    transaction_scoped: bool = False,
    wait_seconds: float | None = None,
    poll_interval: float = 1.0,
) -> None:
    """Poll for the migration advisory lock, raising after ``wait_seconds``.

    A session-scoped lock is held until ``release_migration_lock`` or disconnect;
    a transaction-scoped one until the current transaction ends.
    """
    try_lock = (
        "pg_try_advisory_xact_lock" if transaction_scoped else "pg_try_advisory_lock"
    )
    if wait_seconds is None:
        wait_seconds = settings.DB.MIGRATION_LOCK_WAIT_SECONDS
    deadline = time.monotonic() + wait_seconds
    waiting = False
    while not connection.scalar(
        text(f"SELECT {try_lock}(:key)"), {"key": MIGRATION_LOCK_KEY}
    ):
        holder = _lock_holder_pid(connection)
        if not transaction_scoped:
            connection.commit()
        if time.monotonic() >= deadline:
            raise MigrationLockTimeout(
                f"Timed out after {wait_seconds}s waiting for the migration lock"
                + f" (held by pid {holder})"
            )
        if not waiting:
            logger.warning("Waiting for migration lock held by pid %s", holder)
            waiting = True
        time.sleep(poll_interval)
    if not transaction_scoped:
        connection.commit()


def release_migration_lock(connection: Connection) -> None:
    """Release a session-scoped migration lock."""
    connection.scalar(
        text("SELECT pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK_KEY}
    )


def run_with_lock_retry[T](
    connection: Connection,
    fn: Callable[[], T],
    *,
    lock_timeout_ms: int = 1000,
    max_attempts: int = 8,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
) -> T:
    """Run ``fn`` under a short ``lock_timeout``, retrying on lock contention.

    ``fn`` must own its transactions on ``connection``. A failed attempt is
    rolled back, then retried after jittered exponential backoff.
    """
    attempt = 1
    while True:
        connection.execute(
            text("SELECT set_config('lock_timeout', :value, false)"),
            {"value": f"{lock_timeout_ms}ms"},
        )
        try:
            return fn()
        except DBAPIError as exc:
            sqlstate = getattr(exc.orig, "sqlstate", None)
            if sqlstate != LOCK_NOT_AVAILABLE or attempt >= max_attempts:
                raise
            connection.rollback()
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = random.uniform(delay / 2, delay)
            logger.warning(
                "Migration attempt %d/%d hit lock_timeout, retrying in %.1fs",
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def upgrade(revision: str = "head") -> None:
    """Upgrade the database to ``revision``; ``migrations/env.py`` takes the lock."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(config, revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Honcho database migrations.")
    parser.add_argument("revision", nargs="?", default="head")
    upgrade(parser.parse_args().revision)


if __name__ == "__main__":
    main()
