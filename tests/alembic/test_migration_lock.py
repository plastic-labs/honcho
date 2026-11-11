"""Concurrency tests for the migration advisory lock and lock_timeout retry."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, text

import src.migrate
from src.migrate import (
    MIGRATION_LOCK_KEY,
    MigrationLockTimeout,
    acquire_migration_lock,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _head(alembic_cfg: Config) -> str:
    head = ScriptDirectory.from_config(alembic_cfg).get_current_head()
    assert head is not None
    return head


def _current(alembic_engine: Engine) -> str | None:
    with alembic_engine.connect() as conn:
        return conn.scalar(text("SELECT version_num FROM alembic_version"))


def _start_migrator(alembic_database: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "src.migrate"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "DB_CONNECTION_URI": alembic_database},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_concurrent_migrators_serialize(
    alembic_database: str, alembic_cfg: Config, alembic_engine: Engine
) -> None:
    migrators = [_start_migrator(alembic_database) for _ in range(2)]
    outputs = [proc.communicate(timeout=300)[0] for proc in migrators]

    assert [proc.returncode for proc in migrators] == [0, 0], outputs
    ran = ["Running upgrade" in output for output in outputs]
    assert sorted(ran) == [False, True], outputs
    assert _current(alembic_engine) == _head(alembic_cfg)


def test_terminated_holder_releases_lock(
    alembic_cfg: Config, alembic_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = alembic_engine.connect()
    holder.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
    holder_pid = holder.scalar(text("SELECT pg_backend_pid()"))
    holder.commit()

    events: dict[str, float] = {}

    def acquire_and_record(*args: Any, **kwargs: Any) -> None:
        acquire_migration_lock(*args, **kwargs)
        events["acquired"] = time.monotonic()

    monkeypatch.setattr(src.migrate, "acquire_migration_lock", acquire_and_record)

    def terminate() -> None:
        with alembic_engine.connect() as conn:
            conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": holder_pid})
        events["terminated"] = time.monotonic()

    timer = threading.Timer(2.0, terminate)
    timer.start()
    try:
        command.upgrade(alembic_cfg, "head")
    finally:
        timer.join()
        holder.invalidate()
        holder.close()

    assert events["acquired"] >= events["terminated"]
    assert _current(alembic_engine) == _head(alembic_cfg)


def test_lock_wait_times_out(alembic_engine: Engine) -> None:
    with alembic_engine.connect() as holder, alembic_engine.connect() as waiter:
        holder.execute(
            text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY}
        )
        holder_pid = holder.scalar(text("SELECT pg_backend_pid()"))
        waiter.execute(text("SET idle_in_transaction_session_timeout = '300ms'"))
        waiter.commit()
        try:
            with pytest.raises(MigrationLockTimeout, match=f"pid {holder_pid}"):
                acquire_migration_lock(waiter, wait_seconds=1.5, poll_interval=0.5)
        finally:
            holder.execute(
                text("SELECT pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK_KEY}
            )


def test_lock_timeout_retries_until_table_is_free(
    alembic_cfg: Config, alembic_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ScriptDirectory.from_config(alembic_cfg)
    head = _head(alembic_cfg)
    previous = script.get_revision(head).down_revision
    # A merge head has several parents; stop on all of them so only the head
    # revision itself is left for the blocked upgrade.
    parents = (previous,) if isinstance(previous, str) else tuple(previous or ())
    assert parents
    for parent in parents:
        command.upgrade(alembic_cfg, parent)

    backoffs: list[float] = []
    real_sleep = time.sleep

    def record_sleep(seconds: float) -> None:
        backoffs.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", record_sleep)

    blocker = alembic_engine.connect()
    blocker.execute(text("LOCK TABLE alembic_version IN ACCESS EXCLUSIVE MODE"))
    timer = threading.Timer(2.5, blocker.rollback)
    timer.start()
    try:
        command.upgrade(alembic_cfg, "head")
    finally:
        timer.join()
        blocker.close()

    assert backoffs, "upgrade never hit lock_timeout"
    assert _current(alembic_engine) == head


def test_reconciler_check_keeps_runner_lock_timeout(
    alembic_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c4e8a2b91d57 sets its own lock_timeout and must put the runner's back."""
    command.upgrade(alembic_cfg, "b7d2f4a81c39")

    real_retry = src.migrate.run_with_lock_retry
    seen: list[str | None] = []

    def retry_and_record(connection: Connection, fn: Any, **_: Any) -> Any:
        result = real_retry(connection, fn, lock_timeout_ms=1234)
        seen.append(connection.scalar(text("SHOW lock_timeout")))
        return result

    monkeypatch.setattr(src.migrate, "run_with_lock_retry", retry_and_record)
    command.upgrade(alembic_cfg, "c4e8a2b91d57")

    assert seen == ["1234ms"]
